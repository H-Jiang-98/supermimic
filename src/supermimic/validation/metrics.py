"""Same-frame world errors and square-root RPV9 posterior scores."""

import math

import numpy as np
import torch

from supermimic.utils.lie_state import local_error, so3_log

ODOMETRY_SUMMARY = (
  "root_rotation_rmse_deg",
  "root_position_rmse_m",
  "root_velocity_rmse_m_s",
  "root_nll",
  "root_nees",
)
ODOMETRY_COMPONENTS = (
  "root_x_rmse_m",
  "root_y_rmse_m",
  "root_z_rmse_m",
  "root_roll_rmse_deg",
  "root_pitch_rmse_deg",
  "root_yaw_rmse_deg",
)
ODOMETRY = ODOMETRY_SUMMARY + ODOMETRY_COMPONENTS
TRACKING = (
  "root_rotation_rmse_deg",
  "root_position_rmse_m",
  "root_velocity_rmse_m_s",
  "mpkpe_m",
  "ee_position_error_m",
  "ee_orientation_error_deg",
  "joint_position_rmse_deg",
  "joint_velocity_rmse_deg_s",
)


def posterior_values(belief, truth):
  lower = belief.covariance_sqrt[..., :9, :9].double()
  if not torch.isfinite(lower).all() or (lower.diagonal(dim1=-2, dim2=-1) <= 0).any():
    raise ValueError("Invalid RPV9 covariance square root")
  return posterior_values_kernel(belief, truth)


def world_rpy(rotation):
  """Roll, pitch, yaw for R = Rz(yaw) Ry(pitch) Rx(roll), in radians."""
  r = rotation.double()
  horizontal = torch.hypot(r[..., 0, 0], r[..., 1, 0])
  singular = horizontal < 1e-6
  roll = torch.where(singular, 0.0, torch.atan2(r[..., 2, 1], r[..., 2, 2]))
  pitch = torch.atan2(-r[..., 2, 0], horizontal)
  # At a vertical pitch, choose roll = 0 and put the remaining rotation in yaw.
  yaw = torch.where(
    singular,
    torch.atan2(-r[..., 0, 1], r[..., 1, 1]),
    torch.atan2(r[..., 1, 0], r[..., 0, 0]),
  )
  return torch.stack((roll, pitch, yaw), dim=-1)


def posterior_values_kernel(belief, truth):
  error = local_error(belief.mean, truth)[..., :9].double()
  lower = belief.covariance_sqrt[..., :9, :9].double()
  white = torch.linalg.solve_triangular(lower, error[..., None], upper=False).squeeze(
    -1
  )
  nees = white.square().sum(-1)
  nll = (
    9 * math.log(2 * math.pi)
    + 2 * lower.diagonal(dim1=-2, dim2=-1).log().sum(-1)
    + nees
  ) / 2
  position_sq = (belief.mean.position.double() - truth.position.double()).square()
  rpy_delta = world_rpy(belief.mean.rotation) - world_rpy(truth.rotation)
  rpy_sq_deg = torch.rad2deg(
    torch.atan2(torch.sin(rpy_delta), torch.cos(rpy_delta))
  ).square()
  summary = torch.stack(
    (
      error[..., :3].square().sum(-1) * (180 / math.pi) ** 2,
      position_sq.sum(-1),
      (belief.mean.world_velocity.double() - truth.world_velocity.double())
      .square()
      .sum(-1),
      nll,
      nees,
    ),
    -1,
  )
  return torch.cat((summary, position_sq, rpy_sq_deg), dim=-1)


def tracking_values(truth, reference, command, ee_indices):
  return tracking_values_kernel(
    truth,
    reference,
    command.robot_body_pos_w,
    command.body_pos_w,
    command.robot_body_quat_w[:, ee_indices],
    command.body_quat_w[:, ee_indices],
    command.joint_pos,
    command.joint_vel,
    ee_indices,
  )


def tracking_values_kernel(
  truth,
  reference,
  robot_pos,
  reference_pos,
  robot_ee_quat,
  reference_ee_quat,
  joint_pos,
  joint_vel,
  ee_indices,
):
  from mjlab.utils.lab_api.math import matrix_from_quat

  root_rot = so3_log(truth.rotation @ reference.rotation[:, 0].mT)
  pos = robot_pos - reference_pos
  ee_rot = matrix_from_quat(robot_ee_quat) @ matrix_from_quat(reference_ee_quat).mT
  return torch.stack(
    (
      root_rot.double().square().sum(-1) * (180 / math.pi) ** 2,
      (truth.position.double() - reference.position[:, 0].double()).square().sum(-1),
      (truth.world_velocity.double() - reference.velocity[:, 0].double())
      .square()
      .sum(-1),
      pos.double().norm(dim=-1).mean(-1),
      pos[:, ee_indices].double().norm(dim=-1).mean(-1),
      so3_log(ee_rot).double().norm(dim=-1).mean(-1) * (180 / math.pi),
      (truth.joint_pos.double() - joint_pos.double()).square().mean(-1)
      * (180 / math.pi) ** 2,
      (truth.joint_vel.double() - joint_vel.double()).square().mean(-1)
      * (180 / math.pi) ** 2,
    ),
    -1,
  )


class BinnedMetrics:
  """Keep sums before square root; never average framewise root norms as RMSE."""

  def __init__(self, names, batch, bin_steps, device):
    self.names, self.width = tuple(names), bin_steps
    self.total = torch.zeros(batch, len(names), dtype=torch.float64, device=device)
    self.current = torch.zeros_like(self.total)
    self.valid_count = torch.zeros(batch, dtype=torch.long, device=device)
    self.bin_valid = torch.zeros_like(self.valid_count)
    self.count = self.current_count = 0
    self.bins, self.counts = [], []

  def update(self, values, valid=None):
    if valid is None:
      valid = torch.ones_like(self.valid_count, dtype=torch.bool)
    values = torch.where(valid[:, None], values, 0)
    if values.shape != self.total.shape or not torch.isfinite(values).all():
      raise ValueError("Invalid or non-finite evaluation metrics")
    self.valid_count += valid
    self.bin_valid += valid
    self.total += values
    self.current += values
    self.count += 1
    self.current_count += 1
    if self.current_count == self.width:
      self.flush()

  def finish_mean(self, sums, n):
    values = sums / n[:, None]
    for i, name in enumerate(self.names):
      if "rmse" in name:
        values[:, i] = values[:, i].clamp_min(0).sqrt()
    return values.cpu().numpy()

  def flush(self):
    if self.current_count:
      self.bins.append(self.finish_mean(self.current, self.bin_valid))
      self.counts.append(self.current_count)
      self.current.zero_()
      self.bin_valid.zero_()
      self.current_count = 0

  def result(self):
    if not self.count:
      raise ValueError("No frames scored")
    self.flush()
    return dict(
      valid_frame_counts=self.valid_count.cpu().numpy(),
      full=self.finish_mean(self.total, self.valid_count),
      bins=np.stack(self.bins),
      bin_counts=np.array(self.counts),
      frames=np.array(self.count),
    )


LOCAL_METRICS = (
  "gravity_direction_rmse_deg",
  "height_rmse_m",
  "body_velocity_rmse_m_s",
  "body_angular_velocity_rmse_rad_s",
  "posterior_joint_position_rmse_rad",
  "posterior_joint_velocity_rmse_rad_s",
  "local_nees",
  "local_nll",
)
RELATIVE_METRICS = ("relative_displacement_1s_rmse_m", "relative_rotation_1s_rmse_deg")


def local_posterior_values(belief, truth):
  """Use two orthonormal gravity coordinates and all 65 remaining local coordinates."""
  from supermimic.local_control import local_state, projected_factor

  mean = belief.mean
  y, target = local_state(mean).double(), local_state(truth).double()
  gravity = y[..., :3]
  axis = torch.eye(3, device=y.device, dtype=y.dtype)[gravity.abs().argmin(-1)]
  tangent = torch.nn.functional.normalize(torch.linalg.cross(gravity, axis), dim=-1)
  basis = torch.stack((tangent, torch.linalg.cross(gravity, tangent)), -2)
  error = target - y
  residual = torch.cat((basis @ error[..., :3, None], error[..., 3:, None]), -2)
  factor = projected_factor(mean, belief.covariance_sqrt).double()
  projected = torch.cat((basis @ factor[..., :3, :], factor[..., 3:, :]), -2)
  lower, info = torch.linalg.cholesky_ex(projected @ projected.mT)
  white = torch.linalg.solve_triangular(lower, residual, upper=False)
  nees = white.square().sum((-2, -1))
  nll = (67 * math.log(2 * math.pi) + nees) / 2 + lower.diagonal(
    dim1=-2, dim2=-1
  ).log().sum(-1)
  angle = torch.atan2(
    torch.linalg.cross(gravity, target[..., :3]).norm(dim=-1),
    (gravity * target[..., :3]).sum(-1),
  )
  scores = torch.stack(
    (
      angle.rad2deg().square(),
      error[..., 3].square(),
      error[..., 4:7].square().sum(-1),
      error[..., 7:10].square().sum(-1),
      error[..., 10:39].square().mean(-1),
      error[..., 39:].square().mean(-1),
      nees,
      nll,
    ),
    -1,
  )
  return torch.where((info == 0)[..., None], scores, float("nan"))


def relative_pose_squared(start, end):
  """Compare two pose increments, each expressed in its own starting body frame."""

  def increments(poses):
    rotation, position = poses
    return rotation, position.unsqueeze(-1)

  r0, p0 = increments(start)
  r1, p1 = increments(end)
  translations = (r0.mT @ (p1 - p0)).squeeze(-1)
  rotations = r0.mT @ r1
  return torch.stack(
    (
      (translations[0] - translations[1]).square().sum(-1),
      so3_log(rotations[0].mT @ rotations[1]).rad2deg().square().sum(-1),
    ),
    -1,
  )
