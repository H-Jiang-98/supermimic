"""Same-frame world errors and square-root RPV9 posterior scores."""

import math

import numpy as np
import torch

from supermimic.utils.lie_state import local_error, so3_log

ODOMETRY = (
  "root_rotation_rmse_deg",
  "root_position_rmse_m",
  "root_velocity_rmse_m_s",
  "root_nll",
  "root_nees",
)
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
  return torch.stack(
    (
      error[..., :3].square().sum(-1) * (180 / math.pi) ** 2,
      (belief.mean.position.double() - truth.position.double()).square().sum(-1),
      (belief.mean.velocity.double() - truth.velocity.double()).square().sum(-1),
      nll,
      nees,
    ),
    -1,
  )


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
      (truth.velocity.double() - reference.velocity[:, 0].double()).square().sum(-1),
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
    self.count = self.current_count = 0
    self.bins, self.counts = [], []

  def update(self, values):
    if values.shape != self.total.shape or not torch.isfinite(values).all():
      raise ValueError("Invalid or non-finite evaluation metrics")
    self.total += values
    self.current += values
    self.count += 1
    self.current_count += 1
    if self.current_count == self.width:
      self.flush()

  def finish_mean(self, sums, n):
    values = sums / n
    for i, name in enumerate(self.names):
      if "rmse" in name:
        values[:, i] = values[:, i].clamp_min(0).sqrt()
    return values.cpu().numpy()

  def flush(self):
    if self.current_count:
      self.bins.append(self.finish_mean(self.current, self.current_count))
      self.counts.append(self.current_count)
      self.current.zero_()
      self.current_count = 0

  def result(self):
    if not self.count:
      raise ValueError("No frames scored")
    self.flush()
    return dict(
      full=self.finish_mean(self.total, self.count),
      bins=np.stack(self.bins),
      bin_counts=np.array(self.counts),
      frames=np.array(self.count),
    )
