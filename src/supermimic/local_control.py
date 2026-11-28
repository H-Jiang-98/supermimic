"""Reference-heading motion goals and local posterior coordinates for EIT v7."""

from dataclasses import dataclass

import torch

from supermimic.utils.lie_state import (
  LieState,
  _matrix_from_quat,
  skew,
  so3_left_jacobian_inverse,
  so3_log,
)

OFFSETS = (1, 5, 10, 15, 20, 25, 30, 35, 40, 45)
GOAL_DIM = 730
LOCAL_DIM = 68
VECH_DIM = 2346


def local_state(state: LieState):
  return torch.cat(
    (
      -state.rotation[..., 2, :],
      state.position[..., 2:3],
      state.velocity,
      state.omega,
      state.joint_pos,
      state.joint_vel,
    ),
    -1,
  )


def projected_factor(mean, lower):
  """Apply the sparse 68x70 projection directly to the full posterior factor."""
  gravity = -mean.rotation[..., 2, :]
  return torch.cat(
    (
      skew(gravity) @ lower[..., :3, :],
      -gravity.unsqueeze(-2) @ lower[..., 3:6, :],
      lower[..., 6:, :],
    ),
    -2,
  )


def local_covariance_features(mean, lower):
  factor = projected_factor(mean, lower)
  covariance = factor @ factor.mT
  rows, cols = torch.tril_indices(LOCAL_DIM, LOCAL_DIM, device=lower.device)
  return covariance[..., rows, cols]


def heading(rotation):
  yaw = torch.atan2(rotation[..., 1, 0], rotation[..., 0, 0])
  c, s = yaw.cos(), yaw.sin()
  zero, one = torch.zeros_like(c), torch.ones_like(c)
  return torch.stack((c, -s, zero, s, c, zero, zero, zero, one), -1).reshape_as(
    rotation
  )


@dataclass(frozen=True)
class LocalReference:
  goal: torch.Tensor
  y: torch.Tensor
  next_joint: torch.Tensor

  def index(self, idx):
    return LocalReference(self.goal[idx], self.y[idx], self.next_joint[idx])


def motion_reference(motion, times, last):
  """Gather current plus ten future frames for arbitrary time/batch dimensions."""
  idx = torch.minimum(
    times[..., None] + times.new_tensor((0, *OFFSETS)), last[..., None]
  )
  rotation = _matrix_from_quat(motion.root_quat_w[idx])
  position = motion.root_pos_w[idx]
  velocity = (rotation.mT @ motion.root_lin_vel_w[idx].unsqueeze(-1)).squeeze(-1)
  omega = (rotation.mT @ motion.root_ang_vel_w[idx].unsqueeze(-1)).squeeze(-1)
  q, qdot = motion.joint_pos[idx], motion.joint_vel[idx]
  current = LieState(
    rotation[..., 0, :, :],
    position[..., 0, :],
    velocity[..., 0, :],
    omega[..., 0, :],
    q[..., 0, :],
    qdot[..., 0, :],
  )
  inv = heading(current.rotation).mT.unsqueeze(-3)
  origin = torch.cat(
    (current.position[..., :2], torch.zeros_like(current.position[..., 2:])), -1
  )
  relative = (
    inv @ (position[..., 1:, :] - origin.unsqueeze(-2)).unsqueeze(-1)
  ).squeeze(-1)
  orientation = inv @ rotation[..., 1:, :, :]
  goal = torch.cat(
    (
      relative,
      orientation[..., :2].flatten(-2),
      velocity[..., 1:, :],
      omega[..., 1:, :],
      q[..., 1:, :],
      qdot[..., 1:, :],
    ),
    -1,
  ).flatten(-2)
  return LocalReference(goal, local_state(current), q[..., 1, :])


def privileged_goal(truth, reference):
  rotation = reference.rotation[:, 0]
  delta = truth.position - reference.position[:, 0]
  relative = (heading(rotation).mT @ delta.unsqueeze(-1)).squeeze(-1)
  relative_heading = heading(rotation).mT @ heading(truth.rotation)
  yaw = torch.atan2(relative_heading[:, 1, 0], relative_heading[:, 0, 0])
  return torch.cat((reference.local.goal, relative[:, :2], yaw[:, None]), -1)


def root_tracking_error(state, rotation, position, world_velocity):
  """[Log(T_state^-1 T_ref), R_state.T v_ref_world - v_state_body]."""
  inverse = state.rotation.mT
  theta = so3_log(inverse @ rotation)
  delta = (inverse @ (position - state.position).unsqueeze(-1)).squeeze(-1)
  rho = (so3_left_jacobian_inverse(theta) @ delta.unsqueeze(-1)).squeeze(-1)
  velocity = (inverse @ world_velocity.unsqueeze(-1)).squeeze(-1) - state.velocity
  return torch.cat((theta, rho, velocity), -1)


def root_tracking_jacobian(state, rotation, position, world_velocity):
  """Analytic 9x9 task projection; tensor-only, including pose cross terms.

  Differentiate Log(Exp(-delta_pose) T_state^-1 T_ref) at zero and
  the additive body velocity error. No nested autodiff or host reads in rollout.
  """
  inverse = state.rotation.mT
  theta = so3_log(inverse @ rotation)
  p = (inverse @ (position - state.position).unsqueeze(-1)).squeeze(-1)
  j = so3_left_jacobian_inverse(theta)
  squared = theta.square().sum(-1)[..., None, None]
  safe = squared.clamp_min(0.01)
  half = 0.5 * safe.sqrt()
  cot = 1 / half.tan()
  c = torch.where(
    squared < 0.01,
    1 / 12 + squared / 720 + squared.square() / 30240 + squared.pow(3) / 1209600,
    (1 - half * cot) / safe,
  )
  dc = torch.where(
    squared < 0.01,
    1 / 720 + squared / 15120 + squared.square() / 403200,
    (-1 + half * cot / 2 + half.square() * (1 + cot.square()) / 2) / safe.square(),
  )
  eye = torch.eye(3, device=p.device, dtype=p.dtype).expand_as(j)
  zero = torch.zeros_like(j)
  dot = (theta * p).sum(-1)[..., None, None]
  outer = theta.unsqueeze(-1) * p.unsqueeze(-2)
  k2p = theta * dot.squeeze(-1) - squared.squeeze(-1) * p
  derivative = (
    0.5 * skew(p)
    + c * (dot * eye + outer - 2 * outer.mT)
    + 2 * dc * k2p.unsqueeze(-1) * theta.unsqueeze(-2)
  )
  velocity = (inverse @ world_velocity.unsqueeze(-1)).squeeze(-1)
  return torch.cat(
    (
      torch.cat((-j, zero, zero), -1),
      torch.cat((-derivative @ j + j @ skew(p), -j, zero), -1),
      torch.cat((skew(velocity), zero, -eye), -1),
    ),
    -2,
  )


def root_tracking_covariance(state, lower, rotation, position, world_velocity):
  jacobian = root_tracking_jacobian(state, rotation, position, world_velocity)
  factor = jacobian @ lower[..., :9, :9]
  return factor @ factor.mT
