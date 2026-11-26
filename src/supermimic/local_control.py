"""Reference-heading motion goals and local posterior coordinates for EIT v7."""

from dataclasses import dataclass

import torch

from supermimic.utils.lie_state import LieState, _matrix_from_quat, skew

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
