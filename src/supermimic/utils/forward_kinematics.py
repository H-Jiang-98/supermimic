"""Differentiable MuJoCo tree kinematics for a floating-base hinge/slide robot.

Inputs use entity joint order and world-frame root velocities. Outputs are body
frame origins (not centers of mass), matching mjlab's ``body_link_*`` fields.
MuJoCo is used only to extract constants; all runtime operations are PyTorch.
"""

from __future__ import annotations

import mujoco
import torch
from torch import nn


def _mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  aw, av, bw, bv = a[..., :1], a[..., 1:], b[..., :1], b[..., 1:]
  return torch.cat(
    (
      aw * bw - (av * bv).sum(-1, keepdim=True),
      aw * bv + bw * av + torch.linalg.cross(av, bv),
    ),
    -1,
  )


def _rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
  axis, v = torch.broadcast_tensors(q[..., 1:], v)
  t = 2 * torch.linalg.cross(axis, v)
  return v + q[..., :1] * t + torch.linalg.cross(axis, t)


class ForwardKinematics(nn.Module):
  """Kinematics in arbitrary batch dimensions, including joint-rate derivatives."""

  def __init__(
    self,
    model: mujoco.MjModel,
    body_ids: torch.Tensor,
    joint_q_adr: torch.Tensor,
    tracked_body_rows: torch.Tensor,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
  ):
    super().__init__()
    bodies = [int(i) for i in body_ids]
    if not bodies or len(set(bodies)) != len(bodies):
      raise ValueError("body_ids must contain one robot's distinct bodies")
    root = bodies[0]
    body_set = set(bodies)
    if any(int(model.body_parentid[b]) not in body_set for b in bodies[1:]):
      raise ValueError("body_ids must include every link below the root")
    root_joints = range(
      model.body_jntadr[root], model.body_jntadr[root] + model.body_jntnum[root]
    )
    if (
      len(root_joints) != 1
      or model.jnt_type[root_joints[0]] != mujoco.mjtJoint.mjJNT_FREE
    ):
      raise ValueError("ForwardKinematics requires a single free joint on the root")
    ordered = [root] + sorted(bodies[1:])  # MuJoCo parents precede their children.
    row_of = {body: row for row, body in enumerate(ordered)}
    col_of = {int(adr): col for col, adr in enumerate(joint_q_adr)}
    self._links: list[tuple[int, tuple[tuple[int, int, bool], ...]]] = []
    used_columns = set()
    for body in ordered[1:]:
      joints = []
      for j in range(
        model.body_jntadr[body], model.body_jntadr[body] + model.body_jntnum[body]
      ):
        kind = int(model.jnt_type[j])
        if kind not in (
          int(mujoco.mjtJoint.mjJNT_HINGE),
          int(mujoco.mjtJoint.mjJNT_SLIDE),
        ):
          raise ValueError("Only hinge and slide joints are supported below the root")
        column = col_of.get(int(model.jnt_qposadr[j]))
        if column is None:
          raise ValueError(f"joint_q_adr is missing joint {j}")
        used_columns.add(column)
        joints.append((j, column, kind == mujoco.mjtJoint.mjJNT_HINGE))
      self._links.append((row_of[int(model.body_parentid[body])], tuple(joints)))
    if used_columns != set(range(len(col_of))):
      raise ValueError("joint_q_adr must contain exactly this robot's scalar joints")

    def buffer(name, values):
      self.register_buffer(
        name, torch.tensor(values.copy(), dtype=dtype, device=device)
      )

    buffer("offset", model.body_pos[ordered])
    buffer("orientation", model.body_quat[ordered])
    buffer("axis", model.jnt_axis)
    buffer("anchor", model.jnt_pos)
    buffer("reference", model.qpos0[model.jnt_qposadr])
    self._tracked = [row_of[bodies[int(row)]] for row in tracked_body_rows]

  def forward(
    self,
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    joint_pos: torch.Tensor,
    root_lin_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
    joint_vel: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    positions, quaternions = [root_pos], [root_quat]
    linear, angular = [root_lin_vel], [root_ang_vel]
    for row, (parent, joints) in enumerate(self._links, start=1):
      offset = _rotate(quaternions[parent], self.offset[row])
      pos = positions[parent] + offset
      quat = _mul(quaternions[parent], self.orientation[row].expand_as(root_quat))
      vel = linear[parent] + torch.linalg.cross(angular[parent], offset)
      omega = angular[parent]
      for j, col, hinge in joints:
        axis_w = _rotate(quat, self.axis[j])
        angle = joint_pos[..., col : col + 1] - self.reference[j]
        rate = joint_vel[..., col : col + 1]
        if hinge:
          pivot = pos + _rotate(quat, self.anchor[j])
          rotation = torch.cat(
            (torch.cos(angle / 2), torch.sin(angle / 2) * self.axis[j]), -1
          )
          quat = _mul(quat, rotation)
          next_pos = pivot - _rotate(quat, self.anchor[j])
          vel = (
            vel
            + torch.linalg.cross(omega, next_pos - pos)
            + torch.linalg.cross(axis_w * rate, next_pos - pivot)
          )
          pos = next_pos
          omega = omega + axis_w * rate
        else:
          offset = axis_w * angle
          pos = pos + offset
          vel = vel + torch.linalg.cross(omega, offset) + axis_w * rate
      positions.append(pos)
      quaternions.append(quat)
      linear.append(vel)
      angular.append(omega)
    return tuple(
      torch.stack([values[i] for i in self._tracked], -2)
      for values in (positions, quaternions, linear, angular)
    )
