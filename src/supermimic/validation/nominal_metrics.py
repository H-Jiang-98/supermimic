"""Action-endpoint tracking metrics in the paper's units."""

# PyTorch 2.14 dynamically re-exports public tensor factories and operations.
# pyright: reportPrivateImportUsage=false

import torch
from mjlab.utils.lab_api.math import matrix_from_quat

from supermimic.local_control import heading
from supermimic.validation.metrics import world_rpy


def tracking_errors(robot_pos, ref_pos, robot_rot, ref_rot, robot_v, ref_v):
  def skeleton(pos, rotation):
    return (pos - pos[:, :1]) @ heading(rotation)

  difference = skeleton(robot_pos, robot_rot) - skeleton(ref_pos, ref_rot)
  pelvis = robot_pos[:, 0] - ref_pos[:, 0]
  delta = world_rpy(robot_rot)[:, :2] - world_rpy(ref_rot)[:, :2]
  angles = torch.atan2(delta.sin(), delta.cos()).abs().rad2deg()
  body_v = (robot_rot.mT @ robot_v.unsqueeze(-1)) - (ref_rot.mT @ ref_v.unsqueeze(-1))
  return torch.stack(
    (
      difference.norm(dim=-1).mean(-1) * 100,
      pelvis[:, 2].abs() * 100,
      angles[:, 0],
      angles[:, 1],
      body_v.squeeze(-1).norm(dim=-1),
      pelvis.norm(dim=-1) * 100,
    ),
    -1,
  )


def reward_terms(raw):
  names = (
    "motion_global_root_pos",
    "motion_global_root_ori",
    "motion_body_pos",
    "motion_body_ori",
    "motion_body_lin_vel",
    "motion_body_ang_vel",
  )
  terms = [raw.reward_manager.get_term_cfg(name) for name in names]
  if [t.weight for t in terms] != [0.5, 0.5, 1.0, 1.0, 1.0, 1.0]:
    raise ValueError("Nominal tracking reward weights differ from the paper protocol")
  return terms


def frame_scores(raw, command, terms):
  errors = tracking_errors(
    command.robot_body_pos_w,
    command.body_pos_w,
    matrix_from_quat(command.robot_anchor_quat_w),
    matrix_from_quat(command.anchor_quat_w),
    command.robot_body_lin_vel_w[:, 0],
    command.body_lin_vel_w[:, 0],
  )
  weighted: list[torch.Tensor] = [t.weight * t.func(raw, **t.params) for t in terms]
  reward = weighted[0]
  for term in weighted[1:]:
    reward = reward + term
  reward = reward / 5
  return torch.cat((errors, reward[:, None]), -1)
