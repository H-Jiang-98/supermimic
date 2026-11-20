"""Fixed episode references and the clean sensor contract, without simulation."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from mjlab.utils.lab_api.math import matrix_from_quat, quat_apply, quat_mul

from supermimic.estimator_inputs import EpisodeAlignment
from supermimic.motion_command import MotionCommand
from supermimic.tasks.env_cfgs import (
  configure_estimator_env_cfg,
  gencoord_env_cfg,
  joint_vel,
  maxcoord_env_cfg,
  root_ang_vel,
)
from supermimic.utils.tracking_reward import _reanchor_targets, tracking_reward


def _command() -> MotionCommand:
  command = object.__new__(MotionCommand)
  command.cfg = SimpleNamespace(
    fixed_reference_alignment=True,
    sampling_mode="start",
    pose_range={"x": (0.4, 0.4), "y": (-0.3, -0.3), "yaw": (0.5, 0.5)},
    velocity_range={},
    joint_position_range=(0.0, 0.0),
    body_names=("root", "foot"),
  )
  origins = torch.tensor([[0.0, 0.0, 0.0], [4.0, 7.0, 0.0]])
  command._env = SimpleNamespace(
    num_envs=2,
    device="cpu",
    scene=SimpleNamespace(env_origins=origins),
    sim=SimpleNamespace(forward=MagicMock()),
  )
  root_pos = torch.tensor([[1.0, 2.0, 0.8], [1.1, 2.1, 0.9], [1.3, 2.0, 0.9]])
  root_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(3, 1)
  root_velocity = torch.tensor([[0.3, 0.2, 0.1]]).repeat(3, 1)
  root_angular = torch.tensor([[0.1, 0.2, 0.3]]).repeat(3, 1)
  command.motion = SimpleNamespace(
    root_pos_w=root_pos,
    root_quat_w=root_quat,
    root_lin_vel_w=root_velocity,
    body_pos_w=torch.stack([root_pos, root_pos + torch.tensor([0.2, 0.1, -0.7])], 1),
    body_quat_w=root_quat[:, None].repeat(1, 2, 1),
    body_lin_vel_w=root_velocity[:, None].repeat(1, 2, 1),
    body_ang_vel_w=root_angular[:, None].repeat(1, 2, 1),
    joint_pos=torch.zeros(3, 29),
    joint_vel=torch.zeros(3, 29),
    last_frame=lambda frames: torch.full_like(frames, 2),
  )
  command.robot = SimpleNamespace(
    data=SimpleNamespace(
      soft_joint_pos_limits=torch.tensor([[[-10.0, 10.0]]]).repeat(2, 29, 1),
      body_link_pos_w=torch.zeros(2, 2, 3),
      body_link_quat_w=root_quat[0].repeat(2, 2, 1),
    ),
    write_joint_state_to_sim=MagicMock(),
    write_root_state_to_sim=MagicMock(),
    reset=MagicMock(),
  )
  command.reference_alignment = EpisodeAlignment.identity(2)
  command.time_steps = torch.zeros(2, dtype=torch.long)
  command._just_reset = torch.zeros(2, dtype=torch.bool)
  command._pending_forward = False
  command.body_pos_relative_w = torch.zeros(2, 2, 3)
  command.body_quat_relative_w = torch.zeros(2, 2, 4)
  command.motion_anchor_body_index = 0
  command.robot_anchor_body_index = 0
  command.body_indexes = torch.arange(2)
  return command


def test_reset_alignment_uses_actual_reset_pose_without_old_transform():
  command = _command()
  command.reference_alignment.translation[:] = 100.0
  old_body = command.motion.body_pos_w.clone()
  command._resample_command(torch.tensor([0, 1]))
  reset_state = command.robot.write_root_state_to_sim.call_args.args[0]
  expected_position = command.motion.root_pos_w[0] + command._env.scene.env_origins
  expected_position = expected_position + torch.tensor([0.4, -0.3, 0.0])
  torch.testing.assert_close(reset_state[:, :3], expected_position)
  torch.testing.assert_close(command.anchor_pos_w, reset_state[:, :3])
  torch.testing.assert_close(
    matrix_from_quat(command.anchor_quat_w), matrix_from_quat(reset_state[:, 3:7])
  )
  torch.testing.assert_close(command.motion.body_pos_w, old_body)
  assert torch.count_nonzero(command.reference_alignment.translation[:, 2]) == 0


def test_fixed_references_advance_without_reanchoring_and_rotate_velocities():
  command = _command()
  command._resample_command(torch.tensor([0, 1]))
  rotation = command.reference_alignment.rotation.clone()
  translation = command.reference_alignment.translation.clone()
  command._update_command()  # Exposes sampled frame on first update.
  command.robot.data.body_link_pos_w += 50.0
  command._update_command()
  assert command.time_steps.tolist() == [1, 1]
  assert torch.equal(command.reference_alignment.rotation, rotation)
  assert torch.equal(command.reference_alignment.translation, translation)
  relative_pos, relative_rot = _reanchor_targets(
    command.body_pos_w,
    command.body_quat_w,
    command.robot_anchor_pos_w,
    command.robot_anchor_quat_w,
  )
  torch.testing.assert_close(command.body_pos_relative_w, relative_pos)
  torch.testing.assert_close(
    matrix_from_quat(command.body_quat_relative_w), relative_rot
  )
  expected_velocity = (rotation @ command.motion.root_lin_vel_w[1, :, None]).squeeze(-1)
  torch.testing.assert_close(command.anchor_lin_vel_w, expected_velocity)
  expected_angular = (rotation @ command.motion.body_ang_vel_w[1, 0, :, None]).squeeze(
    -1
  )
  torch.testing.assert_close(command.anchor_ang_vel_w, expected_angular)
  expected_position = (
    (rotation[:, None] @ command.motion.body_pos_w[1].unsqueeze(-1)).squeeze(-1)
    + translation[:, None]
    + command._env.scene.env_origins[:, None]
  )
  torch.testing.assert_close(command.body_pos_w, expected_position)


@pytest.mark.parametrize(
  "offset,yaw", [(0.0, 0.0), (1.0, 0.0), (0.0, 0.7), (1.0, -0.6)]
)
def test_hybrid_reward_matches_simulation_with_fixed_reference(offset, yaw):
  command = _command()
  command._resample_command(torch.tensor([0, 1]))
  target_pos = command.body_pos_w
  target_quat = command.body_quat_w
  # The real reward manager runs before the command refresh for this step.
  command.update_relative_body_poses()
  cached_pos = command.body_pos_relative_w.clone()
  delta = torch.tensor(
    [torch.cos(torch.tensor(yaw / 2)), 0.0, 0.0, torch.sin(torch.tensor(yaw / 2))]
  )
  delta = delta.expand(2, 2, 4)
  command.robot.data.body_link_pos_w = (
    target_pos[:, :1]
    + quat_apply(delta, target_pos - target_pos[:, :1])
    + torch.tensor([offset, 0.0, 0.0])
  )
  command.robot.data.body_link_quat_w = quat_mul(delta, target_quat)
  # Shape error and velocity errors ensure equality is not just the reward ceiling.
  command.robot.data.body_link_pos_w[:, 1, 2] += 0.04
  command.robot.data.body_link_lin_vel_w = command.body_lin_vel_w + 0.1
  command.robot.data.body_link_ang_vel_w = command.body_ang_vel_w - 0.2
  command.robot.data.joint_pos = torch.full((2, 29), 0.2)
  command.robot.data.joint_pos[:, 0] = 10.2  # Exercise the joint-limit penalty.
  action = torch.full((2, 29), 0.2)
  last_action = torch.full((2, 29), 0.1)
  reward_env = SimpleNamespace(
    command_manager=SimpleNamespace(get_term=lambda _: command),
    action_manager=SimpleNamespace(action=action, prev_action=last_action),
    scene={"robot": command.robot},
  )
  cfg = configure_estimator_env_cfg(gencoord_env_cfg())
  if offset:
    assert not torch.allclose(cached_pos[:, 0], command.robot_anchor_pos_w)
  terms = {
    name: term.func(reward_env, **term.params) for name, term in cfg.rewards.items()
  }
  real_reward = 0.02 * sum(
    cfg.rewards[name].weight * value for name, value in terms.items()
  )
  origin = command._env.scene.env_origins
  imagined_reward = tracking_reward(
    torch.cat([command.robot_anchor_pos_w - origin, command.robot_anchor_quat_w], -1),
    command.robot_body_pos_w - origin[:, None],
    matrix_from_quat(command.robot_body_quat_w),
    command.robot_body_lin_vel_w,
    command.robot_body_ang_vel_w,
    torch.cat([command.robot.data.joint_pos, torch.zeros(2, 29)], -1),
    action,
    last_action,
    target_pos - origin[:, None],
    target_quat,
    command.body_lin_vel_w,
    command.body_ang_vel_w,
    command.robot.data.soft_joint_pos_limits[0],
    0.02,
  )
  torch.testing.assert_close(imagined_reward, real_reward, atol=1e-7, rtol=1e-5)
  assert (terms["motion_body_pos"] > 0.9).all()
  assert (terms["motion_body_ori"] > 0.99).all()
  if offset:
    assert (terms["motion_global_root_pos"] < 1e-4).all()
  if yaw:
    assert (terms["motion_global_root_ori"] < 0.2).all()
  # Scoring changes neither the actor/global reference nor its fixed transform.
  torch.testing.assert_close(command.body_pos_w, target_pos)
  torch.testing.assert_close(command.body_quat_w, target_quat)


def test_partial_reset_preserves_other_environment_alignment():
  command = _command()
  command._resample_command(torch.tensor([0, 1]))
  other_rotation = command.reference_alignment.rotation[1].clone()
  other_translation = command.reference_alignment.translation[1].clone()
  command.cfg.pose_range = {"yaw": (-0.8, -0.8), "x": (2.0, 2.0)}
  command._resample_command(torch.tensor([0]))
  assert torch.equal(command.reference_alignment.rotation[1], other_rotation)
  assert torch.equal(command.reference_alignment.translation[1], other_translation)
  assert not torch.allclose(command.reference_alignment.rotation[0], other_rotation)
  assert not torch.allclose(
    command.reference_alignment.translation[0], other_translation
  )


def test_reference_window_helpers_use_same_saved_transform():
  command = _command()
  command._resample_command(torch.tensor([0, 1]))
  position_window = command.motion.root_pos_w[None].repeat(2, 1, 1)
  quaternion_window = command.motion.root_quat_w[None].repeat(2, 1, 1)
  aligned_pos = command.align_reference_positions(position_window)
  aligned_quat = command.align_reference_quaternions(quaternion_window)
  torch.testing.assert_close(
    aligned_pos[:, 0] + command._env.scene.env_origins, command.anchor_pos_w
  )
  torch.testing.assert_close(aligned_quat[:, 0], command.anchor_quat_w)


def test_estimator_env_config_exposes_calibrated_clean_sensor_mean():
  cfg = gencoord_env_cfg(play=True)
  assert configure_estimator_env_cfg(cfg) is cfg
  assert cfg.commands["motion"].fixed_reference_alignment
  actor = cfg.observations["actor"]
  assert not actor.enable_corruption
  assert all(term.noise is None for term in actor.terms.values())
  assert actor.terms["base_ang_vel"].func is root_ang_vel
  assert actor.terms["joint_vel"].func is joint_vel
  assert actor.terms["joint_pos"].params["biased"] is False


def test_estimator_config_rejects_maxcoord():
  with pytest.raises(ValueError, match="GenCoord"):
    configure_estimator_env_cfg(maxcoord_env_cfg(play=True))
