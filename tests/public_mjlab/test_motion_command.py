from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from supermimic.motion_command import MotionCommand, MotionCommandCfg, motion_end


@pytest.fixture
def command(g1_model, motion_file):
  names = [g1_model.body(i).name for i in range(1, g1_model.nbody)]
  joints = np.where(g1_model.jnt_type == 3)[0]
  num_envs = 4
  robot = SimpleNamespace(
    num_joints=29,
    body_names=names,
    find_bodies=lambda requested, **_: (
      [names.index(n) for n in requested],
      list(requested),
    ),
    indexing=SimpleNamespace(
      body_ids=torch.arange(1, g1_model.nbody),
      joint_q_adr=torch.tensor(g1_model.jnt_qposadr[joints]),
    ),
    data=SimpleNamespace(
      soft_joint_pos_limits=torch.tensor([[-10.0, 10.0]]).repeat(num_envs, 29, 1)
    ),
    write_joint_state_to_sim=MagicMock(),
    write_root_state_to_sim=MagicMock(),
    reset=MagicMock(),
  )
  env = SimpleNamespace(
    num_envs=num_envs,
    device="cpu",
    step_dt=0.02,
    scene={"robot": robot},
    sim=SimpleNamespace(mj_model=g1_model, forward=MagicMock()),
  )

  # The scene's mapping and env origins are both used by the public command.
  class Scene(dict):
    env_origins = torch.zeros(num_envs, 3)

  env.scene = Scene(env.scene)
  cfg = MotionCommandCfg(
    motion_file=[motion_file("a.npz", frames=103), motion_file("b.npz", frames=52)],
    entity_name="robot",
    anchor_body_name="pelvis",
    body_names=tuple(names),
    resampling_time_range=(1e9, 1e9),
    sampling_mode="external",
    pose_range={},
    velocity_range={},
    joint_position_range=(0.0, 0.0),
  )
  cmd = cfg.build(env)
  cmd.update_relative_body_poses = MagicMock()
  env.command_manager = SimpleNamespace(get_term=lambda _: cmd)
  return cmd


def test_constructor_uses_extended_loader(command):
  assert isinstance(command, MotionCommand)
  assert command.motion.motion_ends.tolist() == [103, 155]
  assert command.motion.fk is not None


def test_external_bins_match_return_sampler_and_include_final_frame(command):
  ids = torch.arange(command.num_envs)
  for b in range(command.bin_count):
    weights = torch.zeros(command.bin_count)
    weights[b] = 1
    command.set_sampling_distribution(weights)
    samples = []
    for _ in range(50):
      command._sample_time_steps(ids)
      samples.append(command.time_steps.clone())
    samples = torch.cat(samples)
    bins = samples * command.bin_count // command.motion.time_step_total
    assert (bins == b).all()
  with patch("torch.rand", return_value=torch.full((len(ids),), 0.999999)):
    command._sample_time_steps(ids)
  assert (command.time_steps == 154).all()


@pytest.mark.parametrize(
  "bad",
  [
    [0, 0, 0, 0],
    [-1, 2, 1, 1],
    [float("nan"), 1, 1, 1],
    [float("inf"), 1, 1, 1],
    [torch.finfo(torch.float32).max] * 4,
    [1, 2],
  ],
)
def test_invalid_external_distribution_is_rejected(command, bad):
  with pytest.raises(ValueError):
    command.set_sampling_distribution(torch.tensor(bad, dtype=torch.float32))


def test_distribution_is_copied_and_normalized(command):
  weights = torch.arange(1, command.bin_count + 1, dtype=torch.float32)
  expected = weights / weights.sum()
  command.set_sampling_distribution(weights)
  weights.zero_()
  torch.testing.assert_close(command._sampling_weights, expected)


def test_reset_does_not_skip_sampled_frame_or_advance_other_envs(command):
  command.time_steps[:] = torch.tensor([10, 20, 30, 40])
  command.cfg.sampling_mode = "start"
  command.reset(torch.tensor([1, 3]))
  command._update_command(torch.tensor([1, 3]))
  assert command.time_steps.tolist() == [10, 0, 30, 0]
  command._update_command()
  assert command.time_steps.tolist() == [11, 1, 31, 1]


def test_auto_reset_keeps_new_reference_frame(command):
  command.time_steps[:] = torch.tensor([10, 20, 30, 40])
  command.cfg.sampling_mode = "start"
  command.reset(torch.tensor([1]))
  command._update_command()
  assert command.time_steps.tolist() == [11, 0, 31, 41]


def test_clip_end_is_truncation_without_cross_clip_jump(command):
  command.time_steps[:] = torch.tensor([101, 102, 153, 154])
  assert motion_end(command._env).tolist() == [False, True, False, True]
  command._update_command()
  assert command.time_steps.tolist() == [102, 102, 154, 154]
  assert motion_end(command._env).all()


def test_public_uniform_and_start_modes_still_work(command):
  ids = torch.arange(command.num_envs)
  command.cfg.sampling_mode = "uniform"
  command._sample_time_steps(ids)
  assert ((command.time_steps >= 0) & (command.time_steps < 155)).all()
  command.cfg.sampling_mode = "start"
  command._sample_time_steps(ids)
  assert (command.time_steps == 0).all()
  with pytest.raises(ValueError, match="external"):
    command.set_sampling_distribution(torch.ones(command.bin_count))
