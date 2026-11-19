"""Explicit collector resets preserve replay and return-sampling episode boundaries."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from tensordict import TensorDict

from supermimic.replay_buffer import FieldSpec, ReplayBuffer
from supermimic.runners.reset_sampling import ReturnSampler, ReturnSamplerCfg
from supermimic.runners.supermimic_runner import InputStagePolicy, build_actor
from tests.test_estimator_runner import _PhysicalEnv, _runner
from tests.test_estimator_world_model_env import BATCH, _MeasurementPosterior


@pytest.mark.parametrize("steps", [1, 10])
def test_manual_collector_reset_restores_exact_new_episode_history(steps):
  runner = _runner("B", provider=_MeasurementPosterior())
  runner.env = _PhysicalEnv(runner)
  runner.action_manager = runner.env.action_manager
  runner.replay_buffer = ReplayBuffer(runner._buffer_schema(), 8, BATCH)
  runner._pending_norm_updates = []
  runner.world_model_learner = SimpleNamespace(update_normalization=MagicMock())
  runner.logger = SimpleNamespace(process_env_step=MagicMock())
  runner._sampler = None
  runner.policy_learner = InputStagePolicy(
    build_actor(runner.cfg, runner._wm, "cpu"), runner.cfg.policy
  )
  runner.reset_collection()
  for _ in range(steps):
    runner.step()
  old_last = (runner.replay_buffer._step - 1) % 8
  terminated = runner.replay_buffer._storage["terminated"][:, old_last].clone()
  count = runner.replay_buffer.num_transitions
  runner.reset_collection()
  assert runner.replay_buffer.num_transitions == count
  assert torch.equal(
    runner.replay_buffer._storage["terminated"][:, old_last], terminated
  )
  assert torch.equal(
    runner.replay_buffer._storage["truncated"][:, old_last], ~terminated
  )
  actual_history = runner._sim_history.clone()
  runner.step()
  newest = runner.replay_buffer.num_transitions - 1
  with patch(
    "torch.randint", side_effect=[torch.arange(BATCH), torch.full((BATCH,), newest)]
  ):
    history, padded = runner.replay_buffer.sample_causal(5, BATCH)
  assert torch.equal(history["actor_obs"], actual_history)
  assert padded[:, :-1].all() and not padded[:, -1].any()


def test_buffer_truncation_is_idempotent_preserves_ends_and_excludes_forward_crossing():
  buffer = ReplayBuffer(
    {
      "state": FieldSpec(1),
      "terminated": FieldSpec(dtype=torch.bool),
      "truncated": FieldSpec(dtype=torch.bool),
    },
    8,
    3,
  )
  buffer.truncate_open_episodes()  # Empty buffer is unchanged.
  assert buffer.num_transitions == 0
  for frame in range(10):
    buffer.insert(
      TensorDict(
        {
          "state": torch.full((3, 1), float(frame)),
          "terminated": torch.tensor([frame == 9, False, False]),
          "truncated": torch.tensor([False, False, frame == 9]),
        },
        batch_size=[3],
      )
    )
  buffer.truncate_open_episodes()
  buffer.truncate_open_episodes()
  assert buffer._storage["terminated"][:, 1].tolist() == [True, False, False]
  assert buffer._storage["truncated"][:, 1].tolist() == [False, True, True]
  for frame in [20, 21]:
    buffer.insert(
      TensorDict(
        {
          "state": torch.full((3, 1), float(frame)),
          "terminated": torch.zeros(3, dtype=torch.bool),
          "truncated": torch.zeros(3, dtype=torch.bool),
        },
        batch_size=[3],
      )
    )
  torch.manual_seed(7)
  sequences = buffer.sample(2, 128)["state"].squeeze(-1)
  assert torch.equal(sequences[:, 1] - sequences[:, 0], torch.ones(128))
  assert (sequences[:, 0] == 20).any()


def test_manual_reset_also_cuts_pending_return_sampler_window():
  sampler = ReturnSampler(ReturnSamplerCfg(gamma=0.9, horizon=3), 2, 30, 3, 1.0)
  sampler.truncate_open_episodes()
  sampler.record(
    torch.zeros(3, dtype=torch.long),
    torch.ones(3),
    torch.tensor([True, False, False]),
    torch.tensor([False, False, True]),
    torch.zeros(3, dtype=torch.bool),
  )
  runner = _runner()
  runner.env = SimpleNamespace(reset=MagicMock())
  runner.replay_buffer = ReplayBuffer(runner._buffer_schema(), 8, BATCH)
  runner._seed_sim_history = MagicMock()
  runner._sampler = sampler
  runner.reset_collection()
  assert sampler._timed_out[:, 0].tolist() == [False, True, False]
  sampler.record(
    torch.full((3,), 20),
    torch.full((3,), 100.0),
    torch.zeros(3, dtype=torch.bool),
    torch.zeros(3, dtype=torch.bool),
    torch.zeros(3, dtype=torch.bool),
  )
  # New-episode rewards cannot flow into the pre-reset return.
  torch.testing.assert_close(sampler._returns()[:, 0], torch.tensor([1.0, 10.0, 10.0]))


def test_manual_reset_can_clear_simulator_caches_created_during_collection():
  with torch.inference_mode():
    contact_history = torch.ones(BATCH, 2)

  runner = _runner()
  runner.env = SimpleNamespace(reset=lambda: contact_history.zero_())
  runner.replay_buffer = ReplayBuffer(runner._buffer_schema(), 8, BATCH)
  runner._seed_sim_history = MagicMock()
  runner._sampler = None
  runner.reset_collection()
  assert torch.count_nonzero(contact_history) == 0
  runner._seed_sim_history.assert_called_once()
