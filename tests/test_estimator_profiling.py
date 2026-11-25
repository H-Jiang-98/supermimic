"""Auditable training budgets and post-commit timings retain resume guarantees."""

import json
import math
from dataclasses import replace
from unittest.mock import Mock

import torch

from supermimic.utils.profiling import PhaseTimer, StreamTimer, runtime_metadata
from tests.test_estimator_belief_training import training as model_training
from tests.test_estimator_eit_training import training


def test_step_budget_counts_actual_updates_and_cache_reads(tmp_path, monkeypatch):
  monkeypatch.setenv("WANDB_MODE", "disabled")
  loop = training()
  loop.policy.cfg = replace(loop.policy.cfg, num_batches=2)
  rows = loop.run(4, tmp_path)
  row = rows[-1]
  assert row["Counters/collected_frames"] == 8
  assert row["Budget/mean_updates"] == 1
  assert row["Budget/covariance_updates"] == 2
  assert row["Budget/covariance_pair_draws"] == 4
  assert row["Budget/actor_updates"] == row["Budget/critic_updates"] == 2
  assert row["Budget/policy_starts_drawn"] == 4
  assert row["Budget/imagined_steps"] == 16
  assert 0 < row["Budget/survival_weighted_imagined_steps"] <= 16
  assert row["Budget/normalization_samples"] == 6
  profiles = [
    json.loads(line) for line in (tmp_path / "profiling.jsonl").read_text().splitlines()
  ]
  seconds = profiles[-1]["seconds"]
  for name in (
    "physical_collection_seconds",
    "mean_rollout_update_seconds",
    "residual_extraction_seconds",
    "covariance_fit_seconds",
    "posterior_cache_normalization_seconds",
    "policy_rollout_backward_seconds",
    "critic_regression_seconds",
    "checkpoint_save_seconds",
  ):
    assert math.isfinite(seconds[name]) and seconds[name] > 0
  assert profiles[-1]["runtime"]["dtype"] == "torch.float64"
  assert profiles[-1]["runtime"]["cuda_peak_allocated_bytes"] is None


def test_collection_budget_excludes_previous_round_and_refresh_boundaries():
  loop = model_training()
  first, second = loop.run_round(), loop.run_round()
  assert (
    first["budget"]["collected_frames"] == second["budget"]["collected_frames"] == 8
  )
  assert second["budget"]["physical_episode_endings"] == 0
  assert second["budget"]["mean_updates"] == 1
  assert first["profiling"]["warmup_rounds_in_process"] == 0
  assert second["profiling"]["warmup_rounds_in_process"] == 1


def test_cpu_profiling_does_not_consume_randomness_or_touch_cuda(monkeypatch):
  def unexpected(*args, **kwargs):
    raise AssertionError("CPU profile touched CUDA")

  monkeypatch.setattr(torch.cuda, "synchronize", unexpected)
  monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", unexpected)
  rng = torch.random.get_rng_state().clone()
  timer = PhaseTimer("cpu")
  with timer.measure("work"):
    runtime_metadata(torch.zeros(1), reset_peak=True)
  torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
  assert timer.seconds["work"] > 0


def test_cache_timing_waits_only_for_the_final_stream_event(monkeypatch):
  stream = object()
  monkeypatch.setattr(torch.cuda, "current_stream", lambda device: stream)
  monkeypatch.setattr(torch.cuda, "synchronize", Mock(side_effect=AssertionError))
  events = [Mock() for _ in range(6)]
  for event in events:
    event.elapsed_time.return_value = 2.0
  monkeypatch.setattr(torch.cuda, "Event", Mock(side_effect=events))
  timer = StreamTimer("cuda:0")
  with timer.measure("normalization"):
    for _ in range(2):
      with timer.measure("read"):
        pass
  for event in events:
    event.record.assert_called_once_with(stream)
    event.synchronize.assert_not_called()
  assert timer.seconds() == {"normalization": 0.002, "read": 0.004}
  events[1].synchronize.assert_called_once_with()
  assert sum(event.synchronize.call_count for event in events) == 1
