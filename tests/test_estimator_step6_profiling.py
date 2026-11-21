"""Auditable per-round budgets and post-commit timings retain resume guarantees."""

import json
import math
from dataclasses import replace

import torch

from supermimic.utils.profiling import PhaseTimer, runtime_metadata
from tests.test_estimator_belief_training import training as model_training
from tests.test_estimator_eit_training import training


def test_round_budget_counts_actual_updates_and_prefix_reconstruction(tmp_path):
  loop = training()
  loop.policy.cfg = replace(loop.policy.cfg, num_batches=2)
  [row] = loop.run(1, tmp_path)
  budget = row["budget"]
  assert budget["collected_frames"] == 8
  assert budget["available_training_windows"] == 2
  assert budget["sampled_training_windows"] == 2
  assert budget["training_pairs"] == budget["validation_pairs"] == 3
  assert budget["mean_updates"] == 1
  assert budget["covariance_updates"] == math.ceil(3 / 2)
  assert budget["covariance_pair_draws"] == 3
  assert budget["actor_updates"] == budget["critic_updates"] == 2
  assert budget["policy_starts_drawn"] == 4
  assert budget["imagined_steps"] == 16
  assert 0 < budget["survival_weighted_imagined_steps"] <= 16
  assert budget["posterior_prefix_episodes"] == 1
  assert budget["posterior_prefix_frames"] == 4
  assert budget["posterior_prefix_length_max"] == 4
  assert budget["physical_episode_endings"] == 0
  state = torch.load(tmp_path / "training.pt", weights_only=True)
  assert state["last_metrics"] == row
  [profile] = [
    json.loads(line) for line in (tmp_path / "profiling.jsonl").read_text().splitlines()
  ]
  seconds = profile["seconds"]
  for name in (
    "physical_collection_seconds",
    "mean_rollout_update_seconds",
    "residual_extraction_seconds",
    "covariance_fit_seconds",
    "posterior_prefix_normalization_seconds",
    "policy_rollout_backward_seconds",
    "critic_regression_seconds",
    "checkpoint_save_seconds",
    "snapshot_and_metrics_seconds",
    "round_seconds",
  ):
    assert math.isfinite(seconds[name]) and seconds[name] > 0
  assert seconds["round_seconds"] >= sum(
    value for name, value in seconds.items() if name != "round_seconds"
  )
  assert profile["runtime"]["dtype"] == "torch.float64"
  assert profile["runtime"]["cuda_peak_allocated_bytes"] is None


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
