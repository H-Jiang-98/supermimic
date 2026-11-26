"""Ratio parity, warmup, bounded fitting and once-only physical statistics."""

from copy import deepcopy
from dataclasses import replace
from unittest.mock import Mock

import pytest
import torch
from tensordict import TensorDict

from supermimic.learners.two_stage_wm import CovarianceFitCfg, MeanFitCfg
from supermimic.runners.supermimic_runner import Ratio
from tests.test_estimator_eit_training import training
from tests.test_estimator_two_stage_wm import fixture, learner_for


@pytest.mark.parametrize("ratio", [0, -1, 0.1, 0.4, 1, 2.5])
def test_ratio_resume_replays_fractional_balance_and_first_trigger(ratio):
  original = Ratio(ratio)
  counts = [original(step) for step in range(7)]
  restored = Ratio(ratio)
  restored.load_state_dict(original.state_dict())
  assert [original(s) for s in range(7, 50)] == [restored(s) for s in range(7, 50)]
  assert counts[0] == (0 if ratio == 0 else 1)
  if ratio == 2.5:
    assert counts == [1, 2, 3, 2, 3, 2, 3]


def test_warmup_discards_opportunities_and_multigroup_does_not_recount_normalization(
  monkeypatch,
):
  loop = training()
  replay = loop.runner.replay_buffer
  enumerate_starts = Mock(wraps=replay.model_fit_starts)
  monkeypatch.setattr(replay, "model_fit_starts", enumerate_starts)
  loop.should_train = Ratio(2.5)
  counts = []
  for _ in range(4):
    row = loop.run_step()
    counts.append(row["groups_this_step"])
  assert counts == [0, 0, 3, 2]
  assert enumerate_starts.call_count == 4
  assert loop.skipped_update_groups == 3
  assert loop.update_groups == 5
  assert loop.learner.mean_steps == loop.policy.actor_steps == 5
  assert int(loop.actor.obs_normalizer.count) == 8
  assert int(loop.learner.dynamics.state_normalizer.count) == 4
  assert loop.runner.posterior_provider.state.time_index.tolist() == [3, 3]
  assert loop.runner.reset_count == 1


def test_zero_ratio_collects_and_normalizes_without_training():
  loop = training(buffer_size=2, replay_ratio=0)
  for _ in range(10):
    loop.run_step()
    assert loop.runner.replay_buffer._round_index is None
  assert loop.update_groups == loop.learner.cov_steps == 0
  assert int(loop.actor.obs_normalizer.count) == 20
  assert int(loop.learner.dynamics.state_normalizer.count) == 10


@pytest.mark.parametrize("ratio", [0.1, -1])
def test_enabled_training_requires_replay_to_fit_a_mean_window(ratio):
  with pytest.raises(ValueError, match=r"buffer_size \(2\).*mean.window_frames \(3\)"):
    training(buffer_size=2, replay_ratio=ratio)


def test_default_covariance_budget_and_batch_sizes_are_independent_of_capacity(
  monkeypatch,
):
  estimator, replay = fixture(episodes=(10, 9), buffer_size=24)
  fit = learner_for(estimator.dynamics)
  fit.mean_cfg = MeanFitCfg()
  fit.cov_cfg = CovarianceFitCfg()
  sizes, residual_sizes = [], []

  def mean(batch):
    fit.mean_steps += 1
    return {"Loss/state": torch.tensor(0.0)}

  def residual(pairs):
    residual_sizes.append(len(pairs))
    # Only bounded samples may be extracted, never the entire replay.
    return TensorDict({"residual": torch.ones(len(pairs), 1)}, [len(pairs)])

  def cov(batch):
    sizes.append(len(batch))
    fit.cov_steps += 1
    return torch.tensor(float(len(batch)))

  monkeypatch.setattr(fit, "update_mean", mean)
  monkeypatch.setattr(fit, "residual_data", residual)
  monkeypatch.setattr(fit, "update_covariance", cov)
  monkeypatch.setattr(fit, "covariance_loss", lambda b: b["residual"].flatten())
  monkeypatch.setattr(
    fit.mean,
    "batch_loss",
    lambda b: (torch.tensor(0.0), torch.zeros(len(fit.mean.loss_group_weights))),
  )
  train, validation = fit.fit(replay, [0], [1], replay.retained_frames([0])[:0], 2)
  assert sizes == [8192] * 17 + [4096]
  assert residual_sizes == sizes + [8192]
  assert train["mean_updates"] == 10 and train["cov_updates"] == 18
  assert train["cov_pair_draws"] == 143360
  assert train["Loss/process_nll"] == (17 * 8192**2 + 4096**2) / 143360
  assert validation["cov_pair_draws"] == 8192
  assert (
    CovarianceFitCfg().training_pairs(
      MeanFitCfg(batch_size=10, num_batches=2, window_frames=5)
    )
    == 80
  )


def test_fit_rng_isolation_and_restore(monkeypatch):
  estimator, replay = fixture(episodes=(6, 4))
  fit = learner_for(estimator.dynamics)
  fit.cov_cfg = CovarianceFitCfg(batch_size=2, pair_budget=7, validation_pairs=4)
  state = fit.state_dict()
  model = deepcopy(fit.dynamics.state_dict())
  fresh = replay.retained_frames([0])[:0]
  a = fit.fit(replay, [0], [1], fresh, 2)
  expected = fit.state_dict()
  fit.dynamics.load_state_dict(model)
  fit.load_state_dict(state)
  torch.manual_seed(9876)
  b = fit.fit(replay, [0], [1], fresh, 2)
  assert a == b
  for name in expected["rng"]:
    assert torch.equal(fit.state_dict()["rng"][name], expected["rng"][name])
  fit.dynamics.load_state_dict(model)
  fit.load_state_dict(state)
  fit.cov_cfg = replace(fit.cov_cfg, validation_pairs=9)
  fit.fit(replay, [0], [1], fresh, 2)
  assert torch.equal(
    fit.generators["cov_train"].get_state(), expected["rng"]["cov_train"]
  )


def test_restore_installs_return_distribution_before_first_new_reset(monkeypatch):
  from types import SimpleNamespace

  from supermimic.runners.reset_sampling import ReturnSampler, ReturnSamplerCfg
  from tests.test_estimator_eit_training import advance

  source, target = training(), training(81)
  for loop in (source, target):
    loop.runner._sampler = ReturnSampler(ReturnSamplerCfg(horizon=2), 4, 40, 2, 1.0)
    loop.runner.motion_cmd = SimpleNamespace(set_sampling_distribution=Mock())
  advance(source)
  sampler = source.runner._sampler
  for _ in range(3):
    sampler.record(
      torch.tensor([1, 12]),
      torch.tensor([0.1, 0.5]),
      torch.ones(2, dtype=torch.bool),
      torch.zeros(2, dtype=torch.bool),
      torch.zeros(2, dtype=torch.bool),
    )
  state = source.state_dict()
  reset = target.runner.reset_collection

  def checked_reset():
    target.runner.motion_cmd.set_sampling_distribution.assert_called_once()
    torch.testing.assert_close(
      target.runner._sampler.weights, sampler.weights, rtol=0, atol=0
    )
    assert target.runner._sampler._step == 0
    reset()

  monkeypatch.setattr(target.runner, "reset_collection", checked_reset)
  target.load_state_dict(state)
  assert target.runner.replay_buffer.insert_count == 0
  assert target.runner.posterior_provider.state is None
  assert target.should_train.state_dict() == source.should_train.state_dict()


@pytest.mark.parametrize("field", ["scheduler", "collection", "model_rng", "sampler"])
def test_checkpoint_preflight_rejects_schedule_rng_or_counter_without_mutation(field):
  from tests.test_estimator_belief_training import assert_tree
  from tests.test_estimator_eit_training import advance

  source, target = training(), training(81)
  advance(source)
  saved = source.state_dict()
  if field == "scheduler":
    saved["scheduler"]["prev"] = float("nan")
  elif field == "collection":
    saved["collected_steps"] += 1
  elif field == "model_rng":
    saved["learner"]["rng"]["cov_train"] = torch.zeros(4, dtype=torch.uint8)
  else:
    saved["return_sampler"] = {}
  before = target.state_dict()
  with pytest.raises((RuntimeError, ValueError)):
    target.load_state_dict(saved)
  assert_tree(target.state_dict(), before)
