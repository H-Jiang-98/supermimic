"""Replay boundaries, physical residuals, and the two independent optimizers."""

from copy import deepcopy
from typing import Literal
from unittest.mock import Mock

import pytest
import torch
from tensordict import TensorDict

from supermimic.estimator import LieBeliefFilter
from supermimic.estimator_inputs import GaussianMeasurementConfig
from supermimic.learners.two_stage_wm import (
  CovarianceFitCfg,
  MeanFitCfg,
  TwoStageWorldModelLearner,
  process_nll,
)
from supermimic.learners.wm_learner import WorldModelLearner, WorldModelLearnerCfg
from supermimic.models import Dynamics, ProcessNoiseConfig
from supermimic.replay_buffer import MODEL_REPLAY_FIELDS, FieldSpec, ReplayBuffer
from supermimic.representations import GenCoordRepresentation
from supermimic.utils.lie_state import LieState, boxplus


def fixture(
  *,
  device="cpu",
  dtype=torch.float64,
  buffer_size=12,
  episodes=(5, 4),
  seed=31,
  replay_mode: Literal["full", "model"] = "full",
  seal=True,
):
  torch.manual_seed(seed)
  representation = GenCoordRepresentation(
    29, torch.zeros(29, device=device, dtype=dtype)
  )
  representation.dt = 0.02
  representation.substep_coeff = 0.5
  dynamics = (
    Dynamics(
      representation,
      29,
      2,
      torch.ones(29, device=device, dtype=dtype),
      torch.zeros(29, device=device, dtype=dtype),
      (12, 12),
      ProcessNoiseConfig(hidden_dims=(8,)),
    )
    .to(device=device, dtype=dtype)
    .eval()
  )
  estimator = LieBeliefFilter(
    dynamics, GaussianMeasurementConfig(), torch.zeros(29, device=device, dtype=dtype)
  )
  schema = {
    "state": FieldSpec(71, dtype),
    "actor_obs": FieldSpec(93, dtype),
    "action": FieldSpec(29, dtype),
    "prev_action": FieldSpec(29, dtype),
    "sim_ctx": FieldSpec(2, dtype),
    "episode_id": FieldSpec(dtype=torch.long),
    "time_index": FieldSpec(dtype=torch.long),
    "terminated": FieldSpec(dtype=torch.bool),
    "truncated": FieldSpec(dtype=torch.bool),
  }
  if replay_mode == "model":
    schema = {key: spec for key, spec in schema.items() if key in MODEL_REPLAY_FIELDS}
  replay = ReplayBuffer(schema, buffer_size, 1, device, replay_mode=replay_mode)
  for episode, length in enumerate(episodes):
    previous = torch.zeros(1, 29, device=device, dtype=dtype)
    for index in range(length):
      state = torch.zeros(1, 71, device=device, dtype=dtype)
      state[:, 2] = 0.8
      state[:, 3] = 1
      state[:, 0] = index * 0.002
      state[:, 36] = 0.1
      state[:, 7:36] = 0.02 * torch.sin(
        torch.arange(29, device=device, dtype=dtype) + index * 0.01
      )
      action = torch.full_like(previous, index * 0.03 + 0.01)
      observation = estimator.observation(LieState.from_full(state))
      noise = torch.randn_like(
        observation
      ) * estimator.measurement_config.standard_deviations(
        29, device=device, dtype=dtype
      )
      frame = TensorDict(
        {
          "state": state,
          "actor_obs": torch.cat((observation + noise, previous), -1),
          "action": action,
          "prev_action": previous,
          "sim_ctx": torch.full((1, 2), episode * 0.1, device=device, dtype=dtype),
          "episode_id": torch.tensor([episode], device=device),
          "time_index": torch.tensor([index], device=device),
          "terminated": torch.tensor([index == length - 1], device=device),
          "truncated": torch.zeros(1, device=device, dtype=torch.bool),
        },
        [1],
      )
      replay.insert(frame.select(*schema))
      previous = action
  if seal:
    replay.finish_collection_round()
  return estimator, replay


def learner_for(model, mean=None, cov=None):
  return TwoStageWorldModelLearner(
    model,
    model._apply_prediction.__self__,
    mean or MeanFitCfg(window_frames=3, batch_size=2, num_batches=1),
    cov or CovarianceFitCfg(batch_size=2, validation_pairs=4),
    str(model.action_scale.device),
  )


def test_free_running_mean_matches_original_predictions_group_losses_and_gradients():
  estimator, replay = fixture()
  fit = learner_for(estimator.dynamics)
  model = fit.dynamics
  rep = fit.mean.representation
  reference = Dynamics(
    rep, 29, 2, model.action_scale, model.action_offset, (12, 12)
  ).double()
  reference.mlp.load_state_dict(model.mlp.state_dict())
  original = WorldModelLearner(
    reference, rep, WorldModelLearnerCfg(), cuda_graphs=False, compile_kernels=False
  )
  batch = replay.model_windows(replay.model_window_starts([0, 1], 4), 4)
  for learner in (original, fit.mean):
    learner.update_normalization(batch["state"], batch["action"])

  def predictions(learner):
    return learner._rollout(
      rep.represent(batch["state"][:, 0]),
      batch["action"],
      rep.extract_aux_state(batch["state"][:, 0]),
      batch["sim_ctx"][:, 0],
    )

  model.mlp.requires_grad_(True)
  for actual, expected in zip(
    predictions(fit.mean), predictions(original), strict=True
  ):
    assert actual.shape[1] == 3
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
  actual, actual_groups = fit.mean.batch_loss(batch)
  expected, expected_groups = original.batch_loss(batch)
  for x, y in ((actual, expected), (actual_groups, expected_groups)):
    torch.testing.assert_close(x, y, rtol=0, atol=0)
  actual.backward()
  expected.backward()
  for a, b in zip(model.mlp.parameters(), reference.mlp.parameters(), strict=True):
    torch.testing.assert_close(a.grad, b.grad, rtol=1e-12, atol=1e-14)
  before = deepcopy(fit.noise_head.state_dict())
  fit.update_mean(batch)
  original.update([batch])
  for a, b in zip(model.mlp.parameters(), reference.mlp.parameters(), strict=True):
    torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-14)
  assert all(torch.equal(v, before[k]) for k, v in fit.noise_head.state_dict().items())


@pytest.mark.parametrize("failure", ["loss", "backward"])
def test_invalid_mean_update_preserves_weights_steps_and_both_optimizers(failure):
  estimator, replay = fixture()
  fit = learner_for(estimator.dynamics)
  windows = replay.model_windows(replay.model_window_starts([0], 3), 3)
  fit.update_mean(windows)
  before = fit.state_dict()
  if failure == "loss":
    windows["sim_ctx"][0, 0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="Non-finite optimizer loss"):
      fit.update_mean(windows)
  else:
    # The forward loss is finite. Reject the entire step if one gradient fails.
    bias = fit.dynamics.mlp[-1].bias
    with bias.register_hook(lambda grad: torch.full_like(grad, float("nan"))):
      with pytest.raises(RuntimeError, match="non-finite"):
        fit.update_mean(windows)
  assert fit.dynamics.model_version == before["model_version"]
  assert fit.mean_steps == before["mean"]["steps"]
  for optimizer, previous in zip(
    fit.mean.optimizers, before["mean"]["optimizers"], strict=True
  ):
    torch.testing.assert_close(
      optimizer.state_dict()["state"], previous["state"], rtol=0, atol=0
    )
  assert all(parameter.grad is None for parameter in fit.dynamics.parameters())


def test_invalid_normalization_update_restores_both_statistics():
  estimator, replay = fixture()
  fit = learner_for(estimator.dynamics)
  batch = replay.model_windows(replay.model_window_starts([0], 2), 2)
  states, actions = batch["state"][:, 0], batch["action"][:, 0]
  before = deepcopy(fit.dynamics.state_dict())
  # Both inputs are finite. Only the action variance overflows on update.
  actions = actions.clone()
  actions[0, 0] = 1e200
  with pytest.raises(ValueError, match="Non-finite model normalization"):
    fit.update_normalization(states, actions)
  assert all(
    torch.equal(value, before[k]) for k, value in fit.dynamics.state_dict().items()
  )
  assert all(not normalizer.training for normalizer in fit.dynamics.normalizers())


def test_covariance_targets_are_right_local_and_full_factor_nll_is_normalized():
  estimator, replay = fixture()
  fit = learner_for(estimator.dynamics)
  pairs = replay.model_windows(replay.model_window_starts([0], 2), 2)
  truth = LieState.from_full(pairs["state"][:, 0])
  action, context = pairs["action"][:, 0], pairs["sim_ctx"][:, 0]
  mean = fit.dynamics.mean_lie(truth, action, context)
  delta = torch.linspace(-0.07, 0.08, 70, dtype=action.dtype).expand(len(pairs), -1)
  pairs["state"][:, 1] = boxplus(mean, delta).to_full()
  data = fit.residual_data(pairs)
  torch.testing.assert_close(data["residual"], delta, atol=1e-12, rtol=1e-12)
  assert all(not x.requires_grad and x.grad_fn is None for x in data.values())
  with torch.no_grad():
    fit.noise_head[-1].bias[1] = 0.4
  _, factor = fit.dynamics.distribution_lie(truth, action, context)
  assert (factor.tril(-1) != 0).any()
  expected = -torch.distributions.MultivariateNormal(
    torch.zeros_like(delta), scale_tril=factor
  ).log_prob(delta)
  torch.testing.assert_close(fit.covariance_loss(data), expected)
  torch.testing.assert_close(process_nll(delta, factor), expected)


def test_covariance_phase_preserves_mean_predictions_and_statistics_bitwise():
  estimator, replay = fixture()
  fit = learner_for(estimator.dynamics)
  pairs = replay.model_windows(replay.model_window_starts([0, 1], 2), 2)
  fit.update_normalization(pairs["state"][:, 0], pairs["action"][:, 0])
  fit.update_mean(pairs)
  model = fit.dynamics
  fixed = (
    LieState.from_full(pairs["state"][:, 0]),
    pairs["action"][:, 0],
    pairs["sim_ctx"][:, 0],
  )
  prediction = model.mean_lie(*fixed).to_full().detach()
  before = deepcopy(model.state_dict())
  data = fit.residual_data(pairs)
  start_loss = float(fit.covariance_loss(data).mean().detach())
  for _ in range(4):
    fit.update_covariance(data)
  assert float(fit.covariance_loss(data).mean().detach()) < start_loss
  assert torch.equal(prediction, model.mean_lie(*fixed).to_full())
  assert all(
    torch.equal(value, before[name])
    for name, value in model.state_dict().items()
    if not name.startswith("noise_head.")
  )
  assert any(
    not torch.equal(value, before["noise_head." + name])
    for name, value in fit.noise_head.state_dict().items()
  )
  assert all(not p.requires_grad for p in model.mlp.parameters())
  assert all(not normalizer.training for normalizer in model.normalizers())


def test_covariance_rejects_finite_gradients_that_overflow_adam_second_moments():
  estimator, replay = fixture(dtype=torch.float32)
  fit = learner_for(estimator.dynamics)
  pairs = replay.model_windows(replay.model_window_starts([0], 2), 2)
  data = fit.residual_data(pairs)
  data["residual"].fill_(0.01)
  fit.update_covariance(data)
  output = fit.noise_head[-1]
  rows, cols = torch.tril_indices(70, 70)
  with torch.no_grad():
    output.weight.zero_()
    output.bias[rows != cols] = -0.5
  loss = fit.covariance_loss(data).mean()
  (gradient,) = torch.autograd.grad(loss, output.bias)
  assert torch.isfinite(loss) and torch.isfinite(gradient).all()
  assert not torch.isfinite(gradient.square()).all()

  before = fit.state_dict()
  with pytest.raises(FloatingPointError, match="squared process covariance gradient"):
    fit.update_covariance(data)
  assert fit.dynamics.model_version == before["model_version"]
  assert fit.cov_steps == before["cov"]["steps"]
  torch.testing.assert_close(
    fit.cov_optimizer.state_dict()["state"],
    before["cov"]["optimizer"]["state"],
    rtol=0,
    atol=0,
  )
  fit.load_state_dict(fit.state_dict())


def test_nonfinite_held_out_predictions_cannot_be_published_as_fit_metrics():
  estimator, replay = fixture(episodes=(4, 2))
  fit = learner_for(estimator.dynamics)
  fresh = replay.retained_frames([0])
  # Only the covariance validation path can score this short held-out episode.
  replay._storage["state"][0, 5, 7] = float("nan")
  with pytest.raises(
    FloatingPointError, match="Non-finite validation metric Loss/process_nll"
  ):
    fit.fit(replay, [0], [1], fresh, 3)


def test_sealed_windows_and_pairs_respect_ring_cut_short_episodes_and_missing_successors():
  _, replay = fixture(episodes=(6, 1, 4), buffer_size=8)

  # The ring holds the last three frames of episode 0, a singleton, and episode 2.
  def times(ids, frames):
    starts = replay.model_window_starts(ids, frames)
    row, slot = starts.unbind(-1)
    return list(
      zip(
        replay._storage["episode_id"][row, slot].tolist(),
        replay._storage["time_index"][row, slot].tolist(),
        strict=True,
      )
    )

  assert times([0, 1], 2) == [(0, 3), (0, 4)]
  assert times([0, 2], 3) == [(0, 3), (2, 0), (2, 1)]
  assert times([1], 2) == []
  assert times([0], 4) == []
  assert times([0, 1, 2], 9) == []
  # A new episode ends frame zero early and skips a physical frame at age two.
  for age, frame in enumerate(replay.retained_frames([2]).split(1)):
    frame["episode_id"][:] = 3
    if age == 0:
      frame["truncated"][:] = True
    if age == 2:
      frame["time_index"] += 5
    replay.insert(frame)
  assert times([3], 2) == []
  with pytest.raises(ValueError, match="two frames"):
    replay.model_window_starts([0, 2], 1)


def test_model_replay_and_full_replay_fit_the_same_windows_with_one_scan(monkeypatch):
  from tests.test_estimator_belief_training import assert_tree

  results = []
  for mode in ("full", "model"):
    estimator, replay = fixture(
      episodes=(6, 4, 3), buffer_size=12, replay_mode=mode, seal=False
    )
    scan = Mock(wraps=replay._recorded_successors)
    monkeypatch.setattr(replay, "_recorded_successors", scan)
    rng = torch.random.get_rng_state()
    replay.finish_collection_round()
    assert torch.equal(rng, torch.random.get_rng_state()) == (mode == "model")
    assert replay.training_episode_lengths() == {0: 5, 1: 4, 2: 3}
    counts = replay.model_fit_starts([0], [2], 3)
    assert [len(starts) for starts in counts] == [3, 1, 4, 2]
    # No held-in or held-out samples from the unlisted episode 1.
    windows = [
      replay.model_windows(starts, size)
      for starts, size in zip(counts, (3, 3, 2, 2), strict=True)
    ]
    fit = learner_for(estimator.dynamics)
    fresh = replay.retained_frames([0], fields=("state", "action"))
    torch.manual_seed(73)
    # Use identical eager arithmetic, including residual extraction and scoring.
    with torch.jit.optimized_execution(False):
      metrics = fit.fit(replay, [0], [2], fresh, 3)
    assert scan.call_count == 1
    results.append(
      (windows, metrics, estimator.dynamics.state_dict(), fit.state_dict())
    )
    if mode == "model":
      assert set(replay._storage.keys()) == MODEL_REPLAY_FIELDS
      assert not replay._episode_heads and not replay._training_heads
      with pytest.raises(ValueError, match="full replay"):
        replay.training_prefix(0, 2)
      with pytest.raises(ValueError, match="full replay"):
        replay.episode_prefix(0, 2)
  full, model = results
  for a, b in zip(full[0], model[0], strict=True):
    for key in a.keys():
      torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
  assert full[1] == model[1]
  assert_tree(full[2:], model[2:])


def test_model_round_index_tracks_insert_retry_and_explicit_truncation(monkeypatch):
  _, replay = fixture(episodes=(3,), buffer_size=3, replay_mode="model", seal=False)
  frame = replay.retained_frames(newest_count=1)
  frame["episode_id"][:] = 1
  frame["time_index"].zero_()
  frame["terminated"].zero_()
  scan = Mock(wraps=replay._recorded_successors)
  monkeypatch.setattr(replay, "_recorded_successors", scan)
  assert len(replay.model_window_starts([0], 2)) == 2
  token = object()
  replay.insert(frame, token=token)
  assert replay.training_episode_lengths() == {0: 2, 1: 1}
  assert len(replay.model_window_starts([0], 2)) == 1
  replay.insert(frame, token=token)
  assert replay.training_episode_lengths() == {0: 2, 1: 1}
  assert scan.call_count == 2 and replay.insert_count == 4
  replay.truncate_open_episodes()
  replay.finish_collection_round()
  replay.truncate_open_episodes()
  replay.finish_collection_round()
  assert scan.call_count == 3 and replay.insert_count == 4
  frame["time_index"][:] = 1
  replay.insert(frame)
  assert len(replay.model_window_starts([1], 2)) == 0
  assert scan.call_count == 4


def test_model_replay_rejects_invalid_heldout_head_before_fitting():
  from tests.test_estimator_belief_training import assert_tree

  estimator, replay = fixture(episodes=(4, 2), replay_mode="model", seal=False)
  replay._storage["state"][0, 4, 3:7].zero_()
  fit = learner_for(estimator.dynamics)
  fresh = replay.retained_frames([0], fields=("state", "action"))
  state = fit.state_dict()
  rng = torch.random.get_rng_state()
  # Private mutation bypassed insert; auditing is explicit before fitting.
  with pytest.raises(ValueError, match="rotation"):
    replay.validate_contents()
    fit.fit(replay, [0], [1], fresh, 2)
  assert_tree(fit.state_dict(), state)
  assert torch.equal(torch.random.get_rng_state(), rng)


@pytest.mark.parametrize("boundary", [None, "reset", "gap", "terminated", "truncated"])
def test_model_replay_checks_previous_action_only_for_recorded_successors(boundary):
  _, replay = fixture(episodes=(2,), buffer_size=2, replay_mode="model")
  frame = replay.retained_frames(newest_count=1)
  frame["time_index"][:] = 2
  frame["terminated"][:] = boundary == "terminated"
  frame["truncated"][:] = boundary == "truncated"
  replay.insert(frame)
  episode = frame["episode_id"] + int(boundary == "reset")
  time_index = frame["time_index"] + (2 if boundary == "gap" else 1)
  replay.validate_previous_action(episode, time_index, frame["action"])
  different = frame["action"] + 0.1
  if boundary is None:
    with pytest.raises(ValueError, match="previous action"):
      replay.validate_previous_action(episode, time_index, different)
  else:
    replay.validate_previous_action(episode, time_index, different)


def test_fit_samples_with_replacement_and_keeps_tail_batch(monkeypatch):
  estimator, replay = fixture(episodes=(6, 4, 1))
  fit = learner_for(
    estimator.dynamics,
    cov=CovarianceFitCfg(batch_size=2, pair_budget=11, validation_pairs=7),
  )
  fresh = replay.retained_frames([0])
  sizes, samples = [], []
  update = fit.update_covariance

  def record(batch):
    sizes.append(len(batch))
    samples.extend(batch["inputs"].tolist())
    assert not batch["residual"].requires_grad
    return update(batch)

  monkeypatch.setattr(fit, "update_covariance", record)
  train, validation = fit.fit(replay, [0], [1, 2], fresh, 3)
  inputs = fit.residual_data(
    replay.model_windows(replay.model_window_starts([0], 2), 2)
  )["inputs"].tolist()
  assert sizes == [2, 2, 2, 2, 2, 1]
  assert all(sample in inputs for sample in samples)
  assert len({tuple(sample) for sample in samples}) < len(samples)
  assert train["cov_pair_draws"] == 11 and train["cov_updates"] == 6
  assert train["cov_pairs"] == 5 and validation["cov_pairs"] == 3
  assert validation["cov_pair_draws"] == 7
  assert int(fit.dynamics.state_normalizer.count) == len(fresh)
  assert all(not p.requires_grad and p.grad is None for p in fit.dynamics.parameters())


def test_short_validation_fragment_still_scores_each_available_pair():
  estimator, replay = fixture(episodes=(4, 2))
  fit = learner_for(estimator.dynamics)
  frames = replay.retained_frames([0])
  _, validation = fit.fit(replay, [0], [1], frames, 4)
  assert validation["mean_windows"] == 0
  assert validation["cov_pairs"] == 1 and "Loss/process_nll" in validation
  with pytest.raises(ValueError, match="complete 3-frame"):
    fit.fit(replay, [1], [0], frames, 4)


def test_validation_samples_only_held_out_windows_and_pairs_with_replacement(
  monkeypatch,
):
  estimator, replay = fixture(episodes=(6, 4, 1))
  fit = learner_for(
    estimator.dynamics,
    cov=CovarianceFitCfg(batch_size=2, pair_budget=3, validation_pairs=7),
  )
  gathered = []
  gather = replay.model_windows

  def record(starts, frames):
    gathered.append((starts.clone(), frames))
    return gather(starts, frames)

  monkeypatch.setattr(replay, "model_windows", record)
  _, validation = fit.fit(replay, [0], [1, 2], replay.retained_frames([0]), 5)
  # One mean training batch, two covariance training batches, then validation.
  validation_draws = gathered[3:]
  assert [(len(starts), frames) for starts, frames in validation_draws] == [
    (5, 3),
    (2, 2),
    (2, 2),
    (2, 2),
    (1, 2),
  ]
  for starts, frames in validation_draws:
    allowed = replay.model_window_starts([1, 2], frames)
    assert (starts[:, None] == allowed[None]).all(-1).any(-1).all()
  # More draws than eligible starts guarantees repeated windows and pairs.
  windows = validation_draws[0][0]
  pairs = torch.cat([starts for starts, _ in validation_draws[1:]])
  assert len(windows.unique(dim=0)) < len(windows)
  assert len(pairs.unique(dim=0)) < len(pairs)
  assert validation["mean_windows"] == 5
  assert validation["cov_pair_draws"] == 7
