"""Online belief lifecycle, reset causality, transactions and restoration."""

from dataclasses import replace

import pytest
import torch

from supermimic.belief import BeliefState
from supermimic.belief_provider import BeliefProvider, FixedBeliefPrior
from supermimic.estimator import LieBeliefFilter
from supermimic.estimator_inputs import GaussianMeasurementConfig
from supermimic.utils.lie_state import LieState, local_error
from tests.test_estimator_stochastic_dynamics import make_dynamics, model_inputs


def setup_provider(batch=3, *, differentiable=False):
  torch.manual_seed(12)
  model, representation = make_dynamics()
  rep, action, aux, context = model_inputs(representation, batch=batch)
  prior = BeliefState(
    LieState.from_rep(aux, rep),
    torch.diag(torch.linspace(0.002, 0.01, 70, dtype=rep.dtype))
    .expand(batch, 70, 70)
    .clone(),
  )
  estimator = LieBeliefFilter(
    model, GaussianMeasurementConfig(), torch.zeros(29, dtype=rep.dtype)
  )
  provider = BeliefProvider(
    estimator, FixedBeliefPrior(prior), differentiable=differentiable
  )
  measurement = estimator.observation(prior.mean) + torch.linspace(
    -0.01, 0.01, 64, dtype=rep.dtype
  )
  return provider, prior, measurement, action, context


def assert_same_belief(actual, expected):
  torch.testing.assert_close(
    local_error(actual.mean, expected.mean),
    torch.zeros_like(actual.mean.to_full()[..., :70]),
    atol=1e-12,
    rtol=0,
  )
  torch.testing.assert_close(
    actual.covariance, expected.covariance, atol=1e-12, rtol=1e-10
  )


def advance(provider, measurement, action, context, reset):
  candidate = provider.prepare(
    measurement=measurement,
    previous_action=action,
    reset_mask=torch.tensor(reset, dtype=torch.bool),
    context=context,
  )
  provider.commit(candidate)
  return candidate


def test_initial_frame_corrects_once_and_partial_reset_preserves_continuing_slots():
  provider, prior, measurement, action, context = setup_provider()
  # Initial action is intentionally large: a reset must never propagate it.
  advance(provider, measurement, action * 100, context, [True, True, True])
  first = provider.estimator.correct(prior, measurement)
  assert_same_belief(provider.state.belief, first)
  assert torch.equal(provider.state.time, torch.zeros(3, dtype=torch.long))
  next_measurement = measurement + 0.002
  candidate = advance(
    provider, next_measurement, action, context, [False, False, False]
  )
  second = provider.estimator.correct(
    provider.estimator.predict(first, action, context), next_measurement
  )
  assert_same_belief(candidate.state.belief, second)
  next_context = context.clone()
  next_context[1] += 0.03
  advance(provider, measurement, action, next_context, [False, True, False])
  expected_reset = provider.estimator.correct(prior.index([1]), measurement[[1]])
  expected_continue = provider.estimator.correct(
    provider.estimator.predict(second.index([0, 2]), action[[0, 2]], context[[0, 2]]),
    measurement[[0, 2]],
  )
  assert_same_belief(provider.state.belief.index([1]), expected_reset)
  assert_same_belief(provider.state.belief.index([0, 2]), expected_continue)
  assert torch.equal(provider.state.time, torch.tensor([2, 0, 2]))


def test_prepare_retry_does_not_mutate_state_or_consume_randomness():
  provider, _, measurement, action, context = setup_provider()
  advance(provider, measurement, action, context, [True] * 3)
  current = provider.state
  bad = measurement.clone()
  bad[0, 0] = float("nan")
  with pytest.raises(ValueError, match="finite"):
    provider.prepare(
      measurement=bad,
      previous_action=action,
      reset_mask=torch.zeros(3, dtype=torch.bool),
    )
  assert provider.state is current
  rng = torch.random.get_rng_state()
  kwargs = dict(
    measurement=measurement,
    previous_action=action,
    reset_mask=torch.zeros(3, dtype=torch.bool),
  )
  candidate = provider.prepare(**kwargs)
  retry = provider.prepare(**kwargs)
  assert provider.state is current
  assert torch.equal(torch.random.get_rng_state(), rng)
  assert_same_belief(candidate.state.belief, retry.state.belief)
  provider.commit(retry)
  assert torch.equal(provider.state.time, current.time + 1)
  with pytest.raises(ValueError, match="Stale or foreign"):
    provider.commit(candidate)
  other, _, _, _, _ = setup_provider()
  with pytest.raises(ValueError, match="Stale or foreign"):
    other.commit(retry)
  assert other.state is None


def test_restore_continues_from_full_corrected_posterior_and_preserves_time():
  provider, _, measurement, action, context = setup_provider()
  advance(provider, measurement, action, context, [True] * 3)
  advance(provider, measurement, action, context, [False] * 3)
  restored = BeliefProvider(provider.estimator)
  restored.restore(
    provider.state.belief,
    context,
    model_version=provider.model_version,
    time_index=provider.state.time,
    episode_id=provider.state.episode_id,
  )
  assert_same_belief(restored.state.belief, provider.state.belief)
  for current in (provider, restored):
    advance(current, measurement + 0.001, action, context, [False] * 3)
  assert_same_belief(restored.state.belief, provider.state.belief)
  assert torch.equal(restored.state.time, torch.full((3,), 2))
  state_before = restored.state
  with pytest.raises(TypeError, match="complete BeliefState"):
    restored.restore(
      provider.state.belief.to_actor_input(),
      context,
      model_version=provider.model_version,
    )
  with pytest.raises(ValueError, match="different model"):
    restored.restore(provider.state.belief, context, model_version="stale")
  with pytest.raises(ValueError, match="nonnegative"):
    restored.restore(
      provider.state.belief,
      context,
      model_version=provider.model_version,
      time_index=torch.tensor([-1, 0, 0]),
      episode_id=provider.state.episode_id,
    )
  assert restored.state is state_before


def test_inference_mode_collection_exits_locally_and_commits_detached_ordinary_tensors():
  provider, _, measurement, action, context = setup_provider()
  with torch.inference_mode():
    for reset in ([True] * 3, [False] * 3):
      advance(provider, measurement.clone(), action.clone(), context.clone(), reset)
  state = provider.state
  for value in (state.belief.mean.to_full(), state.belief.covariance, state.context):
    assert not value.requires_grad
    assert value.grad_fn is None
    assert not torch.is_inference(value)
  assert torch.equal(state.time, torch.ones(3, dtype=torch.long))


def test_differentiable_provider_preserves_action_gradient_across_commits():
  provider, _, measurement, action, context = setup_provider(
    batch=1, differentiable=True
  )
  advance(provider, measurement, action, context, [True])
  action = action.clone().requires_grad_()
  advance(provider, measurement + 0.001, action, context, [False])
  advance(provider, measurement + 0.002, action * 0.9, context, [False])
  objective = (
    provider.state.belief.mean.position.square().sum()
    + provider.state.belief.covariance.square().sum()
  )
  gradient = torch.autograd.grad(objective, action)[0]
  assert torch.isfinite(gradient).all()
  assert gradient.abs().sum() > 0


def test_fused_optimizer_invalidates_posterior_and_checkpoint_roundtrips():
  from supermimic.learners.learner_utils import build_optimizers

  provider, _, measurement, action, context = setup_provider(batch=1)
  advance(provider, measurement, action, context, [True])
  model = provider.estimator.dynamics
  state = provider.state
  # This architecture has only input and output Linear layers. The repo's
  # optimizer puts both the mean and process head parameters in fused Adam.
  (optimizer,) = build_optimizers([model], lr=0.01)
  bias = model.noise_head[-1].bias
  before = bias.detach().clone()
  bias.sum().backward()
  optimizer.step()
  assert not torch.equal(before, bias)
  assert model.model_version != state.model_version
  with pytest.raises(ValueError, match="model version"):
    provider.lookup(state.episode_id, state.time_index, context)

  checkpoint = model.checkpoint()
  restored, _ = make_dynamics()
  restored.load_checkpoint(checkpoint)
  assert restored.model_version == checkpoint["model_version"] == model.model_version
  torch.testing.assert_close(restored.noise_head[-1].bias, bias, rtol=0, atol=0)


def test_live_context_and_model_versions_are_checked_before_prepare_or_commit():
  provider, _, measurement, action, context = setup_provider()
  advance(provider, measurement, action, context, [True] * 3)
  state = provider.state
  with pytest.raises(ValueError, match="Context changed"):
    advance(provider, measurement, action, context + 1, [False] * 3)
  candidate = provider.prepare(
    measurement=measurement,
    previous_action=action,
    reset_mask=torch.zeros(3, dtype=torch.bool),
  )
  provider.estimator.dynamics.state_normalizer.train()
  provider.estimator.dynamics.state_normalizer.update(state.belief.mean.to_rep()[1])
  with pytest.raises(ValueError, match="Model changed"):
    provider.commit(candidate)
  with pytest.raises(ValueError, match="version changed"):
    advance(provider, measurement, action, context, [False] * 3)
  assert provider.state is state


def test_provider_checkpoint_roundtrip_and_configuration_rejection():
  provider, _, measurement, action, context = setup_provider()
  advance(provider, measurement, action, context, [True] * 3)
  advance(provider, measurement, action, context, [False] * 3)
  snapshot = provider.state_dict()
  restored = BeliefProvider(provider.estimator)
  restored.load_state_dict(snapshot)
  assert_same_belief(restored.state.belief, provider.state.belief)
  for current in (provider, restored):
    advance(current, measurement + 0.002, action, context, [False] * 3)
  assert_same_belief(restored.state.belief, provider.state.belief)
  wrong = dict(snapshot, version="unknown")
  before = restored.state
  with pytest.raises(ValueError, match="version"):
    restored.load_state_dict(wrong)
  assert restored.state is before
  for key in ("state_layout_version", "tangent_version"):
    with pytest.raises(ValueError, match="state/tangent layout"):
      restored.load_state_dict(dict(snapshot, **{key: "unsupported"}))
    assert restored.state is before
  incompatible = LieBeliefFilter(
    provider.estimator.dynamics,
    replace(provider.estimator.measurement_config, gravity_std=0.2),
    provider.estimator.default_joint_pos,
  )
  with pytest.raises(ValueError, match="configuration mismatch"):
    BeliefProvider(incompatible).load_state_dict(snapshot)
