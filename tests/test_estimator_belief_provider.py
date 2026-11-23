"""Online belief lifecycle, reset causality, transactions and restoration."""

from dataclasses import fields, replace
from unittest.mock import patch

import pytest
import torch

from supermimic.belief import BeliefState
from supermimic.belief_provider import BeliefProvider, FixedBeliefPrior, belief_rollout
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


def assert_detached_state(state):
  values = [getattr(state.belief.mean, f.name) for f in fields(LieState)]
  for value in (*values, state.belief.covariance, state.context):
    assert not value.requires_grad
    assert value.grad_fn is None
    assert not torch.is_inference(value)


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


def test_provider_preserves_stricter_covariance_contract_after_filter_rows_merge(
  monkeypatch,
):
  provider, _, measurement, action, context = setup_provider()
  advance(provider, measurement, action, context, [True] * 3)
  committed = provider.state
  estimator = provider.estimator
  estimator.config = replace(estimator.config, covariance_rtol=1e-2)
  correct = estimator.correct

  def marginally_indefinite(*args, **kwargs):
    posterior, diagnostics = correct(*args, **kwargs)
    covariance = (
      torch.eye(70, dtype=measurement.dtype).expand_as(posterior.covariance).clone()
    )
    covariance[..., 0, 1] = covariance[..., 1, 0] = 1.0001
    loose = replace(posterior, covariance=covariance)
    estimator._validate_belief(loose)
    return loose, diagnostics

  monkeypatch.setattr(estimator, "correct", marginally_indefinite)
  with (
    belief_rollout(provider),
    pytest.raises(ValueError, match="positive semidefinite"),
  ):
    advance(provider, measurement, action, context, [False, True, False])
  assert provider.state is committed


def test_rollout_reuses_only_committed_results_with_current_tolerances():
  provider, _, measurement, action, context = setup_provider()
  live = [False] * 3

  def step(resets):
    return advance(provider, measurement, action, context, resets)

  estimator = provider.estimator
  with (
    belief_rollout(provider),
    patch.object(
      estimator, "_validate_belief", wraps=estimator._validate_belief
    ) as validate,
  ):
    step([True] * 3)
    validate.reset_mock()
    step(live)
    assert validate.call_count == 2  # new prediction and corrected posterior
    validate.reset_mock()
    step([False, True, False])
    assert validate.call_count == 4  # reset prior and posterior, plus two live checks
    validate.reset_mock()
    estimator.config = replace(estimator.config, covariance_rtol=2e-5)
    step(live)
    assert validate.call_count == 3
    state = provider.state
    provider.restore(
      state.belief,
      state.context,
      model_version=provider.model_version,
      episode_id=state.episode_id,
      time_index=state.time_index,
    )
    validate.reset_mock()
    step(live)
    assert validate.call_count == 3
  # The preceding scope's result is also external at the next rollout boundary.
  with (
    belief_rollout(provider),
    patch.object(
      estimator, "_validate_belief", wraps=estimator._validate_belief
    ) as validate,
  ):
    step(live)
    assert validate.call_count == 3


@pytest.mark.parametrize("when", ["prepared", "committed"])
@pytest.mark.parametrize(
  "corrupt,error", [("covariance", "semidefinite"), ("rotation", "orthonormal")]
)
def test_mutating_posterior_forces_full_revalidation(when, corrupt, error):
  provider, _, measurement, action, context = setup_provider()
  with belief_rollout(provider):
    candidate = provider.prepare(
      measurement=measurement,
      previous_action=action,
      reset_mask=torch.ones(3, dtype=torch.bool),
      context=context,
    )
    if when == "committed":
      provider.commit(candidate)
    belief = candidate.state.belief
    if corrupt == "covariance":
      belief.covariance.detach()[0, 0, 0] = -1
    else:
      belief.mean.rotation.detach()[0].zero_()
    if when == "prepared":
      provider.commit(candidate)
    generation = provider.generation
    with pytest.raises(ValueError, match=error):
      advance(provider, measurement, action, context, [False] * 3)
    assert provider.generation == generation


@pytest.mark.parametrize("name", ["measurement", "previous_action", "context"])
def test_provider_reports_each_nonfinite_input_before_publishing(name):
  provider, _, measurement, action, context = setup_provider()
  inputs = dict(
    measurement=measurement,
    previous_action=action,
    context=context,
    reset_mask=torch.ones(3, dtype=torch.bool),
  )
  inputs[name][0, 0] = torch.nan
  with belief_rollout(provider), pytest.raises(ValueError, match=name):
    provider.prepare(**inputs)
  assert provider.state is None and provider.generation == 0


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


@pytest.mark.parametrize("mode", [torch.no_grad, torch.inference_mode])
def test_differentiable_provider_respects_disabled_grad_modes_and_can_resume(mode):
  provider, _, measurement, action, context = setup_provider(differentiable=True)
  measurement.requires_grad_()
  action.requires_grad_()
  context.requires_grad_()
  advance(provider, measurement, action, context, [True] * 3)
  advance(provider, measurement, action, context, [False] * 3)
  assert provider.state.belief.covariance.grad_fn is not None

  reference = BeliefProvider(provider.estimator, provider.prior_factory)
  reference.restore(
    provider.state.belief,
    context,
    model_version=provider.model_version,
    episode_id=provider.state.episode_id,
    time_index=provider.state.time_index,
  )
  with mode():
    for reset in ([False] * 3, [False, True, False], [False] * 3):
      # Inference-mode inputs still need ordinary clones for the local Jacobian.
      inputs = measurement.clone(), action.clone(), context.clone()
      advance(provider, *inputs, reset)
      advance(reference, *inputs, reset)
      assert_detached_state(provider.state)
      assert_same_belief(provider.state.belief, reference.state.belief)

  assert torch.is_grad_enabled()
  assert provider.differentiable
  assert all(p.requires_grad for p in provider.estimator.dynamics.parameters())
  resumed_action = action.detach().clone().requires_grad_()
  advance(provider, measurement.detach(), resumed_action, context.detach(), [False] * 3)
  objective = (
    provider.state.belief.mean.position.square().sum()
    + provider.state.belief.covariance.square().sum()
  )
  (gradient,) = torch.autograd.grad(objective, resumed_action)
  assert torch.isfinite(gradient).all() and gradient.norm() > 0


@pytest.mark.parametrize(
  "mode, tracks_grad",
  [(torch.enable_grad, True), (torch.no_grad, False), (torch.inference_mode, False)],
)
def test_differentiable_restore_respects_caller_grad_mode(mode, tracks_grad):
  source, _, measurement, action, context = setup_provider(batch=1, differentiable=True)
  action.requires_grad_()
  context.requires_grad_()
  advance(source, measurement, action, context, [True])
  advance(source, measurement, action, context, [False])
  saved = source.state
  restored = BeliefProvider(source.estimator, differentiable=True)
  with mode():
    candidate = restored.prepare_restore(
      saved.belief,
      saved.context,
      model_version=saved.model_version,
      episode_id=saved.episode_id,
      time_index=saved.time_index,
    )
  assert restored.state is None
  restored.commit(candidate)
  assert_same_belief(restored.state.belief, saved.belief)
  if tracks_grad:
    objective = (
      restored.state.belief.mean.position.square().sum()
      + restored.state.belief.covariance.square().sum()
      + restored.state.context.square().sum()
    )
    for gradient in torch.autograd.grad(objective, (action, context)):
      assert torch.isfinite(gradient).all() and gradient.norm() > 0
  else:
    assert_detached_state(restored.state)


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


def test_incremental_snapshot_keeps_source_revision_and_checks_model_at_commit():
  reference, _, measurement, action, context = setup_provider(batch=1)
  provider = BeliefProvider(
    reference.estimator, reference.prior_factory, incremental=True
  )
  advance(provider, measurement, action, context, [True])
  model = provider.estimator.dynamics
  source_revision = model.model_revision
  model.load_state_dict(model.state_dict())
  snapshot = provider.state_dict()
  assert snapshot["posterior_revision"] == source_revision < snapshot["model_revision"]

  restored = BeliefProvider(reference.estimator, incremental=True)
  candidate = restored.prepare_load_state_dict(snapshot)
  # Reload identical weights and revision after prepare.
  model.load_checkpoint(model.checkpoint())
  with pytest.raises(ValueError, match="Model changed"):
    restored.commit(candidate)
  assert restored.state is None
  restored.load_state_dict(snapshot)
  assert restored.state_dict()["posterior_revision"] == source_revision
  assert_same_belief(restored.state.belief, provider.state.belief)

  for changes in (
    {"posterior_revision": -1},
    {"posterior_revision": model.model_revision + 1},
    {"model_revision": model.model_revision + 1},
  ):
    with pytest.raises(ValueError, match="revision"):
      restored.load_state_dict(snapshot | changes)
  with pytest.raises(ValueError, match="Historical posterior"):
    BeliefProvider(reference.estimator).load_state_dict(snapshot)
  with pytest.raises(ValueError, match="Historical posterior"):
    BeliefProvider(
      reference.estimator, incremental=True, differentiable=True
    ).load_state_dict(snapshot)
  advance(restored, measurement, action, context, [False])
  assert restored.state.model_version == model.model_revision
