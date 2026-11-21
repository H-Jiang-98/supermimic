"""Independent filter references and mixed derivatives, without MuJoCo."""

from dataclasses import replace

import pytest
import torch
from torch import nn

from supermimic.belief import BeliefState
from supermimic.estimator import LieBeliefFilter
from supermimic.estimator_inputs import GaussianMeasurementConfig
from supermimic.utils.lie_state import LieState, boxplus, local_error, so3_exp


class _AnalyticDynamics(nn.Module):
  """A nonlinear transition with learnable mean and noise for independent tests."""

  def __init__(self):
    super().__init__()
    self.mean_weight = nn.Parameter(torch.tensor(0.8, dtype=torch.float64))
    self.noise_weight = nn.Parameter(torch.tensor(-3.0, dtype=torch.float64))

  def mean_lie(self, state, action, context):
    dt, weight = 0.02, self.mean_weight
    return LieState(
      state.rotation @ so3_exp(dt * (state.omega + weight * action[..., :3])),
      state.position + dt * state.velocity,
      state.velocity + dt * weight * (action[..., :3] + state.velocity.tanh()),
      state.omega + dt * weight * action[..., :3],
      state.joint_pos + dt * state.joint_vel,
      state.joint_vel + dt * weight * (state.joint_pos + action).tanh(),
    )

  def distribution_lie(self, state, action, context):
    mean = self.mean_lie(state, action, context)
    diagonal = nn.functional.softplus(self.noise_weight) + 1e-6
    factor = diagonal * torch.eye(70, dtype=action.dtype, device=action.device)
    return mean, factor.expand(*action.shape[:-1], 70, 70)


def _fixture(batch=2, *, dtype=torch.float64, measurement=None, config=None):
  generator = torch.Generator().manual_seed(19)

  def rand(*shape):
    return torch.randn(*shape, generator=generator, dtype=dtype)

  mean = LieState(
    so3_exp(rand(batch, 3) * 0.4),
    rand(batch, 3),
    rand(batch, 3) * 0.4,
    rand(batch, 3) * 0.2,
    rand(batch, 29) * 0.2,
    rand(batch, 29) * 0.1,
  )
  lower = rand(batch, 70, 70) * 0.006
  covariance = lower @ lower.transpose(-1, -2) + 0.01 * torch.eye(70, dtype=dtype)
  model = _AnalyticDynamics().to(dtype=dtype)
  estimator = LieBeliefFilter(
    model,
    measurement or GaussianMeasurementConfig(),
    torch.linspace(-0.1, 0.1, 29, dtype=dtype),
    config,
  )
  belief = estimator.initialize(mean, covariance)
  action = rand(batch, 29) * 0.1
  context = rand(batch, 2)
  return estimator, belief, action, context


def test_observation_uses_exact_measurement_layout_and_analytic_jacobian():
  estimator, belief, _, _ = _fixture()
  observation = estimator.observation(belief.mean)
  torch.testing.assert_close(observation[..., :3], -belief.mean.rotation[..., 2, :])
  torch.testing.assert_close(observation[..., 3:6], belief.mean.omega)
  torch.testing.assert_close(
    observation[..., 6:35], belief.mean.joint_pos - estimator.default_joint_pos
  )
  torch.testing.assert_close(observation[..., 35:], belief.mean.joint_vel)
  finite_columns = []
  for index in range(70):
    delta = observation.new_zeros(70)
    delta[index] = 1e-6
    plus = estimator.observation(boxplus(belief.mean, delta))
    minus = estimator.observation(boxplus(belief.mean, -delta))
    finite_columns.append((plus - minus) / 2e-6)
  torch.testing.assert_close(
    estimator.observation_jacobian(belief.mean),
    torch.stack(finite_columns, -1),
    atol=2e-10,
    rtol=1e-7,
  )


def test_transition_jacobian_and_covariance_match_finite_differences():
  estimator, belief, action, context = _fixture()
  predicted, diagnostics = estimator.predict(
    belief, action, context, return_diagnostics=True
  )
  finite_columns = []
  for index in range(70):
    delta = action.new_zeros(70)
    delta[index] = 2e-6
    residuals = []
    for sign in (1, -1):
      perturbed = boxplus(belief.mean, delta * sign)
      moved = estimator.dynamics.mean_lie(perturbed, action, context)
      residuals.append(local_error(predicted.mean, moved))
    finite_columns.append((residuals[0] - residuals[1]) / 4e-6)
  reference = torch.stack(finite_columns, -1)
  torch.testing.assert_close(
    diagnostics.transition_jacobian, reference, atol=5e-10, rtol=5e-7
  )
  torch.testing.assert_close(
    predicted.covariance,
    reference @ belief.covariance @ reference.transpose(-1, -2)
    + diagnostics.process_covariance,
    atol=1e-11,
    rtol=1e-7,
  )


@pytest.mark.parametrize("entrypoint", ["transition_jacobian", "predict"])
def test_filter_rejects_inference_mode_instead_of_dropping_prior_covariance(entrypoint):
  estimator, belief, action, context = _fixture(batch=1)
  state = belief.mean if entrypoint == "transition_jacobian" else belief
  with torch.inference_mode(), pytest.raises(RuntimeError, match="local autograd"):
    getattr(estimator, entrypoint)(state, action, context)


def test_no_grad_prediction_preserves_transition_and_prior_covariance():
  estimator, belief, action, context = _fixture(batch=1)
  expected, reference = estimator.predict(
    belief, action, context, return_diagnostics=True
  )
  with torch.no_grad():
    actual, diagnostics = estimator.predict(
      belief, action, context, return_diagnostics=True
    )
  assert not actual.covariance.requires_grad
  assert not diagnostics.transition_jacobian.requires_grad
  assert (actual.covariance - diagnostics.process_covariance).abs().max() > 0.001
  torch.testing.assert_close(actual.covariance, expected.covariance)
  torch.testing.assert_close(
    diagnostics.transition_jacobian, reference.transition_jacobian
  )
  for output, target in zip(actual.mean.to_rep(), expected.mean.to_rep(), strict=True):
    torch.testing.assert_close(output, target)


def test_euclidean_subproblem_matches_standard_kalman_update():
  estimator, belief, _, _ = _fixture()
  diagonal = torch.linspace(0.01, 0.07, 70, dtype=torch.float64)
  belief = BeliefState(belief.mean, torch.diag(diagonal).expand(2, 70, 70))
  observation = estimator.observation(belief.mean)
  innovation = (
    torch.linspace(-0.04, 0.03, 64, dtype=torch.float64).expand(2, 64).clone()
  )
  innovation[:, :3] = 0
  posterior, diagnostics = estimator.correct(
    belief, observation + innovation, return_diagnostics=True
  )
  v = estimator.measurement_config.covariance(29, dtype=torch.float64).diagonal()
  gain = diagonal[9:] / (diagonal[9:] + v[3:])
  expected_error = gain * innovation[:, 3:]
  expected_variance = diagonal[9:] * v[3:] / (diagonal[9:] + v[3:])
  torch.testing.assert_close(
    local_error(belief.mean, posterior.mean)[:, 9:], expected_error
  )
  torch.testing.assert_close(
    posterior.covariance[:, 9:, 9:].diagonal(dim1=-2, dim2=-1),
    expected_variance.expand(2, 61),
  )
  assert torch.equal(diagnostics.jitter, torch.zeros(2, dtype=torch.float64))
  torch.testing.assert_close(posterior.mean.position, belief.mean.position)


@pytest.mark.parametrize("gravity_std, expected_rank", [(0.0, 63), (0.03, 64)])
def test_joint_gain_survives_mixed_sensor_units_in_float32(gravity_std, expected_rank):
  noise = GaussianMeasurementConfig(gravity_std=gravity_std)
  estimator, belief, _, _ = _fixture(1, dtype=torch.float32, measurement=noise)
  variances = torch.full((70,), 0.01)
  variances[12:41] = 0.01**2
  variances[41:] = 5.0**2
  belief = replace(belief, covariance=torch.diag(variances).unsqueeze(0))
  y = estimator.observation(belief.mean)
  y[:, 6:35] += 0.01
  _, d = estimator.correct(belief, y, return_diagnostics=True)
  expected_gain = variances[12:41] / (variances[12:41] + noise.joint_position_std**2)
  assert d.effective_rank.item() == expected_rank
  torch.testing.assert_close(d.kalman_gain[0, 12:41, 6:35], torch.diag(expected_gain))
  torch.testing.assert_close(d.correction[0, 12:41], 0.01 * expected_gain)


def test_rank_deficient_update_preserves_covariance_and_orientation_derivatives():
  noise = GaussianMeasurementConfig(gravity_std=0)
  estimator, belief, _, _ = _fixture(1, measurement=noise)
  measurement = estimator.observation(belief.mean) + 0.03
  direction = torch.linspace(-0.3, 0.5, 70, dtype=torch.float64)

  def objective(t):
    mean = boxplus(belief.mean, direction * t)
    covariance = belief.covariance + torch.diag((direction * t).square())
    p = estimator.correct(BeliefState(mean, covariance), measurement)
    return (local_error(belief.mean, p.mean) * direction).sum() + p.covariance.sum()

  t = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
  assert torch.autograd.gradcheck(objective, (t,))
  assert torch.autograd.gradgradcheck(objective, (t,))


def test_noiseless_joints_keep_variance_in_a_truncated_mode():
  noise = GaussianMeasurementConfig(joint_position_std=0)
  estimator, prior, _, _ = _fixture(1, dtype=torch.float32, measurement=noise)
  covariance = torch.eye(70).unsqueeze(0)
  covariance[:, 12, 13] = covariance[:, 13, 12] = 1 - 1e-6
  posterior, d = estimator.correct(
    replace(prior, covariance=covariance),
    estimator.observation(prior.mean),
    return_diagnostics=True,
  )
  assert d.effective_rank.item() == 63
  # The observed sum has zero variance. Its unresolved difference retains
  # the prior variance in that direction, including its negative cross term.
  half_difference = (1 - covariance[0, 12, 13]) / 2
  expected = half_difference * torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
  torch.testing.assert_close(
    posterior.covariance[0, 12:14, 12:14], expected, atol=1e-10, rtol=1e-3
  )
  assert torch.count_nonzero(posterior.covariance[0, 14:41]) == 0


def test_full_update_matches_independent_gain_joseph_and_reset_reference():
  estimator, belief, _, _ = _fixture()
  measurement = estimator.observation(belief.mean) + torch.linspace(-0.1, 0.08, 64)
  posterior, d = estimator.correct(belief, measurement, return_diagnostics=True)
  h, p = d.measurement_jacobian, belief.covariance
  v = estimator.measurement_config.covariance(29, dtype=torch.float64)
  s = h @ p @ h.transpose(-1, -2) + v
  gain = torch.linalg.solve(s, h @ p).transpose(-1, -2)
  correction = (
    gain @ (measurement - estimator.observation(belief.mean)).unsqueeze(-1)
  ).squeeze(-1)
  torch.testing.assert_close(d.kalman_gain, gain)
  torch.testing.assert_close(
    local_error(posterior.mean, boxplus(belief.mean, correction)),
    torch.zeros(2, 70, dtype=torch.float64),
    atol=2e-14,
    rtol=0,
  )
  # Differentiate the defining reset map numerically; do not call reset_jacobian.
  reset_columns = []
  for index in range(70):
    epsilon = torch.zeros(70, dtype=torch.float64)
    epsilon[index] = 1e-6
    plus = local_error(posterior.mean, boxplus(belief.mean, correction + epsilon))
    minus = local_error(posterior.mean, boxplus(belief.mean, correction - epsilon))
    reset_columns.append((plus - minus) / 2e-6)
  reset = torch.stack(reset_columns, -1)
  residual = torch.eye(70) - gain @ h
  joseph = residual @ p @ residual.transpose(-1, -2) + gain @ v @ gain.transpose(-1, -2)
  torch.testing.assert_close(
    posterior.covariance,
    reset @ joseph @ reset.transpose(-1, -2),
    atol=2e-11,
    rtol=1e-7,
  )


def test_zero_innovation_preserves_mean_and_shrinks_observed_variance():
  estimator, belief, _, _ = _fixture()
  posterior = estimator.correct(belief, estimator.observation(belief.mean))
  torch.testing.assert_close(
    local_error(belief.mean, posterior.mean),
    torch.zeros(2, 70, dtype=torch.float64),
    atol=2e-15,
    rtol=0,
  )
  assert (
    posterior.covariance.diagonal(dim1=-2, dim2=-1)
    <= belief.covariance.diagonal(dim1=-2, dim2=-1) + 1e-12
  ).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_zero_measurement_noise_is_bounded_and_jitter_consistent(dtype):
  noise = GaussianMeasurementConfig(0, 0, 0, 0)
  estimator, belief, _, _ = _fixture(dtype=dtype, measurement=noise)
  delta = torch.zeros(2, 70, dtype=dtype)
  delta[:, 0] = 0.2
  observation = estimator.observation(boxplus(belief.mean, delta))
  posterior, d = estimator.correct(belief, observation, return_diagnostics=True)
  assert (d.effective_rank == 63).all() and (d.jitter == 0).all()
  reference = (
    belief.covariance @ d.measurement_jacobian.mT
  ).double() @ torch.linalg.pinv(
    d.innovation_covariance.double(), hermitian=True, rtol=64 * torch.finfo(dtype).eps
  )
  torch.testing.assert_close(d.kalman_gain, reference.to(dtype), atol=1e-5, rtol=1e-5)
  assert torch.isfinite(posterior.covariance).all()
  assert torch.linalg.eigvalsh(posterior.covariance).min() >= -1e-8
  gain, h = d.kalman_gain, d.measurement_jacobian
  residual = torch.eye(70, dtype=dtype) - gain @ h
  reference = residual @ belief.covariance @ residual.transpose(-1, -2) + d.jitter[
    :, None, None
  ] * (gain @ gain.transpose(-1, -2))
  torch.testing.assert_close(
    posterior.covariance,
    d.reset_jacobian @ reference @ d.reset_jacobian.mT,
    atol=1e-9 if dtype == torch.float32 else 1e-15,
    rtol=2e-5,
  )


def test_invalid_covariances_fail_explicitly():
  estimator, belief, _, _ = _fixture()
  bad = belief.covariance.clone()
  bad[:, 0, 0] = -1
  with pytest.raises(ValueError, match="semidefinite"):
    estimator.correct(BeliefState(belief.mean, bad), estimator.observation(belief.mean))
  bad = belief.covariance.clone()
  bad[:, 0, 1] += 0.1
  with pytest.raises(ValueError, match="symmetric"):
    estimator.initialize(belief.mean, bad)
  bad[:, 0, 1] = float("nan")
  with pytest.raises(ValueError, match="non-finite"):
    estimator.initialize(belief.mean, bad)


def test_scalar_belief_prediction_with_production_dynamics_and_normalizers():
  from tests.test_estimator_stochastic_dynamics import make_dynamics, model_inputs

  model, rep = make_dynamics()
  # The simulator exposes joint action scale and offset with an environment axis.
  model.action_scale = model.action_scale.reshape(1, 29)
  model.action_offset = model.action_offset.reshape(1, 29)
  represented, action, aux, ctx = model_inputs(rep)
  estimator = LieBeliefFilter(
    model, GaussianMeasurementConfig(), rep._default_joint_pos
  )
  mean = LieState.from_rep(aux, represented)
  prior = BeliefState(mean, torch.eye(70, dtype=action.dtype).expand(2, 70, 70))
  batched = estimator.predict(prior, action, ctx)
  scalar = estimator.predict(prior.index(1), action[1], ctx[1])
  torch.testing.assert_close(scalar.mean.to_full(), batched.mean.to_full()[1])
  torch.testing.assert_close(scalar.covariance, batched.covariance[1])


def test_promote_valid_low_rank_posterior_preserves_psd_and_gradients():
  estimator, prior, _, _ = _fixture(
    4, dtype=torch.float32, measurement=GaussianMeasurementConfig(gravity_std=0)
  )
  posterior = estimator.correct(prior, estimator.observation(prior.mean))
  covariance = posterior.covariance.requires_grad_()
  posterior.validate()
  assert torch.linalg.eigvalsh(covariance.double()).min() < -1e-12
  promoted = posterior.to(torch.float64)
  promoted.validate()
  torch.testing.assert_close(
    promoted.covariance, covariance.double(), atol=3e-9, rtol=1e-6
  )
  torch.testing.assert_close(
    torch.autograd.grad(promoted.covariance.sum(), covariance)[0],
    torch.ones_like(covariance),
  )
  bad = covariance.detach().clone()
  bad[:, 0, 0] = -1
  with pytest.raises(ValueError, match="positive semidefinite"):
    replace(posterior, covariance=bad).to(torch.float64)


def test_scalar_and_batched_filter_have_same_results_and_no_mutation():
  estimator, belief, action, context = _fixture()
  saved = belief.clone()
  batched = estimator.predict(belief, action, context)
  single = estimator.predict(belief.index(0), action[0], context[0])
  torch.testing.assert_close(single.mean.to_full(), batched.mean.to_full()[0])
  torch.testing.assert_close(single.covariance, batched.covariance[0])
  torch.testing.assert_close(belief.mean.to_full(), saved.mean.to_full())
  torch.testing.assert_close(belief.covariance, saved.covariance)


def test_belief_versions_are_checked_by_filter():
  estimator, belief, action, context = _fixture()
  for field in ("state_layout_version", "tangent_version"):
    bad = replace(belief, **{field: "incompatible"})
    with pytest.raises(ValueError, match="version"):
      estimator.predict(bad, action, context)
    with pytest.raises(ValueError, match="version"):
      estimator.correct(bad, estimator.observation(belief.mean))


def test_real_transition_jacobian_mixed_parameter_derivative_matches_difference():
  """Regress silent nested-jacrev/TorchScript errors, not only finite gradients."""
  from tests.test_estimator_stochastic_dynamics import make_dynamics, model_inputs

  torch.manual_seed(81)
  model, representation = make_dynamics()
  rep, action, aux, context = model_inputs(representation, batch=1)
  estimator = LieBeliefFilter(
    model, GaussianMeasurementConfig(), torch.zeros(29, dtype=torch.float64)
  )
  mean = LieState.from_rep(aux, rep)
  parameter = model.mlp[-1].bias
  direction = torch.randn_like(parameter)
  direction = direction / direction.norm()
  weight = torch.randn(1, 70, 70, dtype=torch.float64)

  def objective():
    return (estimator.transition_jacobian(mean, action, context) * weight).sum()

  # Exercise both cold and profiled TorchScript paths before the mixed check.
  objective()
  objective()
  gradient = torch.autograd.grad(objective(), parameter)[0]
  original = parameter.detach().clone()
  values = []
  epsilon = 1e-5
  for sign in (1, -1):
    with torch.no_grad():
      parameter.copy_(original + sign * epsilon * direction)
    values.append(objective().detach())
  with torch.no_grad():
    parameter.copy_(original)
  torch.testing.assert_close(
    (gradient * direction).sum(),
    (values[0] - values[1]) / (2 * epsilon),
    atol=2e-8,
    rtol=2e-6,
  )
