"""Matrix integration, vectorized local Jacobians and frozen-model action gradients."""

from dataclasses import fields, replace

import pytest
import torch

from supermimic.belief import BeliefState
from supermimic.estimator import FilterConfig, LieBeliefFilter
from supermimic.estimator_inputs import GaussianMeasurementConfig
from supermimic.utils.lie_state import LieState, boxplus, local_error, so3_exp
from tests.test_estimator_stochastic_dynamics import make_dynamics, model_inputs


def fixture(batch=2, dtype=torch.float64, device="cpu", config=None):
  torch.manual_seed(81)
  model, rep = make_dynamics(dtype=dtype)
  model.to(device)
  inputs = tuple(x.to(device) for x in model_inputs(rep, batch=batch, dtype=dtype))
  state = LieState.from_rep(inputs[2], inputs[0])
  estimator = LieBeliefFilter(
    model,
    GaussianMeasurementConfig(),
    torch.zeros(29, dtype=dtype, device=device),
    config,
  )
  return model, rep, estimator, state, inputs[1], inputs[3]


def assert_state(actual, expected, **kwargs):
  for field in fields(actual):
    torch.testing.assert_close(
      getattr(actual, field.name), getattr(expected, field.name), **kwargs
    )


def directional_check(objective, parameter, gradient, epsilon=1e-5):
  direction = gradient.detach() / gradient.detach().norm()
  original = parameter.detach().clone()
  values = []
  try:
    for sign in (1, -1):
      with torch.no_grad():
        parameter.copy_(original + sign * epsilon * direction)
      values.append(objective().detach())
  finally:
    with torch.no_grad():
      parameter.copy_(original)
  torch.testing.assert_close(
    (gradient * direction).sum(),
    (values[0] - values[1]) / (2 * epsilon),
    atol=2e-5,
    rtol=3e-4,
  )


@pytest.mark.parametrize("angle", [0.0, 1e-8, 0.8, 3.141592 - 1e-7])
def test_matrix_integrates_same_physical_state_as_quaternion_reference(angle):
  model, rep, _, state, action, context = fixture()
  state = replace(
    state, rotation=so3_exp(state.position.new_tensor([angle, 0, 0]).expand(2, 3))
  )
  aux, features = state.to_rep()
  torch.testing.assert_close(rep.represent_lie(state), features, atol=0, rtol=0)
  prediction = model.mlp(model._normalized_inputs(features, action, context))
  reference = LieState.from_rep(
    *rep.apply_prediction(aux, features, prediction, model.dt)
  )
  matrix = model.mean_lie(state, action, context)
  assert_state(matrix, reference, atol=3e-10, rtol=3e-10)
  # Repeated composition must preserve SO(3), without quaternion renormalization.
  with torch.no_grad():
    for _ in range(100):
      matrix = model.mean_lie(matrix, action, context)
  matrix.validate()
  torch.testing.assert_close(
    matrix.rotation.mT @ matrix.rotation,
    torch.eye(3, dtype=action.dtype).expand(2, 3, 3),
    atol=2e-13,
    rtol=0,
  )


@pytest.mark.parametrize("chunks", [(None, None), (1, 7), (2, 1)])
def test_vectorized_jacobian_and_warmed_mixed_derivative_match_eager(chunks):
  model, _, vector, state, action, context = fixture(
    config=FilterConfig(
      jacobian_batch_chunk_size=chunks[0], jacobian_chunk_size=chunks[1]
    )
  )
  eager = LieBeliefFilter(
    model,
    vector.measurement_config,
    vector.default_joint_pos,
    FilterConfig(jacobian_backend="eager"),
  )
  for _ in range(3):
    vector.transition_jacobian(state, action, context)
  actual = vector.transition_jacobian(state, action, context)
  expected = eager.transition_jacobian(state, action, context)
  torch.testing.assert_close(actual, expected, atol=5e-12, rtol=5e-10)
  weight = torch.randn_like(actual)
  parameter = model.mlp[0].weight
  gradient = torch.autograd.grad((actual * weight).sum(), parameter)[0]
  reference = torch.autograd.grad((expected * weight).sum(), parameter)[0]
  torch.testing.assert_close(gradient, reference, atol=3e-11, rtol=3e-9)
  directional_check(
    lambda: (vector.transition_jacobian(state, action, context) * weight).sum(),
    parameter,
    gradient,
  )


def test_float32_rollout_stays_on_so3_over_a_long_episode():
  model, _, _, state, action, context = fixture(batch=1, dtype=torch.float32)
  state = replace(
    state,
    rotation=torch.eye(3).unsqueeze(0),
    # At this constant yaw rate, uncorrected fp32 products fail at step 456.
    omega=torch.tensor([[0.0, 0.0, 2.3204725]]),
  )
  with torch.no_grad():
    model.mlp[-1].weight.zero_()
    model.mlp[-1].bias.zero_()
    for _ in range(1000):
      state = model.mean_lie(state, action, context)
  state.validate()
  torch.testing.assert_close(
    state.rotation, so3_exp(state.omega * (1000 * model.dt)), atol=8e-6, rtol=0
  )


def test_filter_requires_native_lie_dynamics_at_construction():
  class AuxRepOnly(torch.nn.Module):
    def forward(self, rep, action, aux, context):
      return aux, rep

    def distribution(self, rep, action, aux, context):
      return aux, rep, torch.eye(70)

  with pytest.raises(TypeError, match="mean_lie and distribution_lie"):
    LieBeliefFilter(AuxRepOnly(), GaussianMeasurementConfig(), torch.zeros(29))


def test_all_tangent_directions_and_unbatched_transition():
  _, _, estimator, state, action, context = fixture(batch=1)
  state, action, context = state.index(0), action[0], context[0]
  actual = estimator.transition_jacobian(state, action, context)
  with torch.no_grad():
    nominal = estimator.mean_step(state, action, context)
    columns = []
    for delta in torch.eye(70, dtype=action.dtype) * 1e-6:
      plus = estimator.mean_step(boxplus(state, delta), action, context)
      minus = estimator.mean_step(boxplus(state, -delta), action, context)
      columns.append((local_error(nominal, plus) - local_error(nominal, minus)) / 2e-6)
  torch.testing.assert_close(actual, torch.stack(columns, -1), atol=2e-8, rtol=2e-6)


def test_two_step_frozen_filter_action_gradient_matches_eager_and_finite_difference():
  model, _, estimator, state, action, context = fixture(batch=1)
  with torch.no_grad():
    model.noise_head[-1].weight.normal_(std=0.001)
  model.requires_grad_(False)
  action = action.clone().requires_grad_()
  covariance = torch.eye(70, dtype=action.dtype).unsqueeze(0) * 0.01
  measurement = estimator.observation(state).detach()
  eager = LieBeliefFilter(
    model,
    estimator.measurement_config,
    estimator.default_joint_pos,
    FilterConfig(jacobian_backend="eager"),
  )

  def objective(filt=estimator):
    belief = BeliefState(state, covariance)
    for _ in range(2):
      belief = filt.correct(filt.predict(belief, action, context), measurement)
    return belief.mean.velocity.sum() + belief.covariance.square().sum()

  gradient = torch.autograd.grad(objective(), action)[0]
  assert torch.isfinite(gradient).all() and gradient.norm() > 0
  torch.testing.assert_close(gradient, torch.autograd.grad(objective(eager), action)[0])
  directional_check(objective, action, gradient)


def test_exact_zero_angular_increment_retains_mixed_derivative():
  model, _, estimator, state, action, context = fixture(batch=1)
  with torch.no_grad():
    model.mlp[-1].weight.zero_()
    model.mlp[-1].bias.zero_()
  state = replace(state, omega=torch.zeros_like(state.omega))
  weight = torch.randn(1, 70, 70, dtype=action.dtype)

  def objective():
    return (estimator.transition_jacobian(state, action, context) * weight).sum()

  gradient = torch.autograd.grad(objective(), model.mlp[-1].bias)[0]
  directional_check(objective, model.mlp[-1].bias, gradient)


def test_frozen_model_retains_state_gradient_and_arbitrary_batch_axes():
  model, _, estimator, state, action, context = fixture()
  model.requires_grad_(False)
  joints = state.joint_pos.clone().requires_grad_()
  state = replace(state, joint_pos=joints)
  weights = torch.randn(2, 70, 70, dtype=action.dtype)

  def objective():
    return (estimator.transition_jacobian(state, action, context) * weights).sum()

  gradient = torch.autograd.grad(objective(), joints)[0]
  assert gradient.norm() > 0 and torch.isfinite(gradient).all()
  directional_check(objective, joints, gradient)
  expanded = state._map(lambda tensor: tensor.unsqueeze(1))
  actual = estimator.transition_jacobian(
    expanded, action.unsqueeze(1), context.unsqueeze(1)
  )
  expected = estimator.transition_jacobian(state, action, context)
  torch.testing.assert_close(actual[:, 0], expected, atol=0, rtol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_float32_production_filter_and_action_gradient(device):
  if device == "cuda" and not torch.cuda.is_available():
    pytest.skip("CUDA unavailable")
  model, _, estimator, state, action, context = fixture(
    dtype=torch.float32, device=device
  )
  model.requires_grad_(False)
  action = action.clone().requires_grad_()
  belief = BeliefState(state, torch.eye(70, device=device).expand(2, 70, 70) * 0.01)
  posterior = estimator.correct(
    estimator.predict(belief, action, context), estimator.observation(state)
  )
  gradient = torch.autograd.grad(
    posterior.mean.velocity.sum() + posterior.covariance.square().sum(), action
  )[0]
  assert torch.isfinite(gradient).all() and gradient.norm() > 0
  with torch.no_grad():
    evaluated = estimator.predict(belief, action, context)
  assert not evaluated.covariance.requires_grad


@pytest.mark.parametrize(
  "kwargs",
  [
    {"jacobian_backend": "other"},
    {"jacobian_batch_chunk_size": 0},
    {"jacobian_chunk_size": -1},
    {"jacobian_chunk_size": True},
  ],
)
def test_jacobian_configuration_rejects_invalid_settings(kwargs):
  with pytest.raises(ValueError):
    FilterConfig(**kwargs)
