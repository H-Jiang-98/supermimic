"""Independent filter references and mixed derivatives, without MuJoCo."""

from contextlib import ExitStack
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from supermimic import estimator as filter_module
from supermimic.belief import BeliefState
from supermimic.estimator import LieBeliefFilter
from supermimic.estimator_inputs import GaussianMeasurementConfig
from supermimic.utils.compile import compile_eit_kernel
from supermimic.utils.lie_state import LieState, boxplus, reset_jacobian, so3_exp
from supermimic.utils.square_root import covariance_root


class _AnalyticDynamics(nn.Module):
  """A nonlinear transition with learnable mean and noise for independent tests."""

  def __init__(self):
    super().__init__()
    self.mean_weight = nn.Parameter(torch.tensor(0.8, dtype=torch.float32))
    self.noise_weight = nn.Parameter(torch.tensor(-3.0, dtype=torch.float32))

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


def _fixture(batch=2, *, dtype=torch.float32, measurement=None, config=None):
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


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def on_device(batch, device, **kwargs):
  estimator, belief, action, context = _fixture(batch, **kwargs)
  estimator.dynamics.to(device)
  estimator.default_joint_pos = estimator.default_joint_pos.to(device)
  belief, action, context = belief.to(device), action.to(device), context.to(device)
  # Populate the immutable sensor cache before measuring the hot path.
  estimator.measurement_config._tensors(29, device=action.device, dtype=action.dtype)
  return estimator, belief, action, context


def reference_correct(estimator, belief, measurement):
  h = estimator.observation_jacobian(belief.mean)
  p = belief.covariance
  v = estimator.measurement_config.covariance(29, device=p.device, dtype=p.dtype)
  s = h @ p @ h.mT + v
  gain = torch.linalg.solve(s, h @ p).mT
  correction = (
    gain @ (measurement - estimator.observation(belief.mean)).unsqueeze(-1)
  ).squeeze(-1)
  mean = boxplus(belief.mean, correction)
  residual = torch.eye(70, device=p.device) - gain @ h
  reset = reset_jacobian(correction)
  covariance = reset @ (residual @ p @ residual.mT + gain @ v @ gain.mT) @ reset.mT
  return mean, covariance, gain


@pytest.mark.parametrize("device", DEVICES)
def test_square_root_forward_first_and_second_derivatives(device):
  torch.manual_seed(42)
  a = torch.randn(2, 4, 7, device=device, requires_grad=True)
  a = a + 2 * torch.eye(4, 7, device=device)
  weights = torch.randn(2, 4, 4, device=device)
  direction = torch.randn_like(a)
  actual = covariance_root(a)
  reference = torch.linalg.cholesky(a @ a.mT)
  torch.testing.assert_close(actual, reference, atol=2e-6, rtol=2e-5)
  records = []
  for factor in (actual, reference):
    loss = (factor * weights).sin().sum()
    gradient = torch.autograd.grad(loss, a, create_graph=True, retain_graph=True)[0]
    hessian_vector = torch.autograd.grad(
      (gradient * direction).sum(), a, retain_graph=True
    )[0]
    records.append((gradient, hessian_vector))
  torch.testing.assert_close(records[0], records[1], atol=3e-5, rtol=3e-4)
  epsilon = 2e-3
  finite = (
    (covariance_root(a + epsilon * direction) * weights).sin().sum()
    - (covariance_root(a - epsilon * direction) * weights).sin().sum()
  ) / (2 * epsilon)
  torch.testing.assert_close(
    (records[0][0] * direction).sum(), finite, atol=2e-3, rtol=3e-3
  )


@pytest.mark.parametrize("device", DEVICES)
def test_prediction_matches_covariance_reference_and_model_gradients(device):
  estimator, belief, action, context = on_device(2, device)
  action.requires_grad_()
  result, diagnostics = estimator.predict(
    belief, action, context, return_diagnostics=True
  )
  f = diagnostics.transition_jacobian
  expected = f @ belief.covariance @ f.mT + diagnostics.process_covariance
  torch.testing.assert_close(result.covariance, expected, atol=2e-8, rtol=3e-5)
  inputs = (action, *estimator.dynamics.parameters())
  actual_grad = torch.autograd.grad(
    result.covariance.square().sum(), inputs, retain_graph=True
  )
  expected_grad = torch.autograd.grad(expected.square().sum(), inputs)
  torch.testing.assert_close(actual_grad, expected_grad, atol=2e-7, rtol=5e-4)


@pytest.mark.parametrize("device", DEVICES)
def test_correction_matches_independent_joseph_and_gradients(device):
  estimator, belief, _, _ = on_device(3, device)
  lower = belief.covariance_sqrt.detach().requires_grad_()
  belief = replace(belief, covariance_sqrt=lower)
  measurement = (estimator.observation(belief.mean) + 0.01).detach().requires_grad_()
  posterior, diagnostics = estimator.correct(
    belief, measurement, return_diagnostics=True
  )
  mean, covariance, gain = reference_correct(estimator, belief, measurement)
  torch.testing.assert_close(
    posterior.mean.to_full(), mean.to_full(), atol=2e-6, rtol=1e-4
  )
  torch.testing.assert_close(posterior.covariance, covariance, atol=3e-8, rtol=3e-4)
  torch.testing.assert_close(diagnostics.kalman_gain, gain, atol=5e-6, rtol=2e-4)
  gradients = []
  for m, p in ((posterior.mean, posterior.covariance), (mean, covariance)):
    loss = m.to_full().square().sum() + p.square().sum()
    gradients.append(torch.autograd.grad(loss, (measurement, lower), retain_graph=True))
  torch.testing.assert_close(gradients[0], gradients[1], atol=1e-5, rtol=2e-3)


@pytest.mark.parametrize("device", DEVICES)
def test_shapes_no_grad_and_input_immutability(device):
  estimator, prior, action, context = on_device(6, device)
  saved = prior.clone()
  y = estimator.observation(prior.mean) + 0.01
  with torch.no_grad():
    expected = estimator.predict_correct(prior, action, context, y)
    scalar = estimator.predict_correct(prior.index(0), action[0], context[0], y[0])
    mean = LieState(
      *(
        getattr(prior.mean, f.name).reshape(
          2, 3, *getattr(prior.mean, f.name).shape[1:]
        )
        for f in filter_module.fields(LieState)
      )
    )
    shaped = estimator.predict_correct(
      BeliefState(mean, covariance_sqrt=prior.covariance_sqrt.reshape(2, 3, 70, 70)),
      action.reshape(2, 3, 29),
      context.reshape(2, 3, 2),
      y.reshape(2, 3, 64),
    )
  torch.testing.assert_close(
    scalar.covariance, expected.covariance[0], atol=1e-7, rtol=1e-4
  )
  torch.testing.assert_close(shaped.covariance.reshape(6, 70, 70), expected.covariance)
  torch.testing.assert_close(
    prior.covariance_sqrt, saved.covariance_sqrt, atol=0, rtol=0
  )
  assert not expected.covariance_sqrt.requires_grad


@pytest.mark.parametrize("device", DEVICES)
def test_zero_innovation_and_mixed_sensor_units(device):
  estimator, prior, _, _ = on_device(2, device)
  std = torch.full((70,), 0.1, device=device)
  std[12:41], std[41:] = 0.01, 5.0
  prior = replace(prior, covariance_sqrt=std.diag().expand(2, 70, 70))
  posterior = estimator.correct(prior, estimator.observation(prior.mean))
  torch.testing.assert_close(
    posterior.mean.to_full(), prior.mean.to_full(), atol=1e-6, rtol=1e-5
  )
  noise = estimator.measurement_config.joint_position_std**2
  expected = std[12:41].square() * noise / (std[12:41].square() + noise)
  torch.testing.assert_close(
    posterior.covariance[:, 12:41, 12:41].diagonal(dim1=-2, dim2=-1),
    expected.expand(2, 29),
    atol=1e-9,
    rtol=1e-4,
  )


@pytest.mark.parametrize("device", DEVICES)
def test_long_sequence_keeps_finite_positive_factors(device):
  estimator, belief, action, context = on_device(2, device)
  with torch.no_grad():
    for _ in range(100):
      predicted = estimator.predict(belief, action, context)
      belief = estimator.correct(
        predicted, estimator.observation(predicted.mean) + 0.001
      )
  belief.validate()
  assert torch.isfinite(belief.covariance_sqrt).all()
  assert torch.linalg.cholesky_ex(belief.covariance).info.eq(0).all()


def test_hot_path_has_no_numerical_validation_or_spectral_factorizations():
  estimator, belief, action, context = on_device(2, "cpu")
  measurement = estimator.observation(belief.mean)

  def forbidden(*args, **kwargs):
    raise AssertionError("unexpected numerical validation on hot path")

  with ExitStack() as stack:
    for name in ("cholesky", "cholesky_ex", "eigvalsh", "eigh", "pinv", "matrix_rank"):
      stack.enter_context(patch.object(torch.linalg, name, forbidden))
    stack.enter_context(patch.object(BeliefState, "validate", forbidden))
    stack.enter_context(patch.object(LieState, "validate", forbidden))
    estimator.predict_correct(belief, action, context, measurement)


def test_explicit_validation_and_structure_errors():
  estimator, belief, action, context = _fixture()
  bad = belief.covariance.clone()
  bad[:, 0, 0] = -1
  with pytest.raises(ValueError, match="semidefinite"):
    estimator.initialize(belief.mean, bad)
  estimator.config = replace(estimator.config, validate_numerics=True)
  measurement = estimator.observation(belief.mean)
  measurement[:, 0] = float("nan")
  with pytest.raises(ValueError, match="finite"):
    estimator.correct(belief, measurement)
  with pytest.raises(ValueError, match="version"):
    estimator.predict(replace(belief, tangent_version="invalid"), action, context)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
  "scale,boundary_angle",
  [(scale, None) for scale in (0.0, 0.02, 0.5, 1.0, 3.0)]
  + [(0.5, angle) for angle in (1 - 2**-20, 1.0, 1 + 2**-20)],
)
def test_analytic_reset_matches_exponential_values_and_two_derivatives(
  device, scale, boundary_angle
):
  from supermimic.utils.lie_state import skew

  torch.manual_seed(83)
  delta = torch.randn(2, 70, device=device, dtype=torch.float32) * scale
  if boundary_angle is not None:
    delta[:, :3] = 0
    delta[0, 0], delta[1, 1] = boundary_angle, -boundary_angle
  delta.requires_grad_()
  theta, rho = (skew(delta[..., i : i + 3]) for i in (0, 3))
  zero = torch.zeros_like(theta)
  ad = torch.cat(
    (
      torch.cat((theta, zero, zero), -1),
      torch.cat((rho, theta, zero), -1),
      torch.cat((zero, zero, zero), -1),
    ),
    -2,
  )
  augmented = torch.cat(
    (
      torch.cat((-ad, torch.eye(9, device=device).expand_as(ad)), -1),
      torch.zeros(2, 9, 18, device=device),
    ),
    -2,
  )
  reference = torch.matrix_exp(augmented)[..., :9, 9:]
  actual = reset_jacobian(delta)[..., :9, :9]
  torch.testing.assert_close(actual, reference, atol=4e-6, rtol=3e-5)
  weight = torch.randn_like(actual)
  direction = torch.randn_like(delta)
  records = []
  for jacobian in (actual, reference):
    gradient = torch.autograd.grad(
      (jacobian * weight).sin().sum(), delta, create_graph=True, retain_graph=True
    )[0]
    second = torch.autograd.grad(
      (gradient * direction).sum(), delta, retain_graph=True
    )[0]
    records.append((gradient, second))
  torch.testing.assert_close(records[0], records[1], atol=3e-5, rtol=8e-4)


@pytest.mark.parametrize("device", DEVICES)
def test_compiled_reset_boundary_gradient_float32(device):
  def entries(delta):
    jacobian = reset_jacobian(delta)
    return torch.stack((jacobian[..., 1, 1], jacobian[..., 2, 1]), -1)

  compiled = compile_eit_kernel(entries, dynamic=True)
  delta = torch.zeros(3, 70, device=device, dtype=torch.float32)
  delta[:, 0] = delta.new_tensor([1 - 2**-20, 1.0, 1 + 2**-20])
  delta.requires_grad_()
  angle = delta[:, 0]
  # Independent closed forms for an x-axis correction, including the switch.
  reference = torch.stack((angle.sin() / angle, (angle.cos() - 1) / angle), -1)
  actual = compiled(delta)
  torch.testing.assert_close(actual, reference, atol=2e-7, rtol=2e-6)
  actual_gradient = torch.autograd.grad(actual.square().sum(), delta)[0]
  reference_gradient = torch.autograd.grad(reference.square().sum(), delta)[0]
  torch.testing.assert_close(actual_gradient, reference_gradient, atol=2e-6, rtol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA capture regression")
def test_cuda_filter_forward_backward_capture_and_no_scalar_reads():
  estimator, belief, action, context = on_device(2, "cuda")
  lower = belief.covariance_sqrt.detach().requires_grad_()
  belief = replace(belief, covariance_sqrt=lower)
  measurement = estimator.observation(belief.mean) + 0.01

  def run():
    result = estimator.correct(belief, measurement)
    return torch.autograd.grad(result.covariance_sqrt.square().sum(), lower)[0]

  stream = torch.cuda.Stream()
  stream.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(stream):
    for _ in range(3):
      run()
  torch.cuda.current_stream().wait_stream(stream)
  graph = torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph):
    gradient = run()
  graph.replay()
  torch.testing.assert_close(gradient, run())
  with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
    run()
    estimator.predict_correct(belief, action, context, measurement)
  keys = {e.key for e in prof.key_averages()}
  assert (
    not {"aten::item", "aten::_local_scalar_dense", "aten::_linalg_check_errors"} & keys
  )
