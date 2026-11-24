"""Full-state geometry against independent matrix-exponential references."""

from dataclasses import replace

import pytest
import torch
from torch.autograd import gradcheck, gradgradcheck

from supermimic.belief import BeliefState
from supermimic.estimator_inputs import (
  EITInputConfig,
  PosteriorInput,
  ReferenceWindow,
  RpvState,
  build_actor_features,
)
from supermimic.representations import GenCoordRepresentation
from supermimic.utils.lie_state import (
  LieState,
  boxplus,
  compose,
  exp,
  inverse,
  local_error,
  log,
  reset_jacobian,
  skew,
  so3_exp,
  so3_left_jacobian,
  so3_left_jacobian_inverse,
  so3_log,
)


def _state(batch=2):
  gen = torch.Generator().manual_seed(402)
  delta = torch.randn(batch, 70, generator=gen, dtype=torch.float64) * 0.3
  return exp(delta)


def _matrix_exp(delta):
  """Independent 5x5 SE_2(3) implementation, only for test reference."""
  matrix = delta.new_zeros(*delta.shape[:-1], 5, 5)
  x, y, z = delta[..., :3].unbind(-1)
  matrix[..., 0, 1], matrix[..., 0, 2] = -z, y
  matrix[..., 1, 0], matrix[..., 1, 2] = z, -x
  matrix[..., 2, 0], matrix[..., 2, 1] = -y, x
  matrix[..., :3, 3] = delta[..., 3:6]
  matrix[..., :3, 4] = delta[..., 6:9]
  return torch.matrix_exp(matrix)


def _near_identity_log(matrix):
  # At identity, d Log is vee. Symmetric central differences converge to the
  # exact differential; no production SO3/SE2 logarithm participates here.
  rotation = matrix[..., :3, :3]
  orientation = 0.5 * torch.stack(
    (
      rotation[..., 2, 1] - rotation[..., 1, 2],
      rotation[..., 0, 2] - rotation[..., 2, 0],
      rotation[..., 1, 0] - rotation[..., 0, 1],
    ),
    -1,
  )
  return torch.cat((orientation, matrix[..., :3, 3], matrix[..., :3, 4]), -1)


def test_exp_matches_matrix_exponential_and_retains_coupling():
  generator = torch.Generator().manual_seed(501)
  delta = torch.randn(5, 70, generator=generator, dtype=torch.float64) * 0.8
  state = exp(delta)
  expected = _matrix_exp(delta)
  torch.testing.assert_close(
    state.rotation, expected[..., :3, :3], atol=1e-12, rtol=1e-12
  )
  torch.testing.assert_close(
    state.position, expected[..., :3, 3], atol=1e-12, rtol=1e-12
  )
  torch.testing.assert_close(
    state.velocity, expected[..., :3, 4], atol=1e-12, rtol=1e-12
  )
  assert not torch.allclose(state.position, delta[..., 3:6])
  assert not torch.allclose(state.velocity, delta[..., 6:9])
  torch.testing.assert_close(log(state), delta, atol=1e-12, rtol=1e-12)


def test_full_and_rep_conversion_match_existing_gencoord():
  state = _state()
  state.validate()
  full = state.to_full()
  assert full.shape == (2, 71)
  torch.testing.assert_close(
    local_error(state, LieState.from_full(full)),
    torch.zeros(2, 70, dtype=full.dtype),
    atol=1e-12,
    rtol=0,
  )
  representation = GenCoordRepresentation(29, torch.zeros(29, dtype=full.dtype))
  aux, rep = state.to_rep()
  torch.testing.assert_close(rep, representation.represent(full))
  torch.testing.assert_close(aux, representation.extract_aux_state(full))
  torch.testing.assert_close(LieState.from_rep(aux, rep).to_full(), full)
  torch.testing.assert_close(rep[..., 7:10], state.omega)
  # Existing reconstruction uses rep z as its source of truth.
  inconsistent_aux = aux.clone()
  inconsistent_aux[..., 2] += 4
  torch.testing.assert_close(
    LieState.from_rep(inconsistent_aux, rep).to_full(),
    representation.reconstruct_full_state(inconsistent_aux, rep),
  )


def test_group_inverse_boxplus_and_world_gauge_invariance():
  state = _state()
  delta = torch.linspace(-0.15, 0.2, 70, dtype=torch.float64).expand(2, -1)
  perturbed = boxplus(state, delta)
  torch.testing.assert_close(
    local_error(state, perturbed), delta, atol=1e-12, rtol=1e-12
  )
  zero = torch.zeros_like(delta)
  torch.testing.assert_close(
    log(compose(inverse(state), state)), zero, atol=1e-12, rtol=0
  )
  gauge_delta = torch.zeros_like(delta)
  gauge_delta[..., 2] = 1.2
  gauge_delta[..., 3:6] = torch.tensor([1.0, -4.0, 0.0])
  gauge = exp(gauge_delta)
  torch.testing.assert_close(
    local_error(compose(gauge, state), compose(gauge, perturbed)),
    delta,
    atol=1e-12,
    rtol=1e-12,
  )


@pytest.mark.parametrize("scale", [0.0, 1e-7, 0.2])
def test_so3_first_and_second_derivatives(scale):
  theta = (torch.tensor([0.3, -0.7, 0.2], dtype=torch.float64) * scale).requires_grad_()
  for function in (so3_exp, so3_left_jacobian, so3_left_jacobian_inverse):
    assert gradcheck(function, (theta,), atol=2e-7, rtol=2e-5)
    assert gradgradcheck(function, (theta,), atol=2e-7, rtol=2e-5)
  assert gradcheck(lambda v: so3_log(so3_exp(v)), (theta,), atol=2e-7, rtol=2e-5)
  assert gradgradcheck(lambda v: so3_log(so3_exp(v)), (theta,), atol=2e-7, rtol=2e-5)


def test_zero_state_conversion_and_local_error_mixed_derivatives():
  direction = torch.linspace(-0.2, 0.4, 70, dtype=torch.float64)
  zero = torch.zeros(3, dtype=torch.float64, requires_grad=True)

  def function(value):
    delta = torch.cat((value, direction[3:] * value.square().sum()), -1)
    state = boxplus(_state(1).index(0), delta)
    aux, rep = state.to_rep()
    converted = LieState.from_rep(aux, rep)
    return local_error(_state(1).index(0), converted)

  assert gradcheck(function, (zero,), atol=2e-7, rtol=2e-5)
  assert gradgradcheck(function, (zero,), atol=2e-7, rtol=2e-5)


def test_rep_conversion_vmap_jacrev_remains_valid_after_warmup():
  base = _state(1).index(0)

  def function(delta):
    state = boxplus(base, delta)
    return local_error(base, LieState.from_rep(*state.to_rep()))

  points = torch.zeros(2, 70, dtype=torch.float64)
  expected = torch.eye(70, dtype=points.dtype).expand(2, -1, -1)
  # Scripted quaternion primitives may pass cold calls then fail once their
  # profiling executor installs a differentiable graph. Exercise warmed calls.
  for _ in range(5):
    jacobian = torch.vmap(torch.func.jacrev(function))(points)
    torch.testing.assert_close(jacobian, expected, atol=1e-12, rtol=1e-12)


def test_log_principal_branch_near_pi():
  axis = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)
  axis /= axis.norm()
  for angle in (torch.pi - 1e-7, torch.pi, torch.pi + 1e-7):
    rotation = torch.matrix_exp(skew(axis * angle))
    residual = so3_log(rotation)
    assert torch.isfinite(residual).all()
    assert residual.norm() <= torch.pi + 1e-12
    torch.testing.assert_close(so3_exp(residual), rotation, atol=1e-12, rtol=1e-12)


def test_reset_jacobian_against_independent_group_finite_difference():
  delta = torch.linspace(-0.8, 0.9, 70, dtype=torch.float64)
  h = 1e-6
  columns = []
  inverse_exp = _matrix_exp(-delta)
  for j in range(9):
    direction = torch.zeros_like(delta)
    direction[j] = h
    positive = _near_identity_log(inverse_exp @ _matrix_exp(delta + direction))
    negative = _near_identity_log(inverse_exp @ _matrix_exp(delta - direction))
    columns.append((positive - negative) / (2 * h))
  expected = torch.stack(columns, -1)
  jacobian = reset_jacobian(delta)
  torch.testing.assert_close(jacobian[:9, :9], expected, atol=4e-10, rtol=4e-8)
  assert jacobian[3:6, :3].abs().max() > 0.1
  assert jacobian[6:9, :3].abs().max() > 0.1
  torch.testing.assert_close(jacobian[9:, 9:], torch.eye(61, dtype=delta.dtype))
  torch.testing.assert_close(
    reset_jacobian(torch.zeros_like(delta)), torch.eye(70, dtype=delta.dtype)
  )


@pytest.mark.parametrize("scale", [0.0, 0.5])
def test_reset_jacobian_outer_derivatives_and_vectorized_geometry(scale):
  value = torch.tensor([scale, scale / 3], dtype=torch.float64, requires_grad=True)
  basis = torch.linspace(-0.8, 0.9, 140, dtype=torch.float64).reshape(2, 70)

  def function(v):
    return reset_jacobian(v @ basis)[:9, :9]

  assert gradcheck(function, (value,), atol=2e-7, rtol=2e-5)
  assert gradgradcheck(function, (value,), atol=2e-7, rtol=2e-5)
  points = torch.stack((torch.zeros(70, dtype=torch.float64), basis[0] * 0.1))
  actual = torch.vmap(torch.func.jacrev(lambda v: log(exp(v))))(points)
  torch.testing.assert_close(
    actual, torch.eye(70, dtype=points.dtype).expand(2, -1, -1), atol=1e-12, rtol=1e-12
  )
  torch.testing.assert_close(torch.vmap(reset_jacobian)(points), reset_jacobian(points))


def test_belief_projection_features_and_tensor_lifecycle():
  state = _state()
  gen = torch.Generator().manual_seed(37)
  factor = torch.randn(2, 70, 70, generator=gen, dtype=torch.float64)
  covariance = factor @ factor.transpose(-1, -2) * 0.001
  belief = BeliefState(state, covariance.requires_grad_())
  belief.validate()
  projected = belief.to_actor_input()
  explicit = PosteriorInput(
    RpvState(state.rotation, state.position, state.velocity), covariance[:, :9, :9]
  )
  references = ReferenceWindow(
    torch.zeros(2, 10, 29, dtype=covariance.dtype),
    torch.ones(2, 10, 29, dtype=covariance.dtype),
    state.rotation[:, None].expand(-1, 10, -1, -1),
    state.position[:, None].expand(-1, 10, -1),
    state.velocity[:, None].expand(-1, 10, -1),
  )
  actual = build_actor_features(
    EITInputConfig(group="C"), explicit.mean, projected, references
  )
  expected = build_actor_features(
    EITInputConfig(group="C"), explicit.mean, explicit, references
  )
  for a, e in zip(actual, expected, strict=True):
    torch.testing.assert_close(a, e)
  clone = belief.clone()
  assert clone.covariance.data_ptr() != belief.covariance.data_ptr()
  assert clone.mean.position.data_ptr() != belief.mean.position.data_ptr()
  assert belief.detach().covariance.requires_grad is False
  assert belief.to(dtype=torch.float32).covariance.dtype == torch.float32
  selected = belief.index(torch.tensor([1]))
  assert selected.covariance.shape == (1, 70, 70)
  selected.validate()
  assert belief.index(0).mean.position.shape == (3,)


def test_belief_validation_accepts_psd_and_rejects_invalid_boundaries():
  state = _state()
  covariance = torch.eye(70, dtype=torch.float64).expand(2, -1, -1).clone()
  with pytest.raises(torch.linalg.LinAlgError):
    BeliefState(state, torch.zeros_like(covariance))
  belief = BeliefState(state, covariance)
  with pytest.raises(ValueError, match="tangent version"):
    replace(belief, tangent_version="wrong").validate()
  with pytest.raises(ValueError, match="layout version"):
    replace(belief, state_layout_version="wrong").validate()
  negative = covariance.clone()
  negative[:, 0, 0] = -0.1
  with pytest.raises(ValueError, match="positive semidefinite"):
    BeliefState(belief.mean, negative).validate()
  asymmetric = covariance.clone()
  asymmetric[:, 0, 1] = 0.1
  with pytest.raises(ValueError, match="asymmetric"):
    BeliefState(belief.mean, asymmetric).validate()
  with pytest.raises(ValueError, match="orthonormal"):
    replace(state, rotation=state.rotation * 2).validate()


def test_near_pi_float32_backward_and_valid_promotion():
  theta = torch.tensor([3.14, 0.0, 0.0], requires_grad=True)
  value = so3_log(so3_exp(theta)).sum()
  torch.testing.assert_close(
    torch.autograd.grad(value, theta)[0], torch.ones_like(theta)
  )
  delta = torch.randn(3, 70) * 0.2
  belief = BeliefState(exp(delta), torch.eye(70).expand(3, 70, 70))
  belief.validate()
  promoted = belief.to(torch.float64)
  promoted.validate()
  torch.testing.assert_close(promoted.mean.rotation.float(), belief.mean.rotation)
