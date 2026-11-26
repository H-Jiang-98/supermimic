"""Factored 35-output world-model Jacobians versus full-state differentiation."""

from dataclasses import replace

import pytest
import torch

from supermimic.belief import BeliefState
from supermimic.estimator import FilterConfig, LieBeliefFilter
from supermimic.utils.lie_state import boxplus, local_error
from tests.test_estimator_matrix_dynamics import directional_check, fixture


def incremental(reference, **kwargs):
  return LieBeliefFilter(
    reference.dynamics,
    reference.measurement_config,
    reference.default_joint_pos,
    replace(reference.config, jacobian_backend="increment"),
    **kwargs,
  )


@pytest.mark.parametrize(("count", "widths"), [(2047, [512] * 4), (513, [512, 16])])
def test_compiled_tail_dispatch_preserves_rows_and_gradients(count, widths):
  _, _, estimator, *_ = fixture(config=FilterConfig(jacobian_batch_chunk_size=512))
  seen = []

  def kernel(epsilon, state, nominal, action, context):
    seen.append(len(action))
    return action.square()

  estimator._compiled_jacobian = kernel
  action = torch.randn(count, 2, requires_grad=True)
  actual = estimator._compiled_jacobian_blocks(
    action, (action,), (action,), action, action
  )
  assert seen == widths
  torch.testing.assert_close(actual, action.square())
  torch.testing.assert_close(torch.autograd.grad(actual.sum(), action)[0], 2 * action)


@pytest.mark.parametrize("chunks", [(None, None), (1, 7), (2, 1)])
@pytest.mark.parametrize("offset_nominal", [False, True])
def test_increment_matches_full_jacobian_and_all_outer_gradients(
  chunks, offset_nominal
):
  model, _, full, state, action, context = fixture(
    config=FilterConfig(
      jacobian_batch_chunk_size=chunks[0], jacobian_chunk_size=chunks[1]
    )
  )
  fast = incremental(full)
  # Nontrivial normalizers and action scaling are included in the network AD.
  with torch.no_grad():
    model.state_normalizer._std.fill_(1.4)
    model.state_normalizer._mean.fill_(0.12)
    model.action_normalizer._std.fill_(0.8)
  perturbation = torch.randn(2, 70, dtype=action.dtype).mul_(0.02).requires_grad_()
  state = boxplus(state, perturbation)
  action, context = action.requires_grad_(), context.requires_grad_()
  nominal = full.mean_step(state, action, context)
  if offset_nominal:
    nominal = boxplus(nominal, torch.randn_like(perturbation) * 0.01)
  actual = fast.transition_jacobian(state, action, context, nominal)
  expected = full.transition_jacobian(state, action, context, nominal)
  torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-10)
  weight = torch.randn_like(actual)
  inputs = (perturbation, action, context, *model.mlp.parameters())
  gradient = torch.autograd.grad((actual * weight).sum(), inputs, retain_graph=True)
  reference = torch.autograd.grad((expected * weight).sum(), inputs)
  torch.testing.assert_close(gradient, reference, atol=3e-11, rtol=3e-9)


@pytest.mark.parametrize("coefficient", [0.5, 0.6, 1.0])
def test_increment_matches_finite_differences_for_all_70_directions(coefficient):
  _, rep, full, state, action, context = fixture(batch=1)
  rep.substep_coeff = coefficient
  fast = incremental(full)
  state, action, context = state.index(0), action[0], context[0]
  actual = fast.transition_jacobian(state, action, context)
  with torch.no_grad():
    nominal = full.mean_step(state, action, context)
    columns = []
    for delta in torch.eye(70, dtype=action.dtype) * 1e-6:
      plus = full.mean_step(boxplus(state, delta), action, context)
      minus = full.mean_step(boxplus(state, -delta), action, context)
      columns.append((local_error(nominal, plus) - local_error(nominal, minus)) / 2e-6)
  torch.testing.assert_close(actual, torch.stack(columns, -1), atol=2e-8, rtol=2e-6)


def test_increment_zero_rotation_mixed_derivative():
  model, _, full, state, action, context = fixture(batch=1)
  with torch.no_grad():
    model.mlp[-1].weight.zero_()
    model.mlp[-1].bias.zero_()
  state = replace(state, omega=torch.zeros_like(state.omega))
  fast = incremental(full)
  weight = torch.randn(1, 70, 70, dtype=action.dtype)

  def objective():
    return (fast.transition_jacobian(state, action, context) * weight).sum()

  gradient = torch.autograd.grad(objective(), model.mlp[-1].bias)[0]
  assert torch.isfinite(gradient).all()
  directional_check(objective, model.mlp[-1].bias, gradient)


def test_increment_batch_axes_empty_batch_and_no_grad():
  _, _, full, state, action, context = fixture()
  fast = incremental(full)
  expanded = state._map(lambda tensor: tensor.unsqueeze(1))
  with torch.no_grad():
    result = fast.transition_jacobian(expanded, action[:, None], context[:, None])
    expected = full.transition_jacobian(state, action, context)
    assert not result.requires_grad and result.shape == (2, 1, 70, 70)
    torch.testing.assert_close(result[:, 0], expected, atol=2e-12, rtol=2e-10)
    empty = fast.transition_jacobian(state.index(slice(0, 0)), action[:0], context[:0])
    assert empty.shape == (0, 70, 70)
  with torch.inference_mode(), pytest.raises(RuntimeError, match="local autograd"):
    fast.transition_jacobian(state, action, context)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_increment_two_step_filter_and_frozen_model_actor_gradient(dtype):
  model, _, full, state, action, context = fixture(batch=1, dtype=dtype)
  model.requires_grad_(False)
  fast = incremental(full)
  action = action.clone().requires_grad_()
  prior = BeliefState(state, torch.eye(70, dtype=dtype).unsqueeze(0) * 0.01)
  measurement = full.observation(state).detach() + 0.01

  def evaluate(estimator):
    belief = prior
    for _ in range(2):
      belief = estimator.predict_correct(belief, action, context, measurement)
    loss = belief.mean.velocity.sum() + belief.covariance.square().sum()
    return belief.mean.to_full(), belief.covariance, torch.autograd.grad(loss, action)

  torch.testing.assert_close(evaluate(fast), evaluate(full))


def test_increment_rejects_dynamics_without_split_integration_api():
  from tests.test_estimator_belief_filter import _fixture

  reference, _, _, _ = _fixture()
  with pytest.raises(TypeError, match="prediction_lie and integrate_prediction_lie"):
    incremental(reference)


@pytest.mark.parametrize("backend", [None, "vmap", "eager"])
def test_train_eit_selects_increment_by_default_and_accepts_reference_backends(
  backend, monkeypatch, tmp_path
):
  import sys
  from unittest.mock import MagicMock

  from supermimic.scripts import train_eit

  class ConfigChecked(Exception):
    pass

  def inspect_runner(env, cfg, *args, **kwargs):
    assert cfg.filter_config.jacobian_backend == (backend or "increment")
    raise ConfigChecked

  argv = [
    "train_eit",
    "--group",
    "mean",
    "--device",
    "cpu",
    "--motion-file",
    str(tmp_path / "motion.npz"),
    "--output",
    str(tmp_path),
  ]
  if backend is not None:
    argv.extend(("--jacobian-backend", backend))
  monkeypatch.setattr(sys, "argv", argv)
  monkeypatch.setenv("WANDB_MODE", "disabled")
  monkeypatch.setattr(train_eit, "create_env", MagicMock())
  monkeypatch.setattr(train_eit, "SuperMimicRunner", inspect_runner)
  with pytest.raises(ConfigChecked):
    train_eit.main()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Inductor regression")
def test_compiled_increment_matches_full_values_and_outer_gradients():
  model, _, full, state, action, context = fixture(
    batch=3,
    dtype=torch.float32,
    device="cuda",
    config=FilterConfig(jacobian_batch_chunk_size=2),
  )
  model.requires_grad_(False)
  fast = incremental(full, compile_kernels=True)
  action = action.clone().requires_grad_()
  weight = torch.randn(3, 70, 70, device="cuda")
  for count in (3, 1):
    inputs = state.index(slice(0, count)), action[:count], context[:count]
    actual = fast.transition_jacobian(*inputs)
    expected = full.transition_jacobian(*inputs)
    gradient = torch.autograd.grad((actual * weight[:count]).sum(), action)[0]
    reference = torch.autograd.grad((expected * weight[:count]).sum(), action)[0]
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=1e-4)
    torch.testing.assert_close(gradient, reference, atol=3e-6, rtol=1e-4)
    with torch.no_grad():
      evaluated = fast.transition_jacobian(*inputs)
    assert not evaluated.requires_grad
    torch.testing.assert_close(evaluated, expected, atol=3e-6, rtol=1e-4)
