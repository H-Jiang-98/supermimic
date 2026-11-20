"""Distribution, differentiation and lifecycle checks for stochastic dynamics."""

from dataclasses import replace

import pytest
import torch

from supermimic.models import Dynamics, ProcessNoiseConfig
from supermimic.representations import GenCoordRepresentation


def make_dynamics(*, stochastic=True, dtype=torch.float64):
  representation = GenCoordRepresentation(29, torch.zeros(29, dtype=dtype))
  representation.dt = 0.02
  representation.substep_coeff = 0.6
  model = Dynamics(
    representation,
    29,
    2,
    torch.full((29,), 0.5, dtype=dtype),
    torch.zeros(29, dtype=dtype),
    (16,),
    process_noise=ProcessNoiseConfig(hidden_dims=(12,)) if stochastic else None,
  ).to(dtype=dtype)
  model.eval()
  return model, representation


def model_inputs(representation, batch=2, dtype=torch.float64):
  state = torch.zeros(batch, 71, dtype=dtype)
  state[:, 2] = 0.8
  state[:, 3:7] = torch.tensor([0.9, 0.1, -0.2, 0.3], dtype=dtype)
  state[:, 3:7] /= state[:, 3:7].norm(dim=-1, keepdim=True)
  state[:, 36:42] = torch.tensor([0.3, -0.1, 0.02, 0.1, 0.2, -0.1], dtype=dtype)
  return (
    representation.represent(state),
    torch.full((batch, 29), 0.12, dtype=dtype),
    representation.extract_aux_state(state),
    torch.full((batch, 2), 0.1, dtype=dtype),
  )


def test_distribution_preserves_mean_and_initial_physical_scale():
  torch.manual_seed(17)
  legacy, representation = make_dynamics(stochastic=False)
  torch.manual_seed(17)
  stochastic, _ = make_dynamics()
  args = model_inputs(representation)
  for name, parameter in legacy.named_parameters():
    torch.testing.assert_close(parameter, dict(stochastic.named_parameters())[name])
  expected = legacy(*args)
  aux, rep, lower = stochastic.distribution(*args)
  # The native matrix path and retained quaternion reference implement the
  # same integration, with different floating-point roundoff.
  torch.testing.assert_close(aux, expected[0], atol=3e-10, rtol=3e-10)
  torch.testing.assert_close(rep, expected[1], atol=3e-10, rtol=3e-10)
  actual_mean = stochastic(*args)
  torch.testing.assert_close(aux, actual_mean[0], atol=0, rtol=0)
  torch.testing.assert_close(rep, actual_mean[1], atol=0, rtol=0)
  std = stochastic.process_noise_config.standard_deviations(dtype=lower.dtype)
  torch.testing.assert_close(
    lower, torch.diag(std).expand(2, 70, 70), rtol=1e-6, atol=1e-9
  )
  assert (
    len(
      [p for name, p in stochastic.named_parameters() if name.startswith("noise_head")]
    )
    > 0
  )
  with pytest.raises(ValueError, match="ProcessNoiseConfig"):
    legacy.distribution(*args)


def test_reparameterized_sampling_is_explicit_and_matches_right_local_covariance():
  from supermimic.utils.lie_state import LieState, local_error

  torch.manual_seed(4)
  model, representation = make_dynamics()
  args = model_inputs(representation, batch=1)
  with torch.no_grad():
    # Include cross terms, so a diagonal-only sampler cannot pass this check.
    model.noise_head[-1].bias[1] = 0.004
    model.noise_head[-1].bias[4] = -0.003
    mean_aux, mean_rep, lower = model.distribution(*args)
    mean = LieState.from_rep(mean_aux, mean_rep)
    epsilon = torch.randn(1, 70, dtype=torch.float64)
    rng = torch.random.get_rng_state()
    first = model.sample(*args, epsilon)
    second = model.sample(*args, epsilon)
    assert torch.equal(rng, torch.random.get_rng_state())
    for a, b in zip(first, second, strict=True):
      torch.testing.assert_close(a, b, rtol=0, atol=0)
    zero = LieState.from_rep(*model.sample(*args, torch.zeros_like(epsilon)))
    torch.testing.assert_close(
      local_error(mean, zero), torch.zeros_like(epsilon), atol=1e-14, rtol=0
    )
    errors = []
    for _ in range(16):
      chunk_args = tuple(value.expand(500, -1) for value in args)
      sampled = LieState.from_rep(
        *model.sample(*chunk_args, torch.randn(500, 70, dtype=torch.float64))
      )
      errors.append(local_error(mean, sampled))
    error = torch.cat(errors)
    covariance = lower[0] @ lower[0].T
    scale = covariance.diagonal().sqrt()
    whitened_by_std = error / scale
    empirical = torch.cov(whitened_by_std.T)
    target = covariance / (scale[:, None] * scale[None, :])
    torch.testing.assert_close(
      empirical.diagonal(), target.diagonal(), atol=0.055, rtol=0
    )
    assert (empirical - target).abs().max() < 0.06
    assert (whitened_by_std.mean(0)).abs().max() < 0.04
  with pytest.raises(ValueError, match="70 entries"):
    model.sample(*args, torch.zeros(1, 69, dtype=torch.float64))


def test_sample_gradients_reach_action_mean_and_noise_and_support_second_derivatives():
  model, representation = make_dynamics()
  rep, action, aux, context = model_inputs(representation)
  action.requires_grad_()
  epsilon = torch.linspace(-1.0, 1.0, 70, dtype=torch.float64).expand(2, 70)
  sample_aux, sample_rep = model.sample(rep, action, aux, context, epsilon)
  loss = sample_aux.square().sum() + sample_rep.square().sum()
  action_gradient = torch.autograd.grad(
    loss, action, create_graph=True, retain_graph=True
  )[0]
  second = torch.autograd.grad(
    action_gradient.square().sum(), action, retain_graph=True
  )[0]
  loss.backward()
  for value in (
    action_gradient,
    second,
    model.mlp[-1].weight.grad,
    model.noise_head[-1].weight.grad,
  ):
    assert torch.isfinite(value).all()
    assert value.abs().sum() > 0


def test_model_version_detects_parameter_and_normalizer_updates_and_checkpoint_restore():
  model, representation = make_dynamics()
  original = model.model_version
  assert model.model_version == original
  checkpoint = model.checkpoint()
  with torch.no_grad():
    model.mlp[-1].bias.add_(0.001)
  assert model.model_version != original
  model.load_checkpoint(checkpoint)
  assert model.model_version == original
  # Explicit normalizer updates change the prediction and therefore the version.
  args = model_inputs(representation)
  model.state_normalizer.train()
  model.state_normalizer.update(args[0])
  assert model.model_version != original
  model.load_checkpoint(checkpoint)
  restored, _ = make_dynamics()
  restored.load_checkpoint(checkpoint)
  epsilon = torch.randn(2, 70, dtype=torch.float64)
  for expected, actual in zip(
    model.sample(*args, epsilon), restored.sample(*args, epsilon), strict=True
  ):
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
  wrong = dict(checkpoint)
  wrong["metadata"] = dict(checkpoint["metadata"], tangent_version="other")
  with pytest.raises(ValueError, match="layout mismatch"):
    restored.load_checkpoint(wrong)
  original_restored_version = restored.model_version
  wrong_weights = dict(checkpoint["state_dict"])
  wrong_weights["mlp.3.bias"] = wrong_weights["mlp.3.bias"] + 1.0
  with pytest.raises(ValueError, match="content does not match"):
    restored.load_checkpoint(dict(checkpoint, state_dict=wrong_weights))
  assert restored.model_version == original_restored_version
  wrong_weights["mlp.3.bias"] = wrong_weights["mlp.3.bias"][:-1]
  with pytest.raises(ValueError, match="Invalid Dynamics checkpoint tensor"):
    restored.load_checkpoint(dict(checkpoint, state_dict=wrong_weights))
  assert restored.model_version == original_restored_version
  legacy, _ = make_dynamics(stochastic=False)
  with pytest.raises(RuntimeError, match="Missing key"):
    restored.load_state_dict(legacy.state_dict())


@pytest.mark.parametrize(
  "kwargs",
  [
    {"diagonal_floor": 0.0},
    {"diagonal_floor": float("nan")},
    {"initial_std": (0.01,)},
    {"initial_std": (float("inf"),) * 6},
    {"hidden_dims": (0,)},
  ],
)
def test_process_noise_configuration_rejects_invalid_scales(kwargs):
  with pytest.raises(ValueError):
    replace(ProcessNoiseConfig(), **kwargs)
