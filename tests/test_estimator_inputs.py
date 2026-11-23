"""Geometry/information-permission tests without a running simulator."""

import json
from dataclasses import replace

import pytest
import torch

from supermimic.estimator_inputs import (
  EITInputConfig,
  EpisodeAlignment,
  GaussianMeasurementConfig,
  PosteriorInput,
  ReferenceWindow,
  RpvState,
  _measurement_tensors,
  apply_alignment,
  build_actor_critic_goals,
  build_actor_features,
  build_motion_goal,
  create_episode_alignment,
  encode_covariance,
  extract_rpv_covariance,
  sample_proprio,
  skew,
  task_error_covariance,
  task_error_jacobian,
)


def _inputs(batch=2, dtype=torch.float64):
  generator = torch.Generator().manual_seed(39)

  def rand(*shape):
    return torch.randn(*shape, dtype=dtype, generator=generator)

  state = RpvState(
    torch.matrix_exp(skew(rand(batch, 3))), rand(batch, 3), rand(batch, 3)
  )
  reference = ReferenceWindow(
    rand(batch, 10, 29),
    rand(batch, 10, 29),
    torch.matrix_exp(skew(rand(batch, 10, 3))),
    rand(batch, 10, 3),
    rand(batch, 10, 3),
  )
  factor = rand(batch, 9, 9)
  covariance = factor @ factor.transpose(-1, -2) + torch.eye(9, dtype=dtype) * 0.1
  return state, reference, covariance


def _perturb(state, delta):
  """Independent SE_2(3) exponential, not the Jacobian under test."""
  algebra = delta.new_zeros(*delta.shape[:-1], 5, 5)
  algebra[..., :3, :3] = skew(delta[..., :3])
  algebra[..., :3, 3] = delta[..., 3:6]
  algebra[..., :3, 4] = delta[..., 6:9]
  exponential = torch.matrix_exp(algebra)
  return RpvState(
    state.rotation @ exponential[..., :3, :3],
    state.position + (state.rotation @ exponential[..., :3, 3:4]).squeeze(-1),
    state.velocity + (state.rotation @ exponential[..., :3, 4:5]).squeeze(-1),
  )


def _orientation_residual(state, perturbed, reference):
  base = state.rotation.transpose(-1, -2)[:, None] @ reference.rotation
  moved = perturbed.rotation.transpose(-1, -2)[:, None] @ reference.rotation
  relative = moved @ base.transpose(-1, -2)
  vector = (
    torch.stack(
      (
        relative[..., 2, 1] - relative[..., 1, 2],
        relative[..., 0, 2] - relative[..., 2, 0],
        relative[..., 1, 0] - relative[..., 0, 1],
      ),
      dim=-1,
    )
    * 0.5
  )
  sine = vector.norm(dim=-1, keepdim=True)
  cosine = (relative.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True) - 1) * 0.5
  angle = torch.atan2(sine, cosine)
  return vector * (angle / sine.clamp_min(1e-30))


def test_config_roundtrip_and_permissions():
  for group in ("A", "B", "C"):
    config = EITInputConfig(group=group)
    restored = EITInputConfig.from_dict(json.loads(json.dumps(config.to_dict())))
    assert restored == config
    assert config.uses_posterior == (group != "A")
    assert config.uses_uncertainty == (group == "C")
    assert config.critic_uses_belief == (group != "A")
    assert replace(config, mode="eval").uses_posterior
  with pytest.raises(ValueError, match="version"):
    EITInputConfig.from_dict({"group": "A"})
  with pytest.raises(ValueError, match="version"):
    EITInputConfig(version="old")


@pytest.mark.parametrize(
  "name",
  ["gravity_std", "angular_velocity_std", "joint_position_std", "joint_velocity_std"],
)
@pytest.mark.parametrize("invalid", [0.0, -0.1, float("nan"), float("inf")])
def test_invalid_measurement_noise_identifies_field_in_config_and_checkpoint(
  name, invalid
):
  with pytest.raises(ValueError, match=name):
    GaussianMeasurementConfig(**{name: invalid})
  saved = EITInputConfig().to_dict()
  saved["measurement_noise"][name] = invalid
  with pytest.raises(ValueError, match=name):
    EITInputConfig.from_dict(saved)


@pytest.mark.parametrize("value", [1e-30, 1e20])
def test_noise_variance_is_representable_at_the_point_of_use(value):
  config = GaussianMeasurementConfig(joint_velocity_std=value)
  assert config.covariance(29, dtype=torch.float64)[-1, -1] > 0
  with pytest.raises(ValueError, match="joint_velocity_std.*float32"):
    sample_proprio(torch.zeros(1, 93), config)
  with pytest.raises(ValueError, match="joint_velocity_std.*float32"):
    config.covariance(29, dtype=torch.float32)


def test_noise_cache_isolated_from_callers_and_usable_after_inference():
  config = GaussianMeasurementConfig(gravity_std=0.1234)
  with torch.inference_mode():
    config.standard_deviations(4).zero_()
    config.covariance(4).zero_()
  std, variance = config._tensors(4, device="cpu", dtype=torch.float32)
  assert not torch.is_inference(std) and (std > 0).all()
  torch.testing.assert_close(config.covariance(4).diag(), variance)
  differentiable = torch.ones_like(std, requires_grad=True)
  torch.testing.assert_close(
    torch.autograd.grad((differentiable * std).sum(), differentiable)[0], std
  )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("operation", ["sampling", "covariance"])
def test_noise_cache_supports_repeated_higher_order_transforms(dtype, operation):
  config = GaussianMeasurementConfig()
  if operation == "sampling":
    value = torch.ones(93, dtype=dtype)
    expected = 2 * torch.eye(93, dtype=dtype)

    def objective(value):
      return sample_proprio(value, config).square().sum()

    hessian = torch.func.jacrev(torch.func.jacrev(objective))
  else:
    value = torch.ones(64, dtype=dtype)
    expected = 2 * config.covariance(29, dtype=dtype).square()

    def objective(value):
      covariance = config.covariance(29, device=value.device, dtype=value.dtype)
      return (covariance @ value).square().sum()

    hessian = torch.func.hessian(objective)

  # The first transformed call must create the constants, without prior warmup.
  _measurement_tensors.cache_clear()
  for _ in range(3):
    torch.testing.assert_close(hessian(value), expected)

  # Constants first requested by torch.func also remain usable by ordinary AD.
  covariance = config.covariance(29, dtype=dtype)
  sample = torch.ones(64, dtype=dtype, requires_grad=True)
  gradient = torch.autograd.grad((covariance @ sample).sum(), sample)[0]
  torch.testing.assert_close(gradient, covariance.diag())


@pytest.mark.parametrize("kind", ["zero", "reflection"])
def test_injected_and_reference_rotations_must_belong_to_so3(kind):
  state, reference, cov = _inputs(dtype=torch.float32)
  invalid = (
    torch.zeros_like(state.rotation) if kind == "zero" else state.rotation.clone()
  )
  if kind == "reflection":
    invalid[:, :, 0].neg_()
  posterior = PosteriorInput(replace(state, rotation=invalid), cov)
  with pytest.raises(ValueError, match="Rotation"):
    build_actor_features(EITInputConfig(group="B"), state, posterior, reference)
  with pytest.raises(ValueError, match="Rotation"):
    build_motion_goal(
      state, replace(reference, rotation=invalid[:, None].expand_as(reference.rotation))
    )


@pytest.mark.parametrize("invalid", ["negative", "correlation", "deterministic"])
def test_large_unrelated_variance_cannot_mask_invalid_actor_covariance(invalid):
  cov = torch.eye(9) * 1e-4
  cov[8, 8] = 100
  if invalid == "negative":
    cov[0, 0] *= -1
  elif invalid == "correlation":
    cov[0, 1] = cov[1, 0] = 2e-4
  else:
    cov[0, 0] = 0
    cov[0, 1] = cov[1, 0] = 1e-10
  with pytest.raises(ValueError, match="semidefinite"):
    encode_covariance(cov)


def test_goal_layout_and_required_inputs():
  state, reference, covariance = _inputs()
  mean = replace(state, position=state.position + 0.3)
  posterior = PosteriorInput(mean, covariance)
  a, no_uncertainty = build_actor_features(EITInputConfig(), state, None, reference)
  b, _ = build_actor_features(EITInputConfig(group="B"), None, posterior, reference)
  c, uncertainty = build_actor_features(
    EITInputConfig(group="C"), None, posterior, reference
  )
  assert a.shape == b.shape == c.shape == (2, 700)
  assert uncertainty.shape == (2, 450)
  assert no_uncertainty is None
  torch.testing.assert_close(b, c)
  torch.testing.assert_close(
    a[:, :580], torch.cat((reference.joint_pos, reference.joint_vel), -1).flatten(1)
  )
  inverse = state.rotation.transpose(-1, -2)[:, None]
  torch.testing.assert_close(
    a[:, 610:670], (inverse @ reference.rotation)[..., :2].flatten(1)
  )
  torch.testing.assert_close(
    a[:, 670:],
    (inverse @ (reference.velocity - state.velocity[:, None]).unsqueeze(-1))
    .squeeze(-1)
    .flatten(1),
  )
  assert not torch.allclose(a, b)
  for config in (
    EITInputConfig(group="B"),
    EITInputConfig(group="C"),
    EITInputConfig(mode="eval"),
  ):
    with pytest.raises(ValueError, match="posterior mean"):
      build_actor_features(config, state, None, reference)
  with pytest.raises(ValueError, match="covariance"):
    build_actor_features(
      EITInputConfig(group="C"), state, PosteriorInput(mean), reference
    )


def test_actor_information_isolation():
  state, reference, covariance = _inputs()
  changed_truth = replace(
    state, position=state.position + 100, velocity=state.velocity - 50
  )
  posterior = PosteriorInput(state, covariance)
  for group in ("B", "C"):
    config = EITInputConfig(group=group)
    before = build_actor_features(config, state, posterior, reference)
    after = build_actor_features(config, changed_truth, posterior, reference)
    torch.testing.assert_close(before[0], after[0], atol=0, rtol=0)
    if group == "C":
      torch.testing.assert_close(before[1], after[1], atol=0, rtol=0)
  for group in ("A", "B"):
    config = EITInputConfig(group=group)
    # Invalid covariance proves it is never read by A/B.
    invalid = PosteriorInput(state, torch.full_like(covariance, float("nan")))
    before = build_actor_features(config, state, posterior, reference)
    after = build_actor_features(config, state, invalid, reference)
    omitted = build_actor_features(config, state, PosteriorInput(state), reference)
    assert before[1] is after[1] is omitted[1] is None
    torch.testing.assert_close(before[0], after[0], atol=0, rtol=0)
    torch.testing.assert_close(before[0], omitted[0], atol=0, rtol=0)


@pytest.mark.parametrize(
  "group,mode", [("A", "train"), ("A", "eval"), ("B", "train"), ("C", "train")]
)
def test_shared_reference_goals_match_standalone_actor_and_privileged_critic(
  group, mode
):
  truth, reference, covariance = _inputs()
  posterior = PosteriorInput(replace(truth, position=truth.position + 1), covariance)
  config = EITInputConfig(group=group, mode=mode)
  actor, uncertainty, critic = build_actor_critic_goals(
    config, truth, posterior, reference
  )
  expected_actor, expected_uncertainty = build_actor_features(
    config, truth, posterior, reference
  )
  torch.testing.assert_close(actor, expected_actor, atol=0, rtol=0)
  torch.testing.assert_close(uncertainty, expected_uncertainty, atol=0, rtol=0)
  torch.testing.assert_close(
    critic, build_motion_goal(truth, reference)[0], atol=0, rtol=0
  )
  assert actor.data_ptr() != critic.data_ptr()


def test_right_local_jacobian_finite_difference():
  state, reference, _ = _inputs()
  _, position, velocity = build_motion_goal(state, reference)
  jacobian = task_error_jacobian(position, velocity)
  finite_difference = []
  eps = 1e-6
  for axis in range(9):
    delta = state.position.new_zeros(2, 9)
    delta[:, axis] = eps
    plus, minus = _perturb(state, delta), _perturb(state, -delta)
    _, plus_p, plus_v = build_motion_goal(plus, reference)
    _, minus_p, minus_v = build_motion_goal(minus, reference)
    orient = _orientation_residual(state, plus, reference) - _orientation_residual(
      state, minus, reference
    )
    finite_difference.append(
      torch.cat((orient, plus_p - minus_p, plus_v - minus_v), -1) / (2 * eps)
    )
  torch.testing.assert_close(
    torch.stack(finite_difference, -1), jacobian, atol=2e-9, rtol=2e-8
  )


def test_shared_perturbation_monte_carlo_covariance():
  state, reference, covariance = _inputs(batch=1)
  covariance = covariance * 1e-8
  _, position, velocity = build_motion_goal(state, reference)
  expected = task_error_covariance(covariance, position, velocity)[0]
  samples = 14000
  delta = torch.randn(
    samples, 9, dtype=torch.float64, generator=torch.Generator().manual_seed(719)
  )
  delta = delta @ torch.linalg.cholesky(covariance[0]).T
  perturbed = _perturb(state, delta)
  expanded_ref = ReferenceWindow(
    *(
      v.expand(samples, *v.shape[1:])
      for v in (
        reference.joint_pos,
        reference.joint_vel,
        reference.rotation,
        reference.position,
        reference.velocity,
      )
    )
  )
  # Position/velocity directly, avoiding an unnecessary 700-wide target allocation.
  inverse = perturbed.rotation.transpose(-1, -2)[:, None]
  p = (
    inverse @ (expanded_ref.position - perturbed.position[:, None]).unsqueeze(-1)
  ).squeeze(-1)
  v = (
    inverse @ (expanded_ref.velocity - perturbed.velocity[:, None]).unsqueeze(-1)
  ).squeeze(-1)
  residual = torch.cat(
    (_orientation_residual(state, perturbed, expanded_ref), p - position, v - velocity),
    -1,
  )
  centered = residual - residual.mean(0)
  empirical = torch.einsum("nti,ntj->tij", centered, centered) / (samples - 1)
  assert torch.linalg.norm(empirical - expected) / torch.linalg.norm(expected) < 0.035
  # The first and last reference frames are correlated, not independent samples.
  empirical_cross = centered[:, 0].T @ centered[:, -1] / (samples - 1)
  jacobian = task_error_jacobian(position, velocity)[0]
  cross = jacobian[0] @ covariance[0] @ jacobian[-1].T
  assert torch.linalg.norm(empirical_cross - cross) / torch.linalg.norm(cross) < 0.04


def test_horizontal_invariance_and_fixed_episode_alignment():
  state, reference, covariance = _inputs()
  yaw = state.position.new_tensor([[0, 0, 0.7], [0, 0, -1.2]])
  rotation = torch.matrix_exp(skew(yaw))
  translation = state.position.new_tensor([[3, -2, 0], [-1, 4, 0]])
  transform = EpisodeAlignment(rotation, translation)
  moved = RpvState(
    rotation @ state.rotation,
    (rotation @ state.position.unsqueeze(-1)).squeeze(-1) + translation,
    (rotation @ state.velocity.unsqueeze(-1)).squeeze(-1),
  )
  config = EITInputConfig(group="C")
  before = build_actor_features(
    config, None, PosteriorInput(state, covariance), reference
  )
  after = build_actor_features(
    config,
    None,
    PosteriorInput(moved, covariance),
    apply_alignment(reference, transform),
  )
  torch.testing.assert_close(before[0], after[0], atol=1e-12, rtol=1e-12)
  torch.testing.assert_close(before[1], after[1], atol=1e-12, rtol=1e-12)
  initial_ref = RpvState(
    reference.rotation[:, 0], reference.position[:, 0], reference.velocity[:, 0]
  )
  alignment = create_episode_alignment(state, initial_ref)
  aligned = apply_alignment(reference, alignment)
  torch.testing.assert_close(aligned.position[:, 0, :2], state.position[:, :2])
  torch.testing.assert_close(aligned.position[..., 2], reference.position[..., 2])
  torch.testing.assert_close(
    aligned.velocity,
    (alignment.rotation[:, None] @ reference.velocity.unsqueeze(-1)).squeeze(-1),
  )
  # Applying the fixed transform again gives the same results and never mutates source.
  torch.testing.assert_close(
    apply_alignment(reference, alignment).position, aligned.position, atol=0, rtol=0
  )


def test_encoding_extreme_variance_correlation_and_invalid_covariance():
  _, _, covariance = _inputs(dtype=torch.float32)
  encoded = encode_covariance(covariance)
  assert encoded.shape == (2, 45)
  torch.testing.assert_close(
    encoded[:, :9], covariance.diagonal(dim1=-2, dim2=-1).sqrt().log()
  )
  assert torch.isfinite(encode_covariance(torch.zeros_like(covariance))).all()
  strong = torch.ones(2, 9, 9) + torch.eye(9) * 1e-5
  assert torch.isfinite(encode_covariance(strong)).all()
  for bad in (
    covariance * float("nan"),
    -torch.eye(9),
    torch.eye(9).index_fill(0, torch.tensor([0]), 10.0),
  ):
    with pytest.raises(ValueError):
      encode_covariance(bad)
  full = torch.arange(12 * 12, dtype=torch.float32).reshape(12, 12)
  indices = (4, 5, 6, 0, 1, 2, 8, 9, 10)
  marginal = extract_rpv_covariance(full, indices)
  torch.testing.assert_close(marginal, full[list(indices)][:, list(indices)])


def test_gaussian_noise_distribution_and_known_action_channels():
  config = GaussianMeasurementConfig()
  generator = torch.Generator().manual_seed(93)
  clean = torch.zeros(50000, 93)
  clean[:, 64:] = torch.linspace(-2, 2, 29)
  noisy = sample_proprio(clean, config, generator=generator)
  torch.testing.assert_close(noisy[:, 64:], clean[:, 64:], atol=0, rtol=0)
  std = config.standard_deviations(29)
  assert torch.all(noisy[:, :64].mean(0).abs() < std * 0.025)
  torch.testing.assert_close(noisy[:, :64].var(0), std.square(), rtol=0.025, atol=1e-9)
  torch.testing.assert_close(config.covariance(29).diagonal(), std.square())
  # Additive Gaussian gravity is intentionally not normalized back to the sphere.
  assert (noisy[:, :3].norm(dim=-1) - 1).abs().mean() > 0.9


def test_geometry_preserves_mean_and_covariance_gradients():
  state, reference, covariance = _inputs()
  state.position.requires_grad_()
  state.velocity.requires_grad_()
  covariance.requires_grad_()
  goal, uncertainty = build_actor_features(
    EITInputConfig(group="C"), None, PosteriorInput(state, covariance), reference
  )
  loss = goal.square().mean() + uncertainty.square().mean()
  loss.backward()
  for value in (state.position, state.velocity, covariance):
    assert value.grad is not None and torch.isfinite(value.grad).all()
    assert value.grad.norm() > 0


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_zero_covariance_encoding_has_finite_gradients(dtype):
  covariance = torch.zeros(2, 9, 9, dtype=dtype, requires_grad=True)
  features = encode_covariance(covariance)
  features.sum().backward()
  assert torch.isfinite(features).all()
  assert torch.isfinite(covariance.grad).all()
  # A materially negative variance cannot be hidden by the standard-deviation
  # floor; the PSD validation runs before the feature clamp.
  invalid = torch.eye(9, dtype=dtype)
  invalid[0, 0] = -1e-4
  with pytest.raises(ValueError, match="positive semidefinite"):
    encode_covariance(invalid)
