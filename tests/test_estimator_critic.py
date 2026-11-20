"""RPV-belief permissions, geometry, gradients and atomic critic restoration."""

from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from supermimic.belief import BeliefState
from supermimic.critic_inputs import (
  CRITIC_INPUT_VERSION,
  BeliefCritic,
  CriticInputConfig,
  build_critic_features,
  encode_rpv_covariance,
)
from supermimic.utils.lie_state import boxplus, exp, so3_exp


def inputs(dtype=torch.float64, device="cpu"):
  generator = torch.Generator(device=device).manual_seed(85)

  def rand(*shape):
    return torch.randn(shape, generator=generator, dtype=dtype, device=device)

  truth = exp(rand(3, 70) * 0.2)
  mean = boxplus(truth, rand(3, 70) * 0.1)
  factor = rand(3, 70, 70) * 0.005 + torch.eye(70, dtype=dtype, device=device) * 0.1
  belief = BeliefState(mean, factor @ factor.mT)
  return truth, rand(3, 700), rand(3, 5, 93), rand(3, 4), rand(3).abs(), belief


def test_groups_use_rpv_belief_and_a_never_reads_it():
  args = inputs()
  a = build_critic_features(*args, group="A")
  torch.testing.assert_close(a, build_critic_features(*args[:-1], group="A"))
  torch.testing.assert_close(a, build_critic_features(*args[:-1], object(), group="A"))
  b = build_critic_features(*args, group="B")
  c = build_critic_features(*args, group="C")
  assert a.shape == (3, 1238)
  assert b.shape == c.shape == (3, 1295)
  torch.testing.assert_close(a, b[:, :1238], atol=0, rtol=0)
  torch.testing.assert_close(b, c, atol=0, rtol=0)
  with pytest.raises(ValueError, match="complete BeliefState"):
    build_critic_features(*args[:-1], group="B")
  changed = deepcopy(args[-1])
  changed.mean.position[:, 1] += 0.3
  changed.covariance[:, 1, 7] += 0.0001
  changed.covariance[:, 7, 1] += 0.0001
  torch.testing.assert_close(build_critic_features(*args[:-1], changed, group="A"), a)
  different = build_critic_features(*args[:-1], changed, group="B")
  assert (different[:, 1238:1250] != b[:, 1238:1250]).any()
  assert (different[:, 1250:] != b[:, 1250:]).any()


@pytest.mark.parametrize("group", ["B", "C"])
@pytest.mark.parametrize("component", ["mean", "covariance", "cross_covariance"])
def test_rpv_critic_ignores_other_belief_coordinates(group, component):
  args = inputs()
  expected = build_critic_features(*args, group=group)
  changed = args[-1].clone()
  if component == "mean":
    changed.mean.omega.add_(0.3)
    changed.mean.joint_pos.add_(0.2)
    changed.mean.joint_vel.sub_(0.4)
  elif component == "covariance":
    changed.covariance[:, 9:, 9:] += torch.eye(61, dtype=torch.float64) * 0.01
  else:
    for row, col in ((1, 10), (5, 42), (7, 65)):
      changed.covariance[:, row, col] += 0.0001
      changed.covariance[:, col, row] += 0.0001
  # Legal changes outside the marginal cannot enter the current critic, including
  # through a Schur complement. They may still affect future filter updates.
  actual = build_critic_features(*args[:-1], changed, group=group)
  torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_rpv_mean_encoding_recovers_rotation_position_and_velocity_with_scales():
  truth, goal, history, ctx, remaining, belief = inputs()
  cfg = CriticInputConfig(mean_scales=(2.0, 3.0))
  features = build_critic_features(
    truth, goal, history, ctx, remaining, belief, config=cfg
  )
  mean = features[:, 1238:1250]
  columns = mean[:, :6].reshape(3, 3, 2)
  relative_rotation = torch.cat(
    (columns, torch.linalg.cross(columns[..., 0], columns[..., 1]).unsqueeze(-1)),
    dim=-1,
  )
  torch.testing.assert_close(truth.rotation @ relative_rotation, belief.mean.rotation)
  position = truth.position + (
    truth.rotation @ (mean[:, 6:9] * 2).unsqueeze(-1)
  ).squeeze(-1)
  velocity = truth.velocity + (
    truth.rotation @ (mean[:, 9:12] * 3).unsqueeze(-1)
  ).squeeze(-1)
  torch.testing.assert_close(position, belief.mean.position)
  torch.testing.assert_close(velocity, belief.mean.velocity)
  torch.testing.assert_close(
    features[:, 1250:], encode_rpv_covariance(belief.covariance[:, :9, :9], cfg)
  )


def test_horizontal_transform_preserves_all_features_and_near_pi_is_finite():
  truth, goal, history, ctx, remaining, belief = inputs()
  rotation = so3_exp(torch.tensor([0.0, 0.0, 1.4], dtype=torch.float64))
  translation = torch.tensor([4.1, -3.2, 0.0], dtype=torch.float64)

  def transform(state):
    return replace(
      state,
      rotation=rotation @ state.rotation,
      position=(rotation @ state.position.unsqueeze(-1)).squeeze(-1) + translation,
      velocity=(rotation @ state.velocity.unsqueeze(-1)).squeeze(-1),
    )

  for group in ("A", "B", "C"):
    expected = build_critic_features(
      truth, goal, history, ctx, remaining, belief, group=group
    )
    actual = build_critic_features(
      transform(truth),
      goal,
      history,
      ctx,
      remaining,
      BeliefState(transform(belief.mean), belief.covariance),
      group=group,
    )
    torch.testing.assert_close(actual, expected, atol=1e-13, rtol=1e-13)
  turn = so3_exp(torch.tensor([torch.pi - 1e-9, 0.0, 0.0], dtype=torch.float64))
  almost_pi = BeliefState(
    replace(belief.mean, rotation=truth.rotation @ turn), belief.covariance
  )
  assert torch.isfinite(
    build_critic_features(truth, goal, history, ctx, remaining, almost_pi)
  ).all()


def test_covariance_encoding_roundtrip_rpv_matrix_and_physical_scales():
  covariance = inputs()[-1].covariance[:, :9, :9]
  cfg = CriticInputConfig(covariance_scales=(2.0, 3.0, 4.0))
  encoded = encode_rpv_covariance(covariance, cfg)
  assert encoded.shape == (3, 45)
  scale = torch.tensor([2.0] * 3 + [3.0] * 3 + [4.0] * 3)
  std = encoded[:, :9].exp() * scale
  correlation = torch.eye(9, dtype=torch.float64).expand(3, 9, 9).clone()
  rows, cols = torch.triu_indices(9, 9, offset=1)
  correlation[:, rows, cols] = encoded[:, 9:]
  correlation[:, cols, rows] = encoded[:, 9:]
  reconstructed = correlation * std.unsqueeze(-1) * std.unsqueeze(-2)
  torch.testing.assert_close(reconstructed, covariance, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize(
  "kind", ["nan", "negative", "asymmetric", "indefinite", "wrong_shape"]
)
def test_invalid_covariance_is_rejected(kind):
  covariance = torch.eye(9, dtype=torch.float64)
  if kind == "nan":
    covariance[8, 8] = float("nan")
  elif kind == "negative":
    covariance[8, 8] = -1
  elif kind == "asymmetric":
    covariance[2, 7] = 0.1
  elif kind == "indefinite":
    covariance[2, 7] = covariance[7, 2] = 2
  else:
    covariance = torch.eye(70, dtype=torch.float64)
  with pytest.raises(ValueError):
    encode_rpv_covariance(covariance)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_zero_covariance_encoding_has_finite_values_and_gradients(dtype):
  covariance = torch.zeros(2, 9, 9, dtype=dtype, requires_grad=True)
  features = encode_rpv_covariance(covariance)
  assert torch.isfinite(features).all()
  gradient = torch.autograd.grad(features.sum(), covariance)[0]
  assert torch.isfinite(gradient).all()


@pytest.mark.parametrize("group", ["B", "C"])
@pytest.mark.parametrize("component", ["mean", "covariance"])
@pytest.mark.parametrize(
  "device",
  [
    "cpu",
    pytest.param(
      "cuda:0",
      marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA unavailable"
      ),
    ),
  ],
)
def test_critic_frozen_parameters_preserve_mean_and_covariance_directional_derivatives(
  device,
  group,
  component,
):
  truth, goal, history, ctx, remaining, belief = inputs(device=device)
  parameter = torch.tensor(0.02, dtype=torch.float64, device=device, requires_grad=True)
  direction = torch.linspace(-0.4, 0.5, 70, dtype=torch.float64, device=device).expand(
    3, 70
  )
  covariance_direction = torch.eye(70, dtype=torch.float64, device=device) * 0.001
  covariance_direction[1, 7] = covariance_direction[7, 1] = 0.002
  critic = BeliefCritic(group, 4, (16, 8)).to(dtype=torch.float64, device=device)
  with torch.no_grad():
    critic.mlp[-1].weight.fill_(0.2)
  critic.requires_grad_(False)

  def value(x):
    posterior = BeliefState(
      boxplus(belief.mean, x * direction) if component == "mean" else belief.mean,
      belief.covariance + x * covariance_direction
      if component == "covariance"
      else belief.covariance,
    )
    return critic(
      build_critic_features(
        truth, goal, history, ctx, remaining, posterior, group=group
      )
    ).sum()

  derivative = torch.autograd.grad(value(parameter), parameter)[0]
  h = 1e-5
  numerical = (value(parameter.detach() + h) - value(parameter.detach() - h)) / (2 * h)
  assert derivative.abs() > 1e-8
  torch.testing.assert_close(derivative, numerical, rtol=3e-6, atol=1e-8)
  assert all(p.grad is None for p in critic.parameters())


@pytest.mark.parametrize("group", ["B", "C"])
def test_rpv_projection_has_no_direct_gradient_to_other_belief_coordinates(group):
  args = inputs()
  belief = args[-1]
  tangent = torch.zeros(3, 70, dtype=torch.float64, requires_grad=True)
  covariance = belief.covariance.clone().requires_grad_()
  posterior = BeliefState(boxplus(belief.mean, tangent), covariance)
  features = build_critic_features(*args[:-1], posterior, group=group)
  mean_grad, covariance_grad = torch.autograd.grad(
    features[:, 1238:].square().sum(), (tangent, covariance)
  )
  assert torch.isfinite(mean_grad).all() and torch.isfinite(covariance_grad).all()
  assert torch.count_nonzero(mean_grad[:, :9]) > 0
  assert torch.count_nonzero(covariance_grad[:, :9, :9]) > 0
  assert torch.count_nonzero(mean_grad[:, 9:]) == 0
  assert torch.count_nonzero(covariance_grad[:, 9:, :]) == 0
  assert torch.count_nonzero(covariance_grad[:, :, 9:]) == 0


def test_initialization_copies_bc_exactly_and_a_common_columns_without_aliases():
  b, c, a = [BeliefCritic(group, 4, (16, 8)).double() for group in ("B", "C", "A")]
  features = build_critic_features(*inputs())
  b.update_normalization(features)
  c.copy_common_initialization_from(b)
  a.copy_common_initialization_from(b)
  for name, value in b.state_dict().items():
    assert torch.equal(c.state_dict()[name], value)
    assert c.state_dict()[name].data_ptr() != value.data_ptr()
  torch.testing.assert_close(a.mlp[0].weight, b.mlp[0].weight[:, : a.input_dim])
  for critic in (a, b, c):
    assert torch.count_nonzero(critic(features[:, : critic.input_dim])) == 0
  with torch.no_grad():
    b.mlp[-1].weight.fill_(0.1)
  c.copy_common_initialization_from(b)
  torch.testing.assert_close(c(features), b(features), atol=0, rtol=0)


def test_normalization_uses_only_newest_history_and_is_atomic():
  critic = BeliefCritic("B", 4, (16, 8))
  features = build_critic_features(*inputs(dtype=torch.float32))
  critic.update_normalization(features)
  assert critic.history_normalizer.count == 3
  torch.testing.assert_close(
    critic.history_normalizer.mean, features[:, 1140:1233].mean(0)
  )
  state = deepcopy(critic.state_dict())
  critic.train()
  critic(features)
  for name, value in state.items():
    torch.testing.assert_close(critic.state_dict()[name], value, atol=0, rtol=0)
  bad = features.clone()
  bad[0, 70], bad[1, 70] = 1e38, -1e38
  with pytest.raises(ValueError, match="non-finite statistics"):
    critic.update_normalization(bad)
  for name, value in state.items():
    torch.testing.assert_close(critic.state_dict()[name], value, atol=0, rtol=0)


@pytest.mark.parametrize(
  "fault",
  [
    "contract",
    "missing",
    "shape",
    "nan",
    "negative_std",
    "inconsistent_std",
  ],
)
def test_invalid_checkpoint_does_not_partially_modify_critic(fault):
  source = BeliefCritic("B", 4, (16, 8))
  source.update_normalization(build_critic_features(*inputs(dtype=torch.float32)))
  destination = BeliefCritic("B", 4, (16, 8))
  before = deepcopy(destination.state_dict())
  checkpoint = source.checkpoint()
  if fault == "contract":
    checkpoint["contract"]["config"]["group"] = "C"
  elif fault == "missing":
    checkpoint["state_dict"].pop("mlp.4.weight")
  elif fault == "shape":
    checkpoint["state_dict"]["mlp.4.weight"] = torch.zeros(1)
  elif fault == "nan":
    checkpoint["state_dict"]["mlp.4.weight"].fill_(float("nan"))
  elif fault == "negative_std":
    checkpoint["state_dict"]["goal_normalizer._std"].fill_(-1)
  else:
    checkpoint["state_dict"]["goal_normalizer._std"].fill_(15)
  with pytest.raises(ValueError):
    destination.load_checkpoint(checkpoint)
  for key, value in before.items():
    torch.testing.assert_close(destination.state_dict()[key], value, atol=0, rtol=0)
  destination.load_checkpoint(source.checkpoint())
  for key, value in source.state_dict().items():
    torch.testing.assert_close(destination.state_dict()[key], value, atol=0, rtol=0)


@pytest.mark.parametrize("group", ["A", "B", "C"])
def test_all_groups_reject_full_belief_critic_contract_before_loading_weights(group):
  critic = BeliefCritic(group, 4, (16, 8))
  checkpoint = critic.checkpoint()
  # A's old weights even have the same shape. The contract, including its
  # version, is checked before applying any compatible tensor.
  checkpoint["contract"].update(
    version="eit_full_belief_critic_v1", input_dim=1238 if group == "A" else 3796
  )
  checkpoint["contract"]["config"].update(
    version="eit_full_belief_critic_v1",
    covariance_encoding="full70_log_std_corr_v1",
    mean_scales=(1.0,) * 5,
    covariance_scales=(1.0,) * 6,
  )
  checkpoint["state_dict"]["mlp.0.bias"].add_(1)
  before = deepcopy(critic.state_dict())
  with pytest.raises(
    ValueError,
    match=f"Group {group}.*{CRITIC_INPUT_VERSION}.*eit_full_belief_critic_v1",
  ):
    critic.load_checkpoint(checkpoint)
  for key, value in before.items():
    torch.testing.assert_close(critic.state_dict()[key], value, atol=0, rtol=0)


@pytest.mark.parametrize(
  "kwargs",
  [
    {"group": "D"},
    {"epsilon_std": 0},
    {"mean_scales": (1,)},
    {"mean_scales": (1,) * 5},
    {"mean_scales": (1, float("nan"))},
    {"covariance_scales": (1, 1, -1)},
    {"covariance_scales": (1,) * 6},
    {"version": "old"},
    {"version": "eit_full_belief_critic_v1"},
    {"covariance_encoding": "full70_log_std_corr_v1"},
    {"remaining_time_scale": float("inf")},
  ],
)
def test_config_rejects_invalid_contract(kwargs):
  with pytest.raises(ValueError):
    CriticInputConfig(**kwargs)
