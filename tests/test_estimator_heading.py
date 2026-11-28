"""Sparse heading fusion geometry and online transaction regressions."""

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch
from torch.func import jacrev, vmap

from supermimic.belief import BeliefState
from supermimic.utils.lie_state import boxplus, local_error, reset_jacobian, so3_exp
from tests.test_estimator_belief_filter import _fixture
from tests.test_estimator_belief_provider import assert_same_belief, setup_provider


def test_heading_jacobian_matches_right_local_autodiff():
  estimator, belief, _, _ = _fixture(batch=3, dtype=torch.float64)

  def observation(error, rotation):
    rotated = rotation @ so3_exp(error)
    return torch.atan2(rotated[1, 0], rotated[0, 0])

  expected = vmap(jacrev(observation))(
    torch.zeros(3, 3, dtype=torch.float64), belief.mean.rotation
  )
  jacobian = estimator.heading_jacobian(belief.mean)
  torch.testing.assert_close(jacobian[:, 0, :3], expected)
  assert torch.count_nonzero(jacobian[:, :, 3:]) == 0


def test_heading_correction_matches_dense_joseph_update_across_angle_wrap():
  estimator, belief, _, _ = _fixture(batch=2, dtype=torch.float64)
  angles = torch.tensor([[0.0, 0.0, 3.13], [0.0, 0.0, -3.13]], dtype=torch.float64)
  belief = BeliefState(
    replace(belief.mean, rotation=so3_exp(angles)),
    covariance_sqrt=belief.covariance_sqrt,
  )
  heading = torch.tensor([-3.12, 3.12], dtype=torch.float64)
  std = 0.03
  h, covariance = estimator.heading_jacobian(belief.mean), belief.covariance
  delta = heading - estimator.heading_observation(belief.mean)
  innovation = torch.atan2(delta.sin(), delta.cos())
  gain = torch.linalg.solve(h @ covariance @ h.mT + std**2, h @ covariance).mT
  correction = (gain * innovation[:, None, None]).squeeze(-1)
  reset = reset_jacobian(correction)
  residual = torch.eye(70, dtype=torch.float64) - gain @ h
  expected_covariance = (
    reset @ (residual @ covariance @ residual.mT + (gain @ gain.mT) * std**2) @ reset.mT
  )
  result = estimator.correct_heading(belief, heading, std)
  torch.testing.assert_close(
    local_error(boxplus(belief.mean, correction), result.mean),
    torch.zeros(2, 70, dtype=torch.float64),
    atol=1e-12,
    rtol=0,
  )
  torch.testing.assert_close(result.covariance, expected_covariance)
  assert innovation.abs().max() < 0.04
  assert result.covariance[:, 5, 5].lt(covariance[:, 5, 5]).all()


def test_heading_packet_mask_and_prepare_retry_preserve_committed_state():
  provider, prior, measurement, action, context = setup_provider()
  mask = torch.tensor([True, False, True])
  values = torch.tensor([0.4, float("nan"), -0.3])
  args = dict(
    measurement=measurement,
    previous_action=action,
    reset_mask=torch.ones(3, dtype=torch.bool),
    context=context,
    heading=values,
    heading_mask=mask,
    heading_std=0.04,
  )
  first = provider.prepare(**args)
  retry = provider.prepare(**args)
  assert provider.state is None
  assert_same_belief(first.state.belief, retry.state.belief)
  proprio = provider.estimator.correct(prior, measurement)
  expected = provider.estimator.correct_heading(proprio.index(mask), values[mask], 0.04)
  assert_same_belief(first.state.belief.index(mask), expected)
  assert_same_belief(first.state.belief.index(~mask), proprio.index(~mask))
  provider.commit(first)
  saved = provider.state.belief.clone()
  with pytest.raises(ValueError):
    provider.commit(retry)
  assert_same_belief(provider.state.belief, saved)


def test_missing_heading_skips_update_and_partial_reset_fuses_current_packet_once():
  provider, prior, measurement, action, context = setup_provider()
  reset = torch.ones(3, dtype=torch.bool)
  first = provider.prepare(
    measurement=measurement, previous_action=action, reset_mask=reset, context=context
  )
  provider.commit(first)
  args = dict(
    measurement=measurement,
    previous_action=action,
    reset_mask=torch.tensor([False, True, False]),
    context=context,
  )
  baseline = provider.prepare(**args)
  with patch.object(provider.estimator, "correct_heading", side_effect=AssertionError):
    missing = provider.prepare(
      **args,
      heading=torch.full((3,), float("nan")),
      heading_mask=torch.zeros(3, dtype=torch.bool),
      heading_std=0.04,
    )
  for name in ("rotation", "position", "velocity", "omega", "joint_pos", "joint_vel"):
    assert torch.equal(
      getattr(missing.state.belief.mean, name),
      getattr(baseline.state.belief.mean, name),
    )
  assert torch.equal(
    missing.state.belief.covariance_sqrt, baseline.state.belief.covariance_sqrt
  )
  packet = provider.prepare(
    **args,
    heading=torch.tensor([float("nan"), 0.4, float("nan")]),
    heading_mask=torch.tensor([False, True, False]),
    heading_std=0.04,
  )
  expected = provider.estimator.correct_heading(
    provider.estimator.correct(prior.index([1]), measurement[[1]]),
    torch.tensor([0.4]),
    0.04,
  )
  assert_same_belief(packet.state.belief.index([1]), expected)
  assert_same_belief(
    packet.state.belief.index([0, 2]), baseline.state.belief.index([0, 2])
  )
  provider.commit(packet)
  assert torch.equal(provider.state.time_index, torch.tensor([1, 0, 1]))


@pytest.mark.parametrize("std", [0.0, -0.1, float("nan"), float("inf")])
def test_heading_rejects_invalid_noise_before_provider_commit(std):
  provider, _, measurement, action, context = setup_provider()
  with pytest.raises(ValueError, match="heading packet"):
    provider.prepare(
      measurement=measurement,
      previous_action=action,
      reset_mask=torch.ones(3, dtype=torch.bool),
      context=context,
      heading=torch.zeros(3),
      heading_mask=torch.ones(3, dtype=torch.bool),
      heading_std=std,
    )
  assert provider.state is None
