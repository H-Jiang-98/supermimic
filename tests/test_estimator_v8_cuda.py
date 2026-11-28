"""v8 fixed-noise CUDA gradients, host-read boundaries and motion batching."""

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from supermimic.imagined_rollout import prepare_rollout
from tests import test_estimator_eit_graph as graph_tests


@pytest.fixture
def graph_policy(tmp_path):
  yield from graph_tests.graph_policy.__wrapped__(tmp_path)


@pytest.mark.parametrize("group", ["gt", "mean", "belief"])
def test_v8_cuda_rollout_gradients_and_batched_encoder(graph_policy, group):
  policy, starts = graph_policy(group, compiled=True, samples=1, policy_version="v8")
  policy.cfg = replace(policy.cfg, horizon=8)
  policy.wm_env.dynamics.requires_grad_(False)
  policy.critic.requires_grad_(False)
  if group == "belief":
    with torch.no_grad():
      policy.actor.film[-1].weight.normal_(std=0.01)
  noises = tuple(noise[:, :, 0] for noise in policy.draw_noises(starts))
  prepared = prepare_rollout(policy.wm_env, starts, policy.cfg.horizon, *noises)
  parameters = tuple(policy.actor.parameters())

  def run():
    result = policy._rollout_kernel(prepared, *noises[:2], policy.cfg.horizon)
    gradients = torch.autograd.grad(result.returns.sum(), parameters, allow_unused=True)
    state = result.final_state
    return (
      result.returns.detach(),
      state.history.detach(),
      state.belief.mean.to_full().detach() if state.belief is not None else None,
      state.belief.covariance_sqrt.detach() if state.belief is not None else None,
      tuple(g.detach() if g is not None else None for g in gradients),
      result.invalid,
    )

  with torch.compiler.set_stance("force_eager"):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as p:
      eager = run()
  keys = {event.key for event in p.key_averages()}
  assert (
    not {
      "aten::item",
      "aten::_local_scalar_dense",
      "aten::nonzero",
      "aten::_linalg_check_errors",
    }
    & keys
  )
  with patch.object(policy, "_encode", wraps=policy._encode) as encoder:
    compiled = run()
  assert encoder.call_count == 1
  assert not compiled[-1].any()
  torch.testing.assert_close(compiled, eager, atol=2e-5, rtol=1e-3)
  policy.cfg = replace(policy.cfg, batch_motion_encoder=False)
  with patch.object(policy, "_encode", wraps=policy._encode) as encoder:
    stepwise = run()
  assert encoder.call_count == policy.cfg.horizon
  torch.testing.assert_close(stepwise, compiled, atol=2e-5, rtol=1e-3)
