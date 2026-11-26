"""Functional imagination parity, graph ownership and update rejection boundaries."""

from copy import deepcopy
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from supermimic.belief import BeliefState
from supermimic.estimator_inputs import EpisodeAlignment
from supermimic.imagined_rollout import imagine_step, observe, prepare_rollout
from supermimic.learners.eit_policy_graph import _map_inputs
from tests.test_estimator_belief_filter import DEVICES, on_device
from tests.test_estimator_belief_integration import setup_env
from tests.test_estimator_belief_policy import real_setup


@pytest.mark.parametrize("device", DEVICES)
def test_filter_kernel_preserves_values_first_and_second_derivatives(device):
  estimator, belief, action, context = on_device(2, device)
  lower = belief.covariance_sqrt.detach().requires_grad_()
  action.requires_grad_()
  mean = replace(belief.mean, velocity=belief.mean.velocity.detach().requires_grad_())
  belief = BeliefState(mean, covariance_sqrt=lower)
  measurement = estimator.observation(mean) + 0.01
  std = estimator.measurement_config.standard_deviations(
    29, device=lower.device, dtype=lower.dtype
  )
  inputs = (action, lower, mean.velocity, *estimator.dynamics.parameters())
  baseline = estimator.predict_correct(belief, action, context, measurement)
  next_mean, factor, invalid = estimator.predict_correct_kernel(
    mean, lower, action, context, measurement, measurement_std=std
  )
  assert not invalid.any()
  actual = BeliefState(next_mean, covariance_sqrt=factor)
  torch.testing.assert_close(actual.mean.to_full(), baseline.mean.to_full())
  torch.testing.assert_close(actual.covariance_sqrt, baseline.covariance_sqrt)
  records = []
  for result in (baseline, actual):
    loss = result.mean.velocity.sin().sum() + result.covariance_sqrt.square().sum()
    first = torch.autograd.grad(loss, inputs, create_graph=True, retain_graph=True)
    second = torch.autograd.grad(
      sum(x.square().sum() for x in first), inputs, retain_graph=True
    )
    records.append((first, second))
  torch.testing.assert_close(records[0], records[1])


def inputs_for(learner):
  env = learner.wm_env
  starts = env.prepare_policy_starts(
    torch.tensor([0, 2, 0]), torch.zeros(3, dtype=torch.long)
  )
  # Different clips, repeated posterior keys and a held terminal reference frame.
  env.motion.motion_ends = torch.tensor([2, len(env.motion.joint_pos)])
  starts = replace(starts, motion_digest=env._policy_motion_digest())
  starts.batch["motion_time"] = torch.tensor([0, 2, len(env.motion.joint_pos) - 1])
  noises = tuple(noise[:, :, 0] for noise in learner.draw_noises(starts))
  return starts, noises


@pytest.mark.parametrize("group", ["C", "belief"])
def test_prepare_rollout_views_match_per_frame_without_extra_scalar_reads(group):
  env, _, _ = setup_env(group)
  env.motion.motion_ends = torch.tensor([2, 10])
  starts = env.prepare_policy_starts(torch.arange(3), torch.zeros(3, dtype=torch.long))
  starts.batch["alignment_translation"][:, 0] += torch.arange(3)
  alignment = EpisodeAlignment(
    starts.batch["alignment_rotation"], starts.batch["alignment_translation"]
  )
  reads = []
  for horizon in (1, 8):
    with torch.profiler.profile(
      activities=[torch.profiler.ProfilerActivity.CPU]
    ) as prof:
      prepared = prepare_rollout(
        env, starts, horizon, torch.zeros(horizon, 3, 70), torch.zeros(horizon, 3, 64)
      )
    reads.append(
      sum(e.count for e in prof.key_averages() if e.key == "aten::_local_scalar_dense")
    )
    reference_tensors = []
    _map_inputs(
      (prepared.reference, prepared.bodies, prepared.motion_times),
      reference_tensors.append,
      describe=True,
    )
    assert len(reference_tensors) == (13 if group == "belief" else 10)
    for step in range(horizon + 1):
      current = torch.minimum(starts.motion_times + step, torch.tensor([1, 9, 9]))
      torch.testing.assert_close(prepared.motion_times[step], current)
      expected = env.representation.reference_window(env.motion, current, alignment)
      actual = prepared.reference_at(step)
      for name in ("joint_pos", "joint_vel", "rotation", "position", "velocity"):
        torch.testing.assert_close(getattr(actual, name), getattr(expected, name))
      if expected.local is not None:
        torch.testing.assert_close(vars(actual.local), vars(expected.local))
      torch.testing.assert_close(
        prepared.bodies_at(step),
        env._aligned_reference_bodies(env.motion.body_state(current), alignment),
      )
  assert reads[0] == reads[1]


@pytest.mark.parametrize(
  ("field", "message"),
  [
    ("local", "finite values"),
    ("rotation", "positive determinant"),
    ("body", "bodies must be finite"),
  ],
)
def test_prepare_rollout_rejects_invalid_bootstrap_reference(field, message):
  env, _, _ = setup_env("belief")
  starts = env.prepare_policy_starts(torch.tensor([0]), torch.tensor([0]))
  starts.batch["motion_time"].zero_()
  build = env.representation.reference_window

  def corrupt_reference(motion, times, alignment):
    ref = build(motion, times, alignment)
    if field == "local":
      ref.local.goal[times == 2] = float("nan")
    if field == "rotation":
      ref.rotation[times == 2] *= -1
    return ref

  if field == "body":
    env.motion.body_ang_vel_w[2, 0, 0] = float("inf")
    starts = replace(starts, motion_digest=env._policy_motion_digest())
  with (
    patch.object(env.representation, "reference_window", side_effect=corrupt_reference),
    pytest.raises(ValueError, match=message),
  ):
    prepare_rollout(env, starts, 2, torch.zeros(2, 1, 70), torch.zeros(2, 1, 64))


@pytest.mark.parametrize("group", ["A", "B", "C"])
@pytest.mark.parametrize("backend", ["vmap", "increment"])
def test_every_frame_and_actor_gradient_match_transactional_rollout(group, backend):
  learner = real_setup(group)
  env = learner.wm_env
  estimator = env.posterior_provider.estimator
  estimator.config = replace(estimator.config, jacobian_backend=backend)
  env.dynamics.requires_grad_(False)
  learner.critic.requires_grad_(False)
  with torch.no_grad():
    learner.critic.mlp[-1].weight.normal_(std=0.01)
    if learner.actor.film is not None:
      learner.actor.film[-1].weight.normal_(std=0.01)
  starts, noises = inputs_for(learner)
  prepared = prepare_rollout(env, starts, 2, *noises)
  state = prepared.state
  obs, invalid = observe(env, prepared, state, 0)
  original = env.reset_prepared(starts)
  torch.testing.assert_close(dict(obs.items()), dict(original.items()))
  actions = torch.randn(2, len(starts), 29, requires_grad=True) * 0.1
  losses = [0, 0]
  for t in range(2):
    old, reward, done, _ = env.step(
      actions[t], process_noise=noises[0][t], measurement_noise=noises[1][t]
    )
    state, actual_reward, actual_done, failed = imagine_step(
      env, prepared, state, actions[t], noises[0][t], noises[1][t], t
    )
    actual, bad = observe(env, prepared, state, t + 1)
    assert not (invalid | failed | bad).any()
    torch.testing.assert_close(
      state.belief.mean.to_full(), env.posterior_provider.state.belief.mean.to_full()
    )
    torch.testing.assert_close(
      state.belief.covariance_sqrt, env.posterior_provider.state.belief.covariance_sqrt
    )
    torch.testing.assert_close(dict(actual.items()), dict(old.items()))
    torch.testing.assert_close((actual_reward, actual_done), (reward, done))
    losses[0] = losses[0] + reward.sum() + old["critic"].square().mean()
    losses[1] = losses[1] + actual_reward.sum() + actual["critic"].square().mean()
  torch.testing.assert_close(
    torch.autograd.grad(losses[0], actions)[0],
    torch.autograd.grad(losses[1], actions)[0],
  )
  result = learner._rollout_result(starts, *noises, 2)
  expected = learner._rollout_kernel(None, *noises, 2, starts=starts)
  for name in (
    "returns",
    "critic_data",
    "metrics",
    "auxiliary_loss",
    "reward_return",
    "bootstrap",
  ):
    torch.testing.assert_close(getattr(result, name), getattr(expected, name))
  parameters = tuple(learner.actor.parameters())
  torch.testing.assert_close(
    torch.autograd.grad(result.returns.sum(), parameters),
    torch.autograd.grad(expected.returns.sum(), parameters),
    atol=2e-6,
    rtol=1e-4,
  )


def test_training_never_touches_provider_or_actor_cached_graph():
  learner = real_setup()
  starts, noises = inputs_for(learner)
  env = learner.wm_env
  env.reset_prepared(starts)
  provider = env.posterior_provider
  state, generation = provider.state, provider.generation
  frame, latent = env._belief_frame, env.rep_state
  aux = learner.actor._aux_loss
  from contextlib import ExitStack

  with ExitStack() as stack:
    for owner, names in (
      (provider, ("prepare", "prepare_restore", "lookup", "commit", "clear")),
      (env, ("step", "reset_prepared", "clear_rollout")),
    ):
      for name in names:
        stack.enter_context(patch.object(owner, name, side_effect=AssertionError(name)))
    result = learner._rollout_result(starts, *noises, 2)
    assert result.final_state.belief.covariance_sqrt.grad_fn is not None
    learner.update_from(starts, *learner.draw_noises(starts))
  assert provider.state is state and provider.generation == generation
  assert env._belief_frame is frame and env.rep_state is latent
  assert learner.actor._aux_loss is aux


@pytest.mark.parametrize("failure", ["action", "reward", "model", "motion"])
def test_mid_rollout_failure_rejects_whole_update(failure, monkeypatch):
  learner = real_setup()
  starts, _ = inputs_for(learner)
  noises = learner.draw_noises(starts)
  before = deepcopy(
    (
      learner.actor.state_dict(),
      learner.critic.state_dict(),
      [x.state_dict() for x in (*learner.actor_optimizers, *learner.critic_optimizers)],
    )
  )
  env = learner.wm_env
  original = learner._actor_with_aux
  calls = 0

  def actor(obs):
    nonlocal calls
    calls += 1
    action, aux = original(obs)
    if calls == 3:  # after the first microbatch already accumulated gradients
      if failure == "action":
        action = action + float("inf")  # finite after clipping, but still invalid
      elif failure == "model":
        with torch.no_grad():
          next(env.dynamics.parameters()).add_(0.01)
      elif failure == "motion":
        env.motion.joint_pos.add_(0.01)
    return action, aux

  monkeypatch.setattr(learner, "_actor_with_aux", actor)
  if failure == "reward":
    step = env._wm_step_kernel

    def bad_reward(*args, **kwargs):
      *values, reward = step(*args, **kwargs)
      return (*values, reward * (float("nan") if calls == 3 else 1))

    monkeypatch.setattr(env, "_wm_step_kernel", bad_reward)
  with pytest.raises((FloatingPointError, ValueError)):
    learner.update_from(starts, *noises)
  after = (
    learner.actor.state_dict(),
    learner.critic.state_dict(),
    [x.state_dict() for x in (*learner.actor_optimizers, *learner.critic_optimizers)],
  )
  before_leaves, before_spec = torch.utils._pytree.tree_flatten(before)
  after_leaves, after_spec = torch.utils._pytree.tree_flatten(after)
  assert before_spec == after_spec
  for actual, expected in zip(after_leaves, before_leaves, strict=True):
    if isinstance(actual, torch.Tensor):
      torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    else:
      assert actual == expected
  assert learner.actor_steps == learner.critic_steps == 0
  assert all(
    p.grad is None for p in (*learner.actor.parameters(), *learner.critic.parameters())
  )


@pytest.mark.parametrize("group", ["A", "B", "C"])
def test_single_sample_uses_mean_with_observation_noise_and_actor_gradients(group):
  learner = real_setup(group)
  learner.cfg = replace(learner.cfg, num_samples=1)
  starts, (process, measurement) = inputs_for(learner)
  env = learner.wm_env
  env.dynamics.requires_grad_(False)
  learner.critic.requires_grad_(False)
  with (
    patch.object(env.dynamics, "sample", side_effect=AssertionError("process sample")),
    patch.object(env.dynamics, "forward", wraps=env.dynamics.forward) as mean,
    patch.object(
      env.dynamics, "noise_factor", wraps=env.dynamics.noise_factor
    ) as covariance,
  ):
    expected = learner._rollout_result(starts, process, measurement, 2)
    assert mean.call_count == 2
    assert covariance.call_count > 0  # The filter still propagates process covariance.
    actual = learner._rollout_result(starts, torch.ones_like(process), measurement, 2)
    torch.testing.assert_close(actual.final_state.rep, expected.final_state.rep)
    torch.testing.assert_close(actual.final_state.aux, expected.final_state.aux)
    torch.testing.assert_close(actual.returns, expected.returns)
    parameters = tuple(learner.actor.parameters())
    expected_grad = torch.autograd.grad(expected.returns.sum(), parameters)
    actual_grad = torch.autograd.grad(actual.returns.sum(), parameters)
    torch.testing.assert_close(actual_grad, expected_grad)
    assert sum(g.square().sum() for g in actual_grad) > 0
    without_measurement = learner._rollout_result(
      starts, process, torch.zeros_like(measurement), 2
    )
    assert not torch.equal(
      actual.final_state.history, without_measurement.final_state.history
    )
    learner.update_from(starts, *learner.draw_noises(starts))


def test_multiple_samples_keep_stochastic_dynamics():
  learner = real_setup()
  starts, (process, measurement) = inputs_for(learner)
  dynamics = learner.wm_env.dynamics
  with patch.object(dynamics, "sample", wraps=dynamics.sample) as sample:
    actual = learner._rollout_result(starts, process, measurement, 2)
    assert sample.call_count == 2
    noiseless = learner._rollout_result(
      starts, torch.zeros_like(process), measurement, 2
    )
  assert not torch.equal(actual.final_state.rep, noiseless.final_state.rep)
