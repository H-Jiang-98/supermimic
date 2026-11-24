"""Transactional adapters and fixed-noise policy gradients through full belief."""

from dataclasses import asdict, fields, replace
from unittest.mock import Mock

import pytest
import torch

from supermimic.belief import BeliefState, GaussianStatePrior
from supermimic.belief_provider import BeliefProvider, FixedBeliefPrior
from supermimic.estimator import LieBeliefFilter
from supermimic.models import Dynamics, MultiEncoderAE, ProcessNoiseConfig
from supermimic.replay_buffer import FieldSpec, ReplayBuffer
from supermimic.runners.run_config import load_runner_cfg
from supermimic.runners.supermimic_runner import InputStagePolicy, build_actor
from supermimic.utils.lie_state import LieState, local_error
from tests.test_estimator_runner import _build_current, _PhysicalEnv, _runner
from tests.test_estimator_world_model_env import BATCH, HISTORY, _batch, _env


def setup_env(group="C"):
  torch.manual_seed(17)
  env = _env(group)
  env.dynamics = Dynamics(
    env.representation,
    29,
    1,
    torch.ones(29),
    torch.zeros(29),
    (12,),
    process_noise=ProcessNoiseConfig(hidden_dims=(8,)),
  ).eval()
  estimator = LieBeliefFilter(
    env.dynamics, env.estimator_input_config.measurement_noise, torch.zeros(29)
  )
  batch = _batch()
  mean = LieState.from_full(batch["state"])
  belief = BeliefState(mean, torch.eye(70).expand(BATCH, 70, 70).clone() * 0.005)
  sensor = torch.cat([estimator.observation(mean), batch["prev_action"]], -1)
  batch["sensor_window"] = sensor[:, None].expand(-1, HISTORY, -1).clone()
  batch["actor_obs"] = sensor
  env.posterior_provider = BeliefProvider(estimator, differentiable=True)
  record = batch.exclude("sensor_window", "pad_mask").clone()
  record["episode_id"] = torch.arange(BATCH)
  record["time_index"] = torch.zeros(BATCH, dtype=torch.long)
  record["action"] = batch["prev_action"]
  record["terminated"] = record["truncated"] = torch.zeros(BATCH, dtype=torch.bool)
  schema = {key: FieldSpec(tuple(v.shape[1:]), v.dtype) for key, v in record.items()}
  env.replay_buffer = ReplayBuffer(schema, 4, BATCH)
  env.replay_buffer.insert(record, head_priors=(record["episode_id"], belief))
  env.replay_buffer.finish_collection_round()
  return env, batch, belief


def test_reset_requires_full_versioned_posterior_without_refusing_measurement():
  env, batch, belief = setup_env()
  with pytest.raises(ValueError, match="requires episode_id"):
    env.reset(batch)
  correct = Mock(wraps=env.posterior_provider.estimator.correct)
  env.posterior_provider.estimator.correct = correct
  rng = torch.random.get_rng_state()
  obs = env.reset(episode_id=torch.arange(BATCH), time_index=0)
  assert correct.call_count == BATCH
  assert torch.equal(rng, torch.random.get_rng_state())
  assert torch.equal(obs["history"], batch["sensor_window"])
  for i in range(BATCH):
    prior, _ = env.replay_buffer.training_prefix(i, 0)
    expected = correct(prior.index(None), batch["actor_obs"][i : i + 1, :64])
    torch.testing.assert_close(
      env.posterior_provider.state.belief.covariance[i], expected.covariance[0]
    )
  with pytest.raises(ValueError, match="full restore"):
    env.step(torch.zeros(BATCH, 29), posterior=belief.to_actor_input())


def test_no_grad_imagination_does_not_retain_posterior_graphs():
  env, _, _ = setup_env()
  # eval() leaves model parameters differentiable, as in ordinary evaluation.
  assert not env.dynamics.training
  assert all(p.requires_grad for p in env.dynamics.parameters())
  with torch.no_grad():
    env.reset(episode_id=torch.arange(BATCH), segment_time=0)
    for _ in range(8):
      obs, reward, termination, _ = env.step(
        torch.zeros(BATCH, 29),
        process_noise=torch.zeros(BATCH, 70),
        measurement_noise=torch.zeros(BATCH, 64),
      )
      state = env.posterior_provider.state
      values = [getattr(state.belief.mean, f.name) for f in fields(LieState)]
      for value in (
        *values,
        state.belief.covariance,
        state.context,
        *obs.values(),
        reward,
        termination,
      ):
        assert not value.requires_grad
        assert value.grad_fn is None


def test_failed_imagined_validation_reuses_latent_transition_and_measurement(
  monkeypatch,
):
  env, batch, belief = setup_env()
  initial = env.reset(episode_id=torch.arange(BATCH), time_index=0)
  initial_state = env.posterior_provider.state
  step = Mock(wraps=env._wm_step)
  prepare = Mock(wraps=env.posterior_provider.prepare)
  monkeypatch.setattr(env, "_wm_step", step)
  monkeypatch.setattr(env.posterior_provider, "prepare", prepare)
  goals = env._prepare_estimator_goals
  monkeypatch.setattr(
    env, "_prepare_estimator_goals", Mock(side_effect=ValueError("test failure"))
  )
  action = torch.full((BATCH, 29), 0.01, requires_grad=True)
  with pytest.raises(ValueError, match="test failure"):
    env.step(action)
  assert env.posterior_provider.state is initial_state
  assert torch.equal(env.get_observations()["history"], initial["history"])
  rng = torch.random.get_rng_state()
  with pytest.raises(ValueError, match="same action"):
    env.step(action + 0.01)
  monkeypatch.setattr(env, "_prepare_estimator_goals", goals)
  with pytest.raises(ValueError, match="same differentiable"):
    env.step(action.detach().clone().requires_grad_())
  following, _, _, _ = env.step(action)
  assert step.call_count == 1
  assert prepare.call_count == 2
  assert torch.equal(
    prepare.call_args_list[0].kwargs["measurement"],
    prepare.call_args_list[1].kwargs["measurement"],
  )
  assert torch.equal(rng, torch.random.get_rng_state())
  assert torch.equal(
    following["history"][:, -1, :64], prepare.call_args.kwargs["measurement"]
  )
  assert env.posterior_provider.state.time.tolist() == [1, 1, 1]


@pytest.mark.parametrize(
  ("change", "pending"),
  [
    ("same_frame", False),
    ("same_frame", True),
    ("episode", False),
    ("time", True),
    ("model", True),
  ],
)
def test_imagined_frame_rejects_external_changes_before_observing_or_retrying(
  change, pending, monkeypatch
):
  env, _, _ = setup_env()
  ids = torch.arange(BATCH)
  env.reset(episode_id=ids, time_index=0)
  provider = env.posterior_provider
  action = torch.zeros(BATCH, 29)
  if pending:
    with monkeypatch.context() as patch:
      patch.setattr(
        env, "_prepare_estimator_goals", Mock(side_effect=ValueError("goals"))
      )
      with pytest.raises(ValueError, match="goals"):
        env.step(action)
  latent = env.rep_state
  rng = torch.random.get_rng_state()
  if change == "model":
    with torch.no_grad():
      env.dynamics.mlp[-1].bias.add_(0.03)
  else:
    state = provider.state
    restored = state.belief.clone()
    restored.mean.position.add_(1)
    provider.restore(
      restored,
      state.context,
      model_version=state.model_version,
      episode_id=ids + (BATCH if change == "episode" else 0),
      time_index=state.time_index + (1 if change == "time" else 0),
    )
  current = provider.state
  with pytest.raises(ValueError, match="reset"):
    env.get_observations()
  with pytest.raises(ValueError, match="reset"):
    env.step(action)
  assert env.rep_state is latent and provider.state is current
  assert torch.equal(torch.random.get_rng_state(), rng)
  env.reset(episode_id=ids, time_index=0)
  env.step(action)
  assert torch.equal(provider.state.episode_id, ids)
  assert torch.equal(provider.state.time_index, torch.ones_like(ids))


def test_invalid_rollout_inputs_leave_frame_and_noise_draws_available_for_retry():
  env, _, _ = setup_env()
  env.reset(episode_id=torch.arange(BATCH), time_index=0)
  state = env.posterior_provider.state
  action = torch.zeros(BATCH, 29)
  rng = torch.random.get_rng_state()
  for name, width in (
    ("actions", 29),
    ("process_noise", 70),
    ("measurement_noise", 64),
  ):
    for invalid in (
      torch.full((BATCH, width), float("nan")),
      torch.zeros(BATCH, width + 1),
      torch.zeros(BATCH, width, dtype=torch.float64),
    ):
      args = {"actions": action, name: invalid}
      with pytest.raises(ValueError, match=name):
        env.step(**args)
  assert env.posterior_provider.state is state
  assert torch.equal(torch.random.get_rng_state(), rng)
  env.step(action)
  assert env.posterior_provider.state.time_index.tolist() == [1] * BATCH


def test_mutating_explicit_noise_preserves_samples_and_gradients_on_retry(monkeypatch):
  env, _, _ = setup_env()
  ids = torch.arange(BATCH)
  base_process, base_sensor = torch.randn(BATCH, 70), torch.randn(BATCH, 64)

  def rollout(with_retry):
    env.reset(episode_id=ids, time_index=0)
    action = torch.zeros(BATCH, 29, requires_grad=True)
    process = base_process.clone().requires_grad_()
    sensor = base_sensor.clone().requires_grad_()
    if with_retry:
      with monkeypatch.context() as patch:
        patch.setattr(
          env, "_prepare_estimator_goals", Mock(side_effect=ValueError("goals"))
        )
        with pytest.raises(ValueError, match="goals"):
          env.step(action, process_noise=process, measurement_noise=sensor)
      with pytest.raises(ValueError, match="same differentiable process_noise"):
        env.step(action, process_noise=process.clone())
      with torch.no_grad():
        process.add_(2)
        sensor.sub_(2)
      for name, noise in (("process_noise", process), ("measurement_noise", sensor)):
        with pytest.raises(ValueError, match="same action/noises"):
          env.step(action, **{name: noise})
      obs, reward, _, _ = env.step(action)
      # A reusable input buffer may also be overwritten after a successful step.
      with torch.no_grad():
        process.zero_()
        sensor.zero_()
    else:
      obs, reward, _, _ = env.step(
        action, process_noise=process, measurement_noise=sensor
      )
    objective = reward.sum() + obs["goal_motion"].square().mean()
    return objective.detach(), torch.autograd.grad(objective, (action, process, sensor))

  expected, expected_grads = rollout(False)
  actual, actual_grads = rollout(True)
  torch.testing.assert_close(actual, expected)
  for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
    assert torch.isfinite(actual_grad).all() and actual_grad.norm() > 0
    torch.testing.assert_close(actual_grad, expected_grad)


def test_nonfinite_imagined_results_do_not_poison_retry(monkeypatch):
  env, _, _ = setup_env()
  env.reset(episode_id=torch.arange(BATCH), time_index=0)
  provider_state = env.posterior_provider.state
  latent = env.rep_state
  action = torch.zeros(BATCH, 29)
  process = torch.zeros(BATCH, 70)
  sensor = torch.zeros(BATCH, 64)
  extreme = process.clone()
  extreme[:, 0] = torch.finfo(process.dtype).max
  with pytest.raises(ValueError, match="non-finite"):
    env.step(action, process_noise=extreme, measurement_noise=sensor)
  with monkeypatch.context() as patch:
    # An invalid reward can accompany an entirely finite state and measurement.
    patch.setattr(
      env.representation,
      "compute_wm_reward_from_reference",
      Mock(return_value=torch.full((BATCH,), float("nan"))),
    )
    with pytest.raises(ValueError, match="non-finite"):
      env.step(action, process_noise=process, measurement_noise=sensor)
  assert env.posterior_provider.state is provider_state and env.rep_state is latent
  env.step(action, process_noise=process, measurement_noise=sensor)
  assert env.posterior_provider.state.time_index.tolist() == [1] * BATCH


def test_two_step_fixed_noise_policy_gradient_and_covariance_sensitivity():
  env, batch, belief = setup_env()
  env.dynamics.requires_grad_(False)
  actor = MultiEncoderAE(
    93,
    29,
    env.task_specs,
    {"motion": 1.0},
    8,
    (16, 8),
    0.0,
    HISTORY,
    uncertainty_dim=450,
    film_hidden_dims=(12, 8),
  ).eval()
  # Exercise upstream covariance after the prescribed zero-output init stage.
  with torch.no_grad():
    actor.film[-1].weight.normal_(std=0.01)
  process = torch.randn(2, BATCH, 70)
  measurement = torch.randn(2, BATCH, 64)

  def rollout():
    nonlocal covariance_sqrt
    obs = env.reset(episode_id=torch.arange(BATCH), time_index=0)
    covariance_sqrt = (
      env.posterior_provider.state.belief.covariance_sqrt.requires_grad_()
    )
    rewards = []
    for k in range(2):
      obs, reward, _, _ = env.step(
        actor(obs), process_noise=process[k], measurement_noise=measurement[k]
      )
      rewards.append(reward)
    return torch.stack(rewards).sum() / BATCH

  covariance_sqrt = None
  parameter = actor.decoder[-1].bias
  loss = rollout()
  gradient, covariance_gradient = torch.autograd.grad(
    loss, (parameter, covariance_sqrt)
  )
  assert torch.isfinite(gradient).all() and gradient.norm() > 1e-6
  assert torch.isfinite(covariance_gradient).all() and covariance_gradient.norm() > 1e-7
  assert all(p.grad is None for p in env.dynamics.parameters())
  direction = torch.randn_like(parameter)
  direction /= direction.norm()
  epsilon = 0.002
  original = parameter.detach().clone()
  with torch.no_grad():
    parameter.copy_(original + epsilon * direction)
  plus = rollout().detach()
  with torch.no_grad():
    parameter.copy_(original - epsilon * direction)
  minus = rollout().detach()
  with torch.no_grad():
    parameter.copy_(original)
  torch.testing.assert_close(
    (plus - minus) / (2 * epsilon), gradient @ direction, rtol=0.015, atol=2e-4
  )
  torch.testing.assert_close(rollout().detach(), loss.detach(), atol=0, rtol=0)


def test_real_runner_failure_before_action_retries_and_keeps_model_snapshot(
  monkeypatch,
):
  runner = _runner()
  stochastic_env, _, prior = setup_env()
  runner.cfg.process_noise = stochastic_env.dynamics.process_noise_config
  runner._dynamics = stochastic_env.dynamics
  runner._wm.dynamics = runner._dynamics
  runner.wm_env.dynamics = runner._dynamics

  def draw(*, env_ids, **kwargs):
    sampled = prior.index(env_ids).clone()
    sampled.mean.position.add_(torch.randn_like(sampled.mean.position))
    return sampled

  factory = Mock(side_effect=draw)
  real, imagined = runner.configure_belief_filter(factory)
  assert real is not imagined and real.estimator.dynamics is imagined.estimator.dynamics
  runner.env = _PhysicalEnv(runner)
  runner.action_manager = runner.env.action_manager
  runner.replay_buffer = ReplayBuffer(runner._buffer_schema(), 8, BATCH)
  runner._sampler = None
  runner.logger = Mock()
  runner._pending_norm_updates = []
  runner.world_model_learner = Mock()
  runner.cfg.policy.noise = 0.0
  runner.policy_learner = InputStagePolicy(
    build_actor(runner.cfg, runner._wm, "cpu"), runner.cfg.policy
  )
  runner.reset_collection()
  import supermimic.runners.supermimic_runner as module

  builder = module.build_actor_features
  monkeypatch.setattr(
    module, "build_actor_features", Mock(side_effect=ValueError("test rejection"))
  )
  initial_measurement = runner._sim_history.clone()
  with pytest.raises(ValueError, match="test rejection"):
    runner.step()
  assert real.state is None and not runner.env.actions
  factory.assert_called_once()
  assert torch.equal(runner._sim_history, initial_measurement)
  monkeypatch.setattr(module, "build_actor_features", builder)
  version = runner._dynamics.model_version
  runner.step()
  factory.assert_called_once()
  runner.step()
  assert factory.call_count == 2 and factory.call_args.kwargs["env_ids"].tolist() == [0]
  assert len(runner.env.actions) == 2
  assert real.state.time.tolist() == [0, 1, 1]
  assert real.state.belief.covariance.grad_fn is None
  assert not runner.world_model_learner.update_normalization.called
  assert runner._dynamics.model_version == version
  assert runner.replay_buffer.num_transitions == 2  # One batched record per step.


def test_process_noise_runner_config_roundtrips():
  runner = _runner()
  runner.cfg.process_noise = ProcessNoiseConfig(hidden_dims=(8,))
  restored = load_runner_cfg({"runner_cfg": asdict(runner.cfg)})
  assert restored == runner.cfg


@pytest.mark.parametrize("change", ["model", "context"])
def test_cached_collector_posterior_checks_version_and_context(change):
  runner = _runner()
  env, _, prior = setup_env()
  runner.cfg.process_noise = env.dynamics.process_noise_config
  runner._dynamics = env.dynamics
  runner.configure_belief_filter(FixedBeliefPrior(prior))
  runner._episode_id = torch.arange(BATCH)
  runner._time_index = torch.zeros_like(runner._episode_id)
  _build_current(runner)
  if change == "model":
    with torch.no_grad():
      runner._dynamics.state_normalizer._mean.add_(0.01)
  else:
    runner.posterior_provider.state.context.add_(0.01)
  with pytest.raises(ValueError, match="Cached posterior"):
    _build_current(runner)


def test_imagined_provider_rejects_mismatched_measurement_definition():
  from supermimic.estimator_inputs import GaussianMeasurementConfig

  env, batch, belief = setup_env()
  env.posterior_provider.estimator.measurement_config = GaussianMeasurementConfig(
    gravity_std=0.5
  )
  with pytest.raises(ValueError, match="measurement definition"):
    env.reset(episode_id=torch.arange(BATCH), time_index=0)


@pytest.mark.parametrize("ready_flag", [False, True])
def test_runner_reads_matching_restored_frame_directly_from_provider(ready_flag):
  runner = _runner("B")
  env, batch, prior = setup_env("B")
  runner.cfg.process_noise = env.dynamics.process_noise_config
  runner._dynamics = env.dynamics
  real, _ = runner.configure_belief_filter(FixedBeliefPrior(prior))
  runner._episode_id = torch.arange(BATCH)
  runner._time_index = torch.zeros_like(runner._episode_id)
  before = _build_current(runner)["goal_motion"]
  saved = real.state_dict()
  saved["mean"][:, 0] += 2
  real.load_state_dict(saved)
  runner._posterior_ready = (
    ready_flag  # Only the legacy injection adapter may consult this flag.
  )
  state = real.state
  after = _build_current(runner)["goal_motion"]
  assert real.state is state
  assert (before - after).abs().max() > 1
  assert runner._current_posterior is None
  for change in ("context", "dtype", "width"):
    bad = dict(saved)
    if change == "context":
      bad["context"] = saved["context"].double()
    elif change == "width":
      bad["context"] = saved["context"].repeat(1, 2)
    else:
      bad["mean"], bad["covariance_sqrt"] = (
        saved["mean"].double(),
        saved["covariance_sqrt"].double(),
      )
    with pytest.raises(ValueError):
      real.load_state_dict(bad)
    assert real.state is state


def test_keyed_rebuild_uses_latest_weights_and_exact_retained_prefix():
  env, batch, prior = setup_env()
  buffer = env.replay_buffer
  # Three frames per episode, different measurement at each transition.
  head = buffer._storage[:, 0].clone()
  for k in (1, 2):
    frame = head.clone()
    frame["time_index"].fill_(k)
    frame["actor_obs"][:, 6] += 0.1 * k
    frame["state"][:, 0] += k
    buffer.insert(frame)
  ids = torch.tensor([2, 0, 0])
  times = torch.tensor([2, 0, 1])
  first = env.reset(episode_id=ids, time_index=times)
  old = env.posterior_provider.state
  assert torch.equal(old.time_index, times) and torch.equal(old.episode_id, ids)
  torch.testing.assert_close(env.aux_state[:, 0], times.float())
  with torch.no_grad():
    env.dynamics.mlp[-1].bias.add_(0.1)
  second = env.reset(episode_id=ids, time_index=times)
  new = env.posterior_provider.state
  assert new.model_version == env.dynamics.model_version != old.model_version
  assert not torch.allclose(old.belief.mean.velocity[0], new.belief.mean.velocity[0])
  assert torch.equal(first["history"], second["history"])
  filt = env.posterior_provider.estimator
  for i, (ep, t) in enumerate(zip(ids.tolist(), times.tolist(), strict=True)):
    prior_ep, seq = buffer.training_prefix(ep, t)
    belief = filt.correct(prior_ep.index(None), seq["actor_obs"][:1, :64])
    for k in range(t):
      belief = filt.predict(belief, seq["action"][k : k + 1], seq["sim_ctx"][k : k + 1])
      belief = filt.correct(belief, seq["actor_obs"][k + 1 : k + 2, :64])
    torch.testing.assert_close(belief.mean.to_full()[0], new.belief.mean.to_full()[i])
    torch.testing.assert_close(belief.covariance[0], new.belief.covariance[i])
  for k in (3, 4):
    frame = head.clone()
    frame["time_index"].fill_(k)
    buffer.insert(frame)
  with pytest.raises(ValueError, match="unavailable"):
    env.reset(episode_id=ids, time_index=times)
  assert env.posterior_provider.state is new
  obs = env.reset(episode_id=ids, segment_time=0)
  assert torch.equal(env.posterior_provider.state.time_index, torch.zeros_like(ids))
  oldest = buffer._storage["actor_obs"][ids, 1]
  assert torch.equal(obs["history"], oldest[:, None].expand_as(obs["history"]))
  assert torch.equal(env._segment_start_time, torch.ones_like(ids))


def test_prefix_crosses_ring_wrap_and_ignores_another_episode_in_same_slot():
  env, _, prior = setup_env()
  buffer = env.replay_buffer
  frame = buffer._storage[:, 0].clone()
  for t in (1, 2):
    frame["time_index"].fill_(t)
    buffer.insert(frame)
  episodes = frame["episode_id"] + BATCH
  frame["episode_id"] = episodes
  frame["time_index"].zero_()
  buffer.insert(frame, head_priors=(episodes, prior))
  frame["time_index"].fill_(1)
  buffer.insert(frame)
  for row, episode in enumerate(episodes.tolist()):
    restored_prior, prefix = buffer.episode_prefix(episode, 1)
    torch.testing.assert_close(restored_prior.mean.position, prior.mean.position[row])
    assert prefix["time_index"].tolist() == [0, 1]
    assert prefix["episode_id"].tolist() == [episode, episode]
    with pytest.raises(ValueError, match="prefix"):
      buffer.episode_prefix(episode, 2)
    with pytest.raises(ValueError, match="head"):
      buffer.episode_prefix(episode - BATCH, 1)


def test_training_segments_follow_overwrites_and_independent_resets(monkeypatch):
  env, _, _ = setup_env()
  buffer = env.replay_buffer
  template = buffer._storage[:, 0].clone()
  ids = torch.arange(BATCH)
  times = torch.zeros_like(ids)
  action = torch.zeros(BATCH, 29)
  for k in range(1, 7):
    record = template.clone()
    record["prev_action"] = action
    action = torch.full((BATCH, 29), 0.01 * k)
    record["action"] = action
    times += 1
    if k == 4:
      ids[1] += BATCH
      times[1] = 0
      record["prev_action"][1] = 0
    record["state"][:, 0] = k + 10 * torch.arange(BATCH)
    record["actor_obs"][:, :64] += 0.001 * k
    record["actor_obs"][:, 64:] = record["prev_action"]
    record["episode_id"], record["time_index"] = ids.clone(), times.clone()
    record["terminated"][1] = k == 3
    buffer.insert(record)
  buffer.finish_collection_round()
  rng = torch.random.get_rng_state()
  episodes = torch.tensor([0, 4, 1])
  target = torch.tensor([1, 2, 0])
  obs = env.reset(episode_id=episodes, segment_time=target)
  # Ask the step-1 history sampler for exactly these environment rows and ages.
  with monkeypatch.context() as patch:
    patch.setattr(
      torch,
      "randint",
      Mock(side_effect=[torch.tensor([0, 1, 1]), torch.tensor([1, 3, 0])]),
    )
    window, padded = buffer.sample_causal(HISTORY, BATCH)
  assert torch.equal(window["actor_obs"], obs["history"])
  assert torch.equal(padded.sum(-1), HISTORY - 1 - target)
  state = env.posterior_provider.state
  assert state.time_index.tolist() == [1, 2, 0]
  assert env._segment_start_time.tolist() == [3, 0, 3]
  filt = env.posterior_provider.estimator
  for row, (ep, t) in enumerate(zip(episodes.tolist(), target.tolist(), strict=True)):
    seed, frames = buffer.training_prefix(ep, t, relative=True)
    assert frames["segment_time"].tolist() == list(range(t + 1))
    # At the ring cut the known action from the preceding physical step survives.
    if ep in (0, 1):
      torch.testing.assert_close(frames["prev_action"][0], torch.full((29,), 0.02))
    truth = LieState.from_full(frames["state"][0])
    err = local_error(truth, seed.mean)
    assert 0 < err[:3].norm() and (err.abs() < 6 * seed.covariance.diag().sqrt()).all()
    q = filt.correct(seed, frames["actor_obs"][0, :64])
    for j in range(t):
      q = filt.predict(q, frames["action"][j], frames["sim_ctx"][j])
      q = filt.correct(q, frames["actor_obs"][j + 1, :64])
    torch.testing.assert_close(state.belief.mean.to_full()[row], q.mean.to_full())
    torch.testing.assert_close(state.belief.covariance[row], q.covariance)
    history = frames["actor_obs"][
      (torch.arange(HISTORY) + t + 1 - HISTORY).clamp_min(0)
    ]
    assert torch.equal(obs["history"][row], history)
  # An unavailable physical target cannot replace the current rollout.
  with pytest.raises(ValueError, match="unavailable"):
    env.reset(episode_id=episodes, time_index=0)
  assert env.posterior_provider.state is state
  buffer.finish_collection_round()
  assert torch.equal(torch.random.get_rng_state(), rng)


def test_gaussian_prior_uses_every_right_local_block_and_configured_variance():
  _, batch, _ = setup_env()
  truth = LieState.from_full(batch["state"]).to(torch.float64)
  prior = GaussianStatePrior((0.004, 0.009, 0.03, 0.02, 0.008, 0.04))
  generator = torch.Generator().manual_seed(71)
  sample = prior.sample(truth, generator=generator)
  std = torch.tensor(prior.std, dtype=torch.float64).repeat_interleave(
    torch.tensor([3, 3, 3, 3, 29, 29])
  )
  generator.manual_seed(71)
  expected = torch.randn(BATCH, 70, generator=generator, dtype=torch.float64) * std
  torch.testing.assert_close(local_error(truth, sample.mean), expected)
  torch.testing.assert_close(
    sample.covariance, torch.diag(std.square()).expand(BATCH, 70, 70)
  )


def test_imagined_only_snapshot_restores_without_fusion_replay_or_rng(
  monkeypatch, tmp_path
):
  env, _, _ = setup_env()
  env.dynamics.requires_grad_(False)
  env.reset(episode_id=torch.arange(BATCH), segment_time=0)
  env.step(torch.zeros(BATCH, 29))
  snapshot = env.snapshot()
  first_obs = env.get_observations().clone()
  process, measurement = torch.randn(BATCH, 70), torch.randn(BATCH, 64)

  def next_step():
    action = torch.full((BATCH, 29), 0.03, requires_grad=True)
    obs, reward, term, _ = env.step(
      action, process_noise=process, measurement_noise=measurement
    )
    (gradient,) = torch.autograd.grad(
      reward.sum() + obs["task_uncertainty"].sum() * 0.001, action
    )
    return obs, reward, term, gradient, env.posterior_provider.state.belief.covariance

  expected = next_step()
  provider = env.posterior_provider
  before = provider.state
  bad = dict(snapshot, frame=dict(snapshot["frame"], prev_action=torch.ones(BATCH, 29)))
  with pytest.raises(ValueError, match="measurement actions"):
    env.restore(bad)
  assert provider.state is before
  # A fresh environment can resume a serialized imagined frame without replay.
  path = tmp_path / "imagined.pt"
  torch.save(snapshot, path)
  snapshot = torch.load(path, weights_only=True)
  env, _, _ = setup_env()
  env.dynamics.requires_grad_(False)
  provider = env.posterior_provider
  assert provider.state is None
  rng = torch.random.get_rng_state()
  with monkeypatch.context() as patch:
    patch.setattr(env, "replay_buffer", None)
    patch.setattr(
      provider.estimator,
      "correct",
      Mock(side_effect=AssertionError("duplicate fusion")),
    )
    restored = env.restore(snapshot)
  assert torch.equal(torch.random.get_rng_state(), rng)
  for name in first_obs.keys():
    torch.testing.assert_close(restored[name], first_obs[name], atol=0, rtol=0)
  actual = next_step()
  for name in actual[0].keys():
    torch.testing.assert_close(actual[0][name], expected[0][name])
  for got, want in zip(actual[1:], expected[1:], strict=True):
    torch.testing.assert_close(got, want)


@pytest.mark.parametrize(
  "field",
  [
    "joint_pos",
    "joint_vel",
    "root_pos_w",
    "root_quat_w",
    "root_lin_vel_w",
    "root_ang_vel_w",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
  ],
)
def test_snapshot_rejects_changed_motion_content_before_restoring(field, monkeypatch):
  env, _, _ = setup_env()
  obs = env.reset(episode_id=torch.arange(BATCH), segment_time=0).clone()
  snapshot = env.snapshot()
  provider = env.posterior_provider
  state, generation = provider.state, provider.generation
  latent, frame = env.rep_state, env._belief_frame
  prepare = Mock(wraps=provider.prepare_load_state_dict)
  monkeypatch.setattr(provider, "prepare_load_state_dict", prepare)
  reference = getattr(env.motion, field)
  original = reference.clone()
  # Same object and clip lengths: an identity cached at load time is insufficient.
  reference[..., 0].add_(0.25)
  rng = torch.random.get_rng_state()
  with pytest.raises(ValueError, match="reference motion content mismatch"):
    env.restore(snapshot)
  prepare.assert_not_called()
  assert provider.state is state and provider.generation == generation
  assert env.rep_state is latent and env._belief_frame is frame
  assert torch.equal(torch.random.get_rng_state(), rng)
  for key, value in obs.items():
    torch.testing.assert_close(env.get_observations()[key], value, atol=0, rtol=0)
  reference.copy_(original)
  restored = env.restore(snapshot)
  for key, value in obs.items():
    torch.testing.assert_close(restored[key], value, atol=0, rtol=0)


@pytest.mark.parametrize("legacy", [False, True])
def test_snapshot_requires_reference_content_identity(legacy):
  env, _, _ = setup_env()
  env.reset(episode_id=torch.arange(BATCH), segment_time=0)
  snapshot = env.snapshot()
  snapshot.pop("motion_digest")
  if legacy:
    snapshot["version"] = 1
  state = env.posterior_provider.state
  with pytest.raises(ValueError, match="motion"):
    env.restore(snapshot)
  assert env.posterior_provider.state is state


def test_world_model_bundle_checks_stochastic_configuration_and_preserves_legacy():
  from supermimic.runners.supermimic_runner import WorldModel
  from tests.test_estimator_stochastic_dynamics import make_dynamics

  for enabled in (False, True):
    model, rep = make_dynamics(stochastic=enabled)
    wm = WorldModel(rep, model, torch.nn.Identity(), 700, 29)
    bundle = wm.bundle()
    wm.load_bundle(bundle)
    if enabled:
      model.process_noise_config = replace(
        model.process_noise_config, diagonal_floor=0.0005
      )
      with pytest.raises(ValueError, match="configuration"):
        wm.load_bundle(bundle)


def test_model_bundle_restores_filter_noise_and_prior_without_online_state():
  from supermimic.belief import GaussianStatePrior
  from supermimic.estimator import FilterConfig
  from supermimic.estimator_inputs import GaussianMeasurementConfig
  from supermimic.runners.supermimic_runner import WorldModel
  from tests.test_estimator_stochastic_dynamics import make_dynamics

  def build():
    model, rep = make_dynamics()
    rep.actor_obs_mode = "proprio"
    return WorldModel(rep, model, torch.nn.Identity(), 700, 29)

  wm = build()
  wm.filter_config = FilterConfig(validate_numerics=True)
  wm.initial_prior = GaussianStatePrior((0.02,) * 6)
  wm.representation.configure_estimator_measurement(
    GaussianMeasurementConfig(joint_position_std=0.0003)
  )
  bundle = wm.bundle()
  restored = build()
  restored.load_bundle(bundle)
  assert restored.filter_config == wm.filter_config
  assert restored.initial_prior == wm.initial_prior
  assert (
    restored.representation.measurement_config == wm.representation.measurement_config
  )
  assert restored.dynamics.model_version == wm.dynamics.model_version
  del bundle["estimator"]["filter_config"]["validate_numerics"]
  with pytest.raises(ValueError, match="Incomplete"):
    restored.load_bundle(bundle)


@pytest.mark.parametrize("stochastic", [False, True])
@pytest.mark.parametrize(
  "mutation", ["missing", "shape", "dtype", "nan", "zero_scale", "dynamics"]
)
def test_invalid_bundle_does_not_partially_restore_model_or_configuration(
  stochastic, mutation
):
  from supermimic.belief import GaussianStatePrior
  from supermimic.estimator import FilterConfig
  from supermimic.estimator_inputs import GaussianMeasurementConfig
  from supermimic.runners.supermimic_runner import WorldModel
  from supermimic.sim_context import FootFrictionSpec, SimContext
  from tests.test_estimator_stochastic_dynamics import make_dynamics

  def build(std):
    model, rep = make_dynamics(stochastic=stochastic)
    rep.actor_obs_mode = "proprio"
    rep.configure_estimator_measurement(GaussianMeasurementConfig(gravity_std=std))
    context = SimContext([FootFrictionSpec()], [(0.2, 1.4)])
    return WorldModel(rep, model, context, 700, 29)

  source, target = build(0.02), build(0.01)
  source.filter_config = FilterConfig(validate_numerics=True)
  source.initial_prior = GaussianStatePrior((0.02,) * 6)
  source.sim_context.mean.add_(0.1)
  source.sim_context.scale.mul_(2)
  version = target.dynamics.model_version
  measurement = target.representation.measurement_config
  bundle = source.bundle()
  assert source.dynamics.model_version != version
  if mutation == "missing":
    del bundle["sim_context"]["scale"]
  elif mutation == "shape":
    bundle["sim_context"]["scale"] = torch.ones(2)
  elif mutation == "dtype":
    bundle["sim_context"]["scale"] = torch.ones(1, dtype=torch.float64)
  elif mutation == "nan":
    bundle["sim_context"]["mean"] = torch.full((1,), float("nan"))
  elif mutation == "zero_scale":
    bundle["sim_context"]["scale"] = torch.zeros(1)
  else:
    weights = bundle["dynamics"]["state_dict"] if stochastic else bundle["dynamics"]
    del weights["mlp.3.bias"]
  # state_dict returns tensor aliases: snapshot the live context independently.
  context_before = {k: v.clone() for k, v in target.sim_context.state_dict().items()}
  with pytest.raises(ValueError, match="checkpoint"):
    target.load_bundle(bundle)
  assert target.dynamics.model_version == version
  assert target.filter_config == FilterConfig()
  assert target.initial_prior == GaussianStatePrior()
  assert target.representation.measurement_config == measurement
  for key, value in target.sim_context.state_dict().items():
    torch.testing.assert_close(value, context_before[key], atol=0, rtol=0)
