"""v8 geometry, information boundaries, exact replay and training transactions."""

from copy import deepcopy
from dataclasses import replace
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch.func import jacrev

from supermimic.belief import BeliefState
from supermimic.collection_observations import CollectionObservations
from supermimic.critic_inputs import CriticInputConfig, build_critic_features
from supermimic.estimator_inputs import (
  EITInputConfig,
  EpisodeAlignment,
  perturb_gt_state,
)
from supermimic.local_control import root_tracking_error, root_tracking_jacobian
from supermimic.replay_buffer import FieldSpec, ReplayBuffer
from supermimic.runners.eit_training import EITTraining
from supermimic.runners.supermimic_runner import InputStagePolicy, build_actor
from supermimic.utils.lie_state import LieState, boxplus, so3_exp
from tests.test_estimator_belief_filter import _fixture
from tests.test_estimator_eit_training import training


def v8_training(group="belief", buffer_size=4, film_hidden_dims=(128, 64)):
  base = training(group=group, buffer_size=buffer_size, replay_ratio=0)
  policy_cfg = base.policy.cfg
  runner = base.runner
  runner.cfg.estimator_inputs = EITInputConfig(group=group, policy_version="v8")
  runner.cfg.film_hidden_dims = film_hidden_dims
  runner.cfg.heading_dropout_probability = 0
  runner.cfg.heading_rate_hz = (5, 5)
  runner.collection_observations = CollectionObservations(runner.cfg, "cpu", 31)
  runner.policy_learner = InputStagePolicy(
    build_actor(runner.cfg, runner._wm, "cpu").double(), runner.cfg.policy
  )
  runner.wm_env.estimator_input_config = runner.cfg.estimator_inputs
  runner.wm_env.critic_input_config = None
  runner._schema.update(
    tracking_error=FieldSpec(9, torch.float64),
    heading=FieldSpec(dtype=torch.float64),
    heading_valid=FieldSpec(dtype=torch.bool),
  )
  runner.replay_buffer = ReplayBuffer(
    runner._schema, buffer_size, 2, initial_prior=runner._wm.initial_prior
  )
  runner.wm_env.replay_buffer = runner.replay_buffer
  original_reset = runner.reset_collection

  def reset(self):
    original_reset()
    self._time_index = torch.zeros(2, dtype=torch.long)
    self._measurement_reset = torch.ones(2, dtype=torch.bool)
    self._posterior_ready = False
    self._v8_packet = self._v8_gt_noise = self._v8_history = None
    self._sim_history = None

  @torch.no_grad()
  def step(self, *, record_episode_stats=True):
    frame = self.template[self.index % len(self.template)].expand(2).clone()
    truth = LieState.from_full(frame["state"])
    sensor = torch.cat(
      (self.posterior_provider.estimator.observation(truth), self.previous), -1
    )
    if self._sim_history is None:
      self._sim_history = sensor[:, None].repeat(1, 5, 1)
    else:
      self._sim_history = torch.cat((self._sim_history[:, 1:], sensor[:, None]), 1)
    self.motion_cmd = SimpleNamespace(
      reference_alignment=EpisodeAlignment.identity(2, dtype=torch.float64)
    )
    aux, rep = truth.to_rep()
    obs = self._build_sim_obs_td(
      rep, aux, sensor, frame["motion_time"], frame["sim_ctx"]
    )
    # A retry must reuse GT noise, the heading packet and the complete actor frame.
    retry = self._build_sim_obs_td(
      rep, aux, sensor, frame["motion_time"], frame["sim_ctx"]
    )
    assert torch.equal(obs["history"], retry["history"])
    action = self.policy_learner.actor(obs)
    frame.update(
      dict(
        actor_obs=sensor,
        tracking_error=obs["history"][:, -1, 93:],
        heading=self._v8_packet[0],
        heading_valid=self._v8_packet[1],
        action=action,
        prev_action=self.previous,
        episode_id=self._episode_id.clone(),
        time_index=self._time_index.clone(),
        terminated=torch.zeros(2, dtype=torch.bool),
        truncated=torch.zeros(2, dtype=torch.bool),
      )
    )
    self.replay_buffer.insert(
      frame,
      head_priors=self.posterior_provider.state.head_priors,
      posterior=self.replay_buffer.posterior_cache.capture(
        self.replay_buffer,
        self.posterior_provider.state.belief,
        self._dynamics.model_revision,
      ),
    )
    self.collected.append(
      (self._episode_id.clone(), self._time_index.clone(), obs["history"].clone())
    )
    self.previous = action.clone()
    self._time_index += 1
    self.index += 1
    self._measurement_reset.zero_()
    self._posterior_ready = False
    self._v8_packet = self._v8_gt_noise = None
    return 2

  runner.reset_collection = MethodType(reset, runner)
  runner.step = MethodType(step, runner)
  result = EITTraining(runner, base.cfg, policy_cfg)
  return result


def test_tracking_geometry_and_analytic_covariance_projection():
  _, belief, _, _ = _fixture(batch=1, dtype=torch.float64)
  state = belief.mean.index(0)
  rotation = state.rotation @ so3_exp(
    torch.tensor([0.3, -0.4, 0.5], dtype=torch.float64)
  )
  position = state.position + torch.tensor([1.0, -0.3, 0.4], dtype=torch.float64)
  velocity = torch.tensor([1.2, -0.3, 0.5], dtype=torch.float64)

  def error(delta):
    return root_tracking_error(
      boxplus(state, torch.cat((delta, delta.new_zeros(61)))),
      rotation,
      position,
      velocity,
    )

  expected = jacrev(error)(torch.zeros(9, dtype=torch.float64))
  actual = root_tracking_jacobian(state, rotation, position, velocity)
  torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-9)
  for angle in (0.0, 1e-6, 0.09, 2.9):
    rotation = state.rotation @ so3_exp(
      torch.tensor([0.0, 0.0, angle], dtype=torch.float64)
    )
    torch.testing.assert_close(
      root_tracking_jacobian(state, rotation, position, velocity),
      jacrev(error)(torch.zeros(9, dtype=torch.float64)),
      rtol=1e-8,
      atol=1e-9,
    )
  error0 = root_tracking_error(
    state, state.rotation, state.position, state.world_velocity
  )
  torch.testing.assert_close(
    error0, torch.zeros(9, dtype=torch.float64), atol=1e-12, rtol=0
  )
  moved = root_tracking_error(state, rotation, position, velocity)
  assert moved[:6].abs().sum() > 1
  torch.testing.assert_close(moved[6:], state.rotation.mT @ velocity - state.velocity)


def test_gt_noise_is_fixed_local_pose_and_body_velocity():
  _, belief, _, _ = _fixture(batch=3, dtype=torch.float64)
  cfg = EITInputConfig(group="gt", policy_version="v8")
  noise = torch.randn(3, 9, dtype=torch.float64)
  actual = perturb_gt_state(belief.mean, noise, cfg)
  expected = boxplus(
    belief.mean,
    torch.cat(
      (noise * noise.new_tensor([0.03] * 6 + [0.2] * 3), noise.new_zeros(3, 61)), -1
    ),
  )
  torch.testing.assert_close(actual.to_full(), expected.to_full())
  assert EITInputConfig.from_dict(cfg.to_dict()) == cfg


@pytest.mark.parametrize("group", ["gt", "mean", "belief"])
def test_collector_replay_history_training_and_restore(group, tmp_path):
  loop = v8_training(group)
  bundle = loop.initialization_bundle()
  assert bundle["version"] == "eit_initialization_v8"
  assert set(bundle["actors"]) == {"gt", "mean", "belief"}
  loop.load_initialization(bundle)
  for _ in range(7):
    loop.run_step()
  runner = loop.runner
  for episodes, times, history in runner.collected[-4:]:
    starts = runner.wm_env.prepare_policy_starts(episodes, times)
    torch.testing.assert_close(starts.batch["sensor_window"], history, rtol=0, atol=0)
  loop.runner.reset_collection()
  loop.run_step()
  latest = runner.collected[-1][2]
  assert torch.equal(latest, latest[:, :1].expand_as(latest))
  # New episode first frame has no transitions, so retain old episodes for fitting.
  for _ in range(2):
    loop.run_step()
  train, valid = loop.episode_split()
  loop.learner.fit(
    runner.replay_buffer,
    train,
    valid,
    runner.replay_buffer.retained_frames(train)[:0],
    2,
  )
  starts = loop.policy.sample_starts()
  noises = loop.policy.draw_noises(starts)
  with (
    patch.object(
      runner.posterior_provider.estimator,
      "correct_heading",
      side_effect=AssertionError("imagined heading"),
    ),
    patch.object(
      runner.wm_env.posterior_provider.estimator,
      "correct_heading",
      side_effect=AssertionError("imagined heading"),
    ),
  ):
    metrics = loop.policy.update_from(starts, *noises)
  assert torch.isfinite(metrics["Policy/return"])
  assert loop.actor.obs_normalizer._mean.shape[-1] == 102
  state = loop.state_dict()
  path = tmp_path / "training.pt"
  torch.save(state, path)
  state = torch.load(path, weights_only=True)
  restored = v8_training(group)
  restored.load_state_dict(state)
  assert restored.policy.actor_steps == loop.policy.actor_steps
  assert restored.policy.critic.config.input_dim(100) == (
    1266 if group == "gt" else 1320
  )
  for name in loop.policy.generators:
    assert torch.equal(
      loop.policy.generators[name].get_state(),
      restored.policy.generators[name].get_state(),
    )


def test_v8_permissions_and_critic_belief_sensitivity():
  loop = v8_training()
  loop.run_step()
  runner = loop.runner
  starts = loop.policy.sample_starts()
  truth = LieState.from_full(starts.batch["state"])
  history = starts.batch["sensor_window"]
  reference = runner._representation.reference_window(
    runner.motion,
    starts.motion_times,
    EpisodeAlignment(
      starts.batch["alignment_rotation"], starts.batch["alignment_translation"]
    ),
  )
  from supermimic.local_control import privileged_goal

  goal = privileged_goal(truth, reference)
  changed = BeliefState(
    replace(starts.belief.mean, position=starts.belief.mean.position + 1),
    covariance_sqrt=starts.belief.covariance_sqrt * 2,
  )
  for group in ("gt", "mean", "belief"):
    cfg = CriticInputConfig(group=group, policy_version="v8")
    first = build_critic_features(
      truth, goal, history, starts.batch["sim_ctx"], starts.belief, config=cfg
    )
    second = build_critic_features(
      truth, goal, history, starts.batch["sim_ctx"] + 10, changed, config=cfg
    )
    assert torch.equal(first, second) == (group == "gt")
    assert first.shape[-1] == cfg.input_dim(0)
  # Every method has identical actor weight shapes and shares initialization.
  bundle = loop.initialization_bundle()
  for name in bundle["actors"]["gt"]["actor"]:
    torch.testing.assert_close(
      bundle["actors"]["gt"]["actor"][name], bundle["actors"]["belief"]["actor"][name]
    )


def test_heading_schedule_dropout_frequency_partial_reset_and_rng():
  loop = v8_training()
  cfg = replace(
    loop.runner.cfg, heading_rate_hz=(2.5, 2.5), heading_dropout_probability=0
  )
  source = CollectionObservations(cfg, "cpu", 123)
  _, belief, _, _ = _fixture(batch=3, dtype=torch.float64)
  state = belief.mean
  reset = torch.ones(3, dtype=torch.bool)
  packets = []
  for t in range(200):
    _, valid = source.packet(
      state, reset if t == 0 else ~reset, torch.full((3,), t), 0.02
    )
    packets.append(valid)
  assert torch.equal(torch.stack(packets).sum(0), torch.full((3,), 10))
  previous = source.phase.clone()
  source.packet(
    state, torch.tensor([False, True, False]), torch.tensor([200, 0, 200]), 0.02
  )
  assert torch.equal(source.phase[[0, 2]], previous[[0, 2]])
  assert source.phase[1] != previous[1]
  saved = source.state_dict()
  target = CollectionObservations(cfg, "cpu", 7)
  target.load_state_dict(saved)
  torch.testing.assert_close(
    source.gt_noise(state.position), target.gt_noise(state.position), rtol=0, atol=0
  )
  a = source.packet(state, reset, torch.zeros(3, dtype=torch.long), 0.02)
  b = target.packet(state, reset, torch.zeros(3, dtype=torch.long), 0.02)
  torch.testing.assert_close(a, b, rtol=0, atol=0)
  dropout = CollectionObservations(
    replace(cfg, heading_dropout_probability=1), "cpu", 123
  )
  for t in (0, 1, 100, 999):
    value, mask = dropout.packet(
      state, reset if t == 0 else ~reset, torch.full((3,), t), 0.02
    )
    assert not mask.any() and not value.any()


@pytest.mark.parametrize("group", ["gt", "mean", "belief"])
def test_v8_microbatch_parity_and_exact_optimizer_continuation(group):
  from supermimic.learners.belief_policy_learner import BeliefActorCriticPolicyLearner

  loop = v8_training(group)
  loop.run_step()
  policy = loop.policy
  starts = policy.sample_starts()
  reference = BeliefActorCriticPolicyLearner(
    deepcopy(policy.actor),
    deepcopy(policy.critic),
    policy.wm_env,
    replace(policy.cfg, microbatch_size=1),
  )
  noises = policy.draw_noises(starts)
  torch.testing.assert_close(noises, reference.draw_noises(starts))
  a = policy.update_from(starts, *noises)
  b = reference.update_from(starts, *noises)
  torch.testing.assert_close(a, b, equal_nan=True, atol=1e-10, rtol=1e-8)
  torch.testing.assert_close(
    policy.actor.state_dict(), reference.actor.state_dict(), atol=1e-10, rtol=1e-8
  )
  continuation = BeliefActorCriticPolicyLearner(
    deepcopy(policy.actor), deepcopy(policy.critic), policy.wm_env, policy.cfg
  )
  continuation.load_state_dict(policy.state_dict())
  noises = policy.draw_noises(starts)
  torch.testing.assert_close(noises, continuation.draw_noises(starts), rtol=0, atol=0)
  a = policy.update_from(starts, *noises)
  b = continuation.update_from(starts, *noises)
  torch.testing.assert_close(a, b, equal_nan=True, rtol=0, atol=0)
  from supermimic.runners.eit_resume_reference import equal_tree

  equal_tree(policy.state_dict(), continuation.state_dict())


@pytest.mark.parametrize("group", ["gt", "mean", "belief"])
def test_v8_rollout_has_no_host_reads_and_batches_motion_encoder(group):
  from supermimic.imagined_rollout import prepare_rollout

  loop = v8_training(group)
  loop.run_step()
  policy = loop.policy
  policy.wm_env.dynamics.requires_grad_(False)
  policy.critic.requires_grad_(False)
  starts = policy.sample_starts()
  noises = tuple(n[:, :, 0] for n in policy.draw_noises(starts))
  prepared = prepare_rollout(policy.wm_env, starts, policy.cfg.horizon, *noises)
  # Do all boundary checks before entering the measured tensor-only loop.
  with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
    result = policy._rollout_kernel(prepared, *noises[:2], policy.cfg.horizon)
    (-result.returns.mean()).backward()
  keys = {event.key for event in prof.key_averages()}
  assert (
    not {
      "aten::item",
      "aten::_local_scalar_dense",
      "aten::nonzero",
      "aten::_linalg_check_errors",
    }
    & keys
  )
  assert not result.invalid.any()
  with patch.object(policy, "_encode", wraps=policy._encode) as encode:
    batched = policy._rollout_kernel(prepared, *noises[:2], policy.cfg.horizon)
    assert encode.call_count == 1
  policy.cfg = replace(policy.cfg, batch_motion_encoder=False)
  with patch.object(policy, "_encode", wraps=policy._encode) as encode:
    stepwise = policy._rollout_kernel(prepared, *noises[:2], policy.cfg.horizon)
    assert encode.call_count == policy.cfg.horizon
  torch.testing.assert_close(batched.returns, stepwise.returns)
  parameters = tuple(policy.actor.parameters())
  a = torch.autograd.grad(batched.returns.sum(), parameters, allow_unused=True)
  b = torch.autograd.grad(stepwise.returns.sum(), parameters, allow_unused=True)
  torch.testing.assert_close(a, b)


def test_v8_filter_and_film_gradients_and_actor_permissions():
  from supermimic.imagined_rollout import prepare_rollout

  loop = v8_training()
  loop.run_step()
  policy = loop.policy
  policy.wm_env.dynamics.requires_grad_(False)
  policy.critic.requires_grad_(False)
  with torch.no_grad():
    policy.actor.film[-1].weight.normal_(std=0.02)
  starts = policy.sample_starts()
  noises = tuple(n[:, :, 0] for n in policy.draw_noises(starts))
  prepared = prepare_rollout(policy.wm_env, starts, policy.cfg.horizon, *noises)
  lower = prepared.state.belief.covariance_sqrt.requires_grad_()
  position = prepared.state.belief.mean.position.requires_grad_()
  result = policy._rollout_kernel(prepared, *noises, policy.cfg.horizon)
  gradients = torch.autograd.grad(
    result.returns.sum(),
    (lower, position, policy.actor.film[-1].weight),
    retain_graph=True,
  )
  assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients)
  # Verify that the future covariance itself retains the action->filter path.
  covariance_grad = torch.autograd.grad(
    result.final_state.belief.covariance_sqrt.square().sum(),
    policy.actor.decoder[-1].weight,
  )[0]
  assert torch.isfinite(covariance_grad).all() and covariance_grad.abs().sum() > 0
  for group in ("gt", "mean"):
    baseline = v8_training(group).actor
    latent = torch.randn(2, baseline.latent_dim, dtype=torch.float64)
    history = torch.randn(2, 5, 102, dtype=torch.float64)
    condition = torch.randn(2, 74, dtype=torch.float64)
    with patch.object(
      baseline.film, "forward", side_effect=AssertionError("FiLM bypass")
    ):
      a = baseline.decode_goal(latent, history, condition)[0]
      b = baseline.decode_goal(latent, history, condition + 100)[0]
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_v8_normalization_counts_saved_errors_without_recomputing_gt_noise():
  loop = v8_training("gt")
  loop.run_step()
  first = loop.runner.collected[-1][2][:, -1]
  torch.testing.assert_close(
    loop.actor.obs_normalizer._mean, first.mean(0, keepdim=True)
  )
  rng = loop.runner.collection_observations.state_dict()
  loop.update_input_normalization()
  for name, value in rng.items():
    assert torch.equal(value, loop.runner.collection_observations.state_dict()[name])


def test_v8_geometry_compiles_as_one_differentiable_graph():
  from supermimic.estimator_inputs import (
    covariance_encoding_constants,
    encode_covariance_kernel,
  )
  from supermimic.local_control import root_tracking_covariance

  _, belief, _, _ = _fixture(batch=2, dtype=torch.float64)
  full = belief.mean.to_full().detach().requires_grad_()
  lower = belief.covariance_sqrt.detach().requires_grad_()
  rotation = belief.mean.rotation @ so3_exp(
    full.new_tensor([[0.2, -0.1, 0.5], [0.1, 0.2, -0.3]])
  )
  position = belief.mean.position + 0.5
  velocity = belief.mean.world_velocity + 0.3
  constants = covariance_encoding_constants(full, (1.0, 1.0, 1.0), 1e-6)

  def encode(full, lower):
    state = LieState.from_full(full)
    error = root_tracking_error(state, rotation, position, velocity)
    covariance = root_tracking_covariance(state, lower, rotation, position, velocity)
    return torch.cat((error, encode_covariance_kernel(covariance, constants)), -1)

  compiled = torch.compile(encode, backend="aot_eager", fullgraph=True)
  a, b = encode(full, lower), compiled(full, lower)
  torch.testing.assert_close(a, b)
  ga = torch.autograd.grad(a.square().sum(), (full, lower))
  gb = torch.autograd.grad(b.square().sum(), (full, lower))
  torch.testing.assert_close(ga, gb)


def test_v8_launcher_options_and_checkpoint_layout():
  import tomllib
  from pathlib import Path

  from supermimic.runners.supermimic_runner import (
    actor_checkpoint,
    load_actor_checkpoint,
  )
  from supermimic.scripts.launch_eit import _training_args

  config = tomllib.loads(Path("train_eit_v8.toml").read_text())
  assert set(config["methods"]) == {"gt", "mean", "belief"}
  options = _training_args(config["training"])
  assert "--policy-version=v8" in options
  assert "--policy-cuda-graphs" in options
  assert "--stepwise-motion-encoder" not in options
  index = options.index("--film-hidden-dims")
  assert options[index + 1 : index + 3] == ["128", "64"]
  loop = v8_training()
  checkpoint = actor_checkpoint(loop.actor, loop.runner.cfg)
  assert checkpoint["observation_layout"]["frame_dim"] == 102
  checkpoint["observation_layout"]["frame_dim"] = 93
  with pytest.raises(ValueError, match="layout"):
    load_actor_checkpoint(loop.actor, loop.runner.cfg, checkpoint)


@pytest.mark.parametrize("film_hidden_dims", [(128, 64), (17, 11, 7)])
def test_v8_shared_initialization_is_independent_of_generation_method(film_hidden_dims):
  from supermimic.scripts.train_eit import _preparation_fingerprint

  bundles = [
    v8_training(group, film_hidden_dims=film_hidden_dims).initialization_bundle()
    for group in ("gt", "mean", "belief")
  ]
  assert len({_preparation_fingerprint(bundle) for bundle in bundles}) == 1


@pytest.mark.parametrize("version", ["v7", "v8"])
def test_checkpoint_before_configurable_film_still_restores(version):
  from supermimic.runners.run_config import load_runner_cfg

  loop = training() if version == "v7" else v8_training()
  state = loop.state_dict()
  state["runner_config"].pop("film_hidden_dims")
  if version == "v7":
    for name in ("policy_version", "gt_pose_std", "gt_velocity_std"):
      state["runner_config"]["estimator_inputs"].pop(name)
    for name in ("heading_rate_hz", "heading_dropout_probability", "heading_std"):
      state["runner_config"].pop(name)
  loaded = load_runner_cfg({"runner_cfg": state["runner_config"]})
  assert loaded.estimator_inputs.policy_version == version
  assert loaded.film_hidden_dims == (128, 64)
  loop.load_state_dict(state)


@pytest.mark.parametrize("group", ["gt", "mean", "belief"])
def test_v8_custom_film_initialization_training_and_restore(group):
  import json

  from supermimic.runners.eit_resume_reference import equal_tree
  from supermimic.runners.run_config import load_runner_cfg
  from supermimic.runners.supermimic_runner import load_actor_checkpoint

  widths = (17, 11, 7)
  loop = v8_training(group, film_hidden_dims=widths)
  layers = [m for m in loop.actor.film if isinstance(m, torch.nn.Linear)]
  assert [m.out_features for m in layers[:-1]] == list(widths)
  assert layers[0].in_features == 45
  assert layers[-1].out_features == 2 * loop.runner.cfg.actor_hidden_dims[0]
  assert not layers[-1].weight.any() and not layers[-1].bias.any()
  bundle = loop.initialization_bundle()
  for actor in bundle["actors"].values():
    assert actor["architecture"]["film_hidden_dims"] == list(widths)
  loop.load_initialization(bundle)
  loop.run_step()
  starts = loop.policy.sample_starts()
  loop.policy.update_from(starts, *loop.policy.draw_noises(starts))
  state = loop.state_dict()
  restored_cfg = load_runner_cfg(
    {"runner_cfg": json.loads(json.dumps(state["runner_config"]))}
  )
  assert restored_cfg.film_hidden_dims == widths
  restored = v8_training(group, film_hidden_dims=widths)
  restored.load_state_dict(state)
  noises = loop.policy.draw_noises(starts)
  torch.testing.assert_close(
    noises, restored.policy.draw_noises(starts), rtol=0, atol=0
  )
  torch.testing.assert_close(restored.actor.state_dict(), loop.actor.state_dict())
  equal_tree(restored.policy.state_dict(), loop.policy.state_dict())
  default = v8_training(group)
  with pytest.raises(ValueError, match="architecture"):
    load_actor_checkpoint(default.actor, default.runner.cfg, bundle["actors"][group])


@pytest.mark.parametrize("widths", [(), (0,), (-1, 8), (True, 8)])
def test_film_hidden_dimensions_reject_invalid_layout(widths):
  from supermimic.runners.supermimic_runner import SuperMimicRunnerCfg

  with pytest.raises(ValueError, match="FiLM hidden dimensions"):
    SuperMimicRunnerCfg(film_hidden_dims=widths).validate_estimator_inputs()
