"""Step-1 runner collection, interface parity and checkpoint contracts."""

from __future__ import annotations

import copy
import io
import json
from dataclasses import asdict, replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from tensordict import TensorDict

from supermimic.estimator_inputs import EITInputConfig, EpisodeAlignment, PosteriorInput
from supermimic.replay_buffer import ReplayBuffer
from supermimic.runners.run_config import load_runner_cfg
from supermimic.runners.supermimic_runner import (
  InputStagePolicy,
  SuperMimicRunner,
  SuperMimicRunnerCfg,
  WorldModel,
  actor_checkpoint,
  build_ablation_actors,
  build_actor,
  load_actor_checkpoint,
)
from supermimic.utils.helpers import normalized_action_limits
from tests.test_estimator_world_model_env import (
  BATCH,
  _batch,
  _env,
  _MeasurementPosterior,
)


def _config(group="C", mode="train"):
  return SuperMimicRunnerCfg.for_estimator_inputs(
    estimator_inputs=EITInputConfig(group=group, mode=mode),
    actor_hidden_dims=(24, 12),
    dynamics_hidden_dims=(16,),
  )


def _runner(group="C", mode="train", provider=None):
  imagined = _env(group=group, mode=mode)
  batch = _batch()
  runner = object.__new__(SuperMimicRunner)
  runner.cfg = _config(group, mode)
  runner.device = "cpu"
  runner._representation = imagined.representation
  runner._representation.configure_estimator_measurement(
    runner.cfg.estimator_inputs.measurement_noise
  )
  runner._sim_context = SimpleNamespace(
    output_dim=1, read=lambda _: torch.zeros(BATCH, 1)
  )
  runner._wm = WorldModel(
    runner._representation, imagined.dynamics, runner._sim_context, 700, 29
  )
  runner._num_actions = 29
  runner._num_goals = 700
  runner.wm_env = imagined
  runner.motion = imagined.motion
  runner.motion_cmd = SimpleNamespace(
    motion=runner.motion,
    time_steps=batch["motion_time"].clone(),
    reference_alignment=EpisodeAlignment(
      batch["alignment_rotation"].clone(), batch["alignment_translation"].clone()
    ),
  )
  runner._sim_history = batch["sensor_window"].clone()
  runner._measurement_reset = torch.ones(BATCH, dtype=torch.bool)
  runner._posterior_ready = False
  runner._current_posterior = None
  runner.posterior_provider = provider
  return runner


def _build_current(runner, **kwargs):
  batch = _batch()
  rep = runner._representation
  return runner._build_sim_obs_td(
    rep.represent(batch["state"]),
    rep.extract_aux_state(batch["state"]),
    batch["actor_obs"],
    batch["motion_time"],
    batch["sim_ctx"],
    **kwargs,
  )


@pytest.mark.parametrize("group", ["A", "B", "C"])
def test_collection_and_actual_imagined_adapter_share_identical_features(group):
  provider = _MeasurementPosterior()
  runner = _runner(group, provider=provider)
  runner.wm_env.posterior_provider = _MeasurementPosterior()
  before = torch.random.get_rng_state()
  collected = _build_current(runner)
  imagined = runner.wm_env.reset(_batch())
  assert torch.equal(torch.random.get_rng_state(), before)
  assert torch.equal(collected["history"], imagined["history"])
  assert torch.equal(collected["goal_motion"], imagined["goal_motion"])
  if group == "C":
    assert torch.equal(collected["task_uncertainty"], imagined["task_uncertainty"])
  else:
    assert "task_uncertainty" not in collected.keys()
  _build_current(runner)
  assert len(provider.calls) == 1  # Same measurement must not advance the filter twice.


@pytest.mark.parametrize("group,mode", [("A", "eval"), ("B", "train"), ("C", "train")])
def test_collection_requires_current_posterior_without_truth_fallback(group, mode):
  with pytest.raises(ValueError, match="[Pp]osterior"):
    _build_current(_runner(group, mode))


def test_c_collection_requires_covariance_but_b_does_not():
  def mean_only(**kwargs):
    return PosteriorInput(_MeasurementPosterior()(**kwargs).mean)

  with pytest.raises(ValueError, match="covariance"):
    _build_current(_runner("C", provider=mean_only))
  assert _build_current(_runner("B", provider=mean_only))["goal_motion"].shape == (
    BATCH,
    700,
  )


def test_failed_posterior_input_can_retry_the_same_measurement():
  runner = _runner("B")
  history = runner._sim_history.clone()
  before = torch.random.get_rng_state()
  with pytest.raises(ValueError, match="[Pp]osterior"):
    _build_current(runner)
  assert not runner._posterior_ready
  posterior = _MeasurementPosterior()(
    measurement=history[:, -1, :64],
    previous_action=history[:, -1, 64:],
    reset_mask=runner._measurement_reset,
  )
  obs = _build_current(runner, posterior=posterior)
  assert runner._posterior_ready
  assert torch.equal(obs["history"], history)
  assert torch.equal(torch.random.get_rng_state(), before)


class _PhysicalEnv:
  """Deterministic step/reset adapter; the runner owns sampling and history."""

  def __init__(self, runner):
    self.num_envs = BATCH
    self.unwrapped = self
    self.clip_actions = 0.25
    self.action_manager = SimpleNamespace(action=torch.zeros(BATCH, 29))
    self.command_manager = SimpleNamespace(get_term=lambda _: runner.motion_cmd)
    self.termination_manager = SimpleNamespace(
      terminated=torch.zeros(BATCH, dtype=torch.bool)
    )
    self.state = _batch()["state"].clone()
    self.runner = runner
    self.actions = []
    self.reset_calls = 0

  def reset(self):
    self.reset_calls += 1
    self.state = _batch()["state"].clone()
    self.action_manager.action.zero_()
    self.runner.motion_cmd.time_steps.zero_()
    return self.get_observations(), {}

  def get_observations(self):
    rep = self.runner._representation
    state = self.state.clone()
    rep_state = rep.represent(state)
    sensor = rep.compute_actor_sensor(
      rep_state,
      torch.zeros(BATCH, 1),
      aux_state=rep.extract_aux_state(state),
      joint_state=rep.joint_state(rep_state),
      last_action=self.action_manager.action.clone(),
    )
    return TensorDict({"full_state": state, "actor": sensor}, batch_size=[BATCH])

  def step(self, actions):
    self.actions.append(actions.clone())
    self.action_manager.action.copy_(actions)
    self.state[:, 0] += 0.01
    command = self.runner.motion_cmd
    command.time_steps.add_(1).clamp_(max=9)
    dones = torch.zeros(BATCH, dtype=torch.bool)
    if len(self.actions) == 1:
      dones[0] = True
      self.state[0] = _batch()["state"][0]
      self.action_manager.action[0] = 0.0
      command.time_steps[0] = 0
      command.reference_alignment.translation[0] += torch.tensor([10.0, 20.0, 0.0])
    self.termination_manager.terminated = dones.clone()
    return self.get_observations(), torch.ones(BATCH), dones, {}


@pytest.mark.parametrize("failure", ["sensor", "before_insert", "after_insert"])
def test_post_physics_retry_commits_the_executed_transition_once(failure, monkeypatch):
  provider = _MeasurementPosterior()
  runner = _runner("B", provider=provider)
  runner.env = _PhysicalEnv(runner)
  runner.action_manager = runner.env.action_manager
  runner.replay_buffer = ReplayBuffer(runner._buffer_schema(), 2, BATCH)
  runner._pending_norm_updates = []
  runner.world_model_learner = MagicMock()
  runner.logger = MagicMock()
  runner._sampler = None
  actor = build_actor(runner.cfg, runner._wm, "cpu")
  runner.policy_learner = InputStagePolicy(actor, runner.cfg.policy)
  runner.reset_collection()
  history = runner._sim_history.clone()
  initial_ids = runner._episode_id.clone()
  sample = MagicMock(wraps=runner._representation.add_proprio_noise)
  monkeypatch.setattr(runner._representation, "add_proprio_noise", sample)
  insert = runner.replay_buffer.insert
  if failure == "sensor":
    sample.side_effect = ValueError("sensor")
  else:

    def fail(*args, **kwargs):
      if failure == "after_insert":
        insert(*args, **kwargs)
      raise ValueError(failure)

    monkeypatch.setattr(runner.replay_buffer, "insert", fail)
  with pytest.raises(ValueError, match=failure):
    runner.step()
  assert len(runner.env.actions) == 1
  assert len(provider.calls) == 1
  assert torch.equal(history, runner._sim_history)
  assert runner.replay_buffer.insert_count == int(failure == "after_insert")
  sample.side_effect = None
  monkeypatch.setattr(runner.replay_buffer, "insert", insert)
  rng = torch.random.get_rng_state()
  assert runner.step() == BATCH
  assert sample.call_count == 1 + int(failure == "sensor")
  if failure != "sensor":
    assert torch.equal(rng, torch.random.get_rng_state())
  assert (
    len(runner.env.actions)
    == len(provider.calls)
    == runner.replay_buffer.insert_count
    == 1
  )
  assert runner.logger.process_env_step.call_count == 1
  stored = runner.replay_buffer._storage[:, 0]
  assert torch.equal(stored["action"], runner.env.actions[0])
  assert torch.equal(stored["actor_obs"], history[:, -1])
  assert torch.equal(stored["episode_id"], initial_ids)
  assert runner._time_index.tolist() == [0, 1, 1]
  runner.step()
  assert runner.replay_buffer._storage["time_index"][:, 1].tolist() == [0, 1, 1]


def test_two_real_collection_steps_share_measurements_actions_and_reset_history():
  torch.manual_seed(91)
  provider = _MeasurementPosterior()
  runner = _runner(provider=provider)
  runner.env = _PhysicalEnv(runner)
  runner.action_manager = runner.env.action_manager
  # The light physical adapter exposes wrapper clipping only; use the production
  # limit resolver, without fabricating a MuJoCo action term.
  runner.wm_env._action_lower, runner.wm_env._action_upper = normalized_action_limits(
    SimpleNamespace(clip_actions=0.25, unwrapped=SimpleNamespace(action_manager=None)),
    29,
  )
  runner.replay_buffer = ReplayBuffer(runner._buffer_schema(), 8, BATCH)
  runner._pending_norm_updates = []
  runner.world_model_learner = SimpleNamespace(update_normalization=MagicMock())
  runner.logger = SimpleNamespace(process_env_step=MagicMock())
  runner._sampler = None
  actor = build_actor(runner.cfg, runner._wm, "cpu")
  with torch.no_grad():
    actor.decoder[-1].weight.zero_()
    actor.decoder[-1].bias.fill_(5.0)
  captured = []
  actor.register_forward_pre_hook(lambda _, args: captured.append(args[0].clone()))
  runner.cfg.policy.noise = 0.02
  runner.policy_learner = InputStagePolicy(actor, runner.cfg.policy)
  noise_draws = []
  sample = runner._representation.add_proprio_noise

  def counted(frame):
    sampled = sample(frame)
    noise_draws.append(sampled.clone())
    return sampled

  runner._representation.add_proprio_noise = counted
  runner._posterior_ready = True
  runner._current_posterior = object()
  runner.reset_collection()
  assert runner.env.reset_calls == 1
  assert not runner._posterior_ready and runner._current_posterior is None
  assert runner._measurement_reset.all()
  assert len(provider.calls) == 0  # Padding/reset itself never advances the filter.
  first_alignment = runner.motion_cmd.reference_alignment.translation.clone()
  assert runner.step() == BATCH
  after_reset = runner._sim_history.clone()
  assert runner.step() == BATCH
  # Initial measurement plus one per subsequent time point.
  assert len(noise_draws) == 3
  assert len(provider.calls) == 2
  assert len(captured) == 2
  stored = runner.replay_buffer._storage
  for step in range(2):
    measurement, previous_action, reset_mask = provider.calls[step]
    assert torch.equal(measurement, captured[step]["history"][:, -1, :64])
    assert torch.equal(previous_action, captured[step]["history"][:, -1, 64:])
    assert torch.equal(stored["actor_obs"][:, step], captured[step]["history"][:, -1])
    assert torch.equal(stored["prev_action"][:, step], previous_action)
    assert torch.equal(stored["action"][:, step], runner.env.actions[step])
    assert torch.equal(runner.env.actions[step], torch.full((BATCH, 29), 0.25))
    assert torch.equal(stored["actor_obs"][:, step], noise_draws[step])
  assert provider.calls[0][2].all()
  assert provider.calls[1][2].tolist() == [True, False, False]
  assert torch.equal(after_reset[0], after_reset[0, :1].expand_as(after_reset[0]))
  assert torch.count_nonzero(after_reset[0, :, 64:]) == 0
  assert torch.equal(after_reset[1:, -1, 64:], runner.env.actions[0][1:])
  assert torch.equal(stored["alignment_translation"][:, 0], first_alignment)
  assert not torch.equal(stored["alignment_translation"][0, 1], first_alignment[0])
  assert stored["terminated"][:, 0].tolist() == [True, False, False]
  assert runner.world_model_learner.update_normalization.call_count == 2


def test_json_roundtrip_preserves_complete_input_configuration():
  cfg = _config("B")
  raw = json.loads(json.dumps(asdict(cfg)))
  restored = load_runner_cfg({"runner_cfg": raw})
  restored.validate_estimator_inputs()
  assert restored == cfg
  assert restored.estimator_inputs.critic_uses_belief
  assert not restored.estimator_inputs.uses_uncertainty


@pytest.mark.parametrize("key", ["version", "covariance_encoding", "measurement_noise"])
def test_serialized_input_config_cannot_silently_fill_semantic_defaults(key):
  raw = json.loads(json.dumps(asdict(_config())))
  del raw["estimator_inputs"][key]
  with pytest.raises(ValueError):
    load_runner_cfg({"runner_cfg": raw})


def test_ablation_factory_copies_common_weights_without_shared_storage():
  runner = _runner()
  actors = build_ablation_actors(runner.cfg, runner._wm, "cpu")
  for group in ("B", "C"):
    for name in ("obs_normalizer", "goal_normalizers", "encoders", "decoder"):
      first = getattr(actors["A"], name).state_dict()
      other = getattr(actors[group], name).state_dict()
      for key in first:
        assert torch.equal(first[key], other[key])
        assert first[key].data_ptr() != other[key].data_ptr()
  assert actors["A"].film is None and actors["B"].film is None
  assert actors["C"].film is not None


def test_versioned_actor_checkpoint_roundtrip_and_explicit_eval_mode():
  runner = _runner(provider=_MeasurementPosterior())
  actor = build_actor(runner.cfg, runner._wm, "cpu")
  obs = _build_current(runner)
  actor.update_normalization(obs)
  actor.eval()
  stream = io.BytesIO()
  torch.save(actor_checkpoint(actor, runner.cfg), stream)
  stream.seek(0)
  checkpoint = torch.load(stream, weights_only=True)
  restored = build_actor(runner.cfg, runner._wm, "cpu").eval()
  load_actor_checkpoint(restored, runner.cfg, checkpoint)
  assert torch.equal(actor(obs), restored(obs))
  eval_cfg = replace(
    runner.cfg, estimator_inputs=replace(runner.cfg.estimator_inputs, mode="eval")
  )
  with pytest.raises(ValueError, match="configuration"):
    load_actor_checkpoint(restored, eval_cfg, checkpoint)
  load_actor_checkpoint(restored, eval_cfg, checkpoint, for_evaluation=True)
  assert torch.equal(actor(obs), restored(obs))


@pytest.mark.parametrize(
  "mutation", ["group", "noise", "version", "architecture", "legacy"]
)
def test_incompatible_checkpoint_is_rejected_before_loading_parameters(mutation):
  runner = _runner()
  actor = build_actor(runner.cfg, runner._wm, "cpu")
  checkpoint = copy.deepcopy(actor_checkpoint(actor, runner.cfg))
  if mutation == "group":
    checkpoint["input_config"]["group"] = "B"
  elif mutation == "noise":
    checkpoint["input_config"]["measurement_noise"]["gravity_std"] *= 2
  elif mutation == "version":
    checkpoint["input_config"]["version"] = "unknown"
  elif mutation == "architecture":
    checkpoint["architecture"]["latent_dim"] += 1
  else:
    checkpoint = checkpoint["actor"]
  before = {name: value.clone() for name, value in actor.state_dict().items()}
  with pytest.raises(ValueError):
    load_actor_checkpoint(actor, runner.cfg, checkpoint)
  assert all(
    torch.equal(value, actor.state_dict()[name]) for name, value in before.items()
  )


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("mutation", ["missing", "extra", "shape", "dtype", "nan"])
def test_invalid_actor_weights_leave_all_parameters_and_normalizers_unchanged(
  legacy, mutation
):
  runner = _runner()
  cfg = replace(runner.cfg, estimator_inputs=None) if legacy else runner.cfg
  source = build_actor(cfg, runner._wm, "cpu")
  target = build_actor(cfg, runner._wm, "cpu")
  for name, buffer in source.named_buffers():
    if name.endswith("_mean"):
      buffer.add_(0.1)
  checkpoint = copy.deepcopy(actor_checkpoint(source, cfg))
  state = checkpoint if legacy else checkpoint["actor"]
  key = next(k for k in state if k.startswith("decoder.") and k.endswith("weight"))
  before = copy.deepcopy(target.state_dict())
  assert not torch.equal(state[key], before[key])
  if mutation == "missing":
    del state[key]
  elif mutation == "extra":
    state["unexpected"] = torch.zeros(1)
  elif mutation == "shape":
    state[key] = state[key][:-1]
  elif mutation == "dtype":
    state[key] = state[key].double()
  else:
    state[key].flatten()[0] = float("nan")
  with pytest.raises(ValueError, match="Actor checkpoint"):
    load_actor_checkpoint(target, cfg, checkpoint)
  for name, value in target.state_dict().items():
    torch.testing.assert_close(value, before[name], atol=0, rtol=0)


def test_input_stage_cannot_silently_start_incomplete_policy_training():
  runner = _runner()
  with pytest.raises(NotImplementedError, match="step-1"):
    runner.run()
  policy = InputStagePolicy(
    build_actor(runner.cfg, runner._wm, "cpu"), runner.cfg.policy
  )
  with pytest.raises(NotImplementedError, match="later stage"):
    policy.update()
