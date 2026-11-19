"""Eager integration of estimator inputs with the actual imagined-env adapter."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import Mock

import pytest
import torch
from tensordict import TensorDict

from supermimic.envs.multi_task_world_model_env import MultiTaskWorldModelEnv
from supermimic.estimator_inputs import (
  EITInputConfig,
  EpisodeAlignment,
  PosteriorInput,
  RpvState,
  build_actor_features,
)
from supermimic.models import Dynamics, MultiEncoderAE
from supermimic.representations import GenCoordRepresentation
from supermimic.tasks.task_spec import TaskSpec
from tests.conftest import make_mock_buffer, make_mock_command_env

BATCH = 3
HISTORY = 5


class _RewardRepresentation(GenCoordRepresentation):
  """Real state/sensor/goal implementation; lightweight analytic test reward."""

  def compute_wm_reward_from_reference(self, rep, aux, joints, action, previous, *ref):
    return -(aux[:, :3] - ref[0][:, 0]).square().sum(-1) - action.square().mean(-1)


class _MeasurementPosterior:
  def __init__(self):
    self.calls = []

  def __call__(self, *, measurement, previous_action, reset_mask):
    self.calls.append((measurement, previous_action, reset_mask))
    rotation = torch.eye(3).expand(measurement.shape[0], 3, 3)
    mean = RpvState(rotation, measurement[:, :3], previous_action[:, :3])
    diagonal = 0.01 + measurement[:, :9].square() * 0.001
    return PosteriorInput(mean, torch.diag_embed(diagonal))


def _batch():
  state = torch.zeros(BATCH, 71)
  state[:, 2] = 0.8
  state[:, 3] = 1.0
  state[:, 36] = 0.2
  history = (
    torch.arange(BATCH * HISTORY * 93).reshape(BATCH, HISTORY, 93).float() / 1000
  )
  rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
  return TensorDict(
    {
      "state": state,
      "prev_action": history[:, -1, -29:].clone(),
      "sim_ctx": torch.zeros(BATCH, 1),
      "motion_time": torch.tensor([0, 4, 9]),
      "sensor_window": history,
      "actor_obs": history[:, -1].clone(),
      "pad_mask": torch.zeros(BATCH, HISTORY, dtype=torch.bool),
      "alignment_rotation": rotation.expand(BATCH, 3, 3).clone(),
      "alignment_translation": torch.tensor([[2.0, 3.0, 0.0]]).expand(BATCH, 3).clone(),
    },
    batch_size=[BATCH],
  )


def _env(group="C", provider=None, *, mode="train"):
  rep = _RewardRepresentation(29, torch.zeros(29))
  rep.actor_obs_mode = "proprio"
  rep.dt = 0.02
  rep.substep_coeff = 0.6
  physical_env = make_mock_command_env()
  physical_env.clip_actions = None
  physical_env.unwrapped.action_manager = None
  motion = physical_env.unwrapped.command_manager.get_term.return_value.motion
  motion.root_pos_w[:, 0] = torch.arange(10).float() * 0.1
  motion.root_lin_vel_w = torch.tensor([0.2, 0.0, 0.0]).expand(10, 3).clone()
  physical_env.unwrapped.step_dt = 0.02
  physical_env.cfg.rewards = {}
  dynamics = Dynamics(rep, 29, 1, torch.ones(29), torch.zeros(29), (16,))
  for normalizer in dynamics.normalizers():
    normalizer.eval()
  spec = TaskSpec("motion", 700, lambda *args: None)
  config = EITInputConfig(group=group, mode=mode)
  env = MultiTaskWorldModelEnv(
    dynamics,
    rep,
    physical_env,
    BATCH,
    make_mock_buffer(71),
    [spec],
    HISTORY,
    0.05,
    estimator_input_config=config,
    posterior_provider=provider,
  )
  return env


def test_reset_reuses_actual_measurements_without_sampling_and_matches_shared_builder():
  provider = _MeasurementPosterior()
  env = _env(provider=provider)
  batch = _batch()
  rng_before = torch.random.get_rng_state()
  obs = env.reset(batch)
  assert torch.equal(torch.random.get_rng_state(), rng_before)
  assert torch.equal(obs["history"], batch["sensor_window"])
  assert torch.equal(provider.calls[0][0], batch["sensor_window"][:, -1, :64])
  assert torch.equal(provider.calls[0][1], batch["prev_action"])
  assert provider.calls[0][2].all()
  reference = env.representation.reference_window(
    env.motion,
    batch["motion_time"],
    EpisodeAlignment(batch["alignment_rotation"], batch["alignment_translation"]),
  )
  goal, uncertainty = build_actor_features(
    env.estimator_input_config, None, env._posterior, reference
  )
  torch.testing.assert_close(obs["goal_motion"], goal)
  torch.testing.assert_close(obs["task_uncertainty"], uncertainty)
  assert obs["goal_motion"].shape == (BATCH, 700)
  assert obs["task_uncertainty"].shape == (BATCH, 450)


def test_two_action_steps_keep_causal_history_and_gradients():
  torch.manual_seed(3)
  provider = _MeasurementPosterior()
  env = _env(provider=provider)
  actor = MultiEncoderAE(
    93,
    29,
    env.task_specs,
    {"motion": 1.0},
    32,
    (24, 12),
    0.0,
    HISTORY,
    uncertainty_dim=450,
  )
  actor.eval()
  obs0 = env.reset(_batch())
  action0 = actor(obs0)
  action0.retain_grad()
  obs1, reward1, _, _ = env.step(action0)
  action1 = actor(obs1)
  obs2, reward2, _, _ = env.step(action1)
  assert torch.equal(obs1["history"][:, :-1], obs0["history"][:, 1:])
  assert torch.equal(obs2["history"][:, :-1], obs1["history"][:, 1:])
  assert torch.equal(obs1["history"][:, -1, -29:], action0)
  assert torch.equal(obs2["history"][:, -1, -29:], action1)
  assert len(provider.calls) == 3
  assert not provider.calls[1][2].any()
  assert torch.equal(provider.calls[1][0], obs1["history"][:, -1, :64])
  assert torch.equal(env.motion_times, torch.tensor([2, 6, 9]))
  loss = obs2["history"][:, -2].square().mean() + obs2["goal_motion"].square().mean()
  loss = loss - reward1.mean() - reward2.mean()
  loss.backward()
  assert action0.grad is not None and torch.isfinite(action0.grad).all()
  assert action0.grad.abs().sum() > 0
  assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in actor.parameters())
  assert all(p.grad is None or torch.isfinite(p.grad).all() for p in actor.parameters())


@pytest.mark.parametrize("group", ["B", "C"])
def test_estimated_actor_is_independent_of_latent_truth(group):
  env = _env(group=group, provider=_MeasurementPosterior())
  batch = _batch()
  first = env.reset(batch)
  changed = batch.clone()
  changed["state"][:, :3] += torch.tensor([10.0, -8.0, 2.0])
  changed["state"][:, 36:39] += 5.0
  second = env.reset(changed)
  torch.testing.assert_close(first["goal_motion"], second["goal_motion"])
  assert torch.equal(first["history"], second["history"])
  if group == "C":
    torch.testing.assert_close(first["task_uncertainty"], second["task_uncertainty"])
  else:
    assert "task_uncertainty" not in first.keys()
  assert not torch.equal(first["critic"], second["critic"])


def test_required_posterior_is_not_silently_replaced_or_reused():
  env = _env(group="B")
  with pytest.raises(ValueError, match="[Pp]osterior"):
    env.reset(_batch())
  posterior = _MeasurementPosterior()(
    measurement=_batch()["actor_obs"][:, :64],
    previous_action=_batch()["prev_action"],
    reset_mask=torch.ones(BATCH, dtype=torch.bool),
  )
  env.reset(_batch(), posterior=posterior)
  with pytest.raises(ValueError, match="[Pp]osterior"):
    env.step(torch.zeros(BATCH, 29))


@pytest.mark.parametrize(
  "group,mode,invalid",
  [
    ("A", "eval", "missing"),
    ("B", "train", "missing"),
    ("C", "train", "missing"),
    ("C", "train", "missing_covariance"),
    ("C", "train", "negative_covariance"),
  ],
)
def test_failed_step_preserves_rollout_and_retry_advances_once(group, mode, invalid):
  batch = _batch()
  good = _MeasurementPosterior()(
    measurement=batch["actor_obs"][:, :64],
    previous_action=batch["prev_action"],
    reset_mask=torch.ones(BATCH, dtype=torch.bool),
  )
  env = _env(group=group, mode=mode)
  before_obs = env.reset(batch, posterior=good).clone()
  before_aux, before_rep = env.aux_state.clone(), env.rep_state.clone()
  before_time, before_actions = env.motion_times.clone(), env.last_actions.clone()
  before_bodies = tuple(value.clone() for value in env._ref_body_state)
  before_rng = torch.random.get_rng_state()
  bad = None
  if invalid == "missing_covariance":
    bad = PosteriorInput(good.mean)
  elif invalid == "negative_covariance":
    bad = PosteriorInput(good.mean, -torch.eye(9).expand(BATCH, 9, 9))
  actions = torch.zeros(BATCH, 29, requires_grad=True)
  with pytest.raises(ValueError):
    env.step(actions, posterior=bad)
  assert torch.equal(torch.random.get_rng_state(), before_rng)
  assert torch.equal(env.aux_state, before_aux)
  assert torch.equal(env.rep_state, before_rep)
  assert torch.equal(env.motion_times, before_time)
  assert torch.equal(env.last_actions, before_actions)
  assert env._posterior is good
  for key in before_obs.keys():
    assert torch.equal(env.get_observations()[key], before_obs[key])
  for old, current in zip(before_bodies, env._ref_body_state, strict=True):
    assert torch.equal(old, current)
  expected_aux, expected_rep = env.dynamics(
    before_rep, actions, before_aux, env.sim_ctx
  )
  obs, reward, _, _ = env.step(actions, posterior=good)
  torch.testing.assert_close(env.aux_state, expected_aux)
  torch.testing.assert_close(env.rep_state, expected_rep)
  assert torch.equal(env.motion_times, torch.tensor([1, 5, 9]))
  assert torch.equal(obs["history"][:, :-1], before_obs["history"][:, 1:])
  (-reward.mean()).backward()
  assert actions.grad is not None and torch.isfinite(actions.grad).all()


def test_invalid_provider_output_does_not_commit_next_state():
  provider = _MeasurementPosterior()
  env = _env(group="C", provider=provider)
  before = env.reset(_batch()).clone()
  aux, rep, times = env.aux_state, env.rep_state, env.motion_times
  good = env._posterior
  env.posterior_provider = lambda **kwargs: None
  with pytest.raises(ValueError, match="posterior"):
    env.step(torch.zeros(BATCH, 29))
  assert env.aux_state is aux and env.rep_state is rep and env.motion_times is times
  assert env._posterior is good
  for key in before.keys():
    assert torch.equal(env.get_observations()[key], before[key])
  env.step(torch.zeros(BATCH, 29), posterior=good)
  assert torch.equal(env.motion_times, torch.tensor([1, 5, 9]))


def test_failed_reset_preserves_last_valid_rollout(monkeypatch):
  provider = _MeasurementPosterior()
  env = _env(provider=provider)
  before = env.reset(_batch()).clone()
  aux, rep, times = env.aux_state, env.rep_state, env.motion_times
  alignment, good = env.reference_alignment, env._posterior
  env.posterior_provider = None
  sample = Mock(side_effect=AssertionError("Invalid reset must not sample replay"))
  monkeypatch.setattr(env.replay_buffer, "sample_causal", sample)
  rng = torch.random.get_rng_state()
  with pytest.raises(ValueError, match="posterior"):
    env.reset()  # Reject before sampling a replay batch.
  sample.assert_not_called()
  changed = _batch()
  changed["state"][:, :3] += 50.0
  with pytest.raises(ValueError, match="covariance"):
    env.reset(changed, posterior=PosteriorInput(good.mean))
  assert torch.equal(torch.random.get_rng_state(), rng)
  assert env.aux_state is aux and env.rep_state is rep and env.motion_times is times
  assert env.reference_alignment is alignment and env._posterior is good
  for key in before.keys():
    assert torch.equal(env.get_observations()[key], before[key])


def test_a_uses_truth_in_training_and_requires_estimate_in_evaluation():
  env = _env(group="A")
  obs = env.reset(_batch())
  assert "task_uncertainty" not in obs.keys()
  env.estimator_input_config = replace(env.estimator_input_config, mode="eval")
  with pytest.raises(ValueError, match="[Pp]osterior"):
    env.reset(_batch())


def test_alignment_is_required_and_remains_fixed_across_steps():
  env = _env(group="A")
  batch = _batch()
  with pytest.raises(ValueError, match="alignment"):
    env.reset(batch.exclude("alignment_rotation"))
  env.reset(batch)
  original = env.reference_alignment
  env.step(torch.ones(BATCH, 29))
  assert env.reference_alignment is original
  # The mocked raw reference bodies are all zero: only the saved translation remains.
  torch.testing.assert_close(
    env._ref_body_state[0][:, 0], batch["alignment_translation"]
  )


def test_replay_action_contract_detects_misaligned_current_frame():
  env = _env(group="A")
  batch = _batch()
  batch["prev_action"] = torch.zeros(BATCH, 29)
  with pytest.raises(ValueError, match="prev_action"):
    env.reset(batch)


def test_imagined_dynamics_measurement_and_provider_use_limited_action():
  provider = _MeasurementPosterior()
  env = _env(provider=provider)
  env._action_lower.fill_(-0.1)
  env._action_upper.fill_(0.2)
  env.reset(_batch())
  actions = torch.linspace(-2.0, 2.0, BATCH * 29).reshape(BATCH, 29)
  executed = env.limit_actions(actions)
  expected_aux, expected_rep = env.dynamics(
    env.rep_state, executed, env.aux_state, env.sim_ctx
  )
  obs = env.step(actions)[0]
  torch.testing.assert_close(env.aux_state, expected_aux)
  torch.testing.assert_close(env.rep_state, expected_rep)
  assert torch.equal(env.last_actions, executed)
  assert torch.equal(obs["history"][:, -1, -29:], executed)
  assert torch.equal(provider.calls[-1][1], executed)
