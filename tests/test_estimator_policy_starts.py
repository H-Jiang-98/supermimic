"""Versioned policy starts reuse recorded measurements across K trajectories."""

from dataclasses import replace
from unittest.mock import Mock

import pytest
import torch

from supermimic.critic_inputs import CriticInputConfig
from supermimic.envs import world_model_env
from tests.test_estimator_belief_integration import setup_env


def test_unique_prefix_then_repeat_without_refusion_and_variable_batch():
  env, _, _ = setup_env()
  provider = env.posterior_provider
  provider.estimator.correct = Mock(wraps=provider.estimator.correct)
  start = env.prepare_policy_starts(torch.tensor([0, 0, 1]), torch.tensor([0, 0, 0]))
  assert provider.estimator.correct.call_count == 2
  assert provider.state is None
  expanded = start.repeat_interleave(2)
  assert len(expanded) == 6
  assert expanded.remaining_steps.tolist() == [10, 10, 10, 10, 6, 6]
  rng = torch.random.get_rng_state()
  obs = env.reset_prepared(expanded.index([0, 1]))
  assert env.num_envs == 2
  assert provider.estimator.correct.call_count == 2
  assert torch.equal(rng, torch.random.get_rng_state())
  torch.testing.assert_close(obs["history"][0], obs["history"][1])
  torch.testing.assert_close(
    provider.state.belief.covariance[0], provider.state.belief.covariance[1]
  )
  assert provider.state.belief.covariance.grad_fn is None
  env.clear_rollout()
  assert provider.state is None
  assert env._past_sensors is None
  env.reset_prepared(expanded.index([5]))
  assert env.num_envs == 1


def test_policy_start_sampler_filters_episodes_and_allows_last_record():
  env, _, _ = setup_env()
  start = env.sample_policy_starts(7, [2])
  assert start.batch["episode_id"].tolist() == [2] * 7
  assert start.remaining_steps.tolist() == [1] * 7
  assert start.batch["time_index"].tolist() == [0] * 7
  with pytest.raises(ValueError, match="No training frames"):
    env.sample_policy_starts(1, [])


def test_stale_prepared_start_rejected_without_mutating_live_frame():
  env, _, _ = setup_env()
  start = env.sample_policy_starts(2, [0, 1])
  env.reset_prepared(start)
  before = env.posterior_provider.state
  with torch.no_grad():
    next(env.dynamics.parameters()).add_(0.01)
  with pytest.raises(ValueError, match="Stale"):
    env.reset_prepared(start)
  assert env.posterior_provider.state is before
  assert env.num_envs == 2


def test_readonly_policy_motion_hashes_once_across_preparations_and_resets(monkeypatch):
  env, _, _ = setup_env()
  digest = Mock(wraps=world_model_env.motion_content_digest)
  monkeypatch.setattr(world_model_env, "motion_content_digest", digest)
  with env.readonly_policy_motion():
    starts = env.prepare_policy_starts(torch.tensor([0, 1]), torch.tensor([0, 0]))
    with env.readonly_policy_motion():
      env.reset_prepared(starts.index(0))
    env.clear_rollout()
    env.reset_prepared(starts.index(1))
  assert digest.call_count == 1

  env.motion.body_pos_w[0, 0, 0] += 1
  with pytest.raises(ValueError, match="Stale rollout start"):
    env.reset_prepared(starts)
  assert digest.call_count == 2


@pytest.mark.parametrize("replace_tensor", [False, True])
def test_motion_mutation_is_rejected_during_policy_phase(replace_tensor):
  env, _, _ = setup_env()
  with pytest.raises(ValueError, match="read-only policy"):
    with env.readonly_policy_motion():
      starts = env.prepare_policy_starts(torch.tensor([0]), torch.tensor([0]))
      env.reset_prepared(starts)
      if replace_tensor:
        env.motion.joint_pos = env.motion.joint_pos.clone()
      else:
        env.motion.joint_pos[0, 0] += 1
      env.step(torch.zeros(1, 29))
      pytest.fail("Policy step proceeded with a changed motion")
  # A new phase may use the updated motion after an exceptional exit.
  with env.readonly_policy_motion():
    env.reset_prepared(env.prepare_policy_starts(torch.tensor([0]), torch.tensor([0])))


def test_finished_clip_gets_zero_remaining_time_and_cannot_step_again():
  env, _, _ = setup_env()
  env.eit_critic = True
  env.dynamics.requires_grad_(False)
  start = env.sample_policy_starts(1, [2])
  obs = env.reset_prepared(start)
  assert obs["critic"].shape == (1, 1292)
  assert obs["critic"][0, 1233] == pytest.approx(0.02)
  obs, _, _, _ = env.step(
    torch.zeros(1, 29),
    process_noise=torch.zeros(1, 70),
    measurement_noise=torch.zeros(1, 64),
  )
  assert obs["critic"][0, 1233] == 0
  assert env.motion_times.tolist() == [9]
  with pytest.raises(ValueError, match="Reference clip ended"):
    env.step(torch.zeros(1, 29))


@pytest.mark.parametrize(
  "field", ["state", "sensor_window", "motion_time", "segment_time", "remaining_steps"]
)
def test_invalid_prepared_frame_preserves_live_provider_and_batch_size(field):
  env, _, _ = setup_env()
  env.eit_critic = True
  start = env.sample_policy_starts(2, [0])
  original = env.reset_prepared(start)
  provider_state = env.posterior_provider.state
  changed = start.index([0])
  if field == "remaining_steps":
    changed = replace(changed, remaining_steps=changed.remaining_steps.float())
  elif field == "motion_time":
    changed.batch[field].fill_(1000)
  elif field == "segment_time":
    changed.batch[field].fill_(10)
  else:
    changed.batch[field].flatten()[10] = float("nan")
  with pytest.raises(ValueError):
    env.reset_prepared(changed)
  assert env.num_envs == 2
  assert env.posterior_provider.state is provider_state
  for name, value in original.items():
    torch.testing.assert_close(env.get_observations()[name], value)


def test_eit_critic_reset_preflight_precedes_provider_commit(monkeypatch):
  env, _, _ = setup_env()
  env.eit_critic = True
  start = env.sample_policy_starts(2, [0])
  original = env.reset_prepared(start)
  provider_state = env.posterior_provider.state
  build = env._build_eit_critic

  def reject(*args, **kwargs):
    build(*args, **kwargs)
    raise ValueError("test critic rejection")

  with monkeypatch.context() as context:
    context.setattr(env, "_build_eit_critic", reject)
    with pytest.raises(ValueError, match="test critic rejection"):
      env.reset_prepared(start.index([0]))
  assert env.num_envs == 2
  assert env.posterior_provider.state is provider_state
  for name, value in original.items():
    torch.testing.assert_close(env.get_observations()[name], value)


def test_eit_critic_step_preflight_preserves_transition_for_retry(monkeypatch):
  env, _, _ = setup_env()
  env.eit_critic = True
  env.dynamics.requires_grad_(False)
  original = env.reset_prepared(env.sample_policy_starts(1, [0]))
  provider_state = env.posterior_provider.state
  step = Mock(wraps=env._wm_step)
  monkeypatch.setattr(env, "_wm_step", step)
  env.critic_input_config = CriticInputConfig(group="A")
  action = torch.zeros(1, 29)
  with pytest.raises(ValueError, match="disagree"):
    env.step(action)
  assert env.posterior_provider.state is provider_state
  assert env._pending_belief_step is not None
  env.critic_input_config = CriticInputConfig(group="C")
  for name, value in original.items():
    torch.testing.assert_close(env.get_observations()[name], value)
  env.step(action)
  assert step.call_count == 1
  assert env._pending_belief_step is None


def test_finished_snapshot_restore_cannot_add_another_scoring_action():
  env, _, _ = setup_env()
  env.eit_critic = True
  env.dynamics.requires_grad_(False)
  env.reset_prepared(env.sample_policy_starts(1, [2]))
  final, _, _, _ = env.step(torch.zeros(1, 29))
  snapshot = env.snapshot()
  env.reset_prepared(env.sample_policy_starts(1, [0]))
  restored = env.restore(snapshot)
  for name, value in final.items():
    torch.testing.assert_close(restored[name], value)
  assert restored["critic"][0, 1233] == 0
  with pytest.raises(ValueError, match="Reference clip ended"):
    env.step(torch.zeros(1, 29))


@pytest.mark.parametrize("change", ["scale", "legacy_version"])
def test_eit_critic_snapshot_contract_rejects_encoding_changes(change):
  env, _, _ = setup_env()
  env.eit_critic = True
  env.reset_prepared(env.sample_policy_starts(1, [0]))
  snapshot = env.snapshot()
  state = env.posterior_provider.state
  if change == "scale":
    env.critic_input_config = CriticInputConfig(group="C", remaining_time_scale=2.0)
  else:
    snapshot["critic_config"]["version"] = "unknown_critic_version"
  with pytest.raises(ValueError, match="critic configuration"):
    env.restore(snapshot)
  assert env.posterior_provider.state is state
