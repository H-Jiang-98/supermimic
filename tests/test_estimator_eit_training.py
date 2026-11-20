"""Alternating EIT phases, normalization splits and transactional continuation."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from supermimic.learners.belief_policy_learner import BeliefPolicyLearnerCfg
from supermimic.replay_buffer import FieldSpec, ReplayBuffer
from supermimic.runners.eit_training import EITTraining, EITTrainingCfg
from tests.test_estimator_belief_training import _RoundRunner, assert_tree
from tests.test_estimator_world_model_env import _env


def training(seed=31):
  runner = _RoundRunner(seed)
  runner.cfg.critic_hidden_dims = (12,)
  runner._sim_context = SimpleNamespace(output_dim=2)
  template = runner.template
  template["motion_time"] = torch.arange(len(template))
  template["alignment_rotation"] = (
    torch.eye(3, dtype=torch.float64).expand(len(template), 3, 3).clone()
  )
  template["alignment_translation"] = torch.zeros(len(template), 3, dtype=torch.float64)
  runner._schema = {
    key: FieldSpec(tuple(value.shape[1:]), value.dtype)
    for key, value in template.items()
  }
  runner.replay_buffer = ReplayBuffer(
    runner._schema, 8, 2, initial_prior=runner._wm.initial_prior
  )
  env = _env("B")
  # All simulated measurements/replay/prefix/model interfaces stay real; only
  # physics and reward use the deterministic stream and analytic test function.
  runner._representation.compute_wm_reward_from_reference = (
    env.representation.compute_wm_reward_from_reference
  )
  env.dynamics, env.representation = runner._dynamics, runner._representation
  env.replay_buffer = runner.replay_buffer
  env.estimator_input_config = runner.cfg.estimator_inputs
  env.motion = runner.motion
  for name, value in vars(env.motion).items():
    if isinstance(value, torch.Tensor) and value.is_floating_point():
      setattr(env.motion, name, value.double())
  runner.wm_env = env
  return EITTraining(
    runner,
    EITTrainingCfg(
      collection_steps=4,
      validation_stride=2,
      validation_batch_size=2,
      normalization_batch_size=2,
    ),
    BeliefPolicyLearnerCfg(
      batch_size=2,
      num_samples=2,
      microbatch_size=2,
      horizon=2,
      critic_epochs=1,
      noise=0,
    ),
  )


def test_two_rounds_update_all_components_and_exclude_validation_normalization():
  loop = training()
  model = deepcopy(loop.runner._dynamics.state_dict())
  actor = deepcopy(loop.actor.state_dict())
  critic = deepcopy(loop.policy.critic.state_dict())
  normalized = []
  original = loop.update_input_normalization

  def normalize(frames):
    normalized.extend(frames["episode_id"].tolist())
    return original(frames)

  with patch.object(loop, "update_input_normalization", side_effect=normalize):
    first, second = loop.run_round(), loop.run_round()
  assert all(episode % 2 for episode in normalized)
  assert len(normalized) == 8
  assert second["collection_actor_version"] == first["trained_actor_version"]
  assert second["collection_model_version"] == first["trained_model_version"]
  assert first["collection_actor_version"] != first["trained_actor_version"]
  assert loop.learner.steps == loop.policy.actor_steps == loop.policy.critic_steps == 2
  for before, module in (
    (model, loop.runner._dynamics),
    (actor, loop.actor),
    (critic, loop.policy.critic),
  ):
    assert any(
      not torch.equal(value, module.state_dict()[key]) for key, value in before.items()
    )
  assert int(loop.actor.obs_normalizer.count) == 8
  assert loop.runner.posterior_provider.state is None
  assert loop.runner.imagined_posterior_provider.state is None


@pytest.mark.parametrize(
  "component",
  ["critic", "policy_optimizer", "model_optimizer", "actor", "rng", "config"],
)
def test_failed_restore_changes_no_component_or_rng(component):
  source, target = training(31), training(81)
  source.run_round()
  target.run_round()
  checkpoint = source.state_dict()
  if component == "critic":
    checkpoint["critic"]["state_dict"]["mlp.0.weight"][0, 0] = float("nan")
  elif component == "policy_optimizer":
    checkpoint["policy"]["actor_optimizer"]["state"][0]["exp_avg"] = torch.zeros(1)
  elif component == "model_optimizer":
    checkpoint["learner"]["optimizer"]["state"][0]["exp_avg"] = torch.zeros(1)
  elif component == "actor":
    checkpoint["actor"]["actor"]["decoder.0.weight"][0, 0] = float("nan")
  elif component == "rng":
    checkpoint["rng_cpu"] = torch.ones(2)
  else:
    checkpoint["policy_config"]["num_samples"] += 1
  before = target.state_dict()
  provider, replay = target.runner.posterior_provider, target.runner.replay_buffer
  with pytest.raises((ValueError, TypeError, RuntimeError)):
    target.load_state_dict(checkpoint)
  assert_tree(target.state_dict(), before)
  assert target.runner.posterior_provider is provider
  assert target.runner.replay_buffer is replay


def test_restore_reproduces_next_model_and_policy_updates(tmp_path):
  source, target = training(31), training(81)
  source.run_round()
  source.save(tmp_path / "training.pt")
  saved = torch.load(tmp_path / "training.pt", weights_only=True)
  target.load_state_dict(saved)
  assert target.rounds == source.rounds
  assert_tree(target.actor.state_dict(), source.actor.state_dict())
  assert_tree(target.policy.critic.state_dict(), source.policy.critic.state_dict())
  assert_tree(target.policy.state_dict(), source.policy.state_dict())
  # Identical raw physical inputs are copied into the new replay, as the public
  # restore deliberately starts a fresh physical stream rather than restoring it.
  target.runner.replay_buffer = deepcopy(source.runner.replay_buffer)
  target.runner.wm_env.replay_buffer = target.runner.replay_buffer
  keys = torch.tensor([1, 1])
  times = torch.tensor([1, 2])
  starts = source.runner.wm_env.prepare_policy_starts(keys, times)
  target_starts = target.runner.wm_env.prepare_policy_starts(keys, times)
  process = torch.randn(2, 2, 2, 70, dtype=torch.float64)
  sensor = torch.randn(2, 2, 2, 64, dtype=torch.float64)
  expected = source.policy.update_from(starts, process, sensor)
  actual = target.policy.update_from(target_starts, process, sensor)
  assert_tree(actual, expected)
  assert_tree(target.actor.state_dict(), source.actor.state_dict())
  assert_tree(target.policy.critic.state_dict(), source.policy.critic.state_dict())
  assert_tree(target.policy.state_dict(), source.policy.state_dict())


def test_initialization_copies_exact_weights_and_rejects_after_training():
  source, target = training(31), training(81)
  bundle = source.initialization_bundle()
  source.load_initialization(bundle)
  target.load_initialization(bundle)
  assert_tree(target.actor.state_dict(), source.actor.state_dict())
  assert_tree(target.policy.critic.state_dict(), source.policy.critic.state_dict())
  assert_tree(
    target.runner._dynamics.state_dict(), source.runner._dynamics.state_dict()
  )
  assert source.initialization_digest == target.initialization_digest
  assert all(
    a.data_ptr() != b.data_ptr()
    for a, b in zip(source.actor.parameters(), target.actor.parameters(), strict=True)
  )
  source.run_round()
  with pytest.raises(ValueError, match="active training"):
    source.load_initialization(bundle)


def test_model_only_checkpoint_cannot_masquerade_as_joint_resume():
  loop = training()
  state = loop.state_dict()
  state["version"] = "belief_model_training_v2"
  with pytest.raises(ValueError, match="explicit weight import"):
    loop.load_state_dict(state)


def test_no_fresh_training_frames_preserves_actor_normalizers():
  loop = training()
  loop.cfg = replace(loop.cfg, validation_stride=5)
  loop.run_round()
  loop.runner._episode_id = torch.tensor([10, 5])
  count = loop.actor.obs_normalizer.count.clone()
  loop.run_round()
  assert torch.equal(loop.actor.obs_normalizer.count, count)


@pytest.mark.parametrize(
  "component", ["unused_actor", "unused_critic", "missing_group"]
)
def test_initialization_preflights_all_groups_before_any_mutation(component):
  source, target = training(31), training(81)
  bundle = source.initialization_bundle()
  if component == "unused_actor":
    bundle["actors"]["C"]["actor"]["decoder.0.weight"][0, 0] = float("nan")
  elif component == "unused_critic":
    bundle["critics"]["C"]["state_dict"]["mlp.0.weight"][0, 0] = float("nan")
  else:
    bundle["actors"].pop("A")
  before = target.state_dict()
  with pytest.raises(ValueError):
    target.load_initialization(bundle)
  assert_tree(target.state_dict(), before)


def test_normalization_late_failure_preserves_all_live_statistics():
  loop = training()
  loop.refresh_collection()
  for _ in range(4):
    loop.runner.step(record_episode_stats=False)
  loop.runner.finish_collection_round()
  frames = loop.runner.replay_buffer.retained_frames()
  frames = frames[frames["episode_id"] == 1]
  actor = deepcopy(loop.actor.state_dict())
  critic = deepcopy(loop.policy.critic.state_dict())
  original = loop.runner.wm_env.reset_prepared
  calls = 0

  def reset(start):
    nonlocal calls
    result = original(start)
    calls += 1
    if calls == 2:
      result["goal_motion"][0, 0] = float("inf")
    return result

  with patch.object(loop.runner.wm_env, "reset_prepared", side_effect=reset):
    with pytest.raises(ValueError, match="Non-finite actor"):
      loop.update_input_normalization(frames)
  assert_tree(loop.actor.state_dict(), actor)
  assert_tree(loop.policy.critic.state_dict(), critic)
  assert loop.runner.imagined_posterior_provider.state is None


def test_joint_output_resume_reconciles_metrics_and_preserves_unrelated_directory(
  tmp_path,
):
  loop = training()
  first = loop.run(1, tmp_path / "run")[0]
  checkpoint = tmp_path / "run" / "training.pt"
  resumed = training(81)
  second = resumed.run(1, tmp_path / "new", resume=checkpoint)[0]
  assert second["round"] == first["round"] + 1
  assert second["collection_actor_version"] == first["trained_actor_version"]
  unrelated = tmp_path / "unrelated"
  unrelated.mkdir()
  marker = unrelated / "training.pt"
  marker.write_bytes(b"unrelated run")
  with pytest.raises(ValueError, match="its own current"):
    training().run(1, unrelated, resume=checkpoint)
  assert marker.read_bytes() == b"unrelated run"


def test_common_initialization_rejects_prefit_critic():
  loop = training()
  initial = loop.initialization_bundle()
  loop.policy.critic_steps = 1
  with pytest.raises(ValueError, match="before training"):
    loop.initialization_bundle()
  with pytest.raises(ValueError, match="active training"):
    loop.load_initialization(initial)
