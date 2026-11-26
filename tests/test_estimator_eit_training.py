"""Alternating EIT phases, normalization splits and transactional continuation."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from supermimic.critic_inputs import BeliefCritic
from supermimic.learners.belief_policy_learner import BeliefPolicyLearnerCfg
from supermimic.replay_buffer import FieldSpec, ReplayBuffer
from supermimic.runners.belief_training import parameter_digest
from supermimic.runners.eit_snapshots import file_sha256
from supermimic.runners.eit_training import (
  EITTraining,
  EITTrainingCfg,
  initialization_fingerprint,
)
from supermimic.runners.supermimic_runner import build_actor
from supermimic.sim_context import SimContext
from tests.test_estimator_belief_training import _RoundRunner, assert_tree
from tests.test_estimator_world_model_env import _env


def training(seed=31, *, buffer_size=8, replay_ratio=1):
  runner = _RoundRunner(seed, group="mean")
  runner.cfg.critic_hidden_dims = (12,)
  runner.cfg.buffer_size = buffer_size
  runner.cfg.replay_ratio = replay_ratio
  runner._sampler = None
  runner.task_id = "test-eit"
  runner.registry_name = None
  from dataclasses import make_dataclass

  from supermimic.utils.logger import Logger

  runner.env.cfg = make_dataclass("EnvCfg", [("seed", object)], namespace={})(None)
  runner.logger = Logger(None, 2, 10, "cpu")
  context_dim = runner._dynamics.sim_ctx_dim
  runner._sim_context = SimContext(
    [SimpleNamespace(dim=context_dim)], [(-1.0, 1.0)] * context_dim
  ).double()
  runner._wm.sim_context = runner._sim_context
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
    runner._schema, buffer_size, 2, initial_prior=runner._wm.initial_prior
  )
  env = _env("mean")
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
    EITTrainingCfg(validation_stride=2, validation_batch_size=2),
    BeliefPolicyLearnerCfg(
      batch_size=2,
      num_samples=2,
      microbatch_size=2,
      horizon=2,
      critic_epochs=1,
      critic_minibatches=1,
      noise=0,
    ),
  )


def advance(loop, checkpoint_path=None, steps=4):
  before_actor = parameter_digest(loop.actor)
  before_model = loop.runner._dynamics.model_revision
  rows = [loop.run_step() for _ in range(steps)]
  updates = [u for row in rows for u in row["updates"]]
  result = {
    **rows[-1],
    **(updates[-1] if updates else {}),
    "collection_actor_version": before_actor,
    "trained_actor_version": parameter_digest(loop.actor),
    "collection_model_version": before_model,
    "trained_model_version": loop.runner._dynamics.model_revision,
  }
  if updates:
    result["budget"] = {
      **updates[-1]["budget"],
      "normalization_samples": sum(row["normalization_samples"] for row in rows),
    }
  if checkpoint_path is not None:
    loop.save(checkpoint_path)
  return result


@pytest.mark.parametrize(
  "component",
  [
    "critic",
    "critic_version",
    "policy_optimizer",
    "model_optimizer",
    "actor",
    "rng",
    "config",
  ],
)
def test_failed_restore_changes_no_component_or_rng(component):
  source, target = training(31), training(81)
  advance(source)
  advance(target)
  checkpoint = source.state_dict()
  if component == "critic":
    checkpoint["critic"]["state_dict"]["mlp.0.weight"][0, 0] = float("nan")
  elif component == "critic_version":
    checkpoint["critic"]["contract"]["version"] = "unknown_critic_version"
  elif component == "policy_optimizer":
    checkpoint["policy"]["actor_optimizers"][0]["state"][0]["exp_avg"] = torch.zeros(1)
  elif component == "model_optimizer":
    checkpoint["learner"]["cov"]["optimizer"]["state"][0]["exp_avg"] = torch.zeros(1)
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


@pytest.mark.parametrize(
  "entry", ["runner", "provider", "initialization", "actor", "world_model_metadata"]
)
def test_zero_measurement_noise_rejected_before_loading(entry):
  loop = training()
  field = "joint_velocity_std"
  if entry == "initialization":
    checkpoint = loop.initialization_bundle()
    checkpoint["world_model"]["estimator"]["measurement_config"][field] = 0.0
    load = loop.load_initialization
  elif entry == "provider":
    loop.refresh_collection()
    loop.runner.step(record_episode_stats=False)
    provider = loop.runner.posterior_provider
    checkpoint = provider.state_dict()
    checkpoint["measurement_config"][field] = 0.0
    load = provider.load_state_dict
  elif entry == "world_model_metadata":
    checkpoint = loop.state_dict()
    checkpoint["world_model"]["estimator"]["measurement_config"][field] = 0.0
    load = loop._check_checkpoint_version
  else:
    advance(loop)
    checkpoint = loop.state_dict()
    inputs = (
      checkpoint["actor"]["input_config"]
      if entry == "actor"
      else checkpoint["runner_config"]["estimator_inputs"]
    )
    inputs["measurement_noise"][field] = 0.0
    load = loop.load_state_dict
  with pytest.raises(ValueError, match=field):
    load(checkpoint)


def test_restore_reproduces_next_model_and_policy_updates(tmp_path):
  source, target = training(31), training(81)
  advance(source)
  source.save(tmp_path / "training.pt")
  saved = torch.load(tmp_path / "training.pt", weights_only=True)
  target.load_state_dict(saved)
  assert target.update_groups == source.update_groups
  assert target.runner.wm_env.replay_buffer is target.runner.replay_buffer
  assert target.runner.replay_buffer.replay_mode == "full"
  assert target.runner.replay_buffer.num_transitions == 0
  assert_tree(target.actor.state_dict(), source.actor.state_dict())
  assert_tree(target.policy.critic.state_dict(), source.policy.critic.state_dict())
  assert_tree(target.policy.state_dict(), source.policy.state_dict())
  # Identical raw physical inputs are copied into the new replay, as the public
  # restore deliberately starts a fresh physical stream rather than restoring it.
  target.runner.replay_buffer = deepcopy(source.runner.replay_buffer)
  target.runner.wm_env.replay_buffer = target.runner.replay_buffer
  replay = source.runner.replay_buffer
  windows = replay.model_windows(replay.model_window_starts([1], 3), 3)
  pairs = replay.model_windows(replay.model_window_starts([1], 2), 2)
  assert_tree(target.learner.update_mean(windows), source.learner.update_mean(windows))
  assert_tree(
    target.learner.update_covariance(target.learner.residual_data(pairs)),
    source.learner.update_covariance(source.learner.residual_data(pairs)),
  )
  assert_tree(target.learner.state_dict(), source.learner.state_dict())
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
  for section, key in (("actors", "actor"), ("critics", "state_dict")):
    mean, belief = (bundle[section][group][key] for group in ("mean", "belief"))
    assert_tree(mean, belief)
    assert all(
      value.data_ptr() != belief[name].data_ptr() for name, value in mean.items()
    )
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
  advance(source)
  with pytest.raises(ValueError, match="active training"):
    source.load_initialization(bundle)


def test_shared_initialization_matches_legacy_v3_weight_identity(tmp_path):
  source = training()
  bundle = source.initialization_bundle()
  # The v3 world-model snapshot used content identity and no training revision.
  legacy = dict(bundle, world_model=source.runner._wm.bundle())
  del legacy["world_model"]["dynamics"]["model_revision"]
  source.load_initialization(legacy)
  assert initialization_fingerprint(bundle) == source.initialization_digest
  # An intermediate v3 writer included the revision. Preserve its recorded
  # identity while comparing the actual starting weights across formats.
  intermediate = dict(bundle, world_model=source.runner._wm.bundle())
  torch.save(intermediate, tmp_path / "initialization.pt")
  restored = torch.load(tmp_path / "initialization.pt", weights_only=True)
  target = training(81)
  target.load_initialization(restored)
  assert target.initialization_digest == initialization_fingerprint(intermediate)
  assert (
    initialization_fingerprint(restored, include_model_revision=False)
    == source.initialization_digest
  )
  assert initialization_fingerprint(
    target.runner._wm.bundle(), include_model_revision=False
  ) == initialization_fingerprint(bundle["world_model"])


def test_occupied_output_checks_model_contents_at_equal_revision(tmp_path):
  loop = training()
  loop.run(4, tmp_path)
  model = loop.runner._dynamics
  disk_hash = file_sha256(tmp_path / "training.pt")
  original = model.checkpoint()
  with torch.no_grad():
    next(model.parameters()).add_(1)
  replacement = model.checkpoint()
  assert replacement["model_version"] != original["model_version"]
  replacement["model_revision"] = original["model_revision"]
  model.load_checkpoint(replacement)
  with pytest.raises(ValueError, match="differs from the output checkpoint"):
    loop.run(4, tmp_path)
  assert file_sha256(tmp_path / "training.pt") == disk_hash


def test_model_only_checkpoint_cannot_masquerade_as_joint_resume():
  loop = training()
  state = loop.state_dict()
  state["stage"] = "model"
  with pytest.raises(ValueError, match="training stage"):
    loop.load_state_dict(state)


def test_policy_training_requires_full_replay_before_collection():
  runner = _RoundRunner(replay_mode="model")
  with pytest.raises(ValueError, match="policy starts require full replay"):
    EITTraining(runner)
  assert runner.replay_buffer.num_transitions == 0


def test_joint_training_rejects_retired_group_before_collection():
  runner = _RoundRunner(group="B")
  with pytest.raises(ValueError, match="mean or belief"):
    EITTraining(runner)
  assert runner.reset_count == 0


def test_model_validation_frames_update_policy_but_not_model_normalizers():
  loop = training()
  loop.cfg = replace(loop.cfg, validation_stride=5)
  advance(loop)
  loop.refresh_collection()
  loop.runner._episode_id = torch.tensor([10, 5])
  count = loop.actor.obs_normalizer.count.clone()
  model_count = loop.learner.dynamics.state_normalizer.count.clone()
  advance(loop)
  assert int(loop.actor.obs_normalizer.count - count) == 8
  assert torch.equal(loop.learner.dynamics.state_normalizer.count, model_count)


@pytest.mark.parametrize(
  "component", ["unused_actor", "unused_critic", "critic_version", "missing_group"]
)
def test_initialization_preflights_all_groups_before_any_mutation(component):
  source, target = training(31), training(81)
  bundle = source.initialization_bundle()
  if component == "unused_actor":
    bundle["actors"]["belief"]["actor"]["decoder.0.weight"][0, 0] = float("nan")
  elif component == "unused_critic":
    bundle["critics"]["belief"]["state_dict"]["mlp.0.weight"][0, 0] = float("nan")
  elif component == "critic_version":
    bundle["critics"]["belief"]["contract"]["version"] = "unknown_critic_version"
  else:
    bundle["actors"].pop("mean")
  before = target.state_dict()
  with pytest.raises(ValueError):
    target.load_initialization(bundle)
  assert_tree(target.state_dict(), before)


@pytest.mark.parametrize(
  "field", ["actor_obs", "alignment_rotation", "alignment_translation", "state"]
)
def test_normalization_late_failure_preserves_all_live_statistics(field):
  loop = training()
  loop.refresh_collection()
  for _ in range(4):
    loop.runner.step(record_episode_stats=False)
  loop.runner.finish_collection_round()
  frames = loop.runner.replay_buffer.retained_frames()
  frames = frames[frames["episode_id"] == 1]
  actor = deepcopy(loop.actor.state_dict())
  critic = deepcopy(loop.policy.critic.state_dict())
  # A bad field in the last selected frame must invalidate the entire batch.
  value = loop.runner.replay_buffer._storage[field][1, 3]
  value.flatten()[0] = 2.0 if field == "alignment_rotation" else float("inf")
  with pytest.raises(ValueError, match="Non-finite|Rotation"):
    loop.update_input_normalization(frames)
  assert_tree(loop.actor.state_dict(), actor)
  assert_tree(loop.policy.critic.state_dict(), critic)
  assert loop.runner.imagined_posterior_provider.state is None


@pytest.mark.parametrize("group", ["mean", "belief"])
def test_replay_normalization_matches_full_reset_statistics_without_rollout(group):
  # Two environments yield 18 fresh frames, exceeding the former 16-frame chunk.
  loop = training(buffer_size=9)
  loop.refresh_collection()
  for _ in range(9):
    loop.runner.step(record_episode_stats=False)
  loop.runner.finish_collection_round()
  env, replay = loop.runner.wm_env, loop.runner.replay_buffer
  cfg = replace(
    loop.runner.cfg, estimator_inputs=replace(env.estimator_input_config, group=group)
  )
  env.estimator_input_config = cfg.estimator_inputs
  loop.policy.actor = build_actor(cfg, loop.runner._wm, "cpu").double()
  loop.policy.critic = BeliefCritic(
    group, loop.runner._dynamics.sim_ctx_dim, (12,)
  ).double()
  env.critic_input_config = loop.policy.critic.config
  expected_actor, expected_critic = deepcopy(loop.actor), deepcopy(loop.policy.critic)
  expected_actor.train()
  with torch.no_grad(), env.readonly_policy_motion():
    locations = replay.posterior_cache.locations(replay)
    for chunk in env.iter_cached_policy_starts(locations, batch_size=len(locations[0])):
      obs = env.reset_prepared(chunk)
      expected_actor.update_normalization(obs)
      expected_actor.condition_normalizer.update(obs["task_uncertainty"][..., :-29])
      expected_critic.update_normalization(obs["critic"])
  env.clear_rollout()
  with (
    patch.object(env, "reset_prepared", side_effect=AssertionError("rollout reset")),
    patch.object(
      replay.posterior_cache, "read", side_effect=AssertionError("full belief read")
    ),
    patch.object(
      loop, "_normalization_inputs", wraps=loop._normalization_inputs
    ) as inputs,
  ):
    loop.update_input_normalization()
  inputs.assert_called_once()
  assert len(inputs.call_args.args[0]) == 18
  assert_tree(loop.actor.state_dict(), expected_actor.state_dict())
  assert_tree(loop.policy.critic.state_dict(), expected_critic.state_dict())


def test_normalization_reads_fresh_slots_without_copying_frame_keys_to_python():
  loop = training()
  loop.runner.replay_buffer.enable_posterior_cache(2, loop.runner.cfg.history_length)
  loop.refresh_collection()
  for _ in range(8):
    loop.runner.step(record_episode_stats=False)
  loop.runner.finish_collection_round()
  replay = loop.runner.replay_buffer
  policy_keys = torch.tensor([[1, 6], [1, 0], [1, 6]])
  generator = torch.Generator().manual_seed(33)
  reference_rng = generator.clone_state()
  expected_times = 2 * torch.cat(
    [torch.randint(4, (3,), generator=reference_rng) for _ in range(2)]
  )
  with (
    patch.object(torch.Tensor, "tolist", side_effect=AssertionError("host key copy")),
    patch.object(replay, "posterior_frames", side_effect=AssertionError("key scan")),
  ):
    episodes, times = loop.runner.wm_env.sample_policy_keys(
      3, [1], generator, num_batches=2
    )
    starts = loop.update_input_normalization(policy_keys=policy_keys)
  assert torch.equal(episodes, torch.ones_like(times))
  assert torch.equal(times, expected_times)
  assert torch.equal(generator.get_state(), reference_rng.get_state())
  assert loop._cache_budget["normalization_samples"] == 8
  assert loop._cache_budget["available_policy_starts"] == 8
  assert int(loop.actor.obs_normalizer.count) == 8
  assert torch.equal(starts.batch["time_index"], policy_keys[:, 1])
  # An explicit selection includes model validation, deduplicates frames,
  # ignores unsaved rows, and can include earlier frames.
  frames = replay.retained_frames(fields=("episode_id", "time_index"))
  loop.update_input_normalization(frames[[0, 8, 8, 9, 10]])
  assert loop._cache_budget["normalization_samples"] == 3
  assert int(loop.actor.obs_normalizer.count) == 11


def test_common_initialization_rejects_prefit_critic():
  loop = training()
  initial = loop.initialization_bundle()
  loop.policy.critic_steps = 1
  with pytest.raises(ValueError, match="before training"):
    loop.initialization_bundle()
  with pytest.raises(ValueError, match="active training"):
    loop.load_initialization(initial)


def test_round_policy_samples_singleton_training_episode():
  loop = training()
  # Seal one-frame episodes 0 and 1. The round collects four-frame episodes 2
  # and 3. Odd IDs are training episodes; only 3 has a recorded successor.
  loop.refresh_collection()
  loop.runner.step(record_episode_stats=False)
  loop.runner.finish_collection_round()
  loop.refresh_collection()
  loop.policy.generators["sampling"].manual_seed(1)
  with (
    patch.object(
      loop.policy, "update_prepared", wraps=loop.policy.update_prepared
    ) as update,
    patch.object(
      loop.runner.replay_buffer.posterior_cache,
      "lookup",
      side_effect=AssertionError("key lookup"),
    ),
    patch.object(
      loop, "policy_episode_ids", side_effect=AssertionError("episode enumeration")
    ),
  ):
    advance(loop)
  starts = update.call_args.args[0]
  assert starts.batch["episode_id"].tolist() == [1, 3]
  assert starts.batch["time_index"][0].item() == 0


def test_policy_sampling_includes_model_validation_and_ignores_split():
  loop = training()
  loop.refresh_collection()
  for _ in range(4):
    loop.runner.step(record_episode_stats=False)
  loop.update_input_normalization()
  replay = loop.runner.replay_buffer
  assert loop.policy_episode_ids() == [0, 1]
  assert loop._cache_budget["available_policy_starts"] == 8
  keys = []
  for stride in (2, 5, 7):
    loop.cfg = replace(loop.cfg, validation_stride=stride)
    episodes, times = loop.runner.wm_env.sample_policy_keys(
      128, loop.policy_episode_ids(), torch.Generator().manual_seed(42)
    )
    keys.append(torch.stack((episodes, times), -1))
    assert set(episodes.tolist()) == {0, 1}
    starts = loop.runner.wm_env.prepare_policy_starts(episodes, times)
    assert len(starts) == 128
  assert all(torch.equal(keys[0], sample) for sample in keys[1:])
  assert replay.model_window_starts([0], 3).shape[0] == 2
