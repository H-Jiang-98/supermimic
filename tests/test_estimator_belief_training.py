"""Round lifecycle and transactional restore with real learners and providers."""

import json
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from tensordict import TensorDict

from supermimic.learners.belief_wm_learner import (
  BeliefWorldModelLearner,
  BeliefWorldModelLearnerCfg,
)
from supermimic.learners.wm_learner import WorldModelLearner
from supermimic.replay_buffer import FieldSpec, ReplayBuffer
from supermimic.runners.belief_training import BeliefModelTraining, BeliefTrainingCfg
from supermimic.runners.supermimic_runner import (
  BaseRunner,
  InputStagePolicy,
  SuperMimicRunner,
  SuperMimicRunnerCfg,
  WorldModel,
  build_actor,
)
from supermimic.utils.lie_state import LieState
from tests.conftest import make_mock_motion
from tests.test_estimator_belief_learner import fixture


class _RoundRunner(SuperMimicRunner):
  """Deterministic physical stream; production providers/actor/replay are real."""

  def __init__(self, seed=31):
    estimator, source = fixture(episodes=(4,), seed=seed)
    self.template = source.retained_frames([0])
    self._schema = {
      key: FieldSpec(tuple(value.shape[2:]), value.dtype)
      for key, value in source._storage.items()
    }
    self._dynamics = estimator.dynamics
    self._representation = self._dynamics._apply_prediction.__self__
    self._representation.actor_obs_mode = "proprio"
    self._representation.configure_estimator_measurement(estimator.measurement_config)
    self.cfg = SuperMimicRunnerCfg.for_estimator_inputs(
      "B",
      process_noise=self._dynamics.process_noise_config,
      dynamics_hidden_dims=(12,),
      actor_hidden_dims=(12,),
      buffer_size=8,
      belief_wm=BeliefWorldModelLearnerCfg(
        batch_size=2, microbatch_size=2, num_batches=1, horizon=2
      ),
    )
    self.device = "cpu"
    self.env = SimpleNamespace(num_envs=2)
    self.motion = make_mock_motion()
    self.motion.joint_pos.zero_()
    self.motion.joint_vel.zero_()
    self._wm = WorldModel(
      self._representation, self._dynamics, torch.nn.Identity(), 700, 29
    )
    self.world_model_learner = BeliefWorldModelLearner(estimator, self.cfg.belief_wm)
    self.policy_learner = InputStagePolicy(
      build_actor(self.cfg, self._wm, "cpu").double(), self.cfg.policy
    )
    self.replay_buffer = ReplayBuffer(
      self._schema, self.cfg.buffer_size, 2, initial_prior=self._wm.initial_prior
    )
    self.wm_env = SimpleNamespace(replay_buffer=self.replay_buffer)
    self._pending_collection = None
    self._pending_norm_updates = []
    self._episode_id = torch.tensor([-2, -1])
    self.reset_count = 0
    self.collected = []

  def _buffer_schema(self):
    return self._schema

  def reset_collection(self):
    self.replay_buffer.truncate_open_episodes()
    self._episode_id += 2
    self.index = 0
    self.previous = torch.zeros(2, 29, dtype=torch.float64)
    self.reset_count += 1

  @torch.no_grad()
  def step(self, *, record_episode_stats=True):
    frame = self.template[self.index].expand(2).clone()
    # Held-out rows differ observably, so leaked normalization is detectable.
    frame["state"][0, 2] = 3.0
    frame["state"][0, 7:36] += 0.4
    mean = LieState.from_full(frame["state"])
    measurement = self.posterior_provider.estimator.observation(mean)
    sensor = torch.cat((measurement, self.previous), -1)
    reset = torch.full((2,), self.index == 0)
    prepared = self.posterior_provider.prepare(
      measurement=measurement,
      previous_action=self.previous,
      reset_mask=reset,
      context=frame["sim_ctx"],
      episode_id=self._episode_id,
      time_index=torch.full((2,), self.index),
      reset_truth=mean if self.index == 0 else None,
    )
    self.posterior_provider.commit(prepared)
    obs = TensorDict(
      {
        "history": sensor[:, None].expand(2, 5, 93),
        "goal_motion": sensor.new_zeros(2, 700),
      },
      [2],
    )
    action = self.policy_learner.actor(obs)
    frame["actor_obs"], frame["action"], frame["prev_action"] = (
      sensor,
      action,
      self.previous,
    )
    frame["episode_id"] = self._episode_id.clone()
    frame["time_index"] = torch.full((2,), self.index)
    frame["terminated"] = frame["truncated"] = torch.zeros(2, dtype=torch.bool)
    self.replay_buffer.insert(
      frame, head_priors=self.posterior_provider.state.head_priors
    )
    self._pending_norm_updates.append((frame["state"], action, obs))
    self._flush_normalization()
    self.collected.append(
      (self._dynamics.model_version, frame.clone(), self.posterior_provider.state)
    )
    self.previous = action.clone()
    self.index += 1


def training(seed=31):
  runner = _RoundRunner(seed)
  return BeliefModelTraining(
    runner,
    BeliefTrainingCfg(collection_steps=4, validation_stride=2, validation_batch_size=2),
  )


def assert_tree(actual, expected):
  if isinstance(expected, torch.Tensor):
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
  elif isinstance(expected, dict):
    assert actual.keys() == expected.keys()
    for key in expected:
      assert_tree(actual[key], expected[key])
  elif isinstance(expected, (tuple, list)):
    assert len(actual) == len(expected)
    for a, b in zip(actual, expected, strict=True):
      assert_tree(a, b)
  else:
    assert actual == expected


def test_round_with_only_new_validation_frames_still_optimizes_retained_training_data():
  loop = training()
  loop.cfg = replace(loop.cfg, validation_stride=5)
  loop.run_round()
  # Both current slots now belong to held-out episodes following asynchronous resets.
  loop.runner._episode_id = torch.tensor([10, 5])
  previous_count = loop.learner.dynamics.state_normalizer.count.clone()
  metrics = loop.run_round()
  assert metrics["training_episodes"] == [1]
  assert metrics["validation_episodes"] == [0, 5, 10]
  assert metrics["optimizer_steps"] == 2
  assert torch.equal(loop.learner.dynamics.state_normalizer.count, previous_count)


def test_round_metrics_weight_ragged_updates_by_valid_transitions():
  loop = training()
  loop.learner.cfg = replace(loop.learner.cfg, num_batches=2)
  updates = [
    {
      "valid_transitions": torch.tensor(count),
      "Loss/posterior_nll": torch.tensor(nll),
      "Grad/total_norm": torch.tensor(norm),
    }
    for count, nll, norm in ((2, 4.0, 1.0), (6, 8.0, 3.0))
  ]
  with patch.object(loop.learner, "update", side_effect=updates):
    metrics = loop.run_round()["train"]
  assert metrics == {
    "valid_transitions": 8,
    "Loss/posterior_nll": 7.0,
    "Grad/total_norm": 2.0,
  }


def test_two_rounds_freeze_actor_exclude_validation_and_refresh_current_model():
  loop = training()
  runner = loop.runner
  actor = deepcopy(runner.policy_learner.actor.state_dict())
  normalizer_inputs = []
  original = loop.learner.update_normalization

  def normalize(states, actions):
    normalizer_inputs.append((states.clone(), actions.clone()))
    original(states, actions)

  with patch.object(loop.learner, "update_normalization", side_effect=normalize):
    first = loop.run_round()
    old = runner.collected[-1][2]
    second = loop.run_round()
  assert loop.rounds == loop.learner.steps == 2
  assert runner.reset_count == 3
  assert loop.collected_frames == 16
  assert first["collection_model_version"] != first["trained_model_version"]
  assert second["collection_model_version"] == first["trained_model_version"]
  assert second["trained_model_version"] != first["trained_model_version"]
  assert [v for v, _, _ in runner.collected[:4]] == [
    first["collection_model_version"]
  ] * 4
  assert [v for v, _, _ in runner.collected[4:]] == [
    second["collection_model_version"]
  ] * 4
  assert_tree(runner.policy_learner.actor.state_dict(), actor)
  assert all(
    not parameter.requires_grad
    for parameter in runner.policy_learner.actor.parameters()
  )
  for round_index, (states, actions) in enumerate(normalizer_inputs):
    expected = runner.collected[round_index * 4 : (round_index + 1) * 4]
    assert_tree(states, torch.stack([frame["state"][1] for _, frame, _ in expected]))
    assert_tree(actions, torch.stack([frame["action"][1] for _, frame, _ in expected]))
  assert set(second["training_episodes"]).isdisjoint(second["validation_episodes"])
  assert all(e % 2 for e in second["training_episodes"])
  assert all(not e % 2 for e in second["validation_episodes"])
  assert old.model_version != runner.posterior_provider.model_version
  assert runner.posterior_provider.state is None
  runner.step()
  assert (
    runner.posterior_provider.state.model_version == second["trained_model_version"]
  )
  assert runner.posterior_provider.state.episode_id.tolist() == [4, 5]
  assert runner.posterior_provider.state.time_index.tolist() == [0, 0]
  with pytest.raises(ValueError, match="version"):
    runner.posterior_provider.restore(
      old.belief,
      old.context,
      model_version=old.model_version,
      episode_id=old.episode_id,
      time_index=old.time_index,
    )


@pytest.mark.parametrize(
  "component", ["dynamics", "actor", "optimizer", "filter", "prior", "measurement"]
)
def test_malformed_checkpoint_does_not_partially_restore_any_live_component(component):
  source, target = training(31), training(91)
  source.run_round()
  target.run_round()
  checkpoint = source.state_dict()
  if component == "dynamics":
    checkpoint["world_model"]["dynamics"]["state_dict"]["mlp.0.weight"][0, 0] = float(
      "nan"
    )
  elif component == "actor":
    checkpoint["actor"]["actor"]["decoder.0.weight"][0, 0] = float("nan")
  elif component == "optimizer":
    checkpoint["learner"]["optimizer"]["state"][0]["exp_avg"] = torch.zeros(1)
  elif component == "filter":
    checkpoint["world_model"]["estimator"]["filter_config"]["jitter_initial"] *= 2
  elif component == "prior":
    checkpoint["world_model"]["estimator"]["initial_prior"]["std"] = (0.02,) * 6
  else:
    checkpoint["world_model"]["estimator"]["measurement_config"]["gravity_std"] *= 2
  before = target.state_dict()
  provider, replay = target.runner.posterior_provider, target.runner.replay_buffer
  with pytest.raises((ValueError, TypeError)):
    target.load_state_dict(checkpoint)
  assert_tree(target.state_dict(), before)
  assert target.runner.posterior_provider is provider
  assert target.runner.replay_buffer is replay


def test_saved_round_restores_optimizer_for_identical_next_input_and_fresh_collection(
  tmp_path,
):
  source, target = training(31), training(91)
  path = tmp_path / "training.pt"
  source.run_round(path)
  checkpoint = torch.load(path, weights_only=True)
  target.load_state_dict(checkpoint)
  assert target.rounds == source.rounds == 1
  assert target.runner.replay_buffer.num_transitions == 0
  assert target.runner.posterior_provider.state is None
  fixed = source.runner.replay_buffer.sample_belief_sequences(2, 2, episode_ids=[1])
  expected = source.learner.update(fixed)
  actual = target.learner.update(fixed)
  assert_tree(actual, expected)
  assert_tree(
    target.runner._dynamics.state_dict(), source.runner._dynamics.state_dict()
  )
  assert_tree(target.learner.state_dict(), source.learner.state_dict())
  assert not path.with_suffix(".pt.tmp").exists()


def test_restore_rejects_changed_motion_and_legacy_without_modifying_runner():
  source, target = training(31), training(91)
  source.run_round()
  state = source.state_dict()
  target.runner.motion.body_pos_w[2, 0, 0] += 1
  before = target.runner._dynamics.model_version
  with pytest.raises(ValueError, match="motion content mismatch"):
    target.load_state_dict(state)
  assert target.runner._dynamics.model_version == before
  state.pop("motion_digest")
  state["version"] = "belief_model_training_v1"
  with pytest.raises(ValueError, match="no verified motion identity"):
    target.load_state_dict(state)


def test_cpu_checkpoint_never_touches_cuda_generators():
  source, target = training(31), training(91)
  source.run_round()
  with (
    patch("torch.cuda.get_rng_state", side_effect=AssertionError),
    patch("torch.cuda.get_rng_state_all", side_effect=AssertionError),
    patch("torch.cuda.set_rng_state_all", side_effect=AssertionError),
    patch("torch.cuda.device_count", return_value=0),
  ):
    state = source.state_dict()
    assert state["rng_cuda"] is None
    target.load_state_dict(state)
  assert target.learner.steps == source.learner.steps


def test_resume_preserves_unrelated_output_and_rejects_stale_checkpoint(tmp_path):
  current = tmp_path / "current"
  source = training()
  source.run(1, current)
  checkpoint = current / "training.pt"
  older = checkpoint.read_bytes()
  unrelated = tmp_path / "unrelated"
  unrelated.mkdir()
  (unrelated / "training.pt").write_bytes(b"other run")
  with pytest.raises(ValueError, match="its own current training.pt"):
    training().run(1, unrelated, resume=checkpoint)
  assert list(unrelated.iterdir()) == [unrelated / "training.pt"]
  assert (unrelated / "training.pt").read_bytes() == b"other run"
  training().run(1, current, resume=checkpoint)
  checkpoint.write_bytes(older)
  history = (current / "metrics.jsonl").read_bytes()
  with pytest.raises(ValueError, match="different rounds"):
    training().run(1, current, resume=checkpoint)
  assert (current / "metrics.jsonl").read_bytes() == history
  assert checkpoint.read_bytes() == older


@pytest.mark.parametrize("partial_length", [0, 27])
def test_resume_recovers_checkpointed_metric_once_and_supports_new_output(
  tmp_path, partial_length
):
  original = tmp_path / "original"
  source = training()
  first, second = source.run(2, original)
  checkpoint = original / "training.pt"
  history = original / "metrics.jsonl"
  history.write_text(
    json.dumps(first) + "\n" + json.dumps(second, sort_keys=True)[:partial_length]
  )
  [third] = training().run(1, original, resume=checkpoint)
  assert [json.loads(line) for line in history.read_text().splitlines()] == [
    first,
    second,
    third,
  ]
  [fourth] = training().run(1, tmp_path / "new", resume=checkpoint)
  assert fourth["round"] == 4


@pytest.mark.parametrize("stochastic", [False, True])
def test_runner_dispatches_distinct_learner_data_contracts(stochastic):
  runner = _RoundRunner()
  wm = runner._wm
  if not stochastic:
    from tests.test_estimator_stochastic_dynamics import make_dynamics

    wm.dynamics, wm.representation = make_dynamics(stochastic=False)
  runner.env.unwrapped = SimpleNamespace(
    action_manager=object(),
    command_manager=SimpleNamespace(
      get_term=lambda _: SimpleNamespace(motion=object())
    ),
  )
  with patch("supermimic.runners.supermimic_runner.build_world_model", return_value=wm):
    BaseRunner._init_common(runner)
  expected = BeliefWorldModelLearner if stochastic else WorldModelLearner
  assert isinstance(runner.world_model_learner, expected)
  if not stochastic:
    with pytest.raises(ValueError, match="estimator_inputs and process_noise"):
      BeliefModelTraining(runner)


def test_explicit_bundle_migration_preserves_source_and_adopts_target_jacobian_config():
  wm = _RoundRunner()._wm
  current = deepcopy(wm.bundle())
  legacy = deepcopy(current)
  metadata = legacy["dynamics"]["metadata"]
  metadata["format_version"] = "stochastic_dynamics_v1"
  metadata["integration"].pop("version")
  legacy["dynamics"]["model_version"] = wm.dynamics._content_digest(
    metadata, legacy["dynamics"]["state_dict"]
  )
  legacy["estimator"]["filter_config"] = {
    key: value
    for key, value in legacy["estimator"]["filter_config"].items()
    if not key.startswith("jacobian_")
  }
  untouched = deepcopy(legacy)
  version_before = wm.dynamics.model_version
  wm.filter_config = replace(
    wm.filter_config, jacobian_chunk_size=7, jacobian_batch_chunk_size=1
  )
  with pytest.raises(ValueError, match="Incomplete"):
    wm.load_bundle(legacy)
  migrated = wm.migrate_bundle(legacy)
  assert_tree(legacy, untouched)
  assert wm.dynamics.model_version == version_before
  assert migrated["dynamics"]["model_version"] != legacy["dynamics"]["model_version"]
  wm.load_bundle(migrated)
  assert wm.filter_config.jacobian_chunk_size == 7
  assert wm.filter_config.jacobian_batch_chunk_size == 1
  assert wm.dynamics.model_version == migrated["dynamics"]["model_version"]
  assert_tree(wm.dynamics.state_dict(), current["dynamics"]["state_dict"])


def test_matrix_v1_weights_migrate_but_old_posterior_version_expires():
  wm = _RoundRunner()._wm
  previous = deepcopy(wm.bundle())
  previous["estimator"]["filter_config"]["jacobian_chunk_size"] = 7
  dynamics = previous["dynamics"]
  dynamics["metadata"]["integration"]["version"] = "right_matrix_v1"
  dynamics["model_version"] = wm.dynamics._content_digest(
    dynamics["metadata"], dynamics["state_dict"]
  )
  with pytest.raises(ValueError, match="configuration or state layout mismatch"):
    wm.load_bundle(previous)
  migrated = wm.migrate_bundle(previous)
  wm.load_bundle(migrated)
  assert wm.filter_config.jacobian_chunk_size == 7
  assert wm.dynamics.model_version != dynamics["model_version"]
  assert_tree(wm.dynamics.state_dict(), dynamics["state_dict"])
