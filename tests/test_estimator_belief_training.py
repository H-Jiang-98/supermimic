"""Round lifecycle and transactional restore with real learners and providers."""

import json
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from tensordict import TensorDict

from supermimic.learners.two_stage_wm import (
  CovarianceFitCfg,
  MeanFitCfg,
  TwoStageWorldModelLearner,
)
from supermimic.learners.wm_learner import WorldModelLearner
from supermimic.replay_buffer import MODEL_REPLAY_FIELDS, FieldSpec
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
from tests.test_estimator_two_stage_wm import fixture


class _RoundRunner(SuperMimicRunner):
  """Deterministic physical stream; production providers/actor/replay are real."""

  def __init__(self, seed=31, replay_mode="full"):
    estimator, source = fixture(episodes=(4,), seed=seed)
    self.template = source.retained_frames([0])
    self.replay_mode = replay_mode
    self._schema = {
      key: FieldSpec(tuple(value.shape[2:]), value.dtype)
      for key, value in source._storage.items()
      if replay_mode == "full" or key in MODEL_REPLAY_FIELDS
    }
    self._dynamics = estimator.dynamics
    self._representation = self._dynamics._apply_prediction.__self__
    self._representation.actor_obs_mode = "proprio"
    self._representation.configure_estimator_measurement(estimator.measurement_config)
    self.cfg = SuperMimicRunnerCfg.for_estimator_inputs(
      "B",
      process_noise=self._dynamics.process_noise_config,
      dynamics_hidden_dims=(12, 12),
      actor_hidden_dims=(12,),
      buffer_size=8,
      mean=MeanFitCfg(batch_size=2, num_batches=1, window_frames=3),
      cov=CovarianceFitCfg(batch_size=2),
    )
    self.device = "cpu"
    self.env = SimpleNamespace(num_envs=2)
    self.motion = make_mock_motion()
    self.motion.joint_pos.zero_()
    self.motion.joint_vel.zero_()
    self._wm = WorldModel(
      self._representation, self._dynamics, torch.nn.Identity(), 700, 29
    )
    self.world_model_learner = TwoStageWorldModelLearner(
      self._dynamics, self._representation, self.cfg.mean, self.cfg.cov, self.device
    )
    self.policy_learner = InputStagePolicy(
      build_actor(self.cfg, self._wm, "cpu").double(), self.cfg.policy
    )
    self.replay_buffer = self._make_replay_buffer()
    self.wm_env = SimpleNamespace(replay_buffer=self.replay_buffer)
    self._pending_collection = None
    self._pending_norm_updates = []
    self._episode_id = None
    self.reset_count = 0
    self.collected = []

  def _buffer_schema(self):
    return self._schema

  def reset_collection(self):
    self.replay_buffer.truncate_open_episodes()
    self._episode_id = (
      torch.arange(self.env.num_envs)
      if self._episode_id is None
      else self._episode_id + self.env.num_envs
    )
    self.index = 0
    self.previous = torch.zeros(2, 29, dtype=torch.float64)
    self.reset_count += 1

  @torch.no_grad()
  def step(self, *, record_episode_stats=True):
    frame = self.template[self.index % len(self.template)].expand(2).clone()
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
    if self.replay_mode == "model":
      self.replay_buffer.validate_previous_action(
        frame["episode_id"], frame["time_index"], self.previous
      )
      self.replay_buffer.insert(frame.select(*MODEL_REPLAY_FIELDS))
    else:
      self.replay_buffer.insert(
        frame,
        head_priors=self.posterior_provider.state.head_priors,
        posterior=(
          self.replay_buffer.posterior_cache.capture(
            self.replay_buffer,
            self.posterior_provider.state.belief,
            self._dynamics.model_revision,
          )
          if self.replay_buffer.posterior_cache is not None
          else None
        ),
      )
    self._pending_norm_updates.append((frame["state"], action, obs))
    self._flush_normalization()
    self.collected.append(
      (
        self.posterior_provider.model_version,
        frame.clone(),
        self.posterior_provider.state,
      )
    )
    self.previous = action.clone()
    self.index += 1


def training(seed=31):
  runner = _RoundRunner(seed, replay_mode="model")
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
  assert metrics["mean_steps"] == 2
  assert torch.equal(loop.learner.dynamics.state_normalizer.count, previous_count)


def test_two_rounds_freeze_actor_exclude_validation_and_refresh_current_model():
  loop = training()
  runner = loop.runner
  actor = deepcopy(runner.policy_learner.actor.state_dict())
  normalizer_inputs = []
  original = loop.learner.stage_normalization

  def normalize(states, actions, staged=None):
    normalizer_inputs.append((states.clone(), actions.clone()))
    return original(states, actions, staged)

  with patch.object(loop.learner, "stage_normalization", side_effect=normalize):
    first = loop.run_round()
    old = runner.collected[-1][2]
    first_mean = deepcopy(runner._dynamics.mlp.state_dict())
    second = loop.run_round()
  assert any(
    not torch.equal(v, first_mean[k])
    for k, v in runner._dynamics.mlp.state_dict().items()
  )
  assert loop.rounds == loop.learner.mean_steps == 2
  assert loop.learner.cov_steps == 5
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
  "component",
  [
    "dynamics",
    "actor",
    "mean_optimizer",
    "cov_optimizer",
    "filter",
    "prior",
    "measurement",
    "replay_mode",
  ],
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
  elif component == "mean_optimizer":
    checkpoint["learner"]["mean"]["optimizers"][0]["state"][0]["exp_avg"] = torch.zeros(
      1
    )
  elif component == "cov_optimizer":
    checkpoint["learner"]["cov"]["optimizer"]["state"][0]["exp_avg"] = torch.zeros(1)
  elif component == "filter":
    checkpoint["world_model"]["estimator"]["filter_config"]["jitter_initial"] *= 2
  elif component == "prior":
    checkpoint["world_model"]["estimator"]["initial_prior"]["std"] = (0.02,) * 6
  elif component == "replay_mode":
    checkpoint["replay_mode"] = "full"
  else:
    checkpoint["world_model"]["estimator"]["measurement_config"]["gravity_std"] *= 2
  before = target.state_dict()
  provider, replay = target.runner.posterior_provider, target.runner.replay_buffer
  with pytest.raises((ValueError, TypeError)):
    target.load_state_dict(checkpoint)
  assert_tree(target.state_dict(), before)
  assert target.runner.posterior_provider is provider
  assert target.runner.replay_buffer is replay


def test_checkpoint_before_first_collection_restores_environment_count_and_episode_ids():
  source, target = training(31), training(91)
  state = source.state_dict()
  assert state["num_envs"] == 2 and state["episode_id"] is None
  target.run_round()
  target.load_state_dict(state)
  assert target.rounds == 0
  assert target.runner._episode_id.tolist() == [0, 1]
  assert target.run_round()["mean_steps"] == 1


def test_saved_round_restores_optimizer_for_identical_next_input_and_fresh_collection(
  tmp_path,
):
  source, target = training(31), training(91)
  path = tmp_path / "training.pt"
  source.run_round(path)
  checkpoint = torch.load(path, weights_only=True)
  assert checkpoint["replay_mode"] == "model"
  target.load_state_dict(checkpoint)
  assert target.rounds == source.rounds == 1
  assert target.runner.replay_buffer.num_transitions == 0
  assert target.runner.wm_env.replay_buffer is target.runner.replay_buffer
  assert target.runner.replay_buffer.replay_mode == "model"
  assert (
    set(target.runner.replay_buffer.retained_frames().keys()) == MODEL_REPLAY_FIELDS
  )
  assert target.runner.posterior_provider.state is None
  replay = source.runner.replay_buffer
  windows = replay.model_windows(replay.model_window_starts([1], 3), 3)
  pairs = replay.model_windows(replay.model_window_starts([1], 2), 2)
  assert_tree(target.learner.update_mean(windows), source.learner.update_mean(windows))

  def covariance_step(loop):
    return loop.learner.update_covariance(loop.learner.residual_data(pairs))

  assert_tree(covariance_step(target), covariance_step(source))
  assert_tree(
    target.runner._dynamics.state_dict(), source.runner._dynamics.state_dict()
  )
  assert_tree(target.learner.state_dict(), source.learner.state_dict())
  assert not path.with_suffix(".pt.tmp").exists()


def test_restore_rejects_changed_motion_without_modifying_runner():
  source, target = training(31), training(91)
  source.run_round()
  state = source.state_dict()
  target.runner.motion.body_pos_w[2, 0, 0] += 1
  before = target.runner._dynamics.model_version
  with pytest.raises(ValueError, match="motion content mismatch"):
    target.load_state_dict(state)
  assert target.runner._dynamics.model_version == before


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
  assert target.learner.mean_steps == source.learner.mean_steps
  assert target.learner.cov_steps == source.learner.cov_steps


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
  expected = TwoStageWorldModelLearner if stochastic else WorldModelLearner
  assert isinstance(runner.world_model_learner, expected)
  if not stochastic:
    with pytest.raises(ValueError, match="estimator_inputs and process_noise"):
      BeliefModelTraining(runner)


@pytest.mark.parametrize("steps", [9, 10])
def test_model_normalization_streams_overwritten_frames_before_mean_fit(steps):
  loop = training()
  runner = loop.runner
  loop.cfg = replace(loop.cfg, collection_steps=steps)
  runner.template = torch.cat([runner.template] * 3)
  runner.cfg.buffer_size = 3
  runner.replay_buffer = runner._make_replay_buffer()
  runner.wm_env.replay_buffer = runner.replay_buffer
  reference = deepcopy(runner._dynamics.normalizers())
  with patch.object(
    loop.learner, "stage_normalization", wraps=loop.learner.stage_normalization
  ) as stage:
    log = loop.run_round()
  assert [len(call.args[0]) for call in stage.call_args_list] == [
    min(3, steps - offset) for offset in range(0, steps, 3)
  ]
  assert all(
    version == log["collection_model_version"] for version, _, _ in runner.collected
  )
  # Only row 1 belongs to the training split; this reference sees every fresh
  # frame at once, including those overwritten more than one ring cycle ago.
  training_frames = torch.stack([frame[1] for _, frame, _ in runner.collected])
  loop.learner.mean.update_normalization(
    training_frames["state"], training_frames["action"], normalizers=reference
  )
  for actual, expected in zip(runner._dynamics.normalizers(), reference, strict=True):
    assert int(actual.count) == steps
    torch.testing.assert_close(actual.state_dict(), expected.state_dict())
  assert log["train"]["cov_pairs"] == log["validation"]["cov_pairs"] == 2


def test_invalid_heldout_frame_preserves_published_round_and_can_resume(tmp_path):
  loop = training()
  [first] = loop.run(1, tmp_path)
  path, history = tmp_path / "training.pt", tmp_path / "metrics.jsonl"
  checkpoint, rows = path.read_bytes(), history.read_bytes()
  seal = loop.runner.finish_collection_round

  def corrupt_then_seal():
    replay = loop.runner.replay_buffer
    slot = (replay._step - 2) % replay.buffer_size
    assert replay._storage["episode_id"][0, slot] % 2 == 0
    replay._storage["state"][0, slot, 7] = float("nan")
    seal()

  with (
    patch.object(loop.runner, "finish_collection_round", side_effect=corrupt_then_seal),
    pytest.raises(ValueError, match="Non-finite replay state"),
  ):
    loop.run(1, tmp_path)
  assert loop.rounds == 1 and loop.learner.mean_steps == first["mean_steps"]
  assert loop.runner._dynamics.model_version == first["trained_model_version"]
  with pytest.raises(ValueError):
    loop.save(path, metrics={"validation": {"Loss/state": float("nan")}})
  assert path.read_bytes() == checkpoint and history.read_bytes() == rows
  [second] = loop.run(1, tmp_path, resume=path)
  assert second["round"] == 2


def test_resume_recognizes_a_matching_preexisting_nan_metric(tmp_path):
  loop = training()
  path = tmp_path / "training.pt"
  # Legacy v3 output predates the immutable snapshot registry.
  loop.run_round(path)
  state = torch.load(path, weights_only=True)
  state["last_metrics"]["validation"]["Loss/process_nll"] = float("nan")
  torch.save(state, path)
  (tmp_path / "metrics.jsonl").write_text(json.dumps(state["last_metrics"]) + "\n")
  [second] = training().run(1, tmp_path, resume=path)
  assert second["round"] == 2
  json.dumps(second, allow_nan=False)
