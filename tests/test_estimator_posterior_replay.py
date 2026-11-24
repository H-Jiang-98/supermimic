"""Sparse storage, true episode history, continuous EIT and revision contracts."""

from dataclasses import replace
from unittest.mock import patch
from weakref import ref

import pytest
import torch
from tensordict import TensorDict

from supermimic.belief import BeliefState
from supermimic.posterior_replay import pack_belief, unpack_belief
from supermimic.replay_buffer import FieldSpec, ReplayBuffer
from supermimic.runners.eit_training import EITTraining
from supermimic.runners.run_config import validate_eit_checkpoint_contract
from supermimic.utils.lie_state import LieState
from tests.test_estimator_belief_integration import setup_env
from tests.test_estimator_eit_training import advance, training
from tests.test_estimator_runner import _collection_runner


def belief(batch=3):
  full = torch.zeros(batch, 71)
  full[:, 3] = 1
  return BeliefState(
    LieState.from_full(full), torch.eye(70).expand(batch, 70, 70).clone()
  )


def test_packed_factor_roundtrip_is_exact_and_keeps_covariance_positive():
  b = belief()
  torch.manual_seed(8)
  factor = torch.randn(3, 70, 70).tril(-1) * 0.005 + 0.1 * torch.eye(70)
  b = replace(b, covariance_sqrt=factor)
  mean, lower = pack_belief(b)
  assert mean.shape == (3, 76) and lower.shape == (3, 2485)
  assert mean.dtype == lower.dtype == torch.float32
  restored = unpack_belief(mean, lower, dtype=torch.float32)
  assert torch.equal(restored.covariance_sqrt, factor)
  assert torch.equal(restored.covariance, b.covariance)
  assert torch.equal(restored.mean.rotation, b.mean.rotation)
  restored.validate()


def replay_fixture(size, stride, history):
  schema = {
    "state": FieldSpec(71),
    "action": FieldSpec(29),
    "prev_action": FieldSpec(29),
    "actor_obs": FieldSpec(93),
    "sim_ctx": FieldSpec(2),
    "episode_id": FieldSpec(dtype=torch.long),
    "time_index": FieldSpec(dtype=torch.long),
    "terminated": FieldSpec(dtype=torch.bool),
    "truncated": FieldSpec(dtype=torch.bool),
  }
  replay = ReplayBuffer(schema, size, 3)
  replay.enable_posterior_cache(stride, history)
  return replay


@pytest.mark.parametrize(
  "size,stride", [(8, 1), (8, 2), (8, 4), (8, 8), (9, 2), (9, 4), (9, 8), (3, 2)]
)
@pytest.mark.parametrize("history", [1, 5, 12])
def test_sparse_ring_retains_exact_histories_across_asynchronous_resets(
  size, stride, history
):
  replay = replay_fixture(size, stride, history)
  cache = replay.posterior_cache
  b = belief()
  episodes = torch.arange(3)
  times = torch.zeros(3, dtype=torch.long)
  previous = torch.zeros(3, 29)
  windows = torch.zeros(3, history, 93)
  expected = {}
  for seq in range(size * 4 + 1):
    sensor = torch.cat((torch.full((3, 64), float(seq)), previous), -1)
    windows = torch.cat((windows[:, 1:], sensor[:, None]), 1)
    windows[times == 0] = sensor[times == 0, None]
    action = torch.full((3, 29), float(seq + 1))
    done = (times + 1) % torch.tensor([3, 5, 7]) == 0
    frame = TensorDict(
      {
        "state": b.mean.to_full(),
        "action": action,
        "prev_action": previous,
        "actor_obs": sensor,
        "sim_ctx": torch.zeros(3, 2),
        "episode_id": episodes.clone(),
        "time_index": times.clone(),
        "terminated": done,
        "truncated": torch.zeros(3, dtype=torch.bool),
      },
      [3],
    )
    token = object()
    posterior = cache.capture(replay, b, seq // 3)
    replay.insert(frame, posterior=posterior, token=token)
    replay.insert(frame, posterior=posterior, token=token)
    assert replay.insert_count == seq + 1
    if seq % stride == 0:
      for row in range(3):
        expected[(int(episodes[row]), int(times[row]))] = (seq, windows[row].clone())
    replay.finish_collection_round()
    frames = replay.posterior_frames(fields=("episode_id", "time_index"))
    keys = torch.stack((frames["episode_id"], frames["time_index"]), -1)
    # Deliberately duplicate and reverse requests.
    keys = torch.cat((keys.flip(0), keys[:1]))
    batch, _ = cache.read(replay, keys)
    for i, key in enumerate(keys.tolist()):
      saved_seq, window = expected[tuple(key)]
      assert saved_seq >= seq + 1 - size
      assert batch["insert_sequence"][i] == saved_seq
      assert torch.equal(batch["sensor_window"][i], window)
      assert batch["posterior_revision"][i] == saved_seq // 3
    previous = torch.where(done[:, None], 0, action)
    episodes += 3 * done.long()
    times = torch.where(done, 0, times + 1)
  assert cache.storage["posterior_sqrt_tril"].shape == (
    3,
    (size + stride - 1) // stride,
    2485,
  )
  assert "sensor_window" not in replay._storage.keys()
  assert not replay._training_heads and not replay._episode_heads
  assert cache.boundary["actor_obs"].numel() == 3 * (history - 1) * 93
  if history > 1:
    # Corrupt identity while leaving sensor bytes intact: never pad a ring boundary.
    cache.boundary_sequences.fill_(-1)
    earliest = cache.frames(replay, fields=("episode_id", "time_index"))
    key = torch.stack((earliest["episode_id"][:1], earliest["time_index"][:1]), -1)
    saved_seq = expected[tuple(key[0].tolist())][0]
    if saved_seq - min(int(key[0, 1]), history - 1) < replay.insert_count - size:
      with pytest.raises(ValueError, match="history gap"):
        cache.read(replay, key)


@pytest.mark.parametrize("stride", [1, 2, 4, 8])
def test_three_rounds_keep_live_belief_and_filter_each_new_measurement_once(stride):
  loop = training()
  loop.cfg = replace(loop.cfg, posterior_save_stride=stride)
  loop.runner.posterior_save_stride = stride
  loop.runner.replay_buffer.enable_posterior_cache(stride, 5)
  loop.refresh_collection()
  provider = loop.runner.posterior_provider
  estimator = provider.estimator
  imagined_calls = [0, 0]
  update = loop._update_policy

  def counted_update():
    before = (correct.call_count, predict.call_count)
    result = update()
    imagined_calls[0] += correct.call_count - before[0]
    imagined_calls[1] += predict.call_count - before[1]
    return result

  with (
    patch.object(loop, "_update_policy", side_effect=counted_update),
    patch.object(estimator, "_correct", wraps=estimator._correct) as correct,
    patch.object(
      estimator, "transition_jacobian", wraps=estimator.transition_jacobian
    ) as predict,
    patch.object(
      loop.runner._dynamics,
      "_content_digest",
      side_effect=AssertionError("training hashed dynamics"),
    ),
  ):
    logs = [advance(loop) for _ in range(3)]
  assert correct.call_count - imagined_calls[0] == 12
  assert predict.call_count - imagined_calls[1] == 11
  assert min(imagined_calls) > 0
  assert loop.runner.posterior_provider is provider
  assert provider.state.time_index.tolist() == [11, 11]
  assert loop.runner.reset_count == 1
  assert loop.runner.replay_buffer.insert_count == 12
  assert logs[-1]["budget"]["posterior_save_stride"] == stride
  assert logs[-1]["budget"]["available_training_windows"] == 6
  assert int(loop.actor.obs_normalizer.count) == len(range(0, 12, stride))
  assert (
    logs[1]["budget"]["normalization_samples"] == len(range(4, 8, stride))
    if stride <= 4
    else logs[1]["budget"]["normalization_samples"] == 0
  )


def test_cache_preparation_has_no_filter_calls_and_keeps_mixed_sources_and_order():
  loop = training()
  advance(loop)
  advance(loop)
  env = loop.runner.wm_env
  keys = torch.tensor([[1, 7], [1, 1], [1, 1], [1, 4]])
  estimator = env.posterior_provider.estimator
  with (
    patch.object(estimator, "predict", side_effect=AssertionError("prefix predict")),
    patch.object(estimator, "correct", side_effect=AssertionError("prefix correct")),
  ):
    starts = loop.update_input_normalization(
      loop.runner.replay_buffer.posterior_frames([1]), keys
    )
    obs = env.reset_prepared(starts.repeat_interleave(2))
  assert len(obs) == 8
  assert torch.equal(starts.batch["time_index"], keys[:, 1])
  assert starts.batch["posterior_revision"].unique().numel() == 3
  # The source revision differs from the current model, but next imagined filter
  # steps still retain action derivatives (also exercised by policy regressions).
  assert (starts.batch["posterior_revision"] < starts.model_version).all()


def test_saved_stride_wins_on_restore_and_v3_remains_readable_but_not_resumable():
  source = training()
  source.cfg = replace(source.cfg, posterior_save_stride=4)
  source.runner.posterior_save_stride = 4
  source.runner.replay_buffer.enable_posterior_cache(4, 5)
  advance(source)
  checkpoint = source.state_dict()
  target = training()
  target.load_state_dict(checkpoint)
  assert target.cfg.posterior_save_stride == 4
  assert target.runner.replay_buffer.posterior_cache.stride == 4
  assert target.runner.replay_buffer.insert_count == 0
  assert target.runner.posterior_provider.state is None
  assert target.runner._dynamics.model_revision == checkpoint["model_revision"]
  old = dict(checkpoint, version="eit_training_v3")
  with pytest.raises(ValueError, match="checkpoint"):
    validate_eit_checkpoint_contract(old)
  with pytest.raises(ValueError, match="checkpoint"):
    EITTraining._check_checkpoint_version(old)


@pytest.mark.parametrize("target_stride", [1, 4])
def test_fresh_restore_reuses_matching_allocations_and_releases_superseded_cache(
  target_stride,
):
  source = training()
  advance(source)
  saved = source.state_dict()
  target = training()
  replay = target.runner.replay_buffer
  replay.enable_posterior_cache(target_stride, 5)
  target.runner.posterior_save_stride = target_stride
  target.cfg = replace(target.cfg, posterior_save_stride=target_stride)
  old_cache = ref(replay.posterior_cache)
  raw_pointer = replay._storage["actor_obs"].data_ptr()
  # Even an explicitly inspected empty cache is reused when the layouts match.
  old_cov = ref(replay.posterior_cache.storage["posterior_sqrt_tril"])
  with patch.object(
    target.runner,
    "_make_replay_buffer",
    side_effect=AssertionError("allocated raw replay"),
  ):
    target.load_state_dict(saved)
  assert replay is target.runner.replay_buffer is target.runner.wm_env.replay_buffer
  assert replay._storage["actor_obs"].data_ptr() == raw_pointer
  assert replay.posterior_cache.stride == 1
  assert replay.insert_count == 0
  cache = replay.posterior_cache
  if target_stride == 1:
    assert old_cache() is cache and old_cov() is cache.storage["posterior_sqrt_tril"]
  else:
    assert old_cache() is None and old_cov() is None
    assert cache.storage_bytes()["posterior_storage_bytes"] == cache.sequences.nbytes
    target.runner.step()
    assert cache.storage["posterior_sqrt_tril"].shape == (2, 8, 2485)
    assert cache.storage_bytes()["posterior_storage_bytes"] > cache.sequences.nbytes


def test_populated_restore_preserves_old_replay_on_failure_and_clears_it_on_success():
  loop = training()
  advance(loop)
  saved = loop.state_dict()
  runner = loop.runner
  old = runner.replay_buffer
  old_cache = old.posterior_cache
  previous_ids = runner._episode_id.clone()
  assert runner.replay_buffer.episode_head_storage_bytes > 0
  before = runner._dynamics.model_revision
  malformed = dict(
    saved, loop_config=dict(saved["loop_config"], posterior_save_stride=9)
  )
  with pytest.raises(ValueError, match="posterior replay checkpoint contract"):
    loop.load_state_dict(malformed)
  with patch.object(runner, "_make_replay_buffer", side_effect=MemoryError):
    with pytest.raises(MemoryError):
      loop.load_state_dict(saved)
  assert runner.replay_buffer is runner.wm_env.replay_buffer is old
  assert runner._dynamics.model_revision == before
  assert old.posterior_cache is old_cache and len(old.posterior_frames()) == 8

  loop.load_state_dict(saved)
  replay = runner.replay_buffer
  assert not replay.insert_count and not replay.num_transitions
  assert not replay._episode_heads and not replay.episode_head_storage_bytes
  assert not len(replay.posterior_frames())
  assert (replay.posterior_cache.boundary_sequences == -1).all()
  assert (runner._episode_id != previous_ids).all()
  runner.step()
  assert replay.insert_count == 1
  assert replay.posterior_cache.sequences[0] == 0
  assert (replay.posterior_frames()["time_index"] == 0).all()


def test_physical_heads_remain_diagnostic_even_when_unsaved_and_expire_with_raw_frames():
  runner = _collection_runner("full")
  runner.replay_buffer.enable_posterior_cache(2, 5)
  stochastic, _, _ = setup_env("B")
  runner._dynamics = runner.wm_env.dynamics = runner._wm.dynamics = stochastic.dynamics
  runner.cfg.process_noise = stochastic.dynamics.process_noise_config
  provider, _ = runner.configure_belief_filter()
  replay = runner.replay_buffer
  first = runner._episode_id.clone()
  for _ in range(2):
    runner.step()
    episodes, priors = provider.state.head_priors
    for row, episode in enumerate(episodes.tolist()):
      recorded, frames = replay.episode_prefix(episode, 0)
      assert frames["time_index"].tolist() == [0]
      assert torch.equal(recorded.mean.to_full(), priors.mean.to_full()[row])
      assert torch.equal(recorded.covariance, priors.covariance[row])
      recorded.mean.position.zero_()
      assert torch.equal(
        replay.episode_prefix(episode, 0)[0].mean.position, priors.mean.position[row]
      )
  # The asynchronous reset was at sequence one, between saved posteriors.
  unsaved = int(episodes.item())
  assert unsaved not in replay.posterior_frames()["episode_id"].tolist()
  bytes_per_head = (76 + 2485) * replay._storage["actor_obs"].element_size()
  assert replay.episode_head_storage_bytes == (len(first) + 1) * bytes_per_head
  runner.step()
  assert replay.episode_head_storage_bytes == bytes_per_head
  for episode in first.tolist():
    with pytest.raises(ValueError, match="head or initial prior unavailable"):
      replay.episode_prefix(episode, 0)
  assert replay.episode_prefix(unsaved, 1)[1]["time_index"].tolist() == [0, 1]


def test_custom_correlated_head_preserves_float32_factor_exactly():
  replay = replay_fixture(8, 1, 5)
  b = belief(3)
  b.covariance_sqrt[:, 1, 0] = 0.25
  b.mean.position.fill_(1.001)
  frame = replay._storage[:, 0].clone()
  frame["episode_id"] = torch.arange(3)
  replay.insert(
    frame,
    head_priors=(frame["episode_id"], b),
    posterior=replay.posterior_cache.capture(replay, b, 0),
  )
  restored, _ = replay.episode_prefix(1, 0)
  assert torch.equal(restored.mean.position, b.mean.position[1])
  assert torch.equal(restored.covariance_sqrt, b.covariance_sqrt[1])


def test_revision_rejects_updates_during_collection_and_entire_policy_update():
  loop = training()
  advance(loop)
  model = loop.runner._dynamics
  before = model.model_revision
  with model.readonly():
    with pytest.raises(ValueError, match="forbidden"):
      loop.learner.update_covariance(None)
    with pytest.raises(ValueError, match="forbidden"):
      model.load_state_dict(model.state_dict())
  assert model.model_revision == before
  start = loop.runner.wm_env.prepare_policy_starts(
    torch.tensor([1, 1]), torch.tensor([0, 1])
  )
  noises = loop.policy.draw_noises(start)
  original = loop.policy.critic_optimizers[0].step

  def forbidden_step(*args, **kwargs):
    original(*args, **kwargs)
    model.begin_write()

  with patch.object(
    loop.policy.critic_optimizers[0], "step", side_effect=forbidden_step
  ):
    with pytest.raises(ValueError, match="forbidden"):
      loop.policy.update_from(start, *noises)


def test_checkpoint_weight_load_invalidates_prepared_start_even_at_same_revision():
  loop = training()
  advance(loop)
  env, model = loop.runner.wm_env, loop.runner._dynamics
  start = env.prepare_policy_starts(torch.tensor([1]), torch.tensor([0]))
  checkpoint = model.checkpoint()
  candidate = env.posterior_provider.prepare_restore(
    start.belief,
    start.batch["sim_ctx"],
    model_version=start.model_version,
    episode_id=start.batch["episode_id"],
    time_index=start.batch["time_index"],
  )
  model.load_checkpoint(checkpoint)
  assert model.model_revision == start.model_version
  with pytest.raises(ValueError, match="Stale rollout"):
    env.reset_prepared(start)
  with pytest.raises(ValueError, match="Model changed"):
    env.posterior_provider.commit(candidate)


@pytest.mark.parametrize("stride", [0, -1, 9, True, 1.5])
def test_invalid_stride_is_rejected_before_allocating_cache(stride):
  with pytest.raises(ValueError, match="posterior_save_stride"):
    replay_fixture(8, stride, 5)


def test_sparse_policy_sampler_rejects_missing_starts_without_filter_fallback():
  loop = training()
  loop.runner.replay_buffer.enable_posterior_cache(4, 5)
  loop.runner.posterior_save_stride = 4
  loop.cfg = replace(loop.cfg, posterior_save_stride=4)
  advance(loop)
  env = loop.runner.wm_env
  with pytest.raises(ValueError, match="missing or overwritten"):
    env.prepare_policy_starts(torch.tensor([1]), torch.tensor([1]))
  with pytest.raises(ValueError, match="No training frames"):
    env.sample_policy_keys(1, [12345])
  episodes, times = env.sample_policy_keys(100, [1])
  assert (episodes == 1).all() and (times == 0).all()
  # Model training still includes every valid physical pair, including t=1,2.
  assert len(loop.runner.replay_buffer.model_window_starts([1], 2)) == 3
