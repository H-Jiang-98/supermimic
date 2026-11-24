"""Replay checks are transactional at the write boundary and bounded per frame."""

import pytest
import torch
from tensordict import TensorDict

from supermimic.replay_buffer import FieldSpec, ReplayBuffer
from tests.test_estimator_posterior_replay import belief, replay_fixture


def frame(step, num_envs=3):
  state = torch.zeros(num_envs, 71)
  state[:, 3] = 1
  previous = torch.full((num_envs, 29), float(step))
  return TensorDict(
    {
      "state": state,
      "action": previous + 1,
      "prev_action": previous,
      "actor_obs": torch.cat((torch.zeros(num_envs, 64), previous), -1),
      "sim_ctx": torch.zeros(num_envs, 2),
      "episode_id": torch.arange(num_envs),
      "time_index": torch.full((num_envs,), step, dtype=torch.long),
      "terminated": torch.zeros(num_envs, dtype=torch.bool),
      "truncated": torch.zeros(num_envs, dtype=torch.bool),
    },
    [num_envs],
  )


@pytest.mark.parametrize("field", ["state", "action", "sim_ctx"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), 1e100])
def test_invalid_insert_preserves_ring_posterior_index_and_retry_token(field, bad):
  replay = replay_fixture(4, 2, 3)
  cache = replay.posterior_cache
  b = belief()
  for step in range(4):
    replay.insert(frame(step), posterior=cache.capture(replay, b, 0))
  replay.finish_collection_round()
  index, rng = replay._round_index, torch.random.get_rng_state().clone()
  before = replay._storage.clone()
  saved = cache.storage.clone()
  sequences = cache.sequences.clone()
  boundary = cache.boundary.clone()
  token = object()
  candidate = frame(4)
  candidate[field] = candidate[field].double()
  candidate[field][1, 0] = bad
  posterior = cache.capture(replay, b, 1)
  with pytest.raises(ValueError, match=f"Non-finite replay {field}"):
    replay.insert(candidate, token=token, posterior=posterior)
  torch.testing.assert_close(replay._storage, before)
  torch.testing.assert_close(cache.storage, saved)
  torch.testing.assert_close(cache.boundary, boundary)
  assert torch.equal(cache.sequences, sequences)
  assert replay._round_index is index and replay.insert_count == 4 and replay._step == 0
  assert torch.equal(torch.random.get_rng_state(), rng)
  # The rejected token must not suppress a corrected retry.
  replay.insert(frame(4), token=token, posterior=posterior)
  replay.insert(frame(4), token=token, posterior=posterior)
  assert replay.insert_count == 5


@pytest.mark.parametrize("capacity", [1, 4, 17])
def test_writes_and_audits_survive_multiple_wraps_with_bounded_numeric_checks(
  capacity, monkeypatch
):
  replay = replay_fixture(capacity, 1, 3)
  cache, b = replay.posterior_cache, belief()
  original = torch.isfinite
  sizes = []

  def checked(value):
    sizes.append(value.numel())
    assert value.numel() <= 3 * 71
    return original(value)

  with monkeypatch.context() as context:
    context.setattr(torch, "isfinite", checked)
    for step in range(capacity * 3 + 2):
      replay.insert(frame(step), posterior=cache.capture(replay, b, 0))
      replay.finish_collection_round()
    before = torch.random.get_rng_state().clone()
    replay.validate_contents()
    assert torch.equal(torch.random.get_rng_state(), before)
  assert sizes and max(sizes) == 3 * 71
  starts = replay.model_window_starts([0], 2)
  expected = [[0, (replay._step + age) % capacity] for age in range(capacity - 1)]
  assert starts.tolist() == expected


@pytest.mark.parametrize("field", ["state", "action", "sim_ctx"])
def test_explicit_audit_detects_private_storage_corruption(field):
  replay = replay_fixture(4, 1, 3)
  for step in range(6):
    replay.insert(
      frame(step), posterior=replay.posterior_cache.capture(replay, belief(), 0)
    )
  replay.finish_collection_round()
  # This deliberately bypasses the validated write API.
  replay._storage[field][0, (replay._step + 1) % 4, 0] = float("nan")
  with pytest.raises(ValueError, match=f"Non-finite replay {field}"):
    replay.validate_contents()


def test_insert_detaches_storage_from_caller_changes_and_audit_handles_empty():
  replay = ReplayBuffer(
    {
      "state": FieldSpec(2),
      "terminated": FieldSpec(dtype=torch.bool),
      "truncated": FieldSpec(dtype=torch.bool),
    },
    3,
    2,
  )
  replay.validate_contents()
  value = TensorDict(
    {
      "state": torch.ones(2, 2),
      "terminated": torch.zeros(2, dtype=torch.bool),
      "truncated": torch.zeros(2, dtype=torch.bool),
    },
    [2],
  )
  replay.insert(value)
  value["state"].fill_(float("nan"))
  replay.validate_contents()


@pytest.mark.parametrize(
  "field,message",
  [
    ("actor_obs", "Recorded sensor action"),
    ("prev_action", "Executed action"),
    ("sim_ctx", "simulation context"),
  ],
)
def test_explicit_audit_checks_retained_successor_edges_after_wrap(field, message):
  replay = replay_fixture(4, 1, 3)
  for step in range(6):
    replay.insert(
      frame(step), posterior=replay.posterior_cache.capture(replay, belief(), 0)
    )
  slot = (replay._step + 1) % 4
  if field == "actor_obs":
    replay._storage[field][0, slot, 64] += 0.5
  else:
    replay._storage[field][0, slot, 0] += 0.5
    if field == "prev_action":
      replay._storage["actor_obs"][0, slot, 64] += 0.5
  with pytest.raises(ValueError, match=message):
    replay.validate_contents()


def test_invalid_rotation_is_rejected_before_commit():
  replay = replay_fixture(4, 1, 3)
  candidate = frame(0)
  candidate["state"][:, 3:7] = 0
  with pytest.raises(ValueError, match="rotation"):
    replay.insert(candidate)
  assert replay.insert_count == replay.num_transitions == replay._step == 0
