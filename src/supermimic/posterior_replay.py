"""Sparse Q[t|t] storage and exact causal histories across replay overwrites."""

from functools import cached_property

import torch
from tensordict import TensorDict

from supermimic.belief import BeliefState
from supermimic.utils.lie_state import LieState

POSTERIOR_REPLAY_VERSION = "se3_body_posterior_sqrt_tril_f32_v3"


def pack_mean(mean):
  return torch.cat(
    (
      mean.rotation.flatten(-2),
      mean.position,
      mean.velocity,
      mean.omega,
      mean.joint_pos,
      mean.joint_vel,
    ),
    -1,
  )


def unpack_mean(mean):
  if mean.shape[-1] != 76:
    raise ValueError("Packed posterior mean requires 76 entries")
  return LieState(
    mean[..., :9].reshape(*mean.shape[:-1], 3, 3),
    mean[..., 9:12],
    mean[..., 12:15],
    mean[..., 15:18],
    mean[..., 18:47],
    mean[..., 47:76],
  )


def pack_belief(belief):
  packed = pack_mean(belief.mean)
  rows, cols = torch.tril_indices(70, 70, device=packed.device)
  return packed.detach().to(torch.float32).clone(), belief.covariance_sqrt[
    ..., rows, cols
  ].detach().to(torch.float32).clone()


def unpack_belief(mean, lower, *, dtype):
  rows, cols = torch.tril_indices(70, 70, device=mean.device)
  factor = lower.new_zeros(*lower.shape[:-1], 70, 70)
  factor[..., rows, cols] = lower
  return BeliefState(unpack_mean(mean), covariance_sqrt=factor).to(dtype=dtype)


class PosteriorReplay:
  @staticmethod
  def validate_config(buffer_size, stride, history_length):
    if type(stride) is not int or not 1 <= stride <= buffer_size:
      raise ValueError("posterior_save_stride must be between 1 and buffer_size")
    if type(history_length) is not int or history_length < 1:
      raise ValueError("history_length must be positive")

  def __init__(self, replay, stride, history_length):
    self.validate_config(replay.buffer_size, stride, history_length)
    self.stride, self.history_length = stride, history_length
    self.capacity = (replay.buffer_size + stride - 1) // stride
    e, c, device = replay.num_envs, self.capacity, replay.device
    self.sequences = torch.full((c,), -1, dtype=torch.long, device=device)
    self.frame_sequences = torch.full(
      (replay.buffer_size,), -1, dtype=torch.long, device=device
    )
    h = history_length - 1
    self.boundary = TensorDict(
      {
        "actor_obs": replay._storage["actor_obs"].new_zeros(e, h, 93),
        "episode_id": torch.full((e, h), -1, dtype=torch.long, device=device),
        "time_index": torch.full((e, h), -1, dtype=torch.long, device=device),
      },
      [e, h],
    )
    self.history_fields = ("actor_obs",)
    if "tracking_error" in replay._storage.keys():
      self.history_fields += ("tracking_error",)
      self.boundary["tracking_error"] = replay._storage["tracking_error"].new_zeros(
        e, h, 9
      )
    self.boundary_sequences = torch.full((h,), -1, dtype=torch.long, device=device)

  @cached_property
  @torch.inference_mode(False)
  def storage(self):
    # Fresh-process restore may select a different stride before collection.
    # Materialize the large ring once, before the first physical action.
    e, c, device = self.boundary.batch_size[0], self.capacity, self.sequences.device
    return TensorDict(
      {
        "posterior_mean": torch.zeros(e, c, 76, device=device, dtype=torch.float32),
        "posterior_sqrt_tril": torch.zeros(
          e, c, 2485, device=device, dtype=torch.float32
        ),
        "posterior_revision": torch.full((e, c), -1, dtype=torch.long, device=device),
      },
      [e, c],
    )

  def capture(self, replay, belief, revision):
    """Copy before env.step: simulators may mutate aliased input tensors."""
    if replay.insert_count % self.stride:
      return None
    if type(revision) is not int or revision < 0:
      raise ValueError("Posterior source revision must be a nonnegative integer")
    if belief.covariance_sqrt.shape != (replay.num_envs, 70, 70):
      raise ValueError("Posterior batch must match replay environments")
    mean, lower = pack_belief(belief)
    _ = self.storage
    return mean, lower, revision

  def insert(self, replay, posterior):
    seq, slot = replay.insert_count, replay._step
    if seq % self.stride == 0:
      if posterior is None:
        raise ValueError("Saved replay frames require a captured posterior")
      mean, lower, revision = posterior
      if mean.shape != (replay.num_envs, 76) or lower.shape != (replay.num_envs, 2485):
        raise ValueError("Invalid packed posterior shape")
    elif posterior is not None:
      raise ValueError("Posterior supplied outside the global save stride")
    # Preserve exactly the latest H-1 evictions, including episode/time identity.
    h = self.history_length - 1
    if h and replay.num_transitions == replay.buffer_size:
      old = seq - replay.buffer_size
      boundary_slot = old % h
      self.boundary[:, boundary_slot] = replay._storage.select(*self.boundary.keys())[
        :, slot
      ]
      self.boundary_sequences[boundary_slot] = old
    if posterior is not None:
      pslot = (seq // self.stride) % self.capacity
      self.storage["posterior_mean"][:, pslot] = mean
      self.storage["posterior_sqrt_tril"][:, pslot] = lower
      self.storage["posterior_revision"][:, pslot] = revision
      self.sequences[pslot] = seq
    self.frame_sequences[slot] = seq

  def saved_ordinals(self, replay, *, newest_count=None) -> range:
    """Retained multiples of stride, in units of saved steps per environment."""
    end = replay.insert_count
    count = replay.num_transitions
    if newest_count is not None:
      count = min(count, max(0, newest_count))
    first = end - count
    return range(
      (first + self.stride - 1) // self.stride,
      (end + self.stride - 1) // self.stride,
    )

  def locations_at(self, replay, saved: range, indices: torch.Tensor):
    """Map flat row-major draws to both rings, preserving order and duplicates."""
    if not len(indices):
      return indices, indices, indices
    rows = indices // len(saved)
    ordinals = saved.start + indices % len(saved)
    return rows, (ordinals * self.stride) % replay.buffer_size, ordinals % self.capacity

  def locations(self, replay, episode_ids=None, newest_count=None):
    saved = self.saved_ordinals(replay, newest_count=newest_count)
    indices = torch.arange(replay.num_envs * len(saved), device=replay.device)
    rows, slots, pslots = self.locations_at(replay, saved, indices)
    if episode_ids is not None:
      mask = torch.isin(
        replay._storage["episode_id"][rows, slots],
        torch.as_tensor(list(episode_ids), device=replay.device, dtype=torch.long),
      )
      rows, slots, pslots = rows[mask], slots[mask], pslots[mask]
    return rows, slots, pslots

  def frames(self, replay, episode_ids=None, *, newest_count=None, fields=None):
    rows, slots, _ = self.locations(replay, episode_ids, newest_count)
    storage = replay._storage if fields is None else replay._storage.select(*fields)
    return storage[rows, slots].detach().clone()

  def lookup(self, replay, keys):
    if (
      not isinstance(keys, torch.Tensor)
      or keys.ndim != 2
      or keys.shape[1] != 2
      or not len(keys)
      or keys.dtype != torch.long
      or (keys < 0).any()
    ):
      raise ValueError("Posterior keys must be nonempty nonnegative int64 pairs")
    locations = self.locations(replay)
    index = self._match_frame_indices(
      replay, locations, keys[:, 0], keys[:, 1], require_all=True
    )
    return tuple(value[index] for value in locations)

  def frame_mask(self, replay, locations, frames):
    """Mark saved slots appearing in an explicit selection, once per slot."""
    index = self._match_frame_indices(
      replay, locations, frames["episode_id"], frames["time_index"], require_all=False
    )
    mask = torch.zeros_like(locations[0], dtype=torch.bool)
    mask[index] = True
    return mask

  @staticmethod
  def _match_frame_indices(
    replay, locations, wanted_episodes, wanted_times, *, require_all
  ):
    rows, slots, _ = locations
    if not len(wanted_times):
      return rows[:0]
    if not len(rows):
      if require_all:
        raise ValueError("No saved posterior available for policy starts")
      return rows[:0]
    episodes, times = (
      replay._storage["episode_id"][rows, slots],
      replay._storage["time_index"][rows, slots],
    )
    # Batched key lookup preserves duplicates and caller order without a B x replay mask.
    width = torch.maximum(times.max(), wanted_times.max()) + 1
    encoded = episodes * width + times
    ordered, order = encoded.sort()
    requested = wanted_episodes * width + wanted_times
    index = torch.searchsorted(ordered, requested).clamp(max=len(ordered) - 1)
    index = order[index]
    matches = (episodes[index] == wanted_episodes) & (times[index] == wanted_times)
    if require_all:
      if not matches.all():
        raise ValueError("Policy start posterior missing or overwritten in replay")
      return index
    return index[matches]

  def read(self, replay, keys=None, *, locations=None, include_belief=True):
    rows, slots, pslots = self.lookup(replay, keys) if locations is None else locations
    batch = replay._storage[rows, slots].detach().clone()
    batch["sensor_window"] = self.history(replay, rows, slots)
    batch["segment_time"] = batch["time_index"].clone()
    if not include_belief:
      return batch, None
    batch["posterior_revision"] = self.storage["posterior_revision"][
      rows, pslots
    ].clone()
    batch["insert_sequence"] = self.sequences[pslots].clone()
    belief = unpack_belief(
      self.storage["posterior_mean"][rows, pslots],
      self.storage["posterior_sqrt_tril"][rows, pslots],
      dtype=batch["state"].dtype,
    )
    return batch, belief

  def history(self, replay, rows, slots):
    seq = self.frame_sequences[slots]
    times = replay._storage["time_index"][rows, slots]
    episode = replay._storage["episode_id"][rows, slots]
    offsets = torch.arange(1 - self.history_length, 1, device=replay.device)
    expected_time = (times[:, None] + offsets).clamp_min(0)
    wanted = seq[:, None] - times[:, None] + expected_time
    row_grid = rows[:, None].expand_as(wanted)
    main = wanted >= replay.insert_count - replay.num_transitions
    source_slots = wanted % replay.buffer_size
    obs = torch.cat(
      [replay._storage[key][row_grid, source_slots] for key in self.history_fields], -1
    )
    got_seq = self.frame_sequences[source_slots].clone()
    got_ep = replay._storage["episode_id"][row_grid, source_slots].clone()
    got_time = replay._storage["time_index"][row_grid, source_slots].clone()
    if (~main).any():
      h = self.history_length - 1
      if not h:
        raise ValueError("Unexplained replay history gap")
      boundary_slots = wanted[~main] % h
      boundary_rows = row_grid[~main]
      obs[~main] = torch.cat(
        [
          self.boundary[key][boundary_rows, boundary_slots]
          for key in self.history_fields
        ],
        -1,
      )
      got_ep[~main] = self.boundary["episode_id"][boundary_rows, boundary_slots]
      got_time[~main] = self.boundary["time_index"][boundary_rows, boundary_slots]
      got_seq[~main] = self.boundary_sequences[boundary_slots]
    if not (
      torch.equal(got_seq, wanted)
      and torch.equal(got_time, expected_time)
      and torch.equal(got_ep, episode[:, None].expand_as(wanted))
    ):
      raise ValueError("Unexplained replay history gap or episode boundary mismatch")
    return obs

  def storage_bytes(self):
    def size(values):
      return sum(t.numel() * t.element_size() for t in values)

    return {
      "posterior_storage_bytes": size(vars(self).get("storage", {}).values())
      + size([self.sequences]),
      "boundary_history_storage_bytes": size(self.boundary.values())
      + size([self.boundary_sequences]),
      "replay_sequence_storage_bytes": size([self.frame_sequences]),
    }
