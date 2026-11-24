from dataclasses import dataclass
from typing import Literal

import torch
from tensordict import TensorDict

from supermimic.belief import BeliefState, GaussianStatePrior
from supermimic.posterior_replay import (
  PosteriorReplay,
  pack_mean,
  unpack_belief,
)
from supermimic.utils.lie_state import LieState
from supermimic.utils.validation import validate_rotation

MODEL_REPLAY_FIELDS = frozenset(
  ("state", "action", "sim_ctx", "episode_id", "time_index", "terminated", "truncated")
)


@dataclass(frozen=True)
class FieldSpec:
  dim: int | tuple[int, ...] = 0  # 0 = scalar; tuple preserves structured fields
  dtype: torch.dtype = torch.float32

  def shape(self, num_envs: int, buffer_size: int) -> tuple[int, ...]:
    if isinstance(self.dim, tuple):
      return (num_envs, buffer_size, *self.dim)
    return (num_envs, buffer_size, self.dim) if self.dim else (num_envs, buffer_size)


@dataclass(frozen=True)
class _RecordedHeadPrior:
  """Native-precision square-root reset seed for explicit diagnostics."""

  mean: torch.Tensor
  covariance_sqrt: torch.Tensor

  @staticmethod
  def capture(priors):
    means = pack_mean(priors.mean).detach().clone()
    factor = priors.covariance_sqrt.detach()
    rows, cols = torch.tril_indices(70, 70, device=factor.device)
    packed = factor[:, rows, cols].clone()
    return [
      _RecordedHeadPrior(mean, lower) for mean, lower in zip(means, packed, strict=True)
    ]

  def restore(self):
    return unpack_belief(self.mean.clone(), self.covariance_sqrt, dtype=self.mean.dtype)

  @property
  def nbytes(self):
    return self.mean.nbytes + self.covariance_sqrt.nbytes


@dataclass(frozen=True)
class RecordedEpisodeHeads:
  replay: "ReplayBuffer"
  insert_count: int
  episode_ids: torch.Tensor
  priors: list[_RecordedHeadPrior]


@dataclass(frozen=True)
class _ReplayRound:
  slots: torch.Tensor
  episodes: torch.Tensor
  successors: torch.Tensor
  episode_lengths: dict[int, int]

  def window_starts(
    self, window_frames: int, *episode_groups
  ) -> tuple[torch.Tensor, ...]:
    """Enumerate each group's starts in environment then chronological order."""
    if self.slots.numel() < window_frames:
      return tuple(self.slots.new_empty((0, 2)) for _ in episode_groups)
    complete = self.successors
    if window_frames > 2:
      complete = complete.unfold(1, window_frames - 1, 1).all(-1)
    rows, ages = complete.nonzero(as_tuple=True)
    starts = torch.stack((rows, self.slots[ages]), dim=1)
    episodes = self.episodes[rows, ages]
    return tuple(
      starts[
        torch.isin(
          episodes, torch.as_tensor(ids, device=episodes.device, dtype=torch.long)
        )
      ]
      for ids in episode_groups
    )


class ReplayBuffer:
  """Per-env circular buffer over a TensorDict, with env-stratified window sampling."""

  def __init__(
    self,
    schema: dict[str, FieldSpec],
    buffer_size: int,
    num_envs: int,
    device: str = "cpu",
    *,
    initial_prior: GaussianStatePrior | None = None,
    replay_mode: Literal["full", "model"] = "full",
  ):
    if replay_mode not in ("full", "model"):
      raise ValueError(f"Unknown replay_mode: {replay_mode}")
    if replay_mode == "model" and schema.keys() != MODEL_REPLAY_FIELDS:
      raise ValueError("Model replay requires the seven model transition fields")
    if not {"terminated", "truncated"} <= schema.keys():
      raise ValueError("schema needs 'terminated' and 'truncated'")
    self.posterior_cache: PosteriorReplay | None = None
    self.replay_mode = replay_mode
    self._episode_heads: dict[int, tuple[int, int, _RecordedHeadPrior]] = {}
    self.episode_head_storage_bytes = 0
    self.initial_prior = (
      (initial_prior or GaussianStatePrior()) if replay_mode == "full" else None
    )
    self._training_heads: dict[int, tuple[int, int, int, BeliefState]] = {}
    self._round_index: _ReplayRound | None = None
    self.insert_count = 0
    self._last_insert_token = None
    self.num_transitions = 0
    self.buffer_size = buffer_size
    self.num_envs = num_envs
    self.device = device
    self._step = 0
    self._storage = TensorDict(
      {
        name: torch.zeros(
          spec.shape(num_envs, buffer_size), dtype=spec.dtype, device=device
        )
        for name, spec in schema.items()
      },
      batch_size=[num_envs, buffer_size],
    )

  def enable_posterior_cache(self, stride: int, history_length: int) -> None:
    self.posterior_cache = self.prepare_posterior_cache(stride, history_length)

  def prepare_posterior_cache(
    self, stride: int, history_length: int
  ) -> PosteriorReplay:
    """Prepare an empty ring layout without mutating replay during preflight."""
    if self.replay_mode != "full" or self.insert_count:
      raise ValueError("Enable posterior replay before collecting full replay")
    PosteriorReplay.validate_config(self.buffer_size, stride, history_length)
    if self.posterior_cache is not None:
      if (self.posterior_cache.stride, self.posterior_cache.history_length) == (
        stride,
        history_length,
      ):
        return self.posterior_cache
    return PosteriorReplay(self, stride, history_length)

  def posterior_frames(self, *args, **kwargs):
    if self.posterior_cache is None:
      raise ValueError("Posterior replay is not enabled")
    return self.posterior_cache.frames(self, *args, **kwargs)

  def validate_head_priors(
    self,
    episode_id: torch.Tensor,
    time_index: torch.Tensor,
    head_priors: tuple[torch.Tensor, BeliefState] | None,
  ) -> None:
    """Check the replay seeds before a real environment can advance."""
    if self.replay_mode == "model":
      raise ValueError("Episode head priors require full replay")
    heads = episode_id[time_index == 0]
    if head_priors is None:
      if heads.numel():
        raise ValueError("Frame-zero replay requires the realized episode head priors")
      return
    ids, priors = head_priors
    if (
      ids.dtype != torch.long
      or ids.device != heads.device
      or not torch.equal(ids, heads)
      or ids.unique().numel() != heads.numel()
      or not isinstance(priors, BeliefState)
      or priors.covariance_sqrt.shape[:-2] != heads.shape
    ):
      raise ValueError("Recorded priors must match the episode heads being inserted")
    sensor = self._storage["actor_obs"]
    if (
      priors.covariance_sqrt.device != sensor.device
      or priors.covariance_sqrt.dtype != sensor.dtype
    ):
      raise ValueError(
        "Episode head priors must match replay measurement dtype and device"
      )
    priors.validate()

  def capture_head_priors(self, episode_id, time_index, head_priors):
    """Validate and snapshot physical reset seeds before advancing the simulator."""
    self.validate_head_priors(episode_id, time_index, head_priors)
    if head_priors is None:
      return None
    ids, priors = head_priors
    return RecordedEpisodeHeads(
      self, self.insert_count, ids.detach().clone(), _RecordedHeadPrior.capture(priors)
    )

  def validate_previous_action(
    self,
    episode_id: torch.Tensor,
    time_index: torch.Tensor,
    previous_action: torch.Tensor,
  ) -> None:
    """Check the current measurement against the last committed physical action."""
    if not self.num_transitions:
      return
    last = (self._step - 1) % self.buffer_size
    consecutive = (
      (self._storage["episode_id"][:, last] == episode_id)
      & (self._storage["time_index"][:, last] + 1 == time_index)
      & ~(self._storage["terminated"][:, last] | self._storage["truncated"][:, last])
    )
    action = self._storage["action"][:, last]
    if (
      action.shape != previous_action.shape
      or (consecutive & (action != previous_action).any(dim=-1)).any()
    ):
      raise ValueError("Executed action disagrees with the successor's previous action")

  def insert(
    self,
    td: TensorDict,
    *,
    head_priors=None,
    require_head_priors: bool = False,
    token=None,
    posterior=None,
  ) -> None:
    """Store a batched step. Optional live priors support exact episode diagnostics."""
    if token is not None and token is self._last_insert_token:
      return
    if td.batch_size != torch.Size([self.num_envs]):
      raise ValueError(
        f"insert expects batch_size=[{self.num_envs}], got {list(td.batch_size)}"
      )
    if set(td.keys()) != set(self._storage.keys()):
      raise ValueError("schema mismatch")
    for name, value in td.items():
      target = self._storage[name][:, self._step]
      if value.shape != target.shape:
        raise ValueError(f"Invalid replay field shape for {name}")
    if isinstance(head_priors, RecordedEpisodeHeads):
      # Contents were validated and detached before env.step. Check the identity
      # again at commit without another full covariance eigendecomposition.
      if (
        head_priors.replay is not self
        or head_priors.insert_count != self.insert_count
        or not torch.equal(
          td["episode_id"][td["time_index"] == 0], head_priors.episode_ids
        )
      ):
        raise ValueError("Captured physical priors disagree with committing frame")
    elif head_priors is not None or (require_head_priors and "episode_id" in td.keys()):
      head_priors = self.capture_head_priors(
        td["episode_id"], td["time_index"], head_priors
      )
    new_heads = {}
    if head_priors is not None:
      rows = (td["time_index"] == 0).nonzero().flatten().tolist()
      new_heads = {
        episode: (env_row, self._step, prior)
        for episode, env_row, prior in zip(
          head_priors.episode_ids.tolist(), rows, head_priors.priors, strict=True
        )
      }
    expired = []
    if self._episode_heads and self.num_transitions == self.buffer_size:
      oldest = self._storage[:, self._step]
      expired = oldest["episode_id"][oldest["time_index"] == 0].tolist()
    if self.posterior_cache is not None:
      self.posterior_cache.insert(self, posterior)
    self._storage[:, self._step] = td
    for episode in expired:
      old_head = self._episode_heads.pop(episode, None)
      if old_head is not None:
        self.episode_head_storage_bytes -= old_head[2].nbytes
    self.episode_head_storage_bytes += sum(
      head[2].nbytes for head in new_heads.values()
    )
    self._episode_heads.update(new_heads)
    self._step = (self._step + 1) % self.buffer_size
    self.num_transitions = min(self.buffer_size, self.num_transitions + 1)
    self.insert_count += 1
    self._last_insert_token = token
    self._round_index = None

  def finish_collection_round(self) -> None:
    """Prepare the retained data for fitting and, in full mode, belief reconstruction."""
    self._sealed_round()

  @torch.no_grad()
  def _sealed_round(self) -> _ReplayRound:
    """Validate and index retained transitions once per sealed collection round."""
    if self._round_index is not None:
      return self._round_index
    if "episode_id" not in self._storage.keys() or not self.num_transitions:
      raise ValueError("A training segment requires retained estimator frames")
    base = self._step if self.num_transitions == self.buffer_size else 0
    slots = (
      base + torch.arange(self.num_transitions, device=self.device)
    ) % self.buffer_size
    episodes = self._storage["episode_id"][:, slots]
    n = self.num_transitions
    for field in ("state", "action", "sim_ctx"):
      if not torch.isfinite(self._storage[field][:, :n]).all():
        raise ValueError(f"Non-finite replay {field}")
    # Compare only recorded physical successors, including the pair across wrap.
    successors = self._recorded_successors(slots, episodes)
    rows, ages = successors.nonzero(as_tuple=True)
    current, successor = slots[ages], slots[ages + 1]
    if self.replay_mode == "full":
      previous_actions = self._storage["prev_action"]
      if not torch.equal(
        self._storage["actor_obs"][:, :n, 64:], previous_actions[:, :n]
      ):
        raise ValueError("Recorded sensor action disagrees with the previous action")
      if not torch.equal(
        self._storage["action"][rows, current], previous_actions[rows, successor]
      ):
        raise ValueError(
          "Executed action disagrees with the successor's previous action"
        )
    context = self._storage["sim_ctx"]
    if not torch.equal(context[rows, current], context[rows, successor]):
      raise ValueError("Inconsistent simulation context across a replay transition")

    begins = torch.ones_like(episodes, dtype=torch.bool)
    begins[:, 1:] = episodes[:, 1:] != episodes[:, :-1]
    rows, ages = begins.nonzero(as_tuple=True)
    ids = episodes[rows, ages].tolist()
    if len(set(ids)) != len(ids):
      raise ValueError(
        "Replay episode IDs must identify a single contiguous environment segment"
      )
    # Every environment starts a segment at age zero, so differences of flat
    # offsets also give the correct length for the last segment of each row.
    offsets = rows * n + ages
    lengths = offsets.diff(append=offsets.new_tensor([self.num_envs * n]))
    episode_lengths = dict(sorted(zip(ids, lengths.tolist(), strict=True)))
    if self.replay_mode == "full" and self.posterior_cache is None:
      self._realize_training_heads(rows, slots[ages], ids)
    else:
      # Full replay checks these rotations when drawing the segment priors.
      head_states = self._storage["state"][rows, slots[ages]]
      validate_rotation(LieState.from_full(head_states).rotation)
    self._round_index = _ReplayRound(slots, episodes, successors, episode_lengths)
    return self._round_index

  def _realize_training_heads(self, rows, start_slots, ids) -> None:
    times = self._storage["time_index"][rows, start_slots].tolist()
    row_numbers, slot_numbers = rows.tolist(), start_slots.tolist()
    heads = {}
    missing = []
    for i, (episode, row, slot, t) in enumerate(
      zip(ids, row_numbers, slot_numbers, times, strict=True)
    ):
      previous = self._training_heads.get(episode)
      if previous is not None and previous[:3] == (row, slot, t):
        heads[episode] = previous
      else:
        missing.append(i)
    if missing:
      assert self.initial_prior is not None
      full = self._storage["state"][rows[missing], start_slots[missing]]
      priors = self.initial_prior.sample(LieState.from_full(full))
      for j, i in enumerate(missing):
        # Each row owns its mean; the fixed diagonal covariance can be shared.
        prior = BeliefState(
          priors.mean.index(j).clone(), covariance_sqrt=priors.covariance_sqrt[j]
        )
        heads[ids[i]] = (row_numbers[i], slot_numbers[i], times[i], prior)
    self._training_heads = heads

  def training_prefix(
    self, episode: int, t: int, *, relative: bool = False
  ) -> tuple[BeliefState, TensorDict]:
    """Return the realized prior and frames from retained segment frame zero to t."""
    if self.replay_mode == "model":
      raise ValueError("Belief prefix reconstruction requires full replay")
    self.finish_collection_round()
    if self.posterior_cache is not None:
      index = self._sealed_round()
      begins = torch.ones_like(index.episodes, dtype=torch.bool)
      begins[:, 1:] = index.episodes[:, 1:] != index.episodes[:, :-1]
      rows, ages = begins.nonzero(as_tuple=True)
      self._realize_training_heads(
        rows, index.slots[ages], index.episodes[rows, ages].tolist()
      )
    head = self._training_heads.get(episode)
    if head is None:
      raise ValueError(f"Episode {episode} unavailable in replay")
    row, slot, first, prior = head
    length = t if relative else t - first
    frames = self._gather_prefix(episode, row, slot, first, length)
    frames["segment_time"] = frames["time_index"] - first
    return prior.clone(), frames

  def retained_frames(
    self,
    episode_ids=None,
    *,
    newest_count: int | None = None,
    fields: tuple[str, ...] | None = None,
  ) -> TensorDict:
    """Chronological retained frames, optionally restricted to training episodes.

    A detached copy prevents normalization or callers from modifying replay.
    Episode filtering also keeps validation measurements out of normalization.
    """
    base = self._step if self.num_transitions == self.buffer_size else 0
    slots = (
      base + torch.arange(self.num_transitions, device=self.device)
    ) % self.buffer_size
    if newest_count is not None:
      if not isinstance(newest_count, int) or newest_count < 1:
        raise ValueError("newest_count must be a positive integer")
      slots = slots[-newest_count:]
    rows = torch.arange(self.num_envs, device=self.device)[:, None].expand(
      self.num_envs, len(slots)
    )
    slots = slots.expand_as(rows)
    if episode_ids is not None:
      ids = torch.as_tensor(list(episode_ids), device=self.device, dtype=torch.long)
      selected = torch.isin(self._storage["episode_id"][rows, slots], ids)
      rows, slots = rows[selected], slots[selected]
    storage = self._storage if fields is None else self._storage.select(*fields)
    return storage[rows.reshape(-1), slots.reshape(-1)].detach().clone()

  def training_episode_lengths(self) -> dict[int, int]:
    """Retained frame counts for sealed fragments, including singleton episodes."""
    return self._sealed_round().episode_lengths.copy()

  def _recorded_successors(self, slots, episodes) -> torch.Tensor:
    times = self._storage["time_index"][:, slots]
    source = slots[:-1]
    done = (
      self._storage["terminated"][:, source] | self._storage["truncated"][:, source]
    )
    return (
      (episodes[:, :-1] == episodes[:, 1:])
      & (times[:, 1:] == times[:, :-1] + 1)
      & ~done
    )

  def model_window_starts(self, episode_ids, window_frames: int) -> torch.Tensor:
    """Enumerate (environment, physical slot) for each complete, consecutive window."""
    if type(window_frames) is not int or window_frames < 2:
      raise ValueError("A model window needs at least two frames")
    if self.num_transitions < window_frames:
      return torch.empty((0, 2), device=self.device, dtype=torch.long)
    return self._sealed_round().window_starts(window_frames, episode_ids)[0]

  def model_fit_starts(
    self, train_ids, valid_ids, window_frames: int
  ) -> tuple[torch.Tensor, ...]:
    """Build both episode splits' mean windows and covariance pairs from one index."""
    if type(window_frames) is not int or window_frames < 2:
      raise ValueError("A model window needs at least two frames")
    if self.num_transitions < 2:
      empty = torch.empty((0, 2), device=self.device, dtype=torch.long)
      return empty, empty, empty, empty
    index = self._sealed_round()
    pairs = index.window_starts(2, train_ids, valid_ids)
    windows = (
      pairs
      if window_frames == 2
      else index.window_starts(window_frames, train_ids, valid_ids)
    )
    return windows + pairs

  def model_windows(self, starts: torch.Tensor, window_frames: int) -> TensorDict:
    """Gather model inputs using starts from the current sealed replay."""
    rows, slots = starts.unbind(-1)
    offsets = torch.arange(window_frames, device=self.device)
    storage = self._storage.select("state", "action", "sim_ctx")
    return storage[rows[:, None], (slots[:, None] + offsets) % self.buffer_size]

  def episode_prefix(self, episode: int, t: int) -> tuple[BeliefState, TensorDict]:
    """Gather frames 0..t in chronological order from one physical env slot."""
    if self.replay_mode == "model":
      raise ValueError("Episode prefix reconstruction requires full replay")
    head = self._episode_heads.get(episode)
    if head is None:
      raise ValueError(f"Episode {episode} head or initial prior unavailable in replay")
    env_row, start_slot, prior = head
    return prior.restore(), self._gather_prefix(episode, env_row, start_slot, 0, t)

  def _gather_prefix(self, episode, env_row, start_slot, first, length):
    if not 0 <= length < self.num_transitions:
      raise ValueError(
        f"Episode {episode} frame {first + length} unavailable in replay"
      )
    row = self._storage[env_row]
    count = self.num_transitions
    times = torch.arange(length + 1, device=self.device)
    slots = (start_slot + times) % self.buffer_size
    if (slots >= count).any():
      raise ValueError("Requested frame has not been recorded")
    frames = row[slots]
    if (
      (frames["episode_id"] != episode).any()
      or not torch.equal(frames["time_index"], first + times)
      or (frames["terminated"][:-1] | frames["truncated"][:-1]).any()
    ):
      raise ValueError("Replay prefix is incomplete or crosses an episode boundary")
    if (
      not torch.equal(frames["prev_action"][1:], frames["action"][:-1])
      or not torch.equal(frames["actor_obs"][:, 64:], frames["prev_action"])
      or not torch.equal(
        frames["sim_ctx"], frames["sim_ctx"][:1].expand_as(frames["sim_ctx"])
      )
    ):
      raise ValueError("Inconsistent executed actions or context in replay prefix")
    return frames

  def truncate_open_episodes(self) -> None:
    """End unfinished episodes before an explicit all-environment reset.

    The last inserted transition precedes the reset. Mark only unfinished rows
    as truncated, preserving real terminations/timeouts and the rest of replay.
    Both forward sequence sampling and causal history then respect the boundary.
    """
    if self.num_transitions == 0:
      return
    last = (self._step - 1) % self.buffer_size
    terminated = self._storage["terminated"][:, last]
    truncated = self._storage["truncated"][:, last]
    open_rows = ~(terminated | truncated)
    if open_rows.any():
      truncated.logical_or_(open_rows)
      self._round_index = None

  def sample_causal(
    self, sequence_length: int, batch_size: int
  ) -> tuple[TensorDict, torch.Tensor]:
    """Sample a window of ``L`` frames ENDING at a uniformly-drawn filled frame.

    The newest frame (index ``-1``) is the sampled "current" state — drawn like
    a 1-step ``sample``, so its distribution is independent of ``L``. Preceding
    slots hold the current frame's real in-episode predecessors; slots before
    the episode start (a stored ``terminated``/``truncated``) or before the
    oldest filled frame are clamped to the first in-episode frame, matching the
    real-sim history ring's reset padding.

    Returns ``(window, padded)``: the ``(B, L)`` window and a ``(B, L)`` bool
    mask that is ``True`` on the clamp-padded slots, so the caller can noise the
    oldest real frame once and share that realization across them.
    """
    L, B, E, T = sequence_length, batch_size, self.num_envs, self.buffer_size
    dev, n = self.device, self.num_transitions
    if L > T and "episode_id" not in self._storage.keys():
      raise ValueError(f"sequence_length {L} exceeds buffer_size {T}")
    if n < 1:
      raise RuntimeError("sample_causal on empty buffer")
    base = self._step if n == T else 0
    env_idx = torch.randint(E, (B,), device=dev)
    current_age = torch.randint(n, (B,), device=dev)
    ages = current_age[:, None] + (
      torch.arange(L, device=dev) - (L - 1)
    )  # (B, L), newest last
    ended = (self._storage["terminated"] | self._storage["truncated"])[
      env_idx[:, None], (base + ages.clamp(min=0)) % T
    ]
    # Reachable = contiguous in-episode suffix ending at the (always-valid) newest
    # frame; a stored done at an earlier slot ends the prior episode there.
    valid = (ages >= 0) & ~ended.bool()
    if "episode_id" in self._storage.keys():
      episodes = self._storage["episode_id"][
        env_idx[:, None], (base + ages.clamp(min=0)) % T
      ]
      valid &= episodes == episodes[:, -1:]
    valid[:, -1] = True
    reachable = torch.cummin(valid.flip(-1).int(), dim=-1).values.flip(-1).bool()
    first_age = torch.where(reachable, ages, current_age[:, None]).amin(
      dim=-1, keepdim=True
    )
    slot = (base + torch.where(reachable, ages, first_age)) % T
    return self._storage[env_idx[:, None], slot], ~reachable

  def sample(self, sequence_length: int, batch_size: int) -> TensorDict:
    L, B, E, T = sequence_length, batch_size, self.num_envs, self.buffer_size
    dev, n = self.device, self.num_transitions
    if L > T:
      raise ValueError(f"sequence_length {L} exceeds buffer_size {T}")
    # Index by age (0 = oldest filled, n-1 = newest); slot = (base + age) % T.
    # In age-space windows are linear, so wrap and warm-up collapse into one predicate.
    base = self._step if n == T else 0
    ages = torch.arange(T, device=dev)
    ended = (self._storage["terminated"] | self._storage["truncated"])[
      :, (base + ages) % T
    ].long()
    # Interior [a, a+L-2] is done-free iff c[a+L-1] == c[a] (c is cumsum with leading 0).
    c = torch.cat(
      [ended.new_zeros(E, 1), ended.cumsum(1), ended.new_zeros(E, L - 1)], dim=1
    )
    valid = (c[:, L - 1 : L - 1 + T] == c[:, :T]) & (ages + L - 1 < n)
    csum = valid.cumsum(dim=1, dtype=torch.long)
    total = csum[:, -1]
    if not (total > 0).any():
      raise RuntimeError(f"no valid sequences L={L} n={n}")
    env_idx = torch.multinomial((total > 0).float(), B, replacement=True)
    u = (torch.rand(B, device=dev) * total[env_idx].float()).long()
    age_start = torch.searchsorted(csum[env_idx], u[:, None], right=True).squeeze(-1)
    storage_idx = (base + age_start[:, None] + torch.arange(L, device=dev)) % T
    return self._storage[env_idx[:, None], storage_idx]
