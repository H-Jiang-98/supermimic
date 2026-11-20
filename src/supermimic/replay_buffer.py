from bisect import bisect_right
from dataclasses import dataclass

import torch
from tensordict import TensorDict

from supermimic.belief import BeliefState, GaussianStatePrior
from supermimic.utils.lie_state import LieState


@dataclass(frozen=True)
class FieldSpec:
  dim: int | tuple[int, ...] = 0  # 0 = scalar; tuple preserves structured fields
  dtype: torch.dtype = torch.float32

  def shape(self, num_envs: int, buffer_size: int) -> tuple[int, ...]:
    if isinstance(self.dim, tuple):
      return (num_envs, buffer_size, *self.dim)
    return (num_envs, buffer_size, self.dim) if self.dim else (num_envs, buffer_size)


@dataclass(frozen=True)
class BeliefTrainingSequence:
  """An immutable sampling record, with a realized segment prior.

  ``prefix`` includes the window's first frame. Its measurements are consumed
  exactly once while rebuilding the current-model boundary posterior. ``frames``
  contains T+1 observations/truth states; only its first T actions are scored.
  Short windows contain fewer frames, rather than fabricated reset successors.
  Tensors are detached copies of replay and remain valid if replay is overwritten.
  """

  prior: BeliefState
  prefix: TensorDict
  frames: TensorDict

  @property
  def episode(self) -> int:
    return int(self.frames["episode_id"][0])

  @property
  def start_time(self) -> int:
    return int(self.frames["time_index"][0])

  @property
  def transitions(self) -> int:
    return len(self.frames) - 1


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
  ):
    if not {"terminated", "truncated"} <= schema.keys():
      raise ValueError("schema needs 'terminated' and 'truncated'")
    self._episode_heads: dict[int, tuple[int, int, BeliefState]] = {}
    self.initial_prior = initial_prior or GaussianStatePrior()
    self._training_heads: dict[int, tuple[int, int, int, BeliefState]] = {}
    self._validated_lengths: dict[int, int] | None = None
    self._finalized_at = -1
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

  def validate_head_priors(
    self,
    episode_id: torch.Tensor,
    time_index: torch.Tensor,
    head_priors: tuple[torch.Tensor, BeliefState] | None,
  ) -> None:
    """Check the replay seeds before a real environment can advance."""
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
      or priors.covariance.shape[:-2] != heads.shape
    ):
      raise ValueError("Recorded priors must match the episode heads being inserted")
    sensor = self._storage["actor_obs"]
    if (
      priors.covariance.device != sensor.device
      or priors.covariance.dtype != sensor.dtype
    ):
      raise ValueError(
        "Episode head priors must match replay measurement dtype and device"
      )
    priors.validate()

  def insert(
    self,
    td: TensorDict,
    *,
    head_priors=None,
    require_head_priors: bool = False,
    token=None,
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
    if head_priors is not None or (require_head_priors and "episode_id" in td.keys()):
      self.validate_head_priors(td["episode_id"], td["time_index"], head_priors)
    new_heads = {}
    if head_priors is not None:
      ids, priors = head_priors
      rows = (td["time_index"] == 0).nonzero().flatten().tolist()
      new_heads = {
        episode: (env_row, self._step, priors.index(i).detach().clone())
        for i, (episode, env_row) in enumerate(zip(ids.tolist(), rows, strict=True))
      }
    expired = []
    if "episode_id" in td.keys() and self.num_transitions == self.buffer_size:
      oldest = self._storage[:, self._step]
      expired = oldest["episode_id"][oldest["time_index"] == 0].tolist()
    self._storage[:, self._step] = td
    for episode in expired:
      self._episode_heads.pop(episode, None)
    self._episode_heads.update(new_heads)
    self._step = (self._step + 1) % self.buffer_size
    self.num_transitions = min(self.buffer_size, self.num_transitions + 1)
    self.insert_count += 1
    self._last_insert_token = token

  @torch.no_grad()
  def finish_collection_round(self) -> None:
    """Realize one training prior per retained episode fragment, including cuts."""
    if self._finalized_at == self.insert_count:
      return
    if "episode_id" not in self._storage.keys() or not self.num_transitions:
      raise ValueError("A training segment requires retained estimator frames")
    base = self._step if self.num_transitions == self.buffer_size else 0
    slots = (
      base + torch.arange(self.num_transitions, device=self.device)
    ) % self.buffer_size
    episodes = self._storage["episode_id"][:, slots]
    begins = torch.ones_like(episodes, dtype=torch.bool)
    begins[:, 1:] = episodes[:, 1:] != episodes[:, :-1]
    rows, ages = begins.nonzero(as_tuple=True)
    start_slots = slots[ages]
    ids = episodes[rows, ages].tolist()
    if len(set(ids)) != len(ids):
      raise ValueError(
        "Replay episode IDs must identify a single contiguous environment segment"
      )
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
      full = self._storage["state"][rows[missing], start_slots[missing]]
      priors = self.initial_prior.sample(LieState.from_full(full))
      for j, i in enumerate(missing):
        # Each row owns its mean; the fixed diagonal covariance can be shared.
        prior = BeliefState(priors.mean.index(j).clone(), priors.covariance[j])
        heads[ids[i]] = (row_numbers[i], slot_numbers[i], times[i], prior)
    self._training_heads = heads
    self._validated_lengths = None
    self._finalized_at = self.insert_count

  def training_prefix(
    self, episode: int, t: int, *, relative: bool = False
  ) -> tuple[BeliefState, TensorDict]:
    """Return the realized prior and frames from retained segment frame zero to t."""
    self.finish_collection_round()
    head = self._training_heads.get(episode)
    if head is None:
      raise ValueError(f"Episode {episode} unavailable in replay")
    row, slot, first, prior = head
    length = t if relative else t - first
    frames = self._gather_prefix(episode, row, slot, first, length)
    frames["segment_time"] = frames["time_index"] - first
    return prior.clone(), frames

  def retained_frames(
    self, episode_ids=None, *, newest_count: int | None = None
  ) -> TensorDict:
    """Chronological retained frames, optionally restricted to training episodes.

    A detached copy prevents normalization or callers from modifying replay.
    Episode filtering also keeps validation measurements out of normalization.
    """
    self.finish_collection_round()
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
    return self._storage[rows.reshape(-1), slots.reshape(-1)].detach().clone()

  def training_episode_lengths(self) -> dict[int, int]:
    """Retained frame counts for sealed fragments, including singleton episodes."""
    self.finish_collection_round()
    if self._validated_lengths is not None:
      return self._validated_lengths.copy()
    base = self._step if self.num_transitions == self.buffer_size else 0
    slots = (
      base + torch.arange(self.num_transitions, device=self.device)
    ) % self.buffer_size
    ids, counts = self._storage["episode_id"][:, slots].unique(return_counts=True)
    lengths = dict(zip(ids.tolist(), counts.tolist(), strict=True))
    # Check full fragments once here: discontinuous timestamps or a reset marked
    # inside an episode cannot silently become scored training transitions.
    for episode, length in lengths.items():
      self.training_prefix(episode, length - 1, relative=True)
    self._validated_lengths = lengths
    return lengths.copy()

  def belief_sequence(
    self, episode: int, start_time: int, horizon: int
  ) -> BeliefTrainingSequence:
    """Gather a window and its current-segment prefix without crossing a reset."""
    if not isinstance(horizon, int) or horizon < 1:
      raise ValueError("Belief training horizon must be a positive integer")
    lengths = self.training_episode_lengths()
    if episode not in lengths:
      raise ValueError(f"Episode {episode} unavailable in replay")
    return self._belief_sequence(episode, start_time, horizon, lengths[episode])

  def _belief_sequence(self, episode, start_time, horizon, length):
    first = self._training_heads[episode][2]
    start = start_time - first
    if not 0 <= start < length - 1:
      raise ValueError("Belief window requires a recorded successor in the episode")
    end = min(start + horizon, length - 1)
    prior, retained = self.training_prefix(episode, end, relative=True)
    return BeliefTrainingSequence(
      prior.detach().clone(),
      retained[: start + 1].detach().clone(),
      retained[start:].detach().clone(),
    )

  def sample_belief_sequences(
    self,
    horizon: int,
    batch_size: int,
    *,
    episode_ids=None,
    generator: torch.Generator | None = None,
  ) -> list[BeliefTrainingSequence]:
    """Uniformly sample valid start transitions, allowing shorter tail windows.

    The last retained action has no recorded successor and is never selected.
    Priors persist across optimizer updates; posteriors are deliberately absent
    from these records and must be rebuilt using the current model.
    """
    if not isinstance(batch_size, int) or batch_size < 1:
      raise ValueError("Belief training batch_size must be a positive integer")
    if not isinstance(horizon, int) or horizon < 1:
      raise ValueError("Belief training horizon must be a positive integer")
    lengths = self.training_episode_lengths()
    allowed = set(lengths) if episode_ids is None else set(episode_ids)
    unknown = allowed - lengths.keys()
    if unknown:
      raise ValueError(f"Training episodes unavailable in replay: {sorted(unknown)}")
    eligible = [
      episode
      for episode, length in lengths.items()
      if episode in allowed and length > 1
    ]
    if not eligible:
      raise ValueError("No recorded in-episode successor transitions for training")
    # Sample flattened transition indices without building a Python tuple for
    # every retained frame (large environment banks can contain millions).
    cumulative = [0]
    for episode in eligible:
      cumulative.append(cumulative[-1] + lengths[episode] - 1)
    choice = torch.randint(
      cumulative[-1], (batch_size,), device=self.device, generator=generator
    ).tolist()
    sequences = []
    for index in choice:
      segment = bisect_right(cumulative, index) - 1
      episode = eligible[segment]
      start = self._training_heads[episode][2] + index - cumulative[segment]
      sequences.append(self._belief_sequence(episode, start, horizon, lengths[episode]))
    return sequences

  def episode_prefix(self, episode: int, t: int) -> tuple[BeliefState, TensorDict]:
    """Gather frames 0..t in chronological order from one physical env slot."""
    head = self._episode_heads.get(episode)
    if head is None:
      raise ValueError(f"Episode {episode} head or initial prior unavailable in replay")
    env_row, start_slot, prior = head
    return prior.clone(), self._gather_prefix(episode, env_row, start_slot, 0, t)

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
    truncated.logical_or_(~(terminated | truncated))

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
