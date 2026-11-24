"""Differentiable world model environment for motion tracking.

Replaces the physics simulator with a learned ``Dynamics`` + ``StateRepresentation``
while a reference motion clip supplies per-frame goals. Follows a template-method
pattern inspired by mjlab's ManagerBasedRlEnv:

  reset()  →  _on_reset()  →  get_observations()
  step()   →  _wm_step()  →  _compute_commands()  →  get_observations()

``_wm_step`` bundles dynamics + termination + reward for one env step. The env
holds state only after ``reset()``. Subclasses implement only
``get_observations``, whose actor keys are the one thing that differs between a
single-actor policy and the multi-encoder actor, and may override
``_build_task_goals`` to emit extra per-task goal tensors.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields, replace
from itertools import batched, groupby
from operator import itemgetter
from typing import TYPE_CHECKING, Any, NamedTuple

import torch
from mjlab.utils.lab_api.math import quat_mul
from tensordict import TensorDict

from supermimic.belief import BeliefState
from supermimic.belief_provider import BeliefProvider, PreparedBelief
from supermimic.estimator_inputs import (
  EITInputConfig,
  EpisodeAlignment,
  PosteriorInput,
  PosteriorProvider,
  build_actor_critic_goals,
  sample_proprio,
)
from supermimic.motion_loader import motion_content_digest, motion_tensor_versions
from supermimic.replay_buffer import ReplayBuffer
from supermimic.utils.helpers import matrix_to_quat, normalized_action_limits
from supermimic.utils.lie_state import LieState
from supermimic.utils.validation import validate_rotation

if TYPE_CHECKING:
  from supermimic.critic_inputs import CriticInputConfig
  from supermimic.models import Dynamics
  from supermimic.representations import StateRepresentation


class _EstimatorGoals(NamedTuple):
  body_state: tuple[torch.Tensor, ...]
  actor: torch.Tensor
  critic: torch.Tensor
  uncertainty: torch.Tensor | None


class _BeliefFrame(NamedTuple):
  """Provider revision used to build this latent frame's policy observation."""

  provider: BeliefProvider
  generation: int
  episode_id: torch.Tensor
  time_index: torch.Tensor
  model_write_generation: int


def _cat_beliefs(beliefs: list[BeliefState]) -> BeliefState:
  return BeliefState(
    LieState(
      *(torch.cat([getattr(b.mean, f.name) for b in beliefs]) for f in fields(LieState))
    ),
    covariance_sqrt=torch.cat([b.covariance_sqrt for b in beliefs]),
  )


@dataclass(frozen=True)
class BeliefRolloutStart:
  """Detached starts with mixed source beliefs, pinned to one rollout identity."""

  batch: TensorDict
  belief: BeliefState
  model_version: str | int
  replay_buffer: ReplayBuffer
  insert_count: int
  motion_digest: str
  model_write_generation: int | None = None

  def __len__(self):
    return len(self.batch)

  @property
  def motion_times(self):
    return self.batch["motion_time"]

  def index(self, selection):
    if isinstance(selection, int):
      selection = [selection]
    return replace(
      self,
      batch=self.batch[selection].clone(),
      belief=self.belief.index(selection).clone(),
    )

  @staticmethod
  def cat(starts: list[BeliefRolloutStart]) -> BeliefRolloutStart:
    if not starts:
      raise ValueError("Cannot concatenate an empty list of policy starts")
    first = starts[0]
    for start in starts[1:]:
      if (
        start.replay_buffer is not first.replay_buffer
        or start.insert_count != first.insert_count
        or start.model_version != first.model_version
        or start.model_write_generation != first.model_write_generation
        or start.motion_digest != first.motion_digest
      ):
        raise ValueError("Policy starts must share replay, model and motion versions")
    return replace(
      first,
      batch=torch.cat([start.batch for start in starts]),
      belief=_cat_beliefs([start.belief for start in starts]),
    )

  def repeat_interleave(self, repeats: int):
    if type(repeats) is not int or repeats < 1:
      raise ValueError("Rollout repeats must be positive")
    return self.index(
      torch.arange(len(self), device=self.motion_times.device).repeat_interleave(
        repeats
      )
    )


class WorldModelEnv(ABC):
  """Differentiable world model env for motion tracking.

  Owns the reference-motion buffers, the bundled ``_wm_step``, goal updates,
  and the reset/observation plumbing. ``get_observations`` is the only abstract
  method; subclasses choose how to assemble the actor observation.
  """

  _supports_estimator_inputs = False

  def __init__(
    self,
    dynamics: Dynamics,
    representation: StateRepresentation,
    env: Any,
    batch_size: int,
    replay_buffer: ReplayBuffer,
    history_length: int,
    termination_softness: float,
    device: str = "cpu",
    estimator_input_config: EITInputConfig | None = None,
    posterior_provider: PosteriorProvider | None = None,
  ) -> None:
    self.device = device
    self.dynamics = dynamics
    self.representation = representation
    self.num_envs = batch_size
    self.replay_buffer = replay_buffer
    self.history_length = history_length
    self.estimator_input_config = estimator_input_config
    self.posterior_provider = posterior_provider
    self._posterior: PosteriorInput | None = None
    self._belief_frame: _BeliefFrame | None = None
    self._task_uncertainty: torch.Tensor | None = None
    self._pending_belief_step = None
    self.eit_critic = False
    self.critic_input_config: CriticInputConfig | None = None
    if estimator_input_config is not None:
      if not self._supports_estimator_inputs:
        raise ValueError("Estimator inputs require the MultiTaskWorldModelEnv adapter")
      if representation.actor_obs_mode != "proprio" or env.num_actions != 29:
        raise ValueError(
          "Estimator inputs require G1 proprio observations (29 actions)"
        )
      if history_length != estimator_input_config.history_length:
        raise ValueError("history_length must match estimator_input_config")
      self._action_lower, self._action_upper = normalized_action_limits(
        env, env.num_actions, device=device
      )

    # Analytic per-step reward ceiling: positive tracking-reward weights, dt-scaled.
    self.max_step_reward = env.unwrapped.step_dt * sum(
      term.weight for term in env.cfg.rewards.values() if term.weight > 0.0
    )

    # Reference-motion buffers.
    self._num_actions = env.num_actions
    motion_cmd = env.unwrapped.command_manager.get_term("motion")
    self.motion = motion_cmd.motion
    self._readonly_motion = None
    self._anchor_index = motion_cmd.motion_anchor_body_index

    if termination_softness <= 0.0:
      raise ValueError(
        f"termination_softness must be positive, got {termination_softness}"
      )
    terms = env.cfg.terminations
    self._termination_softness = termination_softness
    self._anchor_pos_threshold = terms["anchor_pos"].params["threshold"]
    self._anchor_ori_threshold = terms["anchor_ori"].params["threshold"]

  # ------------------------------------------------------------------
  # Subclass surface
  # ------------------------------------------------------------------

  @abstractmethod
  def get_observations(self) -> TensorDict:
    """Build the observation TensorDict: a ``"critic"`` key plus the
    actor's input keys."""
    ...

  def _build_task_goals(self) -> None:  # noqa: B027
    """Build extra per-task goal tensors after a goal update. Default: none."""

  def _observations_with_critic(self, history, critic) -> TensorDict:
    """Adapters can reuse the critic features validated before frame commit."""
    return self.get_observations()

  # ------------------------------------------------------------------
  # Template methods
  # ------------------------------------------------------------------

  @property
  def num_actions(self) -> int:
    """Dimension of the action space for the high-level policy."""
    return self._num_actions

  def reset(
    self,
    batch: TensorDict | None = None,
    *,
    posterior: PosteriorInput | None = None,
    episode_id: int | torch.Tensor | None = None,
    time_index: int | torch.Tensor | None = None,
    segment_time: int | torch.Tensor | None = None,
  ) -> TensorDict:
    """Rebuild through a physical time_index or a zero-based retained segment_time."""
    candidate = None
    if self.estimator_input_config is not None:
      self._check_posterior_source(posterior)
    if isinstance(self.posterior_provider, BeliefProvider):
      if (
        batch is not None
        or posterior is not None
        or episode_id is None
        or (time_index is None) == (segment_time is None)
      ):
        raise ValueError(
          "Belief reset requires episode_id and one of time_index or segment_time from replay"
        )
      batch, candidate = self._rebuild_from_replay(
        episode_id,
        time_index if segment_time is None else segment_time,
        relative=segment_time is not None,
      )
    elif episode_id is not None or time_index is not None or segment_time is not None:
      raise ValueError("Keyed replay reset requires a full belief provider")
    if batch is None:
      batch = self.sample_initial_states()
    if self.estimator_input_config is not None:
      return self._reset_estimator_inputs(batch, posterior, candidate=candidate)
    self.aux_state, self.rep_state = self._unpack_initial_states(batch)
    self.joint_state = self.representation.joint_state(self.rep_state)
    # Raw policy outputs (pre-scale/offset). Dynamics handles the transform;
    # the reward path consumes these directly.
    self.last_actions = batch["prev_action"].detach()
    self.sim_ctx = batch["sim_ctx"]
    self._on_reset(batch)
    return self.get_observations()

  def step(
    self,
    actions: torch.Tensor,
    *,
    posterior: PosteriorInput | None = None,
    process_noise: torch.Tensor | None = None,
    measurement_noise: torch.Tensor | None = None,
  ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict[str, object]]:
    self._check_belief_frame()
    if isinstance(self.posterior_provider, BeliefProvider):
      if posterior is not None:
        raise ValueError(
          "A live belief provider requires full restore, not RPV injection"
        )
      return self._step_belief_inputs(actions, process_noise, measurement_noise)
    if process_noise is not None or measurement_noise is not None:
      raise ValueError("Explicit rollout noises require a full belief provider")
    if self.estimator_input_config is not None:
      return self._step_estimator_inputs(actions, posterior)
    next_aux_state, next_rep_state, next_joint_state, term_prob, rewards = (
      self._wm_step(
        self.rep_state,
        self.aux_state,
        actions,
        self.last_actions,
        self.sim_ctx,
      )
    )
    self.rep_state = next_rep_state
    self.aux_state = next_aux_state
    self.joint_state = next_joint_state
    self.last_actions = actions

    self._compute_commands()

    if self.history_length > 1:
      self._past_sensors = torch.cat(
        [self._past_sensors[:, 1:], self._latest_sensor.unsqueeze(1)], dim=1
      )
    self._latest_sensor = self._synth_sensor()

    return self.get_observations(), rewards, term_prob, {}

  def _check_posterior_source(self, posterior: PosteriorInput | None) -> None:
    config = self.estimator_input_config
    if isinstance(self.posterior_provider, BeliefProvider):
      estimator = self.posterior_provider.estimator
      if (
        config is None
        or estimator.dynamics is not self.dynamics
        or estimator.measurement_config != config.measurement_noise
        or not torch.equal(
          estimator.default_joint_pos, self.representation._default_joint_pos
        )
      ):
        raise ValueError(
          "Belief provider must use the environment dynamics and measurement definition"
        )
      if not self.posterior_provider.differentiable:
        raise ValueError("Imagined belief provider must retain differentiable state")
    if config.uses_posterior and posterior is None and self.posterior_provider is None:
      raise ValueError(
        f"Group {config.group} in {config.mode} mode requires posterior mean"
      )

  def _check_belief_frame(self) -> None:
    """Cached goals and policy actions must still refer to the committed posterior."""
    if self._readonly_motion is not None:
      self._policy_motion_digest()
    provider, frame = self.posterior_provider, self._belief_frame
    if frame is None and not isinstance(provider, BeliefProvider):
      return
    if (
      frame is None
      or provider is not frame.provider
      or provider.generation != frame.generation
      or self.dynamics.write_generation != frame.model_write_generation
      or provider.lookup(frame.episode_id, frame.time_index, self.sim_ctx) is None
    ):
      raise ValueError("Imagined posterior changed; reset from keyed replay")

  def _commit_belief_frame(self, candidate: PreparedBelief) -> None:
    provider = self.posterior_provider
    provider.commit(candidate)
    self._belief_frame = _BeliefFrame(
      provider,
      provider.generation,
      candidate.state.episode_id.clone(),
      candidate.state.time_index.clone(),
      candidate.model_write_generation,
    )

  def _rebuild_from_replay(self, episode_id, time_index, *, relative=False):
    """Use the recorded prior, actions and measurements, at one model version."""

    def indices(value):
      value = torch.as_tensor(value, device=self.device)
      if value.ndim == 0:
        value = value.expand(self.num_envs)
      if (
        value.shape != (self.num_envs,)
        or value.dtype != torch.long
        or (value < 0).any()
      ):
        raise ValueError(
          "Replay keys must be nonnegative int64 indices per rollout slot"
        )
      return value

    episodes, times = indices(episode_id), indices(time_index)
    if self.replay_buffer.posterior_cache is not None:
      if relative:
        raise ValueError("Cached starts require absolute episode time indices")
      batch, belief = self.replay_buffer.posterior_cache.read(
        self.replay_buffer, torch.stack((episodes, times), -1)
      )
      candidate = self.posterior_provider.prepare_restore(
        belief,
        batch["sim_ctx"],
        model_version=self.posterior_provider.model_version,
        episode_id=episodes,
        time_index=times,
      )
      return batch, candidate
    provider = self.posterior_provider
    version = provider.model_version
    with torch.inference_mode(False), torch.no_grad():
      batches, beliefs = [], []
      for episode, t in zip(episodes.tolist(), times.tolist(), strict=True):
        batch, belief = next(
          self._rebuild_episode(episode, [t], batch_size=1, relative=relative)
        )
        batches.append(batch)
        beliefs.append(belief)
      batch = torch.cat(batches)
      candidate = provider.prepare_restore(
        _cat_beliefs(beliefs),
        batch["sim_ctx"],
        model_version=version,
        episode_id=episodes,
        time_index=batch["segment_time"],
      )
    return batch, candidate

  def _rebuild_episode(self, episode, times, *, batch_size, relative=False):
    """Filter a sealed episode through its sorted requested times in one pass."""
    estimator = self.posterior_provider.estimator
    with torch.inference_mode(False), torch.no_grad():
      prior, frames = self.replay_buffer.training_prefix(
        episode, times[-1], relative=relative
      )
      # Physical indices may start after zero when replay wrapped; relative
      # indices already count from that retained head.
      first = times[-1] - len(frames) + 1
      if times[0] < first:
        raise ValueError(f"Episode {episode} frame {times[0]} unavailable in replay")
      sensors = frames["actor_obs"]
      offsets = torch.arange(1 - self.history_length, 1, device=self.device)
      current = estimator.correct(prior.index(None).clone(), sensors[:1, :64])
      next_step = 1

    for requested in batched(times, batch_size, strict=False):
      with torch.inference_mode(False), torch.no_grad():
        means, factors = [], []
        positions = [t - first for t in requested]
        for position in positions:
          for k in range(next_step, position + 1):
            predicted = estimator.predict(
              current, frames["action"][k - 1 : k], frames["sim_ctx"][k - 1 : k]
            )
            current = estimator.correct(predicted, sensors[k : k + 1, :64])
          next_step = position + 1
          means.append(current.mean.to_full())
          factors.append(current.covariance_sqrt)
        indices = torch.tensor(positions, device=self.device)
        batch = frames[indices]
        batch["sensor_window"] = sensors[(indices[:, None] + offsets).clamp_min(0)]
        belief = BeliefState(
          LieState.from_full(torch.cat(means)), covariance_sqrt=torch.cat(factors)
        )
      yield batch, belief

  @contextmanager
  def readonly_policy_motion(self):
    """Pin the reference content digest for preparation and policy optimization.

    All reference tensors and their backing storage are read-only throughout
    this scope. PyTorch mutation counters catch ordinary in-place writes;
    callers must also respect this contract for externally aliased storage
    and inference tensors, which do not maintain mutation counters.
    """
    if self._readonly_motion is not None:
      self._policy_motion_digest()
      yield
      return
    self._readonly_motion = (
      self.motion,
      motion_tensor_versions(self.motion),
      motion_content_digest(self.motion),
    )
    try:
      yield
      self._policy_motion_digest()
    finally:
      self._readonly_motion = None

  def _policy_motion_digest(self):
    if self._readonly_motion is None:
      return motion_content_digest(self.motion)
    motion, versions, digest = self._readonly_motion
    if self.motion is not motion or motion_tensor_versions(self.motion) != versions:
      raise ValueError("Reference motion changed during read-only policy training")
    return digest

  def _policy_keys(self, episode_id, time_index):
    self._check_posterior_source(None)
    if not isinstance(self.posterior_provider, BeliefProvider):
      raise ValueError("Policy starts require a full belief provider")
    episodes = torch.as_tensor(episode_id, device=self.device)
    times = torch.as_tensor(time_index, device=self.device)
    if (
      episodes.ndim != 1
      or not len(episodes)
      or times.shape != episodes.shape
      or episodes.dtype != torch.long
      or times.dtype != torch.long
      or (episodes < 0).any()
      or (times < 0).any()
    ):
      raise ValueError("Policy start keys require equal nonempty int64 vectors")
    return torch.stack((episodes, times), dim=-1)

  def iter_cached_policy_starts(self, locations, *, batch_size):
    """Stream already selected replay slots without gathering their keys again."""
    if type(batch_size) is not int or batch_size < 1:
      raise ValueError("Policy start batch size must be positive")
    self._check_posterior_source(None)
    if not isinstance(self.posterior_provider, BeliefProvider):
      raise ValueError("Policy starts require a full belief provider")
    replay = self.replay_buffer
    cache = replay.posterior_cache
    if cache is None:
      raise ValueError("Cached policy starts require posterior replay")
    replay.finish_collection_round()
    version = self.posterior_provider.model_version
    write_generation = self.dynamics.write_generation
    insert_count = replay.insert_count
    digest = self._policy_motion_digest()
    for offset in range(0, len(locations[0]), batch_size):
      selected = tuple(t[offset : offset + batch_size] for t in locations)
      batch, belief = cache.read(replay, locations=selected)
      yield BeliefRolloutStart(
        batch,
        belief,
        version,
        replay,
        insert_count,
        digest,
        write_generation,
      )

  def _iter_policy_starts(self, keys, batch_size):
    if type(batch_size) is not int or batch_size < 1:
      raise ValueError("Policy start batch size must be positive")
    replay = self.replay_buffer
    if replay.posterior_cache is not None:
      locations = replay.posterior_cache.lookup(replay, keys)
      yield from self.iter_cached_policy_starts(locations, batch_size=batch_size)
      return
    version = self.posterior_provider.model_version
    insert_count = replay.insert_count
    digest = self._policy_motion_digest()
    for episode, group in groupby(keys.tolist(), key=itemgetter(0)):
      times = [time for _, time in group]
      for batch, belief in self._rebuild_episode(episode, times, batch_size=batch_size):
        with torch.inference_mode(False), torch.no_grad():
          candidate = self.posterior_provider.prepare_restore(
            belief,
            batch["sim_ctx"],
            model_version=version,
            episode_id=batch["episode_id"],
            time_index=batch["segment_time"],
          )
          start = BeliefRolloutStart(
            batch,
            candidate.state.belief,
            version,
            replay,
            insert_count,
            digest,
          )
        yield start

  def iter_policy_starts(self, episode_id, time_index, *, batch_size):
    """Yield detached unique starts by episode and time, in bounded batches."""
    keys = self._policy_keys(episode_id, time_index).unique(dim=0)
    yield from self._iter_policy_starts(keys, batch_size)

  def prepare_policy_starts(self, episode_id, time_index) -> BeliefRolloutStart:
    """Return current-model starts in request order, including repeated keys."""
    keys = self._policy_keys(episode_id, time_index)
    with torch.inference_mode(False), torch.no_grad():
      if self.replay_buffer.posterior_cache is not None:
        # Cache lookup preserves both requested order and repeated draws.
        return next(self._iter_policy_starts(keys, len(keys)))
      keys, inverse = keys.unique(dim=0, return_inverse=True)
      start = BeliefRolloutStart.cat(list(self._iter_policy_starts(keys, len(keys))))
      return start.index(inverse)

  def sample_policy_keys(
    self, batch_size: int, episode_ids, generator=None, *, num_batches: int = 1
  ):
    """Draw consecutive policy batches from the eligible replay locations."""
    if type(batch_size) is not int or batch_size < 1:
      raise ValueError("Policy start batch size must be positive")
    if type(num_batches) is not int or num_batches < 1:
      raise ValueError("Number of policy start batches must be positive")

    def draw(count):
      if not count:
        raise ValueError(
          "No training frames with available posterior for policy starts"
        )
      return torch.cat(
        [
          torch.randint(count, (batch_size,), device=self.device, generator=generator)
          for _ in range(num_batches)
        ]
      )

    replay = self.replay_buffer
    if replay.posterior_cache is not None:
      rows, slots, _ = replay.posterior_cache.locations(replay, episode_ids=episode_ids)
      selected = draw(len(rows))
      rows, slots = rows[selected], slots[selected]
      return replay._storage["episode_id"][rows, slots], replay._storage["time_index"][
        rows, slots
      ]
    frames = replay.retained_frames(
      episode_ids=episode_ids, fields=("episode_id", "time_index")
    )
    selected = draw(len(frames))
    return frames["episode_id"][selected], frames["time_index"][selected]

  def sample_policy_starts(self, batch_size: int, episode_ids, generator=None):
    return self.prepare_policy_starts(
      *self.sample_policy_keys(batch_size, episode_ids, generator)
    )

  def reset_prepared(self, start: BeliefRolloutStart) -> TensorDict:
    """Restore a copied Q0 without sampling or fusing the first measurement again."""
    self._check_posterior_source(None)
    if not isinstance(start, BeliefRolloutStart) or not len(start):
      raise ValueError("A nonempty BeliefRolloutStart is required")
    if (
      start.replay_buffer is not self.replay_buffer
      or start.insert_count != self.replay_buffer.insert_count
      or start.motion_digest != self._policy_motion_digest()
      or start.model_version != self.posterior_provider.model_version
      or (
        start.model_write_generation is not None
        and start.model_write_generation != self.dynamics.write_generation
      )
    ):
      raise ValueError("Stale rollout start: model, replay or motion changed")
    batch = start.batch.detach().clone()
    shape = (len(start),)
    reference = next(self.dynamics.parameters())
    for key, tail in (
      ("state", (71,)),
      ("sensor_window", (self.history_length, 93)),
      ("prev_action", (29,)),
      ("sim_ctx", (self.dynamics.sim_ctx_dim,)),
      ("alignment_rotation", (3, 3)),
      ("alignment_translation", (3,)),
    ):
      value = batch.get(key)
      if (
        not isinstance(value, torch.Tensor)
        or value.shape != (*shape, *tail)
        or value.dtype != reference.dtype
        or value.device != reference.device
        or not torch.isfinite(value).all()
      ):
        raise ValueError(f"Invalid prepared rollout field {key}")
    for key in ("motion_time", "episode_id", "time_index", "segment_time"):
      value = batch.get(key)
      if (
        not isinstance(value, torch.Tensor)
        or value.shape != shape
        or value.dtype != torch.long
        or value.device != reference.device
        or (value < 0).any()
      ):
        raise ValueError(f"Invalid prepared rollout index {key}")
    if (batch["time_index"] < batch["segment_time"]).any():
      raise ValueError("Prepared rollout segment starts after its physical frame")
    if (batch["motion_time"] >= self.motion.motion_ends[-1]).any():
      raise ValueError("Prepared reference frame exceeds motion bounds")
    validate_rotation(batch["alignment_rotation"])
    candidate = self.posterior_provider.prepare_restore(
      start.belief.detach(),
      batch["sim_ctx"],
      model_version=start.model_version,
      episode_id=batch["episode_id"],
      time_index=batch["segment_time"],
    )
    previous_size = self.num_envs
    try:
      self.num_envs = len(start)
      return self._reset_estimator_inputs(batch, None, candidate=candidate)
    except Exception:
      self.num_envs = previous_size
      raise

  def clear_rollout(self):
    """Drop all references to a completed imagined graph."""
    if isinstance(self.posterior_provider, BeliefProvider):
      self.posterior_provider.clear()
    for name in (
      "_belief_frame",
      "_pending_belief_step",
      "_posterior",
      "_task_uncertainty",
      "aux_state",
      "rep_state",
      "joint_state",
      "last_actions",
      "sim_ctx",
      "motion_times",
      "reference_alignment",
      "_past_sensors",
      "_latest_sensor",
      "_ref_body_state",
      "_actor_goal_obs",
      "_wm_goal_obs",
    ):
      setattr(self, name, None)
    self._task_goal_obs = {}

  def snapshot(self) -> dict:
    """Save the last committed imagined frame as a detached rollout starting point."""
    if not isinstance(self.posterior_provider, BeliefProvider):
      raise ValueError("An imagined snapshot requires a full belief provider")
    self._check_belief_frame()
    frame = {
      "aux_state": self.aux_state,
      "rep_state": self.rep_state,
      "sensor_window": self._sensor_history(),
      "prev_action": self.last_actions,
      "sim_ctx": self.sim_ctx,
      "motion_time": self.motion_times,
      "alignment_rotation": self.reference_alignment.rotation,
      "alignment_translation": self.reference_alignment.translation,
      "episode_id": self._belief_frame.episode_id,
      "segment_time": self._belief_frame.time_index,
      "segment_start_time": self._segment_start_time,
    }
    snapshot = {
      "version": 2,
      "estimator_inputs": self.estimator_input_config.to_dict(),
      "motion_ends": self.motion.motion_ends.detach().clone(),
      "motion_digest": motion_content_digest(self.motion),
      "provider": self.posterior_provider.state_dict(),
      "frame": {key: value.detach().clone() for key, value in frame.items()},
    }
    if self.eit_critic:
      snapshot["eit_critic"] = True
      snapshot["critic_config"] = asdict(self._eit_critic_config())
    return snapshot

  def restore(self, snapshot: dict) -> TensorDict:
    """Resume an imagined frame on the same model and motion without replay or fusion."""
    if not isinstance(self.posterior_provider, BeliefProvider):
      raise ValueError("An imagined restore requires a full belief provider")
    self._check_posterior_source(None)
    if snapshot.get("version") != 2:
      raise ValueError(
        "Imagined snapshot version 2 with a motion content fingerprint is required"
      )
    if snapshot.get(
      "estimator_inputs"
    ) != self.estimator_input_config.to_dict() or not torch.equal(
      snapshot["motion_ends"], self.motion.motion_ends
    ):
      raise ValueError("Imagined snapshot configuration mismatch")
    if snapshot.get("motion_digest") != motion_content_digest(self.motion):
      raise ValueError("Imagined snapshot reference motion content mismatch")
    if snapshot.get("eit_critic", False) != self.eit_critic or (
      self.eit_critic
      and snapshot.get("critic_config") != asdict(self._eit_critic_config())
    ):
      raise ValueError("Imagined snapshot critic configuration mismatch")
    candidate = self.posterior_provider.prepare_load_state_dict(snapshot["provider"])
    state = candidate.state
    frame = snapshot["frame"]
    shapes = {
      "aux_state": (7,),
      "rep_state": (68,),
      "sensor_window": (self.history_length, 93),
      "prev_action": (29,),
      "sim_ctx": (self.dynamics.sim_ctx_dim,),
      "alignment_rotation": (3, 3),
      "alignment_translation": (3,),
    }
    batch = {}
    for key, shape in shapes.items():
      value = frame[key]
      if (
        value.shape != (self.num_envs, *shape)
        or value.device != state.context.device
        or value.dtype != state.context.dtype
        or not torch.isfinite(value).all()
      ):
        raise ValueError(f"Invalid imagined snapshot field {key}")
      batch[key] = value.detach().clone()
    for key in ("segment_start_time", "motion_time", "episode_id", "segment_time"):
      value = frame[key]
      if (
        value.shape != (self.num_envs,)
        or value.dtype != torch.long
        or value.device != state.context.device
        or (value < 0).any()
      ):
        raise ValueError(f"Invalid imagined snapshot index {key}")
      batch[key] = value.detach().clone()
    if (
      not torch.equal(batch["episode_id"], state.episode_id)
      or not torch.equal(batch["segment_time"], state.time_index)
      or not torch.equal(batch["sim_ctx"], state.context)
      or (batch["motion_time"] >= self.motion.motion_ends[-1]).any()
    ):
      raise ValueError("Imagined snapshot frame and posterior disagree")
    validate_rotation(batch["alignment_rotation"])
    aux, rep = batch.pop("aux_state"), batch.pop("rep_state")
    LieState.from_rep(aux, rep).validate()
    batch["time_index"] = batch["segment_start_time"] + batch["segment_time"]
    batch["actor_obs"] = batch["sensor_window"][:, -1]
    return self._reset_estimator_inputs(
      TensorDict(batch, batch_size=[self.num_envs]),
      None,
      candidate=candidate,
      latent=(aux, rep),
    )

  def _reset_estimator_inputs(
    self,
    batch: TensorDict,
    posterior: PosteriorInput | None,
    *,
    candidate: PreparedBelief | None = None,
    latent: tuple[torch.Tensor, torch.Tensor] | None = None,
  ) -> TensorDict:
    """Validate the complete reset input before replacing a live rollout."""
    aux, rep = self._unpack_initial_states(batch) if latent is None else latent
    joints = self.representation.joint_state(rep)
    actions = batch["prev_action"].detach()
    required = {"alignment_rotation", "alignment_translation"}
    if not required.issubset(batch.keys()):
      raise ValueError("Estimator replay requires the stored episode alignment")
    alignment = EpisodeAlignment(
      batch["alignment_rotation"].detach(), batch["alignment_translation"].detach()
    )
    if self.history_length > 1:
      if "sensor_window" not in batch.keys():
        raise ValueError("Estimator replay requires the stored noisy sensor_window")
      history = batch["sensor_window"].detach()
    else:
      history = batch["actor_obs"].detach().unsqueeze(1)
    if history.shape != (self.num_envs, self.history_length, 93):
      raise ValueError("Estimator sensor history must have shape (batch, H_obs, 93)")
    if not torch.equal(history[:, -1, -29:], actions):
      raise ValueError("Stored measurement actions must equal replay prev_action")
    sensor = history[:, -1]
    if candidate is not None:
      posterior = candidate.to_actor_input()
    else:
      posterior = self._resolve_posterior(posterior, sensor, actions, reset=True)
    goals = self._prepare_estimator_goals(
      aux, rep, batch["motion_time"], alignment, posterior
    )
    critic = None
    if self.eit_critic:
      if candidate is None:
        raise ValueError("Full critic reset requires a complete keyed belief")
      critic = self._build_eit_critic(
        aux,
        rep,
        goals.critic,
        history,
        batch["sim_ctx"],
        candidate.state.belief,
      )
    # Commit only after posterior and all actor/critic reference inputs are valid.
    if candidate is not None:
      self._commit_belief_frame(candidate)
      self._segment_start_time = batch["time_index"] - batch["segment_time"]
    else:
      self._belief_frame = None
    self.aux_state, self.rep_state, self.joint_state = aux, rep, joints
    self.last_actions, self.sim_ctx = actions, batch["sim_ctx"]
    self.motion_times, self.reference_alignment = batch["motion_time"], alignment
    self._past_sensors, self._latest_sensor = history[:, :-1], sensor
    self._posterior = posterior if candidate is None else None
    self._commit_estimator_goals(goals)
    self._pending_belief_step = None
    if critic is not None:
      return self._observations_with_critic(self._sensor_history(), critic)
    return self.get_observations()

  def _step_belief_inputs(self, actions, process_noise, measurement_noise):
    """Stage latent state, one measurement and belief as one transaction.

    Input validation precedes any random draw. After sampling, retries reuse
    that transition with the same action. Noise arguments are standard normal
    samples. Omit them on retry to reuse the original samples and graphs.
    A keyed reset abandons the pending transition.
    """
    self._check_posterior_source(None)
    frame = self._belief_frame
    assert frame is not None
    inputs = (
      ("actions", actions, self.num_actions),
      ("process_noise", process_noise, 70),
      ("measurement_noise", measurement_noise, 64),
    )
    for name, value, width in inputs:
      if value is not None and (
        value.shape != (self.num_envs, width)
        or value.dtype != self.rep_state.dtype
        or value.device != self.rep_state.device
        or not torch.isfinite(value).all()
      ):
        raise ValueError(
          f"{name} must be finite with shape ({self.num_envs},{width}) "
          "and match the rollout dtype and device"
        )
    actions = self.limit_actions(actions.clone())
    pending = self._pending_belief_step
    if pending is None:
      # Autograd and retries must both see the realized samples, even when the
      # caller reuses a noise tensor as mutable storage for another transition.
      def sample(value, width):
        if value is None:
          return torch.randn(
            self.num_envs, width, device=actions.device, dtype=actions.dtype
          )
        return value.clone()

      process_noise = sample(process_noise, 70)
      measurement_noise = sample(measurement_noise, 64)
      aux, rep, joints, term_prob, rewards = self._wm_step(
        self.rep_state,
        self.aux_state,
        actions,
        self.last_actions,
        self.sim_ctx,
        process_noise=process_noise,
      )
      sensor = self._sensor_from_state(
        rep, aux, joints, actions, self.sim_ctx, measurement_noise=measurement_noise
      )
      outputs = (aux, rep, joints, term_prob, rewards, sensor)
      if not torch.stack([torch.isfinite(value).all() for value in outputs]).all():
        raise ValueError("World model produced a non-finite imagined transition")
      pending = dict(
        gradient_sources={
          name: value
          for name, value, _ in inputs
          if value is not None and value.requires_grad
        },
        actions=actions,
        process_noise=process_noise,
        measurement_noise=measurement_noise,
        aux=aux,
        rep=rep,
        joints=joints,
        term_prob=term_prob,
        rewards=rewards,
        sensor=sensor,
        times=self._next_motion_times(),
      )
      self._pending_belief_step = pending
    else:
      for name, value, _ in inputs:
        if value is None:
          continue  # Omitted explicit noise reuses the original sample and graph.
        realized = actions if name == "actions" else value
        if not torch.equal(realized, pending[name]):
          raise ValueError(
            "Pending transition retry must use the same action/noises or reset"
          )
        source = pending["gradient_sources"].get(name)
        if (source is not None or value.requires_grad) and value is not source:
          raise ValueError(
            f"Retry must reuse the same differentiable {name} tensor or reset"
          )
    candidate = frame.provider.prepare(
      measurement=pending["sensor"][..., :64],
      previous_action=pending["actions"],
      reset_mask=torch.zeros(self.num_envs, dtype=torch.bool, device=actions.device),
      context=self.sim_ctx,
      episode_id=frame.episode_id,
      time_index=frame.time_index + 1,
    )
    if candidate.generation != frame.generation:
      raise ValueError("Imagined posterior changed; reset from keyed replay")
    posterior = candidate.to_actor_input()
    goals = self._prepare_estimator_goals(
      pending["aux"],
      pending["rep"],
      pending["times"],
      self.reference_alignment,
      posterior,
    )
    past = self._past_sensors
    if self.history_length > 1:
      past = torch.cat((past[:, 1:], self._latest_sensor.unsqueeze(1)), dim=1)
    frame_features = None
    if self.eit_critic:
      history = torch.cat((past, pending["sensor"].unsqueeze(1)), dim=1)
      critic = self._build_eit_critic(
        pending["aux"],
        pending["rep"],
        goals.critic,
        history,
        self.sim_ctx,
        candidate.state.belief,
      )
      frame_features = (history, critic)
    self._commit_belief_frame(candidate)
    self._past_sensors = past
    self.aux_state, self.rep_state = pending["aux"], pending["rep"]
    self.joint_state, self.last_actions = pending["joints"], pending["actions"]
    self.motion_times, self._latest_sensor = pending["times"], pending["sensor"]
    self._commit_estimator_goals(goals)
    self._pending_belief_step = None
    obs = (
      self._observations_with_critic(*frame_features)
      if frame_features is not None
      else self.get_observations()
    )
    return obs, pending["rewards"], pending["term_prob"], {}

  def _step_estimator_inputs(
    self, actions: torch.Tensor, posterior: PosteriorInput | None
  ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict[str, object]]:
    """Stage the next frame; failed posterior validation leaves env state intact."""
    self._check_posterior_source(posterior)
    actions = self.limit_actions(actions)
    aux, rep, joints, term_prob, rewards = self._wm_step(
      self.rep_state, self.aux_state, actions, self.last_actions, self.sim_ctx
    )
    times = self._next_motion_times()
    sensor = None
    if posterior is None and self.posterior_provider is not None:
      sensor = self._sensor_from_state(rep, aux, joints, actions, self.sim_ctx)
      posterior = self._resolve_posterior(posterior, sensor, actions, reset=False)
    goals = self._prepare_estimator_goals(
      aux, rep, times, self.reference_alignment, posterior
    )
    # A direct injection can be validated before drawing any new measurement.
    if sensor is None:
      sensor = self._sensor_from_state(rep, aux, joints, actions, self.sim_ctx)
    if self.history_length > 1:
      past = torch.cat(
        [self._past_sensors[:, 1:], self._latest_sensor.unsqueeze(1)], dim=1
      )
      self._past_sensors = past
    self.aux_state, self.rep_state, self.joint_state = aux, rep, joints
    self.last_actions, self.motion_times = actions, times
    self._latest_sensor, self._posterior = sensor, posterior
    self._commit_estimator_goals(goals)
    return self.get_observations(), rewards, term_prob, {}

  def limit_actions(self, actions: torch.Tensor) -> torch.Tensor:
    """Same post-exploration normalized command limit used by real collection."""
    if self.estimator_input_config is None:
      return actions
    return torch.clamp(
      actions, min=self._action_lower.to(actions), max=self._action_upper.to(actions)
    )

  def _synth_sensor(self) -> torch.Tensor:
    """Synthesize the actor's sensor from the current WM state."""
    return self._sensor_from_state(
      self.rep_state, self.aux_state, self.joint_state, self.last_actions, self.sim_ctx
    )

  def _sensor_from_state(
    self,
    rep: torch.Tensor,
    aux: torch.Tensor,
    joints: torch.Tensor,
    actions: torch.Tensor,
    sim_ctx: torch.Tensor,
    *,
    measurement_noise: torch.Tensor | None = None,
  ) -> torch.Tensor:
    sensor = self.representation.compute_actor_sensor(
      rep,
      sim_ctx,
      aux_state=aux,
      joint_state=joints,
      last_action=actions,
      noisy=self.estimator_input_config is None,
    )
    if self.estimator_input_config is not None:
      if measurement_noise is not None:
        if (
          measurement_noise.shape != sensor[..., :64].shape
          or not torch.isfinite(measurement_noise).all()
        ):
          raise ValueError("measurement_noise must be finite with shape (B,64)")
        std = self.estimator_input_config.measurement_noise.standard_deviations(
          29, device=sensor.device, dtype=sensor.dtype
        )
        return torch.cat(
          [sensor[..., :64] + measurement_noise * std, sensor[..., 64:]], -1
        )
      return sample_proprio(sensor, self.estimator_input_config.measurement_noise)
    return sensor

  def _resolve_posterior(
    self,
    posterior: PosteriorInput | None,
    sensor: torch.Tensor,
    actions: torch.Tensor,
    *,
    reset: bool,
  ) -> PosteriorInput | None:
    """Consume a posterior for this time point, never reuse a stale injection.

    The provider sees exactly the sensor sample consumed by the actor and the
    previous executed action. It has no privileged-state argument. A reset call
    restores/initializes its state; history padding is not replayed as updates.
    """
    if posterior is None and self.posterior_provider is not None:
      posterior = self.posterior_provider(
        measurement=sensor[..., :64],
        previous_action=actions,
        reset_mask=torch.full(
          (self.num_envs,), reset, dtype=torch.bool, device=sensor.device
        ),
      )
    return posterior

  def _sensor_history(self) -> torch.Tensor:
    """The ``(B, history_length, sensor_dim)`` window; newest frame last."""
    latest = self._latest_sensor.unsqueeze(1)
    if self.history_length == 1:
      return latest
    return torch.cat([self._past_sensors, latest], dim=1)

  def _critic_obs(self, history: torch.Tensor | None = None) -> torch.Tensor:
    """The privileged critic input — full WM state, not the actor's sensor obs."""
    if self.eit_critic:
      return self._build_eit_critic(
        self.aux_state,
        self.rep_state,
        self._wm_goal_obs,
        self._sensor_history() if history is None else history,
        self.sim_ctx,
        self.posterior_provider.state.belief,
      )
    return torch.cat([self.rep_state, self._wm_goal_obs, self.sim_ctx], dim=-1)

  def _eit_critic_config(self):
    from supermimic.critic_inputs import CriticInputConfig

    return self.critic_input_config or CriticInputConfig(
      group=self.estimator_input_config.group
    )

  def _build_eit_critic(self, aux, rep, goal, history, context, belief):
    """Preflight staged frames before committing provider or environment state."""
    from supermimic.critic_inputs import build_critic_features

    return build_critic_features(
      LieState.from_rep(aux, rep),
      goal,
      history,
      context,
      belief,
      config=self._eit_critic_config(),
      group=self.estimator_input_config.group,
    )

  def _wm_step(
    self,
    rep_state: torch.Tensor,
    aux_state: torch.Tensor,
    action: torch.Tensor,
    last_action: torch.Tensor,
    sim_ctx: torch.Tensor,
    *,
    process_noise: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bundled dynamics + termination + reward for one env step.

    Returns ``(next_aux_state, next_rep_state, next_joint_state, term_prob,
    reward)``. ``action`` / ``last_action`` are raw policy outputs; Dynamics
    applies the scale/offset transform internally. Termination and reward both
    score the post-step state against the cached reference body state, the way
    the simulator scores it before advancing the clip.
    """
    if process_noise is None:
      next_aux_state, next_rep_state = self.dynamics(
        rep_state, action, aux_state, sim_ctx
      )
    else:
      next_aux_state, next_rep_state = self.dynamics.sample(
        rep_state, action, aux_state, sim_ctx, process_noise
      )
    next_joint_state = self.representation.joint_state(next_rep_state)
    ref_body_pos_w, ref_body_quat_w, ref_body_lin_vel_w, ref_body_ang_vel_w = (
      self._ref_body_state
    )
    term_prob = self._compute_term_prob(next_aux_state, ref_body_pos_w, ref_body_quat_w)
    reward = self.representation.compute_wm_reward_from_reference(
      next_rep_state,
      next_aux_state,
      next_joint_state,
      action,
      last_action,
      ref_body_pos_w,
      ref_body_quat_w,
      ref_body_lin_vel_w,
      ref_body_ang_vel_w,
    )
    return next_aux_state, next_rep_state, next_joint_state, term_prob, reward

  def _compute_term_prob(
    self,
    aux_state: torch.Tensor,
    ref_body_pos_w: torch.Tensor,
    ref_body_quat_w: torch.Tensor,
  ) -> torch.Tensor:
    """Soft termination from the real threshold rules."""
    anchor_pos_w = ref_body_pos_w[:, self._anchor_index]
    anchor_quat_w = ref_body_quat_w[:, self._anchor_index]
    root_pos, root_quat = aux_state[..., :3], aux_state[..., 3:7]
    height_error = (anchor_pos_w[..., 2] - root_pos[..., 2]).abs()
    # Gravity's z in a body frame is 2(x^2 + y^2) - 1 for a unit quaternion.
    ref_gravity_z = 2 * (anchor_quat_w[..., 1:3] ** 2).sum(-1) - 1.0
    root_gravity_z = 2 * (root_quat[..., 1:3] ** 2).sum(-1) - 1.0
    tilt_error = (ref_gravity_z - root_gravity_z).abs()
    normalized_error = torch.maximum(
      height_error / self._anchor_pos_threshold,
      tilt_error / self._anchor_ori_threshold,
    )
    return torch.sigmoid((normalized_error - 1.0) / self._termination_softness)

  def _on_reset(self, batch: TensorDict) -> None:
    """Seed motion times from the sampled batch and refresh goals."""
    self.motion_times = batch["motion_time"]
    self._update_goal()
    if self.history_length > 1:
      self._past_sensors = self._warmstart_past_sensors(
        batch["sensor_window"], batch["pad_mask"]
      )
    self._latest_sensor = self._synth_sensor()

  def _compute_commands(self) -> None:
    """Advance the reference clip one frame (clamped at clip ends) and refresh."""
    self.motion_times = self._next_motion_times()
    self._update_goal()

  def _next_motion_times(self) -> torch.Tensor:
    next_motion_times = self.motion_times + 1
    # Membership in the sorted clip ends; torch.isin syncs the host on long lists.
    ends = self.motion.motion_ends
    pos = torch.searchsorted(ends, next_motion_times).clamp(max=ends.numel() - 1)
    at_end = ends[pos] == next_motion_times
    return torch.where(at_end, self.motion_times, next_motion_times)

  def _update_goal(self) -> None:
    """Compute the reference body state at ``motion_times`` and rebuild goals."""
    if self.estimator_input_config is not None:
      goals = self._prepare_estimator_goals(
        self.aux_state,
        self.rep_state,
        self.motion_times,
        self.reference_alignment,
        self._posterior,
      )
      self._commit_estimator_goals(goals)
      return
    self._ref_body_state = self.motion.body_state(self.motion_times)
    self._wm_goal_obs, self._actor_goal_obs = self.representation.compute_wm_goal(
      self.motion, self.motion_times, self.aux_state
    )
    self._build_task_goals()

  def _prepare_estimator_goals(
    self,
    aux: torch.Tensor,
    rep: torch.Tensor,
    times: torch.Tensor,
    alignment: EpisodeAlignment,
    posterior: PosteriorInput | None,
  ) -> _EstimatorGoals:
    assert self.estimator_input_config is not None
    reference = self.representation.reference_window(self.motion, times, alignment)
    truth = self.representation.true_rpv(aux, rep)
    actor, uncertainty, critic = build_actor_critic_goals(
      self.estimator_input_config, truth, posterior, reference
    )
    bodies = self._aligned_reference_bodies(self.motion.body_state(times), alignment)
    return _EstimatorGoals(bodies, actor, critic, uncertainty)

  def _commit_estimator_goals(self, goals: _EstimatorGoals) -> None:
    self._ref_body_state = goals.body_state
    self._actor_goal_obs, self._wm_goal_obs = goals.actor, goals.critic
    self._task_uncertainty = goals.uncertainty
    self._build_task_goals()

  def _aligned_reference_bodies(
    self,
    body_state: tuple[torch.Tensor, ...],
    alignment: EpisodeAlignment,
  ) -> tuple[torch.Tensor, ...]:
    """Apply the saved episode transform to every privileged reference channel."""
    position, quaternion, linear_velocity, angular_velocity = body_state
    rotation = alignment.rotation[:, None]
    translation = alignment.translation[:, None]

    def rotate(value: torch.Tensor) -> torch.Tensor:
      return (rotation @ value.unsqueeze(-1)).squeeze(-1)

    alignment_quaternion = matrix_to_quat(rotation).expand_as(quaternion)
    return (
      rotate(position) + translation,
      quat_mul(alignment_quaternion, quaternion),
      rotate(linear_velocity),
      rotate(angular_velocity),
    )

  # ------------------------------------------------------------------
  # Shared helpers
  # ------------------------------------------------------------------

  def sample_initial_states(self) -> TensorDict:
    """Sample initial state snapshots from the replay buffer.

    With history_length > 1 the warm start draws the init state 1-step
    (H-independent) and keeps its real in-episode predecessors to seed the ring:
    ``sensor_window`` holds the H sensor frames ending at the init state, and
    ``pad_mask`` flags its clamp-padded slots.
    """
    if self.history_length == 1:
      return self.replay_buffer.sample(1, self.num_envs).squeeze(1)
    window, padded = self.replay_buffer.sample_causal(
      self.history_length, self.num_envs
    )
    batch = window[:, -1]
    batch["sensor_window"] = window["actor_obs"]
    batch["pad_mask"] = padded
    return batch

  def _warmstart_past_sensors(
    self, sensor_window: torch.Tensor, pad_mask: torch.Tensor
  ) -> torch.Tensor:
    """Noise the sampled predecessor window; clamp-padded slots (before the
    episode start) share the oldest real frame's single noise realization, the
    way the real-sim ring's reset padding does."""
    noisy = self.representation.add_proprio_noise(sensor_window[:, :-1])
    pad = pad_mask[:, :-1]  # padding is a leading run (sample_causal)
    first_real = (~pad).int().argmax(dim=1)
    shared = noisy[torch.arange(self.num_envs, device=self.device), first_real]
    return torch.where(pad.unsqueeze(-1), shared.unsqueeze(1), noisy)

  def _unpack_initial_states(
    self, batch: TensorDict
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract (aux_state, rep_state) from a sampled batch."""
    full_states = batch["state"]
    aux_state = self.representation.extract_aux_state(full_states)
    rep_state = self.representation.represent(full_states)
    return aux_state.detach(), rep_state.detach()
