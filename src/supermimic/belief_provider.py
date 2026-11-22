"""Transactional per-environment filtering, separate from the pure filter core.

An episode prior is supplied explicitly. Imagined replay starts restore an
already corrected *full* posterior; history padding is never a filter update.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, replace

import torch

from supermimic.belief import BeliefState, GaussianStatePrior
from supermimic.estimator import LieBeliefFilter
from supermimic.utils.lie_state import STATE_LAYOUT_VERSION, TANGENT_VERSION, LieState

_FIELDS = ("rotation", "position", "velocity", "omega", "joint_pos", "joint_vel")
PROVIDER_VERSION = "right_local_provider_v3"


@dataclass(frozen=True)
class ProviderState:
  belief: BeliefState
  context: torch.Tensor
  episode_id: torch.Tensor
  time_index: torch.Tensor
  # Legacy content SHA; incremental providers use the lightweight model revision.
  model_version: str | int
  # Realized pre-observation priors for the rows currently at frame zero.
  head_priors: tuple[torch.Tensor, BeliefState] | None = None

  @property
  def time(self) -> torch.Tensor:
    return self.time_index


@dataclass(frozen=True)
class PreparedBelief:
  state: ProviderState
  generation: int
  owner: object
  model_version: str | int
  model_write_generation: int
  diagnostics: dict[str, torch.Tensor] | None = None

  @property
  def head_priors(self):
    return self.state.head_priors

  def to_actor_input(self):
    return self.state.belief.to_actor_input()


@dataclass(frozen=True)
class FixedBeliefPrior:
  """Explicit per-environment episode prior, supplied by the experiment setup.

  Its coordinates and covariance must describe available initialization
  knowledge. No simulator state is read by this object or by the provider.
  """

  belief: BeliefState

  def __call__(self, *, measurement, context, env_ids) -> BeliefState:
    del context
    return (
      self.belief.index(env_ids.to(self.belief.covariance.device))
      .to(measurement)
      .clone()
    )


class BeliefProvider:
  def __init__(
    self,
    estimator: LieBeliefFilter,
    prior_factory: Callable[..., BeliefState] | GaussianStatePrior | None = None,
    *,
    differentiable: bool = False,
    incremental: bool = False,
  ):
    self.estimator = estimator
    self.prior_factory = prior_factory
    self.differentiable = differentiable
    self.incremental = incremental
    self.state: ProviderState | None = None
    self.diagnostics: dict[str, torch.Tensor] | None = None
    self._generation = 0
    self._pending_prior: tuple[torch.Tensor, torch.Tensor, BeliefState] | None = None

  @property
  def model_version(self) -> str | int:
    # The hash remains available for legacy callers and explicit snapshots.
    model = self.estimator.dynamics
    return model.model_revision if self.incremental else model.model_version

  def clear(self) -> None:
    """Release an active graph and invalidate outstanding transactions."""
    self.state = None
    self.diagnostics = None
    self._pending_prior = None
    self._generation += 1

  @property
  def generation(self) -> int:
    """Changes on every committed filter update or restore."""
    return self._generation

  def _ordinary(self, value: torch.Tensor) -> torch.Tensor:
    return value.clone() if self.differentiable else value.detach().clone()

  def _belief(self, value: BeliefState) -> BeliefState:
    return value.clone() if self.differentiable else value.detach().clone()

  def _validate_context(self, context, batch, reference):
    model = self.estimator.dynamics
    spec = model.action_scale
    if (
      context.shape != (batch, model.sim_ctx_dim)
      or context.device != spec.device
      or context.dtype != spec.dtype
      or reference.device != spec.device
      or reference.dtype != spec.dtype
      or not torch.isfinite(context).all()
    ):
      raise ValueError(
        "Context and belief must match model context width, dtype and device"
      )

  @staticmethod
  def _check_index(value, batch, device):
    if (
      value.shape != (batch,)
      or value.dtype != torch.long
      or value.device != device
      or (value < 0).any()
    ):
      raise ValueError(
        "Frame indices must be nonnegative int64 batch vectors on the model device"
      )

  def lookup(self, episode_id, time_index, context):
    """Return the committed Q for exactly these frames, or None for a new frame."""
    state = self.state
    if state is None or not (
      torch.equal(state.episode_id, episode_id)
      and torch.equal(state.time_index, time_index)
    ):
      return None
    if (
      (not self.incremental or self.differentiable)
      and state.model_version != self.model_version
    ) or not torch.equal(state.context, context):
      raise ValueError(
        "Cached posterior model version or context changed; rebuild or reset"
      )
    return state.belief

  def prepare(
    self,
    *,
    measurement: torch.Tensor,
    previous_action: torch.Tensor,
    reset_mask: torch.Tensor,
    context: torch.Tensor | None = None,
    episode_id: torch.Tensor | None = None,
    time_index: torch.Tensor | None = None,
    reset_truth: LieState | None = None,
  ) -> PreparedBelief:
    """Compute a candidate without mutating the current posterior.

    Real collection runs in inference mode. Local Jacobians still need AD;
    clones made inside inference_mode(False) are ordinary autograd tensors.
    Outer gradients are retained only when enabled by both provider and caller.
    """
    # Capture the caller's mode before inference_mode(False) can enable gradients.
    track_grad = self.differentiable and torch.is_grad_enabled()
    with torch.inference_mode(False), torch.set_grad_enabled(track_grad):
      measurement = self._ordinary(measurement)
      previous_action = self._ordinary(previous_action)
      reset_mask = reset_mask.clone()
      batch = measurement.shape[0]
      if measurement.shape != (batch, 64) or previous_action.shape != (batch, 29):
        raise ValueError(
          "Provider expects measurement (B,64) and executed action (B,29)"
        )
      if reset_mask.shape != (batch,) or reset_mask.dtype != torch.bool:
        raise ValueError("reset_mask must be a boolean batch vector")
      if (
        not torch.isfinite(measurement).all()
        or not torch.isfinite(previous_action).all()
      ):
        raise ValueError("Provider inputs must be finite")
      if context is None:
        if self.state is None:
          raise ValueError("Initial provider call requires explicit context")
        context = self.state.context
      context = self._ordinary(context)
      self._validate_context(context, batch, measurement)
      if (
        previous_action.device != measurement.device
        or previous_action.dtype != measurement.dtype
      ):
        raise ValueError("Previous action must match measurement dtype and device")
      keep = ~reset_mask
      version = self.model_version
      model_write_generation = self.estimator.dynamics.write_generation
      slots = torch.arange(batch, device=measurement.device)
      if self.state is None or self.state.time_index.shape != (batch,):
        next_episode = slots
        next_time = torch.zeros_like(slots)
      else:
        next_episode = self.state.episode_id + batch * reset_mask.long()
        next_time = torch.where(reset_mask, 0, self.state.time_index + 1)
      episode_id = next_episode if episode_id is None else episode_id.clone()
      time_index = next_time if time_index is None else time_index.clone()
      self._check_index(episode_id, batch, measurement.device)
      self._check_index(time_index, batch, measurement.device)
      if not torch.equal(time_index, next_time) or not torch.equal(
        episode_id[keep], next_episode[keep]
      ):
        raise ValueError(
          "Requested frame does not follow the committed episode/time_index"
        )
      if self.state is not None and self.state.episode_id.shape == (batch,):
        if (episode_id[reset_mask] == self.state.episode_id[reset_mask]).any():
          raise ValueError("Episode reset requires a new episode_id")
      if keep.any():
        if self.state is None or self.state.time.shape != (batch,):
          raise ValueError("Non-reset slots require a matching initialized provider")
        if self.state.model_version != version and (
          not self.incremental or self.differentiable
        ):
          raise ValueError(
            "Model/normalization version changed; restore a rebuilt belief"
          )
        if not torch.equal(context[keep], self.state.context[keep]):
          raise ValueError(
            "Context changed for live slots; reset or restore explicitly"
          )

      head_priors = None
      pieces = []
      diagnostics = []
      if reset_mask.any():
        if self.prior_factory is None:
          raise ValueError("New episodes require an explicit prior_factory")
        ids = reset_mask.nonzero().flatten()
        episodes = episode_id[ids]
        cached = self._pending_prior
        if (
          cached is not None
          and torch.equal(cached[0], ids)
          and torch.equal(cached[1], episodes)
        ):
          prior = cached[2]
        else:
          if isinstance(self.prior_factory, GaussianStatePrior):
            if reset_truth is None or reset_truth.position.shape != (ids.numel(), 3):
              raise ValueError(
                "Truth-based initialization requires the reset rows' full states"
              )
            prior = self.prior_factory.sample(reset_truth)
          else:
            prior = self.prior_factory(
              measurement=measurement[ids], context=context[ids], env_ids=ids
            )
          prior = self._belief(prior)
          prior.validate()
          self._pending_prior = (ids, episodes, prior)
        head_priors = (episodes, prior.detach())
        corrected, diagnostic = self.estimator.correct(
          prior, measurement[ids], return_diagnostics=True
        )
        pieces.append((ids, corrected))
        diagnostics.append((ids, diagnostic))
      if keep.any():
        ids = keep.nonzero().flatten()
        old = self._belief(self.state.belief.index(ids))
        predicted = self.estimator.predict(old, previous_action[ids], context[ids])
        corrected, diagnostic = self.estimator.correct(
          predicted, measurement[ids], return_diagnostics=True
        )
        pieces.append((ids, corrected))
        diagnostics.append((ids, diagnostic))

      # index_copy is differentiable and preserves both partial-reset graphs.
      fields = {}
      for name in _FIELDS:
        exemplar = getattr(pieces[0][1].mean, name)
        value = exemplar.new_zeros((batch, *exemplar.shape[1:]))
        for ids, piece in pieces:
          value = value.index_copy(0, ids, getattr(piece.mean, name))
        fields[name] = value
      covariance = measurement.new_zeros(batch, 70, 70)
      for ids, piece in pieces:
        covariance = covariance.index_copy(0, ids, piece.covariance)
      belief = BeliefState(LieState(**fields), covariance)
      belief.validate()
      if not track_grad:
        belief, context = belief.detach(), context.detach()
      summary = {
        "innovation_norm": measurement.new_zeros(batch),
        "jitter": measurement.new_zeros(batch),
        "cholesky_attempts": torch.zeros(
          batch, dtype=torch.long, device=measurement.device
        ),
      }
      for ids, diagnostic in diagnostics:
        summary["innovation_norm"][ids] = diagnostic.innovation.detach().norm(dim=-1)
        summary["jitter"][ids] = diagnostic.jitter.detach()
        summary["cholesky_attempts"][ids] = diagnostic.cholesky_attempts.detach()
      return PreparedBelief(
        ProviderState(belief, context, episode_id, time_index, version, head_priors),
        self._generation,
        self,
        version,
        model_write_generation,
        summary,
      )

  def prepare_restore(
    self,
    belief: BeliefState,
    context: torch.Tensor,
    *,
    model_version: str | int,
    time_index: torch.Tensor | None = None,
    episode_id: torch.Tensor | None = None,
    head_priors: tuple[torch.Tensor, BeliefState] | None = None,
  ) -> PreparedBelief:
    """Restore Q[t|t]; the next prepare consumes a[t], o[t+1].

    To record restored frame-zero rows, also pass their realized priors in
    batch-row order as (episode_ids, prior_before_observation).
    The restored tensors follow the same caller gradient mode as prepare().
    """
    if not isinstance(belief, BeliefState):
      raise TypeError("Restore requires a complete BeliefState, not an RPV view")
    track_grad = self.differentiable and torch.is_grad_enabled()
    with torch.inference_mode(False), torch.set_grad_enabled(track_grad):
      model_write_generation = self.estimator.dynamics.write_generation
      if model_version != self.model_version:
        raise ValueError(
          "Cannot restore belief from a different model/normalization version"
        )
      belief = self._belief(belief)
      context = self._ordinary(context)
      belief.validate()
      if belief.covariance.ndim != 3:
        raise ValueError("Provider restore requires one batch dimension")
      batch = belief.covariance.shape[0]
      self._validate_context(context, batch, belief.covariance)
      if episode_id is None:
        raise ValueError("Restoring a posterior requires its episode_id")
      episode_id = episode_id.detach().clone()
      self._check_index(episode_id, batch, context.device)
      if time_index is None:
        raise ValueError("Restoring a posterior requires its time_index")
      time_index = time_index.detach().clone()
      self._check_index(time_index, batch, context.device)
      if head_priors is not None:
        ids, prior = head_priors
        expected = episode_id[time_index == 0]
        self._check_index(ids, expected.numel(), context.device)
        if not torch.equal(ids, expected) or prior.covariance.shape != (
          expected.numel(),
          70,
          70,
        ):
          raise ValueError("Restored head priors must match the frame-zero episodes")
        prior.validate()
        if (
          prior.covariance.dtype != context.dtype
          or prior.covariance.device != context.device
        ):
          raise ValueError("Restored head priors must match model dtype and device")
        head_priors = (ids.detach().clone(), prior.detach().clone())
      return PreparedBelief(
        ProviderState(
          belief, context, episode_id, time_index, model_version, head_priors
        ),
        self._generation,
        self,
        model_version,
        model_write_generation,
      )

  def commit(self, candidate: PreparedBelief) -> None:
    if candidate.owner is not self or candidate.generation != self._generation:
      raise ValueError("Stale or foreign provider candidate")
    if (
      candidate.model_version != self.model_version
      or candidate.model_write_generation != self.estimator.dynamics.write_generation
    ):
      raise ValueError("Model changed before committing posterior")
    self.state = candidate.state
    self.diagnostics = candidate.diagnostics
    self._pending_prior = None
    self._generation += 1

  def restore(
    self,
    belief,
    context,
    *,
    model_version,
    episode_id=None,
    time_index=None,
    head_priors=None,
  ) -> None:
    self.commit(
      self.prepare_restore(
        belief,
        context,
        model_version=model_version,
        episode_id=episode_id,
        time_index=time_index,
        head_priors=head_priors,
      )
    )

  def __call__(self, **kwargs):
    """Convenience single transaction; adapters use prepare/commit instead."""
    candidate = self.prepare(**kwargs)
    self.commit(candidate)
    return candidate.to_actor_input()

  def state_dict(self) -> dict:
    if self.state is None:
      raise ValueError("Provider has no posterior to save")
    state = self.state
    heads = None
    if state.head_priors is not None:
      ids, prior = state.head_priors
      heads = {
        "episode_id": ids.detach().clone(),
        "mean": prior.mean.to_full().detach().clone(),
        "covariance": prior.covariance.detach().clone(),
      }
    return {
      "version": PROVIDER_VERSION,
      "state_layout_version": state.belief.state_layout_version,
      "tangent_version": state.belief.tangent_version,
      "model_version": self.estimator.dynamics.model_version
      if self.incremental
      else state.model_version,
      "model_revision": self.estimator.dynamics.model_revision,
      "posterior_revision": state.model_version if self.incremental else None,
      "filter_config": asdict(self.estimator.config),
      "measurement_config": asdict(self.estimator.measurement_config),
      "default_joint_pos": self.estimator.default_joint_pos.detach().clone(),
      "mean": state.belief.mean.to_full().detach().clone(),
      "rotation": state.belief.mean.rotation.detach().clone(),
      "covariance": state.belief.covariance.detach().clone(),
      "context": state.context.detach().clone(),
      "episode_id": state.episode_id.detach().clone(),
      "time_index": state.time_index.detach().clone(),
      "head_priors": heads,
    }

  def prepare_load_state_dict(self, data: dict) -> PreparedBelief:
    if data.get("version") != PROVIDER_VERSION:
      raise ValueError("Unsupported provider state version")
    if (
      data.get("state_layout_version") != STATE_LAYOUT_VERSION
      or data.get("tangent_version") != TANGENT_VERSION
    ):
      raise ValueError("Unsupported provider state/tangent layout")
    if (
      data.get("filter_config") != asdict(self.estimator.config)
      or data.get("measurement_config") != asdict(self.estimator.measurement_config)
      or not torch.equal(data["default_joint_pos"], self.estimator.default_joint_pos)
    ):
      raise ValueError("Provider/filter configuration mismatch")
    mean = LieState.from_full(data["mean"])
    rotation = data["rotation"]
    tol = 128 * torch.finfo(mean.rotation.dtype).eps
    if (
      rotation.shape != mean.rotation.shape
      or rotation.dtype != mean.rotation.dtype
      or rotation.device != mean.rotation.device
      or not torch.allclose(mean.rotation, rotation, atol=tol, rtol=tol)
    ):
      raise ValueError("Snapshot rotation disagrees with the packed mean")
    mean = replace(mean, rotation=rotation)
    belief = BeliefState(mean, data["covariance"])
    heads = data["head_priors"]
    head_priors = (
      None
      if heads is None
      else (
        heads["episode_id"],
        BeliefState(LieState.from_full(heads["mean"]), heads["covariance"]),
      )
    )
    if data["model_version"] != self.estimator.dynamics.model_version:
      raise ValueError("Provider snapshot model version mismatch")
    version = self.model_version
    source = data.get("posterior_revision")
    if source is not None:
      revision = data.get("model_revision")
      if (
        type(revision) is not int
        or type(source) is not int
        or not 0 <= source <= revision
      ):
        raise ValueError("Invalid provider snapshot revision")
      if self.incremental and revision != version:
        raise ValueError("Provider snapshot model revision mismatch")
      if source != revision and (not self.incremental or self.differentiable):
        raise ValueError("Historical posterior requires an incremental real provider")
    candidate = self.prepare_restore(
      belief,
      data["context"],
      model_version=version,
      episode_id=data["episode_id"],
      time_index=data["time_index"],
      head_priors=head_priors,
    )
    if self.incremental and source is not None:
      # Preserve the last filter revision while committing against today's model.
      return replace(candidate, state=replace(candidate.state, model_version=source))
    return candidate

  def load_state_dict(self, data: dict) -> None:
    self.commit(self.prepare_load_state_dict(data))
