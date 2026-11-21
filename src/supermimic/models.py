from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.modules import EmpiricalNormalization
from tensordict import TensorDict

from supermimic.utils.lie_state import LieState, boxplus
from supermimic.utils.validation import validate_state_dict

if TYPE_CHECKING:
  from supermimic.representations import StateRepresentation
  from supermimic.tasks.task_spec import TaskSpec


class _CudaGraphNormalization(EmpiricalNormalization):
  """EmpiricalNormalization with in-place _std update for CUDA graph compatibility.

  The upstream update() does ``self._std = torch.sqrt(self._var)`` which creates
  a new tensor, invalidating data pointers captured by CUDA graphs (the
  captured training updates). This subclass uses copy_() instead.
  """

  @torch.jit.unused
  def update(self, x: torch.Tensor) -> None:
    std_buf = self._std
    super().update(x)
    std_buf.copy_(self._std)
    self._std = std_buf


def build_mlp(
  input_dim: int,
  hidden_dims: tuple[int, ...],
  output_dim: int,
) -> nn.Sequential:
  """Build an MLP: ``input_dim → (Linear → LN → GELU)* → Linear(output_dim)``."""
  layers: list[nn.Module] = []
  curr = input_dim
  for h in hidden_dims:
    layers += [nn.Linear(curr, h), nn.LayerNorm(h), nn.GELU()]
    curr = h
  layers.append(nn.Linear(curr, output_dim))
  return nn.Sequential(*layers)


@dataclass(frozen=True)
class ProcessNoiseConfig:
  """One-control-step right-local process noise, in physical tangent units.

  ``initial_std`` follows theta, rho, nu, omega_body, q, qdot. Each value
  expands across its block (3, 3, 3, 3, 29, 29); no additional dt scaling
  is applied. These are initialization settings, not calibrated estimates.
  """

  initial_std: tuple[float, ...] = (0.01, 0.001, 0.05, 0.05, 0.005, 0.05)
  diagonal_floor: float = 1e-6
  hidden_dims: tuple[int, ...] = (128, 64)

  def __post_init__(self) -> None:
    if not math.isfinite(self.diagonal_floor) or self.diagonal_floor <= 0:
      raise ValueError("Process-noise diagonal_floor must be finite and positive")
    if len(self.initial_std) != 6 or any(
      not math.isfinite(s) or s <= self.diagonal_floor for s in self.initial_std
    ):
      raise ValueError("initial_std requires six finite values above diagonal_floor")
    if any(not isinstance(d, int) or d < 1 for d in self.hidden_dims):
      raise ValueError("Process-noise hidden widths must be positive integers")

  def standard_deviations(self, *, device=None, dtype=None) -> torch.Tensor:
    return torch.cat(
      [
        torch.full((width,), value, device=device, dtype=dtype)
        for width, value in zip((3, 3, 3, 3, 29, 29), self.initial_std, strict=True)
      ]
    )


class Dynamics(nn.Module):
  """Differentiable forward model: ``(rep_state, action, aux_state) -> (next_aux, next_rep)``.

  ``action`` is the raw post-clamp policy output — same distribution the
  policy sees back as ``last_action`` in its observation. Dynamics applies
  ``proc = action * scale + offset`` internally before normalization.
  """

  def __init__(
    self,
    representation: StateRepresentation,
    num_actions: int,
    sim_ctx_dim: int,
    action_scale: torch.Tensor,
    action_offset: torch.Tensor,
    hidden_dims: tuple[int, ...],
    process_noise: ProcessNoiseConfig | None = None,
  ):
    super().__init__()

    self.sim_ctx_dim = sim_ctx_dim
    self.num_rep_state = representation.rep_state_dim
    self.num_actions = num_actions
    self.dt = representation.dt
    self._apply_prediction = representation.apply_prediction
    self.process_noise_config = process_noise
    # Bound methods avoid registering the representation (and its simulator
    # helpers) as a submodule. Deterministic legacy forward stays unchanged.
    self._represent_lie = getattr(representation, "represent_lie", None)
    self._apply_prediction_lie = getattr(representation, "apply_prediction_lie", None)

    self.register_buffer("action_scale", action_scale.clone())
    self.register_buffer("action_offset", action_offset.clone())

    self.state_normalizer = _CudaGraphNormalization(self.num_rep_state)
    self.action_normalizer = _CudaGraphNormalization(num_actions)

    self.mlp = build_mlp(
      self.num_rep_state + num_actions + sim_ctx_dim,
      hidden_dims,
      output_dim=representation.prediction_dim,
    )
    self.noise_head: nn.Sequential | None = None
    if process_noise is not None:
      from supermimic.representations import GenCoordRepresentation

      if not isinstance(representation, GenCoordRepresentation) or num_actions != 29:
        raise ValueError(
          "Stochastic dynamics currently requires G1 GenCoord (29 joints)"
        )
      if representation.num_joints != 29:
        raise ValueError("Stochastic dynamics requires a 70-dimensional tangent")
      self.noise_head = build_mlp(
        self.num_rep_state + num_actions + sim_ctx_dim,
        process_noise.hidden_dims,
        output_dim=2485,
      )
      rows, cols = torch.tril_indices(70, 70)
      self.register_buffer("_noise_indices", rows * 70 + cols, persistent=False)
      output = self.noise_head[-1]
      nn.init.zeros_(output.weight)
      nn.init.zeros_(output.bias)
      std = process_noise.standard_deviations(dtype=torch.float64)
      target = std - process_noise.diagonal_floor
      # Numerically stable inverse softplus, giving the requested initial std.
      inverse = target + torch.log(-torch.expm1(-target))
      with torch.no_grad():
        output.bias[rows == cols] = inverse.to(output.bias)

  def proc(self, action: torch.Tensor) -> torch.Tensor:
    """Map raw policy output to proc joint commands: ``action * scale + offset``."""
    return action * self.action_scale + self.action_offset

  def forward(
    self,
    rep_state: torch.Tensor,
    action: torch.Tensor,
    aux_state: torch.Tensor,
    sim_ctx: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    if self.process_noise_config is not None:
      return self.mean_lie(
        LieState.from_rep(aux_state, rep_state), action, sim_ctx
      ).to_rep()
    return self.mean_rep(rep_state, action, aux_state, sim_ctx)

  def mean_rep(
    self,
    rep_state: torch.Tensor,
    action: torch.Tensor,
    aux_state: torch.Tensor,
    sim_ctx: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """SuperMimic mean transition in rollout coordinates."""
    prediction = self.mlp(self._normalized_inputs(rep_state, action, sim_ctx))
    return self._apply_prediction(aux_state, rep_state, prediction, self.dt)

  def mean_lie(
    self, state: LieState, action: torch.Tensor, sim_ctx: torch.Tensor
  ) -> LieState:
    """Matrix mean transition; validation belongs outside jacrev/vmap."""
    if self._represent_lie is None or self._apply_prediction_lie is None:
      raise ValueError("Lie dynamics requires GenCoord representation")
    inputs = self.lie_inputs(state, action, sim_ctx)
    return self._apply_prediction_lie(state, self.mlp(inputs), self.dt)

  def lie_inputs(
    self, state: LieState, action: torch.Tensor, sim_ctx: torch.Tensor
  ) -> torch.Tensor:
    if self._represent_lie is None:
      raise ValueError("Lie dynamics requires GenCoord representation")
    return self._normalized_inputs(self._represent_lie(state), action, sim_ctx)

  def noise_factor(self, inputs: torch.Tensor) -> torch.Tensor:
    if self.noise_head is None or self.process_noise_config is None:
      raise ValueError("distribution() requires an explicit ProcessNoiseConfig")
    values = self.noise_head(inputs)
    indices = self._noise_indices.expand_as(values)
    raw = values.new_zeros(*values.shape[:-1], 4900).scatter(-1, indices, values)
    raw = raw.reshape(*values.shape[:-1], 70, 70)
    scale = (
      F.softplus(raw.diagonal(dim1=-2, dim2=-1))
      + self.process_noise_config.diagonal_floor
    )
    # Each row has a different physical unit. Learn unrestricted off-diagonals
    # in units of its diagonal so an Adam step has a comparable relative size.
    return raw.tril(-1) * scale.unsqueeze(-1) + torch.diag_embed(scale)

  def distribution_lie(
    self, state: LieState, action: torch.Tensor, sim_ctx: torch.Tensor
  ) -> tuple[LieState, torch.Tensor]:
    """Matrix mean and full right-local process factor, sharing model inputs."""
    if self._represent_lie is None or self._apply_prediction_lie is None:
      raise ValueError("Lie dynamics requires GenCoord representation")
    inputs = self.lie_inputs(state, action, sim_ctx)
    return (
      self._apply_prediction_lie(state, self.mlp(inputs), self.dt),
      self.noise_factor(inputs),
    )

  def sample_lie(
    self,
    state: LieState,
    action: torch.Tensor,
    sim_ctx: torch.Tensor,
    epsilon: torch.Tensor,
  ) -> LieState:
    """Reparameterized matrix sample with caller-owned standard normal noise."""
    if epsilon.shape != (*state.position.shape[:-1], 70):
      raise ValueError("Process epsilon must match the batch shape and have 70 entries")
    mean, lower = self.distribution_lie(state, action, sim_ctx)
    return boxplus(mean, (lower @ epsilon.unsqueeze(-1)).squeeze(-1))

  def _normalized_inputs(
    self, rep_state: torch.Tensor, action: torch.Tensor, sim_ctx: torch.Tensor
  ) -> torch.Tensor:
    state_norm = self.state_normalizer(rep_state).reshape_as(rep_state)
    action_norm = self.action_normalizer(self.proc(action)).reshape_as(action)
    return torch.cat([state_norm, action_norm, sim_ctx], dim=-1)

  def distribution(
    self,
    rep_state: torch.Tensor,
    action: torch.Tensor,
    aux_state: torch.Tensor,
    sim_ctx: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return deterministic mean and its full right-local Cholesky factor."""
    if self.noise_head is None or self.process_noise_config is None:
      raise ValueError("distribution() requires an explicit ProcessNoiseConfig")
    mean, factor = self.distribution_lie(
      LieState.from_rep(aux_state, rep_state), action, sim_ctx
    )
    next_aux, next_rep = mean.to_rep()
    return next_aux, next_rep, factor

  def sample(
    self,
    rep_state: torch.Tensor,
    action: torch.Tensor,
    aux_state: torch.Tensor,
    sim_ctx: torch.Tensor,
    epsilon: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Reparameterized sample using caller-owned standard normal noise."""
    return self.sample_lie(
      LieState.from_rep(aux_state, rep_state), action, sim_ctx, epsilon
    ).to_rep()

  def _checkpoint_metadata(self) -> dict[str, Any]:
    from supermimic.utils.lie_state import STATE_LAYOUT_VERSION, TANGENT_VERSION

    return {
      # The row-scaled process factor changes the stochastic snapshot format.
      "format_version": (
        "stochastic_dynamics_v3"
        if self.process_noise_config
        else "stochastic_dynamics_v2"
      ),
      "state_layout": STATE_LAYOUT_VERSION,
      "tangent_version": TANGENT_VERSION,
      "process_noise": (
        asdict(self.process_noise_config) if self.process_noise_config else None
      ),
      "dt": self.dt,
      "integration": {
        "version": "right_matrix_v2" if self.process_noise_config else "quaternion_v1",
        "representation": type(self._apply_prediction.__self__).__name__,
        "substep_coeff": getattr(
          self._apply_prediction.__self__, "substep_coeff", None
        ),
      },
    }

  @property
  def model_version(self) -> str:
    """Fingerprint the current model at a provider or checkpoint boundary.

    Fused optimizers and CUDA graph replay can change weights without
    incrementing PyTorch's tensor version counters.
    """
    return self._content_digest(self._checkpoint_metadata())

  def _content_digest(self, metadata, state=None) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(metadata, sort_keys=True).encode())
    for name, current in (*self.named_parameters(), *self.named_buffers()):
      # Derived, nonpersistent buffers (e.g. the triangle indices) come
      # from the model configuration and are also part of the fingerprint.
      value = current if state is None else state.get(name, current)
      digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}".encode())
      raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
      digest.update(raw.numpy())
    return digest.hexdigest()

  def checkpoint(self) -> dict[str, Any]:
    """Self-describing model snapshot, separate from online belief state."""
    metadata = self._checkpoint_metadata()
    state = {key: value.detach().clone() for key, value in self.state_dict().items()}
    return {
      "metadata": metadata,
      "state_dict": state,
      "model_version": self._content_digest(metadata, state),
    }

  def validate_checkpoint(self, checkpoint: dict[str, Any]) -> None:
    """Validate without changing parameters, buffers or the model version."""
    if checkpoint.get("metadata") != self._checkpoint_metadata():
      raise ValueError("Dynamics checkpoint configuration or state layout mismatch")
    supplied = checkpoint.get("state_dict")
    validate_state_dict(self, supplied, "Dynamics")
    digest = self._content_digest(checkpoint["metadata"], supplied)
    if digest != checkpoint.get("model_version"):
      raise ValueError("Dynamics checkpoint content does not match its model version")

  def load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
    """Validate the complete snapshot before mutating the live model."""
    self.validate_checkpoint(checkpoint)
    self.load_state_dict(checkpoint["state_dict"], strict=True)

  def normalizers(self) -> list[_CudaGraphNormalization]:
    return [self.state_normalizer, self.action_normalizer]


class MultiEncoderAE(nn.Module):
  """Per-task goal encoders + shared decoder, tanh-bounded latent.

  Slots are contiguous blocks of the batch routed to each task's encoder.
  ``slot_mix`` is resolved into ``(lo, hi)`` per forward from the current
  batch size, so one instance serves both sim collection and BPTT rollout.

  Aux loss: latent BC pulling each non-anchor encoder's latent toward the
  anchor encoder's latent on the same state (anchor query stop-grad).

  Obs TensorDict layout:
    ``obs_td["history"]``            — last ``history_length`` obs frames
                                       (sensor or rep_state), (B, H, obs_dim),
                                       newest last; flattened into the
                                       decoder input
    ``obs_td[f"goal_{task.name}"]``  — full-batch goal per task; only the
                                       task's slot range is read
    ``obs_td["task_uncertainty"]``  — task covariance features (B, uncertainty_dim),
                                       required only when FiLM is enabled
  """

  def __init__(
    self,
    obs_dim: int,
    action_dim: int,
    task_specs: list[TaskSpec],
    slot_mix: dict[str, float],
    latent_dim: int,
    hidden_dims: tuple[int, ...],
    latent_bc_weight: float,
    history_length: int,
    uncertainty_dim: int = 0,
    film_hidden_dims: tuple[int, ...] = (128, 64),
    input_version: str = "legacy",
  ):
    super().__init__()
    if not task_specs:
      raise ValueError("task_specs must be non-empty")
    if history_length < 1:
      raise ValueError("history_length must be positive")
    if uncertainty_dim < 0:
      raise ValueError("uncertainty_dim must be non-negative")
    if uncertainty_dim and (not hidden_dims or not film_hidden_dims):
      raise ValueError("FiLM requires decoder and uncertainty hidden layers")
    self.obs_dim = obs_dim
    self.history_length = history_length
    self.latent_dim = latent_dim
    self.latent_bc_weight = latent_bc_weight
    self.task_specs = task_specs
    self.uncertainty_dim = uncertainty_dim
    self.input_version = input_version
    self._task_names = [ts.name for ts in task_specs]
    # Anchor: the first task. The aux-loss hook regularizes each non-anchor
    # encoder against the anchor encoder on the same state.
    self._anchor = self._task_names[0]
    self._aux_loss = torch.zeros(0)

    self.obs_normalizer = _CudaGraphNormalization(obs_dim)
    self.goal_normalizers = nn.ModuleDict(
      {ts.name: _CudaGraphNormalization(ts.goal_dim) for ts in task_specs}
    )
    self.encoders = nn.ModuleDict(
      {
        ts.name: build_mlp(ts.goal_dim, hidden_dims, output_dim=latent_dim)
        for ts in task_specs
      }
    )
    self.decoder = build_mlp(
      latent_dim + history_length * obs_dim,
      hidden_dims,
      output_dim=action_dim,
    )
    # A/B have no uncertainty parameters and never access the uncertainty input.
    # Keep the base decoder Sequential intact for legacy checkpoints; C applies
    # its modulation between that decoder's first LayerNorm and GELU.
    self.film: nn.Sequential | None = None
    if uncertainty_dim:
      self.film = build_mlp(
        uncertainty_dim, film_hidden_dims, output_dim=2 * hidden_dims[0]
      )
      nn.init.zeros_(self.film[-1].weight)
      nn.init.zeros_(self.film[-1].bias)

    self.set_slot_mix(slot_mix)

  @torch.jit.unused
  def copy_common_initialization_from(self, source: MultiEncoderAE) -> None:
    """Copy base parameters and normalization buffers, without sharing storage.

    Call before creating each group's optimizer. FiLM remains independently
    initialized, so copying a B base into C preserves C's identity modulation.
    Input-version checks prevent equal-width but differently ordered layouts
    from being treated as compatible. The runner persists the full input config.
    """
    signature = (
      "obs_dim",
      "history_length",
      "latent_dim",
      "input_version",
      "_task_names",
    )
    if any(getattr(self, key) != getattr(source, key) for key in signature):
      raise ValueError(
        "Cannot copy common initialization across different actor inputs"
      )
    for name in ("obs_normalizer", "goal_normalizers", "encoders", "decoder"):
      dst_state = getattr(self, name).state_dict()
      src_state = getattr(source, name).state_dict()
      if dst_state.keys() != src_state.keys() or any(
        dst_state[key].shape != src_state[key].shape for key in dst_state
      ):
        raise ValueError(f"Cannot copy common initialization: incompatible {name}")
    for name in ("obs_normalizer", "goal_normalizers", "encoders", "decoder"):
      getattr(self, name).load_state_dict(getattr(source, name).state_dict())

  def set_slot_mix(self, slot_mix: dict[str, float]) -> None:
    """Set fractional slot allocation per task. Missing entries default to 0."""
    unknown = set(slot_mix) - set(self._task_names)
    if unknown:
      raise ValueError(f"slot_mix has unknown tasks {sorted(unknown)}")
    fractions = [slot_mix.get(name, 0.0) for name in self._task_names]
    if any(frac < 0.0 for frac in fractions):
      raise ValueError(f"slot_mix fractions must be non-negative; got {slot_mix}")
    total = sum(fractions)
    if abs(total - 1.0) > 1e-6:
      raise ValueError(f"slot_mix must sum to 1.0; got {total}")
    self._slot_fractions = fractions

  def slot_ranges(self, batch_size: int) -> list[tuple[int, int]]:
    """Contiguous ``(lo, hi)`` per task at the given batch size; last absorbs remainder."""
    ranges: list[tuple[int, int]] = []
    off = 0
    last = len(self._slot_fractions) - 1
    for i, frac in enumerate(self._slot_fractions):
      hi = batch_size if i == last else off + int(round(batch_size * frac))
      ranges.append((off, hi))
      off = hi
    return ranges

  def forward(self, obs_td: TensorDict) -> torch.Tensor:
    history = obs_td["history"]
    if history.shape[1:] != (self.history_length, self.obs_dim):
      raise ValueError(
        "Actor history must have shape "
        f"(batch, {self.history_length}, {self.obs_dim}); got {tuple(history.shape)}"
      )
    slots = self.slot_ranges(obs_td.shape[0])
    enc_outs: list[torch.Tensor] = []
    for name, (lo, hi) in zip(self._task_names, slots, strict=True):
      if lo == hi:
        continue
      goal_n = self.goal_normalizers[name](obs_td[f"goal_{name}"][lo:hi])
      enc_outs.append(self.encoders[name](goal_n))

    self._aux_loss = self._compute_aux_loss(obs_td, slots)

    z = torch.cat([torch.tanh(eo) for eo in enc_outs], dim=0)
    history_n = self.obs_normalizer(history)
    decoder_input = torch.cat([z, history_n.flatten(start_dim=1)], dim=-1)
    if self.film is None:
      return self.decoder(decoder_input)
    if "task_uncertainty" not in obs_td.keys():
      raise ValueError("FiLM actor requires task_uncertainty input")
    uncertainty = obs_td["task_uncertainty"]
    if uncertainty.shape != (history.shape[0], self.uncertainty_dim):
      raise ValueError(
        "task_uncertainty must have shape "
        f"(batch, {self.uncertainty_dim}); got {tuple(uncertainty.shape)}"
      )
    gamma, beta = self.film(uncertainty).chunk(2, dim=-1)
    hidden = self.decoder[1](self.decoder[0](decoder_input))
    hidden = (1.0 + gamma) * hidden + beta
    for layer in self.decoder[2:]:
      hidden = layer(hidden)
    return hidden

  def _compute_aux_loss(
    self, obs_td: TensorDict, slots: list[tuple[int, int]]
  ) -> torch.Tensor:
    """Per-env BC loss pulling each non-anchor encoder's latent toward the
    anchor encoder's latent on the same state.

    Goals are detached before the BC encoder forwards so BC's gradient only
    updates the non-anchor encoder's parameters — never the decoder or the
    anchor encoder.

    Anchor query is also under ``no_grad``. Skipped in eval mode.
    Returns ``(num_envs,)``; anchor / empty slots contribute zero.
    """
    out = torch.zeros(obs_td.shape[0], device=obs_td.device)
    if not self.training or self.latent_bc_weight == 0.0:
      return out
    anchor = self._anchor
    anchor_goal_d = self.goal_normalizers[anchor](obs_td[f"goal_{anchor}"]).detach()
    for name, (lo, hi) in zip(self._task_names, slots, strict=True):
      if name == anchor or lo == hi:
        continue
      goal_n_d = self.goal_normalizers[name](obs_td[f"goal_{name}"][lo:hi]).detach()
      z = torch.tanh(self.encoders[name](goal_n_d))
      with torch.no_grad():
        z_anchor = torch.tanh(self.encoders[anchor](anchor_goal_d[lo:hi]))
      out[lo:hi] = self.latent_bc_weight * ((z - z_anchor) ** 2).mean(dim=-1)
    return out

  def auxiliary_loss(self) -> torch.Tensor:
    return self._aux_loss

  def auxiliary_stats(self) -> dict[str, torch.Tensor]:
    if self._aux_loss.numel() == 0:
      return {}
    return {"bc_loss": self._aux_loss.mean().detach()}

  @torch.jit.unused
  def update_normalization(self, obs_td: TensorDict) -> None:
    obs = obs_td["history"][:, -1]
    self.obs_normalizer.update(obs)
    for name, (lo, hi) in zip(
      self._task_names, self.slot_ranges(obs.shape[0]), strict=True
    ):
      if lo == hi:
        continue
      self.goal_normalizers[name].update(obs_td[f"goal_{name}"][lo:hi])
