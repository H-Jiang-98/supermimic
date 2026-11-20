"""Versioned, differentiable inputs for the estimator-in-the-loop actor.

This module deliberately depends only on PyTorch. A caller supplies a current
state (or posterior) and an already aligned reference window; no simulator state
is looked up here. Rotations are matrices, linear velocities are world-frame,
and covariance coordinates are right-local ``[theta, rho, nu]``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Literal, Protocol

import torch

from supermimic.utils.validation import check_covariance, validate_rotation

INPUT_VERSION = "eit_rpv_v1"
COVARIANCE_ENCODING = "log_std_corr_v1"
NUM_REFERENCE_FRAMES = 10
GOAL_BLOCK_ORDER = (
  "joint_reference",
  "relative_position",
  "relative_rotation",
  "relative_velocity",
)
RPV_ERROR_ORDER = ("theta", "rho", "nu")
UNCERTAINTY_DIM = 45 * NUM_REFERENCE_FRAMES


@dataclass(frozen=True)
class GaussianMeasurementConfig:
  """Additive sensor standard deviations; known actions are never corrupted."""

  gravity_std: float = 0.05 / math.sqrt(3.0)
  angular_velocity_std: float = 0.2 / math.sqrt(3.0)
  joint_position_std: float = 0.01 / math.sqrt(3.0)
  joint_velocity_std: float = 0.5 / math.sqrt(3.0)

  def __post_init__(self) -> None:
    if any(not math.isfinite(x) or x < 0 for x in asdict(self).values()):
      raise ValueError("Measurement standard deviations must be finite and nonnegative")

  def standard_deviations(
    self, num_joints: int, *, device=None, dtype=None
  ) -> torch.Tensor:
    if num_joints <= 0:
      raise ValueError("num_joints must be positive")
    return torch.tensor(
      [self.gravity_std] * 3
      + [self.angular_velocity_std] * 3
      + [self.joint_position_std] * num_joints
      + [self.joint_velocity_std] * num_joints,
      device=device,
      dtype=dtype,
    )

  def covariance(self, num_joints: int, *, device=None, dtype=None) -> torch.Tensor:
    """The exact observation covariance V, excluding previous actions."""
    return torch.diag(
      self.standard_deviations(num_joints, device=device, dtype=dtype).square()
    )


@dataclass(frozen=True)
class EITInputConfig:
  group: Literal["A", "B", "C"] = "A"
  mode: Literal["train", "eval"] = "train"
  history_length: int = 5
  version: str = INPUT_VERSION
  covariance_encoding: str = COVARIANCE_ENCODING
  covariance_std_floor: float = 1e-6
  covariance_scales: tuple[float, float, float] = (1.0, 1.0, 1.0)
  measurement_noise: GaussianMeasurementConfig = field(
    default_factory=GaussianMeasurementConfig
  )

  def __post_init__(self) -> None:
    if self.group not in ("A", "B", "C"):
      raise ValueError(f"Unknown ablation group: {self.group}")
    if self.mode not in ("train", "eval"):
      raise ValueError(f"Unknown actor input mode: {self.mode}")
    if not isinstance(self.history_length, int) or self.history_length <= 0:
      raise ValueError("history_length must be a positive integer")
    if self.version != INPUT_VERSION or self.covariance_encoding != COVARIANCE_ENCODING:
      raise ValueError("Unsupported estimator actor input/covariance version")
    if not math.isfinite(self.covariance_std_floor) or self.covariance_std_floor <= 0:
      raise ValueError("covariance_std_floor must be finite and positive")
    if len(self.covariance_scales) != 3 or any(
      not math.isfinite(x) or x <= 0 for x in self.covariance_scales
    ):
      raise ValueError(
        "covariance_scales must contain positive rotation/position/velocity scales"
      )
    if not isinstance(self.measurement_noise, GaussianMeasurementConfig):
      raise TypeError("measurement_noise must be GaussianMeasurementConfig")

  @property
  def uses_posterior(self) -> bool:
    return self.group != "A" or self.mode == "eval"

  @property
  def uses_uncertainty(self) -> bool:
    return self.group == "C"

  @property
  def critic_uses_belief(self) -> bool:
    """A future full critic uses belief for B/C, regardless of actor FiLM."""
    return self.group != "A"

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)

  @classmethod
  def from_dict(cls, values: dict[str, Any]) -> EITInputConfig:
    values = dict(values)
    if "version" not in values or "covariance_encoding" not in values:
      raise ValueError("Checkpoint lacks estimator actor input version metadata")
    missing = {item.name for item in fields(cls)} - values.keys()
    if missing:
      raise ValueError(
        f"Checkpoint lacks estimator input configuration: {sorted(missing)}"
      )
    if isinstance(values.get("measurement_noise"), dict):
      required_noise = {item.name for item in fields(GaussianMeasurementConfig)}
      if required_noise != values["measurement_noise"].keys():
        raise ValueError("Checkpoint measurement noise configuration is incomplete")
      values["measurement_noise"] = GaussianMeasurementConfig(
        **values["measurement_noise"]
      )
    if "covariance_scales" in values:
      values["covariance_scales"] = tuple(values["covariance_scales"])
    return cls(**values)


@dataclass(frozen=True)
class RpvState:
  rotation: torch.Tensor
  position: torch.Tensor
  velocity: torch.Tensor


@dataclass(frozen=True)
class PosteriorInput:
  mean: RpvState
  covariance_rpv: torch.Tensor | None = None


class PosteriorProvider(Protocol):
  """Inject a filter or synthetic posterior using measured information only.

  ``measurement`` excludes the previous-action channels. An implementation may
  retain its own per-environment state; reset_mask identifies new collector
  episodes or new imagined rollout slots, never history-padding copies. Starting
  a synthetic rollout is not restoration of a real filter. Full belief providers
  use the separate keyed replay reset and prepare/commit interfaces instead.
  """

  def __call__(
    self,
    *,
    measurement: torch.Tensor,
    previous_action: torch.Tensor,
    reset_mask: torch.Tensor,
  ) -> PosteriorInput: ...


@dataclass(frozen=True)
class ReferenceWindow:
  joint_pos: torch.Tensor
  joint_vel: torch.Tensor
  rotation: torch.Tensor
  position: torch.Tensor
  velocity: torch.Tensor


@dataclass(frozen=True)
class EpisodeAlignment:
  """Fixed per-episode horizontal transform; never updated by actor inference."""

  rotation: torch.Tensor
  translation: torch.Tensor

  @classmethod
  def identity(cls, batch_size: int, *, device=None, dtype=None) -> EpisodeAlignment:
    return cls(
      torch.eye(3, device=device, dtype=dtype).expand(batch_size, 3, 3).clone(),
      torch.zeros(batch_size, 3, device=device, dtype=dtype),
    )

  def apply(self, reference: ReferenceWindow) -> ReferenceWindow:
    rotation = self.rotation[:, None]
    return ReferenceWindow(
      joint_pos=reference.joint_pos,
      joint_vel=reference.joint_vel,
      rotation=rotation @ reference.rotation,
      position=(rotation @ reference.position.unsqueeze(-1)).squeeze(-1)
      + self.translation[:, None],
      velocity=(rotation @ reference.velocity.unsqueeze(-1)).squeeze(-1),
    )


def apply_alignment(
  reference: ReferenceWindow, alignment: EpisodeAlignment
) -> ReferenceWindow:
  """Apply a saved per-episode transform without modifying shared motion data."""
  return alignment.apply(reference)


def create_episode_alignment(state: RpvState, reference: RpvState) -> EpisodeAlignment:
  """Align initial root yaw and xy; keep source reference height unchanged."""
  _validate_state(state)
  _validate_state(reference)
  yaw = torch.atan2(state.rotation[:, 1, 0], state.rotation[:, 0, 0])
  yaw = yaw - torch.atan2(reference.rotation[:, 1, 0], reference.rotation[:, 0, 0])
  c, s = yaw.cos(), yaw.sin()
  z, o = torch.zeros_like(c), torch.ones_like(c)
  rotation = torch.stack((c, -s, z, s, c, z, z, z, o), dim=-1).reshape(-1, 3, 3)
  delta = state.position - (rotation @ reference.position.unsqueeze(-1)).squeeze(-1)
  translation = torch.cat((delta[:, :2], torch.zeros_like(delta[:, 2:])), dim=-1)
  return EpisodeAlignment(rotation, translation)


def sample_proprio(
  frame: torch.Tensor,
  config: GaussianMeasurementConfig | None = None,
  *,
  generator: torch.Generator | None = None,
) -> torch.Tensor:
  """Sample exactly once from clean ``[g, omega, q-default, qdot, action]``.

  The caller owns the result's lifetime and shares it with actor/filter/replay.
  Replayed measurements must not be passed through this function again.
  """
  width = frame.shape[-1]
  if width <= 6 or (width - 6) % 3:
    raise ValueError("Proprio frame must have 6 + 3*num_joints channels")
  joints = (width - 6) // 3
  sensor_width = 6 + 2 * joints
  config = config or GaussianMeasurementConfig()
  std = config.standard_deviations(joints, device=frame.device, dtype=frame.dtype)
  noise = torch.randn(
    frame.shape[:-1] + (sensor_width,),
    device=frame.device,
    dtype=frame.dtype,
    generator=generator,
  )
  return torch.cat(
    (frame[..., :sensor_width] + noise * std, frame[..., sensor_width:]), dim=-1
  )


def select_actor_state(
  config: EITInputConfig, truth: RpvState | None, posterior: PosteriorInput | None
) -> RpvState:
  if config.uses_posterior:
    if posterior is None:
      raise ValueError(
        f"Group {config.group} in {config.mode} mode requires posterior mean"
      )
    state = posterior.mean
  else:
    if truth is None:
      raise ValueError("Group A training requires the privileged true state")
    state = truth
  _validate_state(state)
  return state


def _validate_state(state: RpvState) -> None:
  if state.position.ndim != 2 or state.position.shape[-1] != 3:
    raise ValueError("RpvState.position must have shape (batch, 3)")
  batch = state.position.shape[0]
  if state.rotation.shape != (batch, 3, 3) or state.velocity.shape != (batch, 3):
    raise ValueError("RpvState rotation/velocity shapes must be (batch,3,3)/(batch,3)")
  for value in (state.rotation, state.position, state.velocity):
    if value.device != state.position.device or value.dtype != state.position.dtype:
      raise ValueError("RpvState components must share dtype and device")
    if not torch.isfinite(value).all():
      raise ValueError("RpvState must contain finite values")
  validate_rotation(state.rotation)


def build_motion_goal(
  state: RpvState, reference: ReferenceWindow
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Return goal, local position errors and local velocity errors.

  Goal order is joints, position, row-major first-two-column rot6d, velocity.
  Joint count is generic; G1's 29 joints give 700 channels.
  """
  _validate_state(state)
  batch = state.position.shape[0]
  shape = (batch, NUM_REFERENCE_FRAMES)
  if (
    reference.joint_pos.ndim != 3
    or reference.joint_pos.shape[:2] != shape
    or reference.joint_vel.shape != reference.joint_pos.shape
    or reference.rotation.shape != (*shape, 3, 3)
    or reference.position.shape != (*shape, 3)
    or reference.velocity.shape != (*shape, 3)
  ):
    raise ValueError(
      "ReferenceWindow must contain ten consistent frames per environment"
    )
  for value in (
    reference.joint_pos,
    reference.joint_vel,
    reference.rotation,
    reference.position,
    reference.velocity,
  ):
    if value.dtype != state.position.dtype or value.device != state.position.device:
      raise ValueError("Reference and state must share dtype and device")
    if not torch.isfinite(value).all():
      raise ValueError("ReferenceWindow must contain finite values")
  validate_rotation(reference.rotation)
  inverse = state.rotation.transpose(-1, -2)[:, None]
  position = (
    inverse @ (reference.position - state.position[:, None]).unsqueeze(-1)
  ).squeeze(-1)
  velocity = (
    inverse @ (reference.velocity - state.velocity[:, None]).unsqueeze(-1)
  ).squeeze(-1)
  rotation = inverse @ reference.rotation
  joints = torch.cat((reference.joint_pos, reference.joint_vel), dim=-1)
  goal = torch.cat(
    (
      joints.flatten(1),
      position.flatten(1),
      rotation[..., :2].flatten(1),
      velocity.flatten(1),
    ),
    dim=-1,
  )
  return goal, position, velocity


def skew(vector: torch.Tensor) -> torch.Tensor:
  x, y, z = vector.unbind(-1)
  zero = torch.zeros_like(x)
  return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).reshape(
    *vector.shape[:-1], 3, 3
  )


def task_error_jacobian(
  position_error: torch.Tensor, velocity_error: torch.Tensor
) -> torch.Tensor:
  """Right-local state error -> [left orientation error, position, velocity]."""
  if position_error.shape != velocity_error.shape or position_error.shape[-1] != 3:
    raise ValueError("Position and velocity errors must have identical (...,3) shapes")
  eye = torch.eye(3, dtype=position_error.dtype, device=position_error.device)
  eye = eye.expand(*position_error.shape[:-1], 3, 3)
  zero = torch.zeros_like(eye)
  return torch.cat(
    (
      torch.cat((-eye, zero, zero), dim=-1),
      torch.cat((skew(position_error), -eye, zero), dim=-1),
      torch.cat((skew(velocity_error), zero, -eye), dim=-1),
    ),
    dim=-2,
  )


def extract_rpv_covariance(
  covariance: torch.Tensor, indices: tuple[int, ...] = tuple(range(9))
) -> torch.Tensor:
  """Extract a marginal block, retaining cross terms, from a full posterior.

  ``indices`` must explicitly follow theta/rho/nu order if the full filter's
  tangent ordering differs. No conditional covariance or Schur complement.
  """
  if covariance.ndim < 2 or covariance.shape[-2] != covariance.shape[-1]:
    raise ValueError("Full posterior covariance must be square")
  if (
    len(indices) != 9
    or len(set(indices)) != 9
    or min(indices) < 0
    or max(indices) >= covariance.shape[-1]
  ):
    raise ValueError("RPV marginal requires nine distinct valid tangent indices")
  index = torch.tensor(indices, device=covariance.device, dtype=torch.long)
  return covariance.index_select(-2, index).index_select(-1, index)


def validate_covariance(covariance: torch.Tensor) -> torch.Tensor:
  """Reject invalid covariance, allowing only roundoff-level asymmetry/PSD error."""
  if covariance.shape[-2:] != (9, 9) or not covariance.is_floating_point():
    raise ValueError("RPV/task covariance must be floating point (...,9,9)")
  check_covariance(
    covariance, "Covariance", rtol=64 * torch.finfo(covariance.dtype).eps
  )
  return (covariance + covariance.mT) * 0.5


def task_error_covariance(
  covariance_rpv: torch.Tensor,
  position_error: torch.Tensor,
  velocity_error: torch.Tensor,
) -> torch.Tensor:
  covariance_rpv = validate_covariance(covariance_rpv)
  if covariance_rpv.shape != (position_error.shape[0], 9, 9):
    raise ValueError(
      "One RPV covariance (batch,9,9) is required for the current posterior"
    )
  jacobian = task_error_jacobian(position_error, velocity_error)
  return jacobian @ covariance_rpv[:, None] @ jacobian.transpose(-1, -2)


def encode_covariance(
  covariance: torch.Tensor,
  *,
  std_floor: float = 1e-6,
  scales: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> torch.Tensor:
  """Per-frame [nine log stds, 36 row-major upper correlations]."""
  if not math.isfinite(std_floor) or std_floor <= 0:
    raise ValueError("std_floor must be finite and positive")
  if len(scales) != 3 or any(not math.isfinite(s) or s <= 0 for s in scales):
    raise ValueError("Covariance scales must be positive")
  covariance = validate_covariance(covariance)
  scale = covariance.new_tensor(scales).repeat_interleave(3)
  normalized = covariance / (scale[:, None] * scale[None, :])
  std = normalized.diagonal(dim1=-2, dim2=-1).clamp_min(std_floor**2).sqrt()
  correlation = normalized / (std.unsqueeze(-1) * std.unsqueeze(-2))
  rows, cols = torch.triu_indices(9, 9, offset=1, device=covariance.device)
  return torch.cat((std.log(), correlation[..., rows, cols].clamp(-1.0, 1.0)), dim=-1)


def build_actor_features(
  config: EITInputConfig,
  truth: RpvState | None,
  posterior: PosteriorInput | None,
  reference: ReferenceWindow,
) -> tuple[torch.Tensor, torch.Tensor | None]:
  state = select_actor_state(config, truth, posterior)
  goal, position_error, velocity_error = build_motion_goal(state, reference)
  if not config.uses_uncertainty:
    # Deliberately do not even validate covariance on A/B actor paths.
    return goal, None
  if posterior is None or posterior.covariance_rpv is None:
    raise ValueError("Group C actor requires posterior covariance_rpv")
  covariance = task_error_covariance(
    posterior.covariance_rpv, position_error, velocity_error
  )
  encoded = encode_covariance(
    covariance, std_floor=config.covariance_std_floor, scales=config.covariance_scales
  )
  return goal, encoded.flatten(1)
