"""Versioned, differentiable inputs for the estimator-in-the-loop actor.

This module deliberately depends only on PyTorch. A caller supplies a current
state (or posterior) and an already aligned reference window; no simulator state
is looked up here. Rotations are matrices, linear velocities are world-frame,
and covariance coordinates are right-local ``[theta, rho, nu]``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from functools import lru_cache
from typing import Any, Literal, Protocol

import torch

from supermimic.local_control import (
  GOAL_DIM,
  LocalReference,
  local_covariance_features,
  local_state,
  privileged_goal,
)
from supermimic.utils.lie_state import LieState
from supermimic.utils.validation import (
  check_covariance,
  covariance_encoding_invalid,
  rotation_valid,
  validate_rotation,
)

INPUT_VERSION = "eit_se3_inputs_v2"
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
  """Sensor noise with default +/-3 sigma matching the legacy uniform bounds.

  Known actions are never corrupted; Gaussian samples are not clipped.
  """

  gravity_std: float = 0.05 / 3.0
  angular_velocity_std: float = 0.2 / 3.0
  joint_position_std: float = 0.01 / 3.0
  joint_velocity_std: float = 0.5 / 3.0

  def __post_init__(self) -> None:
    for name, value in asdict(self).items():
      if not math.isfinite(value) or value <= 0:
        raise ValueError(f"Measurement {name} must be finite and strictly positive")

  @classmethod
  def from_dict(cls, values: object) -> GaussianMeasurementConfig:
    if not isinstance(values, dict) or values.keys() != {
      item.name for item in fields(cls)
    }:
      raise ValueError("Checkpoint measurement noise configuration is incomplete")
    return cls(**values)

  def _tensors(self, num_joints: int, *, device=None, dtype=None):
    if num_joints <= 0:
      raise ValueError("num_joints must be positive")
    # Resolve implicit devices before caching, including the current CUDA index.
    if (
      dtype is None
      or not isinstance(device, torch.device)
      or (device.type != "cpu" and device.index is None)
    ):
      reference = torch.empty(0, device=device, dtype=dtype)
      device, dtype = reference.device, reference.dtype
    return _measurement_tensors(self, num_joints, device, dtype)

  def standard_deviations(
    self, num_joints: int, *, device=None, dtype=None
  ) -> torch.Tensor:
    return self._tensors(num_joints, device=device, dtype=dtype)[0].clone()

  def covariance(self, num_joints: int, *, device=None, dtype=None) -> torch.Tensor:
    """The exact observation covariance V, excluding previous actions."""
    return torch.diag(self._tensors(num_joints, device=device, dtype=dtype)[1])


@lru_cache(maxsize=32)
@torch.inference_mode(False)
def _measurement_tensors(
  config: GaussianMeasurementConfig,
  num_joints: int,
  device: torch.device,
  dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Private constants shared by collection and autograd-enabled imagination."""
  if not dtype.is_floating_point:
    raise ValueError("Measurement standard deviations require a floating-point dtype")
  # Cached constants must outlive every torch.func transform that requests them.
  # Disabling inference mode alone still creates transform-wrapped tensors.
  with torch._C._DisableFuncTorch():
    std = torch.tensor(
      [config.gravity_std] * 3
      + [config.angular_velocity_std] * 3
      + [config.joint_position_std] * num_joints
      + [config.joint_velocity_std] * num_joints,
      device=device,
      dtype=dtype,
    )
    variance = std.square()
    valid = variance.isfinite() & (variance > 0)
    if not valid.all():
      group_valid = valid[[0, 3, 6, 6 + num_joints]].tolist()
      name = fields(config)[group_valid.index(False)].name
      raise ValueError(
        f"Measurement {name} variance must be finite and positive in {dtype}"
      )
  return std, variance


@dataclass(frozen=True)
class EITInputConfig:
  group: Literal["A", "B", "C", "mean", "belief"] = "A"
  mode: Literal["train", "eval"] = "train"
  history_length: int = 5
  residual_actor: bool = False
  version: str = INPUT_VERSION
  covariance_encoding: str | None = None
  covariance_std_floor: float = 1e-6
  covariance_scales: tuple[float, float, float] = (1.0, 1.0, 1.0)
  measurement_noise: GaussianMeasurementConfig = field(
    default_factory=GaussianMeasurementConfig
  )

  def __post_init__(self) -> None:
    if self.group not in ("A", "B", "C", "mean", "belief"):
      raise ValueError(f"Unknown ablation group: {self.group}")
    if type(self.residual_actor) is not bool or (
      self.residual_actor and not self.local_control
    ):
      raise ValueError("Residual control requires a local policy")
    if self.mode not in ("train", "eval"):
      raise ValueError(f"Unknown actor input mode: {self.mode}")
    if not isinstance(self.history_length, int) or self.history_length <= 0:
      raise ValueError("history_length must be a positive integer")
    expected_encoding = (
      "local68_raw_vech_v1" if self.local_control else COVARIANCE_ENCODING
    )
    if self.covariance_encoding is None:
      object.__setattr__(self, "covariance_encoding", expected_encoding)
    if self.version != INPUT_VERSION or self.covariance_encoding != expected_encoding:
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
  def local_control(self) -> bool:
    return self.group in ("mean", "belief")

  @property
  def goal_dim(self) -> int:
    return GOAL_DIM if self.local_control else 700

  @property
  def uses_posterior(self) -> bool:
    return self.group != "A" or self.mode == "eval"

  @property
  def uses_uncertainty(self) -> bool:
    return self.group in ("C", "belief")

  @property
  def critic_uses_belief(self) -> bool:
    """The critic uses RPV belief for B/C, regardless of actor FiLM."""
    return self.group != "A"

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)

  @classmethod
  def from_dict(cls, values: dict[str, Any]) -> EITInputConfig:
    values = dict(values)
    if "version" not in values or "covariance_encoding" not in values:
      raise ValueError("Checkpoint lacks estimator actor input version metadata")
    if values["version"] != INPUT_VERSION:
      raise ValueError(
        f"Unsupported estimator input version {values['version']!r}; expected {INPUT_VERSION}"
      )
    missing = {item.name for item in fields(cls)} - values.keys()
    if missing:
      raise ValueError(
        f"Checkpoint lacks estimator input configuration: {sorted(missing)}"
      )
    if isinstance(values.get("measurement_noise"), dict):
      values["measurement_noise"] = GaussianMeasurementConfig.from_dict(
        values["measurement_noise"]
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
  state: LieState | None = None
  factor: torch.Tensor | None = None


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
  local: LocalReference | None = None

  def index(self, rows) -> ReferenceWindow:
    return ReferenceWindow(
      self.joint_pos[rows],
      self.joint_vel[rows],
      self.rotation[rows],
      self.position[rows],
      self.velocity[rows],
      self.local.index(rows) if self.local is not None else None,
    )


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
      local=reference.local,
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
  std, _ = config._tensors(joints, device=frame.device, dtype=frame.dtype)
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
  values = (state.rotation, state.position, state.velocity)
  if any(
    v.device != state.position.device or v.dtype != state.position.dtype for v in values
  ):
    raise ValueError("RpvState components must share dtype and device")
  if not torch.stack([torch.isfinite(v).all() for v in values]).all():
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
  validate_reference_batch(
    reference, batch_size=len(state.position), like=state.position
  )
  return _motion_goal(state, reference)


def validate_reference_batch(
  reference: ReferenceWindow, *, batch_size: int, like: torch.Tensor, bodies=()
) -> None:
  """Check a complete reference batch with one host decision on the valid path."""
  shape = (batch_size, 1 if reference.local is not None else NUM_REFERENCE_FRAMES)
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
  values = (
    reference.joint_pos,
    reference.joint_vel,
    reference.rotation,
    reference.position,
    reference.velocity,
  )
  if reference.local is not None:
    values += (reference.local.goal, reference.local.y, reference.local.next_joint)
  if any(v.device != like.device or v.dtype != like.dtype for v in values):
    raise ValueError("Reference and state must share dtype and device")
  if not reference.rotation.is_floating_point():
    raise ValueError("Rotation must be a floating-point 3 by 3 matrix")
  finite = torch.stack([torch.isfinite(v).all() for v in values]).all()
  checks = [finite, rotation_valid(reference.rotation).all()]
  if bodies:
    checks.append(torch.stack([torch.isfinite(v).all() for v in bodies]).all())
  if not torch.stack(checks).all():
    if not finite:
      raise ValueError("ReferenceWindow must contain finite values")
    validate_rotation(reference.rotation)
    raise ValueError("Reference bodies must be finite")


def _motion_goal(
  state: RpvState, reference: ReferenceWindow
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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


def _goal_with_checked_reference(
  state: RpvState, reference: ReferenceWindow
) -> torch.Tensor:
  """Build a second goal against a reference already validated for the first."""
  _validate_state(state)
  if (
    state.position.shape[0] != reference.position.shape[0]
    or state.position.dtype != reference.position.dtype
    or state.position.device != reference.position.device
  ):
    raise ValueError("Reference and state must share batch, dtype and device")
  return _motion_goal(state, reference)[0]


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
  return task_error_covariance_kernel(covariance_rpv, position_error, velocity_error)


def task_error_covariance_kernel(covariance_rpv, position_error, velocity_error):
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
  constants = covariance_encoding_constants(covariance, scales, std_floor)
  return encode_covariance_kernel(covariance, constants)


def covariance_encoding_constants(reference, scales, std_floor):
  """Prepare and validate dtype-specific constants outside a captured rollout."""
  scale = reference.new_tensor(scales).repeat_interleave(3)
  floor = reference.new_tensor(std_floor**2)
  if not (
    torch.isfinite(scale.square()).all()
    and (scale.square() > 0).all()
    and torch.isfinite(floor)
    and floor > 0
  ):
    raise ValueError("Covariance scales/floor are not representable in this dtype")
  rows, cols = torch.triu_indices(9, 9, offset=1, device=reference.device)
  return scale, floor, rows, cols


def encode_covariance_kernel(covariance, constants):
  scale, floor, rows, cols = constants
  covariance = (covariance + covariance.mT) * 0.5
  normalized = covariance / (scale[:, None] * scale[None, :])
  std = normalized.diagonal(dim1=-2, dim2=-1).clamp_min(floor).sqrt()
  correlation = normalized / (std.unsqueeze(-1) * std.unsqueeze(-2))
  return torch.cat((std.log(), correlation[..., rows, cols].clamp(-1.0, 1.0)), -1)


def velocity_rotation_block(posterior, velocity_error):
  # R.T @ v_ref_world - v_body has rotation derivative [R.T @ v_ref_world]x.
  # The older externally supplied RPV marginal uses world velocity perturbations.
  if posterior.state is None:
    return velocity_error
  return velocity_error + posterior.state.velocity[:, None]


def build_actor_features(
  config: EITInputConfig,
  truth: RpvState | None,
  posterior: PosteriorInput | None,
  reference: ReferenceWindow,
  context: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
  if config.local_control:
    goal, condition = local_actor_features(posterior, reference, context)
    if not torch.isfinite(goal).all() or not torch.isfinite(condition).all():
      raise ValueError("Non-finite local actor input")
    return goal, condition
  state = select_actor_state(config, truth, posterior)
  validate_reference_batch(
    reference, batch_size=len(state.position), like=state.position
  )
  goal, position_error, velocity_error = _motion_goal(state, reference)
  if not config.uses_uncertainty:
    # Deliberately do not even validate covariance on A/B actor paths.
    return goal, None
  if posterior is None or posterior.covariance_rpv is None:
    raise ValueError("Group C actor requires posterior covariance_rpv")
  covariance = task_error_covariance(
    posterior.covariance_rpv,
    position_error,
    velocity_rotation_block(posterior, velocity_error),
  )
  encoded = encode_covariance(
    covariance, std_floor=config.covariance_std_floor, scales=config.covariance_scales
  )
  return goal, encoded.flatten(1)


def build_actor_critic_goals(
  config: EITInputConfig,
  truth: RpvState,
  posterior: PosteriorInput | None,
  reference: ReferenceWindow,
  context: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
  """Build both goals against one reference, using privilege only for the critic
  and actors configured for true-state input.
  """
  actor, uncertainty = build_actor_features(
    config, truth, posterior, reference, context
  )
  if config.local_control:
    return actor, uncertainty, privileged_goal(truth, reference)
  # Actor observations can be handed to a caller separately from the stored
  # privileged goal. Preserve independent storage even when values agree.
  critic = (
    _goal_with_checked_reference(truth, reference)
    if config.uses_posterior
    else actor.clone()
  )
  return actor, uncertainty, critic


def build_actor_critic_goals_kernel(
  config, truth, posterior, reference, constants, context=None
):
  """Tensor-only goals and numerical status for a fixed training configuration."""
  if config.local_control:
    goal, condition = local_actor_features(posterior, reference, context)
    invalid = ~torch.isfinite(condition).all(-1) | ~torch.isfinite(goal).all(-1)
    return goal, condition, privileged_goal(truth, reference), invalid
  state = posterior.mean if config.uses_posterior else truth
  actor, position_error, velocity_error = _motion_goal(state, reference)
  uncertainty = None
  invalid = torch.zeros(actor.shape[0], device=actor.device, dtype=torch.bool)
  if config.uses_uncertainty:
    covariance = posterior.covariance_rpv
    tolerance = 64 * torch.finfo(covariance.dtype).eps
    invalid = covariance_encoding_invalid(covariance, rtol=tolerance)
    covariance = (covariance + covariance.mT) * 0.5
    task_covariance = task_error_covariance_kernel(
      covariance, position_error, velocity_rotation_block(posterior, velocity_error)
    )
    # Inspect the projected covariance before the encoder can hide numerical
    # failures with variance floors or correlation clamps.
    invalid = invalid | covariance_encoding_invalid(
      task_covariance, rtol=tolerance
    ).flatten(1).any(-1)
    uncertainty = encode_covariance_kernel(task_covariance, constants).flatten(1)
  critic = _motion_goal(truth, reference)[0] if config.uses_posterior else actor.clone()
  return actor, uncertainty, critic, invalid


def local_actor_features(posterior, reference, context):
  if posterior is None or posterior.state is None or posterior.factor is None:
    raise ValueError("Local policy requires a full posterior mean and square root")
  if reference.local is None or context is None:
    raise ValueError("Local policy requires the current reference and physical context")
  y = local_state(posterior.state)
  covariance = local_covariance_features(posterior.state, posterior.factor)
  condition = torch.cat(
    (y, y - reference.local.y, context, covariance, reference.local.next_joint), -1
  )
  return reference.local.goal, condition
