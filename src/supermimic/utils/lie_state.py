"""Right-local geometry for the G1 state on SE(3) x R^64.

The 71-vector interchange format uses wxyz quaternions, world linear velocity,
and body angular velocity. The 70-vector tangent is (theta, rho, nu, omega,
q, qdot). Value validation is explicit, outside differentiable transforms.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Callable

import torch

from supermimic.utils.helpers import matrix_to_quat
from supermimic.utils.validation import validate_rotation

NUM_JOINTS = 29
FULL_STATE_DIM = 71
TANGENT_DIM = 70
STATE_LAYOUT_VERSION = "gencoord71_world_io_body_state_v2"
TANGENT_VERSION = "right_se3_additive_body_twist_q_qdot_v2"


def skew(vector: torch.Tensor) -> torch.Tensor:
  """Cross-product matrix, with arbitrary leading batch dimensions."""
  x, y, z = vector.unbind(-1)
  zero = torch.zeros_like(x)
  return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(
    *vector.shape[:-1], 3, 3
  )


def _eye(size: int, reference: torch.Tensor) -> torch.Tensor:
  return torch.eye(size, dtype=reference.dtype, device=reference.device)


def _mv(matrix: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
  return (matrix @ vector.unsqueeze(-1)).squeeze(-1)


def _matrix_from_quat(quaternion: torch.Tensor) -> torch.Tensor:
  """Eager wxyz conversion for nested jacrev/vmap and outer derivatives.

  The simulator's scripted equivalent can install a profiling-generated
  differentiable graph after warmup that does not support vmap backward.
  Keep this small geometry primitive in ordinary tensor operations.
  """
  scalar = quaternion[..., :1]
  vector = quaternion[..., 1:]
  squared = quaternion.square().sum(-1)[..., None, None]
  cross = skew(vector)
  return _eye(3, quaternion) + (2 / squared) * (
    scalar.unsqueeze(-1) * cross + cross @ cross
  )


def _so3_coefficients(theta: torch.Tensor) -> tuple[torch.Tensor, ...]:
  """Rodrigues coefficients with analytic Taylor branches through theta=0.

  Clamp the *inactive* exact branch before sqrt/division too: torch.where alone
  does not prevent NaNs in a backward pass through a singular inactive branch.
  """
  squared = theta.square().sum(-1)[..., None, None]
  safe_squared = squared.clamp_min(1e-4)
  angle = safe_squared.sqrt()
  a = torch.where(
    squared < 1e-4,
    1 - squared / 6 + squared.square() / 120 - squared.pow(3) / 5040,
    angle.sin() / angle,
  )
  b = torch.where(
    squared < 1e-4,
    0.5 - squared / 24 + squared.square() / 720 - squared.pow(3) / 40320,
    (1 - angle.cos()) / safe_squared,
  )
  c = torch.where(
    squared < 1e-4,
    1 / 6 - squared / 120 + squared.square() / 5040 - squared.pow(3) / 362880,
    (angle - angle.sin()) / (safe_squared * angle),
  )
  return a, b, c


def so3_exp(theta: torch.Tensor) -> torch.Tensor:
  a, b, _ = _so3_coefficients(theta)
  cross = skew(theta)
  return _eye(3, theta) + a * cross + b * (cross @ cross)


def so3_log(rotation: torch.Tensor) -> torch.Tensor:
  """Principal SO(3) logarithm; smooth near identity, branch cut at angle pi.

  A largest-component quaternion conversion also resolves the near-pi axis,
  where extracting an axis from the antisymmetric matrix would be singular.
  At exactly pi the axis sign follows the quaternion conversion's deterministic
  branch; derivatives across this unavoidable principal-log cut are undefined.
  """
  quaternion = matrix_to_quat(rotation)
  quaternion = torch.where(quaternion[..., :1] < 0, -quaternion, quaternion)
  scalar, vector = quaternion[..., :1], quaternion[..., 1:]
  squared = vector.square().sum(-1, keepdim=True)
  safe_norm = squared.clamp_min(1e-8).sqrt()
  safe_scalar = torch.where(squared < 1e-8, scalar, torch.ones_like(scalar))
  small_factor = (
    2 / safe_scalar
    - 2 * squared / (3 * safe_scalar.pow(3))
    + 2 * squared.square() / (5 * safe_scalar.pow(5))
    - 2 * squared.pow(3) / (7 * safe_scalar.pow(7))
  )
  factor = torch.where(
    squared < 1e-8,
    small_factor,
    2 * torch.atan2(safe_norm, scalar) / safe_norm,
  )
  return factor * vector


def so3_left_jacobian(theta: torch.Tensor) -> torch.Tensor:
  _, b, c = _so3_coefficients(theta)
  cross = skew(theta)
  return _eye(3, theta) + b * cross + c * (cross @ cross)


def so3_left_jacobian_inverse(theta: torch.Tensor) -> torch.Tensor:
  squared = theta.square().sum(-1)[..., None, None]
  safe_squared = squared.clamp_min(1e-4)
  half_angle = 0.5 * safe_squared.sqrt()
  coefficient = torch.where(
    squared < 1e-4,
    1 / 12 + squared / 720 + squared.square() / 30240 + squared.pow(3) / 1209600,
    (1 - half_angle / half_angle.tan()) / safe_squared,
  )
  cross = skew(theta)
  return _eye(3, theta) - 0.5 * cross + coefficient * (cross @ cross)


@dataclass(frozen=True)
class LieState:
  rotation: torch.Tensor
  position: torch.Tensor
  velocity: torch.Tensor
  omega: torch.Tensor
  joint_pos: torch.Tensor
  joint_vel: torch.Tensor

  def _map(self, operation: Callable[[torch.Tensor], torch.Tensor]) -> LieState:
    return LieState(*(operation(getattr(self, f.name)) for f in fields(self)))

  def clone(self) -> LieState:
    return self._map(torch.clone)

  def detach(self) -> LieState:
    return self._map(torch.Tensor.detach)

  def to(self, *args, **kwargs) -> LieState:
    result = self._map(lambda value: value.to(*args, **kwargs))
    if torch.finfo(result.rotation.dtype).eps < torch.finfo(self.rotation.dtype).eps:
      self.validate()
      # Reconstruct SO(3) at the destination precision.
      result = replace(
        result, rotation=_matrix_from_quat(matrix_to_quat(result.rotation))
      )
    return result

  def index(self, selection) -> LieState:
    """Select batch dimensions (integer selection may remove a batch axis)."""
    return self._map(lambda value: value[selection])

  @classmethod
  def from_full(cls, full: torch.Tensor) -> LieState:
    if full.shape[-1] != FULL_STATE_DIM:
      raise ValueError("G1 full state must have 71 components")
    rotation = _matrix_from_quat(full[..., 3:7])
    return cls(
      rotation,
      full[..., :3],
      _mv(rotation.mT, full[..., 36:39]),
      full[..., 39:42],
      full[..., 7:36],
      full[..., 42:71],
    )

  @property
  def world_velocity(self) -> torch.Tensor:
    return _mv(self.rotation, self.velocity)

  def to_full(self) -> torch.Tensor:
    return torch.cat(
      (
        self.position,
        matrix_to_quat(self.rotation),
        self.joint_pos,
        self.world_velocity,
        self.omega,
        self.joint_vel,
      ),
      dim=-1,
    )

  @classmethod
  def from_rep(cls, aux: torch.Tensor, rep: torch.Tensor) -> LieState:
    if aux.shape[-1] != 7 or rep.shape[-1] != 68:
      raise ValueError("G1 GenCoord requires aux(...,7) and rep(...,68)")
    rotation = _matrix_from_quat(aux[..., 3:7])
    # Match GenCoordRepresentation.reconstruct_full_state's source of height.
    position = torch.cat((aux[..., :2], rep[..., :1]), -1)
    return cls(
      rotation,
      position,
      rep[..., 4:7],
      rep[..., 7:10],
      rep[..., 10:39],
      rep[..., 39:68],
    )

  def to_rep(self) -> tuple[torch.Tensor, torch.Tensor]:
    aux = torch.cat((self.position, matrix_to_quat(self.rotation)), -1)
    # R.T @ (0,0,-1), avoiding an allocated/broadcasted gravity vector.
    gravity = -self.rotation[..., 2, :]
    rep = torch.cat(
      (
        self.position[..., 2:3],
        gravity,
        self.velocity,
        self.omega,
        self.joint_pos,
        self.joint_vel,
      ),
      dim=-1,
    )
    return aux, rep

  def validate_structure(self) -> None:
    """Check tensor metadata without reading device values."""
    batch = self.position.shape[:-1]
    for f in fields(self):
      value = getattr(self, f.name)
      tail = (
        (3, 3)
        if f.name == "rotation"
        else ((NUM_JOINTS,) if f.name in ("joint_pos", "joint_vel") else (3,))
      )
      if value.shape != (*batch, *tail) or not value.is_floating_point():
        raise ValueError(f"Invalid {f.name} shape or dtype in LieState")
      if value.device != self.position.device or value.dtype != self.position.dtype:
        raise ValueError("All LieState fields must have identical device and dtype")

  def validate(self) -> None:
    """Explicit numerical boundary check, outside differentiated kernels."""
    self.validate_structure()
    finite = torch.stack(
      [torch.isfinite(getattr(self, f.name).detach()).all() for f in fields(self)]
    )
    if not finite.all():
      name = fields(self)[finite.tolist().index(False)].name
      raise ValueError(f"LieState {name} must be finite")
    validate_rotation(self.rotation)


def compose(left: LieState, right: LieState) -> LieState:
  """Product group composition; body omega belongs to the Euclidean block."""
  return LieState(
    left.rotation @ right.rotation,
    left.position + _mv(left.rotation, right.position),
    left.velocity + right.velocity,
    left.omega + right.omega,
    left.joint_pos + right.joint_pos,
    left.joint_vel + right.joint_vel,
  )


def inverse(state: LieState) -> LieState:
  rotation = state.rotation.transpose(-1, -2)
  return LieState(
    rotation,
    -_mv(rotation, state.position),
    -state.velocity,
    -state.omega,
    -state.joint_pos,
    -state.joint_vel,
  )


def exp(delta: torch.Tensor) -> LieState:
  if delta.shape[-1] != TANGENT_DIM:
    raise ValueError("G1 right tangent must have 70 components")
  theta = delta[..., :3]
  jacobian = so3_left_jacobian(theta)
  return LieState(
    so3_exp(theta),
    _mv(jacobian, delta[..., 3:6]),
    delta[..., 6:9],
    delta[..., 9:12],
    delta[..., 12:41],
    delta[..., 41:70],
  )


def log(state: LieState) -> torch.Tensor:
  theta = so3_log(state.rotation)
  jacobian_inverse = so3_left_jacobian_inverse(theta)
  return torch.cat(
    (
      theta,
      _mv(jacobian_inverse, state.position),
      state.velocity,
      state.omega,
      state.joint_pos,
      state.joint_vel,
    ),
    dim=-1,
  )


def boxplus(state: LieState, delta: torch.Tensor) -> LieState:
  return compose(state, exp(delta))


def local_error(base: LieState, state: LieState) -> torch.Tensor:
  # Subtract before rotating, avoiding avoidable cancellation in group inverse.
  inverse_rotation = base.rotation.transpose(-1, -2)
  return log(
    LieState(
      inverse_rotation @ state.rotation,
      _mv(inverse_rotation, state.position - base.position),
      state.velocity - base.velocity,
      state.omega - base.omega,
      state.joint_pos - base.joint_pos,
      state.joint_vel - base.joint_vel,
    )
  )


def reset_jacobian(delta: torch.Tensor) -> torch.Tensor:
  """d Log(Exp(-delta) Exp(delta+e)) / de at zero, including cross blocks.

  Only the six pose coordinates require a chart reset.
  See Sola et al., A micro Lie theory, Eq. (149):
  https://arxiv.org/abs/1812.01537
  All branches are tensor selections; no adaptive matrix_exp or host reads.
  """
  if delta.shape[-1] != TANGENT_DIM:
    raise ValueError("G1 reset correction must have 70 components")
  theta = -delta[..., :3]
  squared = theta.square().sum(-1)[..., None, None]
  safe = squared.clamp_min(1.0)
  angle = safe.sqrt()
  sine, cosine = angle.sin(), angle.cos()

  # Horner Taylor polynomials avoid cancellation in float32 near zero and
  # have finite first/second derivatives in both torch.where branches.
  b0 = 0.5 + squared * (
    -1 / 24
    + squared
    * (1 / 720 + squared * (-1 / 40320 + squared * (1 / 3628800 - squared / 479001600)))
  )
  a0 = 1 / 6 + squared * (
    -1 / 120
    + squared
    * (
      1 / 5040
      + squared * (-1 / 362880 + squared * (1 / 39916800 - squared / 6227020800))
    )
  )
  q0 = -1 / 24 + squared * (
    1 / 720
    + squared
    * (
      -1 / 40320
      + squared * (1 / 3628800 + squared * (-1 / 479001600 + squared / 87178291200))
    )
  )
  d0 = -1 / 120 + squared * (
    1 / 5040
    + squared
    * (
      -1 / 362880
      + squared * (1 / 39916800 + squared * (-1 / 6227020800 + squared / 1307674368000))
    )
  )
  # Keep the clamp boundary in the Taylor branch so its first and second
  # derivatives do not depend on clamp_min's subgradient at equality.
  use_taylor = squared <= 1
  b = torch.where(use_taylor, b0, (1 - cosine) / safe)
  a = torch.where(use_taylor, a0, (angle - sine) / (safe * angle))
  q = torch.where(use_taylor, q0, (1 - safe / 2 - cosine) / safe.square())
  d = torch.where(
    use_taylor, d0, (angle - sine - angle * safe / 6) / (safe.square() * angle)
  )
  cross = skew(theta)
  cross2 = cross @ cross
  rotation = _eye(3, delta) + b * cross + a * cross2

  def translation(vector):
    u = skew(-vector)
    middle = cross @ u @ cross
    return (
      0.5 * u
      + a * (cross @ u + u @ cross + middle)
      - q * (cross2 @ u + u @ cross2 - 3 * middle)
      - 0.5 * (q - 3 * d) * (cross @ u @ cross2 + cross2 @ u @ cross)
    )

  zero = torch.zeros_like(cross)
  right_jacobian = torch.cat(
    (
      torch.cat((rotation, zero), -1),
      torch.cat((translation(delta[..., 3:6]), rotation), -1),
    ),
    -2,
  )
  trailing = TANGENT_DIM - 6
  upper = torch.cat(
    (right_jacobian, delta.new_zeros(*delta.shape[:-1], 6, trailing)), -1
  )
  lower = torch.cat(
    (
      delta.new_zeros(*delta.shape[:-1], trailing, 6),
      _eye(trailing, delta).expand(*delta.shape[:-1], trailing, trailing),
    ),
    -1,
  )
  return torch.cat((upper, lower), -2)
