"""Pure, differentiable right-local filtering on SE_2(3) x R^61.

The core consumes already sampled measurements and executed normalized actions.
It owns no online posterior or random-number state. Stateful collection adapters
must explicitly commit its returned candidate belief and manage model versions.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import TYPE_CHECKING

import torch
from torch.func import jacrev, vmap

from supermimic.belief import BeliefState
from supermimic.estimator_inputs import GaussianMeasurementConfig, PosteriorInput
from supermimic.utils.compile import compile_eit_kernel
from supermimic.utils.lie_state import (
  STATE_LAYOUT_VERSION,
  TANGENT_DIM,
  TANGENT_VERSION,
  LieState,
  boxplus,
  local_error,
  reset_jacobian,
  skew,
)
from supermimic.utils.validation import check_covariance

if TYPE_CHECKING:
  from supermimic.models import Dynamics


@dataclass(frozen=True)
class FilterConfig:
  """Bounded absolute covariance jitter, in the corresponding sensor units.

  First try the unmodified covariance, then at most ``jitter_attempts``
  geometrically increasing values, never exceeding ``jitter_max``. Decisions
  use integer factorization status codes. The successful factorization retains
  its full gradient. Jitter is explicitly reported, never silently accumulated.
  """

  jitter_initial: float = 1e-8
  jitter_multiplier: float = 10.0
  jitter_max: float = 1e-4
  jitter_attempts: int = 5
  covariance_atol: float = 1e-10
  covariance_rtol: float = 1e-5
  jacobian_backend: str = "vmap"
  jacobian_batch_chunk_size: int | None = None
  jacobian_chunk_size: int | None = None

  def __post_init__(self) -> None:
    values = {
      key: value
      for key, value in asdict(self).items()
      if not key.startswith("jacobian_")
    }
    if any(not math.isfinite(value) for value in values.values()):
      raise ValueError("Filter numerical settings must be finite")
    if self.jitter_initial <= 0 or self.jitter_max < self.jitter_initial:
      raise ValueError("Require 0 < jitter_initial <= jitter_max")
    if self.jitter_multiplier <= 1:
      raise ValueError("jitter_multiplier must exceed one")
    if not isinstance(self.jitter_attempts, int) or self.jitter_attempts < 0:
      raise ValueError("jitter_attempts must be a nonnegative integer")
    if self.covariance_atol < 0 or self.covariance_rtol < 0:
      raise ValueError("Covariance validation tolerances must be nonnegative")
    if self.jacobian_backend not in ("vmap", "eager"):
      raise ValueError("jacobian_backend must be 'vmap' or 'eager'")
    for chunk in (self.jacobian_batch_chunk_size, self.jacobian_chunk_size):
      if chunk is not None and (type(chunk) is not int or chunk < 1):
        raise ValueError("Jacobian chunk sizes must be positive integers or None")


@dataclass(frozen=True)
class PredictionDiagnostics:
  transition_jacobian: torch.Tensor
  process_covariance: torch.Tensor


@dataclass(frozen=True)
class CorrectionDiagnostics:
  effective_rank: torch.Tensor
  innovation: torch.Tensor
  measurement_jacobian: torch.Tensor
  innovation_covariance: torch.Tensor
  kalman_gain: torch.Tensor
  correction: torch.Tensor
  reset_jacobian: torch.Tensor
  jitter: torch.Tensor
  cholesky_attempts: torch.Tensor


def _symmetric(matrix: torch.Tensor) -> torch.Tensor:
  return (matrix + matrix.transpose(-1, -2)) * 0.5


def _standardized(matrix: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
  return _symmetric(matrix / std.unsqueeze(-1) / std.unsqueeze(-2))


def _effective_rank(standardized: torch.Tensor) -> torch.Tensor:
  """Rank in dimensionless coordinates, with one unit of reference variance."""
  eigenvalues = torch.linalg.eigvalsh(standardized.detach())
  rtol = standardized.shape[-1] * torch.finfo(standardized.dtype).eps
  return (eigenvalues > rtol * eigenvalues[..., -1:].clamp_min(1)).sum(-1)


@torch.no_grad()
def _certifies_full_rank(covariance, standardized, config) -> bool:
  """Bound the spectrum of the symmetric innovation in the rank solver's units."""
  if (
    config.covariance_atol != FilterConfig.covariance_atol
    or config.covariance_rtol != FilterConfig.covariance_rtol
  ):
    return False
  matrix = standardized.double()
  diagonal = matrix.diagonal(dim1=-2, dim2=-1)
  radius = matrix.abs().sum(-1) - diagonal.abs()
  lower = (diagonal - radius).amin(-1)
  upper = (diagonal + radius).amax(-1)
  # A fourfold margin keeps borderline eigenvalues on the spectral path.
  margin = 4 * matrix.shape[-1] * torch.finfo(standardized.dtype).eps
  valid = (covariance.diagonal(dim1=-2, dim2=-1) > 0).all()
  # An infinite or NaN entry also makes its row fail these finite bounds.
  return bool(valid & (lower > margin * upper.clamp_min(1)).all())


def _validate_versions(belief: BeliefState) -> None:
  if belief.state_layout_version != STATE_LAYOUT_VERSION:
    raise ValueError("Unsupported belief state layout version")
  if belief.tangent_version != TANGENT_VERSION:
    raise ValueError("Unsupported belief tangent version")


def _validate_covariance(matrix: torch.Tensor, config: FilterConfig, name: str) -> None:
  """Check PSD independently of jitter; do not repair invalid distributions."""
  check_covariance(
    matrix, name, rtol=config.covariance_rtol, atol=config.covariance_atol
  )


def _cholesky(
  matrix: torch.Tensor,
  std: torch.Tensor,
  config: FilterConfig,
  standardized: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Factor rows of the already validated innovation, retaining the successful graph."""
  # Collection reuses the values already inspected by the rank check. In AD
  # keep the original factorization graph and its gradient accumulation order.
  if standardized is None or matrix.requires_grad:
    matrix = _symmetric(matrix)
    standardized = _standardized(matrix, std)
  jitter = matrix.new_zeros(matrix.shape[:-2])
  attempts = torch.ones_like(jitter, dtype=torch.long)
  factor, info = torch.linalg.cholesky_ex(standardized)
  for retry in range(config.jitter_attempts):
    failed = info != 0
    if not failed.any():
      return factor, jitter, attempts
    amount = min(
      config.jitter_initial * config.jitter_multiplier**retry, config.jitter_max
    )
    jitter = torch.where(failed, jitter.new_tensor(amount), jitter)
    attempts = attempts + failed.to(attempts.dtype)
    eye = torch.eye(matrix.shape[-1], device=matrix.device, dtype=matrix.dtype)
    candidate = matrix + jitter[..., None, None] * eye
    factor, info = torch.linalg.cholesky_ex(_standardized(candidate, std))
  if (info != 0).any():
    raise ValueError(
      "innovation covariance is not positive definite after bounded jitter "
      f"(maximum {config.jitter_max:g}, retries {config.jitter_attempts})"
    )
  return factor, jitter, attempts


def _innovation_gain(cross, covariance, config):
  """Solve the symmetric innovation in its numerically observable subspace."""
  size = covariance.shape[-1]
  variance = covariance.diagonal(dim1=-2, dim2=-1)
  # A common gravity scale preserves its radial and tangent directions.
  gravity = variance[..., :3].mean(-1, keepdim=True)
  variance = torch.cat((gravity.expand_as(variance[..., :3]), variance[..., 3:]), -1)
  std = torch.where(variance > 0, variance, torch.ones_like(variance)).sqrt()
  standardized = _standardized(covariance, std)
  if _certifies_full_rank(covariance, standardized, config):
    rank = std.new_full(std.shape[:-1], size, dtype=torch.int64)
    all_full = True
  else:
    _validate_covariance(covariance, config, "innovation covariance")
    rank = _effective_rank(standardized)
    all_full = bool((rank == size).all())
  matrices = covariance.reshape(-1, size, size)
  rhs = (cross / std.unsqueeze(-2)).reshape(-1, cross.shape[-2], size)
  scales = std.reshape(-1, size)
  normalized = standardized.reshape_as(matrices)
  if all_full:
    factor, jitter, attempts = _cholesky(matrices, scales, config, normalized)
    gain = torch.cholesky_solve(rhs.mT, factor).mT
  else:
    # Mixed ranks require independent solves for the observable subspaces.
    full = rank.reshape(-1) == size
    gain = torch.zeros_like(rhs)
    jitter = matrices.new_zeros(len(matrices))
    attempts = torch.zeros_like(full, dtype=torch.long)
    if full.any():
      factor, added, count = _cholesky(
        matrices[full], scales[full], config, normalized[full]
      )
      gain[full] = torch.cholesky_solve(rhs[full].mT, factor).mT
      jitter[full], attempts[full] = added, count
    rtol = size * torch.finfo(covariance.dtype).eps
    observable_inverse = torch.linalg.pinv(
      normalized[~full], hermitian=True, atol=rtol, rtol=rtol
    )
    gain[~full] = rhs[~full] @ observable_inverse
  return (
    gain.reshape_as(cross) / std.unsqueeze(-2),
    rank,
    jitter.reshape_as(rank),
    attempts.reshape_as(rank),
  )


def _padded_rows(part: torch.Tensor, width: int) -> torch.Tensor:
  # Each Jacobian row is independent. Slicing the result drops the repeated
  # rows from both the loss and its outer gradient.
  missing = width - len(part)
  if missing:
    part = torch.cat((part, part[-1:].expand(missing, *part.shape[1:])))
  if part.storage_offset():
    return part.clone(memory_format=torch.contiguous_format)
  return part.contiguous()


def _propagate_covariance(jacobian, covariance, factor):
  process = factor @ factor.mT
  return _symmetric(jacobian @ covariance @ jacobian.mT + process), process


class LieBeliefFilter:
  """First-order Gaussian belief filter with differentiable model Jacobians.

  Normalization statistics and dynamics parameters are read-only during each
  call. The caller controls whether outer model/action derivatives are needed;
  local propagation differentiation is required even during real collection.
  """

  def __init__(
    self,
    dynamics: Dynamics,
    measurement_config: GaussianMeasurementConfig,
    default_joint_pos: torch.Tensor,
    config: FilterConfig | None = None,
    *,
    compile_kernels: bool = False,
  ) -> None:
    if not callable(getattr(dynamics, "mean_lie", None)) or not callable(
      getattr(dynamics, "distribution_lie", None)
    ):
      raise TypeError("Belief dynamics must implement mean_lie and distribution_lie")
    if default_joint_pos.shape != (29,) or not bool(
      torch.isfinite(default_joint_pos).all()
    ):
      raise ValueError("default_joint_pos must contain 29 finite joint positions")
    self.dynamics = dynamics
    self.measurement_config = measurement_config
    self.default_joint_pos = default_joint_pos.detach().clone()
    self.config = config or FilterConfig()
    self._compiled_jacobian = None
    self._propagate = _propagate_covariance
    if compile_kernels:
      self._compiled_jacobian = compile_eit_kernel(self._vector_jacobian, dynamic=False)
      self._propagate = compile_eit_kernel(_propagate_covariance, dynamic=True)

  def _validate_belief(self, belief: BeliefState) -> None:
    _validate_versions(belief)
    belief.mean.validate()
    expected = (*belief.mean.position.shape[:-1], TANGENT_DIM, TANGENT_DIM)
    if belief.covariance.shape != expected:
      raise ValueError(f"Belief covariance must have shape {expected}")
    if (
      belief.covariance.dtype != belief.mean.position.dtype
      or belief.covariance.device != belief.mean.position.device
    ):
      raise ValueError("Belief mean and covariance must share dtype and device")
    _validate_covariance(belief.covariance, self.config, "belief covariance")

  def initialize(
    self,
    mean: LieState,
    covariance: torch.Tensor,
    context: torch.Tensor | None = None,
  ) -> BeliefState:
    """Create a prior. The caller must fuse the first measurement exactly once."""
    if context is not None and not bool(torch.isfinite(context).all()):
      raise ValueError("Context contains non-finite values")
    belief = BeliefState(mean, covariance)
    self._validate_belief(belief)
    return belief

  def observation(self, mean: LieState) -> torch.Tensor:
    # R.T @ (0, 0, -1) is the negative last row of R.
    gravity = -mean.rotation[..., 2, :]
    return torch.cat(
      (
        gravity,
        mean.omega,
        mean.joint_pos - self.default_joint_pos.to(mean.joint_pos),
        mean.joint_vel,
      ),
      dim=-1,
    )

  def observation_jacobian(self, mean: LieState) -> torch.Tensor:
    gravity = -mean.rotation[..., 2, :]
    identity = torch.eye(61, device=gravity.device, dtype=gravity.dtype)
    identity = identity.expand(*gravity.shape[:-1], 61, 61)
    top = torch.cat((skew(gravity), gravity.new_zeros(*gravity.shape[:-1], 3, 67)), -1)
    bottom = torch.cat((gravity.new_zeros(*gravity.shape[:-1], 61, 9), identity), -1)
    return torch.cat((top, bottom), -2)

  def transition_jacobian(
    self,
    mean: LieState,
    action: torch.Tensor,
    context: torch.Tensor,
    nominal: LieState | None = None,
  ) -> torch.Tensor:
    """Derivative in epsilon only, retaining all outer parameter dependencies.

    The vectorized backend differentiates each independent sample then batches
    the result, without materializing cross-batch blocks. Normalization is fixed.

    Evaluation may use no_grad(), but inference_mode disables the local AD
    needed even when no outer model/action gradients are requested.
    """
    if torch.is_inference_mode_enabled():
      raise RuntimeError(
        "Transition Jacobians require local autograd; use torch.no_grad() "
        "instead of torch.inference_mode(), or use BeliefProvider"
      )
    if nominal is None:
      nominal = self.mean_step(mean, action, context)

    def residual(epsilon: torch.Tensor) -> torch.Tensor:
      perturbed = boxplus(mean, epsilon)
      return local_error(nominal, self.mean_step(perturbed, action, context))

    zero = mean.position.new_zeros(TANGENT_DIM)
    if self.config.jacobian_backend == "eager":
      return torch.autograd.functional.jacobian(
        residual, zero, create_graph=torch.is_grad_enabled(), vectorize=False
      )

    # Flatten arbitrary leading batch axes. The nominal is constant only for
    # epsilon differentiation; passing its tensors preserves outer gradients.
    def flatten(state):
      return tuple(
        getattr(state, field.name).reshape(
          -1,
          *(
            (3, 3)
            if field.name == "rotation"
            else (getattr(state, field.name).shape[-1],)
          ),
        )
        for field in fields(state)
      )

    batch_shape = mean.position.shape[:-1]
    state_values, nominal_values = flatten(mean), flatten(nominal)
    count = state_values[0].shape[0]
    args = (
      zero.expand(count, -1),
      state_values,
      nominal_values,
      action.reshape(count, action.shape[-1]),
      context.reshape(count, context.shape[-1]),
    )
    if self._compiled_jacobian is None or count == 0:
      result = self._vector_jacobian(
        *args, chunk_size=self.config.jacobian_batch_chunk_size
      )
    else:
      result = self._compiled_jacobian_blocks(*args)
    return result.reshape(*batch_shape, TANGENT_DIM, TANGENT_DIM)

  def _vector_jacobian(
    self, epsilon, state, nominal, action, context, *, chunk_size=None
  ):
    derivatives = jacrev(
      self._local_transition, argnums=0, chunk_size=self.config.jacobian_chunk_size
    )
    return vmap(derivatives, chunk_size=chunk_size)(
      epsilon, state, nominal, action, context
    )

  def _compiled_jacobian_blocks(self, epsilon, state, nominal, action, context):
    # vmap specializes its batch dimension. Real episode resets and short
    # imagined clips otherwise compile one backward graph per remainder size.
    block = self.config.jacobian_batch_chunk_size or 256
    assert self._compiled_jacobian is not None
    small = min(block, 16)
    count = len(action)
    blocks = []
    start = 0
    while start < count:
      width = block if count - start >= block else small
      stop = min(start + width, count)
      length = stop - start

      result = self._compiled_jacobian(
        epsilon[:1].expand(width, -1),
        tuple(_padded_rows(x[start:stop], width) for x in state),
        tuple(_padded_rows(x[start:stop], width) for x in nominal),
        _padded_rows(action[start:stop], width),
        _padded_rows(context[start:stop], width),
      )
      blocks.append(result[:length])
      start = stop
    return blocks[0] if len(blocks) == 1 else torch.cat(blocks)

  def _local_transition(self, epsilon, state_values, nominal_values, action, context):
    state = LieState(*state_values)
    nominal = LieState(*nominal_values)
    return local_error(
      nominal, self.mean_step(boxplus(state, epsilon), action, context)
    )

  def mean_step(
    self, mean: LieState, action: torch.Tensor, context: torch.Tensor
  ) -> LieState:
    return self.dynamics.mean_lie(mean, action, context)

  def predict(
    self,
    belief: BeliefState,
    action: torch.Tensor,
    context: torch.Tensor,
    *,
    return_diagnostics: bool = False,
  ) -> BeliefState | tuple[BeliefState, PredictionDiagnostics]:
    result, jacobian, process = self._prediction(belief, action, context)
    self._validate_belief(result)
    if return_diagnostics:
      return result, PredictionDiagnostics(jacobian, process)
    return result

  def predict_correct(
    self,
    belief: BeliefState,
    action: torch.Tensor,
    context: torch.Tensor,
    measurement: torch.Tensor,
    *,
    return_diagnostics: bool = False,
  ) -> BeliefState | tuple[BeliefState, CorrectionDiagnostics]:
    """Advance a live frame; correct validates this fresh prediction exactly once."""
    predicted = self._prediction(belief, action, context)[0]
    return self.correct(predicted, measurement, return_diagnostics=return_diagnostics)

  def _prediction(
    self,
    belief: BeliefState,
    action: torch.Tensor,
    context: torch.Tensor,
    *,
    belief_checked: bool = False,
    inputs_checked: bool = False,
  ) -> tuple[BeliefState, torch.Tensor, torch.Tensor]:
    if not belief_checked:
      self._validate_belief(belief)
    batch_shape = belief.mean.position.shape[:-1]
    if action.shape != (*batch_shape, 29):
      raise ValueError("Action must have shape (*belief_batch, 29)")
    if context.shape[:-1] != batch_shape:
      raise ValueError("Context batch shape must match the belief")
    if not inputs_checked and not bool(
      torch.isfinite(action).all() & torch.isfinite(context).all()
    ):
      raise ValueError("Action and context must be finite")
    nominal, factor = self.dynamics.distribution_lie(belief.mean, action, context)
    jacobian = self.transition_jacobian(belief.mean, action, context, nominal)
    covariance, process = self._propagate(jacobian, belief.covariance, factor)
    return BeliefState(nominal, covariance), jacobian, process

  def correct(
    self,
    belief: BeliefState,
    measurement: torch.Tensor,
    *,
    return_diagnostics: bool = False,
  ) -> BeliefState | tuple[BeliefState, CorrectionDiagnostics]:
    result, diagnostics = self._correct(
      belief, measurement, return_diagnostics=return_diagnostics
    )
    return (result, diagnostics) if diagnostics is not None else result

  def _correct(
    self,
    belief: BeliefState,
    measurement: torch.Tensor,
    *,
    return_diagnostics: bool,
    measurement_checked: bool = False,
  ) -> tuple[BeliefState, CorrectionDiagnostics | None]:
    self._validate_belief(belief)
    predicted = self.observation(belief.mean)
    if measurement.shape != predicted.shape:
      raise ValueError("Measurement must have shape (*belief_batch, 64)")
    if measurement.device != predicted.device or measurement.dtype != predicted.dtype:
      raise ValueError("Measurement and belief must share dtype and device")
    if not measurement_checked and not bool(torch.isfinite(measurement).all()):
      raise ValueError("Measurement contains non-finite values")
    h = self.observation_jacobian(belief.mean)
    covariance = _symmetric(belief.covariance)
    _, variance = self.measurement_config._tensors(
      29, device=measurement.device, dtype=measurement.dtype
    )
    innovation = measurement - predicted
    cross = covariance @ h.transpose(-1, -2)
    innovation_covariance = h @ cross
    innovation_covariance.diagonal(dim1=-2, dim2=-1).add_(variance)
    innovation_covariance = _symmetric(innovation_covariance)
    gain, rank, jitter, attempts = _innovation_gain(
      cross, innovation_covariance, self.config
    )
    correction = (gain @ innovation.unsqueeze(-1)).squeeze(-1)
    mean = boxplus(belief.mean, correction)
    identity = torch.eye(TANGENT_DIM, device=covariance.device, dtype=covariance.dtype)
    residual_map = identity - gain @ h
    # Jitter is effective measurement variance in BOTH gain and Joseph update.
    effective_v = (variance + jitter[..., None]).diag_embed()
    joseph = residual_map @ covariance @ residual_map.transpose(
      -1, -2
    ) + gain @ effective_v @ gain.transpose(-1, -2)
    reset = reset_jacobian(correction)
    result = BeliefState(mean, _symmetric(reset @ joseph @ reset.transpose(-1, -2)))
    self._validate_belief(result)
    if return_diagnostics:
      measurement_identity = torch.eye(64, device=variance.device, dtype=variance.dtype)
      return result, CorrectionDiagnostics(
        rank,
        innovation,
        h,
        innovation_covariance + jitter[..., None, None] * measurement_identity,
        gain,
        correction,
        reset,
        jitter,
        attempts,
      )
    return result, None

  @staticmethod
  def to_actor_input(belief: BeliefState) -> PosteriorInput:
    return belief.to_actor_input()
