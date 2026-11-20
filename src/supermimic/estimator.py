"""Pure, differentiable right-local filtering on SE_2(3) x R^61.

The core consumes already sampled measurements and executed normalized actions.
It owns no online posterior or random-number state. Stateful collection adapters
must explicitly commit its returned candidate belief and manage model versions.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import torch

from supermimic.belief import BeliefState
from supermimic.estimator_inputs import GaussianMeasurementConfig, PosteriorInput
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
  are made on detached matrices; the successful factorization retains its
  full gradient. Jitter is explicitly reported, never silently accumulated.
  """

  jitter_initial: float = 1e-8
  jitter_multiplier: float = 10.0
  jitter_max: float = 1e-4
  jitter_attempts: int = 5
  covariance_atol: float = 1e-10
  covariance_rtol: float = 1e-5

  def __post_init__(self) -> None:
    values = asdict(self)
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


@dataclass(frozen=True)
class NllDiagnostics:
  residual: torch.Tensor
  mahalanobis: torch.Tensor
  log_determinant: torch.Tensor
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
  config: FilterConfig,
  name: str,
  *,
  std: torch.Tensor | None = None,
  require_resolved_rank: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Factor in the requested coordinates, adding jitter in physical units."""
  _validate_covariance(matrix, config, name)
  matrix = _symmetric(matrix)
  eye = torch.eye(matrix.shape[-1], device=matrix.device, dtype=matrix.dtype)

  def coordinates(candidate):
    return _standardized(candidate, std) if std is not None else candidate

  def unresolved(candidate):
    _, info = torch.linalg.cholesky_ex(coordinates(candidate))
    failed = info != 0
    if require_resolved_rank:
      variance = candidate.diagonal(dim1=-2, dim2=-1)
      std_rank = torch.where(variance > 0, variance, 1.0).sqrt()
      rank = _effective_rank(_standardized(candidate, std_rank))
      failed = failed | (rank < candidate.shape[-1])
    return failed

  with torch.no_grad():
    jitter = matrix.new_zeros(matrix.shape[:-2])
    attempts = torch.ones(matrix.shape[:-2], device=matrix.device, dtype=torch.long)
    failed = unresolved(matrix.detach())
    for attempt in range(config.jitter_attempts):
      if not bool(failed.any()):
        break
      amount = min(
        config.jitter_initial * config.jitter_multiplier**attempt,
        config.jitter_max,
      )
      jitter = torch.where(failed, jitter.new_tensor(amount), jitter)
      attempts = attempts + failed.to(attempts.dtype)
      failed = unresolved(matrix.detach() + jitter[..., None, None] * eye)
    if bool(failed.any()):
      raise ValueError(
        f"{name} is not positive definite after bounded jitter "
        f"(maximum {config.jitter_max:g}, retries {config.jitter_attempts})"
      )
  factor = torch.linalg.cholesky(coordinates(matrix + jitter[..., None, None] * eye))
  return factor, jitter, attempts


def _innovation_gain(cross, covariance, config):
  """Solve in the numerically observable subspace of each innovation.

  A 3D noiseless gravity observation has at most two independent directions.
  Roundoff can produce a successful Cholesky even on that null direction.
  """
  _validate_covariance(covariance, config, "innovation covariance")
  size = covariance.shape[-1]
  variance = covariance.diagonal(dim1=-2, dim2=-1)
  # The three projected-gravity coordinates form one Euclidean vector. A
  # common scale preserves its radial null direction and the tangent-plane
  # projection for a noiseless sensor. Scale each scalar sensor separately.
  gravity = variance[..., :3].mean(-1, keepdim=True)
  variance = torch.cat((gravity.expand_as(variance[..., :3]), variance[..., 3:]), -1)
  std = torch.where(variance > 0, variance, torch.ones_like(variance)).sqrt()
  standardized = _standardized(covariance, std)
  rank = _effective_rank(standardized)
  full = rank.reshape(-1) == size
  matrices = covariance.reshape(-1, size, size)
  rhs = (cross / std.unsqueeze(-2)).reshape(-1, cross.shape[-2], size)
  gain = torch.zeros_like(rhs)
  jitter = matrices.new_zeros(len(matrices))
  attempts = torch.zeros_like(full, dtype=torch.long)
  if full.any():
    factor, added, count = _cholesky(
      matrices[full], config, "innovation covariance", std=std.reshape(-1, size)[full]
    )
    gain[full] = torch.cholesky_solve(rhs[full].mT, factor).mT
    jitter[full], attempts[full] = added, count
  if (~full).any():
    rtol = size * torch.finfo(covariance.dtype).eps
    observable_inverse = torch.linalg.pinv(
      standardized.reshape(-1, size, size)[~full],
      hermitian=True,
      atol=rtol,
      rtol=rtol,
    )
    gain[~full] = rhs[~full] @ observable_inverse
  return (
    gain.reshape_as(cross) / std.unsqueeze(-2),
    rank,
    jitter.reshape_as(rank),
    attempts.reshape_as(rank),
  )


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
  ) -> None:
    if default_joint_pos.shape != (29,) or not bool(
      torch.isfinite(default_joint_pos).all()
    ):
      raise ValueError("default_joint_pos must contain 29 finite joint positions")
    self.dynamics = dynamics
    self.measurement_config = measurement_config
    self.default_joint_pos = default_joint_pos.detach().clone()
    self.config = config or FilterConfig()

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

    A shared tangent vector perturbs every independent batch row. Its Jacobian
    has shape ``(*batch, 70, 70)`` without materializing batch cross-blocks.
    Dynamics must not couple batch rows (fixed empirical normalization is OK).

    Evaluation may use no_grad(), but inference_mode disables the local AD
    needed even when no outer model/action gradients are requested.
    """
    if torch.is_inference_mode_enabled():
      raise RuntimeError(
        "Transition Jacobians require local autograd; use torch.no_grad() "
        "instead of torch.inference_mode(), or use BeliefProvider"
      )
    if nominal is None:
      aux, rep = mean.to_rep()
      next_aux, next_rep = self.dynamics(rep, action, aux, context)
      nominal = LieState.from_rep(next_aux, next_rep)

    def residual(epsilon: torch.Tensor) -> torch.Tensor:
      perturbed = boxplus(mean, epsilon)
      aux, rep = perturbed.to_rep()
      next_aux, next_rep = self.dynamics(rep, action, aux, context)
      return local_error(nominal, LieState.from_rep(next_aux, next_rep))

    zero = mean.position.new_zeros(TANGENT_DIM)
    # GenCoord currently calls TorchScript quaternion primitives. Their nested
    # functorch backward can either fail under vmap or silently give incorrect
    # mixed parameter derivatives, including with jacrev(chunk_size=1). Keep
    # this eager reference on ordinary autograd until those primitives have a
    # separately verified transform-safe implementation. create_graph is
    # essential for model learning and action gradients through covariance.
    return torch.autograd.functional.jacobian(
      residual, zero, create_graph=torch.is_grad_enabled(), vectorize=False
    )

  def predict(
    self,
    belief: BeliefState,
    action: torch.Tensor,
    context: torch.Tensor,
    *,
    return_diagnostics: bool = False,
  ) -> BeliefState | tuple[BeliefState, PredictionDiagnostics]:
    self._validate_belief(belief)
    batch_shape = belief.mean.position.shape[:-1]
    if action.shape != (*batch_shape, 29):
      raise ValueError("Action must have shape (*belief_batch, 29)")
    if context.shape[:-1] != batch_shape:
      raise ValueError("Context batch shape must match the belief")
    if not bool(torch.isfinite(action).all() and torch.isfinite(context).all()):
      raise ValueError("Action and context must be finite")
    aux, rep = belief.mean.to_rep()
    next_aux, next_rep, factor = self.dynamics.distribution(rep, action, aux, context)
    nominal = LieState.from_rep(next_aux, next_rep)
    jacobian = self.transition_jacobian(belief.mean, action, context, nominal)
    process = factor @ factor.transpose(-1, -2)
    covariance = _symmetric(
      jacobian @ belief.covariance @ jacobian.transpose(-1, -2) + process
    )
    result = BeliefState(nominal, covariance)
    self._validate_belief(result)
    if return_diagnostics:
      return result, PredictionDiagnostics(jacobian, process)
    return result

  def correct(
    self,
    belief: BeliefState,
    measurement: torch.Tensor,
    *,
    return_diagnostics: bool = False,
  ) -> BeliefState | tuple[BeliefState, CorrectionDiagnostics]:
    self._validate_belief(belief)
    predicted = self.observation(belief.mean)
    if measurement.shape != predicted.shape:
      raise ValueError("Measurement must have shape (*belief_batch, 64)")
    if measurement.device != predicted.device or measurement.dtype != predicted.dtype:
      raise ValueError("Measurement and belief must share dtype and device")
    if not bool(torch.isfinite(measurement).all()):
      raise ValueError("Measurement contains non-finite values")
    h = self.observation_jacobian(belief.mean)
    covariance = _symmetric(belief.covariance)
    v = self.measurement_config.covariance(
      29, device=measurement.device, dtype=measurement.dtype
    )
    innovation = measurement - predicted
    cross = covariance @ h.transpose(-1, -2)
    innovation_covariance = _symmetric(h @ cross + v)
    gain, rank, jitter, attempts = _innovation_gain(
      cross, innovation_covariance, self.config
    )
    correction = (gain @ innovation.unsqueeze(-1)).squeeze(-1)
    mean = boxplus(belief.mean, correction)
    identity = torch.eye(TANGENT_DIM, device=covariance.device, dtype=covariance.dtype)
    residual_map = identity - gain @ h
    measurement_identity = torch.eye(64, device=v.device, dtype=v.dtype)
    # Jitter is effective measurement variance in BOTH gain and Joseph update.
    effective_v = v + jitter[..., None, None] * measurement_identity
    joseph = residual_map @ covariance @ residual_map.transpose(
      -1, -2
    ) + gain @ effective_v @ gain.transpose(-1, -2)
    # H observes omega, q and qdot directly. When an exact scalar constraint
    # is resolved, Joseph's squared residual leaves at most roundoff variance.
    # Keep any variance remaining along a mode omitted by the rank cutoff.
    exact = effective_v.diagonal(dim1=-2, dim2=-1)[..., 3:] == 0
    prior_variance = covariance.diagonal(dim1=-2, dim2=-1)[..., 9:]
    remaining = joseph.diagonal(dim1=-2, dim2=-1)[..., 9:].abs()
    roundoff = h.shape[-2] * torch.finfo(covariance.dtype).eps ** 2 * prior_variance
    resolved = exact & (remaining <= roundoff)
    keep = torch.cat(
      (torch.ones_like(correction[..., :9], dtype=torch.bool), ~resolved), -1
    )
    joseph = joseph * keep.unsqueeze(-2) * keep.unsqueeze(-1)
    reset = reset_jacobian(correction)
    result = BeliefState(mean, _symmetric(reset @ joseph @ reset.transpose(-1, -2)))
    self._validate_belief(result)
    if return_diagnostics:
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
    return result

  @staticmethod
  def to_actor_input(belief: BeliefState) -> PosteriorInput:
    return belief.to_actor_input()


def posterior_nll(
  belief: BeliefState,
  truth: LieState,
  config: FilterConfig | None = None,
  *,
  return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, NllDiagnostics]:
  """Full-dimensional NLL, using bounded jitter on numerically singular rows.

  Rank is assessed in marginal standard-deviation units. Absolute variance
  does not distinguish a precise observation from a deterministic constraint.
  """
  config = config or FilterConfig()
  _validate_versions(belief)
  belief.mean.validate()
  truth.validate()
  if truth.position.shape != belief.mean.position.shape:
    raise ValueError("NLL truth batch must match the belief")
  if (
    truth.position.dtype != belief.mean.position.dtype
    or truth.position.device != belief.mean.position.device
  ):
    raise ValueError("NLL truth and belief must share dtype and device")
  if belief.covariance.shape != (*truth.position.shape[:-1], TANGENT_DIM, TANGENT_DIM):
    raise ValueError("NLL covariance batch must match the belief")
  residual = local_error(belief.mean, truth)
  factor, jitter, attempts = _cholesky(
    belief.covariance, config, "NLL covariance", require_resolved_rank=True
  )
  solution = torch.cholesky_solve(residual.unsqueeze(-1), factor).squeeze(-1)
  mahalanobis = (residual * solution).sum(-1)
  log_determinant = 2 * factor.diagonal(dim1=-2, dim2=-1).log().sum(-1)
  result = 0.5 * (mahalanobis + log_determinant)
  if return_diagnostics:
    return result, NllDiagnostics(
      residual, mahalanobis, log_determinant, jitter, attempts
    )
  return result
