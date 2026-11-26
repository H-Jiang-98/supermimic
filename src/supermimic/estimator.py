"""Pure, differentiable right-local filtering on SE(3) x R^64.

The core consumes already sampled measurements and executed normalized actions.
It owns no online posterior or random-number state. Stateful collection adapters
must explicitly commit its returned candidate belief and manage model versions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Literal

import torch
from torch.func import jacrev, vmap

from supermimic.belief import BeliefState
from supermimic.estimator_inputs import GaussianMeasurementConfig, PosteriorInput
from supermimic.utils.compile import compile_eit_kernel
from supermimic.utils.lie_state import (
  TANGENT_DIM,
  LieState,
  boxplus,
  local_error,
  reset_jacobian,
  skew,
)
from supermimic.utils.square_root import covariance_root
from supermimic.utils.validation import check_covariance, rotation_valid

if TYPE_CHECKING:
  from supermimic.models import Dynamics


@dataclass(frozen=True)
class FilterConfig:
  """Square-root filtering; numerical checks are opt-in outside tensor kernels."""

  validate_numerics: bool = False
  covariance_atol: float = 1e-10
  covariance_rtol: float = 1e-5
  jacobian_backend: str = "vmap"
  jacobian_batch_chunk_size: int | None = None
  jacobian_chunk_size: int | None = None

  def __post_init__(self) -> None:
    if type(self.validate_numerics) is not bool:
      raise ValueError("validate_numerics must be boolean")
    if not all(math.isfinite(v) for v in (self.covariance_atol, self.covariance_rtol)):
      raise ValueError("Filter numerical settings must be finite")
    if self.covariance_atol < 0 or self.covariance_rtol < 0:
      raise ValueError("Covariance validation tolerances must be nonnegative")
    if self.jacobian_backend not in ("vmap", "eager", "increment"):
      raise ValueError("jacobian_backend must be 'vmap', 'eager' or 'increment'")
    for chunk in (self.jacobian_batch_chunk_size, self.jacobian_chunk_size):
      if chunk is not None and (type(chunk) is not int or chunk < 1):
        raise ValueError("Jacobian chunk sizes must be positive integers or None")

  @classmethod
  def from_dict(cls, values):
    """Read current settings or the old covariance-filter weight metadata."""
    if not isinstance(values, dict):
      raise ValueError("Incomplete estimator checkpoint filter_config")
    legacy = {"jitter_initial", "jitter_multiplier", "jitter_max", "jitter_attempts"}
    current = {f.name for f in fields(cls)}
    if values.keys() == (current - {"validate_numerics"}) | legacy:
      values = {k: v for k, v in values.items() if k not in legacy}
      values["validate_numerics"] = False
    if values.keys() != current:
      raise ValueError("Incomplete estimator checkpoint filter_config")
    return cls(**values)


@dataclass(frozen=True)
class PredictionDiagnostics:
  transition_jacobian: torch.Tensor
  process_covariance: torch.Tensor


@dataclass(frozen=True)
class CorrectionDiagnostics:
  innovation: torch.Tensor
  measurement_jacobian: torch.Tensor
  innovation_covariance: torch.Tensor
  kalman_gain: torch.Tensor
  correction: torch.Tensor
  reset_jacobian: torch.Tensor


@dataclass(frozen=True)
class CorrectionSummary:
  innovation_norm: torch.Tensor


def _padded_rows(part: torch.Tensor, width: int) -> torch.Tensor:
  # Each Jacobian row is independent. Slicing the result drops the repeated
  # rows from both the loss and its outer gradient.
  missing = width - len(part)
  if missing:
    part = torch.cat((part, part[-1:].expand(missing, *part.shape[1:])))
  if part.storage_offset():
    return part.clone(memory_format=torch.contiguous_format)
  return part.contiguous()


def _propagate_sqrt(jacobian, lower, process_factor):
  return covariance_root(torch.cat((jacobian @ lower, process_factor), -1))


def _correction_sqrt(lower, h, std, innovation):
  projected = h @ lower
  noise = std.diag_embed().expand(*projected.shape[:-2], std.shape[-1], std.shape[-1])
  innovation_factor = covariance_root(torch.cat((projected, noise), -1))
  cross = lower @ projected.mT
  whitened = torch.linalg.solve_triangular(innovation_factor, cross.mT, upper=False)
  gain = torch.linalg.solve_triangular(innovation_factor.mT, whitened, upper=True).mT
  correction = (gain @ innovation.unsqueeze(-1)).squeeze(-1)
  reset = reset_jacobian(correction)
  posterior = covariance_root(
    torch.cat(
      (
        reset @ (lower - gain @ projected),
        (reset @ gain) * std.unsqueeze(-2),
      ),
      -1,
    )
  )
  return posterior, correction, innovation_factor, gain, reset


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
    if self.config.jacobian_backend == "increment" and not all(
      callable(getattr(dynamics, name, None))
      for name in ("prediction_lie", "integrate_prediction_lie")
    ):
      raise TypeError(
        "Increment Jacobians require prediction_lie and integrate_prediction_lie"
      )
    self._compiled_jacobian = None
    self._propagate = _propagate_sqrt
    self._correct_factor = _correction_sqrt
    if compile_kernels:
      self._compiled_jacobian = compile_eit_kernel(self._vector_jacobian, dynamic=False)
      self._propagate = compile_eit_kernel(_propagate_sqrt, dynamic=True)
      self._correct_factor = compile_eit_kernel(_correction_sqrt, dynamic=True)

  def _validate_belief(self, belief: BeliefState) -> None:
    belief.validate_structure()
    if self.config.validate_numerics:
      belief.validate()

  def initialize(
    self,
    mean: LieState,
    covariance: torch.Tensor,
    context: torch.Tensor | None = None,
  ) -> BeliefState:
    """Create a prior. The caller must fuse the first measurement exactly once."""
    if context is not None and not bool(torch.isfinite(context).all()):
      raise ValueError("Context contains non-finite values")
    mean.validate()
    check_covariance(
      covariance,
      "belief covariance",
      atol=self.config.covariance_atol,
      rtol=self.config.covariance_rtol,
    )
    lower = torch.linalg.cholesky(0.5 * (covariance + covariance.mT))
    belief = BeliefState(mean, covariance_sqrt=lower)
    belief.validate()
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
    The increment backend differentiates the 35 network outputs and composes
    their derivatives with the integration map to recover the full 70x70 map.

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
    if count == 0:
      return mean.position.new_empty(*batch_shape, TANGENT_DIM, TANGENT_DIM)
    args = (
      zero.expand(count, -1),
      state_values,
      nominal_values,
      action.reshape(count, action.shape[-1]),
      context.reshape(count, context.shape[-1]),
    )
    if self._compiled_jacobian is None:
      result = self._vector_jacobian(
        *args, chunk_size=self.config.jacobian_batch_chunk_size
      )
    else:
      result = self._compiled_jacobian_blocks(*args)
    return result.reshape(*batch_shape, TANGENT_DIM, TANGENT_DIM)

  def _vector_jacobian(
    self, epsilon, state, nominal, action, context, *, chunk_size=None
  ):
    if self.config.jacobian_backend == "increment":
      return vmap(self._increment_jacobian, chunk_size=chunk_size)(
        epsilon, state, nominal, action, context
      )
    derivatives = jacrev(
      self._local_transition, argnums=0, chunk_size=self.config.jacobian_chunk_size
    )
    return vmap(derivatives, chunk_size=chunk_size)(
      epsilon, state, nominal, action, context
    )

  def _local_prediction(self, epsilon, state_values, action, context):
    prediction = self.dynamics.prediction_lie(
      boxplus(LieState(*state_values), epsilon), action, context
    )
    # has_aux reuses the forward value without detaching the outer graph.
    return prediction, prediction

  def _local_integration(self, epsilon, prediction, state_values, nominal_values):
    return local_error(
      LieState(*nominal_values),
      self.dynamics.integrate_prediction_lie(
        boxplus(LieState(*state_values), epsilon), prediction
      ),
    )

  def _increment_jacobian(self, epsilon, state, nominal, action, context):
    # g(e) is the network's 35-vector and h(e, d) is local_error after the
    # integrator. F = h_e + h_d @ g_e. Only g_e traverses the world model;
    # the 70-output derivative below contains geometry and integration only.
    # Differentiate the actual integrator, including polar correction and the
    # supplied nominal chart, so outer action/state/model gradients stay exact.
    chunk = self.config.jacobian_chunk_size
    increment_jacobian, prediction = jacrev(
      self._local_prediction, argnums=0, has_aux=True, chunk_size=chunk
    )(epsilon, state, action, context)
    direct, through_increment = jacrev(
      self._local_integration, argnums=(0, 1), chunk_size=chunk
    )(epsilon, prediction, state, nominal)
    return direct + through_increment @ increment_jacobian

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
      # Reuse the full specialization for a nearly full tail after partial resets.
      width = block if count - start > max(small, block // 2) else small
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
    result, jacobian, process_factor = self._prediction(belief, action, context)
    self._validate_belief(result)
    if return_diagnostics:
      return result, PredictionDiagnostics(jacobian, process_factor @ process_factor.mT)
    return result

  def predict_correct(
    self,
    belief: BeliefState,
    action: torch.Tensor,
    context: torch.Tensor,
    measurement: torch.Tensor,
    *,
    return_diagnostics: bool | Literal["summary"] = False,
  ) -> BeliefState | tuple[BeliefState, CorrectionDiagnostics | CorrectionSummary]:
    """Advance one frame, retaining the factor and its complete outer graph."""
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
    if (
      self.config.validate_numerics
      and not inputs_checked
      and not bool(torch.isfinite(action).all() & torch.isfinite(context).all())
    ):
      raise ValueError("Action and context must be finite")
    return self._prediction_kernel(belief.mean, belief.covariance_sqrt, action, context)

  def _prediction_kernel(self, mean, covariance_sqrt, action, context):
    nominal, factor = self.dynamics.distribution_lie(mean, action, context)
    jacobian = self.transition_jacobian(mean, action, context, nominal)
    lower = self._propagate(jacobian, covariance_sqrt, factor)
    return BeliefState(nominal, covariance_sqrt=lower), jacobian, factor

  def predict_correct_kernel(
    self,
    mean: LieState,
    covariance_sqrt: torch.Tensor,
    action: torch.Tensor,
    context: torch.Tensor,
    measurement: torch.Tensor,
    *,
    measurement_std: torch.Tensor,
  ) -> tuple[LieState, torch.Tensor, torch.Tensor]:
    """Unchecked tensor transition with explicit noise constants.

    The caller validates initial tensors and pins model/configuration for the
    rollout. No provider state, host numerical decisions, cache initialization,
    or detach of differentiable outputs occurs here. ``invalid`` is a boolean
    tensor with the belief's leading batch shape. The factor is (...,70,70),
    action (...,29), measurement (...,64), and measurement_std (64,).
    """
    predicted, jacobian, factor = self._prediction_kernel(
      mean, covariance_sqrt, action, context
    )
    result, *_ = self._correction_kernel(predicted, measurement, measurement_std)
    with torch.no_grad():
      invalid = belief_invalid(result) | belief_invalid(predicted)
      invalid = invalid | ~torch.isfinite(jacobian).all(dim=(-2, -1))
      invalid = invalid | ~torch.isfinite(factor).all(dim=(-2, -1))
      invalid = invalid | ~torch.isfinite(measurement).all(dim=-1)
      invalid = invalid | ~torch.isfinite(action).all(dim=-1)
    return result.mean, result.covariance_sqrt, invalid

  def correct(
    self,
    belief: BeliefState,
    measurement: torch.Tensor,
    *,
    return_diagnostics: bool | Literal["summary"] = False,
  ) -> BeliefState | tuple[BeliefState, CorrectionDiagnostics | CorrectionSummary]:
    result, diagnostics = self._correct(
      belief,
      measurement,
      return_diagnostics=bool(return_diagnostics),
      summary_only=return_diagnostics == "summary",
    )
    return (result, diagnostics) if diagnostics is not None else result

  def _correct(
    self,
    belief: BeliefState,
    measurement: torch.Tensor,
    *,
    return_diagnostics: bool,
    measurement_checked: bool = False,
    summary_only: bool = False,
  ) -> tuple[BeliefState, CorrectionDiagnostics | CorrectionSummary | None]:
    self._validate_belief(belief)
    if measurement.shape != (*belief.mean.position.shape[:-1], 64):
      raise ValueError("Measurement must have shape (*belief_batch, 64)")
    if (
      measurement.device != belief.covariance_sqrt.device
      or measurement.dtype != belief.covariance_sqrt.dtype
    ):
      raise ValueError("Measurement and belief must share dtype and device")
    if (
      self.config.validate_numerics
      and not measurement_checked
      and not bool(torch.isfinite(measurement).all())
    ):
      raise ValueError("Measurement contains non-finite values")
    std, _ = self.measurement_config._tensors(
      29, device=measurement.device, dtype=measurement.dtype
    )
    result, innovation, h, innovation_factor, gain, correction, reset = (
      self._correction_kernel(belief, measurement, std)
    )
    self._validate_belief(result)
    if summary_only:
      return result, CorrectionSummary(innovation.detach().norm(dim=-1))
    if return_diagnostics:
      return result, CorrectionDiagnostics(
        innovation,
        h,
        innovation_factor @ innovation_factor.mT,
        gain,
        correction,
        reset,
      )
    return result, None

  def _correction_kernel(self, belief, measurement, std):
    predicted = self.observation(belief.mean)
    h = self.observation_jacobian(belief.mean)
    innovation = measurement - predicted
    lower, correction, innovation_factor, gain, reset = self._correct_factor(
      belief.covariance_sqrt, h, std, innovation
    )
    result = BeliefState(boxplus(belief.mean, correction), covariance_sqrt=lower)
    return result, innovation, h, innovation_factor, gain, correction, reset

  @staticmethod
  def to_actor_input(belief: BeliefState) -> PosteriorInput:
    return belief.to_actor_input()


@torch.no_grad()
def belief_invalid(belief: BeliefState) -> torch.Tensor:
  """Per-sample numerical status; square roots guarantee covariance PSD."""
  mean, lower = belief.mean, belief.covariance_sqrt
  valid = rotation_valid(mean.rotation)
  for field in fields(mean):
    value = getattr(mean, field.name)
    axes = (-2, -1) if field.name == "rotation" else -1
    valid = valid & torch.isfinite(value).all(dim=axes)
  valid = valid & torch.isfinite(lower).all(dim=(-2, -1))
  valid = valid & (lower.diagonal(dim1=-2, dim2=-1) > 0).all(dim=-1)
  return ~valid
