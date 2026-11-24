"""Versioned full-state Gaussian belief; actor views are explicit marginals."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from supermimic.estimator_inputs import PosteriorInput, RpvState
from supermimic.utils.lie_state import (
  STATE_LAYOUT_VERSION,
  TANGENT_DIM,
  TANGENT_VERSION,
  LieState,
  boxplus,
)
from supermimic.utils.validation import check_covariance


@dataclass(frozen=True, init=False)
class BeliefState:
  """Gaussian state with a single authoritative square-root covariance.

  The positional covariance constructor is an external-data boundary. Internal
  code supplies ``covariance_sqrt=`` to transport factors without factorization
  or numerical validation. ``covariance`` materializes a read-only view by value.
  """

  mean: LieState
  covariance_sqrt: torch.Tensor
  state_layout_version: str = STATE_LAYOUT_VERSION
  tangent_version: str = TANGENT_VERSION

  def __init__(
    self,
    mean,
    covariance=None,
    state_layout_version=STATE_LAYOUT_VERSION,
    tangent_version=TANGENT_VERSION,
    *,
    covariance_sqrt=None,
  ):
    if covariance is not None:
      if covariance_sqrt is not None:
        raise ValueError("Supply covariance or covariance_sqrt, not both")
      check_covariance(
        covariance, "Belief covariance", rtol=128 * torch.finfo(covariance.dtype).eps
      )
      covariance_sqrt = torch.linalg.cholesky(0.5 * (covariance + covariance.mT))
    if covariance_sqrt is None:
      raise ValueError("Belief requires a covariance or covariance_sqrt")
    object.__setattr__(self, "mean", mean)
    object.__setattr__(self, "covariance_sqrt", covariance_sqrt)
    object.__setattr__(self, "state_layout_version", state_layout_version)
    object.__setattr__(self, "tangent_version", tangent_version)

  @property
  def covariance(self) -> torch.Tensor:
    return self.covariance_sqrt @ self.covariance_sqrt.mT

  @property
  def covariance_rpv(self) -> torch.Tensor:
    # L is lower triangular; the leading marginal needs only its 9x9 block.
    lower = self.covariance_sqrt[..., :9, :9]
    return lower @ lower.mT

  def _with(self, mean: LieState, lower: torch.Tensor) -> BeliefState:
    return BeliefState(
      mean,
      covariance_sqrt=lower,
      state_layout_version=self.state_layout_version,
      tangent_version=self.tangent_version,
    )

  def clone(self) -> BeliefState:
    return self._with(self.mean.clone(), self.covariance_sqrt.clone())

  def detach(self) -> BeliefState:
    return self._with(self.mean.detach(), self.covariance_sqrt.detach())

  def to(self, *args, **kwargs) -> BeliefState:
    return self._with(
      self.mean.to(*args, **kwargs), self.covariance_sqrt.to(*args, **kwargs)
    )

  def index(self, selection) -> BeliefState:
    return self._with(self.mean.index(selection), self.covariance_sqrt[selection])

  def to_actor_input(self) -> PosteriorInput:
    return PosteriorInput(
      RpvState(self.mean.rotation, self.mean.position, self.mean.velocity),
      self.covariance_rpv,
    )

  @property
  def covariance_rtol(self) -> float:
    return 128 * torch.finfo(self.covariance_sqrt.dtype).eps

  def validate_structure(self) -> None:
    """Metadata-only checks, safe on the GPU hot path."""
    if self.state_layout_version != STATE_LAYOUT_VERSION:
      raise ValueError("Unsupported belief state layout version")
    if self.tangent_version != TANGENT_VERSION:
      raise ValueError("Unsupported belief tangent version")
    self.mean.validate_structure()
    lower = self.covariance_sqrt
    expected = (*self.mean.position.shape[:-1], TANGENT_DIM, TANGENT_DIM)
    if lower.shape != expected or not lower.is_floating_point():
      raise ValueError(
        "Belief covariance factor must match the mean batch and (...,70,70)"
      )
    if (
      lower.device != self.mean.position.device
      or lower.dtype != self.mean.position.dtype
    ):
      raise ValueError("Belief mean and covariance factor must share dtype and device")

  def validate(self) -> None:
    """Explicit initialization/load/debug check; never called by filter kernels."""
    self.validate_structure()
    self.mean.validate()
    lower = self.covariance_sqrt
    if not torch.isfinite(lower).all():
      raise ValueError("Belief covariance factor contains non-finite values")
    if (lower.diagonal(dim1=-2, dim2=-1) <= 0).any() or lower.triu(1).count_nonzero():
      raise ValueError(
        "Belief covariance factor must be lower triangular with positive diagonal"
      )


def to_actor_input(belief: BeliefState) -> PosteriorInput:
  """Project the full posterior to the actor's RPV marginal, never condition it."""
  return belief.to_actor_input()


@dataclass(frozen=True)
class GaussianStatePrior:
  """Initialization noise in right-local theta, rho, nu, omega, q, qdot units."""

  std: tuple[float, ...] = (0.01, 0.01, 0.05, 0.05, 0.005, 0.05)

  def __post_init__(self):
    if len(self.std) != 6 or any(not math.isfinite(s) or s <= 0 for s in self.std):
      raise ValueError("Initial prior requires six finite positive standard deviations")

  def sample(
    self, truth: LieState, *, generator: torch.Generator | None = None
  ) -> BeliefState:
    truth.validate()
    std = truth.position.new_tensor(
      [s for s, n in zip(self.std, (3, 3, 3, 3, 29, 29), strict=True) for _ in range(n)]
    )
    if not (torch.isfinite(std.square()) & (std.square() > 0)).all():
      raise ValueError("Prior variances must be representable in the state dtype")
    shape = (*truth.position.shape[:-1], TANGENT_DIM)
    noise = torch.randn(shape, device=std.device, dtype=std.dtype, generator=generator)
    lower = torch.diag(std).expand(*shape[:-1], TANGENT_DIM, TANGENT_DIM)
    return BeliefState(boxplus(truth, noise * std), covariance_sqrt=lower)
