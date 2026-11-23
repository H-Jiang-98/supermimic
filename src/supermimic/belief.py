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


@dataclass(frozen=True)
class BeliefState:
  mean: LieState
  covariance: torch.Tensor
  state_layout_version: str = STATE_LAYOUT_VERSION
  tangent_version: str = TANGENT_VERSION

  def _with(self, mean: LieState, covariance: torch.Tensor) -> BeliefState:
    return BeliefState(
      mean, covariance, self.state_layout_version, self.tangent_version
    )

  def clone(self) -> BeliefState:
    return self._with(self.mean.clone(), self.covariance.clone())

  def detach(self) -> BeliefState:
    return self._with(self.mean.detach(), self.covariance.detach())

  def to(self, *args, **kwargs) -> BeliefState:
    mean = self.mean.to(*args, **kwargs)
    covariance = self.covariance.to(*args, **kwargs)
    if torch.finfo(covariance.dtype).eps < torch.finfo(self.covariance.dtype).eps:
      self.validate()
      covariance = 0.5 * (covariance + covariance.mT)
      # A PSD matrix rounded at source precision may have tiny negative
      # eigenvalues at destination precision. Remove just that roundoff.
      # A cast transports covariance derivatives unchanged on symmetric
      # inputs; eigenvectors at repeated zero eigenvalues have no derivative.
      with torch.no_grad():
        std = covariance.diagonal(dim1=-2, dim2=-1).sqrt()
        scale = torch.where(std > 0, std, 1)
        correlation = covariance / scale.unsqueeze(-1) / scale.unsqueeze(-2)
        eigenvalues, basis = torch.linalg.eigh(correlation)
        repair = (basis * (-eigenvalues).clamp_min(0).unsqueeze(-2)) @ basis.mT
        repair = repair * std.unsqueeze(-1) * std.unsqueeze(-2)
      covariance = covariance + repair
    return self._with(mean, covariance)

  def index(self, selection) -> BeliefState:
    return self._with(self.mean.index(selection), self.covariance[selection])

  def to_actor_input(self) -> PosteriorInput:
    return PosteriorInput(
      RpvState(self.mean.rotation, self.mean.position, self.mean.velocity),
      self.covariance[..., :9, :9],
    )

  @property
  def covariance_rtol(self) -> float:
    return 128 * torch.finfo(self.covariance.dtype).eps

  def validate(self) -> None:
    """Check a boundary without modifying covariance or its computation graph."""
    if self.state_layout_version != STATE_LAYOUT_VERSION:
      raise ValueError("Unsupported belief state layout version")
    if self.tangent_version != TANGENT_VERSION:
      raise ValueError("Unsupported belief tangent version")
    self.mean.validate()
    covariance = self.covariance
    expected = (*self.mean.position.shape[:-1], TANGENT_DIM, TANGENT_DIM)
    if covariance.shape != expected or not covariance.is_floating_point():
      raise ValueError("Belief covariance must match the mean batch and (...,70,70)")
    if (
      covariance.device != self.mean.position.device
      or covariance.dtype != self.mean.position.dtype
    ):
      raise ValueError(
        "Belief mean and covariance must have identical device and dtype"
      )
    check_covariance(covariance, "Belief covariance", rtol=self.covariance_rtol)


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
    covariance = torch.diag(std.square()).expand(*shape[:-1], TANGENT_DIM, TANGENT_DIM)
    return BeliefState(boxplus(truth, noise * std), covariance)
