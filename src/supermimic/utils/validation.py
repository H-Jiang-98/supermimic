"""Numerical checks at actor and filter input boundaries."""

import torch


@torch.no_grad()
def validate_rotation(rotation: torch.Tensor) -> None:
  if not rotation.is_floating_point() or rotation.shape[-2:] != (3, 3):
    raise ValueError("Rotation must be a floating-point 3 by 3 matrix")
  tolerance = 128 * torch.finfo(rotation.dtype).eps
  matrix = rotation.double()
  gram = matrix.mT @ matrix
  identity = torch.eye(3, device=matrix.device, dtype=matrix.dtype)
  if not torch.allclose(gram, identity.expand_as(gram), atol=tolerance, rtol=tolerance):
    raise ValueError("Rotation must be orthonormal")
  if (torch.linalg.det(matrix) <= 0).any():
    raise ValueError("Rotation must have positive determinant")


@torch.no_grad()
def check_covariance(
  matrix: torch.Tensor, name: str, *, rtol: float, atol: float = 0.0
):
  if (
    matrix.ndim < 2
    or matrix.shape[-1] != matrix.shape[-2]
    or not matrix.is_floating_point()
  ):
    raise ValueError(f"{name} must be a square floating-point matrix")
  if not torch.isfinite(matrix).all():
    raise ValueError(f"{name} contains non-finite values")
  variance = matrix.diagonal(dim1=-2, dim2=-1)
  if (variance < 0).any():
    raise ValueError(
      f"{name} must be positive semidefinite (negative marginal variance)"
    )
  # A deterministic coordinate has zero covariance with every coordinate.
  zero = variance == 0
  if ((zero.unsqueeze(-1) | zero.unsqueeze(-2)) & (matrix != 0)).any():
    raise ValueError(
      f"{name} must be positive semidefinite (nonzero deterministic row)"
    )
  std = torch.where(zero, 1.0, variance.double()).sqrt()
  correlation = matrix.double() / std.unsqueeze(-1) / std.unsqueeze(-2)
  if ((correlation - correlation.mT).abs() > atol + rtol).any():
    raise ValueError(f"{name} is asymmetric")
  if (correlation.abs() > 1 + atol + rtol).any():
    raise ValueError(f"{name} must be positive semidefinite (correlation exceeds one)")
  eig = torch.linalg.eigvalsh((correlation + correlation.mT) * 0.5)
  if (eig[..., 0] < -(atol + rtol * eig[..., -1].clamp_min(1))).any():
    raise ValueError(f"{name} must be positive semidefinite")
