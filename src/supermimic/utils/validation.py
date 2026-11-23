"""Numerical checks at actor and filter input boundaries."""

import torch


@torch.no_grad()
def validate_state_dict(module: torch.nn.Module, supplied: dict, name: str) -> None:
  """Check every tensor before load_state_dict can copy a partial checkpoint."""
  expected = module.state_dict()
  if not isinstance(supplied, dict) or supplied.keys() != expected.keys():
    raise ValueError(f"{name} checkpoint parameter/buffer keys mismatch")
  for key, value in supplied.items():
    if (
      not isinstance(value, torch.Tensor)
      or value.shape != expected[key].shape
      or value.dtype != expected[key].dtype
      or value.layout != expected[key].layout
      or value.is_meta
      or not bool(torch.isfinite(value).all())
    ):
      raise ValueError(f"Invalid {name} checkpoint tensor: {key}")


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
  variance = matrix.diagonal(dim1=-2, dim2=-1)
  # A deterministic coordinate has zero covariance with every coordinate.
  zero = variance == 0
  std = torch.where(zero, 1.0, variance.double()).sqrt()
  correlation = matrix.double() / std.unsqueeze(-1) / std.unsqueeze(-2)
  # Inspect the preconditions together before passing a matrix to the solver.
  # The success path needs just one host decision for all five predicates.
  invalid = torch.stack(
    (
      ~torch.isfinite(matrix).all(),
      (variance < 0).any(),
      ((zero.unsqueeze(-1) | zero.unsqueeze(-2)) & (matrix != 0)).any(),
      ((correlation - correlation.mT).abs() > atol + rtol).any(),
      (correlation.abs() > 1 + atol + rtol).any(),
    )
  )
  if invalid.any():
    reasons = (
      "contains non-finite values",
      "must be positive semidefinite (negative marginal variance)",
      "must be positive semidefinite (nonzero deterministic row)",
      "is asymmetric",
      "must be positive semidefinite (correlation exceeds one)",
    )
    raise ValueError(f"{name} {reasons[invalid.tolist().index(True)]}")
  eig = torch.linalg.eigvalsh((correlation + correlation.mT) * 0.5)
  if (eig[..., 0] < -(atol + rtol * eig[..., -1].clamp_min(1))).any():
    raise ValueError(f"{name} must be positive semidefinite")
