"""Differentiable covariance factors without forming a Gram matrix.

For a full-row-rank A, root(A) is the positive-diagonal lower triangle L
with L L.T = A A.T. QR avoids squaring A's condition number. Its VJP uses
only L and A, so training need not construct or differentiate QR's Q.
"""

import torch


class _CovarianceRoot(torch.autograd.Function):
  @staticmethod
  def forward(ctx, matrix):
    upper = torch.linalg.qr(matrix.mT, mode="r").R
    signs = torch.where(upper.diagonal(dim1=-2, dim2=-1) < 0, -1.0, 1.0)
    lower = upper.mT * signs.unsqueeze(-2)
    # Save the returned L (not a detached intermediate) for double backward.
    ctx.save_for_backward(matrix, lower)
    return lower

  @staticmethod
  def backward(ctx, gradient):
    matrix, lower = ctx.saved_tensors
    product = lower.mT @ gradient
    triangle = product.tril(-1)
    symmetric = triangle + triangle.mT + product.diagonal(dim1=-2, dim2=-1).diag_embed()
    normalized = torch.linalg.solve_triangular(lower, matrix, upper=False)
    return torch.linalg.solve_triangular(lower.mT, symmetric @ normalized, upper=True)


def covariance_root(matrix: torch.Tensor) -> torch.Tensor:
  """Batched square root; callers guarantee full row rank and finite inputs."""
  return _CovarianceRoot.apply(matrix)
