"""TorchInductor settings for the pure tensor kernels used by EIT training."""

from collections.abc import Callable

import torch


def compile_eit_kernel[**P, R](
  fn: Callable[P, R], *, dynamic: bool | None = None
) -> Callable[P, R]:
  # Collection, the first imagined step, and later differentiable steps carry
  # different gradient guards. TensorDict also populates its type cache on the
  # first learner update. Keep room for these variants local to this kernel.
  # The pinned 2.14 implementation accepts both cache options; its overloads omit them.
  return torch.compile(  # pyright: ignore[reportCallIssue]
    fn,
    backend="inductor",
    fullgraph=True,
    dynamic=dynamic,
    isolate_recompiles=True,
    recompile_limit=32,
  )
