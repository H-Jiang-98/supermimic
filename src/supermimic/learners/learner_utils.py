"""Shared learner utilities."""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterable, Sequence

import torch
import torch.nn as nn
from tensordict import TensorDict


def build_optimizers(
  modules: Sequence[nn.Module],
  lr: float,
  betas: tuple[float, float] = (0.9, 0.999),
  device: str = "cpu",
) -> list[torch.optim.Optimizer]:
  """Muon on the Linear weights between the first and the last Linear of each
  ``nn.Sequential`` in ``modules``, fused capturable Adam on the rest, both at
  ``lr``. Muon's ``lr`` is a float, fixed once an update is captured."""
  hidden = [
    linear.weight
    for module in modules
    for seq in module.modules()
    if isinstance(seq, nn.Sequential)
    for linear in [m for m in seq if isinstance(m, nn.Linear)][1:-1]
  ]
  hidden_ids = {id(p) for p in hidden}
  rest = [p for m in modules for p in m.parameters() if id(p) not in hidden_ids]
  adam = torch.optim.Adam(
    rest, lr=torch.tensor(lr, device=device), betas=betas, fused=True, capturable=True
  )
  if not hidden:
    return [adam]
  return [adam, torch.optim.Muon(hidden, lr=lr, adjust_lr_fn="match_rms_adamw")]


def validate_optimizer_state(
  optimizer: torch.optim.Optimizer, saved: dict, steps: int
) -> None:
  """Preflight the Adam or Muon parameter groups and moments before a restore."""
  if type(steps) is not int or steps < 0:
    raise ValueError("Invalid optimizer step count")
  groups = optimizer.state_dict()["param_groups"]
  if (
    not isinstance(saved, dict)
    or saved.keys() != {"param_groups", "state"}
    or saved["param_groups"] != groups
  ):
    raise ValueError("Optimizer parameter groups mismatch")
  params = [
    (index, parameter)
    for saved_group, live_group in zip(groups, optimizer.param_groups, strict=True)
    for index, parameter in zip(
      saved_group["params"], live_group["params"], strict=True
    )
  ]
  moments = saved.get("state")
  if not isinstance(moments, dict) or set(moments) != (
    {index for index, _ in params} if steps else set()
  ):
    raise ValueError("Optimizer state does not match its step count")
  if not steps:
    return
  fields = (
    {"momentum_buffer"}
    if isinstance(optimizer, torch.optim.Muon)
    else {"exp_avg", "exp_avg_sq", "step"}
  )
  for key, parameter in params:
    values = moments[key]
    if not isinstance(values, dict) or set(values) != fields:
      raise ValueError("Invalid optimizer moment fields")
    for name, value in values.items():
      if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.is_meta
        or not torch.isfinite(value).all()
      ):
        raise ValueError("Invalid optimizer state tensor")
      if name == "step":
        if (
          value.ndim != 0
          or value.dtype not in (torch.float32, torch.float64)
          or float(value) != steps
        ):
          raise ValueError("Adam step count mismatch")
      elif value.shape != parameter.shape or value.dtype != parameter.dtype:
        raise ValueError("Optimizer moment shape or dtype mismatch")
      elif name == "exp_avg_sq" and (value < 0).any():
        raise ValueError("Negative Adam second moment")


def _optimize_step(
  optimizers: Sequence[torch.optim.Optimizer],
  loss: torch.Tensor,
  params: Iterable[torch.Tensor] | None = None,
  max_norm: float | None = None,
  clip_value: float = 0.0,
  *,
  check_finite: bool = False,
) -> torch.Tensor | None:
  """Run one optimizer step: zero_grad -> backward -> optional clip -> step."""
  for optimizer in optimizers:
    optimizer.zero_grad(set_to_none=True)
  if check_finite and not torch.isfinite(loss):
    raise FloatingPointError("Non-finite optimizer loss")
  loss.backward()
  grad_norm = None
  if params is not None and max_norm is not None:
    # Captured updates cannot synchronize with the host. The eager model fit
    # checks the pre-clipping norm before either Adam or Muon can mutate state.
    grad_norm = torch.nn.utils.clip_grad_norm_(
      params, max_norm=max_norm, error_if_nonfinite=check_finite
    )
    if clip_value > 0:
      torch.nn.utils.clip_grad_value_(params, clip_value)
  for optimizer in optimizers:
    optimizer.step()
  return grad_norm


def graph_state(
  modules: Iterable[nn.Module], optimizers: Iterable[torch.optim.Optimizer]
) -> list[torch.Tensor]:
  """What a captured update reads that Python can rebind: parameters, buffers,
  optimizer state and learning-rate tensors."""
  state: list[torch.Tensor] = []
  for module in modules:
    state += [*module.parameters(), *module.buffers()]
  for optimizer in optimizers:
    state += [t for s in optimizer.state.values() for t in s.values()]
    state += [
      group["lr"] for group in optimizer.param_groups if torch.is_tensor(group["lr"])
    ]
  return state


def _identity(state: list[torch.Tensor]) -> list[int]:
  return [x.data_ptr() for x in state]


@functools.cache
def _capture_stream_and_pool(device: torch.device):
  return torch.cuda.Stream(device), torch.cuda.graph_pool_handle()


class CapturedUpdate:
  """Runs ``fn`` eagerly for its first ``warmup`` calls, then records it once as
  a CUDA graph and replays it on the device of ``state()``. ``fn`` must be
  capture-safe: no host syncs, fixed shapes, and every tensor it reads besides
  its inputs updated in place. Python values it reads are fixed at capture, and
  each replay overwrites the outputs it returns. All CapturedUpdates on a device
  record on one stream into one memory pool and must replay in the order they
  were recorded. A replay raises if a tensor in ``state()`` moved; a learning rate
  that changes is a device tensor, changed in place with ``group["lr"].fill_(lr)``."""

  def __init__(
    self,
    fn: Callable[[TensorDict], dict[str, torch.Tensor]],
    state: Callable[[], list[torch.Tensor]],
    warmup: int = 3,
    *,
    enabled: bool = True,
  ):
    self._fn = fn
    self._state = state
    self._warmup = warmup
    self._enabled = enabled
    self._inputs: TensorDict | None = None
    self._graph: torch.cuda.CUDAGraph | None = None
    self._identity: list[int] = []
    self._outputs: dict[str, torch.Tensor] = {}

  def __call__(self, inputs: TensorDict) -> dict[str, torch.Tensor]:
    if not self._enabled:
      return self._fn(inputs)
    state = self._state()
    if not state[0].is_cuda:
      return self._fn(inputs)
    with torch.cuda.device(state[0].device):
      stream, pool = _capture_stream_and_pool(state[0].device)
      if self._inputs is None:
        self._inputs = inputs.clone()
      else:
        self._inputs.update_(inputs)
      if self._warmup > 0:
        self._warmup -= 1
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
          outputs = self._fn(self._inputs)
        torch.cuda.current_stream().wait_stream(stream)
        return outputs
      if self._graph is None:
        print(f"[INFO] Capturing CUDA graph: {self._fn.__qualname__}", flush=True)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool, stream=stream):
          self._outputs = self._fn(self._inputs)
        self._graph, self._identity = graph, _identity(self._state())
        print(f"[INFO] CUDA graph ready: {self._fn.__qualname__}", flush=True)
      elif _identity(state) != self._identity:
        raise RuntimeError("a captured tensor moved")
      self._graph.replay()
      torch.cuda.current_stream().synchronize()
      return self._outputs
