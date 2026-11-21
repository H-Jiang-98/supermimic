"""Synchronized wall-clock profiling without touching model state or RNG."""

from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter

import torch


class PhaseTimer:
  """Accumulate complete phases; CUDA work is synchronized at both boundaries."""

  def __init__(self, device):
    self.device = torch.device(device)
    self.seconds: dict[str, float] = {}

  def now(self):
    if self.device.type == "cuda":
      torch.cuda.synchronize(self.device)
    return perf_counter()

  @contextmanager
  def measure(self, name):
    start = self.now()
    try:
      yield
    finally:
      self.seconds[name] = self.seconds.get(name, 0.0) + self.now() - start


def runtime_metadata(parameter, *, reset_peak=False):
  device = parameter.device
  result = {
    "device": str(device),
    "dtype": str(parameter.dtype),
    "cpu_threads": torch.get_num_threads(),
    "cpu_interop_threads": torch.get_num_interop_threads(),
    "torch_version": str(torch.__version__),
    "cuda_synchronized": device.type == "cuda",
    "cuda_peak_allocated_bytes": None,
    "cuda_peak_reserved_bytes": None,
    "cuda_allocated_bytes": None,
    "cuda_reserved_bytes": None,
  }
  if device.type == "cuda":
    torch.cuda.synchronize(device)
    if reset_peak:
      torch.cuda.reset_peak_memory_stats(device)
    result.update(
      device_name=torch.cuda.get_device_name(device),
      cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
      cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
      cuda_allocated_bytes=torch.cuda.memory_allocated(device),
      cuda_reserved_bytes=torch.cuda.memory_reserved(device),
    )
  return result
