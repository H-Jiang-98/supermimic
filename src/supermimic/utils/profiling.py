"""Synchronized wall-clock profiling without touching model state or RNG."""

from __future__ import annotations

import os
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


class StreamTimer:
  """Time interleaved work on one stream, collecting CUDA results at phase end."""

  def __init__(self, device):
    device = torch.device(device)
    self.stream = torch.cuda.current_stream(device) if device.type == "cuda" else None
    self._events = []
    self._seconds = {}

  @contextmanager
  def measure(self, name):
    if self.stream is None:
      start = perf_counter()
      try:
        yield
      finally:
        self._seconds[name] = self._seconds.get(name, 0.0) + (perf_counter() - start)
    else:
      begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
      begin.record(self.stream)
      try:
        yield
      finally:
        end.record(self.stream)
        self._events.append((name, begin, end))

  def seconds(self):
    totals = self._seconds.copy()
    if self._events:
      # Events were appended in completion order on the same stream.
      self._events[-1][2].synchronize()
      for name, begin, end in self._events:
        totals[name] = totals.get(name, 0.0) + begin.elapsed_time(end) / 1000
    return totals


def runtime_metadata(parameter, *, reset_peak=False, synchronize=True):
  device = parameter.device
  result = {
    "device": str(device),
    "dtype": str(parameter.dtype),
    "cpu_threads": torch.get_num_threads(),
    "cpu_interop_threads": torch.get_num_interop_threads(),
    "torch_version": str(torch.__version__),
    "cuda_synchronized": device.type == "cuda" and synchronize,
    "cuda_peak_allocated_bytes": None,
    "cuda_peak_reserved_bytes": None,
    "cuda_allocated_bytes": None,
    "cuda_reserved_bytes": None,
    "cuda_allocator_backend": None,
    "cuda_allocator_config": None,
    "cuda_active_bytes": None,
    "cuda_inactive_split_bytes": None,
    "cuda_allocation_retries": None,
    "cuda_ooms": None,
    "cuda_free_bytes": None,
    "cuda_total_bytes": None,
  }
  if device.type == "cuda":
    if synchronize:
      torch.cuda.synchronize(device)
    if reset_peak:
      torch.cuda.reset_peak_memory_stats(device)
    stats = torch.cuda.memory_stats(device)
    free, total = torch.cuda.mem_get_info(device)
    result.update(
      cuda_allocator_backend=torch.cuda.get_allocator_backend(),
      cuda_allocator_config=os.environ.get(
        "PYTORCH_ALLOC_CONF", os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
      ),
      cuda_active_bytes=stats.get("active_bytes.all.current", 0),
      cuda_inactive_split_bytes=stats.get("inactive_split_bytes.all.current", 0),
      cuda_allocation_retries=stats.get("num_alloc_retries", 0),
      cuda_ooms=stats.get("num_ooms", 0),
      cuda_free_bytes=free,
      cuda_total_bytes=total,
      device_name=torch.cuda.get_device_name(device),
      cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
      cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
      cuda_allocated_bytes=torch.cuda.memory_allocated(device),
      cuda_reserved_bytes=torch.cuda.memory_reserved(device),
    )
  return result
