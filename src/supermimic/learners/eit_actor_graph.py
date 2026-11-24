"""Private CUDA capture for one prepared EIT microbatch and its backward.

Gradients belong to the learner and accumulate across calls. Inputs and returned
outputs belong to this cache; callers must consume/copy outputs before reuse.
No optimizer, RNG, provider state or host numerical checks run inside the graph.
"""

from dataclasses import fields, is_dataclass
from time import perf_counter

import torch


def _map_inputs(value, tensor_fn, *, describe=False):
  if isinstance(value, torch.Tensor):
    return tensor_fn(value)
  if is_dataclass(value):
    values = {
      field.name: _map_inputs(getattr(value, field.name), tensor_fn, describe=describe)
      for field in fields(value)
    }
    return (type(value), values) if describe else type(value)(**values)
  if isinstance(value, tuple):
    return tuple(_map_inputs(v, tensor_fn, describe=describe) for v in value)
  # The estimator is an intentionally shared, read-only dependency, not copied.
  if describe and not isinstance(value, (str, int, float, bool, type(None))):
    return type(value), id(value)
  return value


def _layout(tensor):
  return tuple(tensor.shape), tensor.dtype, tensor.device


def _identity(tensors):
  return [(id(t), t.data_ptr(), _layout(t), t.stride()) for t in tensors]


class CapturedActorBackward:
  """Three real warmup calls, then one replay per submitted microbatch.

  Uses a private pool rather than the original learner's shared update-order pool.
  All eager warmup backwards run on the capture stream. Detached return values
  release AccumulateGrad references before the next warmup/capture.
  """

  def __init__(self, fn, state, stream):
    self._fn, self._state = fn, state
    self._inputs = self._spec = self._outputs = None
    self._input_tensors = []
    self._graph = None
    self._stream = stream
    # Keep old allocations alive, so pointer reuse cannot hide a replacement.
    self._owned_state = state()
    self._identity = _identity(self._owned_state)
    self.warmup_calls = self.replay_calls = self.captures = 0
    self.warmup_seconds = self.capture_seconds = 0.0

  def check_state(self):
    if _identity(self._state()) != self._identity:
      raise RuntimeError("EIT actor graph dependency or gradient storage moved")

  @property
  def stats(self):
    return {
      "warmup_calls": self.warmup_calls,
      "replay_calls": self.replay_calls,
      "captures": self.captures,
      "warmup_seconds": self.warmup_seconds,
      "capture_seconds": self.capture_seconds,
    }

  def __call__(self, prepared, process_noise, measurement_noise, total_samples):
    self.check_state()
    inputs = prepared, process_noise, measurement_noise, total_samples
    spec = _map_inputs(inputs, _layout, describe=True)
    if self._inputs is None:

      def clone(value):
        result = value.detach().clone(memory_format=torch.contiguous_format)
        self._input_tensors.append(result)
        return result

      self._inputs = _map_inputs(inputs, clone)
      self._spec = spec
    else:
      if spec != self._spec:
        raise RuntimeError("EIT actor graph input structure changed")
      destination = iter(self._input_tensors)

      def copy(value):
        next(destination).copy_(value)
        return None

      with torch.no_grad():
        _map_inputs(inputs, copy, describe=True)
    with torch.cuda.device(process_noise.device):
      current = torch.cuda.current_stream()
      if self.warmup_calls < 3:
        begin = perf_counter()
        self._stream.wait_stream(current)
        try:
          with torch.cuda.stream(self._stream):
            outputs = self._fn(*self._inputs)
        finally:
          current.wait_stream(self._stream)
        self.warmup_calls += 1
        self.warmup_seconds += perf_counter() - begin
        return outputs
      if self._graph is None:
        begin = perf_counter()
        self._stream.wait_stream(current)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=self._stream):
          self._outputs = self._fn(*self._inputs)
        self._graph = graph
        self.captures += 1
        self.capture_seconds += perf_counter() - begin
      self._graph.replay()
      self.replay_calls += 1
      return self._outputs
