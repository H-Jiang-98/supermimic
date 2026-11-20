"""Production belief numerical checks and CPU/CUDA scale measurements.

Run with ``OMP_NUM_THREADS=1 python -m supermimic.scripts.check_belief_scale``.
The default covers CPU float64 reference derivatives and CPU/CUDA float32
production networks, including the runner's default (2048, 2048, 2048) mean.
This is a reproducible acceptance/benchmark script, not a convergence study.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import resource
import subprocess
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import torch

from supermimic.belief import BeliefState
from supermimic.estimator import FilterConfig, LieBeliefFilter, posterior_nll
from supermimic.estimator_inputs import GaussianMeasurementConfig
from supermimic.models import Dynamics, ProcessNoiseConfig
from supermimic.representations import GenCoordRepresentation
from supermimic.utils.lie_state import LieState, boxplus, local_error, so3_exp


def fixture(batch, device, dtype, hidden=(16,), noise_hidden=(12,)):
  rep = GenCoordRepresentation(29, torch.zeros(29, device=device, dtype=dtype))
  rep.dt, rep.substep_coeff = 0.02, 0.6
  model = (
    Dynamics(
      rep,
      29,
      2,
      torch.full((29,), 0.5, device=device, dtype=dtype),
      torch.zeros(29, device=device, dtype=dtype),
      hidden,
      process_noise=ProcessNoiseConfig(hidden_dims=noise_hidden),
    )
    .to(device=device, dtype=dtype)
    .eval()
  )
  # Exercise noise hidden-layer gradients, which vanish at exact zero output weights.
  with torch.no_grad():
    model.noise_head[-1].weight.normal_(std=0.0001)
  state = LieState(
    so3_exp(torch.randn(batch, 3, device=device, dtype=dtype) * 0.4),
    torch.randn(batch, 3, device=device, dtype=dtype) * 0.1,
    torch.randn(batch, 3, device=device, dtype=dtype) * 0.2,
    torch.randn(batch, 3, device=device, dtype=dtype) * 0.1,
    torch.randn(batch, 29, device=device, dtype=dtype) * 0.1,
    torch.randn(batch, 29, device=device, dtype=dtype) * 0.1,
  )
  return (
    model,
    state,
    torch.randn(batch, 29, device=device, dtype=dtype) * 0.1,
    torch.randn(batch, 2, device=device, dtype=dtype) * 0.1,
  )


def estimator(model, **kwargs):
  return LieBeliefFilter(
    model,
    GaussianMeasurementConfig(),
    model.action_offset.new_zeros(29),
    FilterConfig(**kwargs),
  )


def close(actual, expected, atol=1e-8, rtol=1e-7):
  torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
  return float((actual - expected).abs().max().detach())


def objective(filt, state, action, context, horizon):
  covariance = (
    torch.eye(70, device=action.device, dtype=action.dtype).expand(
      *action.shape[:-1], 70, 70
    )
    * 0.01
  )
  belief = BeliefState(state, covariance)
  truth = boxplus(state, action.new_tensor(0.001).expand(70)).detach()
  observation = filt.observation(truth).detach()
  losses = []
  for i in range(horizon):
    belief = filt.correct(
      filt.predict(belief, action, context), observation + (i % 2) * 0.001
    )
    losses.append(posterior_nll(belief, truth, filt.config).mean())
  return torch.stack(losses).mean()


def directional_difference(function, parameter, gradient):
  original = parameter.detach().clone()
  direction = gradient.detach() / gradient.detach().norm()
  epsilon = 2e-6
  values = []
  try:
    for sign in (1, -1):
      with torch.no_grad():
        parameter.copy_(original + sign * epsilon * direction)
        values.append(function().detach())
  finally:
    with torch.no_grad():
      parameter.copy_(original)
  finite = (values[0] - values[1]) / (2 * epsilon)
  autodiff = (direction * gradient).sum()
  close(autodiff, finite, atol=2e-5, rtol=3e-4)
  return {
    "autograd": float(autodiff),
    "finite_difference": float(finite),
    "relative_error": float((autodiff - finite).abs() / finite.abs().clamp_min(1e-12)),
  }


@torch.no_grad()
def long_yaw_rollout(device):
  """Check one thousand complete fp32 updates against constant-rate rotation."""
  torch.manual_seed(731)
  model, state, action, context = fixture(1, device, torch.float32)
  model.mlp[-1].weight.zero_()
  model.mlp[-1].bias.zero_()
  identity = torch.eye(3, device=device)
  omega = action.new_tensor([[0.0, 0.0, 2.3204725]])
  state = replace(
    state, rotation=identity.unsqueeze(0), omega=omega, velocity=torch.zeros_like(omega)
  )
  filt = estimator(model)
  belief = BeliefState(state, torch.eye(70, device=device).unsqueeze(0) * 0.01)
  max_drift = torch.zeros((), device=device, dtype=torch.float64)
  steps = 1000
  for _ in range(steps):
    predicted = filt.predict(belief, action, context)
    belief = filt.correct(predicted, filt.observation(predicted.mean))
    rotation = belief.mean.rotation.double()
    max_drift = torch.maximum(
      max_drift, (rotation.mT @ rotation - identity).abs().max()
    )
  expected = so3_exp(omega.double() * (steps * model.dt))
  final_error = close(belief.mean.rotation.double(), expected, atol=8e-6, rtol=0)
  position_error = close(belief.mean.position, state.position, atol=0, rtol=0)
  if max_drift > 1e-6:
    raise AssertionError(f"Long yaw rollout exceeded Gram error tolerance: {max_drift}")
  return {
    "device": str(device),
    "dtype": "torch.float32",
    "steps": steps,
    "yaw_rate_rad_per_second": float(omega[0, 2]),
    "max_abs_gram_error": float(max_drift),
    "final_rotation_max_abs_error": final_error,
    "final_position_max_abs_error": position_error,
  }


def numerical_checks(device, dtype):
  torch.manual_seed(731)
  model, state, action, context = fixture(2, device, dtype)
  action.requires_grad_()
  vector = estimator(model)
  reference = estimator(model, jacobian_backend="eager")
  chunked = estimator(model, jacobian_batch_chunk_size=1, jacobian_chunk_size=16)
  tolerance = 3e-5 if dtype == torch.float32 else 1e-8
  # Repeated evaluation exercises the warmed nested-autograd path.
  for _ in range(3):
    vector.transition_jacobian(state, action, context)
  a = vector.transition_jacobian(state, action, context)
  eager = reference.transition_jacobian(state, action, context)
  chunks = chunked.transition_jacobian(state, action, context)
  result = {
    "device": str(device),
    "dtype": str(dtype),
    "jacobian_vs_eager_max_abs_error": close(a, eager, tolerance, tolerance),
    "chunked_jacobian_max_abs_error": close(chunks, a, tolerance, tolerance),
  }
  weight = torch.randn_like(a)
  mixed = torch.autograd.grad((a * weight).sum(), model.mlp[0].weight)[0]
  mixed_eager = torch.autograd.grad((eager * weight).sum(), model.mlp[0].weight)[0]
  result["mixed_derivative_max_abs_error"] = close(
    mixed, mixed_eager, tolerance * 10, tolerance * 10
  )
  if dtype == torch.float64:
    result["mixed_direction"] = directional_difference(
      lambda: (vector.transition_jacobian(state, action, context) * weight).sum(),
      model.mlp[0].weight,
      mixed,
    )
    columns = []
    with torch.no_grad():
      nominal = model.mean_lie(state, action, context)
      for epsilon in torch.eye(70, device=device, dtype=dtype) * 1e-6:
        plus = model.mean_lie(boxplus(state, epsilon), action, context)
        minus = model.mean_lie(boxplus(state, -epsilon), action, context)
        columns.append(
          (local_error(nominal, plus) - local_error(nominal, minus)) / 2e-6
        )
    result["jacobian_finite_difference_max_abs_error"] = close(
      a, torch.stack(columns, -1), 2e-8, 2e-6
    )
  selected = {
    "mean_first_weight": model.mlp[0].weight,
    "mean_last_bias": model.mlp[-1].bias,
    "noise_first_weight": model.noise_head[0].weight,
    "noise_last_weight": model.noise_head[-1].weight,
    "action": action,
  }
  value = objective(vector, state, action, context, 2)
  expected = objective(reference, state, action, context, 2)
  result["two_step_nll_max_abs_error"] = close(
    value, expected, tolerance * 10, tolerance
  )
  gradients = torch.autograd.grad(value, tuple(selected.values()))
  eager_gradients = torch.autograd.grad(expected, tuple(selected.values()))
  result["nll_gradients"] = {}
  for (name, parameter), gradient, expected_grad in zip(
    selected.items(), gradients, eager_gradients, strict=True
  ):
    assert bool(torch.isfinite(gradient).all()) and float(gradient.norm()) > 0
    entry = {
      "norm": float(gradient.norm()),
      "eager_relative_l2_error": float(
        (gradient - expected_grad).norm() / expected_grad.norm().clamp_min(1e-12)
      ),
      "eager_max_abs_error": close(
        gradient, expected_grad, tolerance * 100, tolerance * 100
      ),
    }
    assert entry["eager_relative_l2_error"] < (2e-5 if dtype == torch.float32 else 1e-9)
    if dtype == torch.float64:
      entry["direction"] = directional_difference(
        lambda: objective(vector, state, action, context, 2), parameter, gradient
      )
    result["nll_gradients"][name] = entry
  model.requires_grad_(False)
  gradient = torch.autograd.grad(objective(vector, state, action, context, 2), action)[
    0
  ]
  assert bool(torch.isfinite(gradient).all()) and float(gradient.norm()) > 0
  result["frozen_model_action_gradient_norm"] = float(gradient.norm())
  if dtype == torch.float64:
    result["frozen_model_action_direction"] = directional_difference(
      lambda: objective(vector, state, action, context, 2), action, gradient
    )
  return result


def measured(function, device, warmup, repeats):
  def synchronize():
    if device.type == "cuda":
      torch.cuda.synchronize(device)

  for _ in range(warmup):
    function()
  synchronize()
  if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats(device)
    baseline = torch.cuda.memory_allocated(device)
  start = time.perf_counter()
  for _ in range(repeats):
    function()
  synchronize()
  result = {"seconds": (time.perf_counter() - start) / repeats}
  if device.type == "cuda":
    peak = torch.cuda.max_memory_allocated(device)
    result.update(
      baseline_allocated_bytes=baseline,
      peak_allocated_bytes=peak,
      incremental_peak_bytes=peak - baseline,
    )
  else:
    result["peak_allocated_bytes"] = None
    # Linux reports KiB; RSS is process lifetime high-water, not an operation delta.
    result["process_lifetime_peak_rss_bytes"] = (
      resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    )
  return result


def scale_case(device, hidden, batch, horizon, chunk, batch_chunk, warmup, repeats):
  gc.collect()
  if device.type == "cuda":
    torch.cuda.empty_cache()
  torch.manual_seed(731)
  model, state, action, context = fixture(
    batch, device, torch.float32, hidden, (128, 64)
  )
  filt = estimator(
    model, jacobian_chunk_size=chunk, jacobian_batch_chunk_size=batch_chunk
  )
  covariance = torch.eye(70, device=device).expand(batch, 70, 70) * 0.01
  belief = BeliefState(state, covariance)
  observation = filt.observation(state).detach()

  @torch.no_grad()
  def jacobian():
    filt.transition_jacobian(state, action, context)

  @torch.no_grad()
  def filtering():
    filt.correct(filt.predict(belief, action, context), observation)

  @torch.no_grad()
  def nll_forward():
    objective(filt, state, action, context, horizon)

  def nll_backward():
    model.zero_grad(set_to_none=True)
    loss = objective(filt, state, action, context, horizon)
    loss.backward()
    assert bool(torch.isfinite(loss))
    assert all(
      p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )

  result = {
    "device": str(device),
    "dtype": "torch.float32",
    "mean_hidden": list(hidden),
    "noise_hidden": [128, 64],
    "batch": batch,
    "horizon": horizon,
    "filter_config": asdict(filt.config),
    "parameters": sum(p.numel() for p in model.parameters()),
  }
  for name, function in (
    ("jacobian", jacobian),
    ("predict_correct", filtering),
    ("posterior_nll_forward", nll_forward),
    ("posterior_nll_forward_backward", nll_backward),
  ):
    result[name] = measured(function, device, warmup, repeats)
  return result


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--device", default="all", help="all, cpu, or cuda:0")
  parser.add_argument("--warmup", type=int, default=1)
  parser.add_argument("--repeats", type=int, default=3)
  parser.add_argument("--output", type=Path)
  args = parser.parse_args()
  if args.warmup < 0 or args.repeats < 1:
    parser.error("warmup must be nonnegative and repeats must be positive")
  torch.set_num_threads(1)
  if args.device == "all":
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
      devices.append(torch.device("cuda:0"))
  else:
    devices = [torch.device(args.device)]
  result = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "torch": torch.__version__,
    "source_commit": subprocess.check_output(
      ["git", "rev-parse", "HEAD"], text=True
    ).strip(),
    "source_files_sha256": {
      path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
      for path in (
        "src/supermimic/scripts/check_belief_scale.py",
        "src/supermimic/models.py",
        "src/supermimic/representations.py",
        "src/supermimic/estimator.py",
        "src/supermimic/utils/lie_state.py",
      )
    },
    "threads": torch.get_num_threads(),
    "warmup": args.warmup,
    "repeats": args.repeats,
    "cuda_synchronized": True,
    "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "numerical_checks": [],
    "long_yaw_rollouts": [],
    "scale": [],
  }
  for device in devices:
    print(f"Checking {device} 1000-step float32 yaw rollout", flush=True)
    result["long_yaw_rollouts"].append(long_yaw_rollout(device))
    dtype = torch.float64 if device.type == "cpu" else torch.float32
    print(f"Checking {device} {dtype}", flush=True)
    result["numerical_checks"].append(numerical_checks(device, dtype))
    cases = (
      ((32, 32), 1, 2, None, None),
      ((32, 32), 4, 4, None, None),
      ((256, 256), 2, 2, None, None),
      ((2048, 2048, 2048), 1, 2, None, None),
      ((2048, 2048, 2048), 2, 4, None, None),
      ((2048, 2048, 2048), 4, 4, None, None),
      ((2048, 2048, 2048), 4, 4, 16, None),
      ((2048, 2048, 2048), 4, 4, None, 1),
    )
    for hidden, batch, horizon, chunk, batch_chunk in cases:
      print(
        f"Timing {device} widths={hidden} B={batch} T={horizon} chunk={chunk} batch_chunk={batch_chunk}",
        flush=True,
      )
      result["scale"].append(
        scale_case(
          device, hidden, batch, horizon, chunk, batch_chunk, args.warmup, args.repeats
        )
      )
  result["status"] = "passed"
  serialized = json.dumps(result, indent=2)
  if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized + "\n")
  print(serialized)


if __name__ == "__main__":
  main()
