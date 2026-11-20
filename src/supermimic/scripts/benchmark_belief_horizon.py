"""NLL forward/backward cost at the configured batch and microbatch size.

Uses the production matrix filter, all model gradients, a fixed boundary belief
and synthetic measurements. Excludes collection, prefix reconstruction, detached
learner diagnostics and optimizer.step(). CUDA memory is PyTorch allocated memory.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import statistics
import subprocess
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import torch

from supermimic.belief import BeliefState
from supermimic.estimator import FilterConfig, LieBeliefFilter, posterior_nll
from supermimic.estimator_inputs import GaussianMeasurementConfig
from supermimic.learners.belief_wm_learner import BeliefWorldModelLearnerCfg
from supermimic.models import Dynamics, ProcessNoiseConfig
from supermimic.representations import GenCoordRepresentation
from supermimic.utils.lie_state import LieState, boxplus, so3_exp


def benchmark(batch, microbatch, horizon, device, warmup, repeats):
  gc.collect()
  torch.cuda.empty_cache()
  torch.manual_seed(731)
  rep = GenCoordRepresentation(29, torch.zeros(29, device=device))
  rep.dt, rep.substep_coeff = 0.02, 0.6
  model = (
    Dynamics(
      rep,
      29,
      33,
      torch.full((29,), 0.5, device=device),
      torch.zeros(29, device=device),
      (2048, 2048, 2048),
      process_noise=ProcessNoiseConfig(),
    )
    .to(device)
    .eval()
  )
  # Also exercise gradients in the noise hidden layers (zero at initial output).
  with torch.no_grad():
    model.noise_head[-1].weight.normal_(std=0.0001)
  filt = LieBeliefFilter(model, GaussianMeasurementConfig(), model.action_offset)
  state = LieState(
    so3_exp(torch.randn(batch, 3, device=device) * 0.4),
    torch.randn(batch, 3, device=device) * 0.1,
    torch.randn(batch, 3, device=device) * 0.2,
    torch.randn(batch, 3, device=device) * 0.1,
    torch.randn(batch, 29, device=device) * 0.1,
    torch.randn(batch, 29, device=device) * 0.1,
  )
  actions = torch.randn(batch, 29, device=device) * 0.1
  contexts = torch.randn(batch, 33, device=device) * 0.1
  initial = BeliefState(
    state, torch.eye(70, device=device).expand(batch, 70, 70) * 0.01
  )
  truth = boxplus(state, torch.full((70,), 0.001, device=device)).detach()
  observation = filt.observation(truth).detach()

  def run():
    model.zero_grad(set_to_none=True)
    total = torch.zeros((), device=device)
    for lo in range(0, batch, microbatch):
      hi = min(lo + microbatch, batch)
      selection = slice(lo, hi)
      belief = initial.index(selection)
      losses = []
      target = truth.index(selection)
      for step in range(horizon):
        belief = filt.correct(
          filt.predict(belief, actions[selection], contexts[selection]),
          observation[selection] + (step % 2) * 0.001,
        )
        losses.append(posterior_nll(belief, target, filt.config).sum())
      loss = torch.stack(losses).sum() / (batch * horizon)
      loss.backward()
      total += loss.detach()
    return total

  def validate(loss):
    if not torch.isfinite(loss):
      raise FloatingPointError("Non-finite NLL in measured sequence")
    if not all(
      p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    ):
      raise FloatingPointError("Non-finite/missing model gradient")

  for _ in range(warmup):
    validate(run())
  torch.cuda.synchronize(device)
  timings, peaks, reserved = [], [], []
  for _ in range(repeats):
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    loss = run()
    torch.cuda.synchronize(device)
    timings.append(time.perf_counter() - start)
    peaks.append(torch.cuda.max_memory_allocated(device))
    reserved.append(torch.cuda.max_memory_reserved(device))
    # Explicit finite checks occur outside timing; filter's production validation
    # and dynamic rank/Cholesky paths are still included in the measured workload.
    validate(loss)
  return {
    "batch_size": batch,
    "microbatch_size": microbatch,
    "horizon": horizon,
    "valid_transitions": batch * horizon,
    "mean_seconds": statistics.mean(timings),
    "median_seconds": statistics.median(timings),
    "min_seconds": min(timings),
    "max_seconds": max(timings),
    "seconds": timings,
    "peak_allocated_bytes": max(peaks),
    "peak_reserved_bytes": max(reserved),
    "peak_allocated_mib": max(peaks) / 2**20,
    "loss": float(loss),
    "loss_and_all_gradients_finite": True,
    "parameters": sum(p.numel() for p in model.parameters()),
  }


def main():
  defaults = BeliefWorldModelLearnerCfg()
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
  parser.add_argument("--microbatch-size", type=int, default=defaults.microbatch_size)
  parser.add_argument("--horizons", nargs="+", type=int, default=[4, 8, 16, 32])
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--warmup", type=int, default=2)
  parser.add_argument("--repeats", type=int, default=5)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  if (
    min(args.batch_size, args.microbatch_size, args.repeats, *args.horizons) < 1
    or args.warmup < 0
  ):
    parser.error("sizes/horizons/repeats must be positive and warmup nonnegative")
  device = torch.device(args.device)
  if device.type != "cuda" or not torch.cuda.is_available():
    parser.error("This benchmark measures CUDA time and allocated GPU memory")
  torch.set_num_threads(1)
  root = Path(__file__).resolve().parents[3]
  sources = (
    "src/supermimic/scripts/benchmark_belief_horizon.py",
    "src/supermimic/models.py",
    "src/supermimic/representations.py",
    "src/supermimic/estimator.py",
    "src/supermimic/utils/lie_state.py",
    "src/supermimic/learners/belief_wm_learner.py",
  )
  result = {
    "timestamp_utc": datetime.now(UTC).isoformat(),
    "torch": str(torch.__version__),
    "source_commit": subprocess.check_output(
      ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip(),
    "source_files_sha256": {
      name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources
    },
    "device": str(device),
    "gpu": torch.cuda.get_device_name(device),
    "dtype": "float32",
    "threads": 1,
    "seed": 731,
    "warmup": args.warmup,
    "repeats": args.repeats,
    "mean_hidden_dims": [2048, 2048, 2048],
    "noise_hidden_dims": [128, 64],
    "sim_context_dim": 33,
    "filter_config": asdict(FilterConfig()),
    "includes": "predict/correct/posterior_nll, full higher-order model backward, microbatch gradient accumulation",
    "excludes": "collection, replay/prefix rebuild, detached diagnostics, optimizer/Adam state",
    "memory": "maximum torch.cuda.max_memory_allocated across synchronized repeats; includes model, inputs, gradients, temporary graph",
    "results": [],
  }
  args.output.parent.mkdir(parents=True, exist_ok=True)
  for horizon in args.horizons:
    print(
      f"Measuring B={args.batch_size} microbatch={args.microbatch_size} T={horizon}",
      flush=True,
    )
    try:
      row = benchmark(
        args.batch_size,
        args.microbatch_size,
        horizon,
        device,
        args.warmup,
        args.repeats,
      )
      row["status"] = "passed"
    except torch.OutOfMemoryError as error:
      row = {
        "batch_size": args.batch_size,
        "microbatch_size": args.microbatch_size,
        "horizon": horizon,
        "status": "out_of_memory",
        "error": str(error),
      }
    result["results"].append(row)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
  main()
