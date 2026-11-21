"""Time the batch-one EIT control path on real MuJoCo observations.

The uninstrumented pass measures measurement/history -> full filter/Jacobian ->
features -> actor -> action limits, excluding physics and offline diagnostics.
A separate pass synchronizes each stage for attribution; its end-to-end time is
reported separately because those extra synchronization points add overhead.
"""

from __future__ import annotations

import argparse
import json
import platform
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter_ns

import numpy as np
import torch

from supermimic.motion_loader import motion_content_digest
from supermimic.runners.eit_inference import (
  EITPolicyAdapter,
  evaluation_config,
  load_evaluation_models,
  load_hashed_checkpoint,
)
from supermimic.runners.eit_snapshots import file_sha256
from supermimic.scripts.eval_eit import _validate_output_paths, evaluation_env_config
from supermimic.scripts.train_utils import create_env


class PhaseTimer:
  """Host wall-clock latency with explicit CUDA completion at both boundaries."""

  def __init__(self, device):
    self.device = torch.device(device)
    self.values: dict[str, float] = {}

  def synchronize(self):
    if self.device.type == "cuda":
      torch.cuda.synchronize(self.device)

  @contextmanager
  def __call__(self, name):
    self.synchronize()
    start = perf_counter_ns()
    try:
      yield
    finally:
      self.synchronize()
      elapsed = (perf_counter_ns() - start) / 1e6
      self.values[name] = self.values.get(name, 0.0) + elapsed


def latency_statistics(values, *, control_period_ms=20.0):
  """Use linear-interpolated empirical quantiles, retaining empty subsets."""
  if not values:
    return {"count": 0, "mean_ms": None, "p95_ms": None, "p99_ms": None}
  data = np.asarray(values, dtype=np.float64)
  if not np.isfinite(data).all() or (data < 0).any():
    raise ValueError("Latency samples must be finite and nonnegative")
  mean, p95, p99 = float(data.mean()), *map(float, np.quantile(data, [0.95, 0.99]))
  return {
    "count": len(values),
    "mean_ms": mean,
    "p95_ms": p95,
    "p99_ms": p99,
    "min_ms": float(data.min()),
    "max_ms": float(data.max()),
    "mean_headroom_ms": control_period_ms - mean,
    "p95_headroom_ms": control_period_ms - p95,
    "p99_headroom_ms": control_period_ms - p99,
    "deadline_miss_fraction": float((data > control_period_ms).mean()),
  }


def _memory(device):
  if torch.device(device).type != "cuda":
    return None
  return {
    "allocated_bytes": torch.cuda.memory_allocated(device),
    "reserved_bytes": torch.cuda.memory_reserved(device),
    "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
    "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
  }


def _cpu_name():
  try:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
      if line.startswith("model name"):
        return line.partition(":")[2].strip()
  except OSError:
    pass
  return platform.processor() or platform.machine()


def _summarize(rows, control_period_ms):
  result = {}
  for subset, selected in (
    ("all", rows),
    ("continuous_full_filter", [row for row in rows if not row["reset"]]),
    ("reset_prior", [row for row in rows if row["reset"]]),
  ):
    names = sorted({key for row in rows for key in row["timings_ms"]})
    result[subset] = {
      name: latency_statistics(
        [row["timings_ms"][name] for row in selected],
        control_period_ms=control_period_ms,
      )
      for name in names
    }
  return result


def run_timing_pass(env, policy, *, seed, warmup, samples, stages=False):
  """Exercise production reset/advance/act, never the cached-action fast path."""
  timer = PhaseTimer(policy.device)
  previous_timer = policy.phase_timer
  policy.phase_timer = timer if stages else None
  try:
    torch.manual_seed(seed)
    with timer("environment_reset"):
      obs, _ = env.reset()
    with timer("adapter_reset"):
      policy.reset(obs)
    with timer("first_act"):
      actions = policy.act()
    first_reset = dict(timer.values)
    rows = []
    warmup_resets = 0
    for index in range(warmup + samples):
      # Physics, reward and termination calculation are outside the control timer.
      obs, _, dones, _ = env.step(actions)
      reset = bool(dones.any())
      if index == warmup and timer.device.type == "cuda":
        timer.synchronize()
        torch.cuda.reset_peak_memory_stats(timer.device)
      timer.values.clear()
      with timer("total"):
        policy.advance(obs, dones)
        actions = policy.act()
      if index < warmup:
        warmup_resets += int(reset)
        continue
      rows.append(
        {
          "sample": index - warmup,
          "reset": reset,
          "episode_step": int(policy.time_index.item()),
          "timings_ms": dict(timer.values),
          "memory": _memory(timer.device),
        }
      )
    return {
      "first_reset": first_reset,
      "warmup_reset_count": warmup_resets,
      "sample_reset_count": sum(row["reset"] for row in rows),
      "statistics": _summarize(rows, 1000 * env.unwrapped.step_dt),
      "samples": rows,
      "memory": _memory(timer.device),
    }
  finally:
    policy.phase_timer = previous_timer


def benchmark(
  checkpoint: Path,
  motion_file: Path,
  *,
  device="cpu",
  seed=2026,
  warmup=10,
  samples=100,
):
  if type(samples) is not int or samples < 1:
    raise ValueError("samples must be a positive integer")
  if type(warmup) is not int or warmup < 0:
    raise ValueError("warmup must be a nonnegative integer")
  saved, digest = load_hashed_checkpoint(checkpoint, device)
  cfg = evaluation_config(saved)
  assert cfg.estimator_inputs is not None
  torch.manual_seed(seed)
  with torch.compiler.set_stance("force_eager"):
    env = create_env(evaluation_env_config(motion_file, seed), 1, device)
    try:
      wm, actor = load_evaluation_models(env, cfg, saved, device)
      # Optimizer tensors and serialized weight copies are not part of the
      # running controller. Release them before warmup and memory measurements.
      del saved
      if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
      policy = EITPolicyAdapter(env, actor, wm, cfg)
      total = run_timing_pass(env, policy, seed=seed, warmup=warmup, samples=samples)
      stages = run_timing_pass(
        env, policy, seed=seed, warmup=warmup, samples=samples, stages=True
      )
      dev = torch.device(device)
      result = {
        "version": "eit_online_latency_v1",
        "checkpoint_sha256": digest,
        "motion_file": str(Path(motion_file).resolve()),
        "motion_sha256": file_sha256(Path(motion_file)),
        "motion_digest": motion_content_digest(policy.command.motion),
        "group": cfg.estimator_inputs.group,
        "batch_size": 1,
        "device": str(dev),
        "device_name": (
          torch.cuda.get_device_name(dev) if dev.type == "cuda" else _cpu_name()
        ),
        "platform": platform.platform(),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "dtype": str(next(actor.parameters()).dtype),
        "cpu_threads": torch.get_num_threads(),
        "cpu_interop_threads": torch.get_num_interop_threads(),
        "seed": seed,
        "warmup_steps": warmup,
        "measured_steps": samples,
        "control_period_ms": 1000 * env.unwrapped.step_dt,
        "posterior_diagnostics": False,
        "checkpoint_released_before_warmup": True,
        "memory_scope": (
          "Process-wide PyTorch allocator only: active inference models and "
          "PyTorch simulation tensors are included; the loaded checkpoint and "
          "optimizer tensors are released before warmup. Non-PyTorch allocations "
          "(including Warp/MuJoCo device allocations) are not measured. "
          "Peak counters reset after warmup and include subsequent physics steps."
        ),
        "compiled": False,
        "cuda_synchronization": (
          "total pass: control boundaries; stage pass: every stage boundary"
          if dev.type == "cuda"
          else "not applicable"
        ),
        "timing_scope": (
          "Observation already supplied by physics; measurement noise/history, "
          "complete filter/Jacobian, reference/features, actor, clamp/commit. "
          "Physics, offline diagnostics and sensor hardware I/O are excluded. "
          "Reset prior samples exclude transition prediction and are listed separately."
        ),
        "first_reset_scope": (
          "First explicit evaluation reset after environment/model construction; "
          "construction may itself reset physics. Initial act precedes warmup."
        ),
        "passes_are_independent_physical_trajectories": True,
        "total": total,
        "stages": stages,
      }
      json.dumps(result, allow_nan=False)
      return result
    finally:
      env.close()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--motion-file", type=Path, required=True)
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--seed", type=int, default=2026)
  parser.add_argument("--warmup", type=int, default=10)
  parser.add_argument("--samples", type=int, default=100)
  parser.add_argument("--cpu-threads", type=int, default=1)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  if args.cpu_threads < 1:
    parser.error("--cpu-threads must be positive")
  _validate_output_paths(
    inputs=(args.checkpoint, args.motion_file), outputs=(args.output,)
  )
  torch.set_num_threads(args.cpu_threads)
  result = benchmark(
    args.checkpoint,
    args.motion_file,
    device=args.device,
    seed=args.seed,
    warmup=args.warmup,
    samples=args.samples,
  )
  args.output.parent.mkdir(parents=True, exist_ok=True)
  with NamedTemporaryFile(mode="w", dir=args.output.parent, delete=False) as stream:
    temporary = Path(stream.name)
    try:
      json.dump(result, stream, indent=2, allow_nan=False)
      stream.write("\n")
      stream.flush()
      temporary.replace(args.output)
    finally:
      temporary.unlink(missing_ok=True)
  print(json.dumps(result["total"]["statistics"], indent=2, allow_nan=False))


if __name__ == "__main__":
  main()
