"""Evaluate the paper's five nominal tracking modes on paired motion and noise."""

# PyTorch 2.14 dynamically re-exports public tensor factories and operations.
# pyright: reportPrivateImportUsage=false

import argparse
import json
from collections import Counter
from pathlib import Path
from time import perf_counter

from filelock import FileLock

from supermimic.validation.config import atomic_json
from supermimic.validation.nominal import (
  ROUTES,
  atomic_npz,
  batches,
  manifest_for,
  read_controls,
  read_episode,
  read_settings,
  save_episode,
  scenario_groups,
  tracking_digest,
)


def episode_ready(output, manifest, scenario, planned, record, audit):
  from supermimic.validation.nominal_posterior import MODE, load_posterior

  track = read_episode(scenario.path(output), manifest["identity"], planned)
  if track is None:
    return False
  if audit:
    try:
      read_controls(scenario.path(output / "controls"), track[1])
    except ValueError:
      return False
  if record and scenario.mode == MODE:
    try:
      load_posterior(output, manifest, scenario, track[0], track[1])
    except ValueError:
      return False
  return True


def run(args, paths, manifest, saved, device, capacity, output):
  import torch

  from supermimic.validation.nominal_runtime import NominalRuntime

  plan = {m["name"]: m["steps"] for m in manifest["motions"]}
  full = list(batches(list(scenario_groups(manifest)), capacity, args.layout))
  slots = max(map(len, full))
  layout = dict(layout=args.layout, capacity=capacity, allocated_slots=slots)
  layout_path = output / "layout.json"
  for path in (layout_path, output / "timing.json"):
    if path.exists():
      prior = json.loads(path.read_text())
      if any(prior[k] != v for k, v in layout.items()):
        raise ValueError(f"Resume requires the original layout {path}")
  atomic_json(layout_path, layout)
  timing_path = output / "timing.json"
  old_timings = (
    json.loads(timing_path.read_text())["batches"] if timing_path.exists() else []
  )
  timings = {b.get("batch_index", i): b for i, b in enumerate(old_timings, 1)}
  queue = [
    (i, batch)
    for i, batch in enumerate(full, 1)
    if (output / f".batch-{i}.pending").exists()
    or i not in timings
    or not all(
      episode_ready(
        output,
        manifest,
        s,
        plan[s.clip],
        args.record_belief_posterior,
        args.audit_controls,
      )
      for s in batch
    )
  ]
  if not queue:
    return
  torch.set_num_threads(1)
  if device.startswith("cuda"):
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
  started = perf_counter()
  # MuJoCo pools contact candidates across worlds. Different seeds of a
  # single clip can hit the floor together, particularly in sequential runs.
  largest_cohort = max(max(Counter(s.clip for s in batch).values()) for batch in full)
  runtime = NominalRuntime(
    paths, manifest, saved, device, slots, largest_cohort, not args.no_compile
  )
  initialization = perf_counter() - started

  audit_digests = {}

  def persist(scenario, status, planned, frames, terms):
    if args.audit_controls:
      audit_digests[scenario] = tracking_digest(frames)
    save_episode(
      scenario.path(output), manifest["identity"], status, planned, frames, terms
    )

  def persist_posterior(scenario, status, frames, dt, fields):
    from supermimic.validation.nominal_posterior import save_posterior

    save_posterior(output, manifest, scenario, status, frames, dt, fields)

  def audit(scenario, actions, noise):
    atomic_npz(
      scenario.path(output / "controls"),
      actions=actions,
      noise=noise,
      tracking_sha256=audit_digests.pop(scenario),
    )

  try:
    for i, batch in queue:
      # Old, individually valid files can survive a crash during a batch repair.
      # Mark the batch complete once all files and timing have been replaced.
      pending = output / f".batch-{i}.pending"
      pending.touch()
      started = perf_counter()
      warm = runtime.warmup_s
      steps, errors, write_s = runtime.run(
        batch,
        persist,
        persist_posterior if args.record_belief_posterior else None,
        audit if args.audit_controls else None,
      )
      elapsed = perf_counter() - started
      warm = runtime.warmup_s - warm
      timings[i] = dict(
        batch_index=i,
        episodes=len(batch),
        effective_steps=steps,
        elapsed_s=elapsed,
        filter_warmup_s=warm,
        write_s=write_s,
      )
      print(
        f"batch {i}/{len(full)}: {len(batch)} episodes, {steps} steps in {elapsed:.2f}s (filter warmup {warm:.2f}s)",
        flush=True,
      )
      atomic_json(
        output / "timing.json",
        dict(
          layout=args.layout,
          capacity=capacity,
          allocated_slots=slots,
          nconmax=runtime.raw.cfg.sim.nconmax,
          dt=getattr(runtime.raw, "step_dt", None),
          initialization_s=initialization,
          batches=[timings[j] for j in sorted(timings)],
          peak_cuda_bytes=torch.cuda.max_memory_allocated(device)
          if device.startswith("cuda")
          else 0,
          steady_steps_s=sum(b["effective_steps"] for b in timings.values())
          / sum(
            b["elapsed_s"] - b["filter_warmup_s"] - b["write_s"]
            for b in timings.values()
          ),
        ),
      )
      pending.unlink()
      if errors:
        raise RuntimeError(
          f"Nonfinite evaluation state in {errors}; inspect invalid episode artifacts"
        )
  finally:
    runtime.close()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "command",
    nargs="?",
    default="run",
    choices=("run", "report", "merge", "posterior-report", "compare-controls"),
  )
  parser.add_argument("--config", type=Path, default=Path("eval_eit_nominal.toml"))
  parser.add_argument("--output", type=Path)
  parser.add_argument("--device")
  parser.add_argument("--num-envs", type=int)
  parser.add_argument(
    "--clip", action="append", help="Select a configured clip by basename (repeatable)"
  )
  parser.add_argument(
    "--seed", type=int, action="append", help="Evaluation noise seed (repeatable)"
  )
  parser.add_argument("--layout", choices=("mixed", "sequential"))
  parser.add_argument("--record-belief-posterior", action="store_true")
  parser.add_argument(
    "--audit-controls",
    action="store_true",
    help="Save executed actions and shared noise for the small recording on/off comparison",
  )
  parser.add_argument(
    "--baseline", type=Path, help="First run directory for the offline merge"
  )
  parser.add_argument("--dry-run", action="store_true")
  parser.add_argument(
    "--no-compile",
    action="store_true",
    help="Use eager EKF kernels inside the CUDA graph",
  )
  args = parser.parse_args()
  if args.command == "compare-controls":
    from supermimic.validation.nominal_report import compare_controls

    if args.baseline is None or args.output is None:
      parser.error("compare-controls requires --baseline OFF and --output ON")
    compare_controls(args.baseline, args.output)
    return
  if args.command == "report":
    from supermimic.validation.nominal_report import report

    report(args.output or read_settings(args.config)[3])
    return
  if args.command in ("merge", "posterior-report"):
    output = args.output
    if output is None:
      first = read_settings(args.config)[3]
      output = first.with_name(first.name + "-posterior")
    if args.command == "merge":
      from supermimic.validation.nominal_report import merged_report

      merged_report(args.baseline or read_settings(args.config)[3], output)
    else:
      from supermimic.validation.nominal_posterior_report import posterior_report

      posterior_report(output)
    return
  checkpoints, paths, seeds, default_output, execution = read_settings(args.config)
  repeat_output = default_output.with_name(default_output.name + "-posterior")
  output = args.output or (
    repeat_output if args.record_belief_posterior else default_output
  )
  if args.record_belief_posterior and output.resolve() == default_output.resolve():
    parser.error("Use a separate output for the posterior repeat")
  if args.clip:
    selected = set(args.clip)
    paths = [p for p in paths if p.name in selected]
    if len(paths) != len(selected):
      parser.error("Unknown clip basename")
  if args.seed:
    seeds = args.seed
    if len(set(seeds)) != len(seeds):
      parser.error("Duplicate seed")
  layout = {}
  layout_paths = [output / "layout.json", output / "timing.json"]
  if args.record_belief_posterior:
    layout_paths.append(default_output / "timing.json")
  for path in layout_paths:
    if path.exists():
      layout = json.loads(path.read_text())
      break
  args.layout = args.layout or layout.get("layout", "mixed")
  capacity = (
    args.num_envs
    if args.num_envs is not None
    else layout.get("capacity", execution["num_envs"])
  )
  device = args.device or execution["device"]
  from supermimic.runners.eit_inference import evaluation_config, load_hashed_checkpoint

  saved, fingerprints = {}, {}
  for key, path in checkpoints.items():
    state, sha = load_hashed_checkpoint(path, device if not args.dry_run else "cpu")
    cfg = evaluation_config(state)
    inputs = cfg.estimator_inputs
    if inputs is None or inputs.group != key or not inputs.residual_actor:
      raise ValueError(f"{path}: expected {key} residual policy")
    saved[key] = state
    fingerprints[key] = dict(
      sha256=sha,
      training_seed=state["training_seed"],
      collected_steps=state["collected_steps"],
    )
  if (
    len({(v["training_seed"], v["collected_steps"]) for v in fingerprints.values()})
    != 1
  ):
    raise ValueError("Expected paired training seed and budget")
  configs = [s["runner_config"] for s in saved.values()]
  if any(
    c["initial_prior"] != configs[0]["initial_prior"]
    or c["estimator_inputs"]["measurement_noise"]
    != configs[0]["estimator_inputs"]["measurement_noise"]
    or c["history_length"] != configs[0]["history_length"]
    for c in configs[1:]
  ):
    raise ValueError("Checkpoints have incompatible nominal noise, prior or history")
  manifest = manifest_for(paths, seeds, fingerprints)
  groups = list(scenario_groups(manifest))
  queue = list(batches(groups, capacity, args.layout))
  print(
    f"{len(paths)} clips × {len(seeds)} seeds × 5 modes = {len(groups) * 5} episodes; {len(queue)} batches, capacity {capacity}",
    flush=True,
  )
  print(json.dumps(ROUTES, ensure_ascii=False), flush=True)
  if args.dry_run:
    return
  output.mkdir(parents=True, exist_ok=True)
  with FileLock(str(output / ".study.lock"), timeout=0):
    path = output / "manifest.json"
    if (
      path.exists() and json.loads(path.read_text())["identity"] != manifest["identity"]
    ):
      raise ValueError(
        "Output contains a different nominal study; select a separate output directory"
      )
    atomic_json(path, manifest)
    run(args, paths, manifest, saved, device, capacity, output)
    from supermimic.validation.nominal_report import report

    report(output)


if __name__ == "__main__":
  main()
