"""Full-clip tracking evaluation and result persistence."""

from __future__ import annotations

import json
from dataclasses import replace
from time import monotonic

import torch

from supermimic.validation.batched import TrackingRuntime, batches
from supermimic.validation.config import atomic_json


def batch_capacity(group, maximum):
  count = sum(len(t[3]) for t in group)
  return next(
    (n for n in sorted({64, 128, 256, 512, 640, maximum}) if count <= n <= maximum),
    maximum,
  )


class RuntimePool:
  def __init__(self, cfg, manifest, saved):
    self.cfg, self.manifest, self.saved = cfg, manifest, saved
    self.runtimes = {}

  def get(self, group):
    capacity = batch_capacity(group, self.cfg.num_envs)
    if capacity not in self.runtimes:
      self.runtimes[capacity] = TrackingRuntime(
        replace(self.cfg, num_envs=capacity),
        self.manifest["motions"],
        self.manifest["domains"],
        self.saved,
      )
    return self.runtimes[capacity]

  def close(self):
    for runtime in self.runtimes.values():
      runtime.close()
    self.runtimes.clear()

  def warm(self, groups, variant):
    seen, results = set(), []
    for group in groups:
      runtime = self.get(group)
      if runtime.cfg.num_envs in seen:
        continue
      seen.add(runtime.cfg.num_envs)
      started = monotonic()
      p = runtime.prepare(group, variant, limit=max(2, self.cfg.warmup_steps))
      initialization_s = monotonic() - started
      started = monotonic()
      correctness = (
        runtime.check_compile(p)
        if self.cfg.compile_kernels
        else {"passed": False, "reason": "eager debug run"}
      )
      if self.cfg.compile_kernels:
        torch.cuda.synchronize(self.cfg.device)
      compile_s = monotonic() - started
      started = monotonic()
      runtime.execute(group, variant, p)
      warmup_s = monotonic() - started
      results.append(
        dict(
          capacity=runtime.cfg.num_envs,
          correctness=correctness,
          initialization_s=initialization_s,
          compile_cold_s=compile_s,
          warmup_s=warmup_s,
        )
      )
    return results


def run_tracking(cfg, manifest, saved, work, output):
  from supermimic.validation.study import load_task, save_task

  if not cfg.max_frames:
    # Allocate one slot per clip/context pair for full-clip evaluation.
    cfg = replace(cfg, num_envs=len(manifest["motions"]) * cfg.num_domain_contexts)
  pool = RuntimePool(cfg, manifest, saved)
  timing_path = output / "tracking_timing.json"
  timings = json.loads(timing_path.read_text()) if timing_path.exists() else {}
  try:
    for variant in cfg.methods:
      pending = [
        t
        for t in work
        if t[2] == variant
        and (
          (cached := load_task(output, t[0], manifest, fields=())) is None
          or cached[0]["status"] not in ("complete", "censored")
          or cached[0]["odometry_status"] not in ("complete", "censored")
        )
      ]
      if not pending:
        continue
      groups = list(batches(pending, cfg.num_envs))
      warm = pool.warm(groups, variant)
      started = monotonic()
      valid_frames = 0
      for group in groups:
        runtime = pool.get(group)
        p = runtime.prepare(group, variant, limit=cfg.max_frames)
        outputs, timing = runtime.execute(group, variant, p)
        valid_frames += timing["effective_frames"]
        by_key = {t[0]: t for t in group}
        for key, arrays, metadata in outputs:
          _, motion, _, ids = by_key[key]
          save_task(
            output,
            key,
            arrays,
            dict(
              **metadata,
              identity=manifest["identity"],
              clip=motion["name"],
              variant=variant,
              context_ids=ids,
              device=cfg.device,
            ),
          )
        del p, outputs
      elapsed = monotonic() - started
      timings[variant] = dict(
        evaluation_wall_s=elapsed,
        warmup_by_capacity=warm,
        effective_frames=valid_frames,
        capacity=cfg.num_envs,
        clips=len(pending),
        resumed_subset=len(pending) != len(manifest["motions"]),
      )
      atomic_json(output / "tracking_timing.json", timings)
  finally:
    pool.close()
