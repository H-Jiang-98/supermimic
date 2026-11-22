"""Audit Step 6 artifacts and plot comparable pilot learning curves.

A complete artifact audit is engineering evidence only. Formal experiment
budgets and the decision to enter Step 7 must be recorded separately.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any, Literal, overload

import torch

from supermimic.runners.eit_resume_contract import (
  RESUME_EXECUTION_FIELDS,
  SUPPORTED_REFERENCE_VERSIONS,
  resume_execution_matches,
)
from supermimic.runners.eit_snapshots import file_sha256
from supermimic.runners.eit_training import initialization_fingerprint
from supermimic.runners.run_config import EIT_INITIALIZATION_VERSION
from supermimic.scripts import batch_eval_eit
from supermimic.scripts.benchmark_eit_online import _summarize as summarize_latency
from supermimic.scripts.pilot_report_diagnostics import (
  diagnostic_structure_errors,
  supplementary,
)

GROUPS = "ABC"
PHASES = (
  "physical_collection_seconds",
  "mean_rollout_update_seconds",
  "residual_extraction_seconds",
  "covariance_fit_seconds",
  "posterior_prefix_normalization_seconds",
  "posterior_cache_normalization_seconds",
  "posterior_cache_read_seconds",
  "policy_rollout_backward_seconds",
  "critic_regression_seconds",
  "checkpoint_save_seconds",
  "round_seconds",
)
POLICY_CURVES = {
  "imagined_return": "Policy/return",
  "bootstrap_fraction": "Policy/bootstrap_fraction",
  "action_saturation": "Policy/action_saturation",
  "film_gamma": "Policy/film_gamma",
  "film_beta": "Policy/film_beta",
}


def _finite(value):
  return (
    isinstance(value, (float, int))
    and not isinstance(value, bool)
    and math.isfinite(value)
  )


def _reject_nonfinite(value):
  raise ValueError(f"Non-finite JSON value: {value}")


class Audit:
  def __init__(self, root):
    self.root = Path(root).resolve()
    self.pending, self.failures = [], []

  @overload
  def load(self, relative: str, *, lines: Literal[False] = False) -> dict[str, Any]: ...

  @overload
  def load(self, relative: str, *, lines: Literal[True]) -> list[dict[str, Any]]: ...

  def load(
    self, relative: str, *, lines: bool = False
  ) -> dict[str, Any] | list[dict[str, Any]]:
    path = self.root / relative
    if not path.is_file():
      self.pending.append(f"Missing {relative}")
      return [] if lines else {}
    try:
      text = path.read_text()
      value: Any = (
        [
          json.loads(line, parse_constant=_reject_nonfinite)
          for line in text.splitlines()
          if line.strip()
        ]
        if lines
        else json.loads(text, parse_constant=_reject_nonfinite)
      )
      if not isinstance(value, list if lines else dict):
        raise ValueError("Unexpected JSON root")
      if lines and any(not isinstance(row, dict) for row in value):
        raise ValueError("Expected JSON objects in each log row")
      return value
    except (OSError, ValueError) as error:
      self.failures.append(f"Unreadable {relative}: {error}")
      return [] if lines else {}

  def require(self, condition, message):
    if not condition:
      self.failures.append(message)

  def load_diagnostic(self, relative: str, *, mode: str = "all") -> dict[str, Any]:
    data = self.load(relative)
    if data:
      errors = diagnostic_structure_errors(
        data, mode=mode, require_gradients="actor_gradients" in data
      )
      if errors:
        self.failures.append(f"Invalid {relative}: {', '.join(errors)}")
        return {}
    return data

  def data(self, condition, message):
    if not condition:
      self.pending.append(message)

  def file(self, path, expected, label):
    if not path.is_file():
      self.pending.append(f"Missing {label}: {path}")
    elif not expected or file_sha256(path) != expected:
      self.failures.append(f"Digest mismatch: {label}")


def _coverage(bins):
  valid = [b for b in bins if b.get("frames", 0) > 0 and _finite(b.get("coverage_95"))]
  count = sum(b["frames"] for b in valid)
  return (
    sum(b["coverage_95"] * b["frames"] for b in valid) / count if count else None,
    count,
  )


def _curve(frames, value):
  return {"collected_frames": frames, "value": value if _finite(value) else None}


def _check_checkpoint_configuration(audit, state, p, label):
  runner = state.get("runner_config", {})
  expected = {
    "snapshot_interval": (state.get("snapshot_interval", 1), 1),
    "num_envs": (state.get("num_envs"), p["num_envs"]),
    "buffer_size": (runner.get("buffer_size"), p["buffer_size"]),
    "sampling_mode": (runner.get("sampling_mode"), p["sampling_mode"]),
    "collection_steps": (
      state.get("loop_config", {}).get("collection_steps"),
      p["collection_steps"],
    ),
  }
  for name, key in (
    ("mean", "dynamics_hidden_dims"),
    ("actor", "actor_hidden_dims"),
    ("critic", "critic_hidden_dims"),
  ):
    expected[f"widths.{name}"] = (runner.get(key), p["widths"][name])
  expected["widths.noise"] = (
    runner.get("process_noise", {}).get("hidden_dims"),
    p["widths"]["noise"],
  )
  for name, configured, budget in (
    ("mean", runner.get("mean", {}), p["mean"]),
    ("covariance", runner.get("cov", {}), p["covariance"]),
    ("policy", state.get("policy_config", {}), p["policy"]),
  ):
    expected.update(
      {f"{name}.{key}": (configured.get(key), value) for key, value in budget.items()}
    )
  for key, (actual, desired) in expected.items():
    if isinstance(actual, tuple):
      actual = list(actual)
    audit.require(
      actual == desired,
      f"{label}: checkpoint {key} differs from pilot protocol ({actual!r} != {desired!r})",
    )


def _initialization_identity(audit, p):
  path = audit.root / "initialization.pt"
  if not path.is_file():
    audit.pending.append("Missing initialization.pt")
    return None
  try:
    bundle = torch.load(path, weights_only=True, mmap=True, map_location="cpu")
    audit.require(
      bundle.get("version") == EIT_INITIALIZATION_VERSION
      and bundle.get("generation_seed") == p["training_seed"],
      "Initialization package version/seed differs from pilot protocol",
    )
    identity = initialization_fingerprint(bundle)
    for group in GROUPS:
      baseline = audit.root / group / "snapshots" / "frames_0.pt"
      if not baseline.is_file():
        continue
      state = torch.load(baseline, weights_only=True, mmap=True, map_location="cpu")
      _check_checkpoint_configuration(audit, state, p, f"{group}/0")
      audit.require(
        state.get("initialization_digest") == identity,
        f"{group}: initialization metadata differs from actual package",
      )
      for saved, original in (
        (state["world_model"], bundle["world_model"]),
        (state["actor"], bundle["actors"][group]),
        (state["critic"], bundle["critics"][group]),
      ):
        audit.require(
          initialization_fingerprint(saved, include_model_revision=False)
          == initialization_fingerprint(original, include_model_revision=False),
          f"{group}: baseline weights differ from actual common initialization",
        )
      final_frames = p["rounds"] * p["num_envs"] * p["collection_steps"]
      final = baseline.with_name(f"frames_{final_frames}.pt")
      if final.is_file():
        state = torch.load(final, weights_only=True, mmap=True, map_location="cpu")
        _check_checkpoint_configuration(audit, state, p, f"{group}/{final_frames}")
    return identity
  except (
    OSError,
    ValueError,
    RuntimeError,
    KeyError,
    TypeError,
    EOFError,
    pickle.UnpicklingError,
  ) as error:
    audit.failures.append(f"Cannot verify initialization package: {error}")
    return None


def _verify_diagnostic_record(
  audit,
  path,
  p,
  group,
  sha,
  evaluated,
  offline,
  *,
  actor_source="posterior",
  posterior=True,
):
  if not path.is_file():
    return
  try:
    record = torch.load(path, weights_only=True, mmap=True, map_location="cpu")
    expected = {
      "version": "eit_diagnostic_episodes_v1",
      "source": "mujoco_evaluation",
      "split": "diagnostic",
      "checkpoint_sha256": sha,
      "group": group,
      "actor_state_source": actor_source,
      "seed": p["diagnostic_seed"],
      "motion_sha256": p["motion_sha256"],
    }
    audit.require(
      all(record.get(k) == v for k, v in expected.items()),
      f"{group}: independent recording header differs from protocol/snapshot",
    )
    audit.require(
      offline.get("recording_checkpoint_sha256") == sha
      and offline.get("actor_state_source") == actor_source
      and offline.get("motion_digest") == record.get("motion_digest")
      and evaluated.get("evaluation_motion_sha256") == p["motion_sha256"]
      and evaluated.get("seed") == p["diagnostic_seed"]
      and evaluated.get("actor_state_source") == actor_source,
      f"{group}: independent scores refer to a different recording/scene/policy",
    )
    episodes = record["episodes"]
    audit.require(
      len(episodes) == p["evaluation_envs"],
      f"{group}: missing independent episode slots",
    )
    observed = [(ep["env_id"], len(ep["reward"]), ep["reason"]) for ep in episodes]
    scored = [
      (ep["env_id"], ep["length"], ep["reason"]) for ep in evaluated.get("episodes", [])
    ]
    audit.require(
      observed == scored,
      f"{group}: independent episode lengths/terminations differ between recording and evaluation",
    )
    frames = sum(len(ep["reward"]) for ep in episodes)
    if posterior:
      audit.require(
        sum(b.get("frames", 0) for b in offline.get("fixed_record_posterior", []))
        == frames,
        f"{group}: posterior calibration frame count differs from recording",
      )
    audit.require(
      offline.get("critic", {}).get("bootstrap_value_error", {}).get("samples")
      == frames,
      f"{group}: critic sample count differs from complete recording",
    )
  except (
    OSError,
    RuntimeError,
    ValueError,
    KeyError,
    TypeError,
    EOFError,
    pickle.UnpicklingError,
  ) as error:
    audit.failures.append(f"Cannot verify independent recording {path}: {error}")


def _validated_sources(audit, p, saved_validation):
  """Reopen checkpoints and raw task episodes; never trust aggregate success flags."""
  manifest = audit.load("manifest.json")
  if not manifest:
    return []
  unsigned = {k: v for k, v in manifest.items() if k != "manifest_digest"}
  audit.require(
    manifest.get("version") == batch_eval_eit.VERSION
    and manifest.get("manifest_digest") == batch_eval_eit.digest(unsigned),
    "Frozen manifest identity is invalid",
  )
  audit.require(
    saved_validation.get("manifest_digest") == manifest.get("manifest_digest"),
    "Validation aggregate belongs to a different manifest",
  )
  frames = [i * p["num_envs"] * p["collection_steps"] for i in range(p["rounds"] + 1)]
  expected_runs = [
    {
      "directory": str(audit.root / group),
      "group": group,
      "training_seed": p["training_seed"],
      "frames": frames,
    }
    for group in GROUPS
  ]
  audit.require(
    manifest.get("runs") == expected_runs, "Manifest runs differ from pilot protocol"
  )
  scenes = manifest.get("validation", [])
  audit.require(
    len(scenes) == 1
    and scenes[0].get("sha256") == p["motion_sha256"]
    and scenes[0].get("seed") == p["validation_seed"]
    and scenes[0].get("num_envs") == p["evaluation_envs"]
    and scenes[0].get("episodes") == p["evaluation_envs"]
    and scenes[0].get("protocol") == "full_clip_v1"
    and scenes[0].get("step_budget") is None,
    "Validation scenes differ from pilot motion, seed, episode count or full-clip protocol",
  )
  try:
    # Existing admission checks load actual tensors, optimizer counters and critic
    # contracts and compare them to each indexed immutable checkpoint.
    candidates = batch_eval_eit.candidates(manifest)
    records = []
    for candidate in candidates:
      if "error" in candidate:
        audit.pending.append(
          f"Unusable snapshot {candidate.get('group')}/{candidate.get('collected_frames')}: {candidate['error']}"
        )
      else:
        audit.require(
          all(
            candidate.get(key) == scene.get(key)
            for scene in scenes
            for key in ("initial_prior", "measurement_noise")
          ),
          f"{candidate['group']}/{candidate['collected_frames']}: snapshot/scene estimator conditions differ",
        )
      records.append(
        batch_eval_eit.evaluate_candidate(
          candidate, scenes, audit.root / "evaluation", run=False
        )
      )
    saved_rows = saved_validation.get("baselines", []) + saved_validation.get(
      "trained", []
    )

    def candidate_key(row):
      candidate = row["candidate"]
      return (
        candidate["group"],
        candidate["training_seed"],
        candidate["collected_frames"],
      )

    keys = [candidate_key(row) for row in saved_rows]
    expected_keys = {
      (group, p["training_seed"], frame) for group in GROUPS for frame in frames
    }
    audit.require(
      len(keys) == len(set(keys)),
      "Validation aggregate contains duplicate candidate keys",
    )
    audit.data(
      set(keys) >= expected_keys,
      "Validation aggregate is missing required candidate keys",
    )
    audit.require(
      set(keys) <= expected_keys,
      "Validation aggregate contains unexpected candidate keys",
    )
    audit.require(
      all(
        r["candidate"]["collected_frames"] == 0
        for r in saved_validation.get("baselines", [])
      )
      and all(
        r["candidate"]["collected_frames"] > 0
        for r in saved_validation.get("trained", [])
      ),
      "Validation aggregate baseline/trained candidate categories disagree",
    )
    stored = {candidate_key(row): row for row in saved_rows}
    for record in records:
      key = candidate_key(record)
      audit.data(
        record["complete"],
        f"{key[0]}/{key[2]}: raw validation tasks are missing, invalid or censored",
      )
      if record["complete"] and key in stored:
        audit.require(
          record == stored[key],
          f"{key[0]}/{key[2]}: validation aggregate differs from verified raw episodes",
        )
    if all(record["complete"] for record in records):
      audit.require(
        batch_eval_eit.select(records) == saved_validation.get("selected"),
        "Selected snapshots differ from verified validation ranking",
      )
    return records
  except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
    audit.failures.append(f"Cannot validate source artifacts: {error}")
    return []


def _audit_online(audit, online, group, p, checkpoint_sha, motion_digest):
  if not online:
    return {}
  runtime_digest = online.get("motion_digest")
  valid_digest = (
    isinstance(runtime_digest, str)
    and len(runtime_digest) == 64
    and all(c in "0123456789abcdef" for c in runtime_digest)
  )
  # MotionLoader preprocessing can differ numerically across CPU/CUDA. Raw file
  # identity is device independent; runtime tensor identity is not.
  same_device = online.get("device") == p["device"]
  identity_matches = (
    online.get("version") == "eit_online_latency_v1"
    and online.get("checkpoint_sha256") == checkpoint_sha
    and online.get("group") == group
    and online.get("batch_size") == 1
    and online.get("posterior_diagnostics") is False
    and online.get("device") == p["online_device"]
    and online.get("motion_sha256") == p["motion_sha256"]
    and valid_digest
    and (not same_device or motion_digest is None or runtime_digest == motion_digest)
    and online.get("cpu_threads") == p["cpu_threads"]
    and online.get("seed") == p.get("online_seed", 2026)
  )
  if not identity_matches:
    audit.failures.append(
      f"{group}: online benchmark identity or configuration mismatch"
    )
    return {}
  period, measured, warmup = (
    online.get(k) for k in ("control_period_ms", "measured_steps", "warmup_steps")
  )
  valid_metadata = (
    _finite(period)
    and period == 20.0
    and type(measured) is int
    and measured > 0
    and type(warmup) is int
    and warmup >= 0
    and type(online.get("cpu_interop_threads")) is int
    and online["cpu_interop_threads"] > 0
    and online.get("dtype") == "torch.float32"
  )
  audit.data(valid_metadata, f"{group}: missing online timing metadata")
  if not valid_metadata:
    return {}
  recomputed = {}
  for name in ("total", "stages"):
    evidence = online.get(name)
    if not isinstance(evidence, dict):
      audit.pending.append(f"{group}/{name}: missing online timing pass")
      continue
    rows = evidence.get("samples")
    audit.data(
      isinstance(rows, list) and len(rows) == measured,
      f"{group}/{name}: missing online raw timing samples",
    )
    reset = evidence.get("first_reset", {})
    if not isinstance(reset, dict):
      reset = {}
    audit.data(
      all(
        _finite(reset.get(key)) and reset[key] >= 0
        for key in ("environment_reset", "adapter_reset", "first_act")
      ),
      f"{group}/{name}: missing first-reset timings",
    )
    required = {"total"}
    if name == "stages":
      required |= {
        "measurement_history",
        "filter_and_jacobian",
        "features",
        "actor",
        "clamp_and_commit",
      }
    if not isinstance(rows, list) or not rows:
      continue
    try:
      valid_rows = [row["sample"] for row in rows] == list(range(measured)) and all(
        type(row["reset"]) is bool
        and type(row["episode_step"]) is int
        and row["episode_step"] >= 0
        and row["reset"] == (row["episode_step"] == 0)
        and set(row["timings_ms"]) == required
        and all(_finite(value) and value >= 0 for value in row["timings_ms"].values())
        for row in rows
      )
      audit.require(
        valid_rows,
        f"{group}/{name}: invalid sample sequence, reset classification or phase timings",
      )
      if not valid_rows:
        continue
      statistics = summarize_latency(rows, period)
      audit.require(
        evidence.get("statistics") == statistics,
        f"{group}/{name}: online statistics differ from raw samples",
      )
      audit.require(
        evidence.get("sample_reset_count") == sum(row["reset"] for row in rows),
        f"{group}/{name}: online reset count differs from raw samples",
      )
      recomputed[name] = statistics
    except (KeyError, TypeError, ValueError) as error:
      audit.failures.append(f"{group}/{name}: unreadable online samples: {error}")
  latency = (
    recomputed.get("total", {}).get("continuous_full_filter", {}).get("total", {})
  )
  audit.data(
    latency.get("count", 0) > 0 and _finite(latency.get("p99_ms")),
    f"{group}: no continuous full-filter online latency samples",
  )
  return latency


def _audit_group(audit, group, p, validation, commands):
  run = audit.root / group
  count = p["rounds"]
  frames_per_round = p["num_envs"] * p["collection_steps"]
  expected_frames = [i * frames_per_round for i in range(count + 1)]
  expected_rounds = list(range(1, count + 1))
  metrics = audit.load(f"{group}/metrics.jsonl", lines=True)
  profiles = audit.load(f"{group}/profiling.jsonl", lines=True)
  index = audit.load(f"{group}/snapshots/index.json")
  snapshots = index.get("snapshots", [])
  by_frame = {row["collected_frames"]: row for row in snapshots}
  audit.data(
    sorted(by_frame) == expected_frames,
    f"{group}: expected baseline and {count} trained snapshots",
  )
  audit.require(
    len(by_frame) == len(snapshots), f"{group}: duplicate snapshot frame keys"
  )
  initializations = set()
  for row in snapshots:
    frames = row["collected_frames"]
    audit.file(run / row["path"], row.get("sha256"), f"{group} snapshot {frames}")
    audit.require(
      row.get("stage") == "policy"
      and row.get("group") == group
      and row.get("training_seed") == p["training_seed"],
      f"{group}/{frames}: wrong stage, group or training seed",
    )
    initial = row.get("initialization_digest")
    audit.data(
      isinstance(initial, str) and len(initial) == 64,
      f"{group}/{frames}: missing initialization fingerprint",
    )
    if initial:
      initializations.add(initial)
    audit.require(
      row.get("round") == frames // frames_per_round,
      f"{group}/{frames}: inconsistent round",
    )
  audit.data(
    sorted(row.get("round", -1) for row in metrics) == expected_rounds,
    f"{group}: missing or duplicate training rounds",
  )
  sums = Counter()
  curves = {
    key: []
    for key in ("real_return", "progress", "coverage_95", "critic_rmse", *POLICY_CURVES)
  }
  budget_rows = []
  for row in sorted(metrics, key=lambda r: r.get("round", -1)):
    number, frames, b = (
      row.get("round"),
      row.get("collected_frames"),
      row.get("budget", {}),
    )
    if not b:
      audit.pending.append(f"{group}/{number}: missing budget counters")
      continue
    label = f"{group} round {number}"
    audit.require(
      frames == number * frames_per_round, f"{label}: cumulative frame budget mismatch"
    )
    expected = {
      "collected_frames": frames_per_round,
      "mean_updates": p["mean"]["num_batches"],
      "sampled_training_windows": p["mean"]["num_batches"] * p["mean"]["batch_size"],
      "actor_updates": p["policy"]["num_batches"],
      "critic_updates": p["policy"]["num_batches"] * p["policy"]["critic_epochs"],
      "covariance_batch_size": p["covariance"]["batch_size"],
      "covariance_epochs": p["covariance"]["epochs"],
      "policy_n": p["policy"]["batch_size"],
      "policy_k": p["policy"]["num_samples"],
      "policy_h": p["policy"]["horizon"],
      "policy_microbatch": p["policy"]["microbatch_size"],
    }
    for key, value in expected.items():
      audit.require(
        b.get(key) == value, f"{label}: {key} expected {value}, got {b.get(key)}"
      )
    for key in ("available_training_windows", "training_pairs", "validation_pairs"):
      audit.data(b.get(key, 0) > 0, f"{label}: no usable {key}")
    pairs = b.get("training_pairs", 0)
    updates = p["covariance"]["epochs"] * math.ceil(
      pairs / p["covariance"]["batch_size"]
    )
    audit.require(
      b.get("covariance_updates") == updates,
      f"{label}: covariance update rule mismatch",
    )
    audit.require(
      b.get("covariance_pair_draws") == pairs * p["covariance"]["epochs"],
      f"{label}: covariance pair draws mismatch",
    )
    maximum = (
      p["policy"]["num_batches"]
      * p["policy"]["batch_size"]
      * p["policy"]["num_samples"]
      * p["policy"]["horizon"]
    )
    audit.require(
      0 < b.get("imagined_steps", 0) <= maximum, f"{label}: invalid imagined step count"
    )
    for key in (
      "collected_frames",
      "mean_updates",
      "covariance_updates",
      "actor_updates",
      "critic_updates",
      "training_pairs",
      "validation_pairs",
      "imagined_steps",
      "survival_weighted_imagined_steps",
      "physical_terminations",
      "physical_episode_endings",
    ):
      if key not in b:
        audit.pending.append(f"{label}: missing {key}")
      else:
        sums[key] += b[key]
    for key, counter in (
      ("mean_updates", "mean_steps"),
      ("covariance_updates", "cov_steps"),
      ("actor_updates", "actor_steps"),
      ("critic_updates", "critic_steps"),
    ):
      if frames in by_frame:
        audit.require(
          by_frame[frames].get(counter) == sums[key],
          f"{label}: snapshot {counter} differs from actual updates",
        )
    for name, key in POLICY_CURVES.items():
      curves[name].append(_curve(frames, row.get("policy", {}).get(key)))
    budget_rows.append({"round": number, "collected_frames": frames, **b})
  for key, value in p["expected_per_group"].items():
    audit.data(
      sums[key] == value, f"{group}: total {key} expected {value}, observed {sums[key]}"
    )
  audit.data(
    sorted(row.get("round", -1) for row in profiles) == expected_rounds,
    f"{group}: missing or duplicate profiling rounds",
  )
  for row in profiles:
    for phase in PHASES:
      seconds = row.get("seconds", {})
      # Prefix rebuilding and incremental cache reading are distinct protocols.
      if (
        phase == "posterior_prefix_normalization_seconds"
        and "posterior_cache_normalization_seconds" in seconds
      ):
        continue
      if (
        phase
        in ("posterior_cache_normalization_seconds", "posterior_cache_read_seconds")
        and phase not in seconds
      ):
        continue
      value = seconds.get(phase)
      audit.data(
        _finite(value) and value >= 0,
        f"{group}/{row.get('round')}: missing timing {phase}",
      )
  phase_summary = {
    key: {
      "rounds": len(values),
      "mean_seconds": fmean(values) if values else None,
      "total_seconds": sum(values) if values else None,
    }
    for key in PHASES
    for values in [
      [
        row["seconds"][key]
        for row in profiles
        if _finite(row.get("seconds", {}).get(key))
      ]
    ]
  }
  resume = audit.load(f"resume/{group}.json")
  mid = by_frame.get(p["resume_after_round"] * frames_per_round, {})
  audit.data(
    resume.get("version") == "eit_fixed_input_resume_v2"
    and resume.get("status") == "passed"
    and resume.get("live_optimizer_reference") is True
    and resume.get("expected_source") == "live_training_before_checkpoint_commit"
    and resume.get("reference_version") in SUPPORTED_REFERENCE_VERSIONS
    and resume.get("fixed_input_policy_resume_exact") is True
    and resume.get("fixed_input_model_resume_exact") is True,
    f"{group}: fixed-input next-update restore verification pending",
  )
  if resume:
    audit.require(
      resume.get("checkpoint_sha256") == mid.get("sha256")
      and resume.get("round") == p["resume_after_round"]
      and resume.get("group") == group
      and resume.get("device") == p["device"],
      f"{group}: restore result identity mismatch",
    )
    reference_path = resume.get("reference_path")
    if not isinstance(reference_path, str) or not reference_path:
      audit.pending.append(f"{group}: missing live pre-checkpoint update reference")
    else:
      reference = Path(reference_path)
      audit.file(
        reference,
        resume.get("reference_sha256"),
        f"{group} live pre-checkpoint update reference",
      )
      if reference.is_file():
        try:
          live = torch.load(reference, weights_only=True, mmap=True, map_location="cpu")
          if not isinstance(live, dict):
            raise ValueError("Live update reference must be a dictionary")
          audit.require(
            live.get("version") in SUPPORTED_REFERENCE_VERSIONS
            and live["version"] == resume.get("reference_version")
            and live.get("source") == "live_training_before_checkpoint_commit"
            and live.get("checkpoint_sha256") == mid.get("sha256")
            and live.get("round") == p["resume_after_round"]
            and live.get("group") == group
            and live.get("device") == resume.get("device"),
            f"{group}: live reference version/source/checkpoint identity mismatch",
          )
          have_execution = all(
            isinstance(record.get("verification_execution"), dict)
            and RESUME_EXECUTION_FIELDS.issubset(record["verification_execution"])
            for record in (live, resume)
          )
          audit.data(
            have_execution,
            f"{group}: privately warmed fixed-input restore execution evidence pending",
          )
          if have_execution:
            audit.require(
              resume_execution_matches(
                resume, live, p.get("training_process_environment")
              ),
              f"{group}: restore execution contract differs from live reference or pilot protocol",
            )
        except (
          OSError,
          ValueError,
          RuntimeError,
          KeyError,
          TypeError,
          EOFError,
          pickle.UnpicklingError,
        ) as error:
          audit.failures.append(f"{group}: unreadable live update reference: {error}")
  continued = any(
    c.get("name") == f"train-{group}-to-{count}"
    and c.get("returncode") == 0
    and "--resume" in c.get("argv", [])
    for c in commands
  )
  audit.data(
    continued, f"{group}: no successful fresh-physics resumed training command"
  )
  records = [r for r in validation if r.get("candidate", {}).get("group") == group]
  audit.data(
    sorted(r["candidate"]["collected_frames"] for r in records) == expected_frames,
    f"{group}: missing validation candidates",
  )
  for row in records:
    candidate, tasks = row["candidate"], row.get("tasks", [])
    frames = candidate["collected_frames"]
    audit.require(
      candidate.get("sha256") == by_frame.get(frames, {}).get("sha256"),
      f"{group}/{frames}: validation snapshot mismatch",
    )
    audit.data(
      row.get("complete") is True
      and bool(tasks)
      and all(t.get("status") == "complete" for t in tasks)
      and row.get("expected_episodes") == row.get("recorded_episodes"),
      f"{group}/{frames}: validation missing, failed or censored",
    )
    audit.data(
      row.get("scores", {}).get("success_rate") is not None,
      f"{group}/{frames}: validation completion score unavailable",
    )
    curves["real_return"].append(_curve(frames, row.get("scores", {}).get("return")))
    curves["progress"].append(_curve(frames, row.get("scores", {}).get("progress")))
  diagnostics = []
  diagnostic_motion_digest: str | None = None
  for frames in expected_frames:
    folder = f"diagnostics/{group}/{frames}"
    evaluated, offline = (
      audit.load(f"{folder}/evaluation.json"),
      audit.load_diagnostic(f"{folder}/offline.json"),
    )
    if frames == expected_frames[-1]:
      diagnostic_motion_digest = offline.get("motion_digest")
    expected_sha = by_frame.get(frames, {}).get("sha256")
    for name, data in (("evaluation", evaluated), ("offline", offline)):
      if data:
        audit.require(
          data.get("checkpoint_sha256") == expected_sha and data.get("group") == group,
          f"{group}/{frames}: {name} identity mismatch",
        )
    audit.data(
      evaluated.get("evaluation_complete") is True
      and evaluated.get("censored_episodes") == 0
      and evaluated.get("timeout_count") == 0
      and evaluated.get("protocol") == "full_clip_v1"
      and evaluated.get("step_budget") is None,
      f"{group}/{frames}: independent episodes incomplete or censored",
    )
    if offline:
      audit.require(
        offline.get("training_overlap_check", {}).get("status") == "distinct_seed"
        and offline.get("recording_seed") == p["diagnostic_seed"]
        and offline.get("recording_source") == "mujoco_evaluation",
        f"{group}/{frames}: diagnostics independence unverified",
      )
      audit.file(
        audit.root / folder / "episodes.pt",
        offline.get("recording_sha256"),
        f"{group}/{frames} independent recording",
      )
      _verify_diagnostic_record(
        audit,
        audit.root / folder / "episodes.pt",
        p,
        group,
        expected_sha,
        evaluated,
        offline,
      )
    coverage, coverage_frames = _coverage(offline.get("fixed_record_posterior", []))
    critic = offline.get("critic", {})
    error = critic.get("bootstrap_value_error", {})
    audit.data(
      coverage_frames > 0,
      f"{group}/{frames}: no independent posterior calibration frames",
    )
    audit.data(
      error.get("samples", 0) > 0
      and _finite(error.get("rmse"))
      and critic.get("censored_episodes") == 0,
      f"{group}/{frames}: independent critic score missing or censored",
    )
    audit.data(
      offline.get("process", {}).get("transitions", 0) > 0,
      f"{group}/{frames}: no independent process transitions",
    )
    curves["coverage_95"].append(_curve(frames, coverage))
    curves["critic_rmse"].append(_curve(frames, error.get("rmse")))
    diagnostics.append(
      {
        "collected_frames": frames,
        "evaluation": str(audit.root / folder / "evaluation.json"),
        "offline": str(audit.root / folder / "offline.json"),
        "coverage_95": coverage,
        "coverage_frames": coverage_frames,
        "critic_error": error,
        "process_transitions": offline.get("process", {}).get("transitions"),
      }
    )
  online = audit.load(f"online/{group}.json")
  latency = _audit_online(
    audit,
    online,
    group,
    p,
    by_frame.get(expected_frames[-1], {}).get("sha256"),
    diagnostic_motion_digest,
  )
  complete_budget = sorted(row.get("round", -1) for row in metrics) == expected_rounds
  return {
    "actual_budget": {
      key: sums.get(key)
      if complete_budget and all(key in row.get("budget", {}) for row in metrics)
      else None
      for key in sorted(set(sums) | set(p["expected_per_group"]))
    },
    "observed_budget_subtotals": dict(sums),
    "budget_rounds_observed": len(metrics),
    "round_budgets": budget_rows,
    "initialization_digests": sorted(initializations),
    "final_snapshot_sha256": by_frame.get(expected_frames[-1], {}).get("sha256"),
    "snapshots": str(run / "snapshots/index.json"),
    "metrics": str(run / "metrics.jsonl"),
    "profiling": str(run / "profiling.jsonl"),
    "phase_costs": phase_summary,
    "runtime": profiles[-1].get("runtime") if profiles else None,
    "runtime_after_save": profiles[-1].get("runtime_after_save") if profiles else None,
    "max_cuda_allocated_bytes": max(
      (
        r.get(scope, {}).get("cuda_peak_allocated_bytes") or 0
        for r in profiles
        for scope in ("runtime", "runtime_after_save")
      ),
      default=0,
    )
    or None,
    "max_cuda_allocated_bytes_scope": "Maximum across all rounds, before and after checkpoint save. Older rows without runtime_after_save provide before-save evidence only.",
    "restore": str(audit.root / "resume" / f"{group}.json"),
    "restore_reference": {
      key: resume.get(key)
      for key in (
        "reference_path",
        "reference_sha256",
        "reference_version",
        "expected_source",
        "verification_execution",
      )
    },
    "fresh_physics_continuation_verified": continued,
    "diagnostics": diagnostics,
    "online": str(audit.root / "online" / f"{group}.json"),
    "online_continuous_full_filter": latency,
    "online_memory_scope": online.get(
      "memory_scope",
      "Legacy benchmark: process-wide PyTorch allocator included the loaded training checkpoint and optimizer tensors as well as inference and simulation tensors; not a deployment-only footprint.",
    ),
    "curves": {
      k: sorted(v, key=lambda point: point["collected_frames"])
      for k, v in curves.items()
    },
  }


def build_report(root):
  audit = Audit(root)
  p = audit.load("protocol.json")
  if not p:
    return {
      "version": "eit_pilot_report_v1",
      "status": "pending",
      "pending": audit.pending,
      "failures": audit.failures,
      "step7_ready": False,
      "groups": {},
    }
  validation = audit.load("evaluation/validation.json")
  commands = audit.load("commands.jsonl", lines=True)
  audit.data(
    validation.get("complete") is True and validation.get("pending_validation") == [],
    "Validation aggregate is incomplete",
  )
  records = _validated_sources(audit, p, validation)
  initialization_identity = _initialization_identity(audit, p)
  groups = {group: _audit_group(audit, group, p, records, commands) for group in GROUPS}
  extra = supplementary(audit, p, groups, _verify_diagnostic_record, _coverage)
  initializations = {
    digest for group in groups.values() for digest in group["initialization_digests"]
  }
  audit.require(
    len(initializations) <= 1, "A/B/C do not share one initialization fingerprint"
  )
  return {
    "version": "eit_pilot_report_v1",
    "status": "failed"
    if audit.failures
    else "pending"
    if audit.pending
    else "complete",
    "step7_ready": False,
    "step7_decision": "Formal uniform budget must be locked separately after reviewing learning curves, failure modes and measured costs.",
    "protocol": str(audit.root / "protocol.json"),
    "initialization_fingerprint": initialization_identity,
    "validation": str(audit.root / "evaluation/validation.json"),
    "selected": validation.get("selected", []),
    "pending": audit.pending,
    "failures": audit.failures,
    "groups": groups,
    "supplementary": extra,
    "calibration_curve_weighting": "Within a snapshot, posterior bins weighted by recorded frames; time-resolved and survival counts remain in offline.json.",
    "return_comparison": "Real return is undiscounted physical episode reward; imagined return is the short-horizon discounted objective including bootstrap. Their levels are not directly interchangeable.",
  }


def plot_report(report, output):
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  labels = (
    ("real_return", "Physical episode return", "undiscounted reward"),
    ("progress", "Physical clip progress", "fraction of reference"),
    ("imagined_return", "Imagined actor objective", "discounted reward + bootstrap"),
    (
      "bootstrap_fraction",
      "Bootstrap numerical fraction",
      "abs(bootstrap) / abs(return)",
    ),
    ("coverage_95", "Independent posterior calibration", "95% RPV ellipsoid coverage"),
    ("action_saturation", "Imagined action clipping", "fraction"),
    ("film_gamma", "FiLM scale", "mean abs(gamma)"),
    ("film_beta", "FiLM shift", "mean abs(beta)"),
    ("critic_rmse", "Independent critic error", "RMSE against hard-terminal return"),
  )
  fig, axes = plt.subplots(3, 3, figsize=(15, 11), constrained_layout=True)
  colors = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c"}
  for ax, (key, title, ylabel) in zip(axes.flat, labels, strict=True):
    for group, summary in report["groups"].items():
      points = summary["curves"][key]
      if points:
        ax.plot(
          [p["collected_frames"] for p in points],
          [float("nan") if p["value"] is None else p["value"] for p in points],
          marker="o",
          markersize=3,
          label=group,
          color=colors[group],
        )
    if key == "coverage_95":
      ax.axhline(0.95, color="gray", linestyle="--", linewidth=1)
      ax.set_ylim(-0.02, 1.02)
    ax.set(title=title, xlabel="Real collected frames", ylabel=ylabel)
    ax.grid(alpha=0.25)
    if ax.get_legend_handles_labels()[0]:
      ax.legend()
  fig.suptitle(f"Step 6 single-seed pilot — artifact audit: {report['status']}")
  output = Path(output)
  output.mkdir(parents=True, exist_ok=True)
  for suffix in ("png", "pdf"):
    fig.savefig(output / f"learning_curves.{suffix}", dpi=180)
  plt.close(fig)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--pilot", type=Path, required=True)
  parser.add_argument("--output", type=Path, help="Defaults to PILOT/report")
  parser.add_argument("--no-plots", action="store_true")
  args = parser.parse_args()
  output = args.output or args.pilot / "report"
  report = build_report(args.pilot)
  output.mkdir(parents=True, exist_ok=True)
  (output / "report.json").write_text(
    json.dumps(report, indent=2, allow_nan=False) + "\n"
  )
  if not args.no_plots:
    plot_report(report, output)
  print(json.dumps({k: report[k] for k in ("status", "pending", "failures")}, indent=2))
  if report["status"] != "complete":
    raise SystemExit(1)


if __name__ == "__main__":
  main()
