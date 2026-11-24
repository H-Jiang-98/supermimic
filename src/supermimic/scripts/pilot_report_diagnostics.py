"""Audit final gradient/critic and shared-trajectory evidence for the pilot report."""

from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import TypeGuard

import torch

from supermimic.runners.eit_snapshots import file_sha256


def _finite(value: object) -> TypeGuard[int | float]:
  return (
    isinstance(value, (float, int))
    and not isinstance(value, bool)
    and math.isfinite(value)
  )


def critic_score_status(critic):
  """Distinguish missing evidence from valid, entirely censored critic scores."""
  if not critic:
    return "missing"
  if critic["bootstrap_value_error"]["samples"] > 0:
    return "scored"
  return "unscorable_censored"


def diagnostic_structure_errors(
  result: object, *, mode: str = "all", require_gradients: bool = False
) -> list[str]:
  """Validate cached JSON without loading models or trusting its numeric summaries.

  Zero-pair/censored scores are valid structures with explicit null statistics;
  the artifact audit separately decides whether they satisfy pilot acceptance.
  Identity checks against checkpoints and raw recordings remain caller-owned.
  """
  errors = []

  def require(condition, path):
    if not condition and not errors:
      errors.append(path)

  def mapping(value, path):
    require(isinstance(value, dict), f"{path} must be an object")
    return value if isinstance(value, dict) else {}

  def sequence(value, path):
    require(isinstance(value, list), f"{path} must be an array")
    return value if isinstance(value, list) else []

  def integer(value, path, minimum=0):
    require(
      type(value) is int and value >= minimum, f"{path} must be an integer >= {minimum}"
    )
    return value if type(value) is int else 0

  def number(value, path, *, nullable=False, nonnegative=False):
    require(
      (nullable and value is None)
      or (_finite(value) and (not nonnegative or value >= 0)),
      f"{path} must be a finite number" + (" or null" if nullable else ""),
    )

  def vector(value, path, width):
    values = sequence(value, path)
    require(len(values) == width, f"{path} must have {width} entries")
    for i, item in enumerate(values):
      number(item, f"{path}[{i}]")

  def finite_tree(value, path):
    if isinstance(value, float):
      require(math.isfinite(value), f"{path} is non-finite")
    elif isinstance(value, dict):
      for key, item in value.items():
        finite_tree(item, f"{path}.{key}")
    elif isinstance(value, list):
      for i, item in enumerate(value):
        finite_tree(item, f"{path}[{i}]")

  result = mapping(result, "diagnostic")
  finite_tree(result, "diagnostic")
  require(mode in ("all", "process", "posterior", "critic"), "Unknown diagnostic mode")
  legacy = result.get("version") == "eit_independent_diagnostics_v1"
  require(
    legacy or result.get("version") == "eit_independent_diagnostics_v2",
    "Invalid diagnostic version",
  )
  integer(result.get("episodes"), "episodes", 1)
  overlap = mapping(result.get("training_overlap_check"), "training_overlap_check")
  require(
    overlap.get("status")
    in (
      "distinct_seed",
      "distinct_motion_file",
      "unverified_missing_training_seed",
      "unverified_missing_training_file_manifest",
    ),
    "Invalid training_overlap_check.status",
  )

  if mode in ("all", "process"):
    process = mapping(result.get("process"), "process")
    windows = integer(process.get("windows"), "process.windows")
    width = integer(process.get("window_frames"), "process.window_frames", 2)
    pairs = integer(process.get("transitions"), "process.transitions")
    successors = integer(
      process.get("window_scored_successors"), "process.window_scored_successors"
    )
    require(
      successors == windows * (width - 1),
      "process successor count disagrees with windows",
    )
    mse = mapping(process.get("multistep_mse"), "process.multistep_mse")
    for key in ("root_pos", "root_ori", "joint_pos", "joint_vel"):
      number(
        mse.get(key),
        f"process.multistep_mse.{key}",
        nullable=windows == 0,
        nonnegative=True,
      )
    number(process.get("process_nll"), "process.process_nll", nullable=pairs == 0)
    for key, size in (
      ("rpv_bias_truth_minus_prediction_right_local", 9),
      ("whitened_mean", 70),
    ):
      if pairs or process.get(key) is not None:
        vector(process.get(key), f"process.{key}", size)
    second = process.get("whitened_second_moment")
    if pairs or second is not None:
      rows = sequence(second, "process.whitened_second_moment")
      require(len(rows) == 70, "process.whitened_second_moment must have 70 rows")
      for i, row in enumerate(rows):
        vector(row, f"process.whitened_second_moment[{i}]", 70)

  if mode in ("all", "posterior"):
    bins = sequence(result.get("fixed_record_posterior"), "fixed_record_posterior")
    require(bool(bins), "fixed_record_posterior must contain scored frames")
    for i, item in enumerate(bins):
      prefix = f"fixed_record_posterior[{i}]"
      item = mapping(item, prefix)
      frames = integer(item.get("frames"), f"{prefix}.frames", 1)
      integer(item.get("surviving_episodes"), f"{prefix}.surviving_episodes", 1)
      for key in (
        "start_s",
        "end_s",
        "position_rmse_m",
        "rotation_rmse_rad",
        "velocity_rmse_m_s",
        "mean_innovation_norm",
      ):
        number(item.get(key), f"{prefix}.{key}", nonnegative=True)
      value = item.get("coverage_95")
      require(
        _finite(value) and 0 <= value <= 1, f"{prefix}.coverage_95 must be in [0,1]"
      )
      for key in ("null_direction_violations",):
        count = integer(item.get(key), f"{prefix}.{key}")
        require(count <= frames, f"{prefix}.{key} exceeds frame count")

  if mode in ("all", "critic"):
    critic = mapping(result.get("critic"), "critic")
    gamma = critic.get("gamma")
    require(_finite(gamma) and 0 <= gamma < 1, "critic.gamma must be in [0,1)")
    number(
      critic.get("bootstrap_value_cap"), "critic.bootstrap_value_cap", nullable=True
    )
    require(
      critic.get("actor_state_source") in ("truth", "posterior"),
      "Invalid critic actor_state_source",
    )
    for key in ("censored_samples", "censored_episodes"):
      integer(critic.get(key), f"critic.{key}")
    for key in ("raw_value_error", "bootstrap_value_error"):
      score = mapping(critic.get(key), f"critic.{key}")
      samples = integer(score.get("samples"), f"critic.{key}.samples")
      for field in ("bias", "mae", "rmse"):
        number(
          score.get(field),
          f"critic.{key}.{field}",
          nullable=samples == 0,
          nonnegative=field != "bias",
        )
    episodes = sequence(critic.get("episodes"), "critic.episodes")
    require(len(episodes) == result.get("episodes"), "critic episode count mismatch")
    for i, item in enumerate(episodes):
      prefix = f"critic.episodes[{i}]"
      item = mapping(item, prefix)
      integer(item.get("env_id"), f"{prefix}.env_id")
      reason = item.get("reason")
      require(
        reason in ("failure", "clip_complete", "timeout", "evaluation_truncated"),
        f"{prefix} has invalid termination reason",
      )
      raw = sequence(item.get("raw_value"), f"{prefix}.raw_value")
      used = sequence(item.get("bootstrap_value"), f"{prefix}.bootstrap_value")
      require(
        len(raw) == len(used) and len(raw) > 0,
        f"{prefix} value arrays have invalid lengths",
      )
      for key, values in (("raw_value", raw), ("bootstrap_value", used)):
        for j, value in enumerate(values):
          number(value, f"{prefix}.{key}[{j}]")
      returns = item.get("discounted_return")
      if reason == "failure" or (legacy and reason == "clip_complete"):
        values = sequence(returns, f"{prefix}.discounted_return")
        require(len(values) == len(raw), f"{prefix} discounted return length mismatch")
        for j, value in enumerate(values):
          number(value, f"{prefix}.discounted_return[{j}]")
      else:
        require(returns is None, f"{prefix} censored return must be null")

  if require_gradients or "actor_gradients" in result:
    gradient = mapping(result.get("actor_gradients"), "actor_gradients")
    starts = integer(gradient.get("starts"), "actor_gradients.starts", 1)
    integer(gradient.get("seed"), "actor_gradients.seed")
    for key in (
      "samples_per_start",
      "horizon",
      "microbatch_size",
      "effective_imagined_steps",
    ):
      integer(gradient.get(key), f"actor_gradients.{key}", 1)
    for key in ("episode_ids", "start_frames"):
      values = sequence(gradient.get(key), f"actor_gradients.{key}")
      require(len(values) == starts, f"actor_gradients.{key} count mismatch")
      for i, value in enumerate(values):
        integer(value, f"actor_gradients.{key}[{i}]")
    for key in (
      "reward_gradient_norm",
      "bootstrap_gradient_norm",
      "total_gradient_norm",
    ):
      number(gradient.get(key), f"actor_gradients.{key}", nonnegative=True)
    for key in (
      "reward_return",
      "bootstrap_return",
      "total_return",
      "gradient_dot_product",
    ):
      number(gradient.get(key), f"actor_gradients.{key}")
    for key in ("process_noise_sha256", "measurement_noise_sha256"):
      value = gradient.get(key)
      require(
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value),
        f"actor_gradients.{key} must be a SHA256",
      )
    zero = (
      gradient.get("reward_gradient_norm") == 0
      or gradient.get("bootstrap_gradient_norm") == 0
    )
    for key, lower, upper in (
      ("gradient_cosine", -1, 1),
      ("gradient_angle_degrees", 0, 180),
    ):
      value = gradient.get(key)
      require(
        value is None if zero else _finite(value) and lower <= value <= upper,
        f"actor_gradients.{key} is invalid",
      )
  return errors


def supplementary(
  audit,
  p,
  groups,
  verify_record,
  coverage,
  *,
  final_frames=None,
  policy_record="policy-baseline",
):
  if final_frames is None:
    final_frames = p["rounds"] * p["num_envs"] * p["collection_steps"]
  gradients = {}
  for group in "ABC":
    offline = audit.load_diagnostic(f"diagnostics/{group}/{final_frames}/offline.json")
    gradient = offline.get("actor_gradients", {})
    gradients[group] = gradient or None
    audit.data(
      bool(gradient),
      f"{group}: final actor reward/bootstrap gradient decomposition missing",
    )
    if not gradient:
      continue
    audit.require(
      gradient.get("objective") == "mean_positive_discounted_soft_survival_return"
      and gradient.get("gradient_scope") == "all_actor_parameters_before_clipping"
      and gradient.get("start_source") == "independent_recorded_episode_heads"
      and gradient.get("actor_state_source")
      == ("truth" if group == "A" else "posterior")
      and gradient.get("starts", 0) > 0
      and gradient.get("samples_per_start") == p["policy"]["num_samples"]
      and gradient.get("horizon") == p["policy"]["horizon"]
      and gradient.get("microbatch_size") == p["policy"]["microbatch_size"],
      f"{group}: actor gradient scope/configuration mismatch",
    )
    audit.require(
      gradient.get("seed") == p.get("gradient_seed", 2028),
      f"{group}: actor gradient seed differs from pilot protocol",
    )
    record_path = audit.root / "diagnostics" / group / str(final_frames) / "episodes.pt"
    if record_path.is_file():
      try:
        recorded = torch.load(
          record_path, weights_only=True, mmap=True, map_location="cpu"
        )
        heads = recorded["episodes"][: p.get("gradient_episodes", 4)]
        horizon = p["policy"]["horizon"]
        steps = horizon * len(heads)
        if offline.get("version") == "eit_independent_diagnostics_v1":
          steps = sum(min(horizon, ep["reference_steps"]) for ep in heads)
        expected_steps = p["policy"]["num_samples"] * steps
        audit.require(
          gradient.get("starts") == len(heads)
          and gradient.get("episode_ids") == [ep["env_id"] for ep in heads]
          and gradient.get("start_frames") == [ep["start_frame"] for ep in heads]
          and gradient.get("effective_imagined_steps") == expected_steps,
          f"{group}: actor gradient heads or effective steps differ from independent recording",
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
        audit.failures.append(f"Cannot verify actor gradient heads {group}: {error}")
    for key in (
      "reward_gradient_norm",
      "bootstrap_gradient_norm",
      "total_gradient_norm",
    ):
      audit.require(
        _finite(gradient.get(key)) and gradient[key] >= 0, f"{group}: invalid {key}"
      )
    for key in (
      "reward_return",
      "bootstrap_return",
      "total_return",
      "gradient_dot_product",
    ):
      audit.require(_finite(gradient.get(key)), f"{group}: invalid {key}")
    for key in ("process_noise_sha256", "measurement_noise_sha256"):
      value = gradient.get(key)
      audit.require(
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value),
        f"{group}: missing fixed-noise identity {key}",
      )
    cosine, angle = (
      gradient.get("gradient_cosine"),
      gradient.get("gradient_angle_degrees"),
    )
    nonzero = bool(
      gradient.get("reward_gradient_norm") and gradient.get("bootstrap_gradient_norm")
    )
    audit.require(
      (_finite(cosine) and -1 <= cosine <= 1 and _finite(angle) and 0 <= angle <= 180)
      if nonzero
      else cosine is None and angle is None,
      f"{group}: gradient angle is inconsistent with component norms",
    )
  folder = f"diagnostics/A/{final_frames}-truth"
  truth_eval, truth = (
    audit.load(f"{folder}/evaluation.json"),
    audit.load_diagnostic(f"{folder}/offline.json", mode="critic"),
  )
  truth_error = truth.get("critic", {}).get("bootstrap_value_error", {})
  truth_status = critic_score_status(truth.get("critic"))
  audit.data(
    truth_status != "missing", "A: independent truth-input critic score missing"
  )
  audit.data(
    truth_eval.get("evaluation_complete") is True
    and truth_eval.get("censored_episodes") == 0
    and truth_eval.get("timeout_count") == 0
    and truth_eval.get("protocol") == "full_clip_v1"
    and truth_eval.get("step_budget") is None,
    "A: independent truth-input episodes incomplete or censored",
  )
  if truth:
    audit.require(
      truth.get("checkpoint_sha256") == groups["A"]["final_snapshot_sha256"],
      "A truth-input score belongs to a different checkpoint",
    )
    audit.file(
      audit.root / folder / "episodes.pt",
      truth.get("recording_sha256"),
      "A truth-input independent recording",
    )
    verify_record(
      audit,
      audit.root / folder / "episodes.pt",
      p,
      "A",
      groups["A"]["final_snapshot_sha256"],
      truth_eval,
      truth,
      actor_source="truth",
      posterior=False,
    )
  common_spec = audit.load("diagnostics/common-recordings.json")
  common = {}
  for name in (policy_record, "walking-reference"):
    location = common_spec.get("recordings", {}).get(name)
    if not location or not Path(location).is_file():
      audit.pending.append(f"Missing shared independent recording: {name}")
      continue
    record_path = Path(location)
    try:
      record = torch.load(record_path, weights_only=True, mmap=True, map_location="cpu")
      audit.require(
        record.get("source") == "mujoco_evaluation"
        and record.get("split") == "diagnostic"
        and record.get("seed") == p["diagnostic_seed"]
        and record.get("motion_sha256") == p["motion_sha256"],
        f"{name}: common trajectory is not independent pilot motion",
      )
      sha = file_sha256(record_path)
      count = sum(len(ep["reward"]) for ep in record["episodes"])
      common[name] = {
        "recording": str(record_path),
        "sha256": sha,
        "frames": count,
        "longest_episode_seconds": max(
          (len(ep["reward"]) * record["control_dt"] for ep in record["episodes"]),
          default=0,
        ),
        "groups": {},
      }
      for group in "ABC":
        result = {}
        for mode in ("process", "posterior"):
          path = f"diagnostics/common/{name}/{group}-{mode}.json"
          scored = audit.load_diagnostic(path, mode=mode)
          if not scored:
            continue
          audit.require(
            scored.get("checkpoint_sha256") == groups[group]["final_snapshot_sha256"]
            and scored.get("recording_sha256") == sha
            and scored.get("group") == group
            and scored.get("recording_seed") == record["seed"]
            and scored.get("recording_checkpoint_sha256") == record["checkpoint_sha256"]
            and scored.get("motion_digest") == record["motion_digest"],
            f"{name}/{group}/{mode}: common score identity mismatch",
          )
          if mode == "process":
            process = scored.get("process", {})
            audit.data(
              process.get("transitions", 0) > 0 and _finite(process.get("process_nll")),
              f"{name}/{group}: common process score unavailable",
            )
            result[mode] = {
              "path": str(audit.root / path),
              "transitions": process.get("transitions"),
              "process_nll": process.get("process_nll"),
              "windows": process.get("windows"),
            }
          else:
            fraction, n_frames = coverage(scored.get("fixed_record_posterior", []))
            audit.data(
              n_frames > 0, f"{name}/{group}: common posterior score unavailable"
            )
            audit.require(
              n_frames == count,
              f"{name}/{group}: common posterior frame count differs from recording",
            )
            result[mode] = {
              "path": str(audit.root / path),
              "frames": n_frames,
              "coverage_95": fraction,
            }
        common[name]["groups"][group] = result
    except (
      OSError,
      ValueError,
      RuntimeError,
      KeyError,
      TypeError,
      EOFError,
      pickle.UnpicklingError,
    ) as error:
      audit.failures.append(f"Cannot read shared recording {name}: {error}")
  return {
    "actor_gradients": gradients,
    "A_truth_critic_error": truth_error,
    "A_truth_critic_status": truth_status,
    "common_trajectories": common,
    "continuous_drift_recording": "Optional separate record_eit_drift artifact; not required by this base audit.",
  }
