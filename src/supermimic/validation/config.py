"""Strict TOML configuration and portable experiment identities."""

from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import tempfile
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path

VARIANTS = ("A-posterior", "B-posterior", "C-posterior", "A-truth")


@dataclass
class StudyConfig:
  methods: list[str] = field(default_factory=lambda: ["A-posterior"])
  checkpoint_root: str = "logs/eit_v6_Lafan_compiled_h32_2048envs_stride1/checkpoints/iteration_2500/seed2027"
  motion_files: list[str] = field(default_factory=lambda: ["data/motions/*.npz"])
  output: str = "artifacts/eit-validation"
  device: str = "auto"
  num_domain_contexts: int = 16
  num_envs: int = 256
  compile_kernels: bool = True
  jacobian_batch_chunk_size: int = 64
  transfer_chunk_steps: int = 256
  host_check_interval_steps: int = 256
  debug_checks: bool = False
  warmup_steps: int = 20
  sobol_seed: int = 2026
  evaluation_seed: int = 2718
  torch_threads: int = 1
  progress_interval_s: float = 30.0
  time_bin_s: float = 2.0
  boxplot_interval_s: float = 2.0
  rotation_rmse_unit: str = "deg"
  iterations: int = 20
  ls_iterations: int = 50
  max_frames: int = 0  # Explicit censored smoke budget, never a full-clip result.
  representative_clips: list[str] = field(default_factory=list)
  ee_body_names: list[str] = field(
    default_factory=lambda: [
      "left_ankle_roll_link",
      "right_ankle_roll_link",
      "left_wrist_yaw_link",
      "right_wrist_yaw_link",
    ]
  )
  replay_context_selection: str = "first_middle_last"
  replay_context_ids: list[int] = field(default_factory=list)
  plot_dpi: int = 150
  plot_width: float = 16.0
  plot_row_height: float = 3.3
  plot_formats: list[str] = field(default_factory=lambda: ["png", "svg"])
  viewer_port: int = 8080
  playback_speed: float = 1.0
  panel_spacing: float = 4.0
  reference_rgba: list[float] = field(default_factory=lambda: [0.3, 0.8, 0.4, 0.35])
  truth_rgba: list[float] = field(default_factory=lambda: [0.65, 0.65, 0.65, 1.0])
  posterior_rgba: list[float] = field(default_factory=lambda: [0.45, 0.4, 0.9, 0.45])
  show_reference: bool = True
  show_truth: bool = True
  show_posterior: bool = True

  def validate(self):
    if (
      not isinstance(self.methods, list)
      or not self.methods
      or any(method not in VARIANTS for method in self.methods)
      or len(set(self.methods)) != len(self.methods)
    ):
      raise ValueError(
        f"methods must be a nonempty list of distinct methods from {VARIANTS}"
      )
    for key in (
      "num_domain_contexts",
      "num_envs",
      "torch_threads",
      "iterations",
      "ls_iterations",
      "plot_dpi",
      "jacobian_batch_chunk_size",
      "transfer_chunk_steps",
      "host_check_interval_steps",
      "warmup_steps",
    ):
      if type(getattr(self, key)) is not int or getattr(self, key) < 1:
        raise ValueError(f"{key} must be a positive integer")
    for key in (
      "time_bin_s",
      "boxplot_interval_s",
      "plot_width",
      "plot_row_height",
      "playback_speed",
      "panel_spacing",
      "progress_interval_s",
    ):
      if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
        raise ValueError(f"{key} must be positive and finite")
    for key in ("compile_kernels", "debug_checks"):
      if type(getattr(self, key)) is not bool:
        raise ValueError(f"{key} must be boolean")
    ratio = self.boxplot_interval_s / self.time_bin_s
    if not math.isclose(ratio, round(ratio)) or ratio < 1:
      raise ValueError("boxplot_interval_s must be an integer multiple of time_bin_s")
    if self.rotation_rmse_unit != "deg":
      raise ValueError("This protocol reports rotation RMSE in deg")
    if type(self.max_frames) is not int or self.max_frames < 0:
      raise ValueError("max_frames must be a nonnegative integer")
    if not self.motion_files or not self.ee_body_names:
      raise ValueError("motion_files and ee_body_names must be nonempty")
    if self.replay_context_selection not in ("first_middle_last", "explicit", "none"):
      raise ValueError("Unknown replay_context_selection")
    if any(
      type(i) is not int or not 0 <= i < self.num_domain_contexts
      for i in self.replay_context_ids
    ):
      raise ValueError("Replay context index outside sample table")
    if not self.plot_formats or set(self.plot_formats) - {"png", "svg", "pdf"}:
      raise ValueError("Unsupported plot format")
    for key in ("reference_rgba", "truth_rgba", "posterior_rgba"):
      values = getattr(self, key)
      if len(values) != 4 or any(
        not math.isfinite(x) or not 0 <= x <= 1 for x in values
      ):
        raise ValueError(f"Invalid {key}")
    if self.truth_rgba[3] != 1:
      raise ValueError("Truth rendering must be opaque")

  def replay_ids(self):
    if self.replay_context_selection == "none":
      return []
    if self.replay_context_selection == "explicit":
      return sorted(set(self.replay_context_ids))
    return sorted({0, self.num_domain_contexts // 2, self.num_domain_contexts - 1})


CONFIG_GROUPS = {
  "evaluation": {"methods"},
  "paths": {"checkpoint_root", "motion_files", "output"},
  "scenarios": {"num_domain_contexts", "sobol_seed", "evaluation_seed", "max_frames"},
  "simulation": {"iterations", "ls_iterations"},
  "execution": {
    "device",
    "num_envs",
    "torch_threads",
    "compile_kernels",
    "warmup_steps",
    "jacobian_batch_chunk_size",
    "transfer_chunk_steps",
  },
  "diagnostics": {"debug_checks", "host_check_interval_steps", "progress_interval_s"},
  "metrics": {"time_bin_s", "rotation_rmse_unit", "ee_body_names"},
  "plots": {
    "representative_clips",
    "boxplot_interval_s",
    "plot_formats",
    "plot_dpi",
    "plot_width",
    "plot_row_height",
  },
  "replay": {"replay_context_selection", "replay_context_ids"},
  "viewer": {
    "viewer_port",
    "playback_speed",
    "panel_spacing",
    "reference_rgba",
    "truth_rgba",
    "posterior_rgba",
    "show_reference",
    "show_truth",
    "show_posterior",
  },
}


def load_config(path, overrides=None):
  path = Path(path).resolve()
  values = {}
  for key, value in tomllib.loads(path.read_text()).items():
    if key in CONFIG_GROUPS:
      if not isinstance(value, dict):
        raise ValueError(f"Expected a TOML table for [{key}]")
      unknown = value.keys() - CONFIG_GROUPS[key]
      if unknown:
        raise ValueError(f"Unknown settings in [{key}]: {sorted(unknown)}")
      entries = value
    else:
      # Flat study files (including odometry) use the same field names.
      entries = {key: value}
    duplicate = values.keys() & entries.keys()
    if duplicate:
      raise ValueError(f"Duplicate settings: {sorted(duplicate)}")
    values.update(entries)
  values.update(overrides or {})
  cfg = StudyConfig(**values)  # Unknown keys are errors.
  for name in ("checkpoint_root", "output"):
    setattr(cfg, name, str((path.parent / getattr(cfg, name)).resolve()))
  cfg.motion_files = [str(path.parent / p) for p in cfg.motion_files]
  cfg.validate()
  return cfg


def motion_paths(cfg):
  result = sorted(
    {str(Path(p).resolve()) for pattern in cfg.motion_files for p in glob.glob(pattern)}
  )
  if not result:
    raise ValueError("No motion files matched")
  if len({Path(p).name for p in result}) != len(result):
    raise ValueError("Motion basenames must be unique")
  return result


def digest(value):
  return hashlib.sha256(
    json.dumps(value, sort_keys=True, allow_nan=False).encode()
  ).hexdigest()


def file_digest(path):
  with open(path, "rb") as stream:
    return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path, value):
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
    tmp = Path(stream.name)
    try:
      json.dump(value, stream, indent=2, allow_nan=False)
      stream.flush()
      os.fsync(stream.fileno())
      tmp.replace(path)
    finally:
      tmp.unlink(missing_ok=True)


def scenario_seed(seed, clip_sha, context_id, purpose):
  return int(digest([seed, clip_sha, int(context_id), purpose])[:15], 16)


def config_dict(cfg):
  return asdict(cfg)
