"""Frozen manifests, resumable batches and joint tracking/odometry evaluation."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock

from supermimic.runners.eit_inference import load_hashed_checkpoint
from supermimic.scripts.batch_eval_eit import cfg_json, conditions
from supermimic.validation.config import (
  CONFIG_GROUPS,
  LOCAL_STUDY_VERSION,
  LOCAL_VARIANTS,
  VARIANTS,
  atomic_json,
  config_dict,
  digest,
  file_digest,
  load_config,
  motion_paths,
)
from supermimic.validation.domains import domain_table
from supermimic.validation.runtime import make_env_config

VERSION = "eit_joint_multiclip_v5"


def freeze(cfg):
  paths = motion_paths(cfg)
  local = any(v in LOCAL_VARIANTS for v in cfg.methods)
  if local and not set(cfg.methods).issubset(LOCAL_VARIANTS):
    raise ValueError("Local and legacy methods require separate studies")
  required = set(cfg.methods) if local else {"C", *(v[0] for v in cfg.methods)}
  saved, checkpoints = {}, {}
  for group in LOCAL_VARIANTS if local else "ABC":
    path = Path(cfg.checkpoint_root) / group / "training.pt"
    if group not in required and not path.is_file():
      continue
    s, sha = load_hashed_checkpoint(path, "cpu")
    if group in required:
      saved[group] = s
    inputs = s["runner_config"]["estimator_inputs"]
    if local and inputs["residual_actor"] != group.endswith("-residual"):
      raise ValueError(f"Checkpoint residual mode mismatch: {path}")
    if inputs["group"] != group.split("-")[0]:
      raise ValueError(f"Checkpoint group mismatch: {path}")
    checkpoints[group] = dict(
      path=str(path),
      sha256=sha,
      collected_steps=s["collected_steps"],
      collected_frames=s["collected_frames"],
    )
    if local:
      from supermimic.runners.eit_training import initialization_fingerprint

      checkpoints[group].update(
        actor_sha256=initialization_fingerprint(s["actor"]),
        estimator_sha256=initialization_fingerprint(s["world_model"]),
      )
  if len({v["collected_steps"] for v in checkpoints.values()}) != 1:
    raise ValueError("A/B/C checkpoint training budgets differ")
  if local:
    paired = {
      (
        s["training_seed"],
        s["initialization_digest"],
        s["num_envs"],
        s["collected_frames"],
        s["runner_config"]["replay_ratio"],
        digest(s["policy_config"]),
      )
      for s in saved.values()
    }
    if len(paired) != 1:
      raise ValueError(
        "Local study requires paired seeds, initialization and training budgets"
      )
  training_sets = [
    {v["sha256"] for v in s["training_motion_files"]} for s in saved.values()
  ]
  train_hashes = training_sets[0]
  if any(t != train_hashes for t in training_sets):
    raise ValueError("Checkpoint training motion sets differ")
  motions = []
  for path in paths:
    with np.load(path, allow_pickle=False) as arrays:
      frames = len(arrays["joint_pos"])
    sha = file_digest(path)
    motions.append(
      dict(
        path=path,
        name=Path(path).name,
        sha256=sha,
        frames=frames,
        seen_in_training=sha in train_hashes,
      )
    )
  env_cfg, failures = make_env_config(cfg, paths[0])
  table = domain_table(env_cfg, cfg.num_domain_contexts, cfg.sobol_seed)
  source_root = Path(__file__).parent.parent
  sources = {
    str(p.relative_to(source_root)): file_digest(p)
    for folder in ("validation",)
    for p in sorted((source_root / folder).glob("*.py"))
  }
  for name in (
    "estimator.py",
    "belief.py",
    "motion_command.py",
    "estimator_inputs.py",
    "sim_context.py",
    "representations.py",
    "local_control.py",
    "utils/lie_state.py",
    "models.py",
  ):
    sources[name] = file_digest(source_root / name)
  config = config_dict(cfg)
  # Method selection schedules work. The same scenario keeps its task key when
  # later runs request additional policies.
  del config["methods"]
  manifest = dict(
    version=LOCAL_STUDY_VERSION if local else VERSION,
    kind="joint",
    config=config,
    checkpoints=checkpoints,
    motions=motions,
    domains=table,
    environment=conditions(env_cfg),
    monitored_failures=cfg_json(failures),
    source_sha256=sources,
  )
  manifest["identity"] = digest(manifest)
  return manifest, saved


def simulation_protocol(manifest):
  # Plot settings and CLI scheduling may change between runs. Keep the original
  # manifest and identity on disk so existing per-method task keys stay valid.
  presentation = CONFIG_GROUPS["plots"] | CONFIG_GROUPS["viewer"] | {"methods"}
  orchestration = {
    "validation/report.py",
    "validation/viewer.py",
    "validation/study.py",
  }
  return {
    **{k: v for k, v in manifest.items() if k != "identity"},
    "config": {k: v for k, v in manifest["config"].items() if k not in presentation},
    "source_sha256": {
      k: v for k, v in manifest["source_sha256"].items() if k not in orchestration
    },
  }


def resume_manifest(frozen, current):
  if frozen["identity"] != digest({k: v for k, v in frozen.items() if k != "identity"}):
    raise ValueError("Frozen manifest checksum mismatch")
  if simulation_protocol(frozen) != simulation_protocol(current):
    raise ValueError("Frozen simulation protocol differs; use a new output directory")
  return frozen


def tasks(manifest, variants=None):
  cfg = manifest["config"]
  if variants is None:
    variants = (
      ["C-posterior"]
      if manifest["kind"] == "odometry"
      else (
        LOCAL_VARIANTS if manifest.get("version") == LOCAL_STUDY_VERSION else VARIANTS
      )
    )
  for motion in manifest["motions"]:
    for variant in variants:
      for start in range(0, cfg["num_domain_contexts"], cfg["num_envs"]):
        ids = list(
          range(start, min(start + cfg["num_envs"], cfg["num_domain_contexts"]))
        )
        key = digest([manifest["identity"], motion["sha256"], variant, ids])[:24]
        yield key, motion, variant, ids


def load_task(output, key, manifest, fields=None):
  meta_path = output / "tasks" / f"{key}.json"
  arrays_path = meta_path.with_suffix(".npz")
  if not meta_path.exists() or not arrays_path.exists():
    return None
  try:
    meta = json.loads(meta_path.read_text())
    if meta["identity"] != manifest["identity"] or meta["array_sha256"] != file_digest(
      arrays_path
    ):
      return None
    with np.load(arrays_path, allow_pickle=False) as data:
      arrays = {k: data[k] for k in (data.files if fields is None else fields)}
    return meta, arrays
  except (ValueError, KeyError, OSError):
    return None


def save_task(output, key, arrays, metadata):
  path = output / "tasks" / f"{key}.npz"
  path.parent.mkdir(parents=True, exist_ok=True)
  with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
    tmp = Path(stream.name)
    try:
      np.savez_compressed(stream, **arrays)
      stream.flush()
      os.fsync(stream.fileno())
      tmp.replace(path)
    finally:
      tmp.unlink(missing_ok=True)
  atomic_json(
    path.with_suffix(".json"), {**metadata, "array_sha256": file_digest(path)}
  )


def run(cfg):
  output = Path(cfg.output)
  output.mkdir(parents=True, exist_ok=True)
  torch.set_num_threads(cfg.torch_threads)
  with FileLock(str(output / ".study.lock"), timeout=0):
    manifest, saved = freeze(cfg)
    frozen = output / "manifest.json"
    if frozen.exists():
      manifest = resume_manifest(json.loads(frozen.read_text()), manifest)
    else:
      atomic_json(frozen, manifest)
      atomic_json(output / "domains.json", manifest["domains"])
    device = (
      ("cuda:0" if torch.cuda.is_available() else "cpu")
      if cfg.device == "auto"
      else cfg.device
    )
    runtime_cfg = replace(cfg, device=device)
    # Versioned bundle validation compares metadata tensors to live buffers;
    # decode on the inference device, as the existing eval_eit loader does.
    if torch.device(device).type != "cpu":
      for group in saved:
        identity = manifest["checkpoints"][group]
        saved[group], sha = load_hashed_checkpoint(identity["path"], device)
        if sha != identity["sha256"]:
          raise ValueError("Checkpoint changed after experiment freeze")
    if cfg.compile_kernels and (
      not device.startswith("cuda") or not torch.cuda.is_available()
    ):
      raise RuntimeError(
        "Compiled tracking requires CUDA; CPU/eager fallback is disabled"
      )
    from supermimic.validation.report import report
    from supermimic.validation.tracking import run_tracking

    with torch.compiler.set_stance("default" if cfg.compile_kernels else "force_eager"):
      run_tracking(
        runtime_cfg, manifest, saved, list(tasks(manifest, cfg.methods)), output
      )
    return 0 if report(cfg)["complete"] else 1


def main():
  parser = argparse.ArgumentParser(
    description="Joint full-clip EIT tracking and odometry evaluation"
  )
  parser.add_argument(
    "command", choices=["run", "report", "view"], nargs="?", default="run"
  )
  parser.add_argument("--config", type=Path, default=Path("eval_eit_validation.toml"))
  parser.add_argument("--device")
  parser.add_argument("--output")
  parser.add_argument("--num-domain-contexts", type=int)
  parser.add_argument("--num-envs", type=int)
  parser.add_argument("--max-frames", type=int)
  args = parser.parse_args()
  overrides = {
    k: v
    for k, v in vars(args).items()
    if k not in ("config", "command") and v is not None
  }
  cfg = load_config(args.config, overrides)
  if args.command == "run":
    raise SystemExit(run(cfg))
  if args.command == "report":
    from supermimic.validation.report import report

    raise SystemExit(0 if report(cfg)["complete"] else 1)
  from supermimic.validation.viewer import view

  view(cfg)
