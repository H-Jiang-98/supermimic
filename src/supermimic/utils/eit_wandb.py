"""Offline EIT metrics, with one persistent W&B identity per training directory.

Offline processes produce separate transaction logs with absolute training steps.
Sync completed segments in start-round order using `wandb sync --append`.
Local training checkpoints and JSONL files remain authoritative if logging fails.
"""

from __future__ import annotations

import json
import math
import os
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from tempfile import NamedTemporaryFile

import wandb

from supermimic.utils.wandb_artifacts import log_model_artifact


def _write_json(path: Path, value: dict) -> None:
  with NamedTemporaryFile(dir=path.parent, mode="w", delete=False) as stream:
    temporary = Path(stream.name)
    try:
      json.dump(value, stream, indent=2, allow_nan=False)
      stream.write("\n")
      stream.close()
      temporary.replace(path)
    finally:
      temporary.unlink(missing_ok=True)


def _scalars(value: dict, prefix: str = "") -> dict[str, int | float]:
  result = {}
  for key, item in value.items():
    name = f"{prefix}/{key}" if prefix else key
    if isinstance(item, dict):
      result.update(_scalars(item, name))
    elif type(item) in (int, float) and math.isfinite(item):
      result[name] = item
  return result


def metric_row(metrics: dict, profile: dict) -> dict[str, int | float]:
  """Flatten numeric fields, excluding IDs, episode lists and missing values."""
  return {
    "Counters/round": metrics["round"],
    "Counters/collected_frames": metrics["collected_frames"],
    "Counters/mean_updates": metrics["mean_steps"],
    "Counters/covariance_updates": metrics["cov_steps"],
    **_scalars(metrics.get("train", {}), "Train"),
    **_scalars(metrics.get("validation", {}), "ModelValidation"),
    **_scalars(metrics.get("policy", {})),
    **_scalars(metrics.get("budget", {}), "Budget"),
    **_scalars(profile.get("seconds", {}), "Perf"),
    **_scalars(profile.get("runtime", {}), "Runtime"),
    **_scalars(profile.get("runtime_after_save", {}), "RuntimeAfterSave"),
  }


class EITOfflineWandb:
  """Observe committed rounds without changing the optimizer/resume contract."""

  def __init__(
    self,
    *,
    project: str | None = None,
    entity: str | None = None,
    name: str | None = None,
    group: str | None = None,
    log_models: bool = False,
  ):
    self.options = dict(project=project, entity=entity, name=name, group=group)
    self.log_models = log_models
    self.run = None
    self.enabled = False
    self.session = None
    self.session_path = None

  def _start(self, training, output: Path) -> None:
    if wandb.run is not None:
      raise ValueError("EIT offline logging requires its own W&B run")
    self.output = output
    self.training = training
    group = training.runner.cfg.estimator_inputs.group
    identity = {
      "group": group,
      "training_seed": training.training_seed,
      "motion_sha256": [f["sha256"] for f in training.training_motion_files],
      "initialization_digest": training.initialization_digest,
    }
    path = output / "wandb_run.json"
    if any((output / name).is_symlink() for name in ("wandb_run.json", "wandb")):
      raise ValueError("W&B metadata and logs must be local to the training directory")
    if path.exists():
      metadata = json.loads(path.read_text())
      if (
        metadata.get("version") != "eit_wandb_offline_v1"
        or metadata.get("identity") != identity
      ):
        raise ValueError("W&B run identity differs from the restored training")
      for key, value in self.options.items():
        if value is not None and metadata.get(key) != value:
          raise ValueError(f"W&B {key} differs from the existing run")
    else:
      metadata = {
        "version": "eit_wandb_offline_v1",
        "identity": identity,
        "id": uuid.uuid4().hex[:16],
        "project": self.options["project"] or training.runner.cfg.wandb_project,
        "entity": self.options["entity"] or os.environ.get("WANDB_ENTITY"),
        "name": self.options["name"] or f"eit-{group}-seed{training.training_seed}",
        "group": self.options["group"] or "eit",
      }
      _write_json(path, metadata)
    config = {
      **identity,
      "training_method": "estimator_in_the_loop",
      "num_envs": training.runner.env.num_envs,
      "runner_cfg": asdict(training.runner.cfg),
      "loop_config": asdict(training.cfg),
      "policy_config": asdict(training.policy.cfg),
      "snapshot_interval": training.snapshot_interval,
      "starting_round": training.rounds,
      "starting_collected_frames": training.collected_frames,
      "log_models": self.log_models,
    }
    # Explicit offline mode overrides environment settings and never logs in.
    # W&B resume is intentionally omitted: each process owns a new offline file.
    self.run = wandb.init(
      mode="offline",
      id=metadata["id"],
      project=metadata["project"],
      entity=metadata["entity"],
      name=metadata["name"],
      group=metadata["group"],
      job_type=group,
      tags=["eit", group, f"seed-{training.training_seed}"],
      dir=str(output),
      config=config,
    )
    self.run.define_metric("Counters/collected_frames")
    self.run.define_metric(
      "*", step_metric="Counters/collected_frames", step_sync=False
    )
    self.session = {
      "run_id": metadata["id"],
      "start_round": training.rounds,
      "last_logged_round": training.rounds,
      "last_logged_frames": training.collected_frames,
      "finished": False,
    }
    self.session_path = Path(self.run.dir).parent / "eit_session.json"
    _write_json(self.session_path, self.session)
    self.enabled = True
    if training.rounds == 0:
      self._log_snapshot(0, 0)

  def _warn(self, phase: str, error: Exception) -> None:
    print(
      f"[WARN] EIT W&B offline {phase} failed: {type(error).__name__}: {error}; local checkpoints/JSONL remain available."
    )

  def _log_snapshot(self, rounds: int, frames: int) -> None:
    if not self.log_models:
      return
    path = self.output / "snapshots" / f"frames_{frames}.pt"
    if path.is_file():
      log_model_artifact(
        [str(path)],
        run=self.run,
        metadata={"round": rounds, "collected_frames": frames},
        aliases=["latest", f"frames-{frames}"],
      )

  def log_round(self, metrics: dict, profile: dict) -> None:
    if not self.enabled:
      return
    try:
      self.run.log(metric_row(metrics, profile), step=metrics["round"])
      self.session.update(
        last_logged_round=metrics["round"],
        last_logged_frames=metrics["collected_frames"],
      )
      self._log_snapshot(metrics["round"], metrics["collected_frames"])
    except Exception as error:
      self.enabled = False
      self._warn("recording", error)

  @contextmanager
  def track(self, training, output: Path):
    # Keep optional artifact staging with the run by default, and restore the
    # caller's environment on exit. Respect explicit shared storage locations.
    self.run, self.session, self.session_path = None, None, None
    self.enabled = False
    previous = {
      key: os.environ.get(key) for key in ("WANDB_DATA_DIR", "WANDB_CACHE_DIR")
    }
    os.environ.setdefault("WANDB_DATA_DIR", str(output / "wandb-data"))
    os.environ.setdefault("WANDB_CACHE_DIR", str(output / "wandb-cache"))
    try:
      try:
        self._start(training, output)
      except Exception as error:
        self._warn("initialization", error)
      exit_code = 0
      try:
        yield self
      except BaseException:
        exit_code = 1
        raise
      finally:
        if self.run is not None:
          try:
            self.run.finish(exit_code=exit_code)
            if self.session is not None:
              self.session.update(
                finished=True, exit_code=exit_code, logging_ok=self.enabled
              )
              _write_json(self.session_path, self.session)
          except Exception as error:
            self._warn("finish", error)
        self.enabled = False
    finally:
      for key, value in previous.items():
        if value is None:
          os.environ.pop(key, None)
        else:
          os.environ[key] = value
