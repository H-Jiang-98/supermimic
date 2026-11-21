"""Publish immutable, frame-indexed EIT checkpoints under the training lock."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from shutil import copyfileobj
from tempfile import NamedTemporaryFile

import torch


def file_sha256(path: Path) -> str:
  with path.open("rb") as stream:
    return hashlib.file_digest(stream, "sha256").hexdigest()


def publish_snapshot(checkpoint: Path, state: dict | None = None) -> None:
  """Register bytes already committed to training.pt. Safe to repeat after a crash."""
  if state is None:
    state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
  frames = state["collected_frames"]
  directory = checkpoint.parent / "snapshots"
  snapshot = directory / f"frames_{frames}.pt"
  index = directory / "index.json"
  if any(path.is_symlink() for path in (directory, snapshot, index)):
    raise ValueError(
      "Snapshot publication requires local files in the training directory"
    )
  directory.mkdir(exist_ok=True)
  digest = file_sha256(checkpoint)
  learner = state["learner"]
  policy = state.get("policy", {})
  row = {
    "path": str(snapshot.relative_to(checkpoint.parent)),
    "sha256": digest,
    "round": state["rounds"],
    "collected_frames": frames,
    "mean_steps": learner["mean"]["steps"],
    "cov_steps": learner["cov"]["steps"],
    "actor_steps": policy.get("actor_steps", 0),
    "critic_steps": policy.get("critic_steps", 0),
    "model_version": learner["model_version"],
    "initialization_digest": state.get("initialization_digest"),
    "actor_version": state.get("actor_version"),
    "critic_version": state.get("critic_version"),
    "stage": state["stage"],
    "group": state["runner_config"]["estimator_inputs"]["group"],
    "training_seed": state.get("training_seed"),
  }
  registry = {"version": "eit_snapshots_v1", "snapshots": []}
  if index.exists():
    registry = json.loads(index.read_text())
    if registry.get("version") != "eit_snapshots_v1":
      raise ValueError("Unexpected snapshot index version")
    if any(r["collected_frames"] > frames for r in registry["snapshots"]):
      raise ValueError("Snapshot index is ahead of the current training checkpoint")
  existing = [r for r in registry["snapshots"] if r["collected_frames"] == frames]
  if existing and existing != [row]:
    raise ValueError(f"Snapshot index conflict for {frames} collected frames")
  if snapshot.exists():
    if file_sha256(snapshot) != digest:
      raise ValueError(f"Snapshot content conflict for {frames} collected frames")
  else:
    # Keep independent inodes so even an in-place edit to training.pt cannot
    # change a published candidate. The registry is replaced only after the file.
    with NamedTemporaryFile(dir=directory, delete_on_close=False) as temporary:
      with checkpoint.open("rb") as source:
        copyfileobj(source, temporary)
      temporary.close()
      os.chmod(temporary.name, 0o444)
      os.link(temporary.name, snapshot)
  if existing:
    return
  registry["snapshots"].append(row)
  with NamedTemporaryFile(dir=directory, mode="w", delete=False) as temporary:
    temp_path = Path(temporary.name)
    try:
      json.dump(registry, temporary, indent=2, allow_nan=False)
      temporary.write("\n")
      temporary.close()
      temp_path.replace(index)
    finally:
      temp_path.unlink(missing_ok=True)
