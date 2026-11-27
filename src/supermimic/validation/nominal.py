"""Frozen paired scenarios for the nominal belief-input ablation."""

from __future__ import annotations

import hashlib
import tempfile
import tomllib
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from supermimic.validation.config import digest, file_digest

# Mode order also groups the shared GT actor and the shared Mean estimator
# into contiguous batches.
ROUTES = {
  "Prop.+KP": ("prop_kp", None),
  "GT-Oracle": ("gt", None),
  "GT-Estimated": ("gt", "mean"),
  "Mean-in-Loop": ("mean", "mean"),
  "Belief-in-Loop": ("belief", "belief"),
}
FRAME_COLUMNS = (
  "mpjpe_cm",
  "height_cm",
  "roll_deg",
  "pitch_deg",
  "body_velocity_m_s",
  "root_position_cm",
  "tracking_reward",
)
PROTOCOL = "eit_nominal_tracking_v1"


@dataclass(frozen=True)
class Scenario:
  clip: str
  seed: int
  mode: str

  def path(self, output):
    return output / "episodes" / self.mode / self.clip / f"{self.seed}.npz"


def read_settings(path):
  path = Path(path).resolve()
  with path.open("rb") as stream:
    cfg = tomllib.load(stream)
  checkpoints = {key: path.parent / p for key, p in cfg["checkpoints"].items()}
  if set(checkpoints) != {actor for actor, _ in ROUTES.values()}:
    raise ValueError("Expected prop_kp, gt, mean and belief checkpoint paths")
  study = cfg["study"]
  clips = [path.parent / p for p in study["motion_files"]]
  seeds = study["seeds"]
  if not clips or len({p.name for p in clips}) != len(clips):
    raise ValueError("Expected nonempty motion list with distinct basenames")
  if (
    not seeds or any(type(s) is not int for s in seeds) or len(set(seeds)) != len(seeds)
  ):
    raise ValueError("Expected distinct integer evaluation seeds")
  return checkpoints, clips, seeds, path.parent / study["output"], cfg["execution"]


def manifest_for(clips, seeds, checkpoints):
  motions = []
  for p in clips:
    with np.load(p, allow_pickle=False) as motion:
      frames = len(motion["joint_pos"])
    if frames < 2:
      raise ValueError(f"{p}: nominal evaluation requires at least one action")
    motions.append(dict(name=p.name, sha256=file_digest(p), steps=frames - 1))
  manifest = dict(
    protocol=PROTOCOL,
    checkpoints=checkpoints,
    motions=motions,
    seeds=seeds,
    routes=ROUTES,
    columns=FRAME_COLUMNS,
    friction=0.75,
    com_offset=[0.0] * 3,
    encoder_bias=[0.0] * 29,
    bootstrap_seed=2718,
    bootstrap_samples=2000,
  )
  manifest["identity"] = digest(manifest)
  return manifest


def scenario_groups(manifest):
  for clip in sorted(manifest["motions"], key=lambda c: (-c["steps"], c["name"])):
    for seed in manifest["seeds"]:
      yield [
        Scenario(clip["name"], seed, mode) for mode in manifest.get("routes", ROUTES)
      ]


def batches(groups, capacity, layout):
  if capacity < (len(groups[0]) if layout == "mixed" else 1):
    raise ValueError("Insufficient environment slots for one group of modes")
  if layout == "sequential":
    for mode in ROUTES:
      singles = [s for group in groups for s in group if s.mode == mode]
      for i in range(0, len(singles), capacity):
        yield singles[i : i + capacity]
    return
  batch = []
  for group in groups:
    if len(batch) + len(group) > capacity:
      yield batch
      batch = []
    batch.extend(group)
  if batch:
    yield batch


def atomic_npz(path, **arrays):
  path.parent.mkdir(parents=True, exist_ok=True)
  with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
    tmp = Path(stream.name)
    try:
      np.savez_compressed(stream, **arrays)
      stream.close()
      tmp.replace(path)
    finally:
      tmp.unlink(missing_ok=True)


def save_episode(path, identity, status, planned, frames, failure_terms):
  atomic_npz(
    path,
    identity=identity,
    status=status,
    planned=planned,
    frames=frames,
    failure_terms=np.asarray(failure_terms, dtype=str),
  )


def read_episode(path, identity, planned):
  """Return None for an incomplete/corrupt artifact so run can retry it."""
  try:
    with np.load(path, allow_pickle=False) as data:
      if data["identity"].item() != identity or int(data["planned"]) != planned:
        return None
      status, frames = str(data["status"].item()), data["frames"]
      if status not in ("success", "failure") or frames.ndim != 2:
        return None
      if frames.shape[1] != len(FRAME_COLUMNS) or not 0 < len(frames) <= planned:
        return None
      if (status == "success" and len(frames) != planned) or not np.isfinite(
        frames
      ).all():
        return None
      return status, frames, data["failure_terms"].tolist()
  except (OSError, KeyError, ValueError, EOFError, zipfile.BadZipFile):
    return None


def tracking_digest(frames):
  return hashlib.sha256(np.ascontiguousarray(frames)).hexdigest()


def read_controls(path, frames):
  """Older pilot files carry only arrays. New audits also bind the tracking frames."""
  try:
    with np.load(path, allow_pickle=False) as data:
      if "tracking_sha256" in data and data[
        "tracking_sha256"
      ].item() != tracking_digest(frames):
        raise ValueError("tracking digest mismatch")
      actions, noise = data["actions"], data["noise"]
    for trace, width in ((actions, 29), (noise, 64)):
      if (
        trace.shape != (len(frames), width)
        or trace.dtype.kind != "f"
        or not np.isfinite(trace).all()
      ):
        raise ValueError("invalid control trace")
  except (OSError, KeyError, ValueError, EOFError, zipfile.BadZipFile) as e:
    raise ValueError(f"{path}: {e}") from e
  return actions, noise
