"""Motion clips for SuperMimic, independent of the private mjlab fork."""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import torch

from supermimic.utils.forward_kinematics import ForwardKinematics


def resolve_motion_files(motion_file: str | list[str]) -> list[str]:
  patterns = [motion_file] if isinstance(motion_file, str) else motion_file
  files = []
  for pattern in patterns:
    matches = sorted(glob.glob(str(Path(pattern).expanduser())))
    if not matches:
      raise FileNotFoundError(f"No motion files matched {pattern!r}")
    files.extend(matches)
  if not files:
    raise ValueError("At least one motion file is required")
  return files


class MotionLoader:
  """Concatenated clips with exclusive ends and FK-consistent body targets.

  Accepts the public mjlab NPZ format (joint and full-body states), or joint
  states plus explicit ``root_{pos,quat,lin_vel,ang_vel}_w`` arrays. Quaternions
  are wxyz and all root velocities are in the world frame.
  """

  def __init__(
    self,
    motion_file: str | list[str],
    fk: ForwardKinematics,
    num_joints: int,
    device: str = "cpu",
    expected_fps: float | None = None,
  ):
    self.fk = fk
    self.files = resolve_motion_files(motion_file)
    fields = {
      name: []
      for name in (
        "joint_pos",
        "joint_vel",
        "root_pos_w",
        "root_quat_w",
        "root_lin_vel_w",
        "root_ang_vel_w",
      )
    }
    lengths = []
    for path in self.files:
      with np.load(path, allow_pickle=False) as data:
        if "fps" in data and expected_fps is not None:
          fps = float(np.asarray(data["fps"]).reshape(()))
          if not np.isclose(fps, expected_fps, rtol=1e-4):
            raise ValueError(
              f"{path}: fps={fps}, expected {expected_fps}; resample the clip first"
            )
        values = {}
        for name in fields:
          if name in data:
            value = data[name]
          elif name.startswith("root_"):
            value = data[name.replace("root_", "body_", 1)][:, 0]
          else:
            raise ValueError(f"{path}: missing {name}")
          values[name] = np.asarray(value, dtype=np.float32)
        frames = values["joint_pos"].shape[0]
        if frames == 0:
          raise ValueError(f"{path}: motion clip is empty")
        for name, value in values.items():
          width = (
            num_joints if name.startswith("joint_") else (4 if "quat" in name else 3)
          )
          if value.shape != (frames, width) or not np.isfinite(value).all():
            raise ValueError(
              f"{path}: {name} must be finite with shape {(frames, width)}"
            )
        quat = values["root_quat_w"]
        norm = np.linalg.norm(quat, axis=-1, keepdims=True)
        if np.any(norm < 1e-6):
          raise ValueError(f"{path}: root quaternion has zero norm")
        values["root_quat_w"] = quat / norm
        for name, value in values.items():
          fields[name].append(value)
        lengths.append(frames)

    for name, chunks in fields.items():
      setattr(self, name, torch.tensor(np.concatenate(chunks), device=device))
    self.motion_ends = torch.tensor(lengths, device=device, dtype=torch.long).cumsum(0)
    self.motion_starts = torch.cat(
      (self.motion_ends.new_zeros(1), self.motion_ends[:-1])
    )
    self.time_step_total = sum(lengths)
    # Compute once from root/joints so simulator targets and WM rewards agree.
    states = []
    with torch.no_grad():
      for start in range(0, self.time_step_total, 4096):
        sl = slice(start, start + 4096)
        states.append(
          self.fk(
            self.root_pos_w[sl],
            self.root_quat_w[sl],
            self.joint_pos[sl],
            self.root_lin_vel_w[sl],
            self.root_ang_vel_w[sl],
            self.joint_vel[sl],
          )
        )
    self.body_pos_w, self.body_quat_w, self.body_lin_vel_w, self.body_ang_vel_w = (
      torch.cat([state[i] for state in states]) for i in range(4)
    )

  def body_state(self, frames: torch.Tensor) -> tuple[torch.Tensor, ...]:
    return tuple(
      value[frames]
      for value in (
        self.body_pos_w,
        self.body_quat_w,
        self.body_lin_vel_w,
        self.body_ang_vel_w,
      )
    )

  def last_frame(self, frames: torch.Tensor) -> torch.Tensor:
    clips = torch.searchsorted(self.motion_ends, frames, right=True)
    return self.motion_ends[clips] - 1
