from pathlib import Path

import numpy as np
import pytest
import torch

from supermimic.motion_loader import MotionLoader


def test_public_and_root_formats_have_same_targets(fk, motion_file):
  public = MotionLoader(motion_file("public.npz", public=True), fk, 29)
  root = MotionLoader(motion_file("root.npz"), fk, 29)
  for a, b in zip(
    public.body_state(torch.arange(6)), root.body_state(torch.arange(6)), strict=True
  ):
    torch.testing.assert_close(a, b)
  # Stored public non-root bodies are intentionally bogus: FK defines targets.
  assert not torch.equal(public.body_pos_w[:, 0], public.body_pos_w[:, -1])


def test_multiple_files_preserve_clip_boundaries(fk, motion_file):
  paths = [motion_file("a.npz", frames=3), motion_file("b.npz", frames=5, root_x=10)]
  motion = MotionLoader(paths, fk, 29)
  assert motion.motion_ends.tolist() == [3, 8]
  assert motion.motion_starts.tolist() == [0, 3]
  assert motion.last_frame(torch.tensor([0, 2, 3, 7])).tolist() == [2, 2, 7, 7]
  assert motion.root_pos_w[:, 0].tolist() == [0.0] * 3 + [10.0] * 5
  assert motion.body_state(torch.tensor([[0, 3], [2, 7]]))[0].shape == (2, 2, 30, 3)
  globbed = MotionLoader(str(Path(paths[0]).parent / "*.npz"), fk, 29)
  torch.testing.assert_close(globbed.motion_ends, motion.motion_ends)


@pytest.mark.parametrize("case", ["empty", "bad_fps", "nan", "bad_joints", "zero_quat"])
def test_bad_motion_is_rejected(fk, motion_file, case):
  path = motion_file(
    frames=0 if case == "empty" else 3, fps=60 if case == "bad_fps" else 50
  )
  with np.load(path) as archive:
    data = dict(archive)
  if case == "nan":
    data["joint_pos"][0, 0] = np.nan
  if case == "bad_joints":
    data["joint_pos"] = data["joint_pos"][:, :-1]
  if case == "zero_quat":
    data["root_quat_w"][:] = 0
  np.savez(path, **data)
  with pytest.raises(ValueError):
    MotionLoader(path, fk, 29, expected_fps=50)


def test_missing_file_fails_before_loading(fk, tmp_path):
  with pytest.raises(FileNotFoundError):
    MotionLoader(str(tmp_path / "*.npz"), fk, 29)
