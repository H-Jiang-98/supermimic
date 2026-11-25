"""Record and score a shared five-second, ground-supported MuJoCo drift fixture.

This generated supine reference is a diagnostic input, not walking performance.
The reference-joint-position controller is independent of the evaluated actors.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import mujoco
import numpy as np
import torch
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
  G1_XML,
  KNEES_BENT_KEYFRAME,
)

from supermimic.runners.eit_inference import evaluation_config, load_hashed_checkpoint
from supermimic.runners.eit_snapshots import file_sha256
from supermimic.scripts.check_estimator import stationary_motion
from supermimic.scripts.diagnose_eit import diagnose
from supermimic.scripts.eval_eit import evaluate


def generate_supine_motion(path: Path, *, frames=250):
  """Reproduce docs/eit_operations.md, retaining real reset and failure rules."""
  if type(frames) is not int or frames < 8:
    raise ValueError("Drift fixture requires at least eight frames")
  joint_pose = KNEES_BENT_KEYFRAME.joint_pos
  assert joint_pose is not None, "G1 drift fixture requires the knees-bent joint pose"
  path = Path(path)
  if path.exists():
    raise FileExistsError(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  stationary_motion(path, frames=frames)
  model = mujoco.MjModel.from_xml_path(str(G1_XML))
  with np.load(path, allow_pickle=False) as motion:
    arrays = dict(motion)
  arrays["root_pos_w"][:] = (0, 0, 0.12)
  arrays["root_quat_w"][:] = (2**-0.5, 0, -(2**-0.5), 0)
  joints = np.flatnonzero(model.jnt_type == int(mujoco.mjtJoint.mjJNT_HINGE))
  for column, joint_id in enumerate(joints):
    for pattern, angle in joint_pose.items():
      if re.fullmatch(pattern, model.joint(joint_id).name):
        arrays["joint_pos"][:, column] = angle
  np.savez(path, **arrays)
  return {
    "kind": "generated_supine_joint_position_reference_v1",
    "source": "docs/eit_operations.md#固定后验漂移样例",
    "robot_xml": str(G1_XML),
    "robot_xml_sha256": file_sha256(Path(G1_XML)),
    "joint_pose_source": "mjlab.unitree_g1.KNEES_BENT_KEYFRAME.joint_pos",
    "joint_pose_patterns": dict(joint_pose),
    "frames": frames,
    "fps": 50.0,
    "duration_s": frames / 50.0,
    "root_position_m": [0, 0, 0.12],
    "root_quaternion_wxyz": [2**-0.5, 0, -(2**-0.5), 0],
    "reference_velocities": "all zero",
    "motion_sha256": file_sha256(path),
  }


def summarize_evidence(process, posterior):
  """Keep separate evidence for mean bias, process scale and filtering drift."""
  values = process["process"]
  if process["recording_sha256"] != posterior["recording_sha256"]:
    raise ValueError("Process and posterior scores must use the same recording")
  if process["checkpoint_sha256"] != posterior["checkpoint_sha256"]:
    raise ValueError("Process and posterior scores must use the same checkpoint")
  noise = {
    "process_nll": values["process_nll"],
    "whitened_mean_norm": None,
    "whitened_second_moment_diagonal_mean": None,
    "whitened_centered_covariance_diagonal_mean": None,
    "whitened_second_moment_off_diagonal_rms": None,
    "calibrated_reference": "whitened mean 0; second moment identity",
  }
  bias = values["rpv_bias_truth_minus_prediction_right_local"]
  if values["transitions"]:
    mean = np.asarray(values["whitened_mean"], dtype=np.float64)
    second = np.asarray(values["whitened_second_moment"], dtype=np.float64)
    centered = second - np.outer(mean, mean)
    off_diagonal = second - np.diag(np.diag(second))
    noise.update(
      whitened_mean_norm=float(np.linalg.norm(mean)),
      whitened_second_moment_diagonal_mean=float(np.trace(second) / len(mean)),
      whitened_centered_covariance_diagonal_mean=float(np.trace(centered) / len(mean)),
      whitened_second_moment_off_diagonal_rms=float(
        np.sqrt(np.square(off_diagonal).sum() / (len(mean) * (len(mean) - 1)))
      ),
    )
  return {
    "checkpoint_sha256": process["checkpoint_sha256"],
    "recording_sha256": process["recording_sha256"],
    **{
      key: process[key]
      for key in ("round", "collected_steps", "update_groups", "collected_frames")
      if key in process
    },
    "training_overlap_check": process["training_overlap_check"],
    "mean_model": {
      "transitions": values["transitions"],
      "windows": values["windows"],
      "window_frames": values["window_frames"],
      "multistep_mse": values["multistep_mse"],
      "rpv_signed_bias_truth_minus_prediction_right_local": bias,
      "rpv_bias_norms": {
        name: float(np.linalg.norm(bias[offset : offset + 3]))
        if bias is not None
        else None
        for name, offset in (("rotation", 0), ("position", 3), ("velocity", 6))
      },
    },
    "process_noise": noise,
    "posterior_time_bins": posterior["fixed_record_posterior"],
  }


def _write_json(path, value):
  path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def record_and_score(
  checkpoint,
  checkpoints,
  output,
  *,
  device="cpu",
  seed=90817,
  frames=250,
  batch_size=128,
  bin_steps=50,
):
  """Create a new output directory; preserve partial evidence if a stage fails."""
  output = Path(output).resolve()
  if output.exists() and (not output.is_dir() or any(output.iterdir())):
    raise ValueError("Use a new or empty drift output directory")
  groups = {}
  for path in checkpoints:
    path = Path(path).resolve()
    saved, digest = load_hashed_checkpoint(path, "cpu")
    cfg = evaluation_config(saved)
    assert cfg.estimator_inputs is not None  # Guaranteed by evaluation_config.
    group = cfg.estimator_inputs.group
    if group in groups:
      raise ValueError(f"Duplicate diagnostic group {group}")
    if saved["stage"] != "policy":
      raise ValueError("Drift comparison expects policy-stage A/B/C checkpoints")
    groups[group] = {"checkpoint": str(path), "sha256": digest}
    del saved
  if set(groups) != {"A", "B", "C"}:
    raise ValueError("Supply exactly one checkpoint for each of A/B/C")
  checkpoint = Path(checkpoint).resolve()
  source, source_digest = load_hashed_checkpoint(checkpoint, "cpu")
  source_cfg = evaluation_config(source)
  assert source_cfg.estimator_inputs is not None  # Guaranteed by evaluation_config.
  source_group = source_cfg.estimator_inputs.group
  del source
  output.mkdir(parents=True, exist_ok=True)
  motion, recording = output / "supine.npz", output / "episodes.pt"
  manifest = {
    "version": "eit_continuous_drift_v1",
    "scope": "independent diagnostic fixture; excluded from walking evaluation",
    "source_checkpoint": str(checkpoint),
    "source_checkpoint_sha256": source_digest,
    "source_group": source_group,
    "seed": seed,
    "device": device,
    "controller": "reference_joint_positions",
    "fixture": generate_supine_motion(motion, frames=frames),
    "groups": groups,
  }
  _write_json(output / "manifest.json", manifest)
  evaluation = evaluate(
    checkpoint,
    motion,
    device=device,
    num_envs=1,
    seed=seed,
    steps=None,
    protocol="full_clip",
    posterior_diagnostics=False,
    record_path=recording,
    reference_controller=True,
  )
  _write_json(output / "recording_evaluation.json", evaluation)
  if evaluation["checkpoint_sha256"] != source_digest:
    raise ValueError("Recording checkpoint changed while preparing the fixture")
  record, recording_digest = load_hashed_checkpoint(recording, "cpu")
  episodes = record["episodes"]
  complete = (
    len(episodes) == 1
    and episodes[0]["reason"] == "clip_complete"
    and len(episodes[0]["reward"]) == frames
    and torch.equal(episodes[0]["time_index"], torch.arange(frames))
  )
  report = {
    "version": "eit_continuous_drift_report_v1",
    "scope": manifest["scope"],
    "recording_sha256": recording_digest,
    "continuous_record_complete": complete,
    "recorded_frames": [len(ep["reward"]) for ep in episodes],
    "termination_reasons": [ep["reason"] for ep in episodes],
    "groups": {},
    "interpretation": (
      "All groups consume identical recorded truth, actions, measurements and prior. "
      "Signed one-step bias and multistep MSE describe the mean model; whitened "
      "residual moments describe bias/scale/correlation mismatch in process noise; "
      "time-binned posterior errors and coverage describe the combined filter. "
      "These are localization evidence, not a causal ablation or walking score."
    ),
  }
  del record
  _write_json(output / "report.json", report)
  with torch.compiler.set_stance("force_eager"):
    for group, item in sorted(groups.items()):
      process = diagnose(
        item["checkpoint"],
        recording,
        mode="process",
        device=device,
        batch_size=batch_size,
        window_frames=8,
      )
      _write_json(output / f"{group}_process.json", process)
      posterior = diagnose(
        item["checkpoint"],
        recording,
        mode="posterior",
        device=device,
        bin_steps=bin_steps,
      )
      _write_json(output / f"{group}_posterior.json", posterior)
      if process["checkpoint_sha256"] != item["sha256"]:
        raise ValueError(f"Group {group} checkpoint changed during scoring")
      report["groups"][group] = summarize_evidence(process, posterior)
      _write_json(output / "report.json", report)
  if not complete:
    raise RuntimeError(
      "Supine diagnostic did not produce the requested uninterrupted clip; "
      "partial recording and scores retained in the output directory"
    )
  return report


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--checkpoints", type=Path, nargs=3, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--seed", type=int, default=90817)
  parser.add_argument("--frames", type=int, default=250)
  parser.add_argument("--batch-size", type=int, default=128)
  parser.add_argument("--bin-steps", type=int, default=50)
  parser.add_argument("--cpu-threads", type=int, default=1)
  args = parser.parse_args()
  if args.cpu_threads < 1 or args.batch_size < 1 or args.bin_steps < 1:
    parser.error("threads, batch size and bin steps must be positive")
  torch.set_num_threads(args.cpu_threads)
  kwargs = vars(args)
  del kwargs["cpu_threads"]
  report = record_and_score(**kwargs)
  print(
    json.dumps(
      {
        "output": str(args.output),
        "continuous_record_complete": report["continuous_record_complete"],
        "recorded_frames": report["recorded_frames"],
        "groups": sorted(report["groups"]),
      },
      allow_nan=False,
    )
  )


if __name__ == "__main__":
  main()
