"""The common drift fixture has fixed provenance and cannot mix scoring inputs."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from supermimic.scripts import record_eit_drift as drift


def process_report(checkpoint="A", recording="recording"):
  return {
    "checkpoint_sha256": checkpoint,
    "recording_sha256": recording,
    "round": 10,
    "training_overlap_check": {"status": "distinct_motion_file"},
    "process": {
      "transitions": 249,
      "windows": 243,
      "window_frames": 8,
      "multistep_mse": {"root_pos": 0.01},
      "rpv_bias_truth_minus_prediction_right_local": [3, 4, 0, 0, 0, 0, 0, 0, 0],
      "whitened_mean": [1, 2],
      "whitened_second_moment": [[2, 2], [2, 5]],
      "process_nll": 7.0,
    },
  }


def posterior_report(checkpoint="A", recording="recording"):
  return {
    "checkpoint_sha256": checkpoint,
    "recording_sha256": recording,
    "fixed_record_posterior": [{"samples": 50, "coverage_95": 0.1}],
  }


def test_supine_fixture_reproduces_documented_pose_and_velocities(tmp_path):
  path = tmp_path / "supine.npz"
  source = drift.generate_supine_motion(path)
  assert source["frames"] == 250 and source["duration_s"] == 5
  assert source["joint_pose_source"].endswith("KNEES_BENT_KEYFRAME.joint_pos")
  assert len(source["motion_sha256"]) == len(source["robot_xml_sha256"]) == 64
  with np.load(path, allow_pickle=False) as motion:
    assert motion["joint_pos"].shape == (250, 29)
    assert float(motion["fps"]) == 50
    np.testing.assert_allclose(motion["root_pos_w"], [[0, 0, 0.12]] * 250)
    np.testing.assert_allclose(
      motion["root_quat_w"], [[2**-0.5, 0, -(2**-0.5), 0]] * 250
    )
    for name in ("joint_vel", "root_lin_vel_w", "root_ang_vel_w"):
      assert not motion[name].any()
    np.testing.assert_array_equal(motion["joint_pos"][0], motion["joint_pos"][-1])
  with pytest.raises(FileExistsError):
    drift.generate_supine_motion(path)


def test_mean_noise_and_posterior_evidence_are_separate_and_hand_checkable():
  report = drift.summarize_evidence(process_report(), posterior_report())
  assert report["mean_model"]["rpv_bias_norms"]["rotation"] == 5
  assert report["process_noise"]["whitened_mean_norm"] == pytest.approx(5**0.5)
  assert report["process_noise"]["whitened_second_moment_diagonal_mean"] == 3.5
  assert report["process_noise"]["whitened_centered_covariance_diagonal_mean"] == 1
  assert report["process_noise"]["whitened_second_moment_off_diagonal_rms"] == 2
  assert report["posterior_time_bins"] == posterior_report()["fixed_record_posterior"]
  with pytest.raises(ValueError, match="same recording"):
    drift.summarize_evidence(process_report(), posterior_report(recording="other"))
  with pytest.raises(ValueError, match="same checkpoint"):
    drift.summarize_evidence(process_report(), posterior_report(checkpoint="B"))


def test_first_action_failure_preserves_missing_process_evidence():
  process = process_report()
  process["process"].update(
    transitions=0,
    windows=0,
    process_nll=None,
    whitened_mean=None,
    whitened_second_moment=None,
    rpv_bias_truth_minus_prediction_right_local=None,
  )
  report = drift.summarize_evidence(process, posterior_report())
  assert report["process_noise"]["whitened_mean_norm"] is None
  assert report["mean_model"]["rpv_bias_norms"]["rotation"] is None
  json.dumps(report, allow_nan=False)


def test_all_groups_consume_one_record_and_failure_is_not_promoted_to_full_clip(
  tmp_path, monkeypatch
):
  completed = True
  calls = []

  def load(path, device):
    if path.name == "episodes.pt":
      length = 250 if completed else 12
      return {
        "episodes": [
          {
            "reason": "clip_complete" if completed else "physical_failure",
            "reward": torch.zeros(length),
            "time_index": torch.arange(length),
          }
        ]
      }, "recording"
    return {"stage": "policy", "group": path.stem}, path.stem

  def evaluate(*args, **kwargs):
    assert kwargs["steps"] is None and kwargs["protocol"] == "full_clip"
    assert kwargs["reference_controller"] and not kwargs["posterior_diagnostics"]
    kwargs["record_path"].write_bytes(b"record")
    return {"checkpoint_sha256": "B"}

  def diagnose(checkpoint, recording, **kwargs):
    calls.append((checkpoint, recording, kwargs["mode"]))
    group = drift.Path(checkpoint).stem
    return (
      process_report(group) if kwargs["mode"] == "process" else posterior_report(group)
    )

  monkeypatch.setattr(drift, "load_hashed_checkpoint", load)
  monkeypatch.setattr(
    drift,
    "evaluation_config",
    lambda state: SimpleNamespace(
      estimator_inputs=SimpleNamespace(group=state["group"])
    ),
  )
  monkeypatch.setattr(drift, "evaluate", evaluate)
  monkeypatch.setattr(drift, "diagnose", diagnose)
  checkpoints = [tmp_path / f"{group}.pt" for group in "ABC"]
  output = tmp_path / "complete"
  report = drift.record_and_score(checkpoints[1], checkpoints, output)
  assert report["continuous_record_complete"]
  assert set(report["groups"]) == set("ABC")
  assert len(calls) == 6 and {call[1] for call in calls} == {output / "episodes.pt"}
  with pytest.raises(ValueError, match="new or empty"):
    drift.record_and_score(checkpoints[1], checkpoints, output)
  completed = False
  partial = tmp_path / "partial"
  with pytest.raises(RuntimeError, match="uninterrupted"):
    drift.record_and_score(checkpoints[1], checkpoints, partial)
  result = json.loads((partial / "report.json").read_text())
  assert not result["continuous_record_complete"]
  assert result["recorded_frames"] == [12]
