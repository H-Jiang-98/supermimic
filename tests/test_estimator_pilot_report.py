"""Pilot audit distinguishes missing evidence, bad budgets and physical failures."""

import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from supermimic.runners.eit_snapshots import file_sha256
from supermimic.scripts import batch_eval_eit, report_eit_pilot
from supermimic.scripts.benchmark_eit_online import _summarize
from supermimic.scripts.report_eit_pilot import PHASES, build_report, plot_report


def write(path, value, *, lines=False):
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(
    "\n".join(json.dumps(row) for row in value) + "\n" if lines else json.dumps(value)
  )


def verification_execution(*, cuda=False):
  return {
    "torch_compile": "force_eager",
    "torchscript_optimized_execution": False,
    "torchscript_import_mode": "disabled_before_import",
    "deterministic_algorithms": True,
    "fixed_input_assembly": "cpu_contiguous_then_shared_device_constructors_v1",
    "discarded_private_policy_warmup_updates": 1,
    "cublas_workspace_config": ":4096:8" if cuda else None,
    "expected_optimizer_origin": "deepcopy_of_live_parameter_optimizer_graph",
  }


def fixture(root, monkeypatch):
  monkeypatch.setattr(
    report_eit_pilot, "_initialization_identity", lambda audit, p: "f" * 64
  )
  p = {
    "rounds": 2,
    "num_envs": 1,
    "collection_steps": 4,
    "training_seed": 7,
    "resume_after_round": 1,
    "diagnostic_seed": 9,
    "validation_seed": 8,
    "evaluation_envs": 2,
    "motion_sha256": "m" * 64,
    "online_device": "cpu",
    "device": "cpu",
    "training_process_environment": {"PYTORCH_JIT": "0"},
    "cpu_threads": 1,
    "mean": {"num_batches": 2, "batch_size": 8},
    "covariance": {"batch_size": 2, "epochs": 1},
    "policy": {
      "num_batches": 1,
      "critic_epochs": 2,
      "batch_size": 2,
      "num_samples": 2,
      "horizon": 3,
      "microbatch_size": 1,
    },
    "expected_per_group": {
      "collected_frames": 8,
      "mean_updates": 4,
      "actor_updates": 2,
      "critic_updates": 4,
    },
  }
  write(root / "protocol.json", p)
  validation = {
    "complete": True,
    "pending_validation": [],
    "baselines": [],
    "trained": [],
    "selected": [],
  }
  commands = []
  for group in "ABC":
    snapshots, metrics, profiles = [], [], []
    for number in range(3):
      frames = number * 4
      snapshot = root / group / "snapshots" / f"frames_{frames}.pt"
      snapshot.parent.mkdir(parents=True, exist_ok=True)
      snapshot.write_bytes(f"snapshot-{group}-{number}".encode())
      sha = file_sha256(snapshot)
      index = {
        "path": str(snapshot.relative_to(root / group)),
        "sha256": sha,
        "collected_frames": frames,
        "round": number,
        "stage": "policy",
        "group": group,
        "training_seed": 7,
        "initialization_digest": "f" * 64,
        "mean_steps": number * 2,
        "cov_steps": number * 2,
        "actor_steps": number,
        "critic_steps": number * 2,
      }
      snapshots.append(index)
      validation["trained" if number else "baselines"].append(
        {
          "candidate": deepcopy(index),
          "complete": True,
          "tasks": [{"status": "complete"}],
          "expected_episodes": 2,
          "recorded_episodes": 2,
          # Both episodes failed physically, but this is a valid complete result.
          "scores": {"success_rate": 0.0, "return": number + 0.5, "progress": 0.001},
        }
      )
      folder = root / "diagnostics" / group / str(frames)
      write(
        folder / "evaluation.json",
        {
          "checkpoint_sha256": sha,
          "group": group,
          "evaluation_complete": True,
          "censored_episodes": 0,
          "timeout_count": 0,
          "protocol": "full_clip_v1",
          "step_budget": None,
          "seed": 9,
          "evaluation_motion_sha256": p["motion_sha256"],
          "actor_state_source": "posterior",
          "episodes": [
            {"env_id": i, "length": 2, "reason": "failure"} for i in range(2)
          ],
        },
      )
      torch.save(
        {
          "version": "eit_diagnostic_episodes_v1",
          "source": "mujoco_evaluation",
          "split": "diagnostic",
          "checkpoint_sha256": sha,
          "group": group,
          "actor_state_source": "posterior",
          "seed": 9,
          "motion_sha256": p["motion_sha256"],
          "motion_digest": "d" * 64,
          "control_dt": 0.02,
          "episodes": [
            {
              "env_id": i,
              "reward": torch.ones(2),
              "reason": "failure",
              "start_frame": 0,
              "reference_steps": 100,
            }
            for i in range(2)
          ],
        },
        folder / "episodes.pt",
      )
      posterior_bin = {
        "start_s": 0.0,
        "end_s": 1.0,
        "surviving_episodes": 2,
        "position_rmse_m": 0.1,
        "rotation_rmse_rad": 0.1,
        "velocity_rmse_m_s": 0.1,
        "mean_innovation_norm": 0.1,
        "null_direction_violations": 0,
        "jitter_frames": 0,
      }
      value_error = {"samples": 4, "bias": -2.0, "mae": 2.0, "rmse": 2.0}
      write(
        folder / "offline.json",
        {
          "version": "eit_independent_diagnostics_v1",
          "episodes": 2,
          "checkpoint_sha256": sha,
          "group": group,
          "training_overlap_check": {"status": "distinct_seed"},
          "recording_seed": 9,
          "recording_source": "mujoco_evaluation",
          "recording_sha256": file_sha256(folder / "episodes.pt"),
          "recording_checkpoint_sha256": sha,
          "actor_state_source": "posterior",
          "motion_digest": "d" * 64,
          "fixed_record_posterior": [
            {**posterior_bin, "frames": 3, "coverage_95": 0.5},
            {**posterior_bin, "frames": 1, "coverage_95": 1.0},
          ],
          "critic": {
            "actor_state_source": "posterior",
            "gamma": 0.99,
            "bootstrap_value_cap": 10.0,
            "raw_value_error": value_error,
            "bootstrap_value_error": value_error,
            "censored_samples": 0,
            "censored_episodes": 0,
            "episodes": [
              {
                "env_id": i,
                "reason": "failure",
                "raw_value": [0.0, 0.0],
                "bootstrap_value": [0.0, 0.0],
                "discounted_return": [2.0, 2.0],
              }
              for i in range(2)
            ],
          },
          "process": {
            "transitions": 2,
            "process_nll": 1.0,
            "windows": 2,
            "window_frames": 2,
            "window_scored_successors": 2,
            "multistep_mse": dict.fromkeys(
              ("root_pos", "root_ori", "joint_pos", "joint_vel"), 0.1
            ),
            "rpv_bias_truth_minus_prediction_right_local": [0.0] * 9,
            "whitened_mean": [0.0] * 70,
            "whitened_second_moment": torch.eye(70).tolist(),
          },
        },
      )
      if number:
        metrics.append(
          {
            "round": number,
            "collected_frames": frames,
            "budget": {
              "collected_frames": 4,
              "mean_updates": 2,
              "sampled_training_windows": 16,
              "actor_updates": 1,
              "critic_updates": 2,
              "covariance_batch_size": 2,
              "covariance_epochs": 1,
              "policy_n": 2,
              "policy_k": 2,
              "policy_h": 3,
              "policy_microbatch": 1,
              "available_training_windows": 1,
              "training_pairs": 3,
              "validation_pairs": 1,
              "covariance_updates": 2,
              "covariance_pair_draws": 3,
              "imagined_steps": 12,
              "survival_weighted_imagined_steps": 11.5,
              "physical_terminations": 2,
              "physical_episode_endings": 2,
            },
            "policy": {
              "Policy/return": 5.0,
              "Policy/bootstrap_fraction": 0.8,
              "Policy/action_saturation": 0.0,
              "Policy/film_gamma": 0.01,
              "Policy/film_beta": 0.02,
            },
          }
        )
        profiles.append(
          {
            "round": number,
            "seconds": dict.fromkeys(PHASES, 0.1),
            "runtime": {"device": "cpu"},
          }
        )
    write(root / group / "snapshots/index.json", {"snapshots": snapshots})
    write(root / group / "metrics.jsonl", metrics, lines=True)
    write(root / group / "profiling.jsonl", profiles, lines=True)
    reference = root / "resume" / f"{group}-reference.pt"
    reference.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
      {
        "version": "eit_live_update_reference_v1",
        "source": "live_training_before_checkpoint_commit",
        "checkpoint_sha256": snapshots[1]["sha256"],
        "round": 1,
        "group": group,
        "device": "cpu",
        "verification_execution": verification_execution(),
      },
      reference,
    )
    write(
      root / "resume" / f"{group}.json",
      {
        "version": "eit_fixed_input_resume_v2",
        "device": "cpu",
        "verification_execution": verification_execution(),
        "live_optimizer_reference": True,
        "expected_source": "live_training_before_checkpoint_commit",
        "reference_version": "eit_live_update_reference_v1",
        "reference_path": str(reference),
        "reference_sha256": file_sha256(reference),
        "group": group,
        "round": 1,
        "checkpoint_sha256": snapshots[1]["sha256"],
        "status": "passed",
        "fixed_input_policy_resume_exact": True,
        "fixed_input_model_resume_exact": True,
      },
    )
    write(
      root / "online" / f"{group}.json",
      {
        "version": "eit_online_latency_v1",
        "group": group,
        "checkpoint_sha256": snapshots[-1]["sha256"],
        "batch_size": 1,
        "posterior_diagnostics": False,
        "device": "cpu",
        "motion_sha256": p["motion_sha256"],
        "motion_digest": "d" * 64,
        "cpu_threads": 1,
        "cpu_interop_threads": 1,
        "dtype": "torch.float32",
        "seed": 2026,
        "warmup_steps": 1,
        "measured_steps": 2,
        "control_period_ms": 20.0,
        "total": timing_fixture(stages=False),
        "stages": timing_fixture(stages=True),
      },
    )
    commands.append(
      {
        "name": f"train-{group}-to-2",
        "returncode": 0,
        "argv": ["train_eit", "--resume", "training.pt"],
      }
    )
  write(root / "evaluation/validation.json", validation)
  write(root / "commands.jsonl", commands, lines=True)
  # Checkpoint admission/physical configuration already have dedicated integration
  # tests. Keep those boundaries small here while exercising the real task-key,
  # episode identity, censoring, aggregation and ranking validators below.
  scene = {
    "path": str(root / "walking.npz"),
    "sha256": p["motion_sha256"],
    "seed": 8,
    "num_envs": 2,
    "episodes": 2,
    "protocol": "full_clip_v1",
    "step_budget": None,
    "evaluation_motion_digest": "d" * 64,
    "clip_frames": 100,
    "control_period_s": 0.02,
    "timeout_steps": 101,
    "timeout_s": 2.02,
    "device": "cpu",
    "initial_prior": {},
    "measurement_noise": {},
  }
  manifest = {
    "version": batch_eval_eit.VERSION,
    "runs": [
      {"directory": str(root / g), "group": g, "training_seed": 7, "frames": [0, 4, 8]}
      for g in "ABC"
    ],
    "validation": [scene],
  }
  manifest["manifest_digest"] = batch_eval_eit.digest(manifest)
  write(root / "manifest.json", manifest)
  candidates = []
  for group in "ABC":
    for indexed in json.loads((root / group / "snapshots/index.json").read_text())[
      "snapshots"
    ]:
      candidate = {
        **indexed,
        "training_motion_digest": "training-motion",
        "initial_prior": {},
        "measurement_noise": {},
      }
      candidates.append(candidate)
      key = batch_eval_eit.task_key(candidate, scene)
      result = {
        **scene,
        "group": group,
        "checkpoint_sha256": candidate["sha256"],
        "training_motion_digest": "training-motion",
        "evaluation_motion_sha256": p["motion_sha256"],
        "actor_state_source": "posterior",
        "actor_backend": "pytorch",
        "mode": "eval",
        "time_limit_protocol": "full_clip_v1",
        "episodes": [
          {
            "env_id": i,
            "length": 2,
            "reference_steps": 100,
            "reason": "failure",
            "censored": False,
            "start_frame": 0,
            "metric_steps": 1,
            "return": candidate["round"] + 0.5,
            "metrics": dict.fromkeys(batch_eval_eit.TRACKING, 0.1),
          }
          for i in range(2)
        ],
      }
      write(
        root / "evaluation/tasks" / f"{key}.json",
        {
          "task_key": key,
          "snapshot_sha256": candidate["sha256"],
          "scene": scene,
          "result": result,
        },
      )
  monkeypatch.setattr(
    batch_eval_eit, "candidates", lambda manifest: deepcopy(candidates)
  )
  monkeypatch.setattr(batch_eval_eit, "validate_scene", lambda scene: None)
  records = [
    batch_eval_eit.evaluate_candidate(c, [scene], root / "evaluation", False)
    for c in candidates
  ]
  validation.update(
    manifest_digest=manifest["manifest_digest"],
    baselines=[r for r in records if r["candidate"]["collected_frames"] == 0],
    trained=[r for r in records if r["candidate"]["collected_frames"] > 0],
    selected=batch_eval_eit.select(records),
  )
  write(root / "evaluation/validation.json", validation)
  supplementary_fixture(root, p)
  return p


def timing_fixture(*, stages):
  rows = []
  for index in range(2):
    times = {"total": 30.0}
    if stages:
      times.update(
        measurement_history=1.0,
        filter_and_jacobian=23.0,
        features=1.0,
        actor=2.0,
        clamp_and_commit=2.0,
      )
    rows.append(
      {
        "sample": index,
        "reset": False,
        "episode_step": index + 1,
        "timings_ms": times,
        "memory": None,
      }
    )
  return {
    "samples": rows,
    "statistics": _summarize(rows, 20.0),
    "sample_reset_count": 0,
    "first_reset": {"environment_reset": 1.0, "adapter_reset": 1.0, "first_act": 1.0},
  }


def supplementary_fixture(root, p):
  final = p["rounds"] * p["num_envs"] * p["collection_steps"]
  final_digests = {}
  for group in "ABC":
    path = root / "diagnostics" / group / str(final) / "offline.json"
    data = json.loads(path.read_text())
    final_digests[group] = data["checkpoint_sha256"]
    data["actor_gradients"] = {
      "objective": "mean_positive_discounted_soft_survival_return",
      "gradient_scope": "all_actor_parameters_before_clipping",
      "start_source": "independent_recorded_episode_heads",
      "actor_state_source": "truth" if group == "A" else "posterior",
      "starts": 2,
      "seed": 2028,
      "episode_ids": [0, 1],
      "start_frames": [0, 0],
      "effective_imagined_steps": 2
      * p["policy"]["num_samples"]
      * p["policy"]["horizon"],
      "samples_per_start": p["policy"]["num_samples"],
      "horizon": p["policy"]["horizon"],
      "microbatch_size": p["policy"]["microbatch_size"],
      "reward_gradient_norm": 1.0,
      "bootstrap_gradient_norm": 0.0,
      "total_gradient_norm": 1.0,
      "reward_return": 1.0,
      "bootstrap_return": 2.0,
      "total_return": 3.0,
      "gradient_dot_product": 0.0,
      "process_noise_sha256": "e" * 64,
      "measurement_noise_sha256": "a" * 64,
      "gradient_cosine": None,
      "gradient_angle_degrees": None,
    }
    write(path, data)
  source = root / "diagnostics/A" / str(final)
  dest = source.with_name(str(final) + "-truth")
  dest.mkdir()
  record = torch.load(source / "episodes.pt", weights_only=True)
  record["actor_state_source"] = "truth"
  torch.save(record, dest / "episodes.pt")
  for name in ("evaluation", "offline"):
    data = json.loads((source / f"{name}.json").read_text())
    data["actor_state_source"] = "truth"
    if name == "offline":
      data["recording_sha256"] = file_sha256(dest / "episodes.pt")
      data["critic"]["actor_state_source"] = "truth"
      del data["actor_gradients"]
    write(dest / f"{name}.json", data)
  baseline = root / "diagnostics/B/0/episodes.pt"
  reference = root / "diagnostics/common-walking-reference/episodes.pt"
  reference.parent.mkdir()
  record = torch.load(baseline, weights_only=True)
  record["actor_state_source"] = "reference_joint_positions"
  torch.save(record, reference)
  common = {"policy-baseline": baseline, "walking-reference": reference}
  write(
    root / "diagnostics/common-recordings.json",
    {"recordings": {k: str(v) for k, v in common.items()}},
  )
  for name, record_path in common.items():
    for group in "ABC":
      for mode in ("process", "posterior"):
        data = json.loads((root / "diagnostics/B/0/offline.json").read_text())
        data.update(
          group=group,
          checkpoint_sha256=final_digests[group],
          recording_sha256=file_sha256(record_path),
        )
        data["process"]["process_nll"] = 1.0
        write(root / "diagnostics/common" / name / f"{group}-{mode}.json", data)


def test_complete_artifacts_allow_physical_failures_and_slow_latency(
  tmp_path, monkeypatch
):
  fixture(tmp_path, monkeypatch)
  report = build_report(tmp_path)
  assert report["status"] == "complete", report
  assert report["pending"] == report["failures"] == []
  assert report["step7_ready"] is False
  assert report["groups"]["A"]["actual_budget"]["covariance_updates"] == 4
  assert report["groups"]["A"]["curves"]["coverage_95"][0]["value"] == 0.625
  assert (
    report["groups"]["A"]["online_continuous_full_filter"]["p99_headroom_ms"] == -10
  )
  assert report["groups"]["A"]["phase_costs"]["round_seconds"]["total_seconds"] == 0.2


def set_diagnostic_terminations(folder, reasons, *, version=2):
  record = torch.load(folder / "episodes.pt", weights_only=True)
  evaluated = json.loads((folder / "evaluation.json").read_text())
  offline = json.loads((folder / "offline.json").read_text())
  offline["version"] = f"eit_independent_diagnostics_v{version}"
  terminal = {"failure", "clip_complete"} if version == 1 else {"failure"}
  critic = offline["critic"]
  censored_frames = censored_episodes = 0
  for ep, physical, scored, reason in zip(
    record["episodes"], evaluated["episodes"], critic["episodes"], reasons, strict=True
  ):
    ep["reason"] = physical["reason"] = scored["reason"] = reason
    if reason == "clip_complete":
      ep["reference_steps"] = len(ep["reward"])
    if reason not in terminal:
      scored["discounted_return"] = None
      censored_frames += len(ep["reward"])
      censored_episodes += 1
  critic.update(censored_samples=censored_frames, censored_episodes=censored_episodes)
  for name in ("raw_value_error", "bootstrap_value_error"):
    score = critic[name]
    score["samples"] = (
      sum(len(ep["reward"]) for ep in record["episodes"]) - censored_frames
    )
    if not score["samples"]:
      score.update(bias=None, mae=None, rmse=None)
  evaluated["censored_episodes"] = sum(
    reason in ("timeout", "evaluation_truncated") for reason in reasons
  )
  evaluated["timeout_count"] = reasons.count("timeout")
  evaluated["evaluation_complete"] = evaluated["censored_episodes"] == 0
  if version == 1 and "actor_gradients" in offline:
    gradient = offline["actor_gradients"]
    gradient["effective_imagined_steps"] = gradient["samples_per_start"] * sum(
      min(gradient["horizon"], ep["reference_steps"]) for ep in record["episodes"]
    )
  torch.save(record, folder / "episodes.pt")
  offline["recording_sha256"] = file_sha256(folder / "episodes.pt")
  write(folder / "evaluation.json", evaluated)
  write(folder / "offline.json", offline)


@pytest.mark.parametrize(
  "version,reasons,status,samples",
  [
    (2, ("clip_complete", "clip_complete"), "unscorable_censored", 0),
    (2, ("failure", "clip_complete"), "scored", 2),
    (1, ("clip_complete", "clip_complete"), "scored", 4),
  ],
)
def test_completed_clips_do_not_block_critic_audit(
  tmp_path, monkeypatch, version, reasons, status, samples
):
  fixture(tmp_path, monkeypatch)
  for relative in ("A/8", "B/8", "C/8", "A/8-truth"):
    set_diagnostic_terminations(
      tmp_path / "diagnostics" / relative, reasons, version=version
    )
  report = build_report(tmp_path)
  assert report["status"] == "complete", report
  assert report["pending"] == report["failures"] == []
  extra = report["supplementary"]
  assert extra["A_truth_critic_status"] == status
  assert extra["A_truth_critic_error"]["samples"] == samples
  if not samples:
    assert extra["A_truth_critic_error"]["rmse"] is None
    assert report["groups"]["A"]["curves"]["critic_rmse"][-1]["value"] is None


@pytest.mark.parametrize("reason", ["timeout", "evaluation_truncated"])
def test_censored_truth_evaluation_remains_pending(tmp_path, monkeypatch, reason):
  fixture(tmp_path, monkeypatch)
  set_diagnostic_terminations(tmp_path / "diagnostics/A/8-truth", (reason, reason))
  report = build_report(tmp_path)
  assert report["status"] == "pending", report
  assert not report["failures"]
  assert (
    "A: independent truth-input episodes incomplete or censored" in report["pending"]
  )


def test_missing_truth_critic_score_remains_pending(tmp_path, monkeypatch):
  fixture(tmp_path, monkeypatch)
  (tmp_path / "diagnostics/A/8-truth/offline.json").unlink()
  report = build_report(tmp_path)
  assert report["status"] == "pending", report
  assert report["supplementary"]["A_truth_critic_status"] == "missing"
  assert "A: independent truth-input critic score missing" in report["pending"]


@pytest.mark.parametrize("damage", ["samples", "censored_samples", "reason"])
def test_critic_censoring_is_checked_against_recorded_episodes(
  tmp_path, monkeypatch, damage
):
  fixture(tmp_path, monkeypatch)
  folder = tmp_path / "diagnostics/A/8-truth"
  set_diagnostic_terminations(folder, ("clip_complete", "clip_complete"))
  path = folder / "offline.json"
  data = json.loads(path.read_text())
  critic = data["critic"]
  if damage == "samples":
    critic["bootstrap_value_error"].update(samples=4, bias=0, mae=0, rmse=0)
  elif damage == "censored_samples":
    critic["censored_samples"] = 0
  else:
    critic["episodes"][0]["reason"] = "timeout"
  write(path, data)
  report = build_report(tmp_path)
  assert report["status"] == "failed", report
  assert any(
    "critic" in failure and "recording" in failure for failure in report["failures"]
  )


def test_missing_diagnostic_preserves_missing_curve_values(tmp_path, monkeypatch):
  fixture(tmp_path, monkeypatch)
  (tmp_path / "diagnostics/B/4/offline.json").unlink()
  report = build_report(tmp_path)
  assert report["status"] == "pending"
  assert report["groups"]["B"]["curves"]["critic_rmse"][1]["value"] is None
  assert any(
    "Missing diagnostics/B/4/offline.json" in item for item in report["pending"]
  )


@pytest.mark.parametrize(
  ("relative", "block"),
  [
    ("diagnostics/B/4/offline.json", "training_overlap_check"),
    ("diagnostics/B/4/offline.json", "fixed_record_posterior"),
    ("diagnostics/C/8/offline.json", "actor_gradients"),
    ("diagnostics/A/8-truth/offline.json", "critic"),
    ("diagnostics/common/walking-reference/B-process.json", "process"),
  ],
)
def test_invalid_diagnostic_block_fails_report(tmp_path, monkeypatch, relative, block):
  fixture(tmp_path, monkeypatch)
  path = tmp_path / relative
  data = json.loads(path.read_text())
  data[block] = None
  write(path, data)
  report = build_report(tmp_path)
  assert report["status"] == "failed"
  assert any(relative in reason and block in reason for reason in report["failures"])


def test_missing_training_log_does_not_claim_zero_actual_budget(tmp_path, monkeypatch):
  fixture(tmp_path, monkeypatch)
  (tmp_path / "A/metrics.jsonl").unlink()
  report = build_report(tmp_path)
  assert report["status"] == "pending"
  assert report["groups"]["A"]["actual_budget"]["mean_updates"] is None
  assert report["groups"]["A"]["budget_rounds_observed"] == 0


@pytest.mark.parametrize("damage", ["covariance", "fingerprint", "snapshot", "restore"])
def test_wrong_evidence_fails_audit(tmp_path, monkeypatch, damage):
  fixture(tmp_path, monkeypatch)
  if damage == "covariance":
    path = tmp_path / "A/metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["budget"]["covariance_updates"] = 3
    write(path, rows, lines=True)
  elif damage == "fingerprint":
    path = tmp_path / "B/snapshots/index.json"
    data = json.loads(path.read_text())
    for row in data["snapshots"]:
      row["initialization_digest"] = "d" * 64
    write(path, data)
  elif damage == "snapshot":
    (tmp_path / "C/snapshots/frames_4.pt").write_bytes(b"mutated")
  else:
    path = tmp_path / "resume/B.json"
    data = json.loads(path.read_text())
    data["checkpoint_sha256"] = "bad"
    write(path, data)
  report = build_report(tmp_path)
  assert report["status"] == "failed"
  assert report["failures"]


def test_censored_validation_and_missing_profile_are_pending(tmp_path, monkeypatch):
  fixture(tmp_path, monkeypatch)
  path = tmp_path / "evaluation/validation.json"
  data = json.loads(path.read_text())
  task_path = data["trained"][0]["tasks"][0]["file"]
  task = json.loads(Path(task_path).read_text())
  task["result"]["episodes"][0].update(
    reason="evaluation_truncated", censored=True, metric_steps=2
  )
  write(Path(task_path), task)
  data["trained"][0]["complete"] = False
  data["trained"][0]["tasks"][0]["status"] = "censored"
  data["trained"][0]["scores"]["success_rate"] = None
  write(path, data)
  path = tmp_path / "A/profiling.jsonl"
  path.write_text(path.read_text().splitlines()[0] + "\n")
  report = build_report(tmp_path)
  assert report["status"] == "pending"
  assert any(
    "validation missing, failed or censored" in item for item in report["pending"]
  )
  assert any("profiling rounds" in item for item in report["pending"])


def test_plot_exports_all_panels_with_missing_values(tmp_path, monkeypatch):
  fixture(tmp_path, monkeypatch)
  (tmp_path / "diagnostics/B/4/offline.json").unlink()
  report = build_report(tmp_path)
  plot_report(report, tmp_path / "plots")
  assert (tmp_path / "plots/learning_curves.png").stat().st_size > 10000
  assert (tmp_path / "plots/learning_curves.pdf").read_bytes().startswith(b"%PDF")


def test_raw_task_identity_is_checked_even_when_summary_claims_complete(
  tmp_path, monkeypatch
):
  fixture(tmp_path, monkeypatch)
  validation = json.loads((tmp_path / "evaluation/validation.json").read_text())
  path = Path(validation["trained"][0]["tasks"][0]["file"])
  data = json.loads(path.read_text())
  data["result"]["seed"] = 999  # Summary flags remain unchanged.
  write(path, data)
  report = build_report(tmp_path)
  assert report["status"] == "pending"
  assert any(
    "raw validation tasks are missing, invalid or censored" in reason
    for reason in report["pending"]
  )


def test_aggregate_return_cannot_replace_actual_episode_return(tmp_path, monkeypatch):
  fixture(tmp_path, monkeypatch)
  path = tmp_path / "evaluation/validation.json"
  data = json.loads(path.read_text())
  data["trained"][0]["scores"]["return"] = 999
  write(path, data)
  report = build_report(tmp_path)
  assert report["status"] == "failed"
  assert any(
    "aggregate differs from verified raw episodes" in reason
    for reason in report["failures"]
  )


def test_final_gradient_and_common_scores_are_required(tmp_path, monkeypatch):
  fixture(tmp_path, monkeypatch)
  path = tmp_path / "diagnostics/C/8/offline.json"
  data = json.loads(path.read_text())
  del data["actor_gradients"]
  write(path, data)
  (tmp_path / "diagnostics/common/policy-baseline/A-process.json").unlink()
  report = build_report(tmp_path)
  assert report["status"] == "pending"
  assert any("gradient decomposition missing" in reason for reason in report["pending"])
  assert any("A-process.json" in reason for reason in report["pending"])


def test_peak_memory_includes_save_phase(tmp_path, monkeypatch):
  fixture(tmp_path, monkeypatch)
  path = tmp_path / "A/profiling.jsonl"
  rows = [json.loads(line) for line in path.read_text().splitlines()]
  rows[0]["runtime"]["cuda_peak_allocated_bytes"] = 100
  rows[0]["runtime_after_save"] = {"cuda_peak_allocated_bytes": 500}
  write(path, rows, lines=True)
  report = build_report(tmp_path)
  assert report["status"] == "complete"
  assert report["groups"]["A"]["max_cuda_allocated_bytes"] == 500


def checkpoint_configuration():
  p = {
    "training_seed": 7,
    "rounds": 10,
    "num_envs": 8,
    "collection_steps": 64,
    "buffer_size": 128,
    "sampling_mode": "uniform",
    "widths": {
      "mean": [2048, 2048, 2048],
      "noise": [128, 64],
      "actor": [2048, 1024, 512, 256],
      "critic": [256, 128],
    },
    "mean": {"num_batches": 10, "batch_size": 2048, "window_frames": 8},
    "covariance": {"batch_size": 256, "epochs": 1},
    "policy": {
      "num_batches": 1,
      "batch_size": 4,
      "num_samples": 2,
      "horizon": 4,
      "microbatch_size": 2,
      "critic_epochs": 2,
    },
  }
  return p, {
    "num_envs": p["num_envs"],
    "loop_config": {"collection_steps": p["collection_steps"]},
    "policy_config": deepcopy(p["policy"]),
    "runner_config": {
      "buffer_size": p["buffer_size"],
      "sampling_mode": p["sampling_mode"],
      "dynamics_hidden_dims": tuple(p["widths"]["mean"]),
      "actor_hidden_dims": tuple(p["widths"]["actor"]),
      "critic_hidden_dims": tuple(p["widths"]["critic"]),
      "process_noise": {"hidden_dims": tuple(p["widths"]["noise"])},
      "mean": deepcopy(p["mean"]),
      "cov": deepcopy(p["covariance"]),
    },
  }


def test_checkpoint_configuration_rejects_small_network_as_default_width(tmp_path):
  from supermimic.scripts.report_eit_pilot import Audit, _check_checkpoint_configuration

  p, state = checkpoint_configuration()
  audit = Audit(tmp_path)
  _check_checkpoint_configuration(audit, state, p, "A/5120")
  assert audit.failures == []
  state["runner_config"]["dynamics_hidden_dims"] = (8,)
  _check_checkpoint_configuration(audit, state, p, "A/5120")
  assert len(audit.failures) == 1
  assert "checkpoint widths.mean differs from pilot protocol" in audit.failures[0]


def test_initialization_audit_compares_baseline_parameters(tmp_path):
  from supermimic.runners.eit_training import initialization_fingerprint
  from supermimic.runners.run_config import EIT_INITIALIZATION_VERSION
  from supermimic.scripts.report_eit_pilot import Audit, _initialization_identity

  bundle = {
    "version": EIT_INITIALIZATION_VERSION,
    "generation_seed": 7,
    "world_model": {"weight": torch.ones(1)},
    "actors": {g: {"weight": torch.ones(1)} for g in "ABC"},
    "critics": {g: {"weight": torch.ones(1)} for g in "ABC"},
  }
  torch.save(bundle, tmp_path / "initialization.pt")
  identity = initialization_fingerprint(bundle)
  p, config = checkpoint_configuration()
  for group in "ABC":
    path = tmp_path / group / "snapshots/frames_0.pt"
    path.parent.mkdir(parents=True)
    state = {
      **config,
      "initialization_digest": identity,
      "world_model": bundle["world_model"],
      "actor": bundle["actors"][group],
      "critic": bundle["critics"][group],
    }
    if group == "C":
      state["actor"] = {"weight": torch.zeros(1)}
    torch.save(state, path)
  audit = Audit(tmp_path)
  assert _initialization_identity(audit, p) == identity
  assert audit.failures == [
    "C: baseline weights differ from actual common initialization"
  ]


@pytest.mark.parametrize(
  "damage",
  [
    "device",
    "motion",
    "runtime_motion",
    "threads",
    "p99",
    "raw_phase",
    "reset",
    "missing_stages",
  ],
)
def test_online_evidence_cannot_claim_different_protocol_or_statistics(
  tmp_path, monkeypatch, damage
):
  fixture(tmp_path, monkeypatch)
  path = tmp_path / "online/A.json"
  value = json.loads(path.read_text())
  if damage == "device":
    value["device"] = "cuda:99"
  elif damage == "motion":
    value["motion_sha256"] = "0" * 64
  elif damage == "runtime_motion":
    value["motion_digest"] = "e" * 64
  elif damage == "threads":
    value["cpu_threads"] = 999
  elif damage == "p99":
    value["total"]["statistics"]["continuous_full_filter"]["total"]["p99_ms"] = 0.001
  elif damage == "raw_phase":
    del value["stages"]["samples"][0]["timings_ms"]["filter_and_jacobian"]
  elif damage == "reset":
    value["total"]["samples"][0]["reset"] = True
  else:
    del value["stages"]
  write(path, value)
  result = build_report(tmp_path)
  assert result["status"] == ("pending" if damage == "missing_stages" else "failed")
  if damage in ("device", "motion", "runtime_motion", "threads"):
    assert result["groups"]["A"]["online_continuous_full_filter"] == {}


@pytest.mark.parametrize("damage", ["missing", "duplicate", "extra_seed"])
def test_saved_validation_requires_complete_unique_candidate_keys(
  tmp_path, monkeypatch, damage
):
  fixture(tmp_path, monkeypatch)
  path = tmp_path / "evaluation/validation.json"
  value = json.loads(path.read_text())
  if damage == "missing":
    value["trained"].pop(0)
  elif damage == "duplicate":
    value["trained"].append(deepcopy(value["trained"][0]))
  else:
    extra = deepcopy(value["trained"][0])
    extra["candidate"]["training_seed"] = 9876
    value["trained"].append(extra)
  write(path, value)
  result = build_report(tmp_path)
  assert result["status"] == ("pending" if damage == "missing" else "failed")


@pytest.mark.parametrize("damage", ["seed", "heads"])
def test_gradient_audit_checks_seed_and_actual_recorded_heads(
  tmp_path, monkeypatch, damage
):
  fixture(tmp_path, monkeypatch)
  path = tmp_path / "diagnostics/A/8/offline.json"
  value = json.loads(path.read_text())
  if damage == "seed":
    value["actor_gradients"]["seed"] = -1
  else:
    value["actor_gradients"]["episode_ids"] = [888, 999]
    value["actor_gradients"]["start_frames"] = [200, 200]
  write(path, value)
  result = build_report(tmp_path)
  assert result["status"] == "failed"


@pytest.mark.parametrize(
  "damage",
  [
    "old_version",
    "missing_reference",
    "wrong_digest",
    "wrong_header",
    "wrong_reference_version",
  ],
)
def test_restore_audit_requires_live_pre_checkpoint_reference(
  tmp_path, monkeypatch, damage
):
  fixture(tmp_path, monkeypatch)
  path = tmp_path / "resume/A.json"
  value = json.loads(path.read_text())
  if damage == "old_version":
    value["version"] = "eit_fixed_input_resume_v1"
  elif damage == "missing_reference":
    del value["reference_path"]
  elif damage == "wrong_digest":
    value["reference_sha256"] = "0" * 64
  elif damage == "wrong_reference_version":
    value["reference_version"] = "eit_live_update_reference_v2"
  else:
    reference = tmp_path / "resume/A-reference.pt"
    live = torch.load(reference, weights_only=True)
    live["checkpoint_sha256"] = "other-checkpoint"
    torch.save(live, reference)
    value["reference_sha256"] = file_sha256(reference)
  write(path, value)
  result = build_report(tmp_path)
  assert result["status"] == (
    "pending" if damage in ("old_version", "missing_reference") else "failed"
  )


@pytest.mark.parametrize(
  "damage",
  [
    "missing_warmup",
    "missing_execution",
    "different_receipt",
    "import_mode",
    "workspace",
    "deterministic",
    "warmup_count",
    "warmup_bool",
    "device",
  ],
)
def test_restore_audit_checks_private_warmup_execution_contract(
  tmp_path, monkeypatch, damage
):
  fixture(tmp_path, monkeypatch)
  path = tmp_path / "resume/A.json"
  reference = tmp_path / "resume/A-reference.pt"
  receipt = json.loads(path.read_text())
  live = torch.load(reference, weights_only=True)
  if damage == "missing_execution":
    del live["verification_execution"]
  elif damage == "missing_warmup":
    for record in (live, receipt):
      del record["verification_execution"]["discarded_private_policy_warmup_updates"]
  elif damage == "device":
    live["device"] = "cuda:0"
  elif damage == "different_receipt":
    receipt["verification_execution"]["torchscript_import_mode"] = "enabled"
  else:
    key, invalid = {
      "import_mode": ("torchscript_import_mode", "enabled"),
      "workspace": ("cublas_workspace_config", ":4096:8"),
      "deterministic": ("deterministic_algorithms", False),
      "warmup_count": ("discarded_private_policy_warmup_updates", 0),
      "warmup_bool": ("discarded_private_policy_warmup_updates", True),
    }[damage]
    for record in (live, receipt):
      record["verification_execution"][key] = invalid
  torch.save(live, reference)
  receipt["reference_sha256"] = file_sha256(reference)
  write(path, receipt)
  result = build_report(tmp_path)
  assert result["status"] == (
    "pending" if damage in ("missing_warmup", "missing_execution") else "failed"
  )
  messages = result["pending"] if result["status"] == "pending" else result["failures"]
  assert any(
    "restore execution" in item or "live reference" in item for item in messages
  )


def test_online_runtime_motion_digest_can_differ_across_devices(tmp_path, monkeypatch):
  fixture(tmp_path, monkeypatch)
  protocol_path = tmp_path / "protocol.json"
  protocol = json.loads(protocol_path.read_text())
  protocol["device"] = "cuda:0"
  protocol["training_process_environment"]["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
  write(protocol_path, protocol)
  for group in "ABC":
    resume_path = tmp_path / "resume" / f"{group}.json"
    resume = json.loads(resume_path.read_text())
    resume["device"] = "cuda:0"
    resume["verification_execution"] = verification_execution(cuda=True)
    reference = tmp_path / "resume" / f"{group}-reference.pt"
    live = torch.load(reference, weights_only=True)
    live["device"] = "cuda:0"
    live["verification_execution"] = verification_execution(cuda=True)
    torch.save(live, reference)
    resume["reference_sha256"] = file_sha256(reference)
    write(resume_path, resume)
    online_path = tmp_path / "online" / f"{group}.json"
    online = json.loads(online_path.read_text())
    online["motion_digest"] = "e" * 64
    write(online_path, online)
  result = build_report(tmp_path)
  assert result["status"] == "complete", result["failures"]


@pytest.mark.parametrize("field", ["total", "stages", "first_reset", "statistics"])
def test_online_null_sections_do_not_crash_audit(tmp_path, monkeypatch, field):
  fixture(tmp_path, monkeypatch)
  path = tmp_path / "online/A.json"
  value = json.loads(path.read_text())
  if field in ("total", "stages"):
    value[field] = None
  else:
    value["total"][field] = None
  write(path, value)
  result = build_report(tmp_path)
  assert result["status"] in ("pending", "failed")


def test_current_step_report_checks_actual_updates_and_profiles(tmp_path, monkeypatch):
  from supermimic.scripts.run_eit_pilot import protocol
  from tests.test_estimator_eit_training import training

  monkeypatch.setenv("WANDB_MODE", "disabled")
  motion = tmp_path / "motion.npz"
  motion.write_bytes(b"fixture")
  p = protocol(motion, 2026, "cpu", "cpu")
  p.update(
    num_envs=2,
    max_iterations=4,
    resume_after_step=3,
    covariance_pair_budget=4,
    updates_per_training_group={"mean": 1, "covariance": 2, "actor": 1, "critic": 1},
  )
  write(tmp_path / "protocol.json", p)
  loop = training()
  loop.training_seed = p["training_seed"]
  loop.run(4, tmp_path / "B")
  write(tmp_path / "resume/B.json", {"status": "passed", "collected_steps": 3})
  write(tmp_path / "evaluation/validation.json", {"complete": True})
  report = build_report(tmp_path)
  assert "Missing manifest.json" in report["pending"]
  assert "B: missing midpoint reference or verification receipt" in report["pending"]
  assert report["version"] == "eit_pilot_report_v2" and report["status"] == "pending"
  group = report["groups"]["B"]
  assert group["collected_steps"] == 4 and group["updates"]["covariance"] == 2
  assert group["phase_seconds"]["covariance_fit_seconds"] > 0
  assert group["cuda_peak_allocated_bytes"] is None
  checkpoint = tmp_path / "B/training.pt"
  saved = torch.load(checkpoint, weights_only=True)
  torch.save({"group": "A", "collected_steps": 3}, tmp_path / "resume/B-reference.pt")
  assert any(
    "midpoint reference, checkpoint or verification receipt mismatched" in issue
    for issue in build_report(tmp_path)["failures"]
  )
  (tmp_path / "resume/B-reference.pt").unlink()
  saved["learner"]["mean"]["steps"] += 1
  torch.save(saved, checkpoint)
  assert build_report(tmp_path)["status"] == "failed"


def test_step_validation_requires_the_raw_final_episodes_and_actor_updates(
  tmp_path, monkeypatch
):
  p = fixture(tmp_path, monkeypatch)
  manifest_path = tmp_path / "manifest.json"
  manifest = json.loads(manifest_path.read_text())
  final_frames = 8
  for run in manifest["runs"]:
    run["frames"] = [final_frames]
  del manifest["manifest_digest"]
  manifest["manifest_digest"] = batch_eval_eit.digest(manifest)
  write(manifest_path, manifest)
  candidates = [
    candidate
    for candidate in batch_eval_eit.candidates(manifest)
    if candidate["collected_frames"] == final_frames
  ]
  monkeypatch.setattr(batch_eval_eit, "candidates", lambda _: candidates)
  records = [
    batch_eval_eit.evaluate_candidate(
      c, manifest["validation"], tmp_path / "evaluation", False
    )
    for c in candidates
  ]
  aggregate = dict(
    manifest_digest=manifest["manifest_digest"],
    complete=True,
    trained=records,
    baselines=[],
    unavailable=[],
    selected=batch_eval_eit.select(records),
  )

  def audit_sources():
    audit = report_eit_pilot.Audit(tmp_path)
    report_eit_pilot._validated_sources(audit, p, aggregate, frames=[final_frames])
    return audit

  good = audit_sources()
  assert good.failures == [] and good.pending == []
  warmup = aggregate["trained"].pop()
  warmup["candidate"]["actor_steps"] = 0
  aggregate["baselines"].append(warmup)
  aggregate["selected"] = batch_eval_eit.select(records)
  assert audit_sources().failures == []
  aggregate["unavailable"].append({"candidate": {"error": "bad counters"}})
  assert (
    "Validation aggregate includes unavailable candidates" in audit_sources().pending
  )
  aggregate["unavailable"].clear()
  Path(records[0]["tasks"][0]["file"]).unlink()
  assert any(
    "raw validation tasks are missing" in item for item in audit_sources().pending
  )


@pytest.mark.parametrize("single_graph", [False, True])
def test_report_accepts_single_graph_and_legacy_policy_timings(
  tmp_path, monkeypatch, single_graph
):
  fixture(tmp_path, monkeypatch)
  for group in "ABC":
    path = tmp_path / group / "profiling.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    omitted = (
      ("policy_rollout_backward_seconds", "critic_regression_seconds")
      if single_graph
      else ("policy_prepare_seconds", "policy_update_seconds", "policy_check_seconds")
    )
    for row in rows:
      for name in omitted:
        row["seconds"].pop(name, None)
    write(path, rows, lines=True)
  report = build_report(tmp_path)
  assert report["status"] == "complete", report
