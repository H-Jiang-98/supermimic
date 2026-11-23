"""Pilot scheduling keeps performance, truth-input and fixed-record scoring distinct."""

import json
import sys
from types import SimpleNamespace

import pytest
import torch

from supermimic.scripts.check_eit_resume import equal_tree
from supermimic.scripts.run_eit_pilot import (
  Pilot,
  diagnostic_cache_valid,
  file_sha256,
  protocol,
  source_audit,
  write_json,
)
from tests.test_estimator_diagnostic_structure import (
  valid_diagnostic_score,
  valid_gradient_score,
)


def test_diagnostic_retry_checks_inputs_requested_mode_and_gradients(tmp_path):
  checkpoint, recording, output = (
    tmp_path / name for name in ("model.pt", "record.pt", "offline.json")
  )
  checkpoint.write_bytes(b"checkpoint")
  recording.write_bytes(b"recording")
  arguments = ["--checkpoint", checkpoint, "--recording", recording, "--output", output]
  result = {
    **valid_diagnostic_score(),
    "checkpoint_sha256": file_sha256(checkpoint),
    "recording_sha256": file_sha256(recording),
  }
  write_json(output, result)
  assert diagnostic_cache_valid(arguments)
  assert not diagnostic_cache_valid([*arguments, "--actor-gradients"])
  result["actor_gradients"] = valid_gradient_score()
  write_json(output, result)
  assert diagnostic_cache_valid([*arguments, "--actor-gradients"])
  assert not diagnostic_cache_valid(
    [*arguments, "--actor-gradients", "--gradient-episodes", 1]
  )
  assert not diagnostic_cache_valid([*arguments, "--window-frames", 3])
  recording.write_bytes(b"different independent recording")
  assert not diagnostic_cache_valid(arguments)


def _pilot(tmp_path):
  motion = tmp_path / "walking.npz"
  motion.write_bytes(b"fixed-motion")
  return Pilot(tmp_path, protocol(motion, 2026, "cpu", "cpu"))


def _option(arguments, name):
  return arguments[arguments.index(name) + 1]


@pytest.mark.parametrize("version", ["eit_pilot_v1", "eit_pilot_v2", "eit_pilot_v3"])
def test_pilot_rejects_previous_round_protocol(tmp_path, monkeypatch, version):
  from supermimic.scripts.run_eit_pilot import main

  current = _pilot(tmp_path).p
  write_json(tmp_path / "protocol.json", {**current, "version": version})
  monkeypatch.setattr(
    sys,
    "argv",
    [
      "run_eit_pilot",
      "--motion-file",
      current["motion_file"],
      "--device",
      "cpu",
      "--output",
      str(tmp_path),
      "--phase",
      "evaluate",
    ],
  )
  with pytest.raises(ValueError, match="Pilot configuration changed"):
    main()


def test_pilot_passes_approved_budgets_to_training(tmp_path):
  from supermimic.scripts.run_eit_pilot import training_arguments

  p = _pilot(tmp_path).p
  args = training_arguments(p)
  assert _option(args, "--num-envs") == "1024"
  assert _option(args, "--replay-ratio") == "16"
  assert _option(args, "--critic-epochs") == "5"
  assert _option(args, "--critic-minibatches") == "4"
  assert "--collection-steps" not in args and "--snapshot-interval" not in args
  assert "--cov-epochs" not in args and "--wandb-mode" not in args
  assert p["covariance_pair_budget"] == 143360


def test_pilot_evaluates_all_snapshots_and_final_gradient_truth_and_common_records(
  tmp_path,
):
  pilot = _pilot(tmp_path)
  calls = []
  pilot.command = lambda name, module, args: calls.append(
    (name, module, list(map(str, args)))
  )
  for group in "ABC":
    folder = tmp_path / group
    folder.mkdir(parents=True)
    (folder / "training.pt").write_bytes(group.encode())
  pilot.evaluate()
  validation = next(args for name, _, args in calls if name == "validation")
  assert _option(validation, "--split") == "validation"
  assert not any(
    "--split" in args and _option(args, "--split") == "test" for _, _, args in calls
  )
  gradients = [(name, args) for name, _, args in calls if "--actor-gradients" in args]
  assert {name for name, _ in gradients} == {
    f"diagnose-{group}-102400" for group in "ABC"
  }
  for _, args in gradients:
    assert _option(args, "--checkpoint").endswith("training.pt")
  truth = next(args for name, _, args in calls if name == "record-A-truth")
  assert "--truth-actor" in truth and "--steps" not in truth
  truth_score = next(args for name, _, args in calls if name == "diagnose-A-truth")
  assert _option(truth_score, "--mode") == "critic"
  reference = next(args for name, _, args in calls if name == "record-common-reference")
  assert "--reference-controller" in reference
  assert _option(reference, "--steps") == "250"
  common = [
    (name, args) for name, _, args in calls if name.startswith("common-policy-final-")
  ]
  assert len(common) == 6
  assert {_option(args, "--mode") for _, args in common} == {"process", "posterior"}
  assert {_option(args, "--recording") for _, args in common} == {
    str(tmp_path / "diagnostics/B/102400/episodes.pt")
  }
  for name, _, args in calls:
    if name.startswith("record-") and name != "record-common-reference":
      assert "--steps" not in args


def test_recording_cache_checks_scene_behavior_and_record_contents(tmp_path):
  pilot = _pilot(tmp_path)
  checkpoint = tmp_path / "checkpoint.pt"
  checkpoint.write_bytes(b"checkpoint")
  from supermimic.runners.eit_snapshots import file_sha256

  folder = tmp_path / "recording"
  folder.mkdir()
  scored = {
    "checkpoint_sha256": file_sha256(checkpoint),
    "evaluation_motion_sha256": pilot.p["motion_sha256"],
    "seed": pilot.p["diagnostic_seed"],
    "num_envs": 2,
    "actor_state_source": "posterior",
    "step_budget": None,
    "time_limit_protocol": "full_clip_v1",
  }
  recorded = {
    "checkpoint_sha256": scored["checkpoint_sha256"],
    "motion_sha256": pilot.p["motion_sha256"],
    "seed": pilot.p["diagnostic_seed"],
    "actor_state_source": "posterior",
    "episodes": [{}, {}],
    "version": "eit_diagnostic_episodes_v1",
    "source": "mujoco_evaluation",
    "split": "diagnostic",
  }
  write_json(folder / "evaluation.json", scored)
  torch.save(recorded, folder / "episodes.pt")
  calls = []
  pilot.command = lambda *args: calls.append(args)
  pilot.record("record", checkpoint, folder)
  assert not calls
  pilot.record("record", checkpoint, folder, truth_actor=True)
  assert len(calls) == 1
  recorded["checkpoint_sha256"] = "mismatched-record"
  torch.save(recorded, folder / "episodes.pt")
  pilot.record("record", checkpoint, folder)
  assert len(calls) == 2
  (folder / "episodes.pt").write_bytes(b"truncated")
  pilot.record("record", checkpoint, folder)
  assert len(calls) == 3


def test_command_audits_source_for_each_subprocess(tmp_path, monkeypatch):
  from supermimic.scripts import run_eit_pilot as module

  pilot = _pilot(tmp_path)
  audit = {
    "source_commit": "commit",
    "source_files_sha256": {"src/supermimic/a.py": "one"},
    "torch_version": "test",
  }
  monkeypatch.setattr(module, "source_audit", lambda: dict(audit))
  monkeypatch.setattr(
    module.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(returncode=0)
  )
  pilot.command("first", "eval_eit", ["--example", "first"])
  audit["source_files_sha256"] = {"src/supermimic/a.py": "two"}
  pilot.command("second", "eval_eit", ["--example", "second"])
  rows = [
    json.loads(line) for line in (tmp_path / "commands.jsonl").read_text().splitlines()
  ]
  assert rows[0]["source_files_sha256"] != rows[1]["source_files_sha256"]
  assert rows[0]["source_commit"] == "commit"
  assert rows[1]["torch_version"] == "test"


def test_source_audit_contains_only_source_paths_and_commit():
  audit = source_audit()
  assert set(audit) == {"source_commit", "source_files_sha256", "torch_version"}
  assert "src/supermimic/scripts/run_eit_pilot.py" in audit["source_files_sha256"]
  assert all(
    path.startswith("src/supermimic/") and path.endswith(".py")
    for path in audit["source_files_sha256"]
  )
  assert all(len(value) == 64 for value in audit["source_files_sha256"].values())


def test_training_and_verification_run_with_the_same_cuda_settings(
  tmp_path, monkeypatch
):
  from supermimic.scripts import run_eit_pilot as module

  motion = tmp_path / "motion"
  motion.write_bytes(b"walking")
  pilot = Pilot(tmp_path, protocol(motion, 2026, "cuda:0", "cpu"))
  monkeypatch.setattr(module, "source_audit", dict)
  pilot.env.pop("PYTORCH_JIT", None)
  captured = []

  def record(*args, env, **kwargs):
    captured.append(env)
    return SimpleNamespace(returncode=0)

  monkeypatch.setattr(module.subprocess, "run", record)
  for name in ("train_eit", "check_eit_resume", "benchmark_eit_online"):
    pilot.command(name, name, [])
  assert captured[0] == captured[1]
  assert captured[0]["PYTORCH_JIT"] == "0"
  assert captured[0]["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
  assert "PYTORCH_JIT" not in captured[2]


def test_exact_resume_comparison_reports_nested_key_without_relaxing_tolerance():
  with pytest.raises(AssertionError, match=r"root.optimizer\[0\].exp_avg"):
    equal_tree(
      {"optimizer": [{"exp_avg": torch.tensor([1.0])}]},
      {"optimizer": [{"exp_avg": torch.tensor([1.0000001])}]},
    )


def write_midpoint_evidence(checkpoint, reference, path, identity=None):
  from supermimic.runners.eit_resume_contract import REFERENCE_VERSION

  execution = {
    "torch_compile": "force_eager",
    "torchscript_optimized_execution": False,
    "torchscript_import_mode": "disabled_before_import",
    "deterministic_algorithms": True,
    "fixed_input_assembly": "cpu_contiguous_then_shared_device_constructors_v1",
    "discarded_private_policy_warmup_updates": 1,
    "cublas_workspace_config": None,
    "expected_optimizer_origin": "deepcopy_of_live_parameter_optimizer_graph",
  }
  live = {
    "version": REFERENCE_VERSION,
    "device": "cpu",
    "verification_execution": execution,
    "source": "live_training_before_checkpoint_commit",
    "checkpoint_sha256": file_sha256(checkpoint),
    "group": "A",
    "collected_steps": 5,
  }
  live.update(identity or {})
  torch.save(live, reference)
  row = {
    "version": "eit_fixed_input_resume_v2",
    "status": "passed",
    "checkpoint_sha256": file_sha256(checkpoint),
    "group": "A",
    "collected_steps": 5,
    "device": "cpu",
    "fixed_input_policy_resume_exact": True,
    "fixed_input_model_resume_exact": True,
    "live_optimizer_reference": True,
    "expected_source": "live_training_before_checkpoint_commit",
    "reference_version": REFERENCE_VERSION,
    "reference_sha256": file_sha256(reference),
    "verification_execution": execution,
  }
  write_json(path, row)
  return live, row


def test_interrupted_resume_rechecks_original_midpoint_after_republication(tmp_path):
  # Launching the second leg can atomically replace training.pt before the
  # first new physical step. The original proof must survive that interruption.
  pilot = _pilot(tmp_path)
  pilot.p["resume_after_step"] = 5
  checkpoint = tmp_path / "A-training.pt"
  checkpoint_state = {"initialization_digest": "shared", "motion_digest": "walk"}
  torch.save(checkpoint_state, checkpoint)
  reference = tmp_path / "A-reference.pt"
  path = tmp_path / "A.json"
  live, row = write_midpoint_evidence(checkpoint, reference, path, checkpoint_state)
  pilot.command = lambda *args: pytest.fail(f"Unexpected invocation: {args}")
  pilot.verify_midpoint("A", checkpoint, reference, 5)
  midpoint = tmp_path / "A-midpoint.pt"
  assert midpoint.samefile(checkpoint)
  replacement = checkpoint.with_suffix(".tmp")
  torch.save({**checkpoint_state, "logger": "restarted"}, replacement)
  replacement.replace(checkpoint)
  assert file_sha256(checkpoint) != live["checkpoint_sha256"]
  pilot.verify_midpoint("A", checkpoint, reference, 5)
  path.unlink()

  def reverify(name, module, args):
    assert name == "resume-A" and module == "check_eit_resume"
    assert _option(args, "--checkpoint") == midpoint
    assert file_sha256(midpoint) == live["checkpoint_sha256"]
    write_json(path, row)

  pilot.command = reverify
  pilot.verify_midpoint("A", checkpoint, reference, 5)
  assert json.loads(path.read_text()) == row


def test_resume_cache_requires_successful_exact_checks_and_valid_json(tmp_path):
  from supermimic.scripts.run_eit_pilot import resume_result_matches

  checkpoint, path, reference = (
    tmp_path / name for name in ("checkpoint.pt", "resume.json", "reference.pt")
  )
  checkpoint.write_bytes(b"unchanged")
  live, row = write_midpoint_evidence(checkpoint, reference, path)
  execution = live["verification_execution"]
  environment = {"PYTORCH_JIT": "0"}

  def matches():
    return resume_result_matches(
      path, checkpoint, "A", 5, "cpu", reference, environment
    )

  assert matches()
  assert not resume_result_matches(path, checkpoint, "A", 5, "cpu")
  for key, invalid in (
    ("status", "failed"),
    ("fixed_input_model_resume_exact", False),
    ("group", "B"),
    ("device", "cuda:0"),
    ("reference_version", "eit_live_update_reference_v2"),
  ):
    write_json(path, {**row, key: invalid})
    assert not matches()
  for key, invalid in (
    ("discarded_private_policy_warmup_updates", 0),
    ("deterministic_algorithms", False),
    ("torchscript_import_mode", "enabled"),
    ("cublas_workspace_config", ":4096:8"),
  ):
    damaged = {**execution, key: invalid}
    # Matching, correctly hashed artifacts still cannot override the protocol.
    torch.save({**live, "verification_execution": damaged}, reference)
    write_json(
      path,
      {
        **row,
        "verification_execution": damaged,
        "reference_sha256": file_sha256(reference),
      },
    )
    assert not matches()
  for source in ("receipt", "live"):
    missing_warmup = {
      key: value
      for key, value in execution.items()
      if key != "discarded_private_policy_warmup_updates"
    }
    torch.save(
      {
        **live,
        "verification_execution": missing_warmup if source == "live" else execution,
      },
      reference,
    )
    write_json(
      path,
      {
        **row,
        "verification_execution": missing_warmup if source == "receipt" else execution,
        "reference_sha256": file_sha256(reference),
      },
    )
    assert not matches()
  torch.save({**live, "device": "cuda:0"}, reference)
  write_json(path, {**row, "reference_sha256": file_sha256(reference)})
  assert not matches()
  path.write_text("{truncated")
  assert not matches()


@pytest.mark.parametrize("change", ["group", "seed", "rounds", "frames"])
def test_resume_rejects_foreign_or_overbudget_training_checkpoints(
  tmp_path, monkeypatch, change
):
  from supermimic.scripts import run_eit_pilot as module

  monkeypatch.setattr(
    module, "validate_eit_checkpoint_contract", lambda *_args, **_kwargs: None
  )
  p = _pilot(tmp_path).p
  state = {
    "runner_config": {"estimator_inputs": {"group": "A"}},
    "training_seed": 2026,
    "collected_steps": 5,
    "collected_frames": 5120,
  }
  assert module.checkpoint_progress(state, p, "A") == 5
  if change == "group":
    state["runner_config"]["estimator_inputs"]["group"] = "C"
  elif change == "seed":
    state["training_seed"] = 700
  elif change == "rounds":
    state["collected_steps"] = 101
  else:
    state["collected_frames"] = 42
  with pytest.raises(ValueError):
    module.checkpoint_progress(state, p, "A")


def test_verify_python_api_also_disables_jit_specialization_and_restores_mode(
  monkeypatch,
):
  from supermimic.scripts import check_eit_resume as module

  monkeypatch.setattr(
    module, "_verify", lambda *_args: torch._C._get_graph_executor_optimize()
  )
  with torch.jit.optimized_execution(True):
    assert module.verify(None, None, "cpu", None) is False
    assert torch._C._get_graph_executor_optimize() is True


def test_existing_round_zero_resumes_and_requests_live_reference(tmp_path, monkeypatch):
  from supermimic.runners.eit_training import EITTraining
  from supermimic.scripts import run_eit_pilot as module

  pilot = _pilot(tmp_path)
  checkpoint = tmp_path / "A/training.pt"
  checkpoint.parent.mkdir()
  torch.save({"collected_steps": 0}, checkpoint)
  monkeypatch.setattr(
    module, "checkpoint_progress", lambda state, *_args: state["collected_steps"]
  )
  monkeypatch.setattr(
    EITTraining, "reconcile_output", lambda *_args, **_kwargs: {"collected_steps": 0}
  )
  calls = []

  def command(name, module, arguments):
    calls.append((name, list(map(str, arguments))))
    if name == "train-A-to-50":
      raise InterruptedError("stop after inspecting first training command")

  pilot.command = command
  with pytest.raises(InterruptedError):
    pilot.train()
  args = calls[-1][1]
  assert _option(args, "--resume") == str(checkpoint)
  assert "--initialization" not in args
  assert _option(args, "--resume-reference") == str(tmp_path / "resume/A-reference.pt")


def test_completed_legacy_run_without_live_reference_is_not_relabelled(
  tmp_path, monkeypatch
):
  from supermimic.runners.eit_training import EITTraining
  from supermimic.scripts import run_eit_pilot as module

  pilot = _pilot(tmp_path)
  checkpoint = tmp_path / "A/training.pt"
  checkpoint.parent.mkdir()
  torch.save({"collected_steps": 100}, checkpoint)
  monkeypatch.setattr(
    module, "checkpoint_progress", lambda state, *_args: state["collected_steps"]
  )
  monkeypatch.setattr(
    EITTraining, "reconcile_output", lambda *_args, **_kwargs: {"collected_steps": 100}
  )
  calls = []
  pilot.command = lambda name, *_args: calls.append(name)
  with pytest.raises(ValueError, match="cannot be reconstructed"):
    pilot.train()
  assert calls == ["initialization"]
  assert torch.load(checkpoint, weights_only=True)["collected_steps"] == 100


@pytest.mark.parametrize("component", ["process", "fixed_record_posterior", "critic"])
def test_diagnostic_cache_rejects_null_empty_and_nonfinite_scores(tmp_path, component):
  checkpoint, record, output = (
    tmp_path / name for name in ("model.pt", "record.pt", "offline.json")
  )
  checkpoint.write_bytes(b"model")
  record.write_bytes(b"record")
  result = {
    **valid_diagnostic_score(),
    "checkpoint_sha256": file_sha256(checkpoint),
    "recording_sha256": file_sha256(record),
  }
  args = ["--checkpoint", checkpoint, "--recording", record, "--output", output]
  for value in (None, {}, [], {"broken": float("nan")}):
    malformed = {**result, component: value}
    output.write_text(json.dumps(malformed))
    assert not diagnostic_cache_valid(args)


def test_wrong_group_is_rejected_before_any_snapshot_or_log_recovery(tmp_path):
  from tests.test_estimator_eit_training import training

  pilot = _pilot(tmp_path)
  run = tmp_path / "A"
  run.mkdir()
  loop = training()  # This is a valid group B checkpoint in the wrong directory.
  loop.training_seed = pilot.p["training_seed"]
  loop.save(run / "training.pt")
  (run / ".belief-training.lock").touch()
  (run / "metrics.jsonl").write_bytes(b'{"interrupted":')
  (run / "profiling.jsonl").write_bytes(b'{"round":1,')
  before = {
    path.name: path.read_bytes()
    for path in run.iterdir()
    if path.name != ".belief-training.lock"
  }
  calls = []
  pilot.command = lambda name, *_args: calls.append(name)
  with pytest.raises(ValueError, match="group or training seed differs"):
    pilot.train()
  assert calls == ["initialization"]
  assert {
    path.name: path.read_bytes()
    for path in run.iterdir()
    if path.name != ".belief-training.lock"
  } == before
  assert not (tmp_path / "recovery/A.json").exists()
