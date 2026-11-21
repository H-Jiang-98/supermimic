"""Offline logging observes committed EIT rounds and retains resume identity."""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from supermimic.utils import eit_wandb, wandb_artifacts
from supermimic.utils.eit_wandb import EITOfflineWandb, metric_row
from tests.test_estimator_belief_training import assert_tree
from tests.test_estimator_eit_training import training


@pytest.fixture
def runs(monkeypatch, tmp_path):
  created = []

  def init(**kwargs):
    output = Path(kwargs["dir"])
    # Starting W&B must not preempt the trainer's empty-directory guard.
    assert (output / "training.pt").is_file()
    files = output / "wandb" / f"offline-run-{len(created)}" / "files"
    files.mkdir(parents=True)
    run = SimpleNamespace(
      id=kwargs["id"],
      dir=str(files),
      offline=True,
      disabled=False,
      options=kwargs,
      rows=[],
      definitions=[],
      finished=[],
    )
    run.define_metric = lambda *a, **k: run.definitions.append((a, k))

    def log(row, *, step):
      state = torch.load(output / "training.pt", weights_only=True)
      assert state["rounds"] == step
      assert (
        json.loads((output / "metrics.jsonl").read_text().splitlines()[-1])["round"]
        == step
      )
      assert (
        json.loads((output / "profiling.jsonl").read_text().splitlines()[-1])["round"]
        == step
      )
      run.rows.append((step, row))

    run.log = log
    run.finish = lambda **k: run.finished.append(k)
    created.append(run)
    return run

  monkeypatch.setattr(eit_wandb.wandb, "init", init)
  monkeypatch.setattr(eit_wandb.wandb, "run", None)
  monkeypatch.setattr(
    eit_wandb.wandb, "Api", lambda *a, **k: pytest.fail("network API")
  )
  monkeypatch.setattr(eit_wandb.wandb, "login", lambda *a, **k: pytest.fail("login"))
  return created


def test_metric_mapping_omits_missing_values_and_preserves_policy_names():
  row = metric_row(
    {
      "round": 2,
      "collected_frames": 1024,
      "mean_steps": 20,
      "cov_steps": 8,
      "training_episodes": [1, 2],
      "train": {"mean": {"loss": 0.5}},
      "validation": {"nll": 1.5, "unavailable": None},
      "policy": {
        "Policy/return": 2.0,
        "Policy/explained_variance": None,
        "Grad/actor_norm": 0.1,
      },
      "budget": {"actor_updates": 1, "bad": float("nan")},
    },
    {
      "seconds": {"round_seconds": 3.0},
      "runtime_after_save": {"cuda_allocated_bytes": 1024},
    },
  )
  assert row["Train/mean/loss"] == 0.5
  assert row["ModelValidation/nll"] == 1.5
  assert row["Policy/return"] == 2.0 and row["Grad/actor_norm"] == 0.1
  assert row["Perf/round_seconds"] == 3.0
  assert row["RuntimeAfterSave/cuda_allocated_bytes"] == 1024
  assert "Policy/explained_variance" not in row
  assert not any("episodes" in key or "bad" in key for key in row)
  json.dumps(row, allow_nan=False)


def test_logged_resume_uses_same_id_absolute_steps_and_persistent_settings(
  tmp_path, runs
):
  first = training()
  first.run(
    2,
    tmp_path,
    snapshot_interval=2,
    logger=EITOfflineWandb(project="test-eit", group="experiment"),
  )
  checkpoint = tmp_path / "training.pt"
  metadata = (tmp_path / "wandb_run.json").read_bytes()
  before = deepcopy(first.state_dict())
  restored = training(81)
  restored.run(1, tmp_path, resume=checkpoint, logger=EITOfflineWandb())
  assert [r.options["mode"] for r in runs] == ["offline", "offline"]
  assert runs[0].id == runs[1].id
  assert runs[1].options["project"] == "test-eit"
  assert runs[1].options["group"] == "experiment"
  assert runs[1].options["config"]["starting_round"] == 2
  assert runs[1].options["config"]["snapshot_interval"] == 2
  assert [step for r in runs for step, _ in r.rows] == [1, 2, 3]
  assert [row["Counters/collected_frames"] for r in runs for _, row in r.rows] == [
    8,
    16,
    24,
  ]
  assert all(r.finished == [{"exit_code": 0}] for r in runs)
  assert (tmp_path / "wandb_run.json").read_bytes() == metadata
  # Logging does not add W&B state to the exact-update checkpoint contract.
  assert not any("wandb" in key for key in before)
  for run in runs:
    session = json.loads((Path(run.dir).parent / "eit_session.json").read_text())
    assert session["finished"] and session["logging_ok"]


def test_logging_does_not_change_model_optimizer_or_rng(tmp_path, runs):
  plain = training(31)
  plain.run(2, tmp_path / "plain", snapshot_interval=2)
  expected = plain.state_dict()
  logged = training(31)
  logged.run(2, tmp_path / "logged", snapshot_interval=2, logger=EITOfflineWandb())
  assert_tree(logged.state_dict(), expected)


def test_offline_model_artifacts_follow_snapshot_interval(tmp_path, runs, monkeypatch):
  artifacts = []
  monkeypatch.setattr(
    eit_wandb, "log_model_artifact", lambda files, **k: artifacts.append((files, k))
  )
  loop = training()
  loop.run(3, tmp_path, snapshot_interval=2, logger=EITOfflineWandb(log_models=True))
  assert [Path(files[0]).name for files, _ in artifacts] == [
    "frames_0.pt",
    "frames_16.pt",
  ]
  assert artifacts[-1][1]["metadata"] == {"round": 2, "collected_frames": 16}
  assert artifacts[-1][1]["run"] is runs[0]


@pytest.mark.parametrize("phase", ["init", "log", "finish"])
def test_logging_errors_do_not_lose_training_checkpoints(
  tmp_path, runs, monkeypatch, capsys, phase
):
  init = eit_wandb.wandb.init

  def fail(*a, **k):
    raise OSError("test logging failure")

  def setup(**kwargs):
    run = init(**kwargs)
    setattr(run, phase, fail)
    return run

  monkeypatch.setattr(eit_wandb.wandb, "init", fail if phase == "init" else setup)
  training().run(2, tmp_path, logger=EITOfflineWandb())
  assert torch.load(tmp_path / "training.pt", weights_only=True)["rounds"] == 2
  assert len((tmp_path / "metrics.jsonl").read_text().splitlines()) == 2
  assert "[WARN] EIT W&B offline" in capsys.readouterr().out


def test_training_failure_finishes_offline_run_and_keeps_last_commit(
  tmp_path, runs, monkeypatch
):
  loop = training()
  original = loop.run_round

  def fail_second(*args, **kwargs):
    if loop.rounds == 1:
      raise RuntimeError("training failed")
    return original(*args, **kwargs)

  monkeypatch.setattr(loop, "run_round", fail_second)
  with pytest.raises(RuntimeError, match="training failed"):
    loop.run(2, tmp_path, logger=EITOfflineWandb())
  assert runs[0].finished == [{"exit_code": 1}]
  assert torch.load(tmp_path / "training.pt", weights_only=True)["rounds"] == 1


def test_new_output_gets_distinct_run_identity_and_no_logger_creates_no_wandb(
  tmp_path, runs
):
  training().run(1, tmp_path / "A", logger=EITOfflineWandb())
  training().run(1, tmp_path / "B", logger=EITOfflineWandb())
  assert runs[0].id != runs[1].id
  training().run(1, tmp_path / "disabled")
  assert len(runs) == 2
  assert not (tmp_path / "disabled/wandb_run.json").exists()


def test_existing_artifact_helper_never_queries_registry_for_offline_run(monkeypatch):
  logged = []
  monkeypatch.setattr(
    wandb_artifacts.wandb,
    "Artifact",
    lambda *a, **k: SimpleNamespace(add_file=lambda p: None),
  )
  monkeypatch.setattr(
    wandb_artifacts.wandb, "Api", lambda: pytest.fail("queried network")
  )
  run = SimpleNamespace(
    id="offline",
    disabled=False,
    offline=True,
    log_artifact=lambda artifact, **k: logged.append(k),
  )
  assert wandb_artifacts.log_model_artifact(["test.pt"], run=run, aliases=["latest"])
  assert logged == [{"aliases": ["latest"]}]


def test_pilot_keeps_logging_disabled():
  from supermimic.scripts.run_eit_pilot import training_arguments

  args = training_arguments(
    {"motion_file": "walking.npz", "device": "cpu", "training_seed": 2026}
  )
  assert args[args.index("--wandb-mode") + 1] == "disabled"


@pytest.mark.parametrize("mode", ["online", "invalid"])
def test_cli_rejects_nonoffline_environment_mode_before_creating_environment(
  monkeypatch, mode
):
  from supermimic.scripts import train_eit

  monkeypatch.setenv("WANDB_MODE", mode)
  monkeypatch.setattr("sys.argv", ["train_eit", "--motion-file", "unused.npz"])
  monkeypatch.setattr(
    train_eit, "create_env", lambda *_: pytest.fail("created environment")
  )
  with pytest.raises(SystemExit) as error:
    train_eit.main()
  assert error.value.code == 2


def test_actual_sdk_creates_readable_offline_segments(tmp_path, monkeypatch):
  from wandb.sdk.lib.proto_util import dict_from_proto_list

  from supermimic.scripts.relocate_wandb import read_records

  monkeypatch.setenv("WANDB_API_KEY", "")
  monkeypatch.setenv("WANDB_MODE", "offline")
  monkeypatch.setenv("WANDB_CONFIG_DIR", str(tmp_path / "config"))
  monkeypatch.setenv("WANDB_DATA_DIR", str(tmp_path / "data"))
  monkeypatch.setenv("WANDB_CACHE_DIR", str(tmp_path / "cache"))
  monkeypatch.setattr(
    eit_wandb.wandb, "Api", lambda *a, **k: pytest.fail("network API")
  )
  monkeypatch.setattr(eit_wandb.wandb, "login", lambda *a, **k: pytest.fail("login"))
  output = tmp_path / "training"
  plain = training()
  plain.run(1, tmp_path / "plain", snapshot_interval=2)
  try:
    logged = training()
    logged.run(
      1,
      output,
      snapshot_interval=2,
      logger=EITOfflineWandb(project="eit-offline-test", log_models=True),
    )
    assert_tree(logged.state_dict(), plain.state_dict())
    training(81).run(
      1, output, resume=output / "training.pt", logger=EITOfflineWandb(log_models=True)
    )
  finally:
    eit_wandb.wandb.teardown()
  paths = sorted((output / "wandb").glob("offline-run-*/*.wandb"))
  assert len(paths) == 2
  ids, rows, artifacts = [], [], []
  for path in paths:
    records = list(read_records(path))
    ids.extend(r.run.run_id for r in records if r.WhichOneof("record_type") == "run")
    rows.extend(
      dict_from_proto_list(r.history.item)
      for r in records
      if r.WhichOneof("record_type") == "history"
    )
    artifacts.extend(
      r.artifact for r in records if r.WhichOneof("record_type") == "artifact"
    )
    assert any(r.WhichOneof("record_type") == "exit" for r in records)
    session = json.loads((path.parent / "eit_session.json").read_text())
    assert session["finished"] and session["logging_ok"]
  assert len(set(ids)) == 1
  assert [row["_step"] for row in rows] == [1, 2]
  assert [row["Counters/collected_frames"] for row in rows] == [8, 16]
  assert all("Policy/return" in row and "Perf/round_seconds" in row for row in rows)
  assert len(artifacts) == 2
  for artifact in artifacts:
    for entry in artifact.manifest.contents:
      assert Path(entry.local_path).is_file()
