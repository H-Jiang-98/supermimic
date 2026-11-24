"""EIT uses common W&B/Logger/artifact entry points, with a fresh offline run."""

import json
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import wandb

from supermimic.runners.eit_resume_reference import equal_tree
from supermimic.utils.logger import Logger
from tests.test_estimator_eit_training import training


def test_common_init_logging_and_artifacts_are_offline_and_new_on_resume(
  tmp_path, monkeypatch
):
  monkeypatch.setenv("WANDB_MODE", "offline")
  runs, configurations, artifacts, logs = [], [], [], []

  def init(**kwargs):
    configurations.append(kwargs)
    run = SimpleNamespace(
      id=kwargs["id"],
      name=kwargs["name"],
      project=kwargs["project"],
      disabled=False,
      offline=True,
    )
    run.log_artifact = lambda artifact, **kw: artifacts.append((run.id, artifact))
    runs.append(run)
    monkeypatch.setattr(wandb, "run", run)
    return run

  monkeypatch.setattr(wandb, "init", init)
  monkeypatch.setattr(wandb, "finish", lambda: None)
  monkeypatch.setattr(wandb, "log", lambda row, **kw: logs.append((row, kw)))
  monkeypatch.setattr(wandb, "Artifact", Mock(side_effect=lambda *a, **k: Mock()))
  monkeypatch.setattr(wandb, "Api", Mock(side_effect=AssertionError("online API")))
  first = training()
  first.runner.registry_name = "must-not-query-offline"
  first.runner.cfg.log_interval = 2
  first.runner.cfg.save_interval = 3
  first.runner.cfg.run_name = "approved-name"
  first.runner.cfg.wandb_project = "approved-project"
  public_log = Mock(wraps=first.runner.logger.log)
  monkeypatch.setattr(first.runner.logger, "log", public_log)
  first.run(4, tmp_path)
  restored = training(81)
  restored.run(6, tmp_path, resume=tmp_path / "training.pt")
  assert len(runs) == 2 and runs[0].id != runs[1].id
  assert configurations[0]["name"] == "approved-name"
  assert configurations[0]["project"] == "approved-project"
  assert configurations[1]["config"]["continuation"] == {
    "source_checkpoint": str(tmp_path / "training.pt"),
    "starting_collected_steps": 4,
    "starting_collected_frames": 8,
    "parent_run": {"id": runs[0].id, "name": runs[0].name, "project": runs[0].project},
  }
  assert public_log.call_count == 3
  assert [kw["step"] for _, kw in logs] == [0, 1, 2, 0]
  assert logs[-1][0]["Counters/collected_steps"] == 6
  assert len(artifacts) == 5
  for _, artifact in artifacts:
    assert artifact.add_file.call_count == 4
  wandb.Api.assert_not_called()


def test_disabled_still_saves_all_local_weights(tmp_path, monkeypatch):
  monkeypatch.setenv("WANDB_MODE", "disabled")
  loop = training()
  loop.run(1, tmp_path)
  assert all(
    (tmp_path / name).is_file()
    for name in ("training.pt", "world_model.pt", "actor.pt", "critic.pt")
  )
  assert torch.load(tmp_path / "training.pt", weights_only=True)["collected_steps"] == 1


def test_offline_sdk_in_bounded_subprocess(tmp_path):
  script = """
import json, sys, torch, wandb
from pathlib import Path
from tests.test_estimator_eit_training import training
from supermimic.utils.logger import Logger
root = Path(sys.argv[1])
wandb.Api = lambda *a, **kw: (_ for _ in ()).throw(AssertionError("online API"))
first = training()
first.run(1, root)
one = torch.load(root / "training.pt", weights_only=True)["wandb_run"]
second = training()
second.run(2, root, resume=root / "training.pt")
saved = torch.load(root / "training.pt", weights_only=True)
assert one["id"] != saved["wandb_run"]["id"]
assert saved["collected_steps"] == 2 and saved["collected_frames"] == 4
assert len(list((root / "wandb").glob("offline-run-*"))) == 2
print("SDK_OK " + json.dumps({"first":one["id"], "second":saved["wandb_run"]["id"]}))
wandb.teardown()
"""
  env = {
    **os.environ,
    "WANDB_MODE": "offline",
    "WANDB_SILENT": "true",
    "WANDB_INIT_TIMEOUT": "15",
    "WANDB_CACHE_DIR": str(tmp_path / "cache"),
    "WANDB_DATA_DIR": str(tmp_path / "data"),
    "OMP_NUM_THREADS": "1",
  }
  result = subprocess.run(
    [sys.executable, "-c", script, str(tmp_path / "run")],
    env=env,
    capture_output=True,
    text=True,
    timeout=45,
  )
  assert result.returncode == 0, result.stdout + result.stderr
  assert "SDK_OK" in result.stdout


def test_failed_common_init_leaves_resumable_initial_checkpoint(tmp_path, monkeypatch):
  monkeypatch.setenv("WANDB_MODE", "disabled")
  loop = training()

  def fail(*args, **kwargs):
    (tmp_path / "wandb").mkdir()
    raise RuntimeError("service unavailable")

  monkeypatch.setattr(loop.runner, "_init_wandb", fail)
  with pytest.raises(RuntimeError, match="service unavailable"):
    loop.run(3, tmp_path)
  saved = torch.load(tmp_path / "training.pt", weights_only=True)
  assert saved["collected_steps"] == 0
  for name in ("world_model", "actor", "critic"):
    equal_tree(torch.load(tmp_path / f"{name}.pt", weights_only=True), saved[name])
  training().run(3, tmp_path, resume=tmp_path / "training.pt")
  assert torch.load(tmp_path / "training.pt", weights_only=True)["collected_steps"] == 3


@pytest.mark.parametrize("filtered", [False, True])
def test_logger_console_selection_preserves_wandb(filtered, monkeypatch, capsys):
  rows = []
  monkeypatch.setattr(wandb, "log", lambda row, **kw: rows.append(row))
  logger = Logger("test", 2, 10, "cpu")
  logger.rewbuffer.append(3.0)
  logger.lenbuffer.append(4)
  logger.ep_extras.append({"EpisodeReward/tracking": 0.5})
  kwargs = {"console_keys": ("Policy/reward", "missing")} if filtered else {}
  logger.log(
    0, 1, 1.0, 0.0, 0.0, {"Policy/reward": 2.0, "Budget/mean_updates": 7}, 2, **kwargs
  )
  output = capsys.readouterr().out
  assert ("Budget/mean_updates:" in output) is not filtered
  assert "Policy/reward:" in output
  assert "Mean reward:" in output and "EpisodeReward/tracking:" in output
  assert rows[0]["Budget/mean_updates"] == 7
  assert rows[0]["Train/mean_reward"] == 3.0


@pytest.mark.parametrize(
  "target,physical",
  [
    (0, []),
    (1, [0]),
    (2, [0, 1]),
    (3, [0, 2]),
    (4, [0, 2, 3]),
    (5, [0, 2, 4]),
  ],
)
def test_log_count_includes_final_partial_interval(
  target, physical, tmp_path, monkeypatch
):
  monkeypatch.setenv("WANDB_MODE", "disabled")
  loop = training()
  loop.runner.cfg.log_interval = 2
  calls = []
  monkeypatch.setattr(loop.runner.logger, "log", lambda **kw: calls.append(kw))
  history = loop.run(target, tmp_path)
  assert [r["step"] for r in history] == physical
  assert [r["it"] for r in calls] == list(range(len(physical)))
  assert all(r["total_it"] == len(physical) for r in calls)


def test_core_aliases_and_resume_preserve_physical_logs(tmp_path, monkeypatch, capsys):
  monkeypatch.setenv("WANDB_MODE", "disabled")
  loop = training()
  loop.runner.cfg.log_interval = 2
  original_step = loop.run_step

  def step():
    metrics = original_step()
    metrics["updates"] = [
      {
        "train": {"Loss/state": 0.25, "Loss/process_nll": 0.75},
        "validation": {"Loss/state": 0.5},
        "policy": {
          "Policy/return": 4.0,
          "Policy/raw_reward": 0.03,
          "Grad/actor_norm": 2.0,
          "Policy/critic_loss": 0.125,
        },
        "budget": {"mean_updates": 1},
      }
    ]
    return metrics

  monkeypatch.setattr(loop, "run_step", step)
  rows = []

  def capture_wandb(runner):
    initialize = runner._init_wandb

    def init(*args, **kwargs):
      initialize(*args, **kwargs)
      monkeypatch.setattr(wandb, "log", lambda row, **kw: rows.append((row, kw)))

    monkeypatch.setattr(runner, "_init_wandb", init)

  capture_wandb(loop.runner)
  history = loop.run(4, tmp_path)
  output = capsys.readouterr().out
  assert "Loss/state:" in output and "Policy/reward:" in output
  for key in (
    "Model/Loss/state:",
    "ModelValidation/",
    "Counters/",
    "Budget/",
    "Policy/return:",
    "Policy/raw_reward:",
    "Grad/",
  ):
    assert key not in output
  expected_reward = 4.0 / (
    loop.policy.cfg.horizon * loop.runner.wm_env.representation.dt
  )
  disk = [
    json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()
  ]
  assert disk == history
  for row in [*history, *(r for r, _ in rows)]:
    assert row["Loss/state"] == row["Model/Loss/state"] == 0.25
    assert row["Policy/reward"] == pytest.approx(expected_reward)
    assert row["Policy/grad_norm"] == row["Grad/actor_norm"] == 2.0
    assert row["Policy/value_function"] == row["Policy/critic_loss"] == 0.125
    assert row["Policy/raw_reward"] == 0.03
    assert row["ModelValidation/Loss/state"] == 0.5
    assert row["Model/Loss/process_nll"] == 0.75
  assert history[-1]["Counters/collected_frames"] == 8
  restored = training(81)
  restored.runner.cfg.log_interval = 2
  capture_wandb(restored.runner)
  restored.run(6, tmp_path, resume=tmp_path / "training.pt")
  assert [kw["step"] for _, kw in rows] == [0, 1, 2, 0, 1]
  disk = [
    json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()
  ]
  assert [row["step"] for row in disk] == [0, 2, 3, 4, 5]
  assert disk[-1]["Counters/collected_steps"] == 6
  assert disk[-1]["Counters/collected_frames"] == 12
