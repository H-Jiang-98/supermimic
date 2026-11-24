"""Recovery retains committed step logs and archives torn or ahead records."""

import json
from pathlib import Path

import pytest
import torch
from filelock import FileLock, Timeout

from supermimic.runners.eit_snapshots import file_sha256
from supermimic.runners.eit_training import EITTraining
from tests.test_estimator_eit_training import training


def test_recovery_archives_uncommitted_logs_without_changing_checkpoint(
  tmp_path, monkeypatch
):
  monkeypatch.setenv("WANDB_MODE", "disabled")
  loop = training()
  loop.runner.cfg.log_interval = 1
  loop.run(4, tmp_path)
  digest = file_sha256(tmp_path / "training.pt")
  for name in ("metrics.jsonl", "profiling.jsonl"):
    path = tmp_path / name
    good = path.read_bytes()
    tail = b'{"step": 4}\n{"step":'
    path.write_bytes(good + tail)
  result = EITTraining.reconcile_output(tmp_path)
  assert result["collected_steps"] == 4
  assert file_sha256(tmp_path / "training.pt") == digest
  for name in ("metrics.jsonl", "profiling.jsonl"):
    rows = [json.loads(line) for line in (tmp_path / name).read_text().splitlines()]
    assert [row["step"] for row in rows] == [0, 1, 2, 3]
    assert next(tmp_path.glob(name + ".interrupted.*")).read_bytes() == tail
  assert not (tmp_path / "snapshots").exists()


def test_recovery_lock_and_validator_prevent_mutation(tmp_path):
  loop = training()
  loop.save(tmp_path / "training.pt")
  with FileLock(tmp_path / ".belief-training.lock"):
    with pytest.raises(Timeout):
      EITTraining.reconcile_output(tmp_path)

  def reject(state):
    raise ValueError("wrong source")

  with pytest.raises(ValueError, match="wrong source"):
    EITTraining.reconcile_output(tmp_path, validate_state=reject)
  assert not (tmp_path / "metrics.jsonl").exists()


def test_failed_reference_keeps_all_four_previously_published_files(
  tmp_path, monkeypatch
):
  monkeypatch.setenv("WANDB_MODE", "disabled")
  from supermimic.runners import eit_training

  loop = training()
  prior = {}

  def record(files, **kwargs):
    prior.update({path: file_sha256(path) for path in map(Path, files)})

  monkeypatch.setattr(eit_training, "log_model_artifact", record)

  def fail(path, state):
    if state["collected_steps"] == 4:
      raise RuntimeError("injected reference failure")

  loop.before_checkpoint_commit = fail
  with pytest.raises(RuntimeError, match="injected reference"):
    loop.run(4, tmp_path)
  assert {p.name for p in prior} == {
    "training.pt",
    "world_model.pt",
    "actor.pt",
    "critic.pt",
  }
  assert all(file_sha256(p) == digest for p, digest in prior.items())


def test_resume_completes_interrupted_sidecar_replacement(tmp_path, monkeypatch):
  from supermimic.runners import eit_training
  from supermimic.runners.eit_resume_reference import equal_tree

  monkeypatch.setenv("WANDB_MODE", "disabled")
  loop = training()
  replace = Path.replace

  def interrupt(self, target):
    if (
      self.parent.name == ".eit-publication"
      and self.name == "critic.pt"
      and loop.collected_steps == 4
    ):
      raise OSError("injected rename failure")
    return replace(self, target)

  with monkeypatch.context() as m:
    m.setattr(Path, "replace", interrupt)
    with pytest.raises(OSError, match="injected rename"):
      loop.run(4, tmp_path)

  eit_training.EITTraining.reconcile_output(tmp_path)
  saved = torch.load(tmp_path / "training.pt", weights_only=True)
  assert saved["collected_steps"] == 4
  for name in ("world_model", "actor", "critic"):
    equal_tree(torch.load(tmp_path / f"{name}.pt", weights_only=True), saved[name])
  assert not (tmp_path / ".eit-publication").exists()


def test_first_publication_can_retry_in_its_own_interrupted_output(
  tmp_path, monkeypatch
):
  monkeypatch.setenv("WANDB_MODE", "disabled")
  loop = training()

  def fail(*args):
    (tmp_path / "wandb").mkdir()
    raise RuntimeError("first save failed after W&B started")

  loop.before_checkpoint_commit = fail
  with pytest.raises(RuntimeError, match="first save failed"):
    loop.run(2, tmp_path)
  assert not (tmp_path / "training.pt").exists()
  # A dead process can leave its partially written weights in staging.
  stage = tmp_path / ".eit-publication"
  stage.mkdir()
  (stage / "staging").touch()
  (stage / "actor.pt").write_bytes(b"partial")
  foreign = tmp_path / "foreign.txt"
  foreign.write_text("unrelated")
  with pytest.raises(ValueError, match="Nonempty output"):
    training().run(2, tmp_path)
  assert foreign.read_text() == "unrelated"
  foreign.unlink()
  marker = tmp_path / ".eit-initializing"
  marker.unlink()
  with pytest.raises(ValueError, match="Nonempty output"):
    training().run(2, tmp_path)
  marker.touch()
  training().run(2, tmp_path)
  assert torch.load(tmp_path / "training.pt", weights_only=True)["collected_steps"] == 2
  assert not marker.exists()
  assert not stage.exists()
  assert (tmp_path / "wandb").is_dir()


def test_reconcile_legacy_split_repairs_weights_without_rewriting_checkpoint(
  tmp_path, monkeypatch
):
  from supermimic.runners.eit_resume_reference import equal_tree

  monkeypatch.setenv("WANDB_MODE", "disabled")
  training().run(2, tmp_path)
  checkpoint = tmp_path / "training.pt"
  digest = file_sha256(checkpoint)
  (tmp_path / ".eit-log-interval.json").unlink()
  torch.save({"stale": True}, tmp_path / "actor.pt")
  (tmp_path / "critic.pt").unlink()
  EITTraining.reconcile_output(tmp_path)
  assert file_sha256(checkpoint) == digest
  state = torch.load(checkpoint, weights_only=True)
  for name in ("actor", "critic", "world_model"):
    equal_tree(torch.load(tmp_path / f"{name}.pt", weights_only=True), state[name])
  with monkeypatch.context() as m:
    m.setattr(torch, "save", lambda *_args: pytest.fail("unexpected second migration"))
    EITTraining.reconcile_output(tmp_path)


@pytest.mark.parametrize("boundary", ["log_and_save", "save_only"])
def test_resume_recovers_actual_interval_budget_and_profile_once(
  tmp_path, monkeypatch, boundary
):
  monkeypatch.setenv("WANDB_MODE", "disabled")
  loop = training()
  loop.runner.cfg.save_interval = 3
  loop.runner.cfg.log_interval = 3 if boundary == "log_and_save" else 8
  log, step = loop.runner.logger.log, loop.run_step

  def interrupt_log(*args, **kwargs):
    if kwargs["loss_dict"]["Counters/collected_steps"] == 4:
      raise RuntimeError("injected interruption")
    return log(*args, **kwargs)

  def interrupt_step(*args, **kwargs):
    if loop.collected_steps == 4:
      raise RuntimeError("injected interruption")
    return step(*args, **kwargs)

  monkeypatch.setattr(loop.runner.logger, "log", interrupt_log)
  monkeypatch.setattr(loop, "run_step", interrupt_step)
  with pytest.raises(RuntimeError, match="injected interruption"):
    loop.run(6, tmp_path)
  interval = json.loads((tmp_path / ".eit-log-interval.json").read_text())
  assert interval["metrics.jsonl"]["Budget/mean_updates"] == 1
  # Simulate interruption between the metrics append and the profiling append.
  with (tmp_path / "metrics.jsonl").open("a") as output:
    output.write(json.dumps(interval["metrics.jsonl"]) + "\n")
  resumed = training()
  resumed.run(5, tmp_path, resume=tmp_path / "training.pt")
  for name, expected in interval.items():
    rows = [json.loads(line) for line in (tmp_path / name).read_text().splitlines()]
    assert [row["step"] for row in rows] == [0, 3, 4]
    assert rows[1] == expected
  assert interval["profiling.jsonl"]["seconds"]["world_model_seconds"] > 0
