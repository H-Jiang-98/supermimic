"""Common zero-based physical-step saving and coherent background artifacts."""

import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Event
from unittest.mock import Mock

import pytest
import torch

from supermimic.runners import eit_training
from tests.test_estimator_eit_training import training


def test_initial_periodic_final_and_resume_save_points(tmp_path, monkeypatch):
  monkeypatch.setenv("WANDB_MODE", "disabled")
  publications = []

  def record(files, **kwargs):
    state = torch.load(files[-1], weights_only=True)
    assert [p.rsplit("/", 1)[-1] for p in files] == [
      "world_model.pt",
      "actor.pt",
      "critic.pt",
      "training.pt",
    ]
    assert (
      torch.load(files[0], weights_only=True)["dynamics"]["model_revision"]
      == state["model_revision"]
    )
    assert (
      torch.load(files[2], weights_only=True)["contract"] == state["critic"]["contract"]
    )
    publications.append((state["collected_steps"], kwargs["run"].id))

  monkeypatch.setattr(eit_training, "log_model_artifact", record)
  loop = training()
  loop.runner.cfg.save_interval = 3
  loop.run(7, tmp_path)
  assert [step for step, _ in publications] == [0, 1, 4, 7]
  state = torch.load(tmp_path / "training.pt", weights_only=True)
  assert "rounds" not in state and "snapshot_interval" not in state
  assert not (tmp_path / "snapshots").exists()
  other = training(81)
  other.runner.cfg.save_interval = 3
  other.run(11, tmp_path, resume=tmp_path / "training.pt")
  assert [step for step, _ in publications] == [0, 1, 4, 7, 7, 10, 11]
  assert other.collected_steps == 11 and other.collected_frames == 22
  assert not other.runner.replay_buffer.insert_count == 11


def test_publication_waits_before_overwriting_files(tmp_path, monkeypatch):
  monkeypatch.setenv("WANDB_MODE", "disabled")
  started, release = Event(), Event()
  versions = []

  def archive(files, **kwargs):
    before = [open(path, "rb").read() for path in files]
    if not versions:
      started.set()
      assert release.wait(15)
    assert before == [open(path, "rb").read() for path in files]
    versions.append(kwargs["metadata"]["collected_steps"])

  monkeypatch.setattr(eit_training, "log_model_artifact", archive)
  loop = training()
  with ThreadPoolExecutor(max_workers=1) as executor:
    future = executor.submit(loop.run, 1, tmp_path)
    assert started.wait(15)
    assert (
      torch.load(tmp_path / "training.pt", weights_only=True)["collected_steps"] == 0
    )
    release.set()
    future.result(timeout=30)
  assert versions == [0, 1]


@pytest.mark.parametrize("interval", [0, -1, True])
def test_invalid_save_interval_precedes_collection(tmp_path, interval):
  loop = training()
  loop.runner.cfg.save_interval = interval
  with pytest.raises(ValueError, match="save_interval"):
    loop.run(3, tmp_path)
  assert loop.collected_steps == 0


def test_final_duplicate_state_is_not_republished(tmp_path, monkeypatch):
  monkeypatch.setenv("WANDB_MODE", "disabled")
  archive = Mock()
  monkeypatch.setattr(eit_training, "log_model_artifact", archive)
  loop = training()
  loop.run(0, tmp_path)
  assert archive.call_count == 1


def test_final_save_time_is_logged_at_a_log_boundary(tmp_path, monkeypatch):
  monkeypatch.setenv("WANDB_MODE", "disabled")

  class Timer(eit_training.PhaseTimer):
    @contextmanager
    def measure(self, name):
      with super().measure(name):
        yield
      if name == "checkpoint_save_seconds":
        self.seconds[name] = 1.0

  monkeypatch.setattr(eit_training, "PhaseTimer", Timer)
  loop = training()
  loop.runner.cfg.log_interval = 2
  loop.runner.cfg.save_interval = 1000
  loop.run(3, tmp_path)
  rows = [
    json.loads(line) for line in (tmp_path / "profiling.jsonl").read_text().splitlines()
  ]
  assert [r["step"] for r in rows] == [0, 2]
  assert sum(r["seconds"].get("checkpoint_save_seconds", 0) for r in rows) == 3


def test_reference_verification_is_timed_separately_and_preserves_training_peak(
  tmp_path, monkeypatch
):
  from supermimic.runners import belief_training

  monkeypatch.setenv("WANDB_MODE", "disabled")

  class Timer(eit_training.PhaseTimer):
    @contextmanager
    def measure(self, name):
      with super().measure(name):
        yield
      if name in ("checkpoint_save_seconds", "reference"):
        self.seconds[name] = 3.0 if name == "checkpoint_save_seconds" else 2.0

  metadata = eit_training.runtime_metadata

  def peak(reported):
    def measure(*args, **kwargs):
      return {**metadata(*args, **kwargs), "cuda_peak_allocated_bytes": reported}

    return measure

  monkeypatch.setattr(eit_training, "PhaseTimer", Timer)
  monkeypatch.setattr(belief_training, "PhaseTimer", Timer)
  monkeypatch.setattr(eit_training, "runtime_metadata", peak(100))
  monkeypatch.setattr(belief_training, "runtime_metadata", peak(400))
  loop = training()
  loop.before_checkpoint_commit = lambda *_: None
  row = loop.run(1, tmp_path)[-1]
  assert row["Perf/checkpoint_save_seconds"] == 2.0
  assert row["Perf/resume_reference_seconds"] == 4.0
  assert row["Perf/cuda_peak_allocated_bytes"] == 400
