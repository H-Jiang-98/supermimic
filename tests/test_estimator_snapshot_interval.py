"""Sparse historical snapshots preserve latest-state recovery and audit guards."""

import json
from copy import deepcopy

import pytest
import torch

from supermimic.runners import belief_training
from supermimic.runners.eit_snapshots import file_sha256
from supermimic.runners.eit_training import EITTraining
from tests.test_estimator_belief_training import assert_tree
from tests.test_estimator_eit_training import training


def snapshot_rounds(output):
  rows = json.loads((output / "snapshots/index.json").read_text())["snapshots"]
  assert {p.name for p in (output / "snapshots").glob("*.pt")} == {
    f"frames_{r['collected_frames']}.pt" for r in rows
  }
  for row in rows:
    assert file_sha256(output / row["path"]) == row["sha256"]
  return [row["round"] for row in rows]


@pytest.mark.parametrize("interval,expected", [(2, [0, 2, 4]), (3, [0, 3])])
def test_interval_keeps_latest_unscheduled_round_and_all_metrics(
  tmp_path, interval, expected
):
  loop = training()
  loop.run(5, tmp_path, snapshot_interval=interval)
  assert snapshot_rounds(tmp_path) == expected
  state = torch.load(tmp_path / "training.pt", weights_only=True)
  assert state["rounds"] == state["policy"]["actor_steps"] == 5
  assert state["snapshot_interval"] == interval
  for name in ("metrics.jsonl", "profiling.jsonl"):
    assert [
      json.loads(line)["round"] for line in (tmp_path / name).read_text().splitlines()
    ] == list(range(1, 6))


def test_resume_preserves_absolute_cadence_and_can_override_future_rounds(tmp_path):
  checkpoint = tmp_path / "training.pt"
  training().run(2, tmp_path, snapshot_interval=3)
  assert snapshot_rounds(tmp_path) == [0]
  resumed = training(81)
  resumed.run(2, tmp_path, resume=checkpoint)
  assert resumed.snapshot_interval == 3
  assert snapshot_rounds(tmp_path) == [0, 3]
  previous = {p.name: p.read_bytes() for p in (tmp_path / "snapshots").glob("*.pt")}
  training(91).run(2, tmp_path, resume=checkpoint, snapshot_interval=2)
  assert snapshot_rounds(tmp_path) == [0, 3, 6]
  assert torch.load(checkpoint, weights_only=True)["snapshot_interval"] == 2
  for name, data in previous.items():
    assert (tmp_path / "snapshots" / name).read_bytes() == data


@pytest.mark.parametrize("rounds,expected", [(2, [0]), (3, [0, 3])])
def test_recovery_after_checkpoint_commit_obeys_recorded_interval(
  tmp_path, monkeypatch, rounds, expected
):
  original = belief_training.publish_snapshot

  def interrupt(checkpoint, state=None):
    state = state or torch.load(checkpoint, weights_only=True)
    if state["rounds"] == rounds:
      raise InterruptedError("checkpoint committed before snapshot and metrics")
    return original(checkpoint, state)

  with monkeypatch.context() as context:
    context.setattr(belief_training, "publish_snapshot", interrupt)
    with pytest.raises(InterruptedError):
      training().run(rounds, tmp_path, snapshot_interval=3)
  checkpoint = tmp_path / "training.pt"
  digest = file_sha256(checkpoint)
  recovered = EITTraining.reconcile_output(tmp_path)
  assert recovered["round"] == rounds
  assert recovered["repaired_metric_bytes"] > 0
  assert snapshot_rounds(tmp_path) == expected
  assert file_sha256(checkpoint) == digest
  assert EITTraining.reconcile_output(tmp_path)["repaired_metric_bytes"] == 0
  assert snapshot_rounds(tmp_path) == expected


def test_unscheduled_resume_still_rejects_rollback_behind_snapshot_index(tmp_path):
  loop = training()
  loop.run(2, tmp_path, snapshot_interval=3)
  checkpoint = tmp_path / "training.pt"
  older = checkpoint.read_bytes()
  loop.run(1, tmp_path)
  checkpoint.write_bytes(older)
  for name in ("metrics.jsonl", "profiling.jsonl"):
    path = tmp_path / name
    path.write_bytes(b"".join(path.read_bytes().splitlines(keepends=True)[:2]))
  with pytest.raises(ValueError, match="index is ahead"):
    training().run(1, tmp_path, resume=checkpoint)
  assert checkpoint.read_bytes() == older
  assert snapshot_rounds(tmp_path) == [0, 3]


def test_legacy_checkpoint_keeps_every_round_publication(tmp_path):
  source = training()
  state = source.state_dict()
  state.pop("snapshot_interval")
  checkpoint = tmp_path / "training.pt"
  torch.save(state, checkpoint)
  restored = training(81)
  restored.snapshot_interval = 10
  restored.run(1, tmp_path, resume=checkpoint)
  assert restored.snapshot_interval == 1
  assert snapshot_rounds(tmp_path) == [0, 1]


@pytest.mark.parametrize("interval", [0, -1, True, 1.5, "10"])
def test_invalid_interval_rejected_before_writing_or_restoring(tmp_path, interval):
  loop = training()
  before = deepcopy(loop.state_dict())
  with pytest.raises(ValueError, match="positive integer"):
    loop.run(1, tmp_path, snapshot_interval=interval)
  assert list(tmp_path.iterdir()) == []
  state = deepcopy(before)
  state["snapshot_interval"] = interval
  with pytest.raises(ValueError, match="positive integer"):
    loop.load_state_dict(state)
  assert_tree(loop.state_dict(), before)


def test_pilot_explicitly_preserves_every_round_snapshots():
  from supermimic.scripts.run_eit_pilot import training_arguments

  args = training_arguments(
    {"motion_file": "walking.npz", "device": "cpu", "training_seed": 2026}
  )
  assert args[args.index("--snapshot-interval") + 1] == "1"


@pytest.mark.parametrize("interval", ["0", "-1"])
def test_cli_rejects_nonpositive_interval_before_creating_environment(
  monkeypatch, interval
):
  from supermimic.scripts import train_eit

  monkeypatch.setattr(
    "sys.argv",
    ["train_eit", "--motion-file", "unused.npz", "--snapshot-interval", interval],
  )
  monkeypatch.setattr(
    train_eit, "create_env", lambda *_: pytest.fail("created environment")
  )
  with pytest.raises(SystemExit) as error:
    train_eit.main()
  assert error.value.code == 2
