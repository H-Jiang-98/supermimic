"""Committed checkpoints survive torn publication/logs without fabricated costs."""

import json

import pytest
from filelock import FileLock, Timeout

from supermimic.runners.eit_snapshots import file_sha256
from supermimic.runners.eit_training import EITTraining
from tests.test_estimator_eit_training import training


@pytest.mark.parametrize("rounds", [0, 5, 10])
def test_reconcile_committed_round_without_another_training_step(tmp_path, rounds):
  loop = training()
  checkpoint = tmp_path / "training.pt"
  if rounds:
    loop.run(rounds, tmp_path)
    registry_path = tmp_path / "snapshots/index.json"
    registry = json.loads(registry_path.read_text())
    last = registry["snapshots"].pop()
    (tmp_path / last["path"]).unlink()
    registry_path.write_text(json.dumps(registry))
    metrics = tmp_path / "metrics.jsonl"
    lines = metrics.read_bytes().splitlines(keepends=True)
    expected_metrics = b"".join(lines)
    metrics.write_bytes(b"".join(lines[:-1]) + lines[-1][:23])
    profiles = tmp_path / "profiling.jsonl"
    lines = profiles.read_bytes().splitlines(keepends=True)
    fragment = lines[-1][:23]
    profiles.write_bytes(b"".join(lines[:-1]) + fragment)
  else:
    loop.save(checkpoint)
  digest = file_sha256(checkpoint)
  result = EITTraining.reconcile_output(tmp_path)
  assert result["round"] == rounds
  assert file_sha256(checkpoint) == digest
  snapshot = tmp_path / "snapshots" / f"frames_{rounds * 8}.pt"
  assert file_sha256(snapshot) == digest
  if rounds:
    assert metrics.read_bytes() == expected_metrics
    assert result["missing_profile_rounds"] == [rounds]
    assert (
      list(tmp_path.glob("profiling.interrupted.*.fragment"))[0].read_bytes()
      == fragment
    )
    assert [
      json.loads(line)["round"] for line in profiles.read_text().splitlines()
    ] == list(range(1, rounds))
    note = json.loads((tmp_path / "profiling_recovery.json").read_text())
    assert note["missing_profile_rounds"] == [rounds]
    assert "no zero" in note["missing_timing_policy"]
  again = EITTraining.reconcile_output(tmp_path)
  assert again["repaired_metric_bytes"] == 0
  assert not again["profiling_tail_repaired"]
  assert again["missing_profile_rounds"] == ([rounds] if rounds else [])


def test_complete_profile_tail_is_sealed_without_losing_its_timing(tmp_path):
  loop = training()
  loop.run(1, tmp_path)
  path = tmp_path / "profiling.jsonl"
  complete = path.read_bytes()
  path.write_bytes(complete[:-1])
  result = EITTraining.reconcile_output(tmp_path)
  assert result["missing_profile_rounds"] == []
  assert path.read_bytes() == complete
  assert result["archived_profile_fragment"] is None


def test_reconciliation_uses_training_lock_and_rejects_corrupt_complete_rows(tmp_path):
  loop = training()
  loop.run(1, tmp_path)
  with FileLock(tmp_path / ".belief-training.lock"):
    with pytest.raises(Timeout):
      EITTraining.reconcile_output(tmp_path)
  path = tmp_path / "profiling.jsonl"
  path.write_bytes(b"not-json\n")
  with pytest.raises(ValueError, match="Malformed complete profiling"):
    EITTraining.reconcile_output(tmp_path)
  assert path.read_bytes() == b"not-json\n"


def test_live_reference_hook_precedes_checkpoint_commit_and_can_retry(tmp_path):
  loop = training()
  loop.run(1, tmp_path)
  checkpoint = tmp_path / "training.pt"
  old_digest = file_sha256(checkpoint)
  reference = tmp_path / "reference.sha256"

  def interrupted(pending, state):
    if state["rounds"] == 2:
      assert file_sha256(checkpoint) == old_digest
      reference.write_text(file_sha256(pending))
      raise InterruptedError("reference written, checkpoint not committed")

  loop.before_checkpoint_commit = interrupted
  with pytest.raises(InterruptedError):
    loop.run(1, tmp_path)
  assert file_sha256(checkpoint) == old_digest
  assert not (tmp_path / "snapshots/frames_16.pt").exists()
  retry = training()
  retry.before_checkpoint_commit = lambda pending, _state: reference.write_text(
    file_sha256(pending)
  )
  retry.run(1, tmp_path, resume=checkpoint)
  assert file_sha256(checkpoint) == reference.read_text()
  assert file_sha256(tmp_path / "snapshots/frames_16.pt") == reference.read_text()
  rows = [
    json.loads(row) for row in (tmp_path / "profiling.jsonl").read_text().splitlines()
  ]
  final = rows[-1]["seconds"]
  assert final["resume_reference_seconds"] > 0
  assert final["round_seconds_including_resume_reference"] == pytest.approx(
    final["round_seconds"] + final["resume_reference_seconds"]
  )
