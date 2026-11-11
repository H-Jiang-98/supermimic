from types import SimpleNamespace
from unittest.mock import Mock

from supermimic.utils import wandb_artifacts


def _mock_wandb(monkeypatch, *, disabled=False, offline=False):
  run = SimpleNamespace(
    id="test",
    entity="test-entity",
    project="test-project",
    disabled=disabled,
    offline=offline,
    log_artifact=Mock(),
  )
  api = Mock()
  artifact = Mock()
  monkeypatch.setattr(wandb_artifacts.wandb, "run", run)
  monkeypatch.setattr(wandb_artifacts.wandb, "Api", api)
  monkeypatch.setattr(wandb_artifacts.wandb, "Artifact", artifact)
  return run, api, artifact


def test_disabled_checkpoint_never_accesses_artifact_api(monkeypatch):
  run, api, artifact = _mock_wandb(monkeypatch, disabled=True)
  wandb_artifacts.log_model_artifact(["world_model.pt", "actor.pt"])
  artifact.assert_not_called()
  run.log_artifact.assert_not_called()
  api.assert_not_called()


def test_offline_checkpoint_is_queued_without_remote_pruning(monkeypatch):
  run, api, artifact = _mock_wandb(monkeypatch, offline=True)
  wandb_artifacts.log_model_artifact(["actor.pt"])
  artifact.return_value.add_file.assert_called_once_with("actor.pt")
  run.log_artifact.assert_called_once_with(artifact.return_value)
  api.assert_not_called()


def test_online_checkpoint_retains_milestones_aliases_and_recent(monkeypatch):
  run, api, artifact = _mock_wandb(monkeypatch)
  versions = [
    SimpleNamespace(version=f"v{i}", aliases=["best"] if i == 2 else [], delete=Mock())
    for i in range(8)
  ]
  api.return_value.artifacts.return_value = list(reversed(versions))
  wandb_artifacts.log_model_artifact(["actor.pt"], keep_last=2, milestone_every=5)
  run.log_artifact.assert_called_once_with(artifact.return_value)
  api.return_value.artifacts.assert_called_once_with(
    "model", "test-entity/test-project/model-test"
  )
  for i, version in enumerate(versions):
    if i in (1, 3, 4):
      version.delete.assert_called_once()
    else:
      version.delete.assert_not_called()
