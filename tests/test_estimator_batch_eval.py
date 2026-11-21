from pathlib import Path
from statistics import fmean

import pytest

from supermimic.runners.run_config import EIT_TRAINING_VERSION
from supermimic.scripts import batch_eval_eit as batch


def test_hand_computed_clip_and_episode_weighting():
  data = batch.read_json(
    Path(__file__).resolve().parents[1] / "docs/validation/eit_clip_aggregation.json"
  )
  for rows in data["clips"].values():
    for row in rows:
      frames = row.pop("mpkpe_frames")
      row.update(
        metrics={"mpkpe": fmean(frames) if frames else None}, metric_steps=len(frames)
      )
  result = batch.aggregate(data["clips"])
  expected = data["expected"]
  for key in ("success_rate", "progress", "mpkpe", "length", "return"):
    assert result["scores"][key] == pytest.approx(expected[key])
  assert result["valid_clips"]["mpkpe"] == expected["mpkpe_valid_clips"]
  for clip, report in result["clips"].items():
    assert report["valid_episodes"]["mpkpe"] == expected["mpkpe_valid_episodes"][clip]
    assert report["metric_frames"] == expected["mpkpe_frames"][clip]


def candidate(frames):
  return dict(
    group="B",
    training_seed=2,
    collected_frames=frames,
    sha256=f"{frames:064x}",
    training_motion_digest="training",
  )


def rank_row(frames, progress, mpkpe, complete=True, success=0):
  return dict(
    candidate=candidate(frames),
    complete=complete,
    scores=dict(success_rate=success, progress=progress, mpkpe=mpkpe),
  )


def test_early_failure_ranking_baseline_missing_and_ties():
  rows = [
    rank_row(0, 1, 0, success=1),
    rank_row(10, 0.1, 0),
    rank_row(20, 0.3, None),
    rank_row(30, 0.3, 0.2),
    rank_row(40, 0.3, 0.1),
    rank_row(50, 0.3, 0.1),
  ]
  assert batch.select(rows) == [candidate(40)]
  rows[-1]["complete"] = False
  assert batch.select(rows) == []
  rows[-1] = rank_row(50, 0.1, 1, success=0.1)
  assert batch.select(rows) == [candidate(50)]


def test_model_stage_rejected_by_formal_admission():
  state = dict(version=EIT_TRAINING_VERSION, stage="model", replay_mode="full")
  with pytest.raises(ValueError, match="stage"):
    batch.admit(state, {}, {}, [])


def spec(tmp_path):
  path = tmp_path / "clip.npz"
  path.write_bytes(b"test")
  motion = dict(path=path.name, subject="one")
  return dict(
    design="seen",
    device="cpu",
    runs=[dict(directory="run", group="B", training_seed=2, frames=[10, 20])],
    training=[motion],
    validation=[dict(motion, seed=12, num_envs=2)],
    test=[dict(motion, seed=13, num_envs=2)],
  )


@pytest.mark.parametrize(
  ("kind", "error"),
  [
    ("clip", "clip splits overlap"),
    ("seed", "independent seeds"),
    ("same_seed", "independent of training"),
  ],
)
def test_holdout_and_seen_split_checks(tmp_path, monkeypatch, kind, error):
  conf = spec(tmp_path)
  if kind == "clip":
    conf["design"] = "heldout_clip"
  elif kind == "seed":
    conf["validation"][0]["seed"] = 13
  else:
    conf["test"][0]["seed"] = 2
  monkeypatch.setattr(
    batch, "probe", lambda path, device: {"evaluation_motion_digest": "loaded"}
  )
  with pytest.raises(ValueError, match=error):
    batch.freeze(conf, tmp_path)


def result(c, scene):
  episodes = [
    dict(
      start_frame=0,
      length=2,
      reference_steps=8,
      reason="failure",
      censored=False,
      metric_steps=1,
      env_id=i,
      metrics=dict.fromkeys(batch.TRACKING, 2),
      **{"return": 0.1},
    )
    for i in range(scene["episodes"])
  ]
  return dict(
    scene,
    group=c["group"],
    episodes=episodes,
    checkpoint_sha256=c["sha256"],
    training_motion_digest=c["training_motion_digest"],
    evaluation_motion_sha256=scene["sha256"],
    actor_state_source="posterior",
    actor_backend="pytorch",
    mode="eval",
    time_limit_protocol=scene["protocol"],
  )


def test_failed_retry_missing_dedup_and_test_uses_only_winner(tmp_path, monkeypatch):
  monkeypatch.setattr(batch, "validate_scene", lambda scene: None)
  monkeypatch.setattr(
    batch,
    "probe",
    lambda path, device: dict(
      evaluation_motion_digest="loaded",
      clip_frames=8,
      control_period_s=0.02,
      timeout_steps=500,
      timeout_s=10,
    ),
  )
  manifest = batch.freeze(spec(tmp_path), tmp_path)
  items = [candidate(10), candidate(20)]
  monkeypatch.setattr(batch, "candidates", lambda m: items)
  calls = []

  def runner(c, scene, path, key):
    calls.append((c["collected_frames"], scene["seed"]))
    if len(calls) == 1:
      raise ValueError("simulator failed")
    batch.write_json(
      path,
      dict(
        task_key=key, snapshot_sha256=c["sha256"], scene=scene, result=result(c, scene)
      ),
    )

  monkeypatch.setattr(batch, "execute", runner)
  report = batch.study(manifest, tmp_path, "validation", False)
  assert [t["status"] for t in report["pending_validation"]] == ["missing", "missing"]
  report = batch.study(manifest, tmp_path, "validation", True)
  assert (
    not report["selected"]
    and report["pending_validation"][0]["error"] == "simulator failed"
  )
  assert not batch.study(manifest, tmp_path, "test", True)["complete"]
  recovered = batch.study(manifest, tmp_path, "validation", True)
  assert recovered["complete"] and recovered["selected"] == items[:1]
  assert recovered == batch.study(manifest, tmp_path, "validation", True)
  scored = recovered["trained"][0]
  assert (
    scored["expected_episodes"]
    == scored["recorded_episodes"]
    == scored["valid_episodes"]["mpkpe"]
    == 2
  )
  assert scored["expected_clips"] == scored["valid_clips"]["mpkpe"] == 1
  assert batch.study(manifest, tmp_path, "test", True)["complete"]
  assert calls == [(10, 12), (20, 12), (10, 12), (10, 13)]
  scene = manifest["validation"][0]
  data = result(items[0], scene)
  data["evaluation_motion_sha256"] = "different-raw-file"
  with pytest.raises(ValueError, match="identity"):
    batch.checked_episodes(data, items[0], scene)
  data["evaluation_motion_sha256"] = scene["sha256"]
  data["episodes"][1]["env_id"] = 0
  with pytest.raises(ValueError, match="duplicate episode"):
    batch.checked_episodes(data, items[0], scene)


def test_censored_completion_and_empty_metric(tmp_path, monkeypatch):
  monkeypatch.setattr(
    batch,
    "probe",
    lambda path, device: dict(
      evaluation_motion_digest="loaded",
      clip_frames=8,
      control_period_s=0.02,
      timeout_steps=500,
      timeout_s=10,
    ),
  )
  scene = batch.freeze(spec(tmp_path), tmp_path)["validation"][0]
  c = candidate(10)
  data = result(c, scene)
  row = data["episodes"][0]
  row.update(censored=True, reason="evaluation_truncated", metric_steps=2)
  records = batch.checked_episodes(data, c, scene)
  summary = batch.aggregate({"clip": records})
  assert summary["scores"]["success_rate"] is None
  row.update(
    reason="failure",
    censored=False,
    length=1,
    metric_steps=0,
    metrics=dict.fromkeys(batch.TRACKING),
  )
  assert batch.aggregate({"clip": [row]})["scores"]["mpkpe"] is None


def test_admission_reads_checkpoint_and_pairs_initialization_by_seed(tmp_path):
  import torch

  from supermimic.critic_inputs import BeliefCritic
  from supermimic.runners.eit_snapshots import publish_snapshot

  train = [{"sha256": "raw"}]
  runs = []
  for group in ("A", "B"):
    directory = tmp_path / group
    directory.mkdir()
    runs.append(
      dict(directory=str(directory), group=group, training_seed=2, frames=[10])
    )
    optimizer = {
      "param_groups": [{"params": [0]}],
      "state": {0: {"step": torch.tensor(1.0)}},
    }
    state = dict(
      version=EIT_TRAINING_VERSION,
      stage="policy",
      replay_mode="full",
      rounds=1,
      collected_frames=10,
      training_seed=2,
      training_motion_files=train,
      motion_digest="loaded",
      initialization_digest=("a" if group == "A" else "b") * 64,
      runner_config={
        "estimator_inputs": {"group": group, "measurement_noise": {}},
        "initial_prior": {},
      },
      learner={
        "model_version": "world",
        "mean": {"steps": 1, "optimizers": [optimizer]},
        "cov": {"steps": 1, "optimizer": optimizer},
      },
      policy={
        "actor_steps": 1,
        "critic_steps": 1,
        "actor_optimizer": optimizer,
        "critic_optimizer": optimizer,
      },
      critic=BeliefCritic(group, 2, (4,)).checkpoint(),
      world_model={"sim_context": {"scale": torch.ones(2)}},
      actor_version="actor",
      critic_version="critic",
    )
    path = directory / "training.pt"
    torch.save(state, path)
    publish_snapshot(path, state)
  items = batch.candidates({"runs": runs, "training": train})
  assert [i["group"] for i in items] == ["A", "B"]
  assert all("initialization fingerprints differ" in i["error"] for i in items)
  record = batch.candidates({"runs": runs[:1], "training": train})[0]
  assert "error" not in record and record["actor_steps"] == 1
  assert (
    "expanded training motion"
    in batch.candidates({"runs": runs[:1], "training": []})[0]["error"]
  )
  index = batch.read_json(tmp_path / "B" / "snapshots/index.json")["snapshots"][0]
  state["policy"]["actor_steps"] = 2
  with pytest.raises(ValueError, match="Actual optimizer"):
    batch.admit(state, runs[1], index, train)
  state["initialization_digest"] = None
  with pytest.raises(ValueError, match="initialization fingerprint"):
    batch.admit(state, runs[1], index, train)


def test_real_validation_fixture_reaggregates():
  sample = batch.read_json(
    Path(__file__).resolve().parents[1] / "docs/validation/eit_batch_walk.json"
  )
  for task in sample["tasks"]:
    summary = batch.aggregate({sample["sha256"]: task["episodes"]})
    assert summary["scores"] == pytest.approx(task["expected_scores"])
  tests = [r for r in sample["tasks"] if r["split"] == "test"]
  assert all(r["snapshot_sha256"] == sample["selected_sha256"] for r in tests)


def test_repacked_clip_cannot_cross_holdout_or_get_two_clip_votes(
  tmp_path, monkeypatch
):
  conf = spec(tmp_path)
  alternate = tmp_path / "repacked.npz"
  alternate.write_bytes(b"other archive, same loaded references")
  conf["validation"][0]["path"] = str(alternate)
  monkeypatch.setattr(
    batch,
    "probe",
    lambda path, device: {"evaluation_motion_digest": "same-packed-contents"},
  )
  conf["design"] = "heldout_clip"
  with pytest.raises(ValueError, match="clip splits overlap"):
    batch.freeze(conf, tmp_path)
  conf["design"] = "seen"
  conf["validation"].append(
    dict(conf["validation"][0], path=conf["training"][0]["path"])
  )
  with pytest.raises(ValueError, match="duplicate validation scenarios"):
    batch.freeze(conf, tmp_path)


@pytest.mark.parametrize(
  "change", ["termination", "reward", "physics", "action", "observation"]
)
def test_config_drift_rejects_cached_and_missing_tasks(tmp_path, monkeypatch, change):
  from copy import deepcopy

  import numpy as np

  from supermimic.scripts import eval_eit

  conf = spec(tmp_path)
  motion = tmp_path / "clip.npz"
  np.savez(motion, joint_pos=np.zeros((8, 29)))
  cfg = eval_eit.evaluation_env_config(motion, 0)
  frozen = batch.conditions(cfg)
  monkeypatch.setattr(
    batch,
    "probe",
    lambda path, device: dict(
      evaluation_motion_digest="loaded",
      clip_frames=8,
      control_period_s=0.02,
      timeout_steps=500,
      timeout_s=10,
      conditions=frozen,
    ),
  )
  manifest = batch.freeze(conf, tmp_path)
  items = [candidate(10), candidate(20)]
  monkeypatch.setattr(batch, "candidates", lambda m: items)
  scene = manifest["validation"][0]
  key = batch.task_key(items[0], scene)
  cached = tmp_path / f"tasks/{key}.json"
  batch.write_json(
    cached,
    dict(
      task_key=key,
      snapshot_sha256=items[0]["sha256"],
      scene=scene,
      result=result(items[0], scene),
    ),
  )
  cached_bytes = cached.read_bytes()
  changed = deepcopy(cfg)
  if change == "termination":
    changed.terminations["anchor_pos"].params["threshold"] *= 2
  elif change == "reward":
    changed.rewards["motion_body_pos"].weight *= 2
  elif change == "physics":
    changed.sim.mujoco.iterations *= 2
  elif change == "action":
    next(iter(changed.actions.values())).scale = 0.5
  else:
    terms = changed.observations["actor"].terms
    changed.observations["actor"].terms = dict(reversed(list(terms.items())))
  assert batch.digest(batch.conditions(changed)) != batch.digest(frozen)
  monkeypatch.setattr(eval_eit, "evaluation_env_config", lambda *args: changed)

  def unexpected_execution(*args):
    pytest.fail("A changed environment must be rejected before evaluation")

  monkeypatch.setattr(batch, "execute", unexpected_execution)
  for run in (False, True):
    report = batch.study(manifest, tmp_path, "validation", run)
    assert not report["complete"] and not report["selected"]
    assert len(report["pending_validation"]) == 2
    assert all(
      task["status"] == "failed" and "configuration differs" in task["error"]
      for task in report["pending_validation"]
    )
    assert all(row["recorded_episodes"] == 0 for row in report["trained"])
  assert cached.read_bytes() == cached_bytes


def test_environment_conditions_separate_seed_and_environment_count(tmp_path):
  import numpy as np

  from supermimic.scripts.eval_eit import evaluation_env_config

  motion = tmp_path / "clip.npz"
  np.savez(motion, joint_pos=np.zeros((8, 29)))
  cfg = evaluation_env_config(motion, 0)
  frozen = batch.conditions(cfg)
  cfg.seed = 12
  cfg.scene.num_envs = 2
  assert batch.conditions(cfg) == frozen
  assert isinstance(batch.digest(frozen), str)


def test_old_partial_configuration_manifest_requires_refreezing(tmp_path, monkeypatch):
  manifest = {"version": "eit_batch_full_clip_v1"}
  manifest["manifest_digest"] = batch.digest(manifest)
  path = tmp_path / "manifest.json"
  batch.write_json(path, manifest)
  monkeypatch.setattr(
    "sys.argv",
    [
      "batch_eval_eit",
      "report",
      "--manifest",
      str(path),
      "--output",
      str(tmp_path),
      "--split",
      "validation",
    ],
  )
  with pytest.raises(SystemExit) as error:
    batch.main()
  assert error.value.code == 2
