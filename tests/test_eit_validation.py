"""Analytic metrics, paired sampling, clip weighting and immutable results."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from supermimic.belief import BeliefState
from supermimic.estimator_inputs import GaussianMeasurementConfig
from supermimic.utils.lie_state import LieState, boxplus
from supermimic.validation.config import StudyConfig, load_config
from supermimic.validation.domains import domain_table
from supermimic.validation.metrics import ODOMETRY, BinnedMetrics, posterior_values
from supermimic.validation.report import aggregate
from supermimic.validation.runtime import PairedSensors, make_env_config


def state(batch=1):
  full = torch.zeros(batch, 71)
  full[:, 3] = 1
  return LieState.from_full(full)


def test_correlated_square_root_nll_nees_and_degree_rmse():
  truth = state()
  error = torch.zeros(1, 70)
  error[0, :9] = torch.tensor([0.0, 0.0, np.pi / 6, 0.2, -0.3, 0.1, 0.4, 0.2, 0.5])
  target = boxplus(truth, error)
  lower = torch.eye(70)[None]
  lower[0, 1, 0] = 0.7
  lower[0, 5, 2] = 0.2
  belief = BeliefState(truth, covariance_sqrt=lower)
  actual = posterior_values(belief, target)[0]
  covariance = (lower.double() @ lower.double().mT)[0, :9, :9]
  e = error[0, :9].double()
  nees = e @ torch.linalg.solve(covariance, e)
  nll = (9 * np.log(2 * np.pi) + torch.linalg.slogdet(covariance).logabsdet + nees) / 2
  assert actual[0].sqrt() == pytest.approx(30, abs=1e-5)
  assert actual[3] == pytest.approx(float(nll), abs=1e-6)
  assert actual[4] == pytest.approx(float(nees), abs=1e-6)
  assert actual[1] == pytest.approx(float(target.position.square().sum()), abs=1e-7)
  lower[0, 0, 0] = 0
  with pytest.raises(ValueError, match="square root"):
    posterior_values(belief, target)


def test_rmse_after_time_mean_and_partial_last_bin():
  bins = BinnedMetrics(ODOMETRY, 1, 2, "cpu")
  for error in [0, 4, 9]:
    bins.update(torch.tensor([[error, error, error, 2.0, 9.0]]))
  result = bins.result()
  assert result["bin_counts"].tolist() == [2, 1]
  assert result["full"][0, 0] == pytest.approx(np.sqrt(13 / 3))
  assert result["bins"][0, 0, 0] == pytest.approx(np.sqrt(2))
  assert result["full"][0, 4] == 9


def test_sobol_prefix_bounds_and_seed(tmp_path):
  from supermimic.scripts.check_estimator import stationary_motion

  path = tmp_path / "motion.npz"
  stationary_motion(path)
  env_cfg, _ = make_env_config(StudyConfig(), path)
  small = domain_table(env_cfg, 16, 2026)
  large = domain_table(env_cfg, 32, 2026)
  assert small["values"] == large["values"][:16]
  values, bounds = np.array(small["values"]), np.array(small["ranges"])
  assert values.shape == (16, 33)
  assert (values >= bounds[:, 0]).all() and (values <= bounds[:, 1]).all()
  assert small != domain_table(env_cfg, 16, 2028)
  assert not env_cfg.terminations and not env_cfg.auto_reset


def test_sensor_pairing_independent_of_context_batch():
  cfg = StudyConfig()
  measurement = GaussianMeasurementConfig()
  a = PairedSensors(cfg, "sha", [3, 7], measurement)
  b = PairedSensors(cfg, "sha", [7], measurement)
  for _ in range(3):
    clean = torch.arange(186, dtype=torch.float32).reshape(2, 93)
    both = a.sample(clean)
    torch.testing.assert_close(both[1:], b.sample(clean[1:]), rtol=0, atol=0)
    assert torch.equal(both[:, 64:], clean[:, 64:])


def test_context_then_clip_weighting_and_no_partial_context_means():
  manifest = dict(
    kind="odometry",
    config=dict(num_domain_contexts=2),
    motions=[dict(name="short"), dict(name="long")],
  )
  loaded = []
  for name, frames, values in [("short", 1, [1.0, 3.0]), ("long", 100, [5.0, 7.0])]:
    loaded.append(
      (
        dict(status="complete", clip=name, variant="C-posterior"),
        dict(
          context_ids=np.array([0, 1]),
          full=np.array(values)[:, None],
          bins=np.tile(np.array(values)[None, :, None], (frames, 1, 1)),
          first_failure_s=np.array([np.nan, 0.2]),
        ),
      )
    )
  _, clips, excluded, _ = aggregate(manifest, loaded)
  assert not excluded
  assert np.mean([c["C-posterior"]["mean"] for c in clips.values()]) == 4
  loaded[0][1]["context_ids"] = np.array([0])
  _, clips, excluded, _ = aggregate(manifest, loaded)
  assert excluded == ["short"] and set(clips) == {"long"}


def test_config_unknown_keys_units_and_defaults(tmp_path):
  path = tmp_path / "test.toml"
  path.write_text(
    "num_domain_contexts = 16\ntime_bin_s = 2.0\nboxplot_interval_s = 2.0\n"
  )
  cfg = load_config(path)
  assert cfg.replay_ids() == [0, 8, 15]
  assert cfg.output.startswith(str(tmp_path))
  path.write_text("typo = 1")
  with pytest.raises(TypeError):
    load_config(path)
  with pytest.raises(ValueError):
    replace(cfg, boxplot_interval_s=3).validate()


def test_world_tracking_retains_global_translation_and_rotation():
  from types import SimpleNamespace

  from supermimic.validation.metrics import tracking_values

  truth = state()
  error = torch.zeros(1, 70)
  error[0, 3] = 2
  truth = boxplus(truth, error)
  ref = SimpleNamespace(
    position=torch.zeros(1, 1, 3),
    rotation=torch.eye(3)[None, None],
    velocity=torch.zeros(1, 1, 3),
  )
  command = SimpleNamespace(
    robot_body_pos_w=torch.tensor([[[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]),
    body_pos_w=torch.zeros(1, 2, 3),
    robot_body_quat_w=torch.tensor([[[2**-0.5, 0.0, 0.0, 2**-0.5]]]).repeat(1, 2, 1),
    body_quat_w=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]).repeat(1, 2, 1),
    joint_pos=torch.zeros(1, 29),
    joint_vel=torch.zeros(1, 29),
  )
  value = tracking_values(truth, ref, command, [1])[0]
  assert value[1] == 4
  assert value[3] == 2
  assert value[4] == 2
  assert value[5] == pytest.approx(90, abs=1e-5)


def test_joint_method_selection_partial_reports_and_stale_figures(tmp_path):
  import json

  from supermimic.validation.config import atomic_json
  from supermimic.validation.metrics import TRACKING
  from supermimic.validation.report import report
  from supermimic.validation.study import load_task, save_task, tasks

  config_path = tmp_path / "config.toml"
  config_path.write_text('[evaluation]\nmethods = ["B-posterior"]\n')
  cfg = replace(
    load_config(config_path), output=str(tmp_path), plot_formats=["png"], plot_dpi=30
  )
  # Even older configs requesting PNG produce the current SVG-only report.
  for methods in ([], ["bad"], ["C-posterior", "C-posterior"]):
    with pytest.raises(ValueError, match="methods"):
      replace(cfg, methods=methods).validate()
  manifest = dict(
    kind="joint",
    version="test",
    identity="identity",
    config=dict(num_domain_contexts=1, num_envs=1, time_bin_s=2),
    motions=[
      dict(name=name, sha256=name, seen_in_training=False) for name in ("a", "b")
    ],
  )
  atomic_json(tmp_path / "manifest.json", manifest)
  work = list(tasks(manifest, ["A-posterior", "B-posterior"]))
  for key, motion, variant, ids in work:
    if variant == "B-posterior" and motion["name"] == "b":
      continue
    save_task(
      tmp_path,
      key,
      dict(
        context_ids=ids,
        first_failure_s=[np.nan],
        full=np.ones((1, len(TRACKING))),
        bins=np.ones((2, 1, len(TRACKING))),
        odometry_full=np.full((1, 5), 7),
        odometry_bins=np.full((2, 1, 5), 7),
      ),
      dict(
        identity="identity",
        clip=motion["name"],
        status="complete",
        odometry_status="complete",
        variant=variant,
      ),
    )
  for kind in ("tracking", "odometry"):
    folder = tmp_path / kind / "figures"
    folder.mkdir(parents=True)
    obsolete = ["summary_full.svg", "old-tracking-time.png"]
    for name in obsolete:
      (folder / name).touch()
    atomic_json(folder.parent / "report.json", {"figures": obsolete})
  assert not report(cfg)["complete"]
  for kind, expected in [("odometry", 7), ("tracking", 1)]:
    summary = json.loads((tmp_path / kind / "report.json").read_text())
    assert summary["completed_methods"] == ["A-posterior"]
    assert summary["excluded_clips_by_method"] == {"B-posterior": ["b"]}
    assert set(summary["variants"]["A-posterior"].values()) == {expected}
    expected_figures = (
      ["summary_A-posterior_time.svg"]
      if kind == "odometry"
      else ["A-posterior-mpkpe-by-clip.svg", "A-posterior-tracking-time.svg"]
    )
    assert summary["figures"] == expected_figures
    assert (
      sorted(p.name for p in (tmp_path / kind / "figures").iterdir())
      == expected_figures
    )
  # A remains reusable when the TOML selects it again.
  from supermimic.validation.tracking import run_tracking

  cached_cfg = replace(cfg, methods=["A-posterior"])
  run_tracking(
    cached_cfg, manifest, {}, list(tasks(manifest, cached_cfg.methods)), tmp_path
  )
  assert not (tmp_path / "tracking_timing.json").exists()
  # Completing the last selected-method shard adds B alongside cached A.
  key, motion, _, _ = work[-1]
  meta, arrays = load_task(tmp_path, work[1][0], manifest)
  save_task(tmp_path, key, arrays, {**meta, "clip": motion["name"]})
  assert report(replace(cfg, boxplot_interval_s=4))["complete"]
  for kind in ("tracking", "odometry"):
    summary = json.loads((tmp_path / kind / "report.json").read_text())
    assert summary["requested_methods"] == ["B-posterior"]
    assert summary["completed_methods"] == ["A-posterior", "B-posterior"]
    assert set(summary["clips"]["b"]) == {"A-posterior", "B-posterior"}
  (tmp_path / "tasks" / f"{key}.npz").unlink()
  (tmp_path / "tasks" / f"{work[0][0]}.npz").unlink()
  assert not report(cfg)["complete"]
  assert not list((tmp_path / "tracking" / "figures").iterdir())
  assert not list((tmp_path / "odometry" / "figures").iterdir())


def test_resume_uses_original_task_keys_after_changing_methods_and_plot_settings():
  from supermimic.validation.config import digest
  from supermimic.validation.study import resume_manifest, tasks

  manifest = dict(
    kind="joint",
    version="v3",
    motions=[dict(sha256="clip")],
    checkpoints={"C": "estimator"},
    source_sha256={"validation/batched.py": "runtime", "validation/study.py": "old"},
    config=dict(num_domain_contexts=2, num_envs=2, time_bin_s=2, boxplot_interval_s=2),
  )
  manifest["identity"] = digest(manifest)
  new = {
    **manifest,
    "config": {
      **manifest["config"],
      "methods": ["C-posterior"],
      "boxplot_interval_s": 4,
    },
    "source_sha256": {**manifest["source_sha256"], "validation/study.py": "new"},
    "identity": "new",
  }
  resumed = resume_manifest(manifest, new)
  assert list(tasks(resumed)) == list(tasks(manifest))
  for change in (
    {"config": {**new["config"], "time_bin_s": 1}},
    {"source_sha256": {**new["source_sha256"], "validation/batched.py": "changed"}},
    {"checkpoints": {"C": "changed"}},
  ):
    with pytest.raises(ValueError, match="protocol differs"):
      resume_manifest(manifest, {**new, **change})
  with pytest.raises(ValueError, match="checksum"):
    resume_manifest({**manifest, "identity": "bad"}, new)
