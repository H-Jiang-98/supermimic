import numpy as np
import pytest
import torch

from supermimic.belief import GaussianStatePrior
from supermimic.utils.lie_state import local_error
from supermimic.validation.batched import noise_block
from supermimic.validation.config import StudyConfig, atomic_json, digest
from supermimic.validation.nominal import ROUTES
from supermimic.validation.nominal_report import tracking_sources
from supermimic.validation.nominal_runtime import evaluation_noise
from supermimic.validation.runtime import paired_prior
from supermimic.validation.sweep_protocol import condition_manifest
from supermimic.validation.sweep_report import elapsed_errors
from tests.test_estimator_belief_filter import on_device


def test_uncertainty_scaling_preserves_paired_draws():
  estimator, belief, *_ = on_device(2, "cpu")
  cfg, prior = StudyConfig(), GaussianStatePrior()
  measurement = estimator.measurement_config
  truth = belief.mean.to(torch.float64)
  original = paired_prior(cfg, "clip", [7, 8], truth, prior)
  targets = [("clip", 7), ("clip", 8)]
  noise = noise_block(cfg, targets, 12, measurement, "cpu")
  for kb, ko in ((1, 1), (2, 1), (4, 1), (1, 2), (1, 4)):
    p, m = evaluation_noise(prior, measurement, kb, ko)
    estimate = paired_prior(cfg, "clip", [7, 8], truth, p)
    torch.testing.assert_close(
      local_error(truth, estimate.mean), kb * local_error(truth, original.mean)
    )
    torch.testing.assert_close(estimate.covariance_sqrt, kb * original.covariance_sqrt)
    torch.testing.assert_close(m.covariance(29), ko**2 * measurement.covariance(29))
    torch.testing.assert_close(
      noise_block(cfg, targets, 12, m, "cpu"), ko * noise, rtol=0, atol=0
    )


def test_manifest_routes_are_paired_and_pilot_is_separate():
  base = dict(
    identity="baseline",
    routes=ROUTES,
    motions=[
      dict(name=n, steps=101)
      for n in ("walk1_subject1.npz", "fallAndGetUp1_subject1.npz")
    ],
    seeds=[7, 8],
  )
  initial = condition_manifest(base, "prior-2")
  sensor = condition_manifest(base, "sensor-2")
  pilot = condition_manifest(base, "sensor-2", True)
  assert len({m["identity"] for m in (initial, sensor, pilot)}) == 3
  assert set(initial["routes"]) == {"GT-Estimated", "Mean-in-Loop", "Belief-in-Loop"}
  assert set(sensor["routes"]) == set(ROUTES) and initial["seeds"] == sensor["seeds"]
  assert pilot["seeds"] == [7] and all(m["steps"] == 32 for m in pilot["motions"])
  assert all(m["steps"] == 101 for m in base["motions"])


def test_split_sources_require_the_declared_checkpoint_and_identity(tmp_path):
  route = {"Belief-in-Loop": ["belief", "belief"]}
  source = dict(
    protocol="v1",
    motions=[],
    seeds=[],
    columns=[],
    friction=0.75,
    encoder_bias=[],
    com_offset=[],
    routes=route,
    checkpoints={"belief": {"sha256": "test-hash"}},
  )
  source["identity"] = digest(source)
  path = tmp_path / "source"
  atomic_json(path / "manifest.json", source)
  base = {
    **source,
    "source_runs": {
      "loop": {"path": str(path), "identity": source["identity"], "modes": list(route)}
    },
  }
  assert tracking_sources(tmp_path, base)["Belief-in-Loop"] == (
    path,
    source["identity"],
  )
  base["checkpoints"] = {"belief": {"sha256": "other"}}
  with pytest.raises(ValueError, match="checkpoint differs"):
    tracking_sources(tmp_path, base)


def test_time_bins_keep_failure_endpoint_and_planned_support():
  def trace(values):
    return ({}, np.repeat(np.array(values)[:, None], 7, axis=1))

  groups = {"short": [trace([2.0, 4.0, 6.0])], "early_failure": [trace([10.0])]}
  rows = elapsed_errors(groups, {"short": 3, "early_failure": 6}, dt=0.5)
  assert [r["contributing_episodes"] for r in rows] == [2, 1, 0]
  assert [r["planned_episodes"] for r in rows] == [2, 2, 1]
  assert [r["scored_frames"] for r in rows] == [3, 1, 0]
  assert [r["mpjpe_cm"] for r in rows] == [16 / 3, 6, None]


def test_paired_report_reuses_prior_baselines_and_pairs_levels(tmp_path, monkeypatch):
  import csv

  from supermimic.validation import nominal_posterior_report, sweep_report
  from supermimic.validation.nominal import save_episode, scenario_groups
  from supermimic.validation.sweep_protocol import CONDITIONS

  base = dict(
    motions=[dict(name="a", steps=2), dict(name="b", steps=1)],
    seeds=[17],
    routes=ROUTES,
    bootstrap_seed=42,
    bootstrap_samples=20,
  )
  base["identity"] = digest(base)
  nominal_dir, output = tmp_path / "baseline", tmp_path / "sweep"
  conditions = [(nominal_dir, base)] + [
    (output / name, condition_manifest(base, name)) for name in CONDITIONS
  ]
  for folder, m in conditions:
    atomic_json(folder / "manifest.json", m)
    atomic_json(folder / "timing.json", dict(dt=0.02))
    for group in scenario_groups(m):
      for scenario in group:
        n = 2 if scenario.clip == "a" else 1
        failed = scenario.clip == "a" and scenario.mode == "Belief-in-Loop"
        frames = np.ones((1 if failed else n, 7), dtype=np.float32)
        frames[:, 0] *= m.get("kappa_o", 1) * m.get("kappa_b", 1)
        save_episode(
          scenario.path(folder),
          m["identity"],
          "failure" if failed else "success",
          n,
          frames,
          ["anchor_ori"] if failed else [],
        )
  posterior_runs = []
  monkeypatch.setattr(
    nominal_posterior_report, "posterior_report", posterior_runs.append
  )
  monkeypatch.setattr(sweep_report, "plot_elapsed", lambda *args: None)
  monkeypatch.setattr(sweep_report, "plot_robustness", lambda *args: None)
  sweep_report.build_report(nominal_dir, base, output, include_sweep=True)
  assert posterior_runs == [output / name for name in CONDITIONS]
  with (output / "report" / "uncertainty-sweep.csv").open() as f:
    rows = list(csv.DictReader(f))
  result = {
    (r["condition"], r["mode"]): r for r in rows if r["metric"] == "mpjpe_cm_p95"
  }
  assert len(result) == 25
  for condition in CONDITIONS:
    for mode, (_, estimator) in ROUTES.items():
      r = result[condition, mode]
      expected = (
        0
        if condition.startswith("prior") and estimator is None
        else int(condition[-1]) - 1
      )
      for key in ("delta_from_nominal", "delta_ci95_low", "delta_ci95_high"):
        assert float(r[key]) == expected
