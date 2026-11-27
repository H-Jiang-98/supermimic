from pathlib import Path

import numpy as np
import pytest
import torch

from supermimic.utils.lie_state import so3_exp
from supermimic.validation.config import atomic_json, digest
from supermimic.validation.nominal import (
  FRAME_COLUMNS,
  ROUTES,
  batches,
  read_episode,
  save_episode,
  scenario_groups,
)
from supermimic.validation.nominal_metrics import tracking_errors
from supermimic.validation.nominal_report import (
  episode_values,
  pooled_p95,
  report,
  weighted_tail,
)
from supermimic.validation.nominal_runtime import NominalRuntime
from tests.test_estimator_belief_filter import DEVICES, on_device


def test_heading_frame_and_world_position():
  ref_pos = torch.tensor([[[1.0, 2.0, 3.0], [1.0, 4.0, 3.0], [0.0, 3.0, 4.0]]])
  eye = torch.eye(3)[None]
  zero = torch.zeros(1, 3)
  assert (
    torch.count_nonzero(tracking_errors(ref_pos, ref_pos, eye, eye, zero, zero)) == 0
  )
  yaw = so3_exp(torch.tensor([[0.0, 0.0, 1.3]]))
  moved = (
    (ref_pos - ref_pos[:, :1]) @ yaw.mT + ref_pos[:, :1] + torch.tensor([2.0, 0.0, 0.0])
  )
  velocity = torch.tensor([[0.4, 0.3, 0.0]])
  errors = tracking_errors(
    moved, ref_pos, yaw, eye, (yaw @ velocity[..., None]).squeeze(-1), velocity
  )
  torch.testing.assert_close(
    errors[:, :5], torch.zeros_like(errors[:, :5]), atol=1e-5, rtol=0
  )
  assert errors[0, -1] == 200
  roll = so3_exp(torch.tensor([[np.deg2rad(179.0), 0.0, 0.0]], dtype=torch.float64))
  angles = tracking_errors(
    ref_pos.double(), ref_pos.double(), roll, roll.mT, zero.double(), zero.double()
  )
  assert abs(angles[0, 2] - 2) < 1e-6


def test_failure_endpoint_and_fractional_tail():
  frames = np.ones((10, len(FRAME_COLUMNS)), dtype=np.float32)
  assert episode_values(frames[:1], 10, "failure") == dict(
    executed=1, sr_pct=0.0, cr_pct=10.0, cost=0.9
  )
  assert episode_values(frames, 10, "failure")["sr_pct"] == 0
  assert episode_values(frames, 10, "failure")["cr_pct"] == 100
  assert episode_values(frames, 10, "success")["cost"] == 0
  costs = np.arange(15) / 14
  np.testing.assert_allclose(
    weighted_tail(costs, np.ones((1, 15))), (1 + 0.5 * 13 / 14) / 1.5
  )
  x = [np.array([0.0, 2.0, 9.0]), np.array([0.0, 3.0])]
  w = np.array([[1, 1], [0, 2], [2, 0]])
  np.testing.assert_allclose(
    pooled_p95(x, w),
    [
      np.quantile(
        np.concatenate([v for v, n in zip(x, row, strict=True) for _ in range(n)]), 0.95
      )
      for row in w
    ],
  )


@pytest.mark.parametrize("device", DEVICES)
def test_inference_filter_uses_correct_local_derivative_without_rollout_graph(device):
  estimator, prior, action, ctx = on_device(2, device)
  estimator.dynamics.requires_grad_(False).eval()
  controller = object.__new__(NominalRuntime)
  controller.filters = {"mean": estimator}
  controller.device = device
  controller.use_graph = device == "cuda"
  controller.predictors = {}
  controller.warmup_s = 0
  sensor = estimator.observation(prior.mean)
  std = estimator.measurement_config.standard_deviations(
    29, device=action.device, dtype=action.dtype
  )
  with torch.no_grad():
    reference = estimator.predict_correct_kernel(
      prior.mean, prior.covariance_sqrt, action, ctx, sensor, measurement_std=std
    )
  with torch.inference_mode():
    result, bad = controller.update_filter(
      "mean", prior.clone(), action.clone(), ctx.clone(), sensor.clone(), std.clone()
    )
    assert not bad.any()
    torch.testing.assert_close(result.covariance_sqrt, reference[1])
    torch.testing.assert_close(result.mean.to_full(), reference[0].to_full())
    assert (
      result.mean.position.grad_fn is None and result.covariance_sqrt.grad_fn is None
    )
    initial = result.mean.to_full().clone()
    # Replay the captured graph at a different observation-noise level.
    scaled, _ = controller.update_filter("mean", prior, action, ctx, sensor, std * 4)
    with torch.inference_mode(False), torch.no_grad():
      expected = estimator.predict_correct_kernel(
        prior.mean, prior.covariance_sqrt, action, ctx, sensor, measurement_std=std * 4
      )
    torch.testing.assert_close(scaled.covariance_sqrt, expected[1])
    assert not torch.allclose(reference[1], expected[1])
    action2 = action.clone()
    action2[0] += 3.0
    changed, _ = controller.update_filter("mean", prior, action2, ctx, sensor, std)
    torch.testing.assert_close(changed.mean.to_full()[1], initial[1])
    assert not torch.equal(changed.mean.to_full()[0], initial[0])


def test_resume_and_paired_report(tmp_path: Path):
  manifest = dict(
    motions=[dict(name="a", steps=3), dict(name="b", steps=5)],
    seeds=[1, 2],
    bootstrap_seed=2,
    bootstrap_samples=40,
  )
  manifest["identity"] = digest(manifest)
  atomic_json(tmp_path / "manifest.json", manifest)
  groups = list(scenario_groups(manifest))
  expected = {s for group in groups for s in group}
  for layout in ("mixed", "sequential"):
    chunks = list(batches(groups, 7, layout))
    assert max(map(len, chunks)) <= 7
    assert {s for batch in chunks for s in batch} == expected
  for s in expected:
    planned = 3 if s.clip == "a" else 5
    frames = np.full((planned - 1, len(FRAME_COLUMNS)), 0.5, dtype=np.float32)
    save_episode(
      s.path(tmp_path), manifest["identity"], "failure", planned, frames, ["anchor_ori"]
    )
    assert read_episode(s.path(tmp_path), "wrong", planned) is None
  report(tmp_path)
  import json

  result = json.loads((tmp_path / "report.json").read_text())
  assert result["episodes"] == 20 and list(result["modes"]) == list(ROUTES)
  for diff in result["paired_differences"].values():
    assert all(s["ci95"] == [0.0, 0.0] for s in diff.values())
  sample = next(iter(expected))
  sample.path(tmp_path).write_bytes(b"truncated")
  assert read_episode(sample.path(tmp_path), manifest["identity"], 3) is None
