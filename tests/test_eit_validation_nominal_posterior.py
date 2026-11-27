from types import SimpleNamespace

import numpy as np
import pytest
import torch

from supermimic.belief import BeliefState
from supermimic.utils.lie_state import LieState, boxplus, so3_exp
from supermimic.validation import nominal_runtime as runtime
from supermimic.validation.nominal import Scenario
from supermimic.validation.nominal_posterior import (
  MODE,
  PosteriorBuffer,
  posterior_sample,
)
from supermimic.validation.nominal_posterior_report import (
  calibration,
  integrate,
  intrinsic_error,
  marginal_values,
  normalized_nees,
)
from tests.test_estimator_belief_filter import on_device


def test_full_projection_matches_boxplus_finite_differences():
  _, prior, *_ = on_device(1, "cpu")
  state = LieState.from_full(prior.mean.to_full().double())
  state.rotation[:] = so3_exp(torch.tensor([[0.52, -0.30, 1.2]], dtype=torch.double))
  generator = torch.Generator().manual_seed(369)
  lower = torch.randn((1, 70, 70), generator=generator, dtype=torch.double).tril()
  buffer = PosteriorBuffer([1])
  sample = posterior_sample(state, BeliefState(state, covariance_sqrt=lower))
  buffer.append(sample)
  buffer.append(sample)
  data = buffer.episode(0, 1)

  def projected(delta):
    perturbed = boxplus(state, delta)
    actual = dict(
      data,
      truth_position=perturbed.position.numpy(),
      truth_rotation=perturbed.rotation.numpy(),
      truth_velocity=perturbed.velocity.numpy(),
      mean_rotation=state.rotation.numpy(),
      mean_height=state.position[:, 2:].numpy(),
      mean_velocity=state.velocity.numpy(),
    )
    actual = {k: v[:1] for k, v in actual.items()}
    intrinsic = intrinsic_error(actual)[0]
    angles = np.deg2rad(marginal_values(actual)[2][0, 1:3])
    return np.r_[intrinsic, angles]

  eps = 1e-6
  jac = np.stack(
    [
      (projected(d[None] * eps) - projected(-d[None] * eps)) / (2 * eps)
      for d in torch.eye(70, dtype=torch.double)
    ],
    axis=1,
  )
  full = jac @ lower[0].numpy()
  covariance = full @ full.T
  np.testing.assert_allclose(
    data["intrinsic_cov_lower"][0], covariance[:6, :6][np.tril_indices(6)], atol=1e-7
  )
  np.testing.assert_allclose(
    data["euler_variance"][0], covariance.diagonal()[6:], atol=1e-7
  )


def test_wrapped_euler_singularity_and_known_orientation_integration():
  _, prior, *_ = on_device(3, "cpu")
  state = LieState.from_full(prior.mean.to_full().double())
  state.rotation[:] = so3_exp(
    torch.deg2rad(
      torch.tensor([[179.0, 0, 0], [0, 89.99999, 0], [0, 0, 90]], dtype=torch.double)
    )
  )
  truth = state.clone()
  truth.rotation[0] = state.rotation[0].T
  sample = posterior_sample(
    truth, BeliefState(state, covariance_sqrt=prior.covariance_sqrt)
  )
  buf = PosteriorBuffer([2])
  for row in sample:
    buf.append(row[None])
  data = buf.episode(0, 2)
  _, _, error, sigma = marginal_values(data)
  np.testing.assert_allclose(error[0, 1], 2.0, atol=1e-5)
  assert np.isnan(error[1, 1]) and error[1, 2] == 0
  assert np.isnan(sigma[1, 1:3]).all()
  calibrated, _ = calibration(error, sigma)
  assert calibrated[1]["accuracy_frames"] == calibrated[1]["valid_frames"] == 2
  assert calibrated[2]["accuracy_frames"] == 3 and calibrated[2]["valid_frames"] == 2
  truth_pole = dict(data, truth_rotation=data["truth_rotation"][[1, 1, 2]])
  _, _, mixed_error, mixed_sigma = marginal_values(truth_pole)
  assert np.isfinite(mixed_error[0, 2]) and np.isfinite(mixed_sigma[0, 2])
  assert (
    np.isfinite(error[1, [0, 3, 4, 5]]).all()
    and np.isfinite(intrinsic_error(data)).all()
  )
  data.update(
    dt=0.5,
    truth_position=np.array([[7, 8, 9], [7, 9, 9], [7, 10, 9]]),
    truth_rotation=np.repeat(state.rotation[-1:].numpy(), 3, axis=0),
    mean_velocity=np.tile([2.0, 0, 0], (3, 1)),
  )
  _, distance, metrics = integrate(data)
  np.testing.assert_allclose(distance, 0, atol=1e-12)
  assert metrics["drift_pct"] == 0
  data["mean_velocity"][-1, 0] += 4
  _, _, metrics = integrate(data)
  np.testing.assert_allclose(
    [metrics["ate_m"], metrics["fde_m"], metrics["drift_pct"]], [1 / np.sqrt(3), 1, 50]
  )
  data["truth_position"] = data["truth_position"][:1].repeat(3, axis=0)
  assert integrate(data)[2]["drift_pct"] is None


def test_gaussian_calibration_and_normalized_nees():
  rng = np.random.default_rng(1701)
  lower = np.tril(rng.normal(size=(6, 6)) * 0.4) + np.eye(6)
  cov = lower @ lower.T
  errors = rng.normal(size=(60000, 6)) @ lower.T
  rows, _ = calibration(errors, np.broadcast_to(np.sqrt(cov.diagonal()), errors.shape))
  for row in rows:
    assert abs(row["coverage_1sigma_pct"] - 68.27) < 0.7
    assert abs(row["coverage_2sigma_pct"] - 95.45) < 0.5
  covs = np.broadcast_to(cov[np.tril_indices(6)], (len(errors), 21))
  assert abs(normalized_nees(errors, covs).mean() - 1) < 0.01


def test_pre_reset_endpoints_and_surviving_actor(monkeypatch):
  # A fails on its first action. B consumes three corrected states and completes.
  # Inject a scalar transition whose value distinguishes the applied action,
  # current measurement and previous belief at every timestamp.
  estimator, prior, *_ = on_device(2, "cpu")
  rt = object.__new__(runtime.NominalRuntime)
  rt.cfg = SimpleNamespace(num_envs=2, evaluation_seed=0)
  rt.device = "cpu"
  rt.motions = {c: dict(sha256=c, steps=3) for c in ("a", "b")}
  rows = [Scenario(c, 5, MODE) for c in rt.motions]
  rt.raw = SimpleNamespace(
    step_dt=0.02, sim=SimpleNamespace(data=SimpleNamespace(qpos=torch.zeros(2, 1)))
  )
  rt.cmd = SimpleNamespace(time_steps=torch.zeros(2, dtype=torch.long))
  rt.motion = SimpleNamespace(last_frame=lambda t: t)
  rt.monitor = SimpleNamespace(
    active_terms=["test"], terminated=torch.zeros(2, dtype=torch.bool)
  )
  rt.monitor.get_term = lambda _: rt.monitor.terminated
  rt.measurement = estimator.measurement_config
  rt.priors = {"belief": None}
  rt.context = torch.zeros(2, 33)
  rt.history_length = 1
  rt.limits = (-10000, 10000)
  rt.terms = []
  rt.check_first_actions = lambda *args: None

  def observations():
    obs = {"actor": torch.zeros(2, 93), "full_state": prior.mean.to_full().clone()}
    obs["actor"][:, 0] = rt.cmd.time_steps.float() + 1
    obs["full_state"][:, 2] = rt.cmd.time_steps.float()
    return obs

  def reset(_):
    rt.cmd.time_steps.zero_()
    return observations(), prior.mean

  rt.reset = reset

  def step(raw, cmd, actions, executing, parked, monitor):
    cmd.time_steps += executing.long()
    monitor.terminated[:] = torch.tensor([True, False])
    return observations()

  monkeypatch.setattr(runtime, "full_clip_step", step)
  monkeypatch.setattr(
    runtime, "noise_block", lambda cfg, unique, n, *_: torch.zeros(n, 2, 64)
  )
  monkeypatch.setattr(runtime, "paired_prior", lambda *args: prior.index(slice(0, 1)))
  monkeypatch.setattr(
    runtime,
    "motion_reference",
    lambda *args: SimpleNamespace(
      y=torch.zeros(2, 68), next_joint=torch.zeros(2, 29), goal=torch.zeros(2, 1)
    ),
  )
  monkeypatch.setattr(runtime, "frame_scores", lambda *args: torch.ones(2, 7))

  def update(key, belief, action, context, sensor, std):
    out = belief.clone()
    out.mean.position[:, 2] = (
      sensor[:, 0]
      if action is None
      else belief.mean.position[:, 2] + action[:, 0] + sensor[:, 0]
    )
    return out, torch.zeros(2, dtype=torch.bool)

  rt.update_filter = update
  rt.actors = {
    "belief": SimpleNamespace(
      encode_goal=lambda x: x,
      decode_goal=lambda goal, history, condition: (condition[:, 3:4].expand(-1, 29),),
    )
  }
  controls = {}

  def audit(s, a, noise):
    controls[s.clip] = a[:, 0].tolist()

  tracks = {}

  def persist(s, status, n, frames, terms):
    tracks[s.clip] = (status, len(frames))

  rt.run(rows, persist, audit=audit)
  reference = controls.copy()
  captured = {}

  def posterior(s, status, frames, dt, fields):
    captured[s.clip] = fields

  rt.run(rows, persist, posterior, audit)
  assert controls == reference == {"a": [1.0], "b": [1.0, 4.0, 11.0]}
  assert tracks == {"a": ("failure", 1), "b": ("success", 3)}
  for clip, values in [("a", [1.0, 4.0]), ("b", [1.0, 4.0, 11.0, 26.0])]:
    np.testing.assert_array_equal(captured[clip]["mean_height"].ravel(), values)
    np.testing.assert_array_equal(
      captured[clip]["truth_position"][:, 2], np.arange(len(values))
    )

  # Repeat the switch with the real square-root EKF and compare each posterior
  # used by the surviving actor, including the tick following its peer's failure.
  rt.use_graph = False
  rt.filters = {"belief": estimator}
  histories = []

  def trace_filter(*args):
    result, bad = runtime.NominalRuntime.update_filter(rt, *args)
    histories.append(
      (result.mean.to_full()[1].clone(), result.covariance_sqrt[1].clone())
    )
    return result, bad

  rt.update_filter = trace_filter
  rt.run(rows, persist, audit=audit)
  off_history = histories.copy()
  off_actions = controls["b"].copy()
  histories.clear()
  rt.run(rows, persist, posterior, audit)
  assert len(off_history) == 3 and len(histories) == 4
  for off, on in zip(off_history, histories[:-1], strict=True):
    for a, b in zip(off, on, strict=True):
      torch.testing.assert_close(a, b, rtol=0, atol=0)
  assert off_actions == controls["b"]


def test_repeat_pooling_and_posterior_resume(tmp_path):
  import csv
  import json

  from supermimic.scripts.eval_eit_nominal import episode_ready
  from supermimic.validation.config import atomic_json, digest
  from supermimic.validation.nominal import (
    FRAME_COLUMNS,
    atomic_npz,
    save_episode,
    scenario_groups,
    tracking_digest,
  )
  from supermimic.validation.nominal_posterior import load_posterior, save_posterior
  from supermimic.validation.nominal_report import merged_report

  manifest = dict(
    motions=[dict(name="clip.npz", steps=3)],
    seeds=[7],
    bootstrap_seed=3,
    bootstrap_samples=8,
    checkpoints={"belief": {"sha256": "checkpoint"}},
  )
  manifest["identity"] = digest(manifest)
  first, repeat = tmp_path / "first", tmp_path / "repeat"
  group = next(scenario_groups(manifest))
  for i, run in enumerate((first, repeat)):
    atomic_json(run / "manifest.json", manifest)
    atomic_json(
      run / "timing.json", dict(layout="mixed", capacity=5, allocated_slots=5)
    )
    for s in group:
      frames = np.full((1 + 2 * i, len(FRAME_COLUMNS)), i, dtype=np.float32)
      status = ("failure", "success")[i]
      save_episode(s.path(run), manifest["identity"], status, 3, frames, [])
  merged_report(first, repeat)
  with (repeat / "merged-report" / "tracking.csv").open() as f:
    rows = {r["metric"]: r for r in csv.DictReader(f) if r["mode"] == MODE}
  assert float(rows["sr_pct"]["pooled"]) == 50
  assert float(rows["mpjpe_cm_mean"]["pooled"]) == 0.75
  assert float(rows["mpjpe_cm_p95"]["pooled"]) == 1
  assert (
    json.loads((repeat / "merged-report" / "report.json").read_text())[
      "tracking_records"
    ]
    == 10
  )
  assert not episode_ready(repeat, manifest, s, 3, True, False)
  _, prior, *_ = on_device(1, "cpu")
  buffer = PosteriorBuffer([3])
  sample = posterior_sample(prior.mean, prior)
  for _ in range(4):
    buffer.append(sample)
  save_posterior(repeat, manifest, s, status, frames, 0.02, buffer.episode(0, 3))
  assert episode_ready(repeat, manifest, s, 3, True, False)
  from supermimic.validation.nominal_posterior_report import posterior_report

  posterior_report(repeat)
  audit = json.loads((repeat / "posterior-report" / "validation.json").read_text())
  assert (audit["control_frames"], audit["state_samples"], audit["invalid"]) == (
    3,
    4,
    [],
  )
  assert (repeat / "merged-report" / "tracking.png").is_file()
  assert (repeat / "posterior-report" / "velocity-integration.png").is_file()
  fields = buffer.episode(0, 3)
  for bad_rotation in (np.diag([1.0, 1.0, -1.0]), np.diag([1.0, 1.0, 2.0])):
    fields["truth_rotation"][1] = bad_rotation
    save_posterior(repeat, manifest, s, status, frames, 0.02, fields)
    with pytest.raises(ValueError, match="proper rotation"):
      load_posterior(repeat, manifest, s, status, frames)
    assert not episode_ready(repeat, manifest, s, 3, True, False)
  save_posterior(repeat, manifest, s, status, frames, 0.02, buffer.episode(0, 3))
  trace = s.path(repeat / "controls")
  atomic_npz(
    trace,
    actions=np.zeros((3, 29)),
    noise=np.zeros((3, 64)),
    tracking_sha256=tracking_digest(frames),
  )
  assert episode_ready(repeat, manifest, s, 3, False, True)
  save_episode(s.path(repeat), manifest["identity"], status, 3, frames * 0.5, [])
  assert not episode_ready(repeat, manifest, s, 3, False, True)
  trace.write_bytes(b"interrupted")
  assert not episode_ready(repeat, manifest, s, 3, False, True)
  assert not episode_ready(repeat, manifest, s, 3, True, False)


def test_missing_posterior_replays_original_whole_batch(tmp_path, monkeypatch):
  from supermimic.scripts.eval_eit_nominal import run
  from supermimic.validation.nominal import FRAME_COLUMNS, scenario_groups
  from supermimic.validation.nominal_posterior import posterior_path

  manifest = dict(
    identity="identity",
    motions=[dict(name=c, steps=1) for c in ("a", "b")],
    seeds=[1],
    checkpoints={"belief": {"sha256": "sha"}},
  )
  groups = list(scenario_groups(manifest))
  _, prior, *_ = on_device(1, "cpu")
  buf = PosteriorBuffer([1])
  for _ in range(2):
    buf.append(posterior_sample(prior.mean, prior))
  states = buf.episode(0, 1)
  replayed = []
  crash = False

  class Runtime:
    def __init__(self, paths, manifest, saved, device, slots, cohort, compile_filter):
      assert slots == 5 and cohort == 5
      self.raw = SimpleNamespace(cfg=SimpleNamespace(sim=SimpleNamespace(nconmax=128)))
      self.warmup_s = 0

    def run(self, batch, persist, posterior, audit):
      replayed.append(batch)
      for s in batch:
        frames = np.zeros((1, len(FRAME_COLUMNS)), dtype=np.float32)
        persist(s, "success", 1, frames, [])
        if s.mode == MODE:
          posterior(s, "success", frames, 0.02, states)
      if crash:
        raise RuntimeError("interrupted repair")
      return len(batch), [], 0.0

    def close(self):
      pass

  monkeypatch.setattr(runtime, "NominalRuntime", Runtime)
  args = SimpleNamespace(
    layout="mixed", record_belief_posterior=True, audit_controls=False, no_compile=True
  )

  def evaluate():
    run(args, [], manifest, {}, "cpu", 5, tmp_path)

  evaluate()
  assert replayed == groups
  replayed.clear()
  posterior_path(tmp_path, next(s for s in groups[1] if s.mode == MODE)).unlink()
  crash = True
  with pytest.raises(RuntimeError, match="interrupted repair"):
    evaluate()
  assert replayed == groups[1:]
  from supermimic.validation.nominal_report import load_manifest

  with pytest.raises(ValueError, match="Incomplete nominal batch"):
    load_manifest(tmp_path)
  crash = False
  replayed.clear()
  evaluate()
  assert replayed == groups[1:]
  replayed.clear()
  evaluate()
  assert replayed == []
