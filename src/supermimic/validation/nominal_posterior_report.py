"""Offline same-trajectory marginal calibration, intrinsic NEES and velocity integration."""

from __future__ import annotations

from math import erf, sqrt
from pathlib import Path

import numpy as np

from supermimic.validation.config import atomic_json
from supermimic.validation.nominal import read_episode, scenario_groups
from supermimic.validation.nominal_posterior import (
  MODE,
  SINGULAR_COS,
  intrinsic_cholesky,
  load_posterior,
  posterior_path,
)
from supermimic.validation.nominal_report import episode_values, write_csv

COMPONENTS = ("Height", "Roll", "Pitch", "Body vx", "Body vy", "Body vz")
UNITS = ("cm", "deg", "deg", "m/s", "m/s", "m/s")
SCALE = np.array([100, 180 / np.pi, 180 / np.pi, 1, 1, 1])
EXAMPLES = {"walk1_subject1.npz", "fallAndGetUp1_subject1.npz"}


def marginal_values(data):
  def angles(rotation):
    a, b, c = rotation[:, 2, :].astype(float).T
    h = np.hypot(b, c)
    result = np.stack((np.arctan2(b, c), np.arctan2(-a, h)), -1)
    # Pitch remains a defined value at a pole; roll has no unique value.
    result[h < SINGULAR_COS, 0] = np.nan
    return result

  actual = np.column_stack(
    (
      data["truth_position"][:, 2],
      angles(data["truth_rotation"]),
      data["truth_velocity"],
    )
  )
  mean = np.column_stack(
    (data["mean_height"], angles(data["mean_rotation"]), data["mean_velocity"])
  )
  lower = data["intrinsic_cov_lower"]
  variance = np.column_stack(
    (lower[:, 0], data["euler_variance"], lower[:, [9, 14, 20]])
  )
  if np.any(variance < 0):
    raise ValueError("Negative posterior marginal variance")
  error = actual - mean
  error[:, 1:3] = np.arctan2(np.sin(error[:, 1:3]), np.cos(error[:, 1:3]))
  return actual * SCALE, mean * SCALE, error * SCALE, np.sqrt(variance) * SCALE


def intrinsic_error(data):
  mean = -data["mean_rotation"][:, 2, :].astype(float)
  truth = -data["truth_rotation"][:, 2, :].astype(float)
  axis = np.eye(3)[np.abs(mean).argmin(axis=1)]
  u = np.cross(mean, axis)
  u /= np.linalg.norm(u, axis=1, keepdims=True)
  v = np.cross(mean, u)
  # Coordinates of the S² logarithm in the same basis used for JΣJᵀ.
  tilt = np.column_stack(((truth * u).sum(1), (truth * v).sum(1)))
  sine = np.linalg.norm(tilt, axis=1)
  cosine = (truth * mean).sum(1)
  if np.any((sine < 1e-12) & (cosine < 0)):
    raise ValueError("Antipodal gravity directions have no unique logarithm")
  tilt *= np.divide(
    np.arctan2(sine, cosine), sine, out=np.ones_like(sine), where=sine > 0
  )[:, None]
  return np.column_stack(
    (
      data["truth_position"][:, 2:3].astype(float) - data["mean_height"],
      tilt,
      data["truth_velocity"].astype(float) - data["mean_velocity"],
    )
  )


def normalized_nees(error, packed):
  white = np.linalg.solve(intrinsic_cholesky(packed), error[..., None])
  return np.square(white).sum((1, 2)) / 6


def integrate(data):
  body = data["mean_velocity"].astype(float)
  velocity = (data["truth_rotation"] @ body[..., None])[..., 0]
  position = data["truth_position"].astype(float)
  increments = (velocity[1:] + velocity[:-1]) * (float(data["dt"]) / 2)
  estimated = position[0] + np.vstack((np.zeros(3), np.cumsum(increments, axis=0)))
  distance = np.linalg.norm(estimated - position, axis=1)
  length = np.linalg.norm(np.diff(position, axis=0), axis=1).sum()
  return (
    estimated,
    distance,
    dict(
      path_m=float(length),
      ate_m=float(np.sqrt(np.mean(distance**2))),
      fde_m=float(distance[-1]),
      drift_pct=100 * distance[-1] / length if length > 1 else None,
    ),
  )


def calibration(errors, sigmas):
  rows, curves = [], []
  grid = np.linspace(0, 4, 81)
  for j, (component, unit) in enumerate(zip(COMPONENTS, UNITS, strict=True)):
    accuracy = errors[np.isfinite(errors[:, j]), j]
    valid = np.isfinite(errors[:, j]) & np.isfinite(sigmas[:, j])
    e, s = errors[valid, j], sigmas[valid, j]
    # Handle a degenerate marginal by the original |error| <= k*sigma inequality.
    ratio = np.divide(abs(e), s, where=s > 0, out=np.where(e == 0, 0.0, np.inf))
    ratio.sort()

    def coverage(k, ordered=ratio):
      return (
        np.searchsorted(ordered, k, side="right") * 100 / len(ordered)
        if len(ordered)
        else None
      )

    rows.append(
      dict(
        component=component,
        unit=unit,
        valid_frames=len(e),
        excluded_frames=len(valid) - len(e),
        accuracy_frames=len(accuracy),
        bias=accuracy.mean() if len(accuracy) else None,
        rmse=np.sqrt(np.mean(accuracy**2)) if len(accuracy) else None,
        abs_p95=np.quantile(abs(accuracy), 0.95) if len(accuracy) else None,
        mean_sigma=s.mean() if len(e) else None,
        coverage_1sigma_pct=coverage(1),
        coverage_2sigma_pct=coverage(2),
      )
    )
    curves.extend(
      dict(
        component=component,
        k=k,
        empirical_pct=coverage(k),
        gaussian_pct=100 * erf(k / sqrt(2)),
      )
      for k in grid
    )
  return rows, curves


def posterior_report(output):
  from supermimic.validation.nominal_plots import (
    plot_coverage,
    plot_example,
    plot_integration,
  )
  from supermimic.validation.nominal_report import load_manifest

  output = Path(output)
  manifest = load_manifest(output)
  folder = output / "posterior-report"
  folder.mkdir(exist_ok=True)
  plans = {m["name"]: m["steps"] for m in manifest["motions"]}
  expected = [s for group in scenario_groups(manifest) for s in group if s.mode == MODE]
  errors, sigmas, episodes, invalid = [], [], [], []
  expected_paths = {posterior_path(output, s) for s in expected}
  actual_paths = {p for p in (output / "posteriors").rglob("*.npz") if p.is_file()}
  for path in sorted(actual_paths - expected_paths):
    invalid.append(dict(path=str(path), reason="unexpected posterior file"))
  total_nees, total_frames = 0.0, 0
  examples = []
  for s in expected:
    try:
      track = read_episode(s.path(output), manifest["identity"], plans[s.clip])
      if track is None:
        raise ValueError("missing or invalid same-run tracking")
      status, frames, _ = track
      data = load_posterior(output, manifest, s, status, frames)
      actual, mean, error, sigma = marginal_values(data)
      controls = {k: v[:-1] for k, v in data.items() if v.ndim}
      nees = normalized_nees(intrinsic_error(controls), controls["intrinsic_cov_lower"])
      estimated, distance, odometry = integrate(data)
    except (ValueError, np.linalg.LinAlgError) as e:
      invalid.append(dict(clip=s.clip, seed=s.seed, reason=str(e)))
      continue
    total_nees += nees.sum()
    total_frames += len(nees)
    errors.append(error[:-1])
    sigmas.append(sigma[:-1])
    episodes.append(
      dict(
        clip=s.clip,
        seed=s.seed,
        mode=s.mode,
        status=status,
        planned=plans[s.clip],
        state_samples=len(error),
        dt_s=float(data["dt"]),
        **episode_values(frames, plans[s.clip], status),
        normalized_nees=nees.mean(),
        **odometry,
      )
    )
    if s.clip in EXAMPLES and s.seed == 2718:
      examples.append((s, status, data, actual, mean, sigma, estimated, distance))
  audit = dict(
    expected_episodes=len(expected),
    valid_episodes=len(episodes),
    invalid=invalid,
    control_frames=total_frames,
    state_samples=sum(e["state_samples"] for e in episodes),
    normalized_mean_nees=total_nees / total_frames if total_frames else None,
    nees_target=1.0,
    gaussian_1sigma_pct=68.27,
    gaussian_2sigma_pct=95.45,
    kappa_b=manifest.get("kappa_b", 1.0),
    kappa_o=manifest.get("kappa_o", 1.0),
    tracking_run=str(output.resolve()),
  )
  atomic_json(folder / "validation.json", audit)
  if invalid:
    raise ValueError(
      f"{len(invalid)} invalid posterior pairs; see {folder / 'validation.json'}"
    )
  rows, curves = calibration(np.concatenate(errors), np.concatenate(sigmas))
  write_csv(folder / "calibration.csv", rows)
  write_csv(folder / "coverage.csv", curves)
  write_csv(folder / "episodes.csv", episodes)
  groups = []
  for status in ("success", "failure"):
    subset = [e for e in episodes if e["status"] == status]
    drift = [e["drift_pct"] for e in subset if e["drift_pct"] is not None]
    groups.append(
      dict(
        status=status,
        episodes=len(subset),
        total_episodes=len(episodes),
        ate_m=np.mean([e["ate_m"] for e in subset]) if subset else None,
        fde_m=np.mean([e["fde_m"] for e in subset]) if subset else None,
        drift_pct=np.mean(drift) if drift else None,
        drift_episodes=len(drift),
      )
    )
  write_csv(folder / "velocity-integration.csv", groups)
  for example in examples:
    plot_example(folder, *example)
  plot_coverage(folder, rows, curves, audit["normalized_mean_nees"])
  plot_integration(folder, episodes)
  print(
    f"{len(episodes)} paired Belief episodes, {total_frames} control frames: {folder}",
    flush=True,
  )
