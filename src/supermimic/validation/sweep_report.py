"""Paper uncertainty curves and elapsed-time tracking errors from executed actions."""

from __future__ import annotations

import json
from typing import NamedTuple

import numpy as np

from supermimic.validation.config import atomic_json
from supermimic.validation.nominal import (
  FRAME_COLUMNS,
  ROUTES,
)
from supermimic.validation.nominal_plots import axes_style, figure, finish
from supermimic.validation.nominal_report import (
  bootstrap_weights,
  estimates,
  interval,
  load_manifest,
  load_tracking,
  tracking_sources,
  write_csv,
)
from supermimic.validation.sweep_protocol import CONDITIONS, condition_manifest


class ModeResult(NamedTuple):
  estimates: dict
  time_bins: list
  episodes: list


ERROR_LABELS = (
  "Pelvis-relative MPJPE (cm)",
  "Height error (cm)",
  "Roll error (°)",
  "Pitch error (°)",
  "Body velocity error (m/s)",
  "Global root position error (cm)",
)
PAPER_METRICS = {
  "sr_pct": "Success (%)",
  "cr_pct": "Completion (%)",
  "mpjpe_cm_p95": "P95 MPJPE (cm)",
  "cost_cvar90": "Cost CVaR₀.₉",
}
COLORS = dict(
  zip(ROUTES, ("#72777e", "#7949ab", "#167897", "#d66b1d", "#136746"), strict=True)
)
MARKERS = dict(zip(ROUTES, ("s", "D", "^", "o", "P"), strict=True))
CONTRASTS = (
  ("GT-Oracle", "Prop.+KP"),
  ("GT-Estimated", "GT-Oracle"),
  ("Mean-in-Loop", "GT-Estimated"),
  ("Belief-in-Loop", "Mean-in-Loop"),
)


def elapsed_errors(episodes, plans, dt, bin_seconds=1.0):
  """Bin action endpoints (a,b]; every executed frame has equal weight."""
  width = round(bin_seconds / dt)
  if width < 1 or not np.isclose(width * dt, bin_seconds):
    raise ValueError("Time bin must contain a whole number of control steps")
  bins = (max(plans.values()) + width - 1) // width
  sums = np.zeros((bins, len(ERROR_LABELS)))
  nframes = np.zeros(bins, dtype=int)
  contributors = np.zeros_like(nframes)
  planned = np.zeros_like(nframes)
  for clip, rows in episodes.items():
    planned[: (plans[clip] + width - 1) // width] += len(rows)
    for _, frames in rows:
      starts = np.arange(0, len(frames), width)
      n = len(starts)
      sums[:n] += np.add.reduceat(frames[:, :-1], starts, axis=0, dtype=np.float64)
      nframes[:n] += np.minimum(width, len(frames) - starts)
      contributors[:n] += 1
  return [
    dict(
      start_s=i * bin_seconds,
      end_s=(i + 1) * bin_seconds,
      contributing_episodes=int(contributors[i]),
      planned_episodes=int(planned[i]),
      scored_frames=int(nframes[i]),
      **{
        k: float(sums[i, j] / nframes[i]) if nframes[i] else None
        for j, k in enumerate(FRAME_COLUMNS[:-1])
      },
    )
    for i in range(bins)
  ]


def plot_elapsed(folder, title, rows):
  fig = figure((13.4, 9.0))
  grid = fig.add_gridspec(3, 3, height_ratios=(1, 1, 0.68))
  errors = [fig.add_subplot(grid[i // 3, i % 3]) for i in range(6)]
  coverage = fig.add_subplot(grid[2, :])
  series_by_mode = {mode: [r for r in rows if r["mode"] == mode] for mode in ROUTES}
  schedule = series_by_mode["Prop.+KP"]
  x = [(r["start_s"] + r["end_s"]) / 2 for r in schedule]
  for mode, series in series_by_mode.items():
    for ax, col in zip(errors, FRAME_COLUMNS[:-1], strict=True):
      ax.plot(x, [r[col] for r in series], lw=1.15, color=COLORS[mode], label=mode)
    coverage.plot(
      x, [r["contributing_episodes"] for r in series], lw=1.35, color=COLORS[mode]
    )
  coverage.plot(
    x,
    [r["planned_episodes"] for r in schedule],
    color="#333333",
    ls=":",
    lw=1.6,
    label="Scheduled episodes",
  )
  for ax, label in zip(errors, ERROR_LABELS, strict=True):
    ax.set(ylabel=label, ylim=(0, None))
  for ax in (*errors, coverage):
    axes_style(ax, "y")
    ax.set_xlim(0, schedule[-1]["end_s"])
  coverage.set(
    xlabel="Elapsed control time (s)", ylabel="Episodes in bin", ylim=(0, None)
  )
  coverage.legend(loc="upper right", frameon=False)
  fig.legend(
    *errors[0].get_legend_handles_labels(),
    loc="outside upper center",
    ncols=5,
    frameon=False,
  )
  errors[1].set_title(
    f"{title}   ·   Mean error over executed frames in 1 s bins", fontsize=12
  )
  finish(fig, folder / "tracking-errors-over-time.png")


def plot_robustness(folder, results):
  fig = figure((12, 10))
  axes = fig.subplots(4, 2, sharex=True, sharey="row")
  for j, axis in enumerate(("prior", "sensor")):
    for ax, (metric, label) in zip(axes[:, j], PAPER_METRICS.items(), strict=True):
      for mode in ROUTES:
        samples = [
          results[name][mode][metric] for name in ("nominal", f"{axis}-2", f"{axis}-4")
        ]
        stats = [interval(s) for s in samples]
        x = [1, 2, 4]
        color = COLORS[mode]
        ax.plot(
          x,
          [s["value"] for s in stats],
          marker=MARKERS[mode],
          ms=5,
          color=color,
          label=mode,
        )
        ax.fill_between(
          x,
          [s["ci95"][0] for s in stats],
          [s["ci95"][1] for s in stats],
          color=color,
          alpha=0.10,
          linewidth=0,
        )
      axes_style(ax, "y")
      if j == 0:
        ax.set_ylabel(label)
      ax.set_xticks([1, 2, 4])
    axes[-1, j].set_xlabel("Standard deviation multiplier")
  axes[0, 0].set_title("Initial belief uncertainty")
  axes[0, 1].set_title("Observation noise")
  fig.legend(
    *axes[0, 0].get_legend_handles_labels(),
    title="50k tracking  ·  Shading shows the 95% clip-bootstrap interval",
    loc="outside upper center",
    ncols=5,
    frameon=False,
  )
  finish(fig, folder / "uncertainty-sweep.png")


def summarize_condition(directory, manifest, folder, *, nominal=None):
  folder.mkdir(parents=True, exist_ok=True)
  weights = bootstrap_weights(manifest)
  plans = {m["name"]: m["steps"] for m in manifest["motions"]}
  sources = tracking_sources(directory, manifest)
  all_stats, all_time, records = {}, [], []
  for mode in ROUTES:
    if mode in sources:
      run, _ = sources[mode]
      dt = json.loads((run / "timing.json").read_text())["dt"]
      data, episodes = load_tracking(directory, manifest, [mode])
      stats = estimates(list(data[mode].values()), weights)
      series = elapsed_errors(data[mode], plans, dt)
      episodes = [dict(r, source_identity=manifest["identity"]) for r in episodes]
    else:
      if (
        nominal is None
        or mode not in ("GT-Oracle", "Prop.+KP")
        or manifest["kappa_o"] != 1
      ):
        raise ValueError(f"No valid source for {mode}")
      stats, series, episodes = nominal[mode]
    all_stats[mode] = ModeResult(stats, series, episodes)
    all_time.extend(dict(mode=mode, **r) for r in series)
    records.extend(episodes)
  write_csv(folder / "episodes.csv", records)
  write_csv(folder / "tracking-time.csv", all_time)
  kb, ko = manifest.get("kappa_b", 1), manifest.get("kappa_o", 1)
  plot_elapsed(folder, f"50k · κb = {kb}, κo = {ko}", all_time)
  modes = {
    mode: {k: interval(a) for k, a in stats.items()}
    for mode, (stats, _, _) in all_stats.items()
  }
  write_csv(
    folder / "tracking.csv",
    [
      dict(mode=mode, **{k: s["value"] for k, s in row.items()})
      for mode, row in modes.items()
    ],
  )
  differences = {
    f"{b} minus {a}": {
      k: interval(v - all_stats[a].estimates[k])
      for k, v in all_stats[b].estimates.items()
    }
    for b, a in CONTRASTS
  }
  extra_key = "GT-Oracle minus Prop.+KP"
  atomic_json(
    folder / "report.json",
    dict(
      identity=manifest["identity"],
      kappa_b=kb,
      kappa_o=ko,
      episodes=len(records),
      clips=len(plans),
      seeds=manifest["seeds"],
      bootstrap_samples=len(weights) - 1,
      modes=modes,
      additional_paired_differences={extra_key: differences.pop(extra_key)},
      paired_differences=differences,
    ),
  )
  return all_stats


def build_report(baseline, base, output, *, include_sweep):
  from supermimic.validation.nominal_posterior_report import posterior_report

  report_dir = output / "report"
  nominal = summarize_condition(baseline, base, report_dir / "nominal")
  print(f"Rebuilt {report_dir / 'nominal' / 'report.json'}", flush=True)
  if not include_sweep:
    return
  results = {"nominal": {m: s.estimates for m, s in nominal.items()}}
  rows = []
  for name in CONDITIONS:
    directory = output / name
    manifest = load_manifest(directory)
    if manifest["identity"] != condition_manifest(base, name)["identity"]:
      raise ValueError(f"Unpaired or incomplete condition: {directory}")
    stats = summarize_condition(directory, manifest, report_dir / name, nominal=nominal)
    results[name] = {m: s.estimates for m, s in stats.items()}
    posterior_report(directory)
    print(f"Completed reports for {name}", flush=True)
  for name, modes in results.items():
    for mode, stats in modes.items():
      for key in PAPER_METRICS:
        score = interval(stats[key])
        delta = interval(stats[key] - results["nominal"][mode][key])
        rows.append(
          dict(
            condition=name,
            mode=mode,
            metric=key,
            value=score["value"],
            ci95_low=score["ci95"][0],
            ci95_high=score["ci95"][1],
            delta_from_nominal=delta["value"],
            delta_ci95_low=delta["ci95"][0],
            delta_ci95_high=delta["ci95"][1],
          )
        )
  write_csv(report_dir / "uncertainty-sweep.csv", rows)
  plot_robustness(report_dir, results)
