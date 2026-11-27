"""Offline figures for the repeated nominal tracking and Belief evaluations."""

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

BLUE = "#17658a"
ORANGE = "#ab4d25"
INK = "#233d48"
GRID = "#dde4e6"


def figure(size):
  fig = Figure(figsize=size, layout="constrained")
  FigureCanvasAgg(fig)
  return fig


def finish(fig, path):
  fig.savefig(path, dpi=200, facecolor="white")
  fig.clear()


def axes_style(ax, grid_axis):
  ax.spines[["top", "right"]].set_visible(False)
  ax.set_axisbelow(True)
  ax.grid(axis=grid_axis, color=GRID, linewidth=0.65)
  ax.tick_params(labelsize=9)


def plot_tracking(folder, summary, clips, bootstrap_samples):
  modes = list(summary)
  fig = figure((11.4, 3.8))
  axes = fig.subplots(1, 4, sharey=True)
  y = np.arange(len(modes))
  for ax, (metric, title, unit) in zip(
    axes,
    (
      ("sr_pct", "Success rate", "%"),
      ("cr_pct", "Completion ratio", "%"),
      ("mpjpe_cm_p95", "P95 MPJPE", "cm"),
      ("cost_cvar90", "Cost CVaR₉₀", "cost"),
    ),
    strict=True,
  ):
    axes_style(ax, "x")
    for i, mode in enumerate(modes):
      stats = summary[mode]
      low, high = stats["pooled"][metric]["ci95"]
      ax.hlines(i, low, high, color=INK, lw=1.8)
      ax.plot(stats["pooled"][metric]["value"], i, "D", color=INK, ms=5)
      ax.plot(
        stats["first"][metric]["value"], i - 0.19, "o", mfc="white", mec=BLUE, ms=5
      )
      ax.plot(
        stats["repeat"][metric]["value"], i + 0.19, "^", mfc=ORANGE, mec=ORANGE, ms=5
      )
    ax.set(title=title, xlabel=unit, yticks=y, yticklabels=modes)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.tick_params(axis="y", length=0)
    ax.margins(x=0.12)
  axes[0].set_ylim(len(modes) - 0.55, -0.55)
  fig.suptitle("Nominal tracking across repeated runs", fontsize=15, weight="semibold")
  fig.legend(
    handles=[
      Line2D(
        [], [], marker="o", color=BLUE, mfc="white", linestyle="none", label="First run"
      ),
      Line2D([], [], marker="^", color=ORANGE, linestyle="none", label="Repeat"),
      Line2D(
        [],
        [],
        marker="D",
        color=INK,
        label=f"Pooled estimate and 95% interval ({clips} clips, {bootstrap_samples:,} paired bootstrap draws)",
      ),
    ],
    loc="outside lower center",
    ncols=3,
    frameon=False,
    fontsize=9,
  )
  finish(fig, folder / "tracking.png")


def plot_integration(folder, episodes):
  fig = figure((11.1, 3.65))
  axes = fig.subplots(1, 3, sharey=True)
  for ax, (key, title, unit) in zip(
    axes,
    (
      ("ate_m", "Trajectory RMSE", "m"),
      ("fde_m", "Final displacement error", "m"),
      ("drift_pct", "Drift for paths longer than 1 m", "% of true path length"),
    ),
    strict=True,
  ):
    axes_style(ax, "both")
    for status, name, color in (
      ("success", "Completed", BLUE),
      ("failure", "Failed prefix", ORANGE),
    ):
      values = np.sort(
        [e[key] for e in episodes if e["status"] == status and e[key] is not None]
      )
      if len(values):
        x = np.r_[0.0, values]
        ax.step(
          x,
          np.arange(len(x)) * 100 / len(values),
          where="post",
          color=color,
          lw=1.7,
          label=f"{name} (n={len(values)})",
        )
        avg = values.mean()
        ax.plot(
          avg,
          100 * np.searchsorted(values, avg, side="right") / len(values),
          "D",
          ms=5,
          color=color,
        )
    ax.set(title=title, xlabel=unit, ylim=(0, 103), xlim=(0, None))
    if ax.lines:
      ax.legend(frameon=False, fontsize=8, loc="lower right")
    else:
      ax.text(
        0.5, 0.5, "No eligible episodes", transform=ax.transAxes, ha="center", color=INK
      )
  axes[0].set_ylabel("Cumulative share of episodes (%)")
  fig.suptitle("Known-orientation velocity integration", fontsize=15, weight="semibold")
  fig.legend(
    handles=[
      Line2D(
        [], [], marker="D", color=INK, linestyle="none", label="Episode-weighted mean"
      )
    ],
    loc="outside lower center",
    frameon=False,
    fontsize=9,
  )
  finish(fig, folder / "velocity-integration.png")


def plot_coverage(folder, rows, curves, nees):
  fig = figure((10.8, 4.6))
  ax, fixed = fig.subplots(1, 2, width_ratios=[1.45, 1])
  axes_style(ax, "both")
  axes_style(fixed, "x")
  palette = ("#246789", "#ad4d23", "#258263", "#843a76", "#96701c", "#53577f")
  for i, (row, color) in enumerate(zip(rows, palette, strict=True)):
    values = [r for r in curves if r["component"] == row["component"]]
    ax.plot(
      [r["k"] for r in values],
      [r["empirical_pct"] for r in values],
      lw=1.8,
      color=color,
      label=row["component"],
    )
    one, two = row["coverage_1sigma_pct"], row["coverage_2sigma_pct"]
    if one is not None and two is not None:
      fixed.plot([one, two], [i, i], color=color, lw=1.2)
      fixed.plot(one, i, "o", color=color, ms=5)
      fixed.plot(two, i, "s", mfc="white", mec=color, ms=5)
  reference = [r for r in curves if r["component"] == rows[0]["component"]]
  ax.plot(
    [r["k"] for r in reference],
    [r["gaussian_pct"] for r in reference],
    "--",
    color=INK,
    label="Gaussian",
  )
  ax.set(
    xlabel="Interval half-width (posterior σ)",
    ylabel="Empirical coverage (%)",
    xlim=(0, 4),
    ylim=(0, 102),
  )
  ax.legend(ncols=2, fontsize=8, frameon=False)
  for target in (68.27, 95.45):
    fixed.axvline(target, color=INK, ls=":", lw=1)
    fixed.text(
      target,
      1.025,
      f"{target:g}%",
      ha="center",
      fontsize=8,
      transform=fixed.get_xaxis_transform(),
    )
  fixed.set(
    yticks=range(len(rows)),
    yticklabels=[r["component"] for r in rows],
    ylim=(len(rows) - 0.5, -0.5),
    xlim=(0, 101),
    xlabel="Empirical coverage (%)",
  )
  fixed.legend(
    handles=[
      Line2D([], [], marker="o", color=INK, ls="none", label="1σ"),
      Line2D([], [], marker="s", mfc="white", color=INK, ls="none", label="2σ"),
      Line2D([], [], color=INK, ls=":", label="Gaussian targets"),
    ],
    loc="lower center",
    bbox_to_anchor=(0.5, -0.32),
    ncols=3,
    fontsize=8,
    frameon=False,
  )
  fig.suptitle(
    f"Belief posterior calibration   ·   normalized mean NEES {nees:.2f} (target 1)",
    fontsize=14,
    weight="semibold",
  )
  finish(fig, folder / "coverage.png")


def plot_example(
  folder, scenario, status, data, actual, mean, sigma, estimated, distance
):
  time = data["step"] * float(data["dt"])
  stem = Path(scenario.clip).stem
  end = "completed" if status == "success" else "failed prefix"
  title = f"{stem}   ·   seed {scenario.seed}   ·   {end} at {time[-1]:.2f} s"
  fig = figure((12.5, 6.0))
  axes = fig.subplots(2, 3, sharex=True)
  for j, (ax, label) in enumerate(
    zip(
      axes.flat,
      (
        "Pelvis height (cm)",
        "Roll (°)",
        "Pitch (°)",
        "Body vx (m/s)",
        "Body vy (m/s)",
        "Body vz (m/s)",
      ),
      strict=True,
    )
  ):
    truth, center = actual[:, j].copy(), mean[:, j].copy()
    if j in (1, 2):
      for segment in np.ma.clump_unmasked(np.ma.masked_invalid(truth + center)):
        center[segment] = np.unwrap(center[segment], period=360)
        truth[segment] = (
          center[segment] + (truth[segment] - center[segment] + 180) % 360 - 180
        )
    axes_style(ax, "y")
    ax.fill_between(
      time,
      center - 2 * sigma[:, j],
      center + 2 * sigma[:, j],
      color="#91bfd5",
      alpha=0.65,
      label="Posterior ±2σ",
    )
    ax.plot(time, truth, color=ORANGE, lw=1.05, label="Truth")
    ax.plot(time, center, color=BLUE, lw=0.85, linestyle="--", label="Posterior mean")
    ax.set(ylabel=label, xlim=(0, time[-1]))
    if j >= 3:
      ax.set_xlabel("Time (s)")
  fig.suptitle(title, fontsize=14, weight="semibold")
  fig.legend(
    *axes[0, 0].get_legend_handles_labels(),
    loc="outside lower center",
    ncols=3,
    frameon=False,
    fontsize=9,
  )
  prefix = f"{stem}-{scenario.seed}"
  finish(fig, folder / f"{prefix}-posterior.png")

  fig = figure((9.1, 4.2))
  path, error = fig.subplots(1, 2)
  axes_style(path, "both")
  axes_style(error, "y")
  pos = data["truth_position"]
  for xy, color, label in ((pos, ORANGE, "Truth"), (estimated, BLUE, "Integrated")):
    path.plot(xy[:, 0], xy[:, 1], color=color, lw=1.3, label=label)
    path.plot(*xy[-1, :2], "X", color=color, ms=7)
  path.plot(*pos[0, :2], "o", mfc="white", mec=INK, ms=6, label="Common start")
  path.plot([], [], "X", color=INK, ms=6, label="Endpoint")
  path.set(
    xlabel="World x (m)", ylabel="World y (m)", aspect="equal", adjustable="datalim"
  )
  path.legend(fontsize=8, ncols=2, frameon=True, loc="best")
  error.plot(time, distance, color=BLUE, lw=1.5)
  error.plot(time[-1], distance[-1], "D", color=BLUE, ms=5)
  error.set(xlabel="Time (s)", ylabel="3D position error (m)", ylim=(0, None))
  error.text(
    0.03,
    0.96,
    f"FDE {distance[-1]:.2f} m",
    transform=error.transAxes,
    va="top",
    color=INK,
  )
  fig.suptitle(title, fontsize=14, weight="semibold")
  finish(fig, folder / f"{prefix}-integration.png")
