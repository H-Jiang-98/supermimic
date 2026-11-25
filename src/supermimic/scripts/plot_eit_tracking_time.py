"""Plot saved tracking errors against elapsed clip time."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

LABELS = {
  "mpkpe_m": "Mean per-keypoint position error (m)",
  "root_position_rmse_m": "Root position RMSE (m)",
  "root_rotation_rmse_deg": "Root rotation RMSE (deg)",
  "root_velocity_rmse_m_s": "Root velocity RMSE (m/s)",
  "ee_position_error_m": "End-effector position error (m)",
  "ee_orientation_error_deg": "End-effector orientation error (deg)",
  "joint_position_rmse_deg": "Joint position RMSE (deg)",
  "joint_velocity_rmse_deg_s": "Joint velocity RMSE (deg/s)",
}


def load_curves(root, variant="A-posterior"):
  manifest = json.loads((root / "manifest.json").read_text())
  motions = {m["name"]: m["frames"] for m in manifest["motions"]}
  contexts = manifest["config"]["num_domain_contexts"]
  width = manifest["config"]["time_bin_s"]
  clips = {}
  for path in sorted((root / "tasks").glob("*.json")):
    meta = json.loads(path.read_text())
    if meta["variant"] != variant:
      continue
    name = meta["clip"]
    if meta["identity"] != manifest["identity"] or meta["status"] != "complete":
      raise ValueError(f"Incompatible or incomplete task: {path}")
    indices = [meta["metric_names"].index(key) for key in LABELS]
    with np.load(path.with_suffix(".npz")) as data:
      scores = data["bins"][:, :, indices]
      if (
        data["bin_counts"].sum() != motions[name]
        or not np.all(data["valid_frame_counts"] == motions[name])
        or not np.isfinite(scores).all()
      ):
        raise ValueError(f"Incomplete scores: {name}")
      rows = clips.setdefault(name, {})
      for i, context in enumerate(data["context_ids"]):
        if int(context) in rows:
          raise ValueError(f"Duplicate context in {name}")
        rows[int(context)] = scores[:, i]
  if set(clips) != set(motions) or any(
    set(rows) != set(range(contexts)) for rows in clips.values()
  ):
    raise ValueError("Expected all clips and contexts in the manifest")
  names = sorted(clips)
  curves = [np.mean(list(clips[name].values()), axis=0) for name in names]
  return names, width, pad_curves(curves)


def pad_curves(curves):
  values = np.full((len(curves), max(map(len, curves)), len(LABELS)), np.nan)
  for row, curve in enumerate(curves):
    values[row, : len(curve)] = curve
  return values


def plot_curves(names, width, values, folder, variant, dpi):
  folder.mkdir(parents=True, exist_ok=True)
  t = np.arange(values.shape[1]) * width
  count = np.isfinite(values[:, :, 0]).sum(axis=0)
  low, med, high = np.nanpercentile(values, [25, 50, 75], axis=0)
  ink, band = "#124967", "#8ad2cf"
  fig = plt.figure(figsize=(13, 13), layout="constrained")
  grid = fig.add_gridspec(5, 2, height_ratios=[1, 1, 1, 1, 0.42])
  fig.suptitle(
    f"{variant} tracking errors over time   |   {len(names)} clips", fontsize=18
  )
  for i, label in enumerate(LABELS.values()):
    ax = fig.add_subplot(grid[i // 2, i % 2])
    ax.plot(t, values[:, :, i].T, color="#8a939e", alpha=0.32, lw=0.6)
    ax.fill_between(t, low[:, i], high[:, i], color=band, alpha=0.8)
    ax.plot(t, med[:, i], color=ink, lw=1.9)
    ax.set(title=label, xlim=(0, t[-1] + width), ylim=(0, None))
    ax.grid(axis="y", alpha=0.2)
    if i == 0:
      ax.legend(
        handles=[
          Line2D([], [], color="#9ba2a9", label="Individual clip"),
          Line2D([], [], color=ink, label="Clip median"),
          Patch(color=band, label="25th–75th percentiles"),
        ],
        fontsize=8,
      )
  ax = fig.add_subplot(grid[-1, :])
  ax.stairs(count, np.append(t, t[-1] + width), fill=True, color=ink)
  ax.set(
    xlim=(0, t[-1] + width),
    ylabel="Clips in bin",
    xlabel=f"Elapsed time from clip start (s), {width:g}-second bins",
  )
  fig.savefig(folder / f"{variant}-tracking-time.svg", dpi=dpi)
  plt.close(fig)

  fig, ax = plt.subplots(figsize=(12, 11), layout="constrained")
  cmap = plt.get_cmap("viridis").with_extremes(bad="#eeeeee")
  heat = ax.imshow(
    values[:, :, 0],
    aspect="auto",
    interpolation="nearest",
    cmap=cmap,
    extent=(0, t[-1] + width, len(names) - 0.5, -0.5),
    vmin=0,
  )
  ax.set_yticks(np.arange(len(names)), [Path(n).stem for n in names], fontsize=8)
  ax.set(
    title=f"{variant} keypoint tracking error by clip",
    xlabel="Elapsed time from clip start (s)",
  )
  fig.colorbar(heat, ax=ax, label="MPKPE (m)")
  fig.savefig(folder / f"{variant}-mpkpe-by-clip.svg", dpi=dpi)
  plt.close(fig)
  print(f"{folder}: {len(names)} clips, {len(t)} bins, {count[-1]} clips in final bin")


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("study", type=Path)
  parser.add_argument("--method", default="A-posterior")
  args = parser.parse_args()
  names, width, values = load_curves(args.study, args.method)
  joint = json.loads((args.study / "manifest.json").read_text())["kind"] == "joint"
  folder = (args.study / "tracking" if joint else args.study) / "figures"
  plot_curves(names, width, values, folder, args.method, 160)


if __name__ == "__main__":
  main()
