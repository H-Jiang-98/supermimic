"""Context-first, clip-equal distributions and separately labelled clip figures."""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import numpy as np

from supermimic.validation.config import (
  LOCAL_STUDY_VERSION,
  LOCAL_VARIANTS,
  PRE_FAILURE_VERSIONS,
  VARIANTS,
  atomic_json,
)
from supermimic.validation.metrics import (
  LOCAL_METRICS,
  ODOMETRY,
  ODOMETRY_COMPONENTS,
  ODOMETRY_SUMMARY,
  RELATIVE_METRICS,
  TRACKING,
)


def validate_score_support(manifest, meta, arrays):
  """Every populated score must cover exactly the pre-failure frame prefix."""
  bins = arrays["bins"]
  full = arrays["full"]
  if manifest.get("version") not in PRE_FAILURE_VERSIONS:
    if not np.isfinite(full).all() or not np.isfinite(bins).all():
      raise ValueError("Non-finite metric in complete task")
    return
  failures = arrays["first_failure_s"]
  frames = meta["clip_frames"]
  ends = np.where(np.isnan(failures), frames, np.rint(failures / meta["control_dt"]))
  counts = arrays["bin_counts"]
  if (
    counts.sum() != frames
    or (counts <= 0).any()
    or (ends < 0).any()
    or (ends > frames).any()
    or not np.array_equal(ends, arrays["valid_frame_counts"])
  ):
    raise ValueError("Incomplete pre-failure frame counts")
  starts = np.cumsum(counts) - counts
  for scores, populated in ((full, ends > 0), (bins, starts[:, None] < ends)):
    if (
      not np.isfinite(scores[populated]).all() or not np.isnan(scores[~populated]).all()
    ):
      raise ValueError("Invalid score for pre-failure support")


def available_mean(values):
  """Equal weight per contributing context (or clip), NaN for an empty bin."""
  return np.ma.masked_invalid(values).mean(axis=0).filled(np.nan)


def aggregate(manifest, loaded, variants=None):
  """Only full clips with every planned context in EVERY variant are comparable."""
  if variants is None:
    variants = ["C-posterior"] if manifest["kind"] == "odometry" else list(VARIANTS)
  n = manifest["config"]["num_domain_contexts"]
  gathered = {}
  invalid = []
  durations = {}
  for meta, arrays in loaded:
    if meta["status"] != "complete":
      invalid.append(
        dict(clip=meta["clip"], variant=meta["variant"], status=meta["status"])
      )
      continue
    validate_score_support(manifest, meta, arrays)
    key = (meta["clip"], meta["variant"])
    durations[key] = (meta["clip_frames"] - 1) * meta["control_dt"]
    for row, context in enumerate(arrays["context_ids"]):
      context = int(context)
      bucket = gathered.setdefault(key, {})
      if context in bucket:
        raise ValueError("Duplicate context result")
      bucket[context] = dict(
        full=arrays["full"][row],
        bins=arrays["bins"][:, row],
        failure=arrays["first_failure_s"][row],
      )
  clips, excluded = {}, []
  for motion in manifest["motions"]:
    name = motion["name"]
    if any(set(gathered.get((name, v), {})) != set(range(n)) for v in variants):
      excluded.append(name)
      continue
    clips[name] = {}
    for variant in variants:
      rows = [gathered[(name, variant)][i] for i in range(n)]
      full, bins = (
        np.stack([r["full"] for r in rows]),
        np.stack([r["bins"] for r in rows]),
      )
      failures = np.array([r["failure"] for r in rows])
      clips[name][variant] = dict(
        full=full,
        bins=bins,
        mean=available_mean(full),
        mean_bins=available_mean(bins),
        clip_duration_s=durations[(name, variant)],
        successful_contexts=int(np.isnan(failures).sum()),
        median_first_failure_s=(
          float(np.nanmedian(failures)) if np.isfinite(failures).any() else None
        ),
        no_failure_rate=float(np.isnan(failures).mean()),
        first_failure_s=[None if np.isnan(v) else float(v) for v in failures],
      )
  return variants, clips, excluded, invalid


def _plot_odometry_time(
  cfg, clips, variant, directory, selected=None, metrics=ODOMETRY_SUMMARY
):
  import matplotlib.pyplot as plt

  data = clips if selected is None else {selected: clips[selected]}
  sample_label = (
    "clips (each averaged over contexts)" if selected is None else "contexts"
  )
  stem = "summary" if selected is None else Path(selected).stem
  count_metrics = len(metrics)
  fig, axes = plt.subplots(
    count_metrics,
    1,
    figsize=(cfg.plot_width, cfg.plot_row_height * count_metrics),
    squeeze=False,
  )
  stride = round(cfg.boxplot_interval_s / cfg.time_bin_s)
  count = max(c[variant]["mean_bins"].shape[0] for c in data.values())
  indices = list(range(0, count, stride))
  for ax, metric in zip(axes[:, 0], metrics, strict=True):
    mi = ODOMETRY.index(metric)
    distributions, labels = [], []
    for bi in indices:
      if selected is None:
        values = [
          c[variant]["mean_bins"][bi, mi]
          for c in data.values()
          if bi < len(c[variant]["mean_bins"])
        ]
      else:
        values = data[selected][variant]["bins"][:, bi, mi]
      values = np.asarray(values)
      values = values[np.isfinite(values)]
      distributions.append(values)
      labels.append(f"{bi * cfg.time_bin_s:g}\nn={len(values)}")
    ax.boxplot(distributions, tick_labels=labels, showmeans=True, widths=0.6)
    ax.tick_params(axis="x", labelrotation=90, labelsize=6)
    ax.set_ylabel(metric)
    ax.set_xlabel(f"Elapsed seconds, bin width {cfg.time_bin_s:g}s")
    if metric == "root_nees":
      ax.axhline(9, linestyle="--", color="gray")
    ax.grid(axis="y", alpha=0.2)
  fig.suptitle(
    f"{selected or 'All clips'}: online posterior during {variant}; samples = {sample_label}"
  )
  fig.tight_layout(rect=(0, 0, 1, 0.98))
  suffix = "pose_components_time" if metrics == ODOMETRY_COMPONENTS else "time"
  fig.savefig(directory / f"{stem}_{variant}_{suffix}.svg", dpi=cfg.plot_dpi)
  plt.close(fig)


def _plot_tracking_success(clips, variants, directory, dpi):
  import matplotlib.pyplot as plt

  names = sorted(clips)
  successes = np.array(
    [[clips[name][v]["successful_contexts"] for v in variants] for name in names]
  )
  total = len(clips[names[0]][variants[0]]["first_failure_s"])
  rates = successes / total
  fig, ax = plt.subplots(
    figsize=(max(5.8, 2.5 + 1.65 * len(variants)), 1.7 + 0.48 * len(names)),
    layout="constrained",
  )
  ax.imshow(rates, cmap="Blues", vmin=0, vmax=1, aspect="auto", interpolation="nearest")
  ax.set_yticks(range(len(names)), [Path(name).stem for name in names])
  ax.set_xticks(range(len(variants)), variants)
  ax.xaxis.tick_top()
  ax.tick_params(length=0, labelsize=9, pad=5)
  ax.set_title("Tracking success by clip", loc="left", pad=16)
  ax.set_xlabel(
    "Successful contexts / total contexts\nMedian first failure / clip duration (seconds)",
    labelpad=10,
  )
  for (i, j), count in np.ndenumerate(successes):
    stats = clips[names[i]][variants[j]]
    label = f"{count}/{total}"
    if stats["median_first_failure_s"] is not None:
      median, duration = (
        f"{stats[k]:.2f}".rstrip("0").rstrip(".")
        for k in ("median_first_failure_s", "clip_duration_s")
      )
      label += f"\n{median}s / {duration}s"
    ax.text(
      j,
      i,
      label,
      ha="center",
      va="center",
      fontsize=9,
      color="white" if rates[i, j] >= 0.63 else "#172839",
    )
  fig.savefig(directory / "tracking-success-by-clip.svg", dpi=dpi)
  plt.close(fig)


def report(cfg):
  from supermimic.validation.study import load_task, tasks

  root = Path(cfg.output)
  manifest = json.loads((root / "manifest.json").read_text())
  if cfg.time_bin_s != manifest["config"]["time_bin_s"]:
    raise ValueError("Cannot change time_bin_s when replotting saved bins")
  joint = manifest["kind"] == "joint"
  kinds = ("tracking", "odometry") if joint else (manifest["kind"],)
  fields = ["context_ids", "full", "bins", "first_failure_s"]
  if manifest.get("version") == LOCAL_STUDY_VERSION:
    fields.extend(["relative_pose_1s", "relative_pose_1s_count"])
  if joint:
    fields.extend(["odometry_full", "odometry_bins"])
  censored_scoring = manifest["version"] in PRE_FAILURE_VERSIONS
  if censored_scoring:
    fields.extend(["valid_frame_counts", "odometry_valid_frame_counts", "bin_counts"])
  loaded = []
  missing = {
    v: []
    for v in (
      LOCAL_VARIANTS if manifest["version"] == LOCAL_STUDY_VERSION else VARIANTS
    )
  }
  for key, _, variant, _ in tasks(manifest):
    task = load_task(root, key, manifest, fields=fields)
    if task is None:
      missing[variant].append(key)
    else:
      loaded.append(task)
  complete = True
  for kind in kinds:
    records = loaded
    if joint and kind == "odometry":
      records = [
        (
          {**meta, "status": meta["odometry_status"]},
          {
            **arrays,
            "full": arrays["odometry_full"],
            "bins": arrays["odometry_bins"],
            "valid_frame_counts": arrays.get("odometry_valid_frame_counts"),
          },
        )
        for meta, arrays in loaded
      ]
    if joint:
      requested = cfg.methods
    else:
      requested = ["C-posterior"] if kind == "odometry" else list(VARIANTS)
    summary = _write_report(
      cfg, manifest, kind, records, missing, requested, root / kind if joint else root
    )
    complete = complete and summary["complete"]
  return {"complete": complete}


def _write_report(cfg, manifest, kind, loaded, missing, requested, output):
  from supermimic.scripts.plot_eit_tracking_time import LABELS, pad_curves, plot_curves

  odometry_names = (
    ODOMETRY if manifest["version"] == "eit_joint_multiclip_v5" else ODOMETRY_SUMMARY
  )
  if manifest["version"] == LOCAL_STUDY_VERSION:
    odometry_names = ODOMETRY + LOCAL_METRICS
  names = odometry_names if kind == "odometry" else TRACKING
  # A method enters its report only when every planned clip and context is complete.
  variants, clips, invalid = [], {}, []
  excluded_by_method = {}
  for variant in missing:
    records = [(m, a) for m, a in loaded if m["variant"] == variant]
    if not records and variant not in requested:
      continue
    _, part, excluded, bad = aggregate(manifest, records, [variant])
    invalid.extend(bad)
    if excluded or missing[variant]:
      excluded_by_method[variant] = excluded
      continue
    variants.append(variant)
    for clip, data in part.items():
      clips.setdefault(clip, {}).update(data)
  directory = output / "figures"
  directory.mkdir(parents=True, exist_ok=True)
  report_path = output / "report.json"
  previous = json.loads(report_path.read_text()) if report_path.exists() else {}
  with tempfile.TemporaryDirectory(dir=output) as tmp:
    staging = Path(tmp)
    if kind == "odometry":
      for variant in variants:
        for name in [None, *(n for n in cfg.representative_clips if n in clips)]:
          _plot_odometry_time(cfg, clips, variant, staging, selected=name)
          if odometry_names == ODOMETRY:
            _plot_odometry_time(
              cfg, clips, variant, staging, selected=name, metrics=ODOMETRY_COMPONENTS
            )
    elif clips:
      indices = [names.index(key) for key in LABELS]
      _plot_tracking_success(clips, variants, staging, cfg.plot_dpi)
      clip_names = sorted(clips)
      for variant in variants:
        curves = [clips[n][variant]["mean_bins"][:, indices] for n in clip_names]
        plot_curves(
          clip_names,
          cfg.time_bin_s,
          pad_curves(curves),
          staging,
          variant,
          cfg.plot_dpi,
        )
    figures = sorted(p.name for p in staging.iterdir())
    for name in figures:
      (staging / name).replace(directory / name)
    for stale in set(previous.get("figures", [])) - set(figures):
      if Path(stale).name == stale:
        (directory / stale).unlink(missing_ok=True)
  summary = dict(
    version=manifest["version"],
    identity=manifest["identity"],
    complete=all(v in variants for v in requested),
    requested_methods=requested,
    completed_methods=variants,
    figures=figures,
    sample_unit="clip_mean_over_contexts",
    scoring_window=(
      "before_first_tracking_failure"
      if manifest["version"] in PRE_FAILURE_VERSIONS
      else "full_clip"
    ),
    num_contexts=manifest["config"]["num_domain_contexts"],
    included_clips=len(clips),
    excluded_clips_by_method=excluded_by_method,
    missing_tasks={v: missing[v] for v in excluded_by_method},
    invalid_tasks=invalid,
    unavailable_representatives=[n for n in cfg.representative_clips if n not in clips],
    training_seen_clips=sum(m["seen_in_training"] for m in manifest["motions"]),
    variants={
      v: {
        name: float(np.mean([c[v]["mean"][i] for c in clips.values()]))
        if clips
        else None
        for i, name in enumerate(names)
      }
      for v in variants
    },
    clips={
      name: {
        v: dict(
          metrics=dict(zip(names, c[v]["mean"].tolist(), strict=True)),
          no_failure_rate=c[v]["no_failure_rate"],
          successful_contexts=c[v]["successful_contexts"],
          median_first_failure_s=c[v]["median_first_failure_s"],
          clip_duration_s=c[v]["clip_duration_s"],
          first_failure_s=c[v]["first_failure_s"],
        )
        for v in variants
      }
      for name, c in clips.items()
    },
  )
  if manifest["version"] == LOCAL_STUDY_VERSION:
    for variant in variants:
      summary["variants"][variant]["success_rate"] = float(
        np.mean([clip[variant]["no_failure_rate"] for clip in clips.values()])
      )
      if kind == "odometry":
        grouped = {}
        for meta, arrays in loaded:
          if meta["variant"] == variant and meta["status"] == "complete":
            grouped.setdefault(meta["clip"], []).extend(
              arrays["relative_pose_1s"][arrays["relative_pose_1s_count"] > 0]
            )
        means = [np.mean(rows, axis=0) for rows in grouped.values() if len(rows)]
        for i, key in enumerate(RELATIVE_METRICS):
          summary["variants"][variant][key] = (
            float(np.mean(means, axis=0)[i]) if means else None
          )
        summary["variants"][variant]["contexts_with_1s_support"] = sum(
          map(len, grouped.values())
        )
  atomic_json(output / "report.json", summary)
  with (output / "clip_metrics.csv").open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(
      [
        "clip",
        "variant",
        "context_count",
        *names,
        "no_failure_rate",
        "successful_contexts",
        "median_first_failure_s",
        "clip_duration_s",
      ]
    )
    for name, groups in clips.items():
      for variant, values in groups.items():
        writer.writerow(
          [
            name,
            variant,
            summary["num_contexts"],
            *values["mean"],
            values["no_failure_rate"],
            values["successful_contexts"],
            values["median_first_failure_s"],
            values["clip_duration_s"],
          ]
        )
  with (output / "context_metrics.csv").open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["clip", "variant", "context_id", "status", *names])
    for meta, arrays in loaded:
      for i, context in enumerate(arrays["context_ids"]):
        writer.writerow(
          [meta["clip"], meta["variant"], context, meta["status"], *arrays["full"][i]]
        )
  print(
    f"Report: {output / 'report.json'}; complete={summary['complete']}, clips={len(clips)}",
    flush=True,
  )
  return summary
