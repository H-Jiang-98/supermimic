"""Context-first, clip-equal distributions and separately labelled clip figures."""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import numpy as np

from supermimic.validation.config import VARIANTS, atomic_json
from supermimic.validation.metrics import ODOMETRY, TRACKING


def aggregate(manifest, loaded, variants=None):
  """Only full clips with every planned context in EVERY variant are comparable."""
  if variants is None:
    variants = ["C-posterior"] if manifest["kind"] == "odometry" else list(VARIANTS)
  n = manifest["config"]["num_domain_contexts"]
  gathered = {}
  invalid = []
  for meta, arrays in loaded:
    if meta["status"] != "complete":
      invalid.append(
        dict(clip=meta["clip"], variant=meta["variant"], status=meta["status"])
      )
      continue
    key = (meta["clip"], meta["variant"])
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
      if not np.isfinite(full).all() or not np.isfinite(bins).all():
        raise ValueError("Non-finite metric in complete task")
      failures = np.array([r["failure"] for r in rows])
      clips[name][variant] = dict(
        full=full,
        bins=bins,
        mean=full.mean(0),
        mean_bins=bins.mean(0),
        no_failure_rate=float(np.isnan(failures).mean()),
        first_failure_s=[None if np.isnan(v) else float(v) for v in failures],
      )
  return variants, clips, excluded, invalid


def _plot_odometry_time(cfg, clips, variant, directory, selected=None):
  import matplotlib.pyplot as plt

  data = clips if selected is None else {selected: clips[selected]}
  sample_label = (
    "clips (each averaged over contexts)" if selected is None else "contexts"
  )
  stem = "summary" if selected is None else Path(selected).stem
  count_metrics = len(ODOMETRY)
  fig, axes = plt.subplots(
    count_metrics,
    1,
    figsize=(cfg.plot_width, cfg.plot_row_height * count_metrics),
    squeeze=False,
  )
  stride = round(cfg.boxplot_interval_s / cfg.time_bin_s)
  count = max(c[variant]["mean_bins"].shape[0] for c in data.values())
  indices = list(range(0, count, stride))
  for mi, metric in enumerate(ODOMETRY):
    ax = axes[mi, 0]
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
    f"{selected or 'All clips'}: C online posterior during {variant}; samples = {sample_label}"
  )
  fig.tight_layout(rect=(0, 0, 1, 0.98))
  fig.savefig(directory / f"{stem}_{variant}_time.svg", dpi=cfg.plot_dpi)
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
  if joint:
    fields.extend(["odometry_full", "odometry_bins"])
  loaded = []
  missing = {v: [] for v in VARIANTS}
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
          {**arrays, "full": arrays["odometry_full"], "bins": arrays["odometry_bins"]},
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

  names = ODOMETRY if kind == "odometry" else TRACKING
  # A method enters its report only when every planned clip and context is complete.
  variants, clips, invalid = [], {}, []
  excluded_by_method = {}
  for variant in VARIANTS:
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
        _plot_odometry_time(cfg, clips, variant, staging)
        for name in cfg.representative_clips:
          if name in clips:
            _plot_odometry_time(cfg, clips, variant, staging, selected=name)
    elif clips:
      indices = [names.index(key) for key in LABELS]
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
          first_failure_s=c[v]["first_failure_s"],
        )
        for v in variants
      }
      for name, c in clips.items()
    },
  )
  atomic_json(output / "report.json", summary)
  with (output / "clip_metrics.csv").open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["clip", "variant", "context_count", *names, "no_failure_rate"])
    for name, groups in clips.items():
      for variant, values in groups.items():
        writer.writerow(
          [
            name,
            variant,
            summary["num_contexts"],
            *values["mean"],
            values["no_failure_rate"],
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
