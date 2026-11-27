"""Exact pooled-frame percentiles and paired, whole-clip bootstrap intervals."""

import csv
import json
from pathlib import Path
from typing import TypedDict

import numpy as np

from supermimic.validation.config import atomic_json, digest
from supermimic.validation.nominal import (
  FRAME_COLUMNS,
  ROUTES,
  Scenario,
  read_controls,
  read_episode,
  scenario_groups,
)


def episode_values(frames, planned, status):
  executed = len(frames)
  return dict(
    executed=executed,
    sr_pct=float(status == "success") * 100,
    cr_pct=100 * executed / planned,
    cost=float(1.0 - frames[:, -1].sum(dtype=np.float64) / planned),
  )


def pooled_p95(by_clip, weights):
  """Exact linear quantile of a multiset of repeated clips, without expanding draws."""
  ordered = [np.sort(x) for x in by_clip]
  grid = np.unique(np.concatenate(ordered))
  sizes = np.array([len(x) for x in ordered])
  rank = (weights @ sizes - 1) * 0.95

  def value_at(r):
    lo = np.zeros(len(r), dtype=np.int64)
    hi = np.full_like(lo, len(grid) - 1)
    while np.any(lo < hi):
      mid = (lo + hi) // 2
      count = np.zeros_like(lo)
      for i, values in enumerate(ordered):
        count += weights[:, i] * np.searchsorted(values, grid[mid], side="right")
      go_left = count > r
      hi = np.where(go_left, mid, hi)
      lo = np.where(go_left, lo, mid + 1)
    return grid[lo].astype(float)

  base = np.floor(rank).astype(np.int64)
  lower = value_at(base)
  return lower + (rank - base) * (value_at(np.ceil(rank).astype(np.int64)) - lower)


def weighted_tail(costs, weights):
  """Each column is one episode; weights are integer bootstrap multiplicities."""
  order = np.argsort(costs)[::-1]
  sorted_weights = weights[:, order]
  tail_mass = weights.sum(1) * 0.1
  before = np.cumsum(sorted_weights, axis=1) - sorted_weights
  tail = np.clip(tail_mass[:, None] - before, 0, sorted_weights)
  return (tail @ costs[order]) / tail_mass


def estimates(episodes, weights):
  # episodes is indexed by clip, then seed; each entry holds metadata and frames.
  clip_frames = [np.concatenate([frames for _, frames in rows]) for rows in episodes]
  counts = np.array([len(x) for x in clip_frames])
  episodes_per_clip = np.array([len(rows) for rows in episodes])
  frames_total = weights @ counts
  result = {}
  for label in ("sr_pct", "cr_pct", "cost"):
    totals = np.array([sum(ep[label] for ep, _ in rows) for rows in episodes])
    result[label + ("_mean" if label == "cost" else "")] = (weights @ totals) / (
      weights @ episodes_per_clip
    )
  costs = np.array([ep["cost"] for rows in episodes for ep, _ in rows])
  result["cost_cvar90"] = weighted_tail(
    costs, np.repeat(weights, episodes_per_clip, axis=1)
  )
  for j, column in enumerate(FRAME_COLUMNS[:-1]):
    by_clip = [x[:, j] for x in clip_frames]
    result[column + "_mean"] = (
      weights @ [v.sum(dtype=np.float64) for v in by_clip]
    ) / frames_total
    if column != "root_position_cm":
      result[column + "_p95"] = pooled_p95(by_clip, weights)
  return result


def write_csv(path, rows):
  with path.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


class Interval(TypedDict):
  value: float
  ci95: list[float]


def interval(samples) -> Interval:
  lo, hi = np.quantile(samples[1:], [0.025, 0.975])
  return {"value": float(samples[0]), "ci95": [float(lo), float(hi)]}


def load_manifest(output, *, allow_pending=False):
  if not allow_pending and any(output.glob(".batch-*.pending")):
    raise ValueError(
      f"Incomplete nominal batch in {output}; resume the run before reporting"
    )
  manifest = json.loads((output / "manifest.json").read_text())
  if manifest["identity"] != digest(
    {k: v for k, v in manifest.items() if k != "identity"}
  ):
    raise ValueError("Nominal manifest checksum mismatch")
  return manifest


def tracking_sources(output, manifest):
  routes = manifest.get("routes", ROUTES)
  if "source_runs" not in manifest:
    return {mode: (output, manifest["identity"]) for mode in routes}
  sources = {}
  for info in manifest["source_runs"].values():
    directory = output / info["path"]
    if not directory.is_dir():
      directory = output.parent / directory.name
    source = load_manifest(directory)
    if source["identity"] != info["identity"]:
      raise ValueError(f"Wrong source run in {directory}")
    fields = (
      "protocol",
      "motions",
      "seeds",
      "columns",
      "friction",
      "encoder_bias",
      "com_offset",
    )
    if any(source[k] != manifest[k] for k in fields):
      raise ValueError(f"Source protocol differs in {directory}")
    for mode in info["modes"]:
      if mode in sources or source["routes"][mode] != routes[mode]:
        raise ValueError(f"Duplicate or inconsistent route: {mode}")
      for key in filter(None, routes[mode]):
        if (
          source["checkpoints"][key]["sha256"] != manifest["checkpoints"][key]["sha256"]
        ):
          raise ValueError(f"Source checkpoint differs: {mode}/{key}")
      sources[mode] = directory, source["identity"]
  if set(sources) != set(routes):
    raise ValueError("Incomplete split study")
  return {mode: sources[mode] for mode in routes}


def load_tracking(output, manifest, modes=None):
  clips = [m["name"] for m in manifest["motions"]]
  plans = {m["name"]: m["steps"] for m in manifest["motions"]}
  sources = tracking_sources(output, manifest)
  data = {mode: {clip: [] for clip in clips} for mode in (modes or sources)}
  records = []
  for mode, rows in data.items():
    directory, identity = sources[mode]
    for clip in clips:
      for seed in manifest["seeds"]:
        scenario = Scenario(clip, seed, mode)
        loaded = read_episode(scenario.path(directory), identity, plans[clip])
        if loaded is None:
          raise ValueError(
            f"Incomplete study: missing, invalid or damaged episode {scenario}"
          )
        status, frames, terms = loaded
        ep = episode_values(frames, plans[clip], status)
        rows[clip].append((ep, frames))
        records.append(
          dict(
            clip=clip,
            seed=seed,
            mode=mode,
            status=status,
            failure_terms=",".join(terms),
            planned=plans[clip],
            **ep,
          )
        )
  return data, records


def bootstrap_weights(manifest):
  n = len(manifest["motions"])
  rng = np.random.default_rng(manifest["bootstrap_seed"])
  draws = rng.multinomial(n, np.full(n, 1 / n), size=manifest["bootstrap_samples"])
  return np.vstack([np.ones(n, dtype=int), draws])


def report(output, destination=None):
  output = Path(output)
  manifest = load_manifest(output)
  destination = Path(destination) if destination else output
  if "source_runs" in manifest and destination == output:
    destination = output / "rebuilt-report"
  destination.mkdir(parents=True, exist_ok=True)
  data, records = load_tracking(output, manifest)
  weights = bootstrap_weights(manifest)
  stats = {
    mode: estimates(list(group.values()), weights) for mode, group in data.items()
  }
  summary = {
    mode: {key: interval(value) for key, value in values.items()}
    for mode, values in stats.items()
  }
  differences = {}
  for a, b in (
    ("GT-Oracle", "GT-Estimated"),
    ("GT-Estimated", "Mean-in-Loop"),
    ("Mean-in-Loop", "Belief-in-Loop"),
  ):
    if a not in stats or b not in stats:
      continue
    differences[f"{b} minus {a}"] = {
      k: interval(v - stats[a][k]) for k, v in stats[b].items()
    }
  write_csv(destination / "episodes.csv", records)
  write_csv(
    destination / "tracking.csv",
    [
      dict(mode=m, **{k: v["value"] for k, v in metrics.items()})
      for m, metrics in summary.items()
    ],
  )
  atomic_json(
    destination / "report.json",
    dict(
      identity=manifest["identity"],
      episodes=len(records),
      clips=len(manifest["motions"]),
      seeds=manifest["seeds"],
      bootstrap_samples=len(weights) - 1,
      modes=summary,
      paired_differences=differences,
    ),
  )
  print(f"{len(records)} episodes: {destination / 'tracking.csv'}", flush=True)


def merged_report(baseline, repeat):
  from supermimic.validation.nominal_plots import plot_tracking

  baseline, repeat = Path(baseline).resolve(), Path(repeat).resolve()
  if baseline == repeat:
    raise ValueError("Two distinct nominal run directories are required")
  manifest = load_manifest(baseline)
  if load_manifest(repeat)["identity"] != manifest["identity"]:
    raise ValueError("Unequal scenario, noise or checkpoint manifests")
  layout_keys = ("layout", "capacity", "allocated_slots")
  layouts = [
    json.loads((run / "timing.json").read_text()) for run in (baseline, repeat)
  ]
  if any(layouts[0][k] != layouts[1][k] for k in layout_keys):
    raise ValueError("Run layouts differ")
  first, first_records = load_tracking(baseline, manifest)
  second, second_records = load_tracking(repeat, manifest)

  def key(row):
    return row["mode"], row["clip"], row["seed"]

  latter = {key(r): r for r in second_records}
  pairs = []
  for a in first_records:
    b = latter[key(a)]
    pair = dict(
      mode=a["mode"],
      clip=a["clip"],
      seed=a["seed"],
      planned=a["planned"],
      success_flip=int(a["status"] != b["status"]),
      executed_delta=b["executed"] - a["executed"],
    )
    for field in ("status", "failure_terms", "executed", "sr_pct", "cr_pct", "cost"):
      pair["first_" + field] = a[field]
      pair["repeat_" + field] = b[field]
    pairs.append(pair)
  weights = bootstrap_weights(manifest)
  folder = repeat / "merged-report"
  folder.mkdir(exist_ok=True)
  write_csv(folder / "paired-episodes.csv", pairs)
  rows, changes, summary = [], [], {}
  pooled_stats = {}
  for mode in ROUTES:
    a, b = list(first[mode].values()), list(second[mode].values())
    combined = [x + y for x, y in zip(a, b, strict=True)]
    estimates_by_run = {
      label: estimates(episodes, weights)
      for label, episodes in (("first", a), ("repeat", b), ("pooled", combined))
    }
    pooled_stats[mode] = estimates_by_run["pooled"]
    delta = {
      k: v - estimates_by_run["first"][k] for k, v in estimates_by_run["repeat"].items()
    }
    estimates_by_run["repeat_minus_first"] = delta
    summary[mode] = {
      label: {k: interval(v) for k, v in values.items()}
      for label, values in estimates_by_run.items()
    }
    selected = [p for p in pairs if p["mode"] == mode]
    changes.append(
      dict(
        mode=mode,
        unique_scenarios=len(selected),
        repeat_episodes=2 * len(selected),
        success_flips=sum(p["success_flip"] for p in selected),
        success_to_failure=sum(
          p["first_status"] == "success" and p["repeat_status"] == "failure"
          for p in selected
        ),
        failure_to_success=sum(
          p["first_status"] == "failure" and p["repeat_status"] == "success"
          for p in selected
        ),
        changed_lengths=sum(p["executed_delta"] != 0 for p in selected),
        total_step_delta=sum(p["executed_delta"] for p in selected),
        max_abs_step_delta=max(abs(p["executed_delta"]) for p in selected),
        **{k: float(v[0]) for k, v in delta.items()},
      )
    )
    for k in delta:
      row = dict(mode=mode, metric=k)
      for label, stats in summary[mode].items():
        row[label], (row[label + "_ci95_low"], row[label + "_ci95_high"]) = (
          stats[k]["value"],
          stats[k]["ci95"],
        )
      rows.append(row)
  differences = {
    f"{b} minus {a}": {
      k: interval(v - pooled_stats[a][k]) for k, v in pooled_stats[b].items()
    }
    for a, b in (
      ("GT-Oracle", "GT-Estimated"),
      ("GT-Estimated", "Mean-in-Loop"),
      ("Mean-in-Loop", "Belief-in-Loop"),
    )
  }
  write_csv(folder / "tracking.csv", rows)
  write_csv(
    folder / "pooled-tracking.csv",
    [
      dict(mode=mode, **{metric: float(sample[0]) for metric, sample in stats.items()})
      for mode, stats in pooled_stats.items()
    ],
  )
  write_csv(folder / "repeat-differences.csv", changes)
  atomic_json(
    folder / "report.json",
    dict(
      design="Two repeated runs of identical clip/seed scenarios",
      identity=manifest["identity"],
      first=str(baseline),
      repeat=str(repeat),
      layout={k: layouts[0][k] for k in layout_keys},
      tracking_records=2 * len(pairs),
      unique_mode_clip_seed=len(pairs),
      runs_per_scenario=2,
      bootstrap_unit="whole clip, all modes, seeds and both runs",
      bootstrap_samples=len(weights) - 1,
      modes=summary,
      pooled_paired_differences=differences,
    ),
  )
  plot_tracking(folder, summary, len(manifest["motions"]), len(weights) - 1)
  print(f"{len(pairs)} matched pairs: {folder / 'tracking.csv'}", flush=True)


def compare_controls(off, on):
  """Compare the small same-layout recording switch experiment frame by frame."""
  off, on = Path(off), Path(on)
  manifest = load_manifest(off)
  if load_manifest(on)["identity"] != manifest["identity"]:
    raise ValueError("Control audit requires paired manifests")
  if json.loads((off / "layout.json").read_text()) != json.loads(
    (on / "layout.json").read_text()
  ):
    raise ValueError("Control audit requires paired layouts")
  plans = {m["name"]: m["steps"] for m in manifest["motions"]}
  rows = []
  for group in scenario_groups(manifest):
    noise_by_mode = []
    for s in group:
      tracks = [
        read_episode(s.path(run), manifest["identity"], plans[s.clip])
        for run in (off, on)
      ]
      off_track, on_track = tracks
      if off_track is None or on_track is None:
        raise ValueError(f"Missing valid tracking for {s}")
      (status_a, a, terms_a), (status_b, b, terms_b) = off_track, on_track
      n = min(len(a), len(b))
      act_a, noise_a = read_controls(s.path(off / "controls"), a)
      act_b, noise_b = read_controls(s.path(on / "controls"), b)
      action_delta = np.abs(act_a[:n] - act_b[:n])
      noise_by_mode.append((noise_a, noise_b))
      rows.append(
        dict(
          mode=s.mode,
          clip=s.clip,
          seed=s.seed,
          off_status=status_a,
          on_status=status_b,
          same_failure_terms=terms_a == terms_b,
          off_steps=len(a),
          on_steps=len(b),
          same_noise_prefix=np.array_equal(noise_a[:n], noise_b[:n]),
          first_action_max_abs=float(action_delta[0].max()),
          all_actions_max_abs=float(action_delta.max()),
          first_different_action=next(
            (int(i) for i in np.flatnonzero(action_delta.any(axis=1))), None
          ),
          frame_metric_max_abs=dict(
            zip(FRAME_COLUMNS, np.abs(a[:n] - b[:n]).max(axis=0).tolist(), strict=True)
          ),
        )
      )
    for run_no in (0, 1):
      streams = [p[run_no] for p in noise_by_mode]
      longest = max(streams, key=len)
      if any(not np.array_equal(v, longest[: len(v)]) for v in streams):
        raise ValueError("Noise differs between modes of the same scenario")
  atomic_json(
    on / "recording-comparison.json",
    dict(shared_noise_across_modes=True, episodes=rows),
  )
  return rows
