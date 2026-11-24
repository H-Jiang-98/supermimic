"""Freeze EIT evaluation scenarios, run independent first episodes and rank snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, is_dataclass
from pathlib import Path
from statistics import fmean

import torch
from filelock import FileLock

from supermimic.belief import GaussianStatePrior
from supermimic.estimator_inputs import GaussianMeasurementConfig
from supermimic.motion_loader import MotionLoader, motion_content_digest
from supermimic.runners.eit_inference import load_hashed_checkpoint
from supermimic.runners.eit_snapshots import file_sha256
from supermimic.runners.run_config import validate_eit_checkpoint_contract

VERSION = "eit_batch_full_clip_v3"
TRACKING = ("mpkpe", "r_mpkpe", "joint_vel_error", "ee_pos_error", "ee_ori_error")
SCORES = ("success_rate", "progress", "return", "length", *TRACKING)


def digest(value):
  return hashlib.sha256(
    json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
  ).hexdigest()


def read_json(path):
  return json.loads(Path(path).read_text())


def write_json(path, data):
  path.parent.mkdir(parents=True, exist_ok=True)
  with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
    tmp = Path(stream.name)
    try:
      json.dump(data, stream, sort_keys=True, indent=2, allow_nan=False)
      stream.write("\n")
      stream.flush()
      tmp.replace(path)
    finally:
      tmp.unlink(missing_ok=True)


def cfg_json(value):
  """Canonical JSON for mjlab configs, including selectors and functions."""
  if is_dataclass(value) and not isinstance(value, type):
    return cfg_json(asdict(value))
  if callable(value):
    return f"{value.__module__}.{value.__qualname__}"
  if isinstance(value, dict):
    return {str(k): cfg_json(v) for k, v in value.items()}
  if isinstance(value, (tuple, list)):
    return [cfg_json(v) for v in value]
  if isinstance(value, slice):
    return {"slice": [value.start, value.stop, value.step]}
  return value


def conditions(cfg):
  """Freeze the environment, excluding display and separately recorded identity."""
  settings = asdict(cfg)
  settings.pop("viewer")
  settings.pop("seed")
  settings["scene"].pop("num_envs")
  # Manager term order determines concatenation and execution order. Preserve it
  # explicitly because JSON object order is ignored by equality and digest().
  settings["manager_term_order"] = {
    name: list(settings[name])
    for name in (
      "observations",
      "actions",
      "events",
      "rewards",
      "terminations",
      "commands",
      "curriculum",
      "metrics",
      "recorders",
    )
  }
  settings["observation_term_order"] = {
    name: list(group["terms"]) for name, group in settings["observations"].items()
  }
  return cfg_json(settings)


def probe(path, device):
  from supermimic.motion_command import MotionCommand
  from supermimic.scripts.eval_eit import evaluation_env_config
  from supermimic.scripts.train_utils import create_env

  cfg = evaluation_env_config(path, 0, "full_clip")
  settings = conditions(cfg)
  with torch.compiler.set_stance("force_eager"):
    env = create_env(cfg, 1, device)
    try:
      raw = env.unwrapped
      command = raw.command_manager.get_term("motion")
      if not isinstance(command, MotionCommand) or not isinstance(
        command.motion, MotionLoader
      ):
        raise ValueError("Expected the SuperMimic tracking motion command")
      motion = command.motion
      return {
        "evaluation_motion_digest": motion_content_digest(motion),
        "clip_frames": motion.time_step_total,
        "control_period_s": raw.step_dt,
        "timeout_steps": int(raw.max_episode_length),
        "timeout_s": cfg.episode_length_s,
        "conditions": settings,
      }
    finally:
      env.close()


def freeze(spec, base):
  design, device = spec["design"], spec["device"]
  if design not in {"seen", "heldout_clip", "heldout_subject"}:
    raise ValueError("Choose seen, heldout_clip or heldout_subject")
  runs = []
  for run in spec["runs"]:
    frames = run["frames"]
    if (
      not frames
      or any(type(n) is not int or n < 0 for n in frames)
      or len(set(frames)) != len(frames)
      or run["group"] not in ("A", "B", "C")
      or type(run["training_seed"]) is not int
    ):
      raise ValueError("Invalid run group, training seed or candidate frame counts")
    runs.append({**run, "directory": str((base / run["directory"]).resolve())})
  if not runs or len({(r["group"], r["training_seed"]) for r in runs}) != len(runs):
    raise ValueError("Specify one run per group and training seed")
  training_seeds = {r["training_seed"] for r in runs}
  splits, subjects, file_hashes, probes = {}, {}, {}, {}
  prior = cfg_json(GaussianStatePrior(**spec.get("initial_prior", {})))
  noise = cfg_json(GaussianMeasurementConfig(**spec.get("measurement_noise", {})))
  for split in ("training", "validation", "test"):
    rows = []
    for entry in spec[split]:
      fields = (
        {"path", "subject"}
        if split == "training"
        else {"path", "subject", "seed", "num_envs"}
      )
      if set(entry) != fields:
        raise ValueError(f"{split} scenario requires exactly {sorted(fields)}")
      path = (base / entry["path"]).resolve(strict=True)
      if path not in file_hashes:
        file_hashes[path] = file_sha256(path)
      if path not in probes:
        probes[path] = probe(path, device)
      clip = probes[path]["evaluation_motion_digest"]
      subject = entry["subject"]
      if not subject or subjects.setdefault(clip, subject) != subject:
        raise ValueError("Conflicting or missing subject for a motion")
      row = {
        "path": str(path),
        "sha256": file_hashes[path],
        "evaluation_motion_digest": clip,
        "subject": subject,
      }
      if split != "training":
        seed, count = entry["seed"], entry["num_envs"]
        if type(seed) is not int or seed in training_seeds:
          raise ValueError("Evaluation seeds must be independent of training seeds")
        if type(count) is not int or count < 1:
          raise ValueError("num_envs must be positive")
        row.update(
          probes[path],
          seed=seed,
          num_envs=count,
          episodes=count,
          protocol="full_clip_v1",
          step_budget=None,
          device=device,
          initial_prior=prior,
          measurement_noise=noise,
        )
      rows.append(row)
    keys = [(r["evaluation_motion_digest"], r.get("seed")) for r in rows]
    if not rows or len(keys) != len(set(keys)):
      raise ValueError(f"Empty or duplicate {split} scenarios")
    splits[split] = rows

  def identities(split, field):
    return {r[field] for r in splits[split]}

  train, val, test = (identities(s, "evaluation_motion_digest") for s in splits)
  if design == "seen":
    if not (val | test) <= train:
      raise ValueError("Seen-motion scenarios must use training clips")
    if identities("validation", "seed") & identities("test", "seed"):
      raise ValueError("Validation and test require independent seeds for seen motion")
  elif train & (val | test) or val & test:
    raise ValueError("Held-out clip splits overlap")
  if design == "heldout_subject":
    a, b, c = (identities(s, "subject") for s in splits)
    if a & (b | c) or b & c:
      raise ValueError("Held-out subject splits overlap")
  manifest = dict(version=VERSION, design=design, runs=runs, **splits)
  return {**manifest, "manifest_digest": digest(manifest)}


def admit(state, run, index, training):
  validate_eit_checkpoint_contract(state, stage="policy", for_inference=True)
  counters = {
    "mean_steps": state["learner"]["mean"]["steps"],
    "cov_steps": state["learner"]["cov"]["steps"],
    "actor_steps": state["policy"]["actor_steps"],
    "critic_steps": state["policy"]["critic_steps"],
  }
  group = state["runner_config"]["estimator_inputs"]["group"]
  init = state.get("initialization_digest")
  if group != run["group"] or state.get("training_seed") != run["training_seed"]:
    raise ValueError("Snapshot group or training seed mismatch")
  if (
    not isinstance(init, str)
    or len(init) != 64
    or any(c not in "0123456789abcdef" for c in init)
  ):
    raise ValueError("Snapshot requires a common initialization fingerprint")
  if [f["sha256"] for f in state["training_motion_files"]] != [
    f["sha256"] for f in training
  ]:
    raise ValueError("Actual expanded training motion files differ from manifest")
  round_num, frames = state["update_groups"], state["collected_frames"]
  if (
    type(round_num) is not int or round_num < 0 or type(frames) is not int or frames < 0
  ):
    raise ValueError("Invalid checkpoint update groups or frames")
  if frames != state["collected_steps"] * state["num_envs"] or any(
    type(n) is not int or n < 0 or (n == 0) != (round_num == 0)
    for n in counters.values()
  ):
    raise ValueError("Checkpoint counters disagree with actual optimization steps")
  optimizers = [
    *((opt, counters["mean_steps"]) for opt in state["learner"]["mean"]["optimizers"]),
    (state["learner"]["cov"]["optimizer"], counters["cov_steps"]),
    *((opt, counters["actor_steps"]) for opt in state["policy"]["actor_optimizers"]),
    *((opt, counters["critic_steps"]) for opt in state["policy"]["critic_optimizers"]),
  ]
  for optimizer, steps in optimizers:
    # Muon has momentum buffers only. Adam saves a step counter for every parameter.
    moments = optimizer["state"]
    param_ids = {p for group in optimizer["param_groups"] for p in group["params"]}
    if set(moments) != (param_ids if steps else set()) or any(
      "step" in m and float(m["step"]) != steps for m in moments.values()
    ):
      raise ValueError("Actual optimizer state differs from checkpoint update count")
  metadata = dict(
    group=group,
    update_groups=round_num,
    collected_steps=state["collected_steps"],
    collected_frames=frames,
    stage="policy",
    initialization_digest=init,
    model_version=state["learner"]["model_version"],
    actor_version=state["actor_version"],
    critic_version=state["critic_version"],
    training_seed=state["training_seed"],
    **counters,
  )
  if index is not None and any(index.get(k) != v for k, v in metadata.items()):
    raise ValueError("Snapshot index metadata differs from checkpoint contents")
  return dict(
    **metadata,
    training_motion_digest=state["motion_digest"],
    training_motion_files=state["training_motion_files"],
    initial_prior=cfg_json(state["runner_config"]["initial_prior"]),
    measurement_noise=cfg_json(
      state["runner_config"]["estimator_inputs"]["measurement_noise"]
    ),
  )


def candidates(manifest):
  output = []
  init_by_seed = defaultdict(set)
  for run in manifest["runs"]:
    directory = Path(run["directory"])
    for frames in run["frames"]:
      record = dict(
        group=run["group"], training_seed=run["training_seed"], collected_frames=frames
      )
      try:
        # The trainer overwrites current weights. Historical candidates must be
        # explicit immutable checkpoint paths (for example downloaded artifacts).
        path = directory / run.get("checkpoints", {}).get(str(frames), "training.pt")
        state, identity = load_hashed_checkpoint(path, "cpu")
        if state["collected_frames"] != frames:
          raise ValueError(
            "Requested frame count is unavailable in this checkpoint; specify an explicit checkpoints path"
          )
        meta = admit(state, run, None, manifest["training"])
        record.update(meta, path=str(path), sha256=identity)
        init_by_seed[run["training_seed"]].add(meta["initialization_digest"])
        del state
      except (
        OSError,
        RuntimeError,
        ValueError,
        KeyError,
        TypeError,
        pickle.UnpicklingError,
        EOFError,
      ) as e:
        record["error"] = str(e)
      output.append(record)
  for record in output:
    if len(init_by_seed[record["training_seed"]]) > 1:
      record["error"] = "Common initialization fingerprints differ within training seed"
  return output


def task_key(candidate, scene):
  return digest([VERSION, candidate["sha256"], scene])


def checked_episodes(result, candidate, scene):
  expected = {
    k: scene[k]
    for k in (
      "evaluation_motion_digest",
      "clip_frames",
      "control_period_s",
      "timeout_steps",
      "timeout_s",
      "device",
      "num_envs",
      "seed",
      "protocol",
      "step_budget",
    )
  }
  expected.update(
    group=candidate["group"],
    checkpoint_sha256=candidate["sha256"],
    training_motion_digest=candidate["training_motion_digest"],
    evaluation_motion_sha256=scene["sha256"],
    actor_state_source="posterior",
    actor_backend="pytorch",
    mode="eval",
    time_limit_protocol=scene["protocol"],
  )
  if any(result.get(k) != v for k, v in expected.items()):
    raise ValueError(
      "Evaluation identity, protocol or timing differs from frozen scenario"
    )
  rows = result["episodes"]
  if len(rows) != scene["episodes"] or {r["env_id"] for r in rows} != set(
    range(len(rows))
  ):
    raise ValueError("Missing or duplicate episode slots")
  for row in rows:
    length, reference = row["length"], row["reference_steps"]
    reason = row["reason"]
    censored = reason == "evaluation_truncated"
    if (
      reason not in ("clip_complete", "failure", "evaluation_truncated")
      or row["censored"] != censored
      or reference != scene["clip_frames"]
      or row["start_frame"] != 0
      or not 1 <= length <= reference
      or (reason == "clip_complete" and length != reference)
      or row["metric_steps"] != length - int(not censored)
      or not math.isfinite(row["return"])
    ):
      raise ValueError("Invalid first-episode termination or length")
    for key in TRACKING:
      value = row["metrics"][key]
      if (value is None) != (row["metric_steps"] == 0) or (
        value is not None and not math.isfinite(value)
      ):
        raise ValueError(f"Invalid episode mean {key}")
  return rows


def validate_scene(scene):
  """Reject changed inputs before running a task or reusing a cached episode."""
  from supermimic.scripts.eval_eit import evaluation_env_config

  GaussianMeasurementConfig.from_dict(scene["measurement_noise"])
  if file_sha256(Path(scene["path"])) != scene["sha256"]:
    raise ValueError("Raw motion file differs from frozen scenario")
  cfg = evaluation_env_config(Path(scene["path"]), scene["seed"], "full_clip")
  if conditions(cfg) != scene["conditions"]:
    raise ValueError(
      "Evaluation environment configuration differs from frozen scenario"
    )


def execute(candidate, scene, result_file, key):
  validate_scene(scene)
  if any(candidate[k] != scene[k] for k in ("initial_prior", "measurement_noise")):
    raise ValueError("Snapshot prior or measurement noise differs from frozen scenario")
  with tempfile.TemporaryDirectory(dir=result_file.parent) as tmp:
    raw = Path(tmp) / "raw.json"
    subprocess.run(
      [
        sys.executable,
        "-m",
        "supermimic.scripts.eval_eit",
        "--checkpoint",
        candidate["path"],
        "--motion-file",
        scene["path"],
        "--protocol",
        "full_clip",
        "--device",
        scene["device"],
        "--num-envs",
        str(scene["num_envs"]),
        "--seed",
        str(scene["seed"]),
        "--output",
        str(raw),
      ],
      check=True,
    )
    result = read_json(raw)
    checked_episodes(result, candidate, scene)
    write_json(
      result_file,
      {
        "task_key": key,
        "snapshot_sha256": candidate["sha256"],
        "scene": scene,
        "result": result,
      },
    )


def aggregate(clips):
  """Rows contain episode-level frame means. Give each scored clip one vote."""
  per_clip = {}
  for clip, rows in sorted(clips.items()):
    metrics = defaultdict(list)
    for row in rows:
      if not row["censored"]:
        metrics["success_rate"].append(float(row["reason"] == "clip_complete"))
      metrics["progress"].append(row["length"] / row["reference_steps"])
      metrics["length"].append(row["length"])
      metrics["return"].append(row["return"])
      for key, value in row["metrics"].items():
        if key in TRACKING and value is not None:
          metrics[key].append(value)
    per_clip[clip] = {
      "scores": {k: fmean(metrics[k]) if metrics[k] else None for k in SCORES},
      "valid_episodes": {k: len(metrics[k]) for k in SCORES},
      "reasons": dict(Counter(row["reason"] for row in rows)),
      "metric_frames": sum(row["metric_steps"] for row in rows),
    }
    if any(r["censored"] for r in rows):
      per_clip[clip]["scores"]["success_rate"] = None
  scores, counts = {}, {}
  for key in SCORES:
    valid = [
      v["scores"][key] for v in per_clip.values() if v["scores"][key] is not None
    ]
    scores[key] = fmean(valid) if valid else None
    counts[key] = len(valid)
  if counts["success_rate"] != len(per_clip):
    scores["success_rate"] = None
  return {
    "scores": scores,
    "valid_clips": counts,
    "clips": per_clip,
    "valid_episodes": {
      key: sum(v["valid_episodes"][key] for v in per_clip.values()) for key in SCORES
    },
  }


def load_episodes(path, key, candidate, scene):
  stored = read_json(path)
  if (
    stored["task_key"] != key
    or stored["snapshot_sha256"] != candidate["sha256"]
    or stored["scene"] != scene
  ):
    raise ValueError("Task result identity mismatch")
  return checked_episodes(stored["result"], candidate, scene)


def evaluate_candidate(candidate, scenes, output, run) -> dict:
  tasks, clips = [], defaultdict(list)
  for scene in scenes:
    task = dict(
      group=candidate["group"],
      training_seed=candidate["training_seed"],
      collected_frames=candidate["collected_frames"],
      motion=scene["path"],
      seed=scene["seed"],
      expected_episodes=scene["episodes"],
    )
    tasks.append(task)
    try:
      if "error" in candidate:
        raise ValueError(candidate["error"])
      key = task_key(candidate, scene)
      file = output / f"tasks/{key}.json"
      task.update(task_key=key, file=str(file))
      validate_scene(scene)
      failure_file = file.with_suffix(".failed.json")
      if run:
        file.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(file) + ".lock"):
          try:
            episodes = load_episodes(file, key, candidate, scene)
          except (FileNotFoundError, ValueError, KeyError, TypeError):
            try:
              execute(candidate, scene, file, key)
              episodes = load_episodes(file, key, candidate, scene)
            except (
              ValueError,
              OSError,
              RuntimeError,
              subprocess.CalledProcessError,
              KeyError,
              TypeError,
            ) as e:
              write_json(failure_file, {"task_key": key, "error": str(e)})
              raise
            failure_file.unlink(missing_ok=True)
      elif file.exists():
        episodes = load_episodes(file, key, candidate, scene)
      elif failure_file.exists():
        raise ValueError(read_json(failure_file)["error"])
      else:
        task["status"] = "missing"
        continue
      clips[scene["evaluation_motion_digest"]].extend(episodes)
      task["status"] = (
        "censored" if any(e["censored"] for e in episodes) else "complete"
      )
    except (
      OSError,
      RuntimeError,
      ValueError,
      KeyError,
      TypeError,
      subprocess.CalledProcessError,
    ) as e:
      task.update(status="failed", error=str(e))
  summary = aggregate(clips)
  complete = all(t["status"] == "complete" for t in tasks)
  if not complete:
    summary["scores"]["success_rate"] = None
  return dict(
    candidate=candidate,
    tasks=tasks,
    complete=complete,
    expected_clips=len({scene["evaluation_motion_digest"] for scene in scenes}),
    expected_episodes=sum(scene["episodes"] for scene in scenes),
    recorded_episodes=sum(len(rows) for rows in clips.values()),
    **summary,
  )


def select(records):
  groups = defaultdict(list)
  for row in records:
    c = row["candidate"]
    groups[(c["group"], c["training_seed"])].append(row)
  winners = []
  for _, group in sorted(groups.items()):
    if not all(r["complete"] for r in group):
      continue
    trained = [r for r in group if r["candidate"]["actor_steps"] > 0]
    if not trained:
      continue

    def priority(row):
      s = row["scores"]
      return (
        -s["success_rate"],
        -s["progress"],
        s["mpkpe"] if s["mpkpe"] is not None else math.inf,
        row["candidate"]["collected_frames"],
      )

    winners.append(min(trained, key=priority)["candidate"])
  return winners


def study(manifest, output, split, run):
  items = candidates(manifest)
  val = [
    evaluate_candidate(c, manifest["validation"], output, run and split == "validation")
    for c in items
  ]
  winners = select(val)
  records = (
    val
    if split == "validation"
    else [evaluate_candidate(c, manifest["test"], output, run) for c in winners]
  )
  pending = [t for r in val for t in r["tasks"] if t["status"] != "complete"]
  admitted = [r for r in records if "error" not in r["candidate"]]
  result = dict(
    manifest_digest=manifest["manifest_digest"],
    split=split,
    selected=winners,
    pending_validation=pending,
    baselines=[r for r in admitted if r["candidate"]["actor_steps"] == 0],
    trained=[r for r in admitted if r["candidate"]["actor_steps"] > 0],
    unavailable=[r for r in records if "error" in r["candidate"]],
  )
  result["complete"] = (
    not pending
    and all(r["complete"] for r in records)
    and (len(winners) == len(manifest["runs"]) if split == "test" else True)
  )
  write_json(output / f"{split}.json", result)
  return result


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  sub = parser.add_subparsers(dest="command", required=True)
  prepare = sub.add_parser("freeze")
  prepare.add_argument("--spec", required=True, type=Path)
  prepare.add_argument("--manifest", required=True, type=Path)
  for action in ("run", "report"):
    cmd = sub.add_parser(action)
    cmd.add_argument("--manifest", required=True, type=Path)
    cmd.add_argument("--output", required=True, type=Path)
    cmd.add_argument("--split", choices=("validation", "test"), required=True)
  args = parser.parse_args()
  if args.command == "freeze":
    frozen = freeze(read_json(args.spec), args.spec.resolve().parent)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(args.manifest) + ".lock"):
      if args.manifest.exists() and read_json(args.manifest) != frozen:
        parser.error("Manifest already exists with different contents")
      write_json(args.manifest, frozen)
  else:
    manifest = read_json(args.manifest)
    stored = manifest.pop("manifest_digest")
    if manifest["version"] != VERSION or stored != digest(manifest):
      parser.error("Invalid frozen manifest identity or version")
    manifest["manifest_digest"] = stored
    result = study(manifest, args.output.resolve(), args.split, args.command == "run")
    print(f"{args.split} report at {args.output / (args.split + '.json')}")
    if not result["complete"]:
      raise SystemExit(1)


if __name__ == "__main__":
  main()
