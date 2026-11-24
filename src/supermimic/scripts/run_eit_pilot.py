"""Run the Step 6 single-seed walking pilot with an auditable external budget.

Each training/evaluation/diagnostic command runs in a separate process. Repeating
the command resumes cumulative physical steps and reuses digest-checked validation tasks.
The test split is reserved and is never evaluated by this pilot.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import torch
from filelock import FileLock

from supermimic.runners.eit_resume_contract import (
  REFERENCE_VERSION,
  resume_execution_matches,
)
from supermimic.runners.eit_snapshots import file_sha256
from supermimic.runners.run_config import (
  validate_eit_checkpoint_contract,
)
from supermimic.scripts.pilot_report_diagnostics import diagnostic_structure_errors


def source_audit():
  """Record only repository source identity, without git/user configuration."""
  root = Path(__file__).resolve().parents[3]
  commit = subprocess.run(
    ["git", "rev-parse", "HEAD"],
    cwd=root,
    capture_output=True,
    text=True,
    check=False,
  )
  return {
    "source_commit": commit.stdout.strip() if commit.returncode == 0 else None,
    "source_files_sha256": {
      str(path.relative_to(root)): file_sha256(path)
      for path in sorted((root / "src" / "supermimic").rglob("*.py"))
    },
    "torch_version": str(torch.__version__),
  }


def write_json(path, value):
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
  temporary.replace(path)


def resume_result_matches(
  path,
  checkpoint,
  group,
  collected_steps,
  device,
  reference=None,
  process_environment=None,
  *,
  expected_identity=None,
):
  if reference is None or not path.is_file() or not Path(reference).is_file():
    return False
  try:
    result = json.loads(path.read_text())
    live = torch.load(reference, weights_only=True, mmap=True, map_location="cpu")
    identity = live.get("checkpoint_sha256")
    if (
      not isinstance(identity, str)
      or len(identity) != 64
      or any(c not in "0123456789abcdef" for c in identity)
      or (checkpoint is not None and file_sha256(checkpoint) != identity)
      or live.get("version") != REFERENCE_VERSION
      or live.get("source") != "live_training_before_checkpoint_commit"
      or live.get("group") != group
      or live.get("collected_steps") != collected_steps
      or any(live.get(k) != v for k, v in (expected_identity or {}).items())
    ):
      return False
    return resume_execution_matches(result, live, process_environment) and all(
      type(result.get(key)) is type(expected) and result[key] == expected
      for key, expected in {
        "version": "eit_fixed_input_resume_v2",
        "status": "passed",
        "checkpoint_sha256": identity,
        "group": group,
        "collected_steps": collected_steps,
        "device": device,
        "fixed_input_policy_resume_exact": True,
        "fixed_input_model_resume_exact": True,
        "live_optimizer_reference": True,
        "expected_source": "live_training_before_checkpoint_commit",
        "reference_version": live["version"],
        "reference_sha256": file_sha256(Path(reference)),
      }.items()
    )
  except (
    OSError,
    ValueError,
    AttributeError,
    RuntimeError,
    TypeError,
    EOFError,
    pickle.UnpicklingError,
  ):
    return False


def checkpoint_progress(saved, p, group):
  validate_eit_checkpoint_contract(saved, stage="policy")
  if (
    saved["runner_config"]["estimator_inputs"]["group"] != group
    or saved.get("training_seed") != p["training_seed"]
  ):
    raise ValueError("Existing pilot checkpoint group or training seed differs")
  steps = saved["collected_steps"]
  if type(steps) is not int or not 0 <= steps <= p["max_iterations"]:
    raise ValueError("Existing pilot checkpoint exceeds cumulative step budget")
  if saved["collected_frames"] != steps * p["num_envs"]:
    raise ValueError("Existing pilot checkpoint has a different physical-frame budget")
  return steps


def protocol(motion, seed, device, online_device):
  return {
    "version": "eit_pilot_v4",
    "training_process_environment": {
      "PYTORCH_JIT": "0",
      **(
        {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
        if torch.device(device).type == "cuda"
        else {}
      ),
    },
    "motion_file": str(motion.resolve()),
    "motion_sha256": file_sha256(motion),
    "training_seed": seed,
    "device": device,
    "online_device": online_device,
    "cpu_threads": 1,
    "max_iterations": 100,
    "resume_after_step": 50,
    "num_envs": 1024,
    "replay_ratio": 16,
    "save_interval": 1000,
    "log_interval": 10,
    "buffer_size": 1000,
    "mean": {"num_batches": 10, "batch_size": 2048, "window_frames": 8},
    "covariance": {"batch_size": 8192, "pair_budget": None, "validation_pairs": 8192},
    "policy": {
      "num_batches": 1,
      "batch_size": 4096,
      "num_samples": 1,
      "horizon": 32,
      "microbatch_size": 16,
      "critic_epochs": 5,
      "critic_minibatches": 4,
    },
    "widths": {
      "mean": [2048, 2048, 2048],
      "noise": [128, 64],
      "actor": [2048, 1024, 512, 256],
      "critic": [256, 128],
    },
    "sampling_mode": "return",
    "evaluation_protocol": "full_clip",
    "validation_seed": 17031,
    "reserved_test_seed": 17032,
    "diagnostic_seed": 4795,
    "evaluation_envs": 2,
    "expected_physical_frames": 102400,
    "updates_per_training_group": {
      "mean": 10,
      "covariance": 18,
      "actor": 1,
      "critic": 20,
    },
    "covariance_pair_budget": 143360,
    "covariance_rule": "sample with replacement; mean updates × batch × (window - 1)",
    "warmup_policy": "discard scheduled opportunities until replay and posterior are ready; repeat after resume",
  }


def training_arguments(p):
  args = [
    "--motion-file",
    p["motion_file"],
    "--device",
    p["device"],
    "--seed",
    str(p["training_seed"]),
  ]
  for flag in (
    "num_envs",
    "buffer_size",
    "replay_ratio",
    "save_interval",
    "log_interval",
    "sampling_mode",
  ):
    args += ["--" + flag.replace("_", "-"), str(p[flag])]
  for flag, value in {
    "mean-updates": p["mean"]["num_batches"],
    "mean-batch-size": p["mean"]["batch_size"],
    "mean-window-frames": p["mean"]["window_frames"],
    "cov-batch-size": p["covariance"]["batch_size"],
    "cov-validation-pairs": p["covariance"]["validation_pairs"],
    "policy-updates": p["policy"]["num_batches"],
    "policy-batch-size": p["policy"]["batch_size"],
    "samples": p["policy"]["num_samples"],
    "policy-horizon": p["policy"]["horizon"],
    "policy-microbatch-size": p["policy"]["microbatch_size"],
    "critic-epochs": p["policy"]["critic_epochs"],
    "critic-minibatches": p["policy"]["critic_minibatches"],
  }.items():
    args += ["--" + flag, str(value)]
  if p["covariance"]["pair_budget"] is not None:
    args += ["--cov-pair-budget", str(p["covariance"]["pair_budget"])]
  for flag, name in (
    ("hidden-dims", "mean"),
    ("actor-hidden-dims", "actor"),
    ("critic-hidden-dims", "critic"),
  ):
    args += ["--" + flag, *map(str, p["widths"][name])]
  return args


def diagnostic_cache_valid(arguments):
  """Reuse complete scores only for the exact snapshot, recording and requested mode."""
  args = list(map(str, arguments))
  supported = {
    "--output",
    "--checkpoint",
    "--recording",
    "--device",
    "--mode",
    "--actor-gradients",
    "--gradient-seed",
    "--gradient-episodes",
  }
  if any(arg.startswith("--") and arg not in supported for arg in args):
    return False

  def value(flag: str, default: str = "") -> str:
    return args[args.index(flag) + 1] if flag in args else default

  try:
    result = json.loads(Path(value("--output")).read_text())
    mode = value("--mode", "all")
    if diagnostic_structure_errors(
      result, mode=mode, require_gradients="--actor-gradients" in args
    ):
      return False
    return (
      result.get("version") == "eit_independent_diagnostics_v2"
      and result.get("checkpoint_sha256") == file_sha256(Path(value("--checkpoint")))
      and result.get("recording_sha256") == file_sha256(Path(value("--recording")))
      and (
        "--actor-gradients" not in args
        or (
          result.get("actor_gradients", {}).get("seed")
          == int(value("--gradient-seed", "2028"))
          and result["actor_gradients"].get("starts")
          == min(result["episodes"], int(value("--gradient-episodes", "4")))
        )
      )
    )
  except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError):
    return False


class Pilot:
  def __init__(self, output, p):
    self.output, self.p = output, p
    self.env = {
      **os.environ,
      "OMP_NUM_THREADS": "1",
      "MKL_NUM_THREADS": "1",
      "MPLCONFIGDIR": str(output / "matplotlib-cache"),
    }

  def command(self, name, module, arguments):
    if module == "diagnose_eit" and diagnostic_cache_valid(arguments):
      print(f"PILOT cached {name}", flush=True)
      return
    audit = source_audit()
    command = [
      sys.executable,
      "-m",
      "supermimic.scripts." + module,
      *map(str, arguments),
    ]
    log = self.output / "commands" / (name + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"PILOT {name}: {log}", flush=True)
    start = time.perf_counter()
    overrides = (
      self.p.get("training_process_environment", {})
      if module in ("train_eit", "check_eit_resume")
      else {}
    )
    with log.open("a") as stream:
      stream.write(json.dumps(command) + "\n")
      stream.flush()
      result = subprocess.run(
        command, env={**self.env, **overrides}, stdout=stream, stderr=stream
      )
    with (self.output / "commands.jsonl").open("a") as stream:
      stream.write(
        json.dumps(
          {
            "name": name,
            "argv": command,
            "returncode": result.returncode,
            "seconds": time.perf_counter() - start,
            "log": str(log),
            "training_process_environment": overrides,
            **audit,
          }
        )
        + "\n"
      )
    if result.returncode:
      raise RuntimeError(f"{name} failed; see {log}")

  def record(
    self,
    name,
    checkpoint,
    folder,
    *,
    truth_actor=False,
    reference_controller=False,
    steps=None,
  ):
    """Reuse only the same checkpoint, scene and behavior-policy recording."""
    p = self.p
    result, record = folder / "evaluation.json", folder / "episodes.pt"
    expected_source = (
      "reference_joint_positions"
      if reference_controller
      else "truth"
      if truth_actor
      else "posterior"
    )
    digest = file_sha256(checkpoint)
    matches = False
    if result.exists() and record.exists():
      try:
        scored = json.loads(result.read_text())
        recorded = torch.load(record, weights_only=True, mmap=True, map_location="cpu")
        matches = (
          all(
            scored.get(key) == value
            for key, value in {
              "checkpoint_sha256": digest,
              "evaluation_motion_sha256": p["motion_sha256"],
              "seed": p["diagnostic_seed"],
              "num_envs": p["evaluation_envs"],
              "actor_state_source": expected_source,
              "step_budget": steps,
              "time_limit_protocol": "full_clip_v1",
            }.items()
          )
          and all(
            recorded.get(key) == value
            for key, value in {
              "checkpoint_sha256": digest,
              "motion_sha256": p["motion_sha256"],
              "seed": p["diagnostic_seed"],
              "actor_state_source": expected_source,
              "version": "eit_diagnostic_episodes_v1",
              "source": "mujoco_evaluation",
              "split": "diagnostic",
            }.items()
          )
          and len(recorded.get("episodes", [])) == p["evaluation_envs"]
        )
      except (
        OSError,
        ValueError,
        RuntimeError,
        EOFError,
        pickle.UnpicklingError,
        AttributeError,
      ):
        matches = False
    if not matches:
      options = []
      if truth_actor:
        options += ["--truth-actor"]
      if reference_controller:
        options += ["--reference-controller"]
      if steps is not None:
        options += ["--steps", steps]
      self.command(
        name,
        "eval_eit",
        [
          "--checkpoint",
          checkpoint,
          "--motion-file",
          p["motion_file"],
          "--device",
          p["device"],
          "--num-envs",
          p["evaluation_envs"],
          "--seed",
          p["diagnostic_seed"],
          "--posterior-bin-steps",
          "50",
          "--record",
          record,
          "--output",
          result,
          *options,
        ],
      )
    return record

  def train(self):
    from supermimic.runners.eit_training import EITTraining

    p, output = self.p, self.output
    common = training_arguments(p)
    initial = output / "initialization.pt"
    self.command(
      "initialization",
      "train_eit",
      [
        *common,
        "--group",
        "B",
        "--prepare-initialization",
        "--initialization",
        initial,
      ],
    )
    for group in "ABC":
      run = output / group
      checkpoint = run / "training.pt"
      reference = output / "resume" / f"{group}-reference.pt"
      for target in (p["resume_after_step"], p["max_iterations"]):
        completed = 0
        if checkpoint.exists():
          recovery = EITTraining.reconcile_output(
            run,
            validate_state=lambda state, group=group: checkpoint_progress(
              state, p, group
            ),
          )
          write_json(output / "recovery" / f"{group}.json", recovery)
          completed = recovery["collected_steps"]
        if completed < target:
          options = (
            ["--resume", checkpoint]
            if checkpoint.exists()
            else ["--initialization", initial]
          )
          if target == p["resume_after_step"]:
            options += ["--resume-reference", reference]
          self.command(
            f"train-{group}-to-{target}",
            "train_eit",
            [
              *common,
              "--group",
              group,
              "--output",
              run,
              "--max-iterations",
              target,
              *options,
            ],
          )
          completed = target
        if target == p["resume_after_step"]:
          self.verify_midpoint(group, checkpoint, reference, completed)

  def verify_midpoint(self, group, checkpoint, reference, completed):
    p = self.p
    step = p["resume_after_step"]
    if not reference.is_file():
      raise ValueError(
        "Missing live reference cannot be reconstructed from a committed checkpoint"
      )
    live = torch.load(reference, map_location="cpu", weights_only=True, mmap=True)
    current = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    identity = {k: current[k] for k in ("motion_digest", "initialization_digest")}
    if any(live.get(k) != v for k, v in identity.items()):
      raise ValueError(
        "Midpoint reference belongs to a different pilot initialization or motion"
      )
    midpoint = reference.with_name(f"{group}-midpoint.pt")
    receipt = reference.with_name(f"{group}.json")
    if (
      not midpoint.exists()
      and completed == step
      and file_sha256(checkpoint) == live["checkpoint_sha256"]
    ):
      # training.pt is atomically replaced on restart. Pin its inode before
      # the second launch so retries can still verify the original live update.
      os.link(checkpoint, midpoint)
    evidence = midpoint if midpoint.is_file() else None

    def verified():
      return resume_result_matches(
        receipt,
        evidence,
        group,
        step,
        p["device"],
        reference,
        p.get("training_process_environment"),
        expected_identity=identity,
      )

    if verified():
      # Existing pilots can already have a verified receipt and reference while
      # their original midpoint bytes have been replaced by a training launch.
      return
    if evidence is None:
      raise ValueError(
        "Missing verified midpoint: recover the checkpoint matching the live reference before continuing"
      )
    self.command(
      f"resume-{group}",
      "check_eit_resume",
      [
        "--checkpoint",
        evidence,
        "--motion-file",
        p["motion_file"],
        "--device",
        p["device"],
        "--reference",
        reference,
        "--output",
        receipt,
      ],
    )
    if not verified():
      raise ValueError(
        f"{group}: midpoint verification did not produce matching evidence"
      )

  def evaluate(self):
    p, output = self.p, self.output
    frames = p["max_iterations"] * p["num_envs"]
    spec = {
      "design": "seen",
      "device": p["device"],
      "training": [{"path": p["motion_file"], "subject": "subject1"}],
      "runs": [
        {
          "group": g,
          "training_seed": p["training_seed"],
          "directory": str(output / g),
          "frames": [frames],
        }
        for g in "ABC"
      ],
    }
    for split, seed in (
      ("validation", p["validation_seed"]),
      ("test", p["reserved_test_seed"]),
    ):
      spec[split] = [
        {
          "path": p["motion_file"],
          "subject": "subject1",
          "seed": seed,
          "num_envs": p["evaluation_envs"],
        }
      ]
    write_json(output / "study.json", spec)
    self.command(
      "freeze",
      "batch_eval_eit",
      [
        "freeze",
        "--spec",
        output / "study.json",
        "--manifest",
        output / "manifest.json",
      ],
    )
    self.command(
      "validation",
      "batch_eval_eit",
      [
        "run",
        "--manifest",
        output / "manifest.json",
        "--output",
        output / "evaluation",
        "--split",
        "validation",
      ],
    )
    for group in "ABC":
      checkpoint = output / group / "training.pt"
      folder = output / "diagnostics" / group / str(frames)
      record = self.record(f"record-{group}-{frames}", checkpoint, folder)
      self.command(
        f"diagnose-{group}-{frames}",
        "diagnose_eit",
        [
          "--checkpoint",
          checkpoint,
          "--recording",
          record,
          "--device",
          p["device"],
          "--output",
          folder / "offline.json",
          "--actor-gradients",
        ],
      )
      self.command(
        f"online-{group}",
        "benchmark_eit_online",
        [
          "--checkpoint",
          checkpoint,
          "--motion-file",
          p["motion_file"],
          "--device",
          p["online_device"],
          "--output",
          output / "online" / f"{group}.json",
        ],
      )

    # Truth-input and shared-record diagnostics remain distinct from validation.
    checkpoint = output / "A" / "training.pt"
    folder = output / "diagnostics" / "A" / f"{frames}-truth"
    record = self.record("record-A-truth", checkpoint, folder, truth_actor=True)
    self.command(
      "diagnose-A-truth",
      "diagnose_eit",
      [
        "--checkpoint",
        checkpoint,
        "--recording",
        record,
        "--device",
        p["device"],
        "--mode",
        "critic",
        "--output",
        folder / "offline.json",
      ],
    )
    reference_folder = output / "diagnostics" / "common-walking-reference"
    reference = self.record(
      "record-common-reference",
      output / "B" / "training.pt",
      reference_folder,
      reference_controller=True,
      steps=250,
    )
    records = {
      "policy-final": output / "diagnostics" / "B" / str(frames) / "episodes.pt",
      "walking-reference": reference,
    }
    write_json(
      output / "diagnostics" / "common-recordings.json",
      {
        "scope": "fixed_inputs_for_process_and_posterior_only_not_policy_performance",
        "motion_sha256": p["motion_sha256"],
        "recordings": {name: str(path) for name, path in records.items()},
      },
    )
    for group in "ABC":
      for name, record in records.items():
        for mode in ("process", "posterior"):
          self.command(
            f"common-{name}-{group}-{mode}",
            "diagnose_eit",
            [
              "--checkpoint",
              output / group / "training.pt",
              "--recording",
              record,
              "--device",
              p["device"],
              "--mode",
              mode,
              "--output",
              output / "diagnostics" / "common" / name / f"{group}-{mode}.json",
            ],
          )


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--motion-file", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--seed", type=int, default=2026)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--online-device", default="cpu")
  parser.add_argument("--phase", choices=("all", "train", "evaluate"), default="all")
  args = parser.parse_args()
  output = args.output.resolve()
  output.mkdir(parents=True, exist_ok=True)
  p = protocol(args.motion_file, args.seed, args.device, args.online_device)
  if args.seed in (p["validation_seed"], p["reserved_test_seed"], p["diagnostic_seed"]):
    parser.error("training seed must be independent of evaluation seeds")
  with FileLock(str(output / ".pilot.lock"), timeout=0):
    path = output / "protocol.json"
    if path.exists():
      previous = json.loads(path.read_text())
      if previous != p:
        raise ValueError("Pilot configuration changed; use a new output directory")
    else:
      write_json(path, p)
    pilot = Pilot(output, p)
    try:
      if args.phase in ("all", "train"):
        pilot.train()
      if args.phase in ("all", "evaluate"):
        pilot.evaluate()
    except Exception as error:
      write_json(
        output / "status.json",
        {"status": "failed", "phase": args.phase, "error": str(error)},
      )
      raise
    write_json(output / "status.json", {"status": "completed", "phase": args.phase})


if __name__ == "__main__":
  main()
