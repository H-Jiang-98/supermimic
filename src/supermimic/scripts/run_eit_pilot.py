"""Run the Step 6 single-seed walking pilot with an auditable external budget.

Each training/evaluation/diagnostic command runs in a separate process. Repeating
the command resumes completed rounds and reuses digest-checked validation tasks.
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

from supermimic.runners.eit_resume_contract import resume_execution_matches
from supermimic.runners.eit_snapshots import file_sha256
from supermimic.runners.run_config import validate_eit_checkpoint_contract
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
  path, checkpoint, group, rounds, device, reference=None, process_environment=None
):
  if reference is None or not path.is_file() or not Path(reference).is_file():
    return False
  try:
    result = json.loads(path.read_text())
    live = torch.load(reference, weights_only=True, mmap=True, map_location="cpu")
    return resume_execution_matches(result, live, process_environment) and all(
      result.get(key) == expected
      for key, expected in {
        "version": "eit_fixed_input_resume_v2",
        "status": "passed",
        "checkpoint_sha256": file_sha256(checkpoint),
        "group": group,
        "round": rounds,
        "device": device,
        "fixed_input_policy_resume_exact": True,
        "fixed_input_model_resume_exact": True,
        "live_optimizer_reference": True,
        "expected_source": "live_training_before_checkpoint_commit",
        "reference_version": "eit_live_update_reference_v1",
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
  if saved.get("snapshot_interval", 1) != 1:
    raise ValueError("Pilot requires a snapshot every round")
  if (
    saved["runner_config"]["estimator_inputs"]["group"] != group
    or saved.get("training_seed") != p["training_seed"]
  ):
    raise ValueError("Existing pilot checkpoint group or training seed differs")
  rounds = saved["rounds"]
  if type(rounds) is not int or not 0 <= rounds <= p["rounds"]:
    raise ValueError("Existing pilot checkpoint exceeds the configured round budget")
  per_round = p["num_envs"] * p["collection_steps"]
  if saved["collected_frames"] != rounds * per_round:
    raise ValueError("Existing pilot checkpoint has a different physical-frame budget")
  return rounds


def protocol(motion, seed, device, online_device):
  return {
    "version": "eit_pilot_v2",
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
    "rounds": 10,
    "resume_after_round": 5,
    "num_envs": 8,
    "collection_steps": 64,
    "buffer_size": 128,
    "mean": {"num_batches": 10, "batch_size": 2048, "window_frames": 8},
    "covariance": {"batch_size": 256, "epochs": 1},
    "policy": {
      "num_batches": 1,
      "batch_size": 4,
      "num_samples": 2,
      "horizon": 4,
      "microbatch_size": 2,
      "critic_epochs": 2,
    },
    "widths": {
      "mean": [2048, 2048, 2048],
      "noise": [128, 64],
      "actor": [2048, 1024, 512, 256],
      "critic": [256, 128],
    },
    "sampling_mode": "uniform",
    "evaluation_protocol": "full_clip",
    "validation_seed": 17031,
    "reserved_test_seed": 17032,
    "diagnostic_seed": 4795,
    "evaluation_envs": 2,
    "expected_per_group": {
      "collected_frames": 5120,
      "mean_updates": 100,
      "actor_updates": 10,
      "critic_updates": 20,
    },
    "covariance_rule": "epochs * ceil(valid_training_pairs / batch_size), each round",
    "continuous_collection_limit_s": 1.28,
  }


def training_arguments(p):
  return [
    "--motion-file",
    p["motion_file"],
    "--device",
    p["device"],
    "--seed",
    str(p["training_seed"]),
    "--num-envs",
    "8",
    "--collection-steps",
    "64",
    "--buffer-size",
    "128",
    "--snapshot-interval",
    "1",
    "--wandb-mode",
    "disabled",
    "--mean-updates",
    "10",
    "--mean-batch-size",
    "2048",
    "--mean-window-frames",
    "8",
    "--cov-batch-size",
    "256",
    "--cov-epochs",
    "1",
    "--policy-updates",
    "1",
    "--policy-batch-size",
    "4",
    "--samples",
    "2",
    "--policy-horizon",
    "4",
    "--policy-microbatch-size",
    "2",
    "--critic-epochs",
    "2",
    "--hidden-dims",
    "2048",
    "2048",
    "2048",
    "--actor-hidden-dims",
    "2048",
    "1024",
    "512",
    "256",
    "--critic-hidden-dims",
    "256",
    "128",
  ]


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
      result.get("version") == "eit_independent_diagnostics_v1"
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
      for target in (p["resume_after_round"], p["rounds"]):
        completed = 0
        if checkpoint.exists():
          recovery = EITTraining.reconcile_output(
            run,
            validate_state=lambda state, expected_group=group: checkpoint_progress(
              state, p, expected_group
            ),
          )
          write_json(output / "recovery" / f"{group}.json", recovery)
          completed = recovery["round"]
        if completed < target:
          options = (
            ["--resume", checkpoint]
            if checkpoint.exists()
            else ["--initialization", initial]
          )
          if target == p["resume_after_round"]:
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
              "--rounds",
              target - completed,
              *options,
            ],
          )
        if target == p["resume_after_round"]:
          fixed = run / "snapshots" / f"frames_{target * 512}.pt"
          result = output / "resume" / f"{group}.json"
          if not reference.is_file():
            raise ValueError(
              f"Missing live pre-commit update reference: {reference}. "
              "It cannot be reconstructed from a committed legacy checkpoint; "
              "preserve this run and start a new pilot output directory."
            )
          if not resume_result_matches(
            result,
            fixed,
            group,
            target,
            p["device"],
            reference,
            p.get("training_process_environment"),
          ):
            self.command(
              f"resume-{group}",
              "check_eit_resume",
              [
                "--checkpoint",
                fixed,
                "--motion-file",
                p["motion_file"],
                "--reference",
                reference,
                "--device",
                p["device"],
                "--output",
                result,
              ],
            )

  def evaluate(self):
    p, output = self.p, self.output
    spec = {
      "design": "seen",
      "device": p["device"],
      "training": [{"path": p["motion_file"], "subject": "subject1"}],
      "runs": [
        {
          "directory": str(output / group),
          "group": group,
          "training_seed": p["training_seed"],
          "frames": list(range(0, 5121, 512)),
        }
        for group in "ABC"
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
      for frames in range(0, 5121, 512):
        checkpoint = output / group / "snapshots" / f"frames_{frames}.pt"
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
            *(["--actor-gradients"] if frames == 5120 else []),
          ],
        )
      checkpoint = output / group / "snapshots" / "frames_5120.pt"
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
    checkpoint = output / "A" / "snapshots" / "frames_5120.pt"
    truth_folder = output / "diagnostics" / "A" / "5120-truth"
    truth_record = self.record(
      "record-A-truth", checkpoint, truth_folder, truth_actor=True
    )
    self.command(
      "diagnose-A-truth",
      "diagnose_eit",
      [
        "--checkpoint",
        checkpoint,
        "--recording",
        truth_record,
        "--device",
        p["device"],
        "--mode",
        "critic",
        "--output",
        truth_folder / "offline.json",
      ],
    )
    # This reference-controller trace is separate from validation. Its explicit
    # budget permits a continuous-record diagnostic even when the full clip is long.
    reference_folder = output / "diagnostics" / "common-walking-reference"
    reference_record = self.record(
      "record-common-reference",
      output / "B" / "snapshots" / "frames_0.pt",
      reference_folder,
      reference_controller=True,
      steps=250,
    )
    common = {
      "policy-baseline": output / "diagnostics" / "B" / "0" / "episodes.pt",
      "walking-reference": reference_record,
    }
    write_json(
      output / "diagnostics" / "common-recordings.json",
      {
        "scope": "fixed_inputs_for_process_and_posterior_only_not_policy_performance",
        "motion_sha256": p["motion_sha256"],
        "reference_step_budget": 250,
        "reference_may_end_early_on_physical_failure": True,
        "recordings": {name: str(path) for name, path in common.items()},
      },
    )
    for group in "ABC":
      checkpoint = output / group / "snapshots" / "frames_5120.pt"
      for name, record in common.items():
        for mode in ("process", "posterior"):
          self.command(
            f"common-{name}-{group}-{mode}",
            "diagnose_eit",
            [
              "--checkpoint",
              checkpoint,
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
      # Historical snapshots can still be rescored under their frozen protocol.
      legacy = {k: v for k, v in p.items() if k != "training_process_environment"}
      legacy["version"] = "eit_pilot_v1"
      if previous != p and not (args.phase == "evaluate" and previous == legacy):
        raise ValueError("Pilot configuration changed; use a new output directory")
      p = previous
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
