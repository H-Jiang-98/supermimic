"""Launch the EIT methods and seeds in train_eit_config.toml."""

from __future__ import annotations

import argparse
import asyncio
import glob
import os
import shlex
import tomllib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
  environment_root: str
  checkpoint_root: str
  dataset_root: str
  experiment: str
  motion_file: str


def _child(root: Path, relative: str) -> Path:
  path = Path(relative)
  if path == Path(".") or path.is_absolute() or ".." in path.parts:
    raise ValueError(f"Expected a path relative to {root}, got {relative!r}")
  return root / path


def _training_args(settings: dict) -> list[str]:
  # Scheduling and output identity belong to the launcher. The worker's
  # argparse parser validates the training options, including network widths.
  reserved = {
    "group",
    "seed",
    "device",
    "motion_file",
    "output",
    "run_name",
    "resume",
    "resume_reference",
    "prepare_initialization",
    "initialization",
    "help",
  }
  args = []
  for name, value in settings.items():
    if not name.isidentifier() or name in reserved:
      raise ValueError(f"Invalid [training] option: {name}")
    flag = "--" + name.replace("_", "-")
    if name in ("eager", "policy_cuda_graphs"):
      if type(value) is not bool:
        raise ValueError(f"training.{name} must be a boolean")
      if value:
        args.append(flag)
    elif isinstance(value, list) and value and all(type(v) is int for v in value):
      args.extend([flag, *map(str, value)])
    elif type(value) in (int, float, str):
      args.append(f"{flag}={value}")
    else:
      raise ValueError(f"Invalid value for training.{name}: {value!r}")
  return args


def _indices(values: list[int], label: str) -> list[int]:
  if (
    not isinstance(values, list)
    or not values
    or any(type(v) is not int or v < 0 for v in values)
    or len(set(values)) != len(values)
  ):
    raise ValueError(
      f"{label} requires a nonempty list of distinct nonnegative integers"
    )
  return values


def _mujoco_args(env: dict) -> list[str]:
  table = env
  for key in ("sim", "mujoco"):
    if not isinstance(table, dict) or set(table) != {key}:
      raise ValueError("Expected [env.sim.mujoco]")
    table = table[key]
  if not isinstance(table, dict) or table.keys() - {"iterations", "ls_iterations"}:
    raise ValueError("[env.sim.mujoco] accepts iterations and ls_iterations")
  args = []
  for name, value in table.items():
    if type(value) is not int or value < 0:
      raise ValueError(f"env.sim.mujoco.{name} must be a nonnegative integer")
    args.append(f"--env.sim.mujoco.{name.replace('_', '-')}={value}")
  return args


def resolve_paths(settings: dict, base: Path):
  paths = Paths(**settings)
  if not all(isinstance(v, str) and v.strip() for v in settings.values()):
    raise ValueError("[paths] values must be nonempty strings")
  environment = base / Path(paths.environment_root).expanduser()
  output = _child(base / Path(paths.checkpoint_root).expanduser(), paths.experiment)
  motion = _child(base / Path(paths.dataset_root).expanduser(), paths.motion_file)
  return environment, motion, output


def worker_command(config: dict, environment: Path, motion: Path) -> list[str]:
  validation = config.get("wm validation", {})
  limits = {
    "validation_stride": 2,
    "validation_batch_size": 1,
    "cov_validation_pairs": 1,
  }
  if validation.keys() - limits.keys():
    raise ValueError(
      f"Unknown [wm validation] options: {validation.keys() - limits.keys()}"
    )
  for name, value in validation.items():
    if type(value) is not int or value < limits[name]:
      raise ValueError(f"wm validation.{name} must be an integer >= {limits[name]}")
    if name in config["training"]:
      raise ValueError(f"Duplicate {name} in [training] and [wm validation]")
  common = [
    str(environment / ".venv/bin/python"),
    "-m",
    "supermimic.scripts.train_eit",
    "--motion-file",
    str(motion),
    "--device",
    "cuda:0",
    *_training_args(config["training"]),
    *_training_args(validation),
  ]
  if "env" in config:
    common.extend(_mujoco_args(config["env"]))
  return common


def plan(config: dict, base: Path):
  if set(config) - {"env", "wm validation"} != {
    "paths",
    "training",
    "methods",
  } or not all(isinstance(table, dict) for table in config.values()):
    raise ValueError("Expected [paths], [training] and [methods] tables")
  environment, motion, output = resolve_paths(config["paths"], base)
  common = worker_command(config, environment, motion)
  methods = config["methods"]
  if not methods or set(methods) - {"A", "B", "C"}:
    raise ValueError("[methods] requires one or more of A, B and C")

  prepare = defaultdict(list)
  train = defaultdict(list)
  prepared_seeds = set()
  for group, method in methods.items():
    if not isinstance(method, dict) or set(method) != {"gpus", "seeds"}:
      raise ValueError(f"[methods.{group}] requires gpus and seeds")
    gpus = _indices(method["gpus"], f"methods.{group}.gpus")
    seeds = _indices(method["seeds"], f"methods.{group}.seeds")
    for i, seed in enumerate(seeds):
      gpu = gpus[i % len(gpus)]
      init = output / "init" / f"seed{seed}.pt"
      args = [*common, "--seed", str(seed), "--initialization", str(init)]
      if seed not in prepared_seeds:
        prepare[gpu].append([*args, "--group", "B", "--prepare-initialization"])
        prepared_seeds.add(seed)
      train[gpu].append(
        [
          *args,
          "--group",
          group,
          "--run-name",
          f"eit-{group}-seed{seed}",
          "--output",
          str(output / f"seed{seed}" / group),
        ]
      )
  return environment, motion, (prepare, train)


async def run_queue(gpu: int, commands: list[list[str]], environment: Path):
  env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(gpu))
  for key, value in (
    ("WANDB_MODE", "offline"),
    ("OMP_NUM_THREADS", "1"),
    ("PYTORCH_JIT", "0"),
    ("CUBLAS_WORKSPACE_CONFIG", ":4096:8"),
  ):
    env.setdefault(key, value)
  # Replay validation/indexing grows until the ring fills. Expandable segments
  # reuse memory for changing sizes while the policy graph retains its private pool.
  # Honor either spelling of an explicitly configured PyTorch allocator.
  setup = environment / "env.sh"
  if not setup.is_file() and not any(
    key in env for key in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")
  ):
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
  for command in commands:
    print(f"GPU {gpu}  {shlex.join(command)}", flush=True)
    if setup.is_file():
      command = [
        "bash",
        "-c",
        # A sourced env.sh can replace the shell's positional arguments with set --.
        'readonly eit_launch_gpu="$2" && readonly -a eit_launch_command=("${@:3}") && '
        'source "$1" && export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$eit_launch_gpu" && '
        "if [[ ! -v PYTORCH_ALLOC_CONF && ! -v PYTORCH_CUDA_ALLOC_CONF ]]; then "
        'export PYTORCH_ALLOC_CONF=expandable_segments:True; fi && exec "${eit_launch_command[@]}"',
        "eit",
        str(setup),
        str(gpu),
        *command,
      ]
    try:
      process = await asyncio.create_subprocess_exec(*command, env=env)
    except OSError as error:
      raise RuntimeError(f"Cannot start EIT process on GPU {gpu}: {error}") from error
    try:
      status = await process.wait()
    except asyncio.CancelledError:
      if process.returncode is None:
        process.terminate()
      await process.wait()
      raise
    if status:
      raise RuntimeError(f"EIT process on GPU {gpu} exited with status {status}")


async def launch(phases, environment, *, wait_for_all: bool = False):
  for queues in phases:
    results = await asyncio.gather(
      *(run_queue(gpu, jobs, environment) for gpu, jobs in queues.items()),
      return_exceptions=wait_for_all,
    )
    failures = [str(result) for result in results if isinstance(result, Exception)]
    if failures:
      raise RuntimeError("\n".join(failures))


def run_cli(plan, default_config: str, description: str | None, *, wait_for_all=False):
  parser = argparse.ArgumentParser(description=description)
  parser.add_argument("--config", type=Path, default=Path(default_config))
  parser.add_argument(
    "--dry-run", action="store_true", help="Print GPU queues without starting workers"
  )
  args = parser.parse_args()
  try:
    with args.config.open("rb") as stream:
      environment, motion, phases = plan(
        tomllib.load(stream), args.config.resolve().parent
      )
    if args.dry_run:
      for queues in phases:
        for gpu, jobs in queues.items():
          for cmd in jobs:
            print(f"GPU {gpu}  {shlex.join(cmd)}")
      return
    if not os.access(environment / ".venv/bin/python", os.X_OK):
      raise ValueError(f"Missing executable: {environment}/.venv/bin/python")
    if not any(Path(match).is_file() for match in glob.iglob(str(motion))):
      raise ValueError(f"No motion files matched {motion}")
    asyncio.run(launch(phases, environment, wait_for_all=wait_for_all))
  except (KeyError, TypeError, ValueError, OSError, RuntimeError) as error:
    parser.exit(1, f"EIT launch failed: {error}\n")


def main():
  run_cli(plan, "train_eit_config.toml", __doc__)


if __name__ == "__main__":
  main()
