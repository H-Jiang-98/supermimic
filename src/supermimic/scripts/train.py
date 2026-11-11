"""Script to train World Model using SuperMimicRunner."""

from __future__ import annotations

import argparse
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import mjlab.tasks  # noqa: F401
import torch
import tyro
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg

import supermimic.tasks  # noqa: F401
from supermimic.motion_command import MotionCommandCfg
from supermimic.runners.supermimic_runner import SuperMimicRunner, SuperMimicRunnerCfg
from supermimic.scripts.config_utils import config_args as _config_args
from supermimic.scripts.train_utils import (
  create_env,
  download_registry_motions,
  setup_device,
)


@dataclass
class TrainConfig:
  env: ManagerBasedRlEnvCfg
  runner: SuperMimicRunnerCfg

  config: Path | None = None
  """TOML settings file. Explicit command-line options override its values."""

  # Experiment & Environment
  num_envs: int = 4096
  device: str | None = None
  registry_name: str | None = "supermimic/wandb-registry-motions/lafan_run1_subject5"
  motion_file: str | None = None
  eager: bool = False
  """Diagnose startup stalls without torch.compile or learner CUDA graphs.

  MuJoCo/Warp simulator kernel compilation is still required.
  """

  @staticmethod
  def from_task(task_id: str) -> "TrainConfig":
    env_cfg = load_env_cfg(task_id, play=False)
    return TrainConfig(
      env=env_cfg,
      runner=SuperMimicRunnerCfg(),
    )


def parse_train_args(argv: list[str] | None = None) -> tuple[str, TrainConfig]:
  """Resolve task defaults, optional TOML settings, then explicit CLI overrides."""
  # Parse first argument to choose the task.
  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    args=argv,
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  config_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
  config_parser.add_argument("--config", type=Path)
  config_option, _ = config_parser.parse_known_args(remaining_args)
  defaults = TrainConfig.from_task(chosen_task)
  if config_option.config is not None:
    config_path = config_option.config.expanduser()
    try:
      with config_path.open("rb") as file:
        file_args = _config_args(tomllib.load(file), defaults)
    except (OSError, ValueError) as error:
      config_parser.error(f"Cannot load training config {config_path}: {error}")
    defaults = tyro.cli(
      TrainConfig,
      args=file_args,
      default=defaults,
      prog=f"train.py --config {config_path}",
      config=mjlab.TYRO_FLAGS,
    )

  args = tyro.cli(
    TrainConfig,
    args=remaining_args,
    default=defaults,
    config=mjlab.TYRO_FLAGS,
  )
  return chosen_task, args


def main() -> None:
  chosen_task, args = parse_train_args()
  if args.config is not None:
    print(f"[INFO] Loaded training config: {args.config}")

  device = setup_device(args.device)
  if args.eager:
    torch.compiler.set_stance("force_eager")
    args.runner.cuda_graphs = False
    print("[INFO] Eager diagnostics: torch.compile and learner CUDA graphs disabled.")
  print(f"[INFO] Loading environment: {chosen_task}")
  motion_cfg = cast(MotionCommandCfg, args.env.commands["motion"])

  if args.motion_file:
    # Local motions handed straight to MotionLoader (globs/concatenates).
    if "," in args.motion_file:
      motion_cfg.motion_file = [
        os.path.expanduser(p.strip()) for p in args.motion_file.split(",")
      ]
    else:
      motion_cfg.motion_file = os.path.expanduser(args.motion_file)
    args.registry_name = None  # avoid linking a stale registry artifact for lineage
    print(f"[INFO] Using local motion file: {args.motion_file}")
  else:
    motion_file = download_registry_motions(args.registry_name)
    if motion_file is not None:
      motion_cfg.motion_file = motion_file

  # Auto-detect representation from task_id if not explicitly set
  if "MaxCoord" in chosen_task and args.runner.representation == "gencoord":
    args.runner.representation = "maxcoord"
    print("[INFO] Auto-detected maxcoord representation from task_id")

  motion_cfg.sampling_mode = (
    "external" if args.runner.sampling_mode == "return" else args.runner.sampling_mode
  )

  env = create_env(args.env, args.num_envs, device)

  # Init Runner
  runner = SuperMimicRunner(
    env=env,
    cfg=args.runner,
    task_id=chosen_task,
    registry_name=args.registry_name,
    device=device,
  )

  runner.run()


if __name__ == "__main__":
  main()
