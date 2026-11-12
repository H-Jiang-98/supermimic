"""Play a SuperMimic multi-encoder actor against the sim."""

from __future__ import annotations

import argparse
import glob
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import mjlab.tasks  # noqa: F401
import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.sim import SimulationCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.torch import configure_torch_backends

import supermimic.tasks  # noqa: F401
from supermimic.runners.run_config import check_representation_matches, load_runner_cfg
from supermimic.runners.supermimic_runner import build_actor, build_world_model
from supermimic.scripts.config_utils import config_args
from supermimic.scripts.viewers import (
  SuperMimicNativeViewer, # 提供本地窗口
  SuperMimicViserViewer,  # 提供网页显示
  create_wm_vis_model,    # 创建用于显示世界模型预测的模型副本
)
from supermimic.utils.wandb_artifacts import (
  download_artifact_cached,  
  download_model_dir,        
)


@dataclass
class PlayMujocoConfig:
  iterations: int | None = None
  """Solver iteration limit. Unset: use the training run's value, then task default."""
  ls_iterations: int | None = None
  """Linesearch limit. Unset: use the training run's value, then task default."""


@dataclass
class PlaySimulationConfig:
  nconmax: int | None = None
  """Contact capacity. Unset: at least 256 for single-environment playback."""
  njmax: int | None = None
  """Constraint capacity. Unset: use the training run's value, then task default."""
  mujoco: PlayMujocoConfig = field(default_factory=PlayMujocoConfig)


@dataclass
class PlayConfig:
  wandb_path: str = ""
  """Required W&B run path: entity/project/run_id."""
  config: Path | None = None
  """TOML settings file. Explicit command-line options override its values."""
  checkpoint: str = "latest"
  """Model artifact version (e.g. v25, v32) or alias (e.g. latest)."""
  motion_file: str | None = None
  initial_task: Literal["motion", "velocity"] = "motion"
  """Task selected when the viewer opens. Switch live via the GUI dropdown."""
  device: str | None = None
  num_steps_rollout: int = 1000000
  viewer: Literal["auto", "native", "viser"] = "viser"
  sim: PlaySimulationConfig = field(default_factory=PlaySimulationConfig)


def configure_play_simulation(
  sim: SimulationCfg, overrides: PlaySimulationConfig, run_config: dict[str, Any]
) -> None:
  """Restore training limits, then apply playback overrides before allocation."""
  saved_sim = run_config.get("env_cfg", {}).get("sim", {})
  for name in ("nconmax", "njmax"):
    value = getattr(overrides, name)
    if value is None:
      value = saved_sim.get(name, getattr(sim, name))
      if name == "nconmax":
        # Training pools contacts across many worlds; play has only one world.
        value = max(256, value or 0)
    if value is not None and value <= 0:
      raise ValueError(f"sim.{name} must be positive")
    setattr(sim, name, value)

  saved_mujoco = saved_sim.get("mujoco", {})
  for name in ("iterations", "ls_iterations"):
    value = getattr(overrides.mujoco, name)
    if value is None:
      value = saved_mujoco.get(name, getattr(sim.mujoco, name))
    if value <= 0:
      raise ValueError(f"sim.mujoco.{name} must be positive")
    setattr(sim.mujoco, name, value)

  print(
    f"[INFO] Playback simulation: nconmax={sim.nconmax}, njmax={sim.njmax}, "
    f"iterations={sim.mujoco.iterations}, ls_iterations={sim.mujoco.ls_iterations}"
  )


def play(task_id: str, cfg: PlayConfig) -> None:
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  print(f"[INFO] Loading environment: {task_id}")
  env_cfg = load_env_cfg(task_id, play=True)
  env_cfg.scene.num_envs = 1
  env_cfg.observations["actor"].enable_corruption = True

  import wandb

  print(f"[INFO] Connecting to WandB run: {cfg.wandb_path}")
  run = wandb.Api().run(cfg.wandb_path)
  runner_cfg = load_runner_cfg(run.config)   # 从训练记录中恢复配置  
  check_representation_matches(task_id, runner_cfg.representation)   # 检查任务与模型是否匹配 
  configure_play_simulation(env_cfg.sim, cfg.sim, run.config)

  motion_artifacts = [a for a in run.used_artifacts() if a.type == "motions"]
  command_term = env_cfg.commands["motion"]
  assert isinstance(command_term, MotionCommandCfg)
  motion_names: list[str] = []
  if cfg.motion_file is not None:
    expanded = os.path.expanduser(cfg.motion_file)
    if any(c in expanded for c in "*?[]"):
      files = sorted(glob.glob(expanded))
    else:
      files = [expanded]
    if not files:
      raise ValueError(f"No motion files matched: {cfg.motion_file}")
    command_term.motion_file = files
    motion_names = [Path(f).stem for f in files]
    print(f"[INFO] Using {len(files)} local motion(s)")
  elif motion_artifacts:
    motion_files = []
    for artifact in motion_artifacts:
      print(f"[INFO] Found linked motion artifact: {artifact.name}")
      motion_files.append(str(download_artifact_cached(artifact) / "motion.npz"))
    command_term.motion_file = (
      motion_files if len(motion_files) > 1 else motion_files[0]
    )
    motion_names = [a.name.split(":")[0] for a in motion_artifacts]
    print(f"[INFO] Loaded {len(motion_files)} motion(s)")

  model_dir = download_model_dir(run, checkpoint=cfg.checkpoint)
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=device))

  wm = build_world_model(runner_cfg, env, device)
  representation = wm.representation
  dynamics = wm.dynamics
  # bundle["dynamics"] 动力学网络参数， bundle["sim_context"] 仿真上下文
  wm.load_bundle(torch.load(model_dir / "world_model.pt", map_location=device))
  dynamics.eval()

  actor = build_actor(runner_cfg, wm, device)
  actor.load_state_dict(torch.load(model_dir / "actor.pt", map_location=device))
  actor.eval()

  wm_vis_model = create_wm_vis_model(env.unwrapped.sim.mj_model)

  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved = "native" if has_display else "viser"
  else:
    resolved = cfg.viewer
  print(f"[INFO] Starting viewer: {resolved}")

  if resolved == "native":
    SuperMimicNativeViewer(
      env=env,
      actor=actor,
      dynamics=dynamics,
      representation=representation,
      wm_vis_model=wm_vis_model,
      num_steps_rollout=cfg.num_steps_rollout,
      sim_context=wm.sim_context,
      task_specs=actor.task_specs,
      initial_task=cfg.initial_task,
    ).run()
  else:
    SuperMimicViserViewer(
      env=env,
      actor=actor,
      dynamics=dynamics,
      representation=representation,
      wm_vis_model=wm_vis_model,
      num_steps_rollout=cfg.num_steps_rollout,
      motion_names=motion_names,
      sim_context=wm.sim_context,
      task_specs=actor.task_specs,
      initial_task=cfg.initial_task,
    ).run()

  env.close()


def parse_play_args(argv: list[str] | None = None) -> tuple[str, PlayConfig]:
  """Resolve code defaults, optional TOML settings, then explicit CLI overrides."""
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
  defaults = PlayConfig()
  if config_option.config is not None:
    config_path = config_option.config.expanduser()
    try:
      with config_path.open("rb") as file:
        file_args = config_args(tomllib.load(file), defaults, context="playback")
    except (OSError, ValueError) as error:
      config_parser.error(f"Cannot load playback config {config_path}: {error}")
    defaults = tyro.cli(
      PlayConfig,
      args=file_args,
      default=defaults,
      prog=f"play.py --config {config_path}",
      config=mjlab.TYRO_FLAGS,
    )

  args = tyro.cli(
    PlayConfig, args=remaining_args, default=defaults, config=mjlab.TYRO_FLAGS
  )
  if not args.wandb_path.strip():
    config_parser.error("Set wandb_path in the config file or pass --wandb-path")
  if not args.checkpoint.strip():
    config_parser.error("checkpoint must be a version (e.g. v32) or alias (latest)")
  for name, value in (
    ("sim.nconmax", args.sim.nconmax),
    ("sim.njmax", args.sim.njmax),
    ("sim.mujoco.iterations", args.sim.mujoco.iterations),
    ("sim.mujoco.ls_iterations", args.sim.mujoco.ls_iterations),
    ("num_steps_rollout", args.num_steps_rollout),
  ):
    if value is not None and value <= 0:
      config_parser.error(f"{name} must be positive")
  return chosen_task, args


def main() -> None:
  chosen_task, args = parse_play_args()
  if args.config is not None:
    print(f"[INFO] Loaded playback config: {args.config}")
  play(chosen_task, args)


if __name__ == "__main__":
  main()
