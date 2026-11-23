"""Shared utilities for training scripts."""

from __future__ import annotations

import argparse

import torch
import wandb
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.torch import configure_torch_backends

from supermimic.learners.two_stage_wm import CovarianceFitCfg, MeanFitCfg
from supermimic.utils.wandb_artifacts import download_artifact_cached


def add_model_fit_args(parser: argparse.ArgumentParser) -> None:
  mean = parser.add_argument_group("autoregressive mean fit")
  mean.add_argument(
    "--mean-lr", type=float, default=MeanFitCfg.lr, help="Muon/Adam learning rate"
  )
  mean.add_argument(
    "--mean-updates",
    type=int,
    default=MeanFitCfg.num_batches,
    help="Mean optimizer updates per training group",
  )
  mean.add_argument(
    "--mean-batch-size",
    type=int,
    default=MeanFitCfg.batch_size,
    help="Windows per mean update",
  )
  mean.add_argument(
    "--mean-window-frames",
    type=int,
    default=MeanFitCfg.window_frames,
    help="Recorded frames per window, including the real initial state",
  )
  cov = parser.add_argument_group("one-step process covariance fit")
  cov.add_argument(
    "--cov-lr",
    type=float,
    default=CovarianceFitCfg.lr,
    help="Noise-head Adam learning rate",
  )
  cov.add_argument(
    "--cov-batch-size",
    type=int,
    default=CovarianceFitCfg.batch_size,
    help="Residuals per covariance update, including a partial final batch",
  )
  cov.add_argument(
    "--cov-pair-budget",
    type=int,
    help="Sampled training transitions per group; default: mean updates × batch × (window - 1)",
  )
  cov.add_argument(
    "--cov-validation-pairs", type=int, default=CovarianceFitCfg.validation_pairs
  )


def model_fit_configs(args) -> tuple[MeanFitCfg, CovarianceFitCfg]:
  return (
    MeanFitCfg(
      lr=args.mean_lr,
      num_batches=args.mean_updates,
      batch_size=args.mean_batch_size,
      window_frames=args.mean_window_frames,
    ),
    CovarianceFitCfg(
      lr=args.cov_lr,
      batch_size=args.cov_batch_size,
      pair_budget=args.cov_pair_budget,
      validation_pairs=args.cov_validation_pairs,
    ),
  )


def download_registry_motions(registry_name: str | None) -> str | list[str] | None:
  """Download motion files from W&B registry artifacts.

  Supports comma-separated registry names for multi-motion.
  Returns a single path string if one motion, a list if multiple, or None.
  """
  if not registry_name:
    return None

  names = [n.strip() for n in registry_name.split(",")]
  api = wandb.Api()

  paths = []
  for name in names:
    if ":" not in name:
      name = name + ":latest"
    artifact = api.artifact(name)
    paths.append(str(download_artifact_cached(artifact) / "motion.npz"))

  print(f"[INFO] Loaded {len(paths)} motion(s)")
  return paths if len(paths) > 1 else paths[0]


def setup_device(device: str | None = None) -> str:
  """Configure torch backends and resolve device."""
  configure_torch_backends()
  return device or ("cuda:0" if torch.cuda.is_available() else "cpu")


def create_env(
  env_cfg: ManagerBasedRlEnvCfg, num_envs: int, device: str
) -> RslRlVecEnvWrapper:
  """Set ``env_cfg.scene.num_envs`` and build a wrapped RL environment."""
  env_cfg.scene.num_envs = num_envs
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  return RslRlVecEnvWrapper(env)
