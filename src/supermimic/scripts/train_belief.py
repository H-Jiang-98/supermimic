"""Fit the autoregressive mean and one-step process covariance with a fixed actor."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import mjlab.tasks  # noqa: F401
import torch
from mjlab.tasks.registry import load_env_cfg

import supermimic.tasks  # noqa: F401
from supermimic.estimator import FilterConfig
from supermimic.models import ProcessNoiseConfig
from supermimic.motion_command import MotionCommandCfg
from supermimic.runners.belief_training import BeliefModelTraining, BeliefTrainingCfg
from supermimic.runners.run_config import load_runner_cfg
from supermimic.runners.supermimic_runner import SuperMimicRunner, SuperMimicRunnerCfg
from supermimic.scripts.train_utils import (
  add_model_fit_args,
  create_env,
  model_fit_configs,
)
from supermimic.tasks.env_cfgs import configure_estimator_env_cfg


def main():
  parser = argparse.ArgumentParser(
    description=__doc__,
    allow_abbrev=False,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument("--motion-file", required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--resume", type=Path)
  parser.add_argument(
    "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
  )
  parser.add_argument("--group", choices=("A", "B", "C"), default="B")
  parser.add_argument("--seed", type=int, default=2026)
  parser.add_argument("--num-envs", type=int, default=8)
  parser.add_argument("--rounds", type=int, default=10)
  parser.add_argument("--collection-steps", type=int, default=64)
  parser.add_argument("--buffer-size", type=int, default=128)
  add_model_fit_args(parser)
  parser.add_argument("--hidden-dims", type=int, nargs="+", default=[2048, 2048, 2048])
  parser.add_argument("--jacobian-batch-chunk-size", type=int)
  parser.add_argument("--jacobian-chunk-size", type=int)
  args = parser.parse_args()
  if args.num_envs < 1 or args.rounds < 1:
    parser.error("num-envs and rounds must be positive")
  torch.manual_seed(args.seed)
  torch.compiler.set_stance("force_eager")
  task_id = "SuperMimic-GenCoord-G1"
  env_cfg = load_env_cfg(task_id)
  env_cfg.seed = args.seed
  motion = cast(MotionCommandCfg, env_cfg.commands["motion"])
  motion.motion_file = str(Path(args.motion_file).expanduser())
  motion.sampling_mode = "uniform"
  configure_estimator_env_cfg(env_cfg)
  if args.resume:
    saved = torch.load(args.resume, map_location=args.device, weights_only=True)
    BeliefModelTraining._check_checkpoint_version(saved)
    runner_cfg = load_runner_cfg({"runner_cfg": saved["runner_config"]})
    loop_cfg = BeliefTrainingCfg(**saved["loop_config"])
    num_envs = saved["num_envs"]
    del saved
  else:
    mean, cov = model_fit_configs(args)
    runner_cfg = SuperMimicRunnerCfg.for_estimator_inputs(
      args.group,
      dynamics_hidden_dims=tuple(args.hidden_dims),
      process_noise=ProcessNoiseConfig(),
      filter_config=FilterConfig(
        jacobian_batch_chunk_size=args.jacobian_batch_chunk_size,
        jacobian_chunk_size=args.jacobian_chunk_size,
      ),
      mean=mean,
      cov=cov,
      buffer_size=args.buffer_size,
      sampling_mode="uniform",
    )
    runner_cfg.policy.batch_size = args.num_envs
    loop_cfg = BeliefTrainingCfg(collection_steps=args.collection_steps)
    num_envs = args.num_envs
  motion.sampling_mode = (
    "external" if runner_cfg.sampling_mode == "return" else runner_cfg.sampling_mode
  )
  env = create_env(env_cfg, num_envs, args.device)
  try:
    runner = SuperMimicRunner(env, runner_cfg, task_id, device=args.device)
    training = BeliefModelTraining(runner, loop_cfg)
    training.run(args.rounds, args.output, resume=args.resume)
  finally:
    env.close()


if __name__ == "__main__":
  main()
