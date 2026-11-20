"""Train random mean/noise models by posterior NLL with a fixed actor.

Example:
  python -m supermimic.scripts.train_belief --motion-file /path/motion.npz \
    --device cuda:0 --output logs/belief-model

Full policy/critic optimization remains a separate training stage.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import mjlab.tasks  # noqa: F401
import torch
from mjlab.tasks.registry import load_env_cfg

import supermimic.tasks  # noqa: F401
from supermimic.estimator import FilterConfig
from supermimic.learners.belief_wm_learner import BeliefWorldModelLearnerCfg
from supermimic.models import ProcessNoiseConfig
from supermimic.motion_command import MotionCommandCfg
from supermimic.runners.belief_training import BeliefModelTraining, BeliefTrainingCfg
from supermimic.runners.supermimic_runner import SuperMimicRunner, SuperMimicRunnerCfg
from supermimic.scripts.train_utils import create_env
from supermimic.tasks.env_cfgs import configure_estimator_env_cfg


def main():
  parser = argparse.ArgumentParser(description=__doc__)
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
  parser.add_argument("--updates", type=int, default=10)
  parser.add_argument("--batch-size", type=int, default=4)
  parser.add_argument("--microbatch-size", type=int, default=2)
  parser.add_argument("--horizon", type=int, default=2)
  parser.add_argument("--lr", type=float, default=2e-4)
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
  runner_cfg = SuperMimicRunnerCfg.for_estimator_inputs(
    args.group,
    dynamics_hidden_dims=tuple(args.hidden_dims),
    process_noise=ProcessNoiseConfig(),
    filter_config=FilterConfig(
      jacobian_batch_chunk_size=args.jacobian_batch_chunk_size,
      jacobian_chunk_size=args.jacobian_chunk_size,
    ),
    belief_wm=BeliefWorldModelLearnerCfg(
      lr=args.lr,
      batch_size=args.batch_size,
      microbatch_size=args.microbatch_size,
      horizon=args.horizon,
      num_batches=args.updates,
    ),
    buffer_size=args.buffer_size,
    sampling_mode="uniform",
  )
  runner_cfg.policy.batch_size = args.batch_size
  env = create_env(env_cfg, args.num_envs, args.device)
  try:
    runner = SuperMimicRunner(env, runner_cfg, task_id, device=args.device)
    training = BeliefModelTraining(
      runner, BeliefTrainingCfg(collection_steps=args.collection_steps)
    )
    training.run(args.rounds, args.output, resume=args.resume)
  finally:
    env.close()


if __name__ == "__main__":
  main()
