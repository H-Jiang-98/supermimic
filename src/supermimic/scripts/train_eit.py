"""Train full estimator-in-the-loop actor, critic and stochastic world model."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import mjlab.tasks  # noqa: F401
import torch
from mjlab.tasks.registry import load_env_cfg

import supermimic.tasks  # noqa: F401
from supermimic.critic_inputs import CRITIC_INPUT_VERSION
from supermimic.estimator import FilterConfig
from supermimic.learners.belief_policy_learner import BeliefPolicyLearnerCfg
from supermimic.models import ProcessNoiseConfig
from supermimic.runners.eit_training import EITTraining, EITTrainingCfg
from supermimic.runners.run_config import EIT_INITIALIZATION_VERSION, load_runner_cfg
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
  parser.add_argument("--output", required=True, type=Path)
  parser.add_argument("--resume", type=Path)
  parser.add_argument(
    "--initialization",
    type=Path,
    help=f"Shared A/B/C initial weights (default: output parent/{EIT_INITIALIZATION_VERSION}_{CRITIC_INPUT_VERSION}_seed<seed>.pt)",
  )
  parser.add_argument("--group", choices=("A", "B", "C"), default="B")
  parser.add_argument(
    "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
  )
  parser.add_argument("--seed", type=int, default=2026)
  parser.add_argument("--num-envs", type=int, default=8)
  parser.add_argument("--rounds", type=int, default=10)
  parser.add_argument("--collection-steps", type=int, default=64)
  parser.add_argument("--buffer-size", type=int, default=128)
  add_model_fit_args(parser)
  parser.add_argument("--policy-updates", type=int, default=1)
  parser.add_argument("--policy-batch-size", type=int, default=4)
  parser.add_argument("--samples", type=int, default=2)
  parser.add_argument("--policy-microbatch-size", type=int, default=2)
  parser.add_argument("--policy-horizon", type=int, default=4)
  parser.add_argument("--critic-epochs", type=int, default=2)
  parser.add_argument("--exploration", type=float, default=0.1)
  parser.add_argument("--hidden-dims", type=int, nargs="+", default=[2048, 2048, 2048])
  parser.add_argument(
    "--actor-hidden-dims", type=int, nargs="+", default=[2048, 1024, 512, 256]
  )
  parser.add_argument("--critic-hidden-dims", type=int, nargs="+", default=[256, 128])
  parser.add_argument("--jacobian-batch-chunk-size", type=int)
  parser.add_argument("--jacobian-chunk-size", type=int)
  args = parser.parse_args()
  if args.num_envs < 1 or args.rounds < 1:
    parser.error("num-envs and rounds must be positive")
  if args.resume and args.initialization:
    parser.error("resume and initialization are mutually exclusive")
  torch.manual_seed(args.seed)
  torch.compiler.set_stance("force_eager")
  task_id = "SuperMimic-GenCoord-G1"
  env_cfg = load_env_cfg(task_id)
  env_cfg.seed = args.seed
  env_cfg.commands["motion"].motion_file = str(Path(args.motion_file).expanduser())
  env_cfg.commands["motion"].sampling_mode = "uniform"
  configure_estimator_env_cfg(env_cfg)
  if args.resume:
    saved = torch.load(args.resume, map_location=args.device, weights_only=True)
    EITTraining._check_checkpoint_version(saved)
    runner_cfg = load_runner_cfg({"runner_cfg": saved["runner_config"]})
    loop_cfg = EITTrainingCfg(**saved["loop_config"])
    policy_cfg = BeliefPolicyLearnerCfg(**saved["policy_config"])
    num_envs = saved["num_envs"]
    del saved
  else:
    mean, cov = model_fit_configs(args)
    runner_cfg = SuperMimicRunnerCfg.for_estimator_inputs(
      args.group,
      dynamics_hidden_dims=tuple(args.hidden_dims),
      actor_hidden_dims=tuple(args.actor_hidden_dims),
      critic_hidden_dims=tuple(args.critic_hidden_dims),
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
    runner_cfg.policy.batch_size = args.policy_microbatch_size
    loop_cfg = EITTrainingCfg(collection_steps=args.collection_steps)
    policy_cfg = BeliefPolicyLearnerCfg(
      batch_size=args.policy_batch_size,
      num_samples=args.samples,
      microbatch_size=args.policy_microbatch_size,
      horizon=args.policy_horizon,
      num_batches=args.policy_updates,
      critic_epochs=args.critic_epochs,
      noise=args.exploration,
    )
    num_envs = args.num_envs
  env_cfg.commands["motion"].sampling_mode = (
    "external" if runner_cfg.sampling_mode == "return" else runner_cfg.sampling_mode
  )
  env = create_env(env_cfg, num_envs, args.device)
  try:
    runner = SuperMimicRunner(env, runner_cfg, task_id, device=args.device)
    training = EITTraining(runner, loop_cfg, policy_cfg)
    if not args.resume:
      initialization = (
        args.initialization
        or args.output.parent
        / f"{EIT_INITIALIZATION_VERSION}_{CRITIC_INPUT_VERSION}_seed{args.seed}.pt"
      )
      if initialization.exists():
        initial = torch.load(
          initialization, map_location=args.device, weights_only=True
        )
      else:
        initial = training.initialization_bundle()
        initialization.parent.mkdir(parents=True, exist_ok=True)
        # A complete file is published atomically without overwriting another
        # concurrently starting ablation's initialization.
        with NamedTemporaryFile(dir=initialization.parent, delete=False) as stream:
          temporary = Path(stream.name)
          torch.save(initial, stream)
        try:
          try:
            os.link(temporary, initialization)
          except FileExistsError:
            initial = torch.load(
              initialization, map_location=args.device, weights_only=True
            )
        finally:
          temporary.unlink(missing_ok=True)
      training.load_initialization(initial)
    training.run(args.rounds, args.output, resume=args.resume)
  finally:
    env.close()


if __name__ == "__main__":
  main()
