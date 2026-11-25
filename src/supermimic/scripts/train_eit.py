"""Train full estimator-in-the-loop actor, critic and stochastic world model."""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import cast

import mjlab.tasks  # noqa: F401
import torch
from mjlab.tasks.registry import load_env_cfg

import supermimic.tasks  # noqa: F401
from supermimic.critic_inputs import CRITIC_INPUT_VERSION
from supermimic.estimator import FilterConfig
from supermimic.learners.belief_policy_learner import BeliefPolicyLearnerCfg
from supermimic.models import ProcessNoiseConfig
from supermimic.motion_command import MotionCommandCfg
from supermimic.runners.eit_training import (
  EITTraining,
  EITTrainingCfg,
  initialization_fingerprint,
)
from supermimic.runners.run_config import EIT_INITIALIZATION_VERSION, load_runner_cfg
from supermimic.runners.supermimic_runner import SuperMimicRunner, SuperMimicRunnerCfg
from supermimic.scripts.train_utils import (
  add_model_fit_args,
  create_env,
  model_fit_configs,
)
from supermimic.tasks.env_cfgs import configure_estimator_env_cfg


def _reference_path(reference: Path, output: Path, *inputs: Path | None) -> Path:
  path = reference.resolve()
  snapshots = (output / "snapshots").resolve()
  reserved = (
    "training.pt",
    "world_model.pt",
    "actor.pt",
    "critic.pt",
    "metrics.jsonl",
    "profiling.jsonl",
    "profiling_recovery.json",
    "wandb_run.json",
    ".belief-training.lock",
    ".eit-log-interval.json",
  )
  protected = {item.resolve() for item in inputs if item is not None}
  protected.update((output / name).resolve() for name in reserved)
  if (
    path in protected
    or path.is_relative_to(snapshots)
    or any(
      path.is_relative_to((output / name).resolve())
      for name in (
        "wandb",
        "wandb-data",
        "wandb-cache",
        ".eit-publication",
        ".eit-initializing",
      )
    )
  ):
    raise ValueError(
      "--resume-reference must be separate from training inputs, logs and snapshots"
    )
  return path


def main():
  parser = argparse.ArgumentParser(
    description=__doc__,
    allow_abbrev=False,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument("--motion-file", required=True)
  parser.add_argument("--env.sim.mujoco.iterations", dest="iterations", type=int)
  parser.add_argument("--env.sim.mujoco.ls-iterations", dest="ls_iterations", type=int)
  parser.add_argument("--output", type=Path)
  parser.add_argument(
    "--prepare-initialization",
    action="store_true",
    help="Publish the shared initialization at --initialization and exit before training",
  )
  parser.add_argument("--resume", type=Path)
  parser.add_argument(
    "--resume-reference",
    type=Path,
    help="Publish live next-update evidence before committing the final checkpoint",
  )
  parser.add_argument(
    "--initialization",
    type=Path,
    help=(
      "Shared A/B/C initial weights (default: output parent/"
      f"{EIT_INITIALIZATION_VERSION}_{CRITIC_INPUT_VERSION}_"
      "noise<fingerprint>_seed<seed>.pt)"
    ),
  )
  parser.add_argument("--group", choices=("A", "B", "C"), default="B")
  parser.add_argument(
    "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
  )
  parser.add_argument(
    "--eager", action="store_true", help="Disable CUDA training kernel compilation"
  )
  parser.add_argument("--seed", type=int, default=2026)
  parser.add_argument(
    "--num-envs", type=int, default=1024, help="Parallel physical environments"
  )
  parser.add_argument(
    "--max-iterations",
    type=int,
    default=1_000_000,
    help="Cumulative physical steps per environment, including before resume",
  )
  parser.add_argument("--replay-ratio", type=float, default=16)
  parser.add_argument(
    "--save-interval",
    type=int,
    help="Physical collection steps between saves (runner default: 1000)",
  )
  parser.add_argument(
    "--log-interval",
    type=int,
    help="Physical collection steps between logs (runner default: 10)",
  )
  parser.add_argument(
    "--sampling-mode", choices=("return", "uniform"), default="return"
  )
  parser.add_argument(
    "--buffer-size", type=int, default=1000, help="Retained frames per environment"
  )
  parser.add_argument(
    "--posterior-save-stride",
    type=int,
    default=1,
    help="Save Q[t|t] every S collection steps; resume uses saved stride",
  )
  add_model_fit_args(parser)
  validation = parser.add_argument_group("world model online validation")
  validation.add_argument(
    "--validation-stride",
    type=int,
    default=EITTrainingCfg.validation_stride,
    help="Hold out episode_id %% stride == 0 for model fitting only; at least 2",
  )
  validation.add_argument(
    "--validation-batch-size",
    type=int,
    default=EITTrainingCfg.validation_batch_size,
    help="Validation windows per training group, sampled uniformly with replacement",
  )
  parser.add_argument(
    "--policy-cuda-graphs",
    action="store_true",
    help="Capture a complete actor/critic update for divisible CUDA microbatches; "
    "disabled by --eager, independent of checkpoint hyperparameters",
  )
  parser.add_argument(
    "--policy-updates",
    type=int,
    default=1,
    help="Actor optimizer updates per training group",
  )
  parser.add_argument(
    "--policy-batch-size",
    type=int,
    default=4096,
    help="Replay starts per actor update, sampled with replacement",
  )
  parser.add_argument(
    "--samples", type=int, default=1, help="Independent noise rollouts per start"
  )
  parser.add_argument(
    "--policy-microbatch-size",
    type=int,
    default=16,
    help="Expanded start/sample trajectories per backward pass; gradients accumulate "
    "over the full policy batch before one actor update",
  )
  parser.add_argument(
    "--policy-horizon", type=int, default=32, help="Maximum imagined steps per rollout"
  )
  parser.add_argument("--critic-epochs", type=int, default=5)
  parser.add_argument("--critic-minibatches", type=int, default=4)
  parser.add_argument("--critic-microbatch-size", type=int)
  parser.add_argument("--policy-lr", type=float, default=BeliefPolicyLearnerCfg.lr)
  parser.add_argument(
    "--critic-lr", type=float, default=BeliefPolicyLearnerCfg.critic_lr
  )
  parser.add_argument("--exploration", type=float, default=0.1)
  parser.add_argument(
    "--mean-hidden-dims",
    "--hidden-dims",
    dest="mean_hidden_dims",
    type=int,
    nargs="+",
    default=[2048, 2048, 2048],
    help="World model mean hidden widths (--hidden-dims is a legacy alias)",
  )
  parser.add_argument(
    "--cov-hidden-dims",
    type=int,
    nargs="+",
    default=list(ProcessNoiseConfig().hidden_dims),
    help="World model process covariance hidden widths",
  )
  parser.add_argument(
    "--actor-hidden-dims", type=int, nargs="+", default=[2048, 1024, 512, 256]
  )
  parser.add_argument("--critic-hidden-dims", type=int, nargs="+", default=[256, 128])
  parser.add_argument(
    "--jacobian-backend",
    choices=("increment", "vmap", "eager"),
    default="increment",
    help="Local Jacobian implementation for new runs; resume uses saved settings",
  )
  parser.add_argument("--jacobian-batch-chunk-size", type=int)
  parser.add_argument("--jacobian-chunk-size", type=int)
  parser.add_argument("--wandb-project", help="Common runner W&B project")
  parser.add_argument("--run-name", help="Common runner run name")
  args = parser.parse_args()
  for name in ("mean_hidden_dims", "cov_hidden_dims"):
    if any(width < 1 for width in getattr(args, name)):
      parser.error(f"--{name.replace('_', '-')} requires positive hidden widths")
  os.environ.setdefault("WANDB_MODE", "offline")
  if os.environ["WANDB_MODE"] not in ("offline", "disabled"):
    parser.error("EIT requires WANDB_MODE=offline or disabled")
  if args.num_envs < 1 or args.max_iterations < 0:
    parser.error("num-envs must be positive and max-iterations nonnegative")
  if any(
    value is not None and value < 1 for value in (args.save_interval, args.log_interval)
  ):
    parser.error("save-interval and log-interval must be positive")
  if args.resume and args.initialization:
    parser.error("resume and initialization are mutually exclusive")
  if args.prepare_initialization and not args.initialization:
    parser.error("--prepare-initialization requires an explicit --initialization path")
  if args.prepare_initialization and args.resume_reference:
    parser.error("--resume-reference requires actual training")
  if not args.prepare_initialization and not args.output:
    parser.error("training requires --output")
  initialization = args.initialization
  if (
    args.initialization
    and not args.prepare_initialization
    and not args.initialization.is_file()
  ):
    parser.error(
      "Explicit --initialization must exist; create it with --prepare-initialization"
    )
  torch.manual_seed(args.seed)
  compile_kernels = (
    torch.device(args.device).type == "cuda"
    and not args.eager
    and os.environ.get("TORCH_COMPILE_DISABLE") != "1"
    and os.environ.get("TORCHDYNAMO_DISABLE") != "1"
  )
  torch.compiler.set_stance("default" if compile_kernels else "force_eager")
  print(
    f"[INFO] EIT execution: {'inductor' if compile_kernels else 'eager'}", flush=True
  )
  task_id = "SuperMimic-GenCoord-G1"
  env_cfg = load_env_cfg(task_id)
  for name in ("iterations", "ls_iterations"):
    value = getattr(args, name)
    if value is not None:
      if value < 0:
        parser.error(f"env.sim.mujoco.{name} must be nonnegative")
      setattr(env_cfg.sim.mujoco, name, value)
  env_cfg.seed = args.seed
  motion = cast(MotionCommandCfg, env_cfg.commands["motion"])
  motion.motion_file = str(Path(args.motion_file).expanduser())
  motion.sampling_mode = "uniform"
  configure_estimator_env_cfg(env_cfg)
  if args.resume:
    saved = torch.load(args.resume, map_location=args.device, weights_only=True)
    EITTraining._check_checkpoint_version(saved)
    runner_cfg = load_runner_cfg({"runner_cfg": saved["runner_config"]})
    loop_cfg = EITTrainingCfg(**saved["loop_config"])
    policy_cfg = BeliefPolicyLearnerCfg(**saved["policy_config"])
    num_envs = saved["num_envs"]
    if saved.get("training_seed") is not None:
      env_cfg.seed = saved["training_seed"]
    del saved
  else:
    mean, cov = model_fit_configs(args)
    runner_cfg = SuperMimicRunnerCfg.for_estimator_inputs(
      args.group,
      dynamics_hidden_dims=tuple(args.mean_hidden_dims),
      actor_hidden_dims=tuple(args.actor_hidden_dims),
      critic_hidden_dims=tuple(args.critic_hidden_dims),
      process_noise=ProcessNoiseConfig(hidden_dims=tuple(args.cov_hidden_dims)),
      filter_config=FilterConfig(
        jacobian_backend=args.jacobian_backend,
        jacobian_batch_chunk_size=args.jacobian_batch_chunk_size,
        jacobian_chunk_size=args.jacobian_chunk_size,
      ),
      mean=mean,
      cov=cov,
      buffer_size=args.buffer_size,
      sampling_mode=args.sampling_mode,
      replay_ratio=args.replay_ratio,
    )
    runner_cfg.policy.batch_size = args.policy_microbatch_size
    loop_cfg = EITTrainingCfg(
      validation_stride=args.validation_stride,
      validation_batch_size=args.validation_batch_size,
      posterior_save_stride=args.posterior_save_stride,
    )
    policy_cfg = BeliefPolicyLearnerCfg(
      batch_size=args.policy_batch_size,
      num_samples=args.samples,
      microbatch_size=args.policy_microbatch_size,
      horizon=args.policy_horizon,
      num_batches=args.policy_updates,
      critic_epochs=args.critic_epochs,
      critic_minibatches=args.critic_minibatches,
      critic_microbatch_size=args.critic_microbatch_size,
      lr=args.policy_lr,
      critic_lr=args.critic_lr,
      noise=args.exploration,
    )
    num_envs = args.num_envs
  if initialization is None and args.resume is None:
    assert runner_cfg.estimator_inputs is not None
    noise_fingerprint = initialization_fingerprint(
      asdict(runner_cfg.estimator_inputs.measurement_noise)
    )[:12]
    initialization = args.output.parent / (
      f"{EIT_INITIALIZATION_VERSION}_{CRITIC_INPUT_VERSION}_"
      f"noise{noise_fingerprint}_seed{args.seed}.pt"
    )
  reference_path = None
  if args.resume_reference:
    try:
      reference_path = _reference_path(
        args.resume_reference,
        args.output,
        Path(args.motion_file),
        initialization,
        args.resume,
      )
    except ValueError as error:
      parser.error(str(error))
  runner_cfg.max_iterations = args.max_iterations
  for name in ("wandb_project", "run_name", "save_interval", "log_interval"):
    value = getattr(args, name)
    if value is not None:
      setattr(runner_cfg, name, value)
  motion.sampling_mode = (
    "external" if runner_cfg.sampling_mode == "return" else runner_cfg.sampling_mode
  )
  env = create_env(env_cfg, num_envs, args.device)
  try:
    runner = SuperMimicRunner(
      env, runner_cfg, task_id, device=args.device, compile_eit=compile_kernels
    )
    training = EITTraining(
      runner,
      loop_cfg,
      policy_cfg,
      policy_cuda_graphs=args.policy_cuda_graphs and not args.eager,
    )
    if reference_path is not None:
      from supermimic.runners.eit_resume_reference import write_live_reference

      def capture_live_reference(pending_checkpoint, state):
        if state["collected_steps"] == args.max_iterations:
          write_live_reference(training, pending_checkpoint, state, reference_path)

      training.before_checkpoint_commit = capture_live_reference
    if not args.resume:
      assert initialization is not None
      expected = None
      if args.prepare_initialization or not initialization.exists():
        initial = training.initialization_bundle()
        expected = (
          initialization_fingerprint(initial) if args.prepare_initialization else None
        )
        initialization.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
          dir=initialization.parent, delete_on_close=False
        ) as stream:
          torch.save(initial, stream)
          stream.close()
          try:
            os.link(stream.name, initialization)
          except FileExistsError:
            pass
        del initial
      if initialization.is_file():
        initial = torch.load(
          initialization, map_location=args.device, weights_only=True
        )
      else:
        raise ValueError(f"Initialization path is not a file: {initialization}")
      training.load_initialization(initial)
      if args.prepare_initialization:
        if (
          training.initialization_digest != expected
          and initialization_fingerprint(initial, include_model_revision=False)
          != expected
        ):
          raise ValueError(
            "Existing initialization differs from the requested seed/group/configuration"
          )
        print(f"EIT_INITIALIZATION {training.initialization_digest} {initialization}")
        return
    training.run(args.max_iterations, args.output, resume=args.resume)
  finally:
    env.close()


if __name__ == "__main__":
  main()
