"""Fixed-input float32 measurements, usable against both source revisions."""

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
parser.add_argument("--rollout", action="store_true")
parser.add_argument(
  "--compare-rollout",
  action="store_true",
  help="Compare transactional and tensor rollout paths",
)
parser.add_argument("--batch", type=int, default=1024)
parser.add_argument("--horizon", type=int, default=32)
parser.add_argument("--repeats", type=int, default=3)
parser.add_argument("--small", action="store_true")
args = parser.parse_args()
torch.set_num_threads(1)
torch.manual_seed(2027)
device = "cuda:0"
reports = []


def measure(fn, **metadata):
  for _ in range(2):
    fn()
  torch.cuda.synchronize()
  torch.cuda.reset_peak_memory_stats()
  times = []
  for _ in range(args.repeats):
    start = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    times.append((time.perf_counter() - start) * 1000)
  row = dict(
    metadata,
    median_ms=statistics.median(times),
    times_ms=times,
    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
  )
  reports.append(row)
  print(json.dumps(row), flush=True)
  Path(args.output).write_text(json.dumps(reports, indent=2))


if not (args.rollout or args.compare_rollout):
  from tests.test_estimator_belief_filter import _fixture

  from supermimic.belief import BeliefState

  for batch in (1024, 2048):
    estimator, prior, _, _ = _fixture(batch, dtype=torch.float32)
    estimator.dynamics.to(device)
    estimator.default_joint_pos = estimator.default_joint_pos.to(device)
    prior = prior.to(device)
    is_sqrt = hasattr(prior, "covariance_sqrt")
    factor = (
      (prior.covariance_sqrt if is_sqrt else prior.covariance).detach().requires_grad_()
    )
    prior = (
      BeliefState(prior.mean, covariance_sqrt=factor)
      if is_sqrt
      else BeliefState(prior.mean, factor)
    )
    measurement = (estimator.observation(prior.mean) + 0.01).detach().requires_grad_()
    for grad in (False, True):

      def call(
        grad=grad,
        estimator=estimator,
        prior=prior,
        measurement=measurement,
        factor=factor,
      ):
        with torch.set_grad_enabled(grad):
          p = estimator.correct(prior, measurement)
          if grad:
            loss = (
              p.mean.to_full().square().mean()
              + p.to_actor_input().covariance_rpv.square().mean()
            )
            torch.autograd.grad(loss, (factor, measurement))

      measure(call, kind="correction", batch=batch, backward=grad, dtype="float32")
else:
  import mjlab.tasks  # noqa: F401
  import warp as wp
  from mjlab.tasks.registry import load_env_cfg

  import supermimic.tasks  # noqa: F401
  from supermimic.critic_inputs import BeliefCritic
  from supermimic.estimator import FilterConfig
  from supermimic.learners.belief_policy_learner import (
    BeliefActorCriticPolicyLearner,
    BeliefPolicyLearnerCfg,
  )
  from supermimic.models import ProcessNoiseConfig
  from supermimic.runners.supermimic_runner import SuperMimicRunner, SuperMimicRunnerCfg
  from supermimic.scripts.check_estimator import stationary_motion
  from supermimic.scripts.train_utils import create_env
  from supermimic.tasks.env_cfgs import configure_estimator_env_cfg

  wp.config.kernel_cache_dir = "/tmp/supermimic-srkf-warp"
  with tempfile.TemporaryDirectory(prefix="srkf-bench-") as directory:
    motion = Path(directory) / "motion.npz"
    stationary_motion(motion, frames=128)
    cfg = load_env_cfg("SuperMimic-GenCoord-G1")
    cfg.seed = 2027
    cfg.commands["motion"].motion_file = str(motion)
    cfg.commands["motion"].sampling_mode = "start"
    configure_estimator_env_cfg(cfg)
    env = create_env(cfg, 2, device)
    try:
      rcfg = SuperMimicRunnerCfg.for_estimator_inputs(
        "C",
        process_noise=ProcessNoiseConfig(hidden_dims=(128, 64)),
        filter_config=FilterConfig(
          jacobian_backend="increment", jacobian_batch_chunk_size=512
        ),
        dynamics_hidden_dims=(32, 32) if args.small else (2048, 2048, 2048),
        actor_hidden_dims=(32, 16) if args.small else (2048, 1024, 512, 256),
        critic_hidden_dims=(32, 16) if args.small else (256, 128),
        buffer_size=12,
        sampling_mode="start",
        replay_ratio=0,
      )
      rcfg.policy.batch_size = 2
      runner = SuperMimicRunner(
        env, rcfg, "SuperMimic-GenCoord-G1", device=device, compile_eit=True
      )
      runner.replay_buffer.enable_posterior_cache(1, runner.cfg.history_length)
      runner.configure_belief_filter()
      runner.reset_collection()
      for _ in range(4):
        runner.step(record_episode_stats=False)
      runner.finish_collection_round()
      actor = runner.policy_learner.actor
      critic = BeliefCritic(
        "C", runner._sim_context.output_dim, rcfg.critic_hidden_dims
      ).to(device)
      with torch.no_grad():
        actor.film[-1].weight.normal_(std=1e-3)
        critic.mlp[-1].weight.normal_(std=1e-3)
      policy = BeliefActorCriticPolicyLearner(
        actor,
        critic,
        runner.wm_env,
        BeliefPolicyLearnerCfg(
          batch_size=args.batch,
          microbatch_size=args.batch,
          num_samples=1,
          horizon=args.horizon,
          noise=0,
        ),
      )
      frames = runner.replay_buffer.retained_frames()
      ids = torch.arange(args.batch, device=device) % len(frames)
      starts = runner.wm_env.prepare_policy_starts(
        frames["episode_id"][ids], frames["time_index"][ids]
      )
      noises = [x[:, :, 0] for x in policy.draw_noises(starts)]
      runner._dynamics.requires_grad_(False)
      critic.requires_grad_(False)

      def rollout():
        actor.zero_grad(set_to_none=True)
        value, _, _ = policy._rollout(starts, *noises, args.horizon)
        (-value.mean()).backward()
        runner.wm_env.clear_rollout()

      metadata = dict(
        batch=args.batch,
        horizon=args.horizon,
        small=args.small,
        dtype="float32",
        compiled=True,
        torch_version=torch.__version__,
        gpu=torch.cuda.get_device_name(),
      )
      if args.compare_rollout:
        from supermimic.imagined_rollout import prepare_rollout

        def transactional():
          actor.zero_grad(set_to_none=True)
          result = policy._rollout_kernel(None, *noises, args.horizon, starts=starts)
          (-result.returns.mean()).backward()
          runner.wm_env.clear_rollout()

        measure(transactional, kind="transactional_rollout_backward", **metadata)
        measure(rollout, kind="prepared_rollout_backward", **metadata)
        prepared = prepare_rollout(runner.wm_env, starts, args.horizon, *noises)

        def tensor_rollout():
          actor.zero_grad(set_to_none=True)
          result = policy._rollout_kernel(prepared, *noises, args.horizon)
          (-result.returns.mean()).backward()
          if result.invalid.any():
            raise FloatingPointError("Invalid benchmark rollout")

        measure(tensor_rollout, kind="tensor_rollout_backward", **metadata)
      else:
        measure(rollout, kind="policy_rollout_backward", **metadata)
    finally:
      env.close()
