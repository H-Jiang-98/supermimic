"""Measure full EIT policy updates and fixed-noise derivatives on CPU/CUDA.

Uses a generated 32-frame MuJoCo reference and production-width networks by
 default. Each timing includes stochastic rollout, belief Jacobians, actor
 backward, critic bootstrap and five critic epochs with four minibatches. Replay prefix
 reconstruction is separately timed; physical collection is not in update time.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import tempfile
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import mjlab.tasks  # noqa: F401
import torch
import warp as wp
from mjlab.tasks.registry import load_env_cfg

import supermimic.tasks  # noqa: F401
from supermimic.critic_inputs import BeliefCritic
from supermimic.learners.belief_policy_learner import (
  BeliefActorCriticPolicyLearner,
  BeliefPolicyLearnerCfg,
  policy_metrics_for_json,
)
from supermimic.models import ProcessNoiseConfig
from supermimic.runners.supermimic_runner import SuperMimicRunner, SuperMimicRunnerCfg
from supermimic.scripts.check_estimator import stationary_motion
from supermimic.scripts.train_utils import create_env
from supermimic.tasks.env_cfgs import configure_estimator_env_cfg


def main():
  root = Path(__file__).resolve().parents[3]
  sources = [
    "src/supermimic/scripts/check_eit_scale.py",
    "src/supermimic/learners/belief_policy_learner.py",
    "src/supermimic/critic_inputs.py",
    "src/supermimic/envs/world_model_env.py",
    "src/supermimic/models.py",
    "src/supermimic/estimator.py",
    "src/supermimic/representations.py",
  ]
  source_hashes = {
    name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources
  }
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--small-width", action="store_true")
  parser.add_argument("--horizons", type=int, nargs="+", default=[2, 4, 8])
  parser.add_argument("--samples", type=int, nargs="+", default=[1, 2, 4])
  parser.add_argument("--batch-size", type=int, default=3)
  parser.add_argument("--microbatch-size", type=int, default=2)
  parser.add_argument("--warmup", type=int, default=1)
  parser.add_argument("--repeats", type=int, default=3)
  parser.add_argument("--output", type=Path)
  args = parser.parse_args()
  if (
    min(
      args.horizons
      + args.samples
      + [args.batch_size, args.microbatch_size, args.warmup, args.repeats]
    )
    < 1
  ):
    parser.error("Sizes and repetitions must be positive")
  torch.set_num_threads(1)
  torch.manual_seed(2026)
  torch.compiler.set_stance("force_eager")
  wp.config.kernel_cache_dir = str(
    Path(tempfile.gettempdir()) / "supermimic-step4-warp"
  )
  cuda = torch.device(args.device).type == "cuda"

  def sync():
    if cuda:
      torch.cuda.synchronize(args.device)

  with tempfile.TemporaryDirectory(prefix="supermimic-eit-scale-") as directory:
    motion_path = Path(directory) / "motion.npz"
    stationary_motion(motion_path, frames=32)
    cfg = load_env_cfg("SuperMimic-GenCoord-G1")
    cfg.commands["motion"].motion_file = str(motion_path)
    cfg.commands["motion"].sampling_mode = "start"
    configure_estimator_env_cfg(cfg)
    env = create_env(cfg, 2, args.device)
    try:
      runner_cfg = SuperMimicRunnerCfg.for_estimator_inputs(
        "belief",
        process_noise=ProcessNoiseConfig(
          hidden_dims=(16,) if args.small_width else (128, 64)
        ),
        dynamics_hidden_dims=(32, 32) if args.small_width else (2048, 2048, 2048),
        actor_hidden_dims=(32, 16) if args.small_width else (2048, 1024, 512, 256),
        critic_hidden_dims=(32, 16) if args.small_width else (256, 128),
        buffer_size=12,
        sampling_mode="start",
      )
      runner_cfg.policy.batch_size, runner_cfg.policy.noise = 2, 0.0
      runner = SuperMimicRunner(
        env, runner_cfg, "SuperMimic-GenCoord-G1", device=args.device
      )
      runner.replay_buffer.enable_posterior_cache(1, runner.cfg.history_length)
      runner.configure_belief_filter()
      runner.reset_collection()
      for _ in range(4):
        runner.step(record_episode_stats=False)
      runner.finish_collection_round()
      actor = runner.policy_learner.actor
      critic = BeliefCritic(
        "belief", runner._sim_context.output_dim, runner_cfg.critic_hidden_dims
      ).to(args.device)
      # Exercise FiLM and terminal-value input gradients before timed fitting.
      with torch.no_grad():
        actor.film[-1].weight.normal_(std=1e-3)
        critic.mlp[-1].weight.normal_(std=1e-3)
      initial_actor = {
        k: v.detach().cpu().clone() for k, v in actor.state_dict().items()
      }
      initial_critic = {
        k: v.detach().cpu().clone() for k, v in critic.state_dict().items()
      }
      model_before = runner._dynamics.model_version
      base_cfg = BeliefPolicyLearnerCfg(
        batch_size=args.batch_size, microbatch_size=args.microbatch_size, noise=0.0
      )

      def learner(policy_cfg):
        actor.load_state_dict(initial_actor)
        critic.load_state_dict(initial_critic)
        return BeliefActorCriticPolicyLearner(actor, critic, runner.wm_env, policy_cfg)

      policy = learner(base_cfg)
      frames = runner.replay_buffer.retained_frames()
      selected = torch.arange(args.batch_size, device=args.device) % len(frames)
      keys = (frames["episode_id"][selected], frames["time_index"][selected])
      sync()
      begin = time.perf_counter()
      starts = runner.wm_env.prepare_policy_starts(*keys)
      sync()
      cache_read_seconds = time.perf_counter() - begin

      # Full production derivative, with fixed process and measurement noises.
      numeric_cfg = replace(base_cfg, num_samples=1, horizon=2)
      policy = learner(numeric_cfg)
      one = starts.index(torch.tensor([0], device=args.device))
      process, measurement = (noise[:, :, 0] for noise in policy.draw_noises(one))
      runner._dynamics.requires_grad_(False)
      critic.requires_grad_(False)
      parameter = actor.decoder[-1].bias
      result = policy._rollout_result(one, process, measurement, 2)
      value = result.returns
      assert result.final_state is not None
      gradient, film_gradient, covariance_gradient = torch.autograd.grad(
        value.sum(),
        (
          parameter,
          actor.film[0].weight,
          result.final_state.belief.covariance_sqrt,
        ),
      )
      assert film_gradient.norm() > 0 and covariance_gradient.norm() > 0
      direction = gradient.detach() / gradient.norm()
      analytical = (gradient * direction).sum().detach()
      original = parameter.detach().clone()
      epsilon = 0.002
      values = []
      for sign in (1, -1):
        with torch.no_grad():
          parameter.copy_(original + sign * epsilon * direction)
        value = policy._rollout_result(one, process, measurement, 2).returns
        values.append(value.detach().sum())
      with torch.no_grad():
        parameter.copy_(original)
      finite_difference = (values[0] - values[1]) / (2 * epsilon)
      torch.testing.assert_close(analytical, finite_difference, rtol=0.03, atol=3e-4)
      assert gradient.norm() > 0
      runner.wm_env.clear_rollout()
      numerical = {
        "actor_directional_analytic": float(analytical),
        "actor_directional_finite_difference": float(finite_difference),
        "absolute_error": float((analytical - finite_difference).abs()),
        "relative_error": float(
          (analytical - finite_difference).abs()
          / torch.maximum(analytical.abs(), finite_difference.abs()).clamp_min(1e-8)
        ),
        "finite_difference_epsilon": epsilon,
        "film_hidden_gradient_norm": float(film_gradient.norm()),
        "terminal_covariance_sqrt_gradient_norm": float(covariance_gradient.norm()),
      }

      del (
        value,
        result,
        gradient,
        analytical,
        finite_difference,
        film_gradient,
        covariance_gradient,
      )

      # Compare all accumulated actor gradients with the same K samples.
      compare_cfg = replace(base_cfg, horizon=2, num_samples=2)
      records = []
      fixed_noises = learner(compare_cfg).draw_noises(starts)
      for micro in (1, len(starts) * 2):
        policy = learner(replace(compare_cfg, microbatch_size=micro))
        logs = policy.update_from(starts, *fixed_noises)
        records.append(
          (
            torch.cat([p.grad.detach().flatten().clone() for p in actor.parameters()]),
            logs["Policy/return"],
          )
        )
      gradient_difference = (records[0][0] - records[1][0]).norm() / records[0][
        0
      ].norm().clamp_min(1e-8)
      assert gradient_difference < 2e-4, gradient_difference
      torch.testing.assert_close(records[0][1], records[1][1], rtol=2e-5, atol=1e-6)
      numerical["microbatch_actor_gradient_relative_l2_error"] = float(
        gradient_difference
      )
      del records, fixed_noises
      reports = []
      for horizon in args.horizons:
        for samples in args.samples:
          policy = learner(replace(base_cfg, horizon=horizon, num_samples=samples))
          noises = policy.draw_noises(starts)
          for _ in range(args.warmup):
            policy.update_from(starts, *noises)
          gc.collect()
          sync()
          if cuda:
            torch.cuda.reset_peak_memory_stats(args.device)
          elapsed, retained = [], []
          for _ in range(args.repeats):
            sync()
            begin = time.perf_counter()
            logs = policy.update_from(starts, *noises)
            sync()
            elapsed.append(time.perf_counter() - begin)
            assert all(
              torch.isfinite(v).all()
              for k, v in logs.items()
              if k != "Policy/explained_variance"
            )
            assert runner.imagined_posterior_provider.state is None
            assert (
              int(logs["Policy/valid_steps"]) == args.batch_size * samples * horizon
            )
            retained.append(torch.cuda.memory_allocated(args.device) if cuda else 0)
          if cuda:
            assert max(retained) - min(retained) < 2 * 1024**2, retained
          reports.append(
            {
              "horizon": horizon,
              "samples": samples,
              "batch_size": args.batch_size,
              "microbatch_size": args.microbatch_size,
              "last_microbatch": (args.batch_size * samples - 1) % args.microbatch_size
              + 1,
              "seconds": elapsed,
              "mean_seconds": sum(elapsed) / len(elapsed),
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(args.device)
              if cuda
              else None,
              "retained_allocated_bytes": retained if cuda else None,
              "metrics": policy_metrics_for_json(logs),
            }
          )
          print(
            f"EIT_SCALE_CASE H={horizon} K={samples} seconds={reports[-1]['mean_seconds']:.4f}",
            flush=True,
          )
      assert runner._dynamics.model_version == model_before
      assert all(p.grad is None for p in runner._dynamics.parameters())
    finally:
      env.close()
  assert source_hashes == {
    name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources
  }, "Source files changed during validation; rerun against the final sources"
  report = {
    "status": "passed",
    "timestamp_utc": datetime.now(UTC).isoformat(),
    "device": args.device,
    "device_name": torch.cuda.get_device_name(args.device) if cuda else "CPU",
    "torch": torch.__version__,
    "dtype": "float32",
    "seed": 2026,
    "cpu_threads": 1,
    "small_width": args.small_width,
    "network_widths": {
      "mean": runner_cfg.dynamics_hidden_dims,
      "noise": runner_cfg.process_noise.hidden_dims,
      "actor": runner_cfg.actor_hidden_dims,
      "critic": runner_cfg.critic_hidden_dims,
    },
    "warmup": args.warmup,
    "repeats": args.repeats,
    "scope": "Fixed model C policy: stochastic rollout, full belief filter, RPV critic, actor backward and five critic epochs with four minibatches; excludes collection and prefix replay",
    "model_unchanged": True,
    "posterior_cache_read_seconds": cache_read_seconds,
    "numerical": numerical,
    "configurations": reports,
    "source_commit": subprocess.check_output(
      ["git", "rev-parse", "HEAD"], text=True, cwd=root
    ).strip(),
    "source_files_sha256": source_hashes,
  }
  rendered = json.dumps(report, indent=2)
  if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n")
  print("EIT_SCALE " + rendered)


if __name__ == "__main__":
  main()
