"""Real continuous-round EIT acceptance and fixed-input model and policy continuation.

Use --device cuda:0 --full-width for the production network benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
import time
from copy import deepcopy
from itertools import pairwise
from pathlib import Path

import mjlab.tasks  # noqa: F401
import torch
import warp as wp
from mjlab.tasks.registry import load_env_cfg

import supermimic.tasks  # noqa: F401
from supermimic.learners.belief_policy_learner import BeliefPolicyLearnerCfg
from supermimic.learners.two_stage_wm import CovarianceFitCfg, MeanFitCfg
from supermimic.models import ProcessNoiseConfig
from supermimic.runners.eit_training import EITTraining, EITTrainingCfg
from supermimic.runners.supermimic_runner import SuperMimicRunner, SuperMimicRunnerCfg
from supermimic.scripts.check_estimator import stationary_motion
from supermimic.scripts.train_utils import create_env
from supermimic.tasks.env_cfgs import configure_estimator_env_cfg


def equal_tree(left, right):
  if isinstance(left, torch.Tensor):
    torch.testing.assert_close(left, right, atol=0, rtol=0, equal_nan=True)
  elif isinstance(left, dict):
    assert left.keys() == right.keys()
    for key in left:
      equal_tree(left[key], right[key])
  elif isinstance(left, (list, tuple)):
    assert len(left) == len(right)
    for a, b in zip(left, right, strict=True):
      equal_tree(a, b)
  else:
    assert left == right


def next_model_update(learner, windows, pairs):
  mean_loss = learner.update_mean(windows)
  cov_loss = learner.update_covariance(learner.residual_data(pairs))
  return mean_loss, cov_loss, learner.state_dict()


def parse_args(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--device", default="cpu")
  parser.add_argument(
    "--groups", nargs="+", choices=("A", "B", "C"), default=["A", "B", "C"]
  )
  parser.add_argument("--full-width", action="store_true")
  parser.add_argument("--policy-horizon", "--horizon", type=int, default=4)
  parser.add_argument("--samples", type=int, default=2)
  parser.add_argument("--rounds", type=int, default=3)
  parser.add_argument("--posterior-save-stride", type=int, default=1)
  parser.add_argument("--output", type=Path)
  return parser.parse_args(argv)


def main():
  args = parse_args()
  torch.set_num_threads(1)
  torch.compiler.set_stance("force_eager")
  wp.config.kernel_cache_dir = str(
    Path(tempfile.gettempdir()) / "supermimic-step4-warp"
  )
  cuda = torch.device(args.device).type == "cuda"
  if cuda:
    torch.cuda.set_device(args.device)
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(args.device)
  reports, initial = [], None
  with tempfile.TemporaryDirectory(prefix="supermimic-eit-") as temporary:
    directory = Path(temporary)
    stationary_motion(directory / "motion.npz")
    for group in args.groups:
      torch.manual_seed(2026)
      cfg = load_env_cfg("SuperMimic-GenCoord-G1")
      cfg.commands["motion"].motion_file = str(directory / "motion.npz")
      cfg.commands["motion"].sampling_mode = "start"
      configure_estimator_env_cfg(cfg)
      env = create_env(cfg, 2, args.device)
      try:
        runner_cfg = SuperMimicRunnerCfg.for_estimator_inputs(
          group,
          process_noise=ProcessNoiseConfig(
            hidden_dims=(128, 64) if args.full_width else (16,)
          ),
          dynamics_hidden_dims=(2048, 2048, 2048) if args.full_width else (32, 32),
          actor_hidden_dims=(2048, 1024, 512, 256) if args.full_width else (32, 16),
          critic_hidden_dims=(256, 128) if args.full_width else (32, 16),
          mean=MeanFitCfg(batch_size=4, window_frames=3, num_batches=2),
          cov=CovarianceFitCfg(batch_size=4),
          buffer_size=12,
          sampling_mode="start",
        )
        runner_cfg.policy.batch_size = 2
        runner = SuperMimicRunner(
          env, runner_cfg, "SuperMimic-GenCoord-G1", device=args.device
        )
        training = EITTraining(
          runner,
          EITTrainingCfg(
            collection_steps=8,
            validation_stride=2,
            validation_batch_size=2,
            posterior_save_stride=args.posterior_save_stride,
          ),
          BeliefPolicyLearnerCfg(
            batch_size=4,
            num_samples=args.samples,
            horizon=args.policy_horizon,
            microbatch_size=2,
            noise=0.0,
          ),
        )
        if initial is None:
          initial = training.initialization_bundle()
        training.load_initialization(initial)
        before = {
          name: deepcopy(module.state_dict())
          for name, module in (
            ("mean", runner._dynamics.mlp),
            ("noise", runner._dynamics.noise_head),
            ("actor", training.actor),
            ("critic", training.policy.critic),
          )
        }
        if cuda:
          torch.cuda.synchronize(args.device)
        begin = time.perf_counter()
        logs = training.run(args.rounds, directory / group)
        if cuda:
          torch.cuda.synchronize(args.device)
        elapsed = time.perf_counter() - begin
        for name, module in (
          ("mean", runner._dynamics.mlp),
          ("noise", runner._dynamics.noise_head),
          ("actor", training.actor),
          ("critic", training.policy.critic),
        ):
          assert any(
            not torch.equal(v, before[name][k]) for k, v in module.state_dict().items()
          ), name
        for previous, current in pairwise(logs):
          assert (
            current["collection_actor_version"] == previous["trained_actor_version"]
          )
          assert (
            current["collection_model_version"] == previous["trained_model_version"]
          )
        assert runner.imagined_posterior_provider.state is None
        train_ids, _ = training.episode_split()
        starts = runner.wm_env.sample_policy_starts(2, training.policy_episode_ids())
        noises = training.policy.draw_noises(starts)
        saved = torch.load(
          directory / group / "training.pt", map_location=args.device, weights_only=True
        )
        replay = runner.replay_buffer
        length = training.learner.mean_cfg.window_frames
        windows = replay.model_windows(
          replay.model_window_starts(train_ids, length)[:4], length
        )
        pairs = replay.model_windows(replay.model_window_starts(train_ids, 2)[:4], 2)
        keys = (starts.batch["episode_id"].clone(), starts.batch["time_index"].clone())
        expected_logs = training.policy.update_from(starts, *noises)
        expected = (
          deepcopy(training.actor.state_dict()),
          deepcopy(training.policy.critic.state_dict()),
          training.policy.state_dict(),
        )
        expected_model = next_model_update(training.learner, windows, pairs)
        # Supply exactly the same raw replay fixture after normal restore clears it.
        training.load_state_dict(saved)
        runner.replay_buffer = runner.wm_env.replay_buffer = replay
        restored_starts = runner.wm_env.prepare_policy_starts(*keys)
        restored_noises = training.policy.draw_noises(restored_starts)
        equal_tree(noises, restored_noises)
        actual_logs = training.policy.update_from(restored_starts, *restored_noises)
        equal_tree(expected_logs, actual_logs)
        equal_tree(
          expected,
          (
            training.actor.state_dict(),
            training.policy.critic.state_dict(),
            training.policy.state_dict(),
          ),
        )
        equal_tree(expected_model, next_model_update(training.learner, windows, pairs))
        reports.append(
          {
            "group": group,
            "critic_contract": training.policy.critic.contract(),
            "rounds": args.rounds,
            "posterior_save_stride": args.posterior_save_stride,
            "round_seconds": elapsed,
            "fixed_input_policy_resume_exact": True,
            "fixed_input_model_resume_exact": True,
            "metrics": logs,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(args.device)
            if cuda
            else None,
          }
        )
        # Evaluation is isolated in a child process, including simulator/RNG state.
        command = [
          str(Path(".venv/bin/python").absolute()),
          "-m",
          "supermimic.scripts.eval_eit",
          "--checkpoint",
          str(directory / group / "training.pt"),
          "--motion-file",
          str(directory / "motion.npz"),
          "--device",
          args.device,
          "--num-envs",
          "2",
          "--steps",
          "4",
        ]
        evaluated = subprocess.run(command, capture_output=True, text=True)
        if evaluated.returncode:
          raise RuntimeError(evaluated.stdout + evaluated.stderr)
        reports[-1]["evaluation"] = json.loads(
          evaluated.stdout.splitlines()[-1].split(" ", 1)[1]
        )
      finally:
        env.close()
  root = Path(__file__).resolve().parents[3]
  files = [
    "src/supermimic/scripts/check_eit_training.py",
    "src/supermimic/critic_inputs.py",
    "src/supermimic/learners/belief_policy_learner.py",
    "src/supermimic/learners/two_stage_wm.py",
    "src/supermimic/learners/wm_learner.py",
    "src/supermimic/models.py",
    "src/supermimic/replay_buffer.py",
    "src/supermimic/posterior_replay.py",
    "src/supermimic/runners/belief_training.py",
    "src/supermimic/runners/eit_training.py",
    "src/supermimic/runners/supermimic_runner.py",
    "src/supermimic/runners/run_config.py",
    "src/supermimic/runners/eit_inference.py",
    "src/supermimic/envs/world_model_env.py",
    "src/supermimic/belief_provider.py",
    "src/supermimic/utils/profiling.py",
    "src/supermimic/scripts/train_eit.py",
    "src/supermimic/scripts/eval_eit.py",
  ]
  report = {
    "status": "passed",
    "device": args.device,
    "device_name": torch.cuda.get_device_name(args.device) if cuda else "CPU",
    "torch": torch.__version__,
    "seed": 2026,
    "cpu_threads": 1,
    "dtype": "float32",
    "timing_scope": "full training rounds including checkpoint save and any first-use compilation; separate evaluation excluded",
    "memory_scope": "cumulative peak including initialization, training and restore validation",
    "full_width": args.full_width,
    "policy_horizon": args.policy_horizon,
    "samples": args.samples,
    "source_commit": subprocess.check_output(
      ["git", "rev-parse", "HEAD"], text=True, cwd=root
    ).strip(),
    "source_files_sha256": {
      f: hashlib.sha256((root / f).read_bytes()).hexdigest() for f in files
    },
    "groups": reports,
  }
  rendered = json.dumps(report, indent=2)
  if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n")
  print("EIT_SMOKE " + rendered)


if __name__ == "__main__":
  main()
