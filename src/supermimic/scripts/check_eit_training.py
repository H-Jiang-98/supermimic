"""Real step-driven EIT acceptance and fixed-input model and policy continuation.

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
  parser.add_argument("--max-iterations", type=int, default=12)
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
          cov=CovarianceFitCfg(batch_size=4, validation_pairs=8),
          replay_ratio=3,
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
        from supermimic.runners.eit_resume_reference import write_live_reference
        from supermimic.scripts.check_eit_resume import verify

        reference = directory / f"{group}-reference.pt"

        def capture(path, state, training=training, reference=reference):
          if state["collected_steps"] == args.max_iterations:
            write_live_reference(training, path, state, reference)

        training.before_checkpoint_commit = capture
        logs = training.run(args.max_iterations, directory / group)
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
        verify(
          directory / group / "training.pt",
          directory / "motion.npz",
          args.device,
          reference,
        )
        reports.append(
          {
            "group": group,
            "critic_contract": training.policy.critic.contract(),
            "collected_steps": args.max_iterations,
            "posterior_save_stride": args.posterior_save_stride,
            "training_seconds": elapsed,
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
    "timing_scope": "physical collection and all training updates including checkpoint save and any first-use compilation; separate evaluation excluded",
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
