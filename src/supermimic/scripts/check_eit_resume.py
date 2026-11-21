"""Verify the next model/actor/critic update of an actual EIT checkpoint.

The independent physical fixture and trial updates are discarded. This command
never replaces a training checkpoint or adds frames to its training budget.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import mjlab.tasks  # noqa: F401
import torch
from mjlab.tasks.registry import load_env_cfg

import supermimic.tasks  # noqa: F401
from supermimic.learners.belief_policy_learner import BeliefPolicyLearnerCfg
from supermimic.motion_command import MotionCommandCfg
from supermimic.runners.eit_inference import load_hashed_checkpoint
from supermimic.runners.eit_resume_reference import (
  REFERENCE_SOURCE,
  REFERENCE_VERSION,
  compare_reference_update,
  equal_tree,  # noqa: F401 — public import retained for callers and tests.
)
from supermimic.runners.eit_snapshots import file_sha256
from supermimic.runners.eit_training import EITTraining, EITTrainingCfg
from supermimic.runners.run_config import load_runner_cfg
from supermimic.runners.supermimic_runner import SuperMimicRunner
from supermimic.scripts.train_utils import create_env
from supermimic.tasks.env_cfgs import configure_estimator_env_cfg


def _verify(checkpoint: Path, motion_file: Path, device: str, reference: Path):
  saved, digest = load_hashed_checkpoint(checkpoint, device)
  evidence, reference_digest = load_hashed_checkpoint(reference, "cpu")
  EITTraining._check_checkpoint_version(saved)
  cfg = load_runner_cfg({"runner_cfg": saved["runner_config"]})
  inputs = cfg.estimator_inputs
  assert inputs is not None
  expected_identity = {
    "version": REFERENCE_VERSION,
    "source": REFERENCE_SOURCE,
    "checkpoint_sha256": digest,
    "round": saved["rounds"],
    "group": inputs.group,
    "motion_digest": saved["motion_digest"],
    "initialization_digest": saved.get("initialization_digest"),
    "device": str(torch.device(device)),
  }
  for key, expected in expected_identity.items():
    if evidence.get(key) != expected:
      raise ValueError(f"Live update reference {key} does not match checkpoint")
  loop = EITTrainingCfg(**saved["loop_config"])
  policy_cfg = BeliefPolicyLearnerCfg(**saved["policy_config"])
  env_cfg = load_env_cfg("SuperMimic-GenCoord-G1")
  env_cfg.seed = saved["training_seed"]
  motion = cast(MotionCommandCfg, env_cfg.commands["motion"])
  motion.motion_file = str(motion_file.resolve())
  motion.sampling_mode = (
    "external" if cfg.sampling_mode == "return" else cfg.sampling_mode
  )
  configure_estimator_env_cfg(env_cfg)
  env = create_env(env_cfg, saved["num_envs"], device)
  try:
    runner = SuperMimicRunner(env, cfg, "SuperMimic-GenCoord-G1", device=device)
    training = EITTraining(runner, loop, policy_cfg)
    # Only the actual side loads the checkpoint. Expected parameters, moments and
    # noises were generated from a private copy of live training before commit.
    training.load_state_dict(saved)
    details = compare_reference_update(training, evidence)
    if file_sha256(checkpoint) != digest or file_sha256(reference) != reference_digest:
      raise RuntimeError("Checkpoint or live reference changed during verification")
    return {
      "version": "eit_fixed_input_resume_v2",
      "status": "passed",
      "checkpoint_sha256": digest,
      "round": saved["rounds"],
      "group": inputs.group,
      "device": str(torch.device(device)),
      "reference_path": str(reference.resolve()),
      "reference_sha256": reference_digest,
      "reference_version": REFERENCE_VERSION,
      "expected_source": REFERENCE_SOURCE,
      "live_optimizer_reference": True,
      "fixed_input_policy_resume_exact": True,
      "fixed_input_model_resume_exact": True,
      "fixture_frames_excluded_from_training": 0,
      **details,
      "comparison": "bitwise parameters, optimizer state, noises and update metrics against pre-restore live reference",
      "verification_execution": evidence["verification_execution"],
    }
  finally:
    env.close()


def verify(checkpoint: Path, motion_file: Path, device: str, reference: Path):
  with torch.compiler.set_stance("force_eager"), torch.jit.optimized_execution(False):
    return _verify(checkpoint, motion_file, device, reference)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--motion-file", type=Path, required=True)
  parser.add_argument("--reference", type=Path, required=True)
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  if args.output.resolve() in (
    args.checkpoint.resolve(),
    args.motion_file.resolve(),
    args.reference.resolve(),
  ):
    parser.error("output must differ from inputs")
  torch.set_num_threads(1)
  result = verify(args.checkpoint, args.motion_file, args.device, args.reference)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
  print("EIT_RESUME " + json.dumps(result))


if __name__ == "__main__":
  main()
