"""Evaluate one EIT checkpoint in an isolated process with online state estimation."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import mjlab.tasks  # noqa: F401
import torch
from mjlab.tasks.registry import load_env_cfg

import supermimic.tasks  # noqa: F401
from supermimic.estimator_inputs import EpisodeAlignment
from supermimic.motion_loader import motion_content_digest
from supermimic.runners.eit_training import EITTraining
from supermimic.runners.run_config import load_runner_cfg
from supermimic.runners.supermimic_runner import SuperMimicRunner, load_actor_checkpoint
from supermimic.scripts.train_utils import create_env
from supermimic.tasks.env_cfgs import configure_estimator_env_cfg
from supermimic.utils.lie_state import LieState, local_error


def evaluate(
  checkpoint: Path, motion_file: Path, *, device="cpu", num_envs=2, steps=100, seed=2026
) -> dict:
  if steps < 1 or num_envs < 1:
    raise ValueError("steps and num_envs must be positive")
  saved = torch.load(checkpoint, map_location=device, weights_only=True)
  EITTraining._check_checkpoint_version(saved)
  cfg = load_runner_cfg({"runner_cfg": saved["runner_config"]})
  cfg.estimator_inputs = replace(cfg.estimator_inputs, mode="eval")
  cfg.policy.noise = 0.0
  cfg.sampling_mode = "uniform"
  cfg.policy.batch_size = num_envs
  torch.manual_seed(seed)
  torch.compiler.set_stance("force_eager")
  task_id = "SuperMimic-GenCoord-G1"
  env_cfg = load_env_cfg(task_id)
  env_cfg.seed = seed
  env_cfg.commands["motion"].motion_file = str(motion_file)
  env_cfg.commands["motion"].sampling_mode = "uniform"
  configure_estimator_env_cfg(env_cfg)
  env = create_env(env_cfg, num_envs, device)
  try:
    runner = SuperMimicRunner(env, cfg, task_id, device=device)
    if motion_content_digest(runner.motion) != saved["motion_digest"]:
      raise ValueError("Evaluation motion content mismatch")
    runner._wm.load_bundle(saved["world_model"])
    load_actor_checkpoint(
      runner.policy_learner.actor, cfg, saved["actor"], for_evaluation=True
    )
    runner.policy_learner.actor.eval().requires_grad_(False)
    runner.configure_belief_filter()
    # Begin paired resets and measurements at the same seed for every actor architecture.
    torch.manual_seed(seed)
    runner.reset_collection()
    rewards = torch.zeros(num_envs, device=device)
    lengths = torch.zeros(num_envs, device=device, dtype=torch.long)
    episodes, errors, variances, tracking = [], [], [], []
    jitter_slots, innovation_max = 0, 0.0
    # Observe the same collection transaction consumed by runner.step, before
    # auto-reset state replaces the terminal state. Only this isolated replay is written.
    original_store = runner._store_collection_step

    def record(step):
      nonlocal rewards, lengths, jitter_slots, innovation_max
      original_store(step)
      _, reward, done, extras = step.result
      rewards += reward
      lengths += 1
      manager = runner.env.unwrapped.termination_manager
      clip_end = manager.get_term("motion_end")
      timeout = extras.get("time_outs", torch.zeros_like(done))
      for row in done.nonzero().flatten().tolist():
        failure = bool(manager.terminated[row])
        reason = (
          "failure"
          if failure
          else ("clip_complete" if bool(clip_end[row]) else "timeout")
        )
        episodes.append(
          {
            "return": float(rewards[row]),
            "length": int(lengths[row]),
            "reason": reason,
            "physical_failure": failure,
            "clip_end": bool(clip_end[row]),
            "timeout": bool(timeout[row]),
          }
        )
      rewards = torch.where(done.bool(), 0.0, rewards)
      lengths = torch.where(done.bool(), 0, lengths)
      belief = runner.posterior_provider.state.belief
      truth = LieState.from_full(step.fields["state"])
      error = local_error(belief.mean, truth)
      errors.append(error.detach().square().mean(0))
      variances.append(belief.covariance.diagonal(dim1=-2, dim2=-1).detach().mean(0))
      reference = runner._representation.reference_window(
        runner.motion,
        step.fields["motion_time"],
        EpisodeAlignment(
          step.fields["alignment_rotation"], step.fields["alignment_translation"]
        ),
      )
      tracking.append(
        torch.stack(
          (
            (truth.position - reference.position[:, 0]).square().mean(),
            (truth.velocity - reference.velocity[:, 0]).square().mean(),
            (truth.joint_pos - reference.joint_pos[:, 0]).square().mean(),
          )
        )
      )
      diagnostics = runner.posterior_provider.diagnostics
      jitter_slots += int((diagnostics["jitter"] > 0).sum())
      innovation_max = max(innovation_max, float(diagnostics["innovation_norm"].max()))

    runner._store_collection_step = record
    for _ in range(steps):
      runner.step(record_episode_stats=False)
    return {
      "group": cfg.estimator_inputs.group,
      "mode": cfg.estimator_inputs.mode,
      "device": device,
      "seed": seed,
      "steps": steps,
      "num_envs": num_envs,
      "episodes": episodes,
      "completed_episodes": len(episodes),
      "mean_episode_return": sum(e["return"] for e in episodes) / len(episodes)
      if episodes
      else None,
      "mean_episode_length": sum(e["length"] for e in episodes) / len(episodes)
      if episodes
      else None,
      "failure_count": sum(e["reason"] == "failure" for e in episodes),
      "clip_complete_count": sum(e["reason"] == "clip_complete" for e in episodes),
      "posterior_rmse": torch.stack(errors).mean(0).sqrt().tolist(),
      "posterior_variance": torch.stack(variances).mean(0).tolist(),
      "pre_action_tracking_rmse": dict(
        zip(
          ("root_position", "root_velocity", "joint_position"),
          torch.stack(tracking).mean(0).sqrt().tolist(),
          strict=True,
        )
      ),
      "jitter_slots": jitter_slots,
      "innovation_norm_max": innovation_max,
    }
  finally:
    env.close()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--motion-file", type=Path, required=True)
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--num-envs", type=int, default=2)
  parser.add_argument("--steps", type=int, default=100)
  parser.add_argument("--seed", type=int, default=2026)
  parser.add_argument("--output", type=Path)
  args = parser.parse_args()
  result = evaluate(
    args.checkpoint,
    args.motion_file,
    device=args.device,
    num_envs=args.num_envs,
    steps=args.steps,
    seed=args.seed,
  )
  output = json.dumps(result, sort_keys=True)
  if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output + "\n")
  print("EIT_EVALUATION " + output)


if __name__ == "__main__":
  main()
