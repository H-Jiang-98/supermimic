"""Evaluate EIT with mjlab's first-episode tracking protocol and metrics.

The loop follows mjlab.tasks.tracking.scripts.evaluate: start at the clip head,
score each environment's first episode, and exclude auto-reset terminal states.
Checkpoint/policy loading, EIT alignment, termination categories and belief
diagnostics are adapted here. An optional step budget is a censored smoke run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import mjlab.tasks  # noqa: F401
import torch
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.tracking.mdp.metrics import (
  compute_ee_orientation_error,
  compute_ee_position_error,
  compute_joint_velocity_error,
  compute_mpkpe,
  compute_root_relative_mpkpe,
)
from mjlab.utils.lab_api.math import quat_from_matrix

import supermimic.tasks  # noqa: F401
from supermimic.motion_loader import motion_content_digest
from supermimic.runners.eit_inference import (
  EITPolicyAdapter,
  evaluation_config,
  load_evaluation_models,
)
from supermimic.scripts.train_utils import create_env
from supermimic.tasks.env_cfgs import configure_estimator_env_cfg
from supermimic.utils.lie_state import LieState, local_error
from supermimic.utils.tracking_reward import _reanchor_targets

DEFAULT_EE_BODY_NAMES = (
  "left_ankle_roll_link",
  "right_ankle_roll_link",
  "left_wrist_yaw_link",
  "right_wrist_yaw_link",
)
_METRIC_NAMES = ("mpkpe", "r_mpkpe", "joint_vel_error", "ee_pos_error", "ee_ori_error")


def _belief_diagnostics(policy, obs):
  """Pre-action diagnostics for the very posterior consumed by the actor."""
  belief = policy.provider.state.belief
  truth = LieState.from_full(obs["full_state"])
  command = policy.env.unwrapped.command_manager.get_term("motion")
  reference = policy.wm.representation.reference_window(
    command.motion, command.time_steps, command.reference_alignment
  )
  diagnostics = policy.provider.diagnostics
  return {
    "error_squared": local_error(belief.mean, truth).square(),
    "variance": belief.covariance.diagonal(dim1=-2, dim2=-1),
    "tracking_squared": torch.stack(
      (
        (truth.position - reference.position[:, 0]).square().mean(-1),
        (truth.velocity - reference.velocity[:, 0]).square().mean(-1),
        (truth.joint_pos - reference.joint_pos[:, 0]).square().mean(-1),
      ),
      -1,
    ),
    "jitter": diagnostics["jitter"],
    "innovation_norm": diagnostics["innovation_norm"],
  }


@torch.no_grad()
def run_evaluation(env, policy, *, steps=None, ee_body_names=DEFAULT_EE_BODY_NAMES):
  """Score one episode per slot; the caller resets the physical environment."""
  if steps is not None and (type(steps) is not int or steps < 1):
    raise ValueError("steps must be a positive integer or None")
  command = env.unwrapped.command_manager.get_term("motion")
  if not ee_body_names or len(set(ee_body_names)) != len(ee_body_names):
    raise ValueError("End-effector names must be nonempty and unique")
  missing = set(ee_body_names) - set(command.cfg.body_names)
  if missing:
    raise ValueError(f"End-effector bodies are not tracked: {sorted(missing)}")
  obs = env.get_observations()
  policy.reset(obs)
  reference_tensor = obs["full_state"]
  batch, device = env.num_envs, reference_tensor.device
  finished = torch.zeros(batch, dtype=torch.bool, device=device)
  rewards = reference_tensor.new_zeros(batch)
  lengths = torch.zeros(batch, dtype=torch.long, device=device)
  metric_steps = torch.zeros_like(lengths)
  metric_sums = reference_tensor.new_zeros(batch, len(_METRIC_NAMES))
  error_sums = reference_tensor.new_zeros(batch, 70)
  variance_sums = torch.zeros_like(error_sums)
  tracking_sums = reference_tensor.new_zeros(batch, 3)
  start_frames = command.time_steps.clone()
  reference_steps = command.motion.last_frame(start_frames) - start_frames + 1
  records = [None] * batch
  jitter_slots, innovation_max, step_count = 0, 0.0, 0

  def episode(row, reason, *, failure=False, clip_end=False, timeout=False):
    return {
      "env_id": row,
      "start_frame": int(start_frames[row]),
      "reference_steps": int(reference_steps[row]),
      "return": float(rewards[row]),
      "length": int(lengths[row]),
      "reason": reason,
      "physical_failure": failure,
      "clip_end": clip_end,
      "timeout": timeout,
      "censored": reason == "evaluation_truncated",
    }

  while not finished.all() and (steps is None or step_count < steps):
    active_before = ~finished
    actions = policy.act(obs)
    diagnostic = _belief_diagnostics(policy, obs)
    for sums, name in (
      (error_sums, "error_squared"),
      (variance_sums, "variance"),
      (tracking_sums, "tracking_squared"),
    ):
      sums[active_before] += diagnostic[name][active_before]
    jitter_slots += int((diagnostic["jitter"][active_before] > 0).sum())
    innovation_max = max(
      innovation_max, float(diagnostic["innovation_norm"][active_before].max())
    )
    # Retain the scoring frame before command advancement, as in mjlab evaluate.
    ref = SimpleNamespace(
      num_envs=batch,
      device=device,
      cfg=command.cfg,
      body_pos_w=command.body_pos_w.clone(),
      body_quat_w=command.body_quat_w.clone(),
      joint_vel=command.joint_vel.clone(),
    )
    obs, reward, dones, _ = env.step(actions)
    dones = dones.bool()
    rewards[active_before] += reward[active_before]
    lengths += active_before.long()
    # Terminal rows already contain reset robot states: exclude those samples.
    active = active_before & ~dones
    ref.robot_body_pos_w = command.robot_body_pos_w
    ref.robot_body_quat_w = command.robot_body_quat_w
    ref.robot_joint_vel = command.robot_joint_vel
    # Re-anchor the old scoring reference at the CURRENT physical root, matching
    # EIT body rewards. Neither pre-step relative caches nor next-frame goals fit.
    ref.body_pos_relative_w, rotation = _reanchor_targets(
      ref.body_pos_w,
      ref.body_quat_w,
      command.robot_anchor_pos_w,
      command.robot_anchor_quat_w,
    )
    ref.body_quat_relative_w = quat_from_matrix(rotation)
    values = torch.stack(
      (
        compute_mpkpe(ref),
        compute_root_relative_mpkpe(ref),
        compute_joint_velocity_error(ref),
        compute_ee_position_error(ref, ee_body_names),
        compute_ee_orientation_error(ref, ee_body_names),
      ),
      -1,
    )
    if not torch.isfinite(values[active]).all():
      raise ValueError("Non-finite tracking metrics on an active evaluation frame")
    metric_sums[active] += values[active]
    metric_steps += active.long()
    manager = env.unwrapped.termination_manager
    failures = manager.terminated.bool()
    clips = manager.get_term("motion_end").bool()
    timeouts = manager.time_outs.bool()
    for row in (active_before & dones).nonzero().flatten().tolist():
      failure, clip_end, timeout = (
        bool(value[row]) for value in (failures, clips, timeouts)
      )
      reason = "failure" if failure else "clip_complete" if clip_end else "timeout"
      records[row] = episode(
        row, reason, failure=failure, clip_end=clip_end, timeout=timeout
      )
    finished |= dones
    policy.advance(obs, dones)
    step_count += 1

  for row in range(batch):
    if records[row] is None:
      records[row] = episode(row, "evaluation_truncated")
    records[row]["metric_steps"] = int(metric_steps[row])
    records[row]["metrics"] = {
      name: float(metric_sums[row, index] / metric_steps[row])
      if metric_steps[row]
      else None
      for index, name in enumerate(_METRIC_NAMES)
    }
  valid = metric_steps > 0
  metrics = {
    name: float((metric_sums[valid, index] / metric_steps[valid]).mean())
    if valid.any()
    else None
    for index, name in enumerate(_METRIC_NAMES)
  }
  completed = int(finished.sum())
  successes = sum(record["reason"] == "clip_complete" for record in records)
  # Diagnostics also average within each episode first, then across episodes.
  return {
    "protocol": "clip_head_first_episode_v1",
    "evaluation_complete": completed == batch,
    "steps": step_count,
    "step_budget": steps,
    "num_envs": batch,
    "episodes": records,
    "completed_episodes": completed,
    "censored_episodes": batch - completed,
    "success_rate": successes / batch if completed == batch else None,
    "mean_episode_return": float(rewards.mean()),
    "mean_episode_length": float(lengths.float().mean()),
    "failure_count": sum(record["reason"] == "failure" for record in records),
    "clip_complete_count": successes,
    "timeout_count": sum(record["reason"] == "timeout" for record in records),
    "metric_episode_count": int(valid.sum()),
    "tracking_metric_scope": "post_action_nonterminal_frames_episode_mean",
    "ee_body_names": list(ee_body_names),
    "posterior_rmse": (error_sums / lengths[:, None]).mean(0).sqrt().tolist(),
    "posterior_variance": (variance_sums / lengths[:, None]).mean(0).tolist(),
    "pre_action_tracking_rmse": dict(
      zip(
        ("root_position", "root_velocity", "joint_position"),
        (tracking_sums / lengths[:, None]).mean(0).sqrt().tolist(),
        strict=True,
      )
    ),
    "jitter_slots": jitter_slots,
    "innovation_norm_max": innovation_max,
    **metrics,
  }


def evaluate(
  checkpoint: Path,
  motion_file: Path,
  *,
  device="cpu",
  num_envs=2,
  steps=None,
  seed=2026,
  ee_body_names=DEFAULT_EE_BODY_NAMES,
) -> dict:
  if num_envs < 1 or (steps is not None and steps < 1):
    raise ValueError("num_envs and the optional steps budget must be positive")
  saved = torch.load(checkpoint, map_location=device, weights_only=True)
  cfg = evaluation_config(saved)
  torch.manual_seed(seed)
  task_id = "SuperMimic-GenCoord-G1"
  env_cfg = load_env_cfg(task_id)
  env_cfg.seed = seed
  env_cfg.commands["motion"].motion_file = str(motion_file)
  env_cfg.commands["motion"].sampling_mode = "start"
  env_cfg.events.pop("push_robot", None)
  configure_estimator_env_cfg(env_cfg)
  # A local stance avoids leaking global compiler configuration to callers.
  with torch.compiler.set_stance("force_eager"):
    env = create_env(env_cfg, num_envs, device)
    try:
      wm, actor = load_evaluation_models(env, cfg, saved, device)
      command = env.unwrapped.command_manager.get_term("motion")
      if command.motion.motion_ends.numel() != 1:
        raise ValueError("Evaluate one motion clip per call; pass each file separately")
      policy = EITPolicyAdapter(env, actor, wm, cfg)
      # Pair the initial reset/noise after architecture-dependent construction.
      # Subsequent trajectories are independent; this is not a noise-tape protocol.
      torch.manual_seed(seed)
      env.reset()
      result = run_evaluation(env, policy, steps=steps, ee_body_names=ee_body_names)
      result.update(
        group=cfg.estimator_inputs.group,
        mode=cfg.estimator_inputs.mode,
        device=str(device),
        seed=seed,
        training_motion_digest=saved["motion_digest"],
        evaluation_motion_digest=motion_content_digest(command.motion),
      )
      return result
    finally:
      env.close()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--motion-file", type=Path, required=True)
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--num-envs", type=int, default=2)
  parser.add_argument(
    "--steps", type=int, help="Optional smoke budget; unfinished episodes are censored"
  )
  parser.add_argument("--seed", type=int, default=2026)
  parser.add_argument("--ee-body-names", nargs="+", default=DEFAULT_EE_BODY_NAMES)
  parser.add_argument("--output", type=Path)
  args = parser.parse_args()
  result = evaluate(
    args.checkpoint,
    args.motion_file,
    device=args.device,
    num_envs=args.num_envs,
    steps=args.steps,
    seed=args.seed,
    ee_body_names=tuple(args.ee_body_names),
  )
  output = json.dumps(result, sort_keys=True, allow_nan=False)
  if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output + "\n")
  print("EIT_EVALUATION " + output)


if __name__ == "__main__":
  main()
