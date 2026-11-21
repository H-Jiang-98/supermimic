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
from tempfile import NamedTemporaryFile
from types import SimpleNamespace

import mjlab.tasks  # noqa: F401
import numpy as np
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
from supermimic.evaluation_diagnostics import EpisodeRecorder, PosteriorBins
from supermimic.motion_command import MotionCommand, MotionCommandCfg
from supermimic.motion_loader import MotionLoader, motion_content_digest
from supermimic.runners.eit_inference import (
  EITPolicyAdapter,
  evaluation_config,
  load_evaluation_models,
  load_hashed_checkpoint,
)
from supermimic.runners.eit_snapshots import file_sha256
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
TIME_LIMIT_PROTOCOLS = {"full_clip": "full_clip_v1", "legacy_10s": "legacy_10s_v1"}


def evaluation_env_config(motion_file: Path, seed: int, protocol: str = "full_clip"):
  """Configure the first episode with a timeout beyond its last scoring action."""
  if protocol not in TIME_LIMIT_PROTOCOLS:
    raise ValueError(f"Unknown evaluation time limit protocol: {protocol}")
  cfg = load_env_cfg("SuperMimic-GenCoord-G1")
  cfg.seed = seed
  motion_cfg = cfg.commands["motion"]
  if not isinstance(motion_cfg, MotionCommandCfg):
    raise ValueError("EIT evaluation requires the SuperMimic motion command")
  motion_cfg.motion_file = str(motion_file)
  motion_cfg.sampling_mode = "start"
  cfg.events.pop("push_robot", None)
  configure_estimator_env_cfg(cfg)
  if protocol == "full_clip":
    with np.load(motion_file, allow_pickle=False) as motion:
      frames = len(motion["joint_pos"])
    dt = cfg.sim.mujoco.timestep * cfg.decimation
    cfg.episode_length_s = max(cfg.episode_length_s, (frames + 1) * dt)
  return cfg


def _belief_diagnostics(policy, obs):
  """Pre-action diagnostics for the very posterior consumed by the actor."""
  belief = policy.provider.state.belief
  truth = LieState.from_full(obs["full_state"])
  command = policy.env.unwrapped.command_manager.get_term("motion")
  reference = policy.wm.representation.reference_window(
    command.motion, command.time_steps, command.reference_alignment
  )
  diagnostics = policy.provider.diagnostics
  error = local_error(belief.mean, truth)
  return {
    "rpv_error": error[:, :9],
    "position_error_w": truth.position - belief.mean.position,
    "velocity_error_w": truth.velocity - belief.mean.velocity,
    "error_squared": error.square(),
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
def run_evaluation(
  env,
  policy,
  *,
  steps=None,
  ee_body_names=DEFAULT_EE_BODY_NAMES,
  posterior_bin_steps=None,
  posterior_diagnostics=True,
  recorder=None,
):
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
  if not posterior_diagnostics and posterior_bin_steps is not None:
    raise ValueError("Time bins require posterior diagnostics")
  bins = (
    PosteriorBins(env.unwrapped.step_dt, posterior_bin_steps)
    if posterior_bin_steps is not None
    else None
  )
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
      "termination_terms": [],
      "physical_failure": failure,
      "clip_end": clip_end,
      "timeout": timeout,
      "censored": reason == "evaluation_truncated",
    }

  while not finished.all() and (steps is None or step_count < steps):
    active_before = ~finished
    actions = policy.act(obs)
    if recorder is not None:
      recorder.before(policy, obs, actions, active_before)
    if posterior_diagnostics:
      diagnostic = _belief_diagnostics(policy, obs)
      if bins is not None:
        state = policy.provider.state
        bins.update(
          state.episode_id[active_before],
          state.time_index[active_before],
          diagnostic["rpv_error"][active_before],
          state.belief.covariance[active_before, :9, :9],
          diagnostic["jitter"][active_before],
          diagnostic["innovation_norm"][active_before],
          diagnostic["position_error_w"][active_before],
          diagnostic["velocity_error_w"][active_before],
        )
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
    if recorder is not None:
      recorder.after(reward, active_before)
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
    # motion_end also contributes to the manager's aggregate time_outs flag.
    timeouts = manager.get_term("time_out").bool()
    for row in (active_before & dones).nonzero().flatten().tolist():
      failure, clip_end, timeout = (
        bool(value[row]) for value in (failures, clips, timeouts)
      )
      reason = "failure" if failure else "clip_complete" if clip_end else "timeout"
      records[row] = episode(
        row, reason, failure=failure, clip_end=clip_end, timeout=timeout
      )
      records[row]["termination_terms"] = [
        name for name in manager.active_terms if bool(manager.get_term(name)[row])
      ]
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
  diagnostics = (
    {
      "posterior_rmse": (error_sums / lengths[:, None]).mean(0).sqrt().tolist(),
      "posterior_variance": (variance_sums / lengths[:, None]).mean(0).tolist(),
      "pre_action_tracking_rmse": dict(
        zip(
          ("root_position", "root_velocity", "joint_position"),
          (tracking_sums / lengths[:, None]).mean(0).sqrt().tolist(),
          strict=True,
        )
      ),
      "posterior_time_bins": bins.result() if bins is not None else None,
      "jitter_slots": jitter_slots,
      "innovation_norm_max": innovation_max,
    }
    if posterior_diagnostics
    else {}
  )
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
    **diagnostics,
    **metrics,
  }


def _validate_output_paths(*, inputs, outputs):
  inputs = {Path(path).resolve() for path in inputs if path is not None}
  outputs = [Path(path).resolve() for path in outputs if path is not None]
  if len(set(outputs)) != len(outputs) or set(outputs).intersection(inputs):
    raise ValueError("Output paths must be distinct from inputs and each other")


def evaluate(
  checkpoint: Path,
  motion_file: Path,
  *,
  device="cpu",
  num_envs=2,
  steps=None,
  protocol="full_clip",
  seed=2026,
  posterior_bin_steps=None,
  posterior_diagnostics=True,
  record_path: Path | str | None = None,
  truth_actor=False,
  reference_controller=False,
  onnx_actor=None,
  export_onnx=None,
  ee_body_names=DEFAULT_EE_BODY_NAMES,
) -> dict:
  if num_envs < 1 or (steps is not None and steps < 1):
    raise ValueError("num_envs and the optional steps budget must be positive")
  if reference_controller and (
    not record_path or truth_actor or export_onnx or onnx_actor
  ):
    raise ValueError(
      "Reference controller requires a diagnostic recording and a standalone controller"
    )
  _validate_output_paths(
    inputs=(checkpoint, motion_file, onnx_actor),
    outputs=(record_path, export_onnx),
  )
  saved, checkpoint_sha = load_hashed_checkpoint(checkpoint, device)
  cfg = evaluation_config(saved)
  estimator_inputs = cfg.estimator_inputs
  assert estimator_inputs is not None  # Guaranteed by evaluation_config.
  motion_sha = file_sha256(Path(motion_file))
  torch.manual_seed(seed)
  env_cfg = evaluation_env_config(motion_file, seed, protocol)
  # A local stance avoids leaking global compiler configuration to callers.
  with torch.compiler.set_stance("force_eager"):
    env = create_env(env_cfg, num_envs, device)
    try:
      wm, actor = load_evaluation_models(env, cfg, saved, device)
      command = env.unwrapped.command_manager.get_term("motion")
      if not isinstance(command, MotionCommand) or not isinstance(
        command.motion, MotionLoader
      ):
        raise ValueError("EIT evaluation requires a loaded SuperMimic motion command")
      motion = command.motion
      if motion.motion_ends.numel() != 1:
        raise ValueError("Evaluate one motion clip per call; pass each file separately")
      onnx_backend = None
      backend_name = "pytorch"
      state_source = "truth" if truth_actor else "posterior"
      if export_onnx is not None or onnx_actor is not None:
        from supermimic.runners.exporter import export_eit_actor, load_eit_backend

        if export_onnx is not None:
          export_eit_actor(actor, env, wm, cfg, checkpoint_sha, Path(export_onnx))
        if onnx_actor is not None:
          backend_name = "onnx"
          onnx_backend = load_eit_backend(
            Path(onnx_actor),
            checkpoint_sha,
            estimator_inputs.group,
            device,
            num_envs,
          )
      backend = onnx_backend
      if reference_controller:
        backend_name = "joint_position_reference"
        state_source = "reference_joint_positions"
        action_offset = wm.dynamics.get_buffer("action_offset")
        action_scale = wm.dynamics.get_buffer("action_scale")

        def reference_action(_inputs):
          return (command.joint_pos - action_offset) / action_scale

        backend = reference_action
      policy = EITPolicyAdapter(
        env, actor, wm, cfg, truth_actor=truth_actor, actor_backend=backend
      )
      # Pair the initial reset/noise after architecture-dependent construction.
      # Subsequent trajectories are independent; this is not a noise-tape protocol.
      torch.manual_seed(seed)
      env.reset()
      recorder = EpisodeRecorder(num_envs) if record_path is not None else None
      result = run_evaluation(
        env,
        policy,
        steps=steps,
        ee_body_names=ee_body_names,
        posterior_bin_steps=posterior_bin_steps,
        posterior_diagnostics=posterior_diagnostics,
        recorder=recorder,
      )
      result.update(
        protocol="step_budget_v1"
        if steps is not None
        else TIME_LIMIT_PROTOCOLS[protocol],
        time_limit_protocol=TIME_LIMIT_PROTOCOLS[protocol],
        control_period_s=env.unwrapped.step_dt,
        clip_frames=int(motion.motion_ends.item()),
        timeout_steps=env.unwrapped.max_episode_length,
        timeout_s=env.unwrapped.max_episode_length_s,
        timeout_effective_s=env.unwrapped.max_episode_length * env.unwrapped.step_dt,
        checkpoint_sha256=checkpoint_sha,
        actor_state_source=state_source,
        actor_backend=backend_name,
        group=estimator_inputs.group,
        mode=estimator_inputs.mode,
        device=str(device),
        seed=seed,
        training_motion_digest=saved["motion_digest"],
        evaluation_motion_sha256=motion_sha,
        evaluation_motion_digest=motion_content_digest(motion),
      )
      if onnx_backend is not None:
        result["onnx_timing"] = {
          "runtime": "onnx.reference.ReferenceEvaluator",
          "device": "cpu",
          "calls": onnx_backend.calls,
          "mean_actor_ms": onnx_backend.total_ns / onnx_backend.calls / 1e6,
        }
      if recorder is not None:
        assert record_path is not None
        path = Path(record_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = dict(
          version="eit_diagnostic_episodes_v1",
          source="mujoco_evaluation",
          split="diagnostic",
          checkpoint_sha256=checkpoint_sha,
          group=result["group"],
          actor_state_source=result["actor_state_source"],
          actor_backend=result["actor_backend"],
          control_dt=env.unwrapped.step_dt,
          max_step_reward=env.unwrapped.step_dt
          * sum(term.weight for term in env_cfg.rewards.values() if term.weight > 0),
          sim_context={k: v.cpu() for k, v in wm.sim_context.state_dict().items()},
          seed=seed,
          motion_file=str(Path(motion_file).resolve()),
          motion_sha256=motion_sha,
          motion_digest=result["evaluation_motion_digest"],
          episodes=recorder.episodes(result["episodes"]),
        )
        with NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
          temporary = Path(tmp.name)
          try:
            torch.save(data, tmp)
            tmp.flush()
            temporary.replace(path)
          finally:
            temporary.unlink(missing_ok=True)
        result["diagnostic_record"] = str(path)
      return result
    finally:
      env.close()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--motion-file", type=Path, required=True)
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--protocol", choices=TIME_LIMIT_PROTOCOLS, default="full_clip")
  parser.add_argument("--num-envs", type=int, default=2)
  parser.add_argument(
    "--steps", type=int, help="Optional smoke budget; unfinished episodes are censored"
  )
  parser.add_argument("--seed", type=int, default=2026)
  parser.add_argument("--ee-body-names", nargs="+", default=DEFAULT_EE_BODY_NAMES)
  parser.add_argument(
    "--posterior-bin-steps",
    type=int,
    help="Enable time-resolved RPV calibration at this bin width",
  )
  parser.add_argument(
    "--record", type=Path, help="Save independent episodes for offline diagnostics"
  )
  parser.add_argument(
    "--reference-controller",
    action="store_true",
    help="Diagnostic recording with physical PD targets taken from reference joint positions",
  )
  parser.add_argument(
    "--truth-actor", action="store_true", help="Group A training-input diagnostic"
  )
  parser.add_argument(
    "--export-onnx", type=Path, help="Export this checkpoint's CPU batch-one EIT graph"
  )
  parser.add_argument(
    "--onnx-actor",
    type=Path,
    help="Evaluate a matching exported graph on CPU with one environment",
  )
  parser.add_argument("--no-posterior-diagnostics", action="store_true")
  parser.add_argument("--output", type=Path)
  args = parser.parse_args()
  _validate_output_paths(
    inputs=(args.checkpoint, args.motion_file, args.onnx_actor),
    outputs=(args.record, args.export_onnx, args.output),
  )
  result = evaluate(
    args.checkpoint,
    args.motion_file,
    device=args.device,
    num_envs=args.num_envs,
    steps=args.steps,
    protocol=args.protocol,
    seed=args.seed,
    posterior_bin_steps=args.posterior_bin_steps,
    posterior_diagnostics=not args.no_posterior_diagnostics,
    record_path=args.record,
    truth_actor=args.truth_actor,
    reference_controller=args.reference_controller,
    onnx_actor=args.onnx_actor,
    export_onnx=args.export_onnx,
    ee_body_names=tuple(args.ee_body_names),
  )
  output = json.dumps(result, sort_keys=True, allow_nan=False)
  if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output + "\n")
  print("EIT_EVALUATION " + output)


if __name__ == "__main__":
  main()
