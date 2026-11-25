"""Online C square-root filter and independently loaded actors, at one frame boundary."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from time import monotonic

import numpy as np
import torch
from mjlab.managers.termination_manager import TerminationManager
from mjlab.utils.lab_api.math import quat_from_matrix

from supermimic.belief import BeliefState
from supermimic.estimator import LieBeliefFilter
from supermimic.estimator_inputs import RpvState, build_actor_features, sample_proprio
from supermimic.runners.eit_inference import evaluation_config
from supermimic.runners.supermimic_runner import (
  build_actor,
  build_world_model,
  load_actor_checkpoint,
)
from supermimic.scripts.eval_eit import evaluation_env_config
from supermimic.scripts.train_utils import create_env
from supermimic.tasks.task_spec import build_actor_obs_td
from supermimic.utils.helpers import normalized_action_limits
from supermimic.utils.lie_state import LieState
from supermimic.validation.config import scenario_seed
from supermimic.validation.domains import apply_domains
from supermimic.validation.metrics import (
  ODOMETRY,
  TRACKING,
  BinnedMetrics,
  posterior_values,
  tracking_values,
)


def make_env_config(cfg, motion):
  env_cfg = evaluation_env_config(Path(motion), cfg.evaluation_seed)
  env_cfg.sim.mujoco.iterations = cfg.iterations
  env_cfg.sim.mujoco.ls_iterations = cfg.ls_iterations
  env_cfg.auto_reset = False
  # Separate failure monitoring: auto_reset=False alone prohibits further steps
  # after any done, so no term in the live manager may raise done.
  failure_terms = {k: v for k, v in env_cfg.terminations.items() if not v.time_out}
  env_cfg.terminations = {}
  return env_cfg, failure_terms


def load_split_models(env, estimator_saved, actor_saved, device):
  """Retain actor group/normalization while using C's frozen estimator bundle."""
  estimator_cfg = evaluation_config(estimator_saved)
  actor_cfg = evaluation_config(actor_saved)
  if estimator_cfg.estimator_inputs.group != "C":
    raise ValueError("This study requires a C estimator")
  for key in ("measurement_noise", "history_length"):
    if getattr(actor_cfg.estimator_inputs, key) != getattr(
      estimator_cfg.estimator_inputs, key
    ):
      raise ValueError(f"Actor and estimator disagree on {key}")
  for key in ("action_scale", "action_offset"):
    a = actor_saved["world_model"]["dynamics"]["state_dict"][key]
    b = estimator_saved["world_model"]["dynamics"]["state_dict"][key]
    if not torch.equal(a, b):
      raise ValueError(f"Incompatible action contract: {key}")
  wm = build_world_model(estimator_cfg, env, device)
  wm.load_bundle(estimator_saved["world_model"])
  actor = build_actor(actor_cfg, wm, device)
  load_actor_checkpoint(actor, actor_cfg, actor_saved["actor"], for_evaluation=True)
  wm.dynamics.eval().requires_grad_(False)
  actor.eval().requires_grad_(False)
  return wm, actor, estimator_cfg, actor_cfg


class PairedSensors:
  """CPU streams per context, independent of device, batching and actor group."""

  def __init__(self, cfg, sha, ids, measurement_config):
    self.config = measurement_config
    self.generators = [
      torch.Generator().manual_seed(
        scenario_seed(cfg.evaluation_seed, sha, i, "sensor")
      )
      for i in ids
    ]

  def sample(self, clean):
    cpu = clean.detach().cpu()
    return torch.cat(
      [
        sample_proprio(cpu[i : i + 1], self.config, generator=g)
        for i, g in enumerate(self.generators)
      ]
    ).to(clean.device)


def paired_prior(cfg, sha, ids, truth, factory):
  cpu = truth.to("cpu")
  priors = [
    factory.sample(
      cpu.index(slice(i, i + 1)),
      generator=torch.Generator().manual_seed(
        scenario_seed(cfg.evaluation_seed, sha, context, "prior")
      ),
    )
    for i, context in enumerate(ids)
  ]
  return BeliefState(
    LieState.from_full(torch.cat([p.mean.to_full() for p in priors])),
    covariance_sqrt=torch.cat([p.covariance_sqrt for p in priors]),
  ).to(truth.position.device)


@torch.no_grad()
def run_batch(
  cfg, kind, motion, sha, ids, table, variant, estimator_saved, actor_saved
):
  device = cfg.device
  env_cfg, failure_terms = make_env_config(cfg, motion)
  env = create_env(env_cfg, len(ids), device)
  frame = -1
  started = last_progress = monotonic()
  try:
    wm, actor, ecfg, acfg = load_split_models(env, estimator_saved, actor_saved, device)
    raw = env.unwrapped
    command = raw.command_manager.get_term("motion")
    dt = raw.step_dt
    if not np.isclose(dt, estimator_saved["world_model"]["dynamics"]["metadata"]["dt"]):
      raise ValueError("Simulation and estimator control periods differ")
    width = round(cfg.time_bin_s / dt)
    if width < 1 or not np.isclose(width * dt, cfg.time_bin_s):
      raise ValueError("time_bin_s must be a positive multiple of control_dt")
    contexts = apply_domains(raw, wm.sim_context, [table["values"][i] for i in ids])
    # Per-row resets make reset perturbations independent of batch shape.
    for row, context in enumerate(ids):
      torch.manual_seed(scenario_seed(cfg.evaluation_seed, sha, context, "reset"))
      raw.reset(env_ids=torch.tensor([row], device=device))
    obs = env.get_observations()
    if command.time_steps.any():
      raise ValueError("Evaluation must start at frame zero")
    monitor = TerminationManager(failure_terms, raw)
    truth = LieState.from_full(obs["full_state"])
    belief = paired_prior(cfg, sha, ids, truth, wm.initial_prior)
    sensors = PairedSensors(cfg, sha, ids, ecfg.estimator_inputs.measurement_noise)
    estimator = LieBeliefFilter(
      wm.dynamics,
      ecfg.estimator_inputs.measurement_noise,
      wm.representation._default_joint_pos,
      config=wm.filter_config,
    )
    actor_inputs = (
      replace(acfg.estimator_inputs, mode="train")
      if variant == "A-truth"
      else acfg.estimator_inputs
    )
    lower, upper = normalized_action_limits(env, wm.num_actions, device=device)
    names = ODOMETRY if kind == "odometry" else TRACKING
    metrics = BinnedMetrics(names, len(ids), width, device)
    total_frames = int(command.motion.time_step_total)
    frames = min(total_frames, cfg.max_frames) if cfg.max_frames else total_frames
    ee = [command.cfg.body_names.index(name) for name in cfg.ee_body_names]
    history, previous_action = None, None
    first_failure = np.full(len(ids), np.nan)
    failure_names = [[] for _ in ids]
    replay_rows = [
      row
      for row, context in enumerate(ids)
      if kind == "tracking" and context in cfg.replay_ids()
    ]
    recorded = {key: [] for key in ("truth", "posterior", "reference")}
    posterior_status = "ok"
    initial_truth = truth.to_full().cpu().numpy()
    initial_prior = belief.mean.to_full().cpu().numpy()
    for frame in range(frames):
      now = monotonic()
      if now - last_progress >= cfg.progress_interval_s:
        print(
          f"  {Path(motion).name} {variant}: frame {frame}/{frames}, "
          f"{frame / max(now - started, 1e-9):.1f} control steps/s",
          flush=True,
        )
        last_progress = now
      if not torch.equal(
        command.time_steps, torch.full_like(command.time_steps, frame)
      ):
        raise ValueError(
          "Reference/frame alignment changed during full-clip evaluation"
        )
      truth = LieState.from_full(obs["full_state"])
      sensor = sensors.sample(obs["actor"])
      history = (
        sensor[:, None].repeat(1, acfg.history_length, 1)
        if history is None
        else torch.cat((history[:, 1:], sensor[:, None]), 1)
      )
      # A-truth's estimator is a spectator; a numerical filter failure must not
      # alter its actions. Preserve missing posterior frames explicitly.
      if posterior_status == "ok":
        try:
          belief = (
            estimator.correct(belief, sensor[:, :64])
            if previous_action is None
            else estimator.predict_correct(
              belief,
              previous_action,
              torch.as_tensor(contexts, device=device),
              sensor[:, :64],
            )
          )
          belief.validate()
        except (ValueError, RuntimeError) as exc:
          if variant != "A-truth":
            raise
          posterior_status = f"failed_at_frame_{frame}: {exc}"
      reference = wm.representation.reference_window(
        command.motion, command.time_steps, command.reference_alignment
      )
      values = (
        posterior_values(belief, truth)
        if kind == "odometry"
        else tracking_values(truth, reference, command, ee)
      )
      metrics.update(values)
      monitor.compute()
      failed = monitor.terminated.cpu().numpy()
      for row in np.flatnonzero(failed & np.isnan(first_failure)):
        first_failure[row] = frame * dt
        failure_names[row] = [
          name for name in monitor.active_terms if bool(monitor.get_term(name)[row])
        ]
      if replay_rows:
        ref_pose = torch.cat(
          (
            reference.position[:, 0],
            quat_from_matrix(reference.rotation[:, 0]),
            command.joint_pos,
          ),
          -1,
        )
        posterior = (
          belief.mean.to_full()[:, :36]
          if posterior_status == "ok"
          else torch.full_like(obs["full_state"][:, :36], float("nan"))
        )
        for key, value in (
          ("truth", obs["full_state"][:, :36]),
          ("posterior", posterior),
          ("reference", ref_pose),
        ):
          recorded[key].append(value[replay_rows].cpu().numpy().copy())
      if frame + 1 == frames:
        break
      goal, uncertainty = build_actor_features(
        actor_inputs,
        RpvState(truth.rotation, truth.position, truth.velocity),
        belief.to_actor_input() if posterior_status == "ok" else None,
        reference,
      )
      actor_obs = build_actor_obs_td(
        {"motion": goal}, history, task_uncertainty=uncertainty
      )
      actions = actor(actor_obs).clamp(min=lower, max=upper)
      if not torch.isfinite(actions).all():
        raise ValueError("Non-finite actor action")
      obs, _, dones, _ = env.step(actions)
      if dones.any():
        raise ValueError("Unexpected reset/termination in full-clip study")
      previous_action = actions
    result = metrics.result()
    result.update(
      context_ids=np.array(ids),
      context_normalized=contexts,
      initial_truth=initial_truth,
      initial_prior=initial_prior,
      first_failure_s=first_failure,
    )
    for key, values in recorded.items():
      if values:
        result["replay_" + key] = np.stack(values)
    result["replay_context_ids"] = np.array(
      [ids[row] for row in replay_rows], dtype=np.int64
    )
    return result, dict(
      status="complete" if frames == total_frames else "censored",
      control_dt=dt,
      clip_frames=total_frames,
      scored_frames=frames,
      metric_names=list(names),
      failure_terms=failure_names,
      posterior_status=posterior_status,
      elapsed_s=monotonic() - started,
    )
  except Exception as exc:
    raise RuntimeError(
      f"{variant}, clip={Path(motion).name}, contexts={ids}, frame={frame}: {exc}"
    ) from exc
  finally:
    env.close()
