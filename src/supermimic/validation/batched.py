"""Persistent multi-clip tracking runtime with device-side failure isolation."""

from __future__ import annotations

from collections import deque
from dataclasses import fields, replace
from time import monotonic

import numpy as np
import torch
from mjlab.managers.termination_manager import TerminationManager
from mjlab.utils.lab_api.math import quat_from_matrix

from supermimic.belief import BeliefState
from supermimic.estimator import LieBeliefFilter, belief_invalid
from supermimic.estimator_inputs import (
  RpvState,
  build_actor_critic_goals_kernel,
  build_actor_features,
  covariance_encoding_constants,
)
from supermimic.scripts.train_utils import create_env
from supermimic.tasks.task_spec import build_actor_obs_td
from supermimic.utils.compile import compile_eit_kernel
from supermimic.utils.helpers import normalized_action_limits
from supermimic.utils.lie_state import LieState
from supermimic.validation.config import scenario_seed
from supermimic.validation.domains import apply_domains
from supermimic.validation.metrics import (
  LOCAL_METRICS,
  ODOMETRY,
  TRACKING,
  local_posterior_values,
  posterior_values_kernel,
  relative_pose_squared,
)
from supermimic.validation.runtime import (
  full_clip_step,
  load_split_models,
  make_env_config,
  paired_prior,
)


def batches(work, capacity):
  """Length-sort whole per-clip tasks; scenario identity never uses a slot."""
  current, count = [], 0
  for task in sorted(work, key=lambda t: (-t[1]["frames"], t[1]["name"], t[3])):
    if len(task[3]) > capacity:
      raise ValueError("Task exceeds environment capacity")
    if count + len(task[3]) > capacity:
      yield current
      current, count = [], 0
    current.append(task)
    count += len(task[3])
  if current:
    yield current


def noise_block(cfg, scenarios, frames, measurement, device):
  """CPU draws preserve the original sequence of contiguous 1x64 calls."""
  std = measurement.standard_deviations(29, device="cpu", dtype=torch.float32)
  result = torch.empty(frames, len(scenarios), 64)
  for row, (sha, context) in enumerate(scenarios):
    generator = torch.Generator().manual_seed(
      scenario_seed(cfg.evaluation_seed, sha, context, "sensor")
    )
    # 64 is an exact CPU normal-vector block. Tested against framewise draws.
    result[:, row] = torch.randn(frames, 64, generator=generator) * std
  return result.to(device)


def select_state(mask, state, fallback):
  return LieState(
    *(
      torch.where(
        mask.reshape(-1, *([1] * (value.ndim - 1))),
        value,
        getattr(fallback, field.name),
      )
      for field in fields(state)
      for value in (getattr(state, field.name),)
    )
  )


def isolate_belief(mask, belief, fallback):
  return BeliefState(
    select_state(mask, belief.mean, fallback.mean),
    covariance_sqrt=torch.where(
      mask[:, None, None], belief.covariance_sqrt, fallback.covariance_sqrt
    ),
  )


class DeviceStatus:
  def __init__(self, lengths):
    self.lengths = lengths
    self.first_invalid = torch.full_like(lengths, -1)
    self.posterior_invalid = torch.full_like(lengths, -1)

  def active(self, frame):
    return (frame < self.lengths) & (self.first_invalid < 0)

  def reject(self, bad, frame, *, posterior=False):
    target = self.posterior_invalid if posterior else self.first_invalid
    target.copy_(
      torch.where(bad & (target < 0) & (frame < self.lengths), frame, target)
    )


class ChunkMetrics:
  """Device frame chunks transferred together; per-scenario counts exclude bad rows."""

  def __init__(self, capacity, width, chunk, replay_rows, device, names=TRACKING):
    self.names = names
    self.width, self.chunk, self.replay_rows = width, chunk, replay_rows
    self.values = torch.empty(
      chunk, capacity, len(names), dtype=torch.float64, device=device
    )
    self.valid = torch.empty(chunk, capacity, dtype=torch.bool, device=device)
    self.poses = torch.empty(chunk, 3, len(replay_rows), 36, device=device)
    self.used = 0
    self.host_values, self.host_valid, self.host_poses = [], [], []

  def update(self, values, valid, poses=None):
    self.values[self.used].copy_(values)
    self.valid[self.used].copy_(valid)
    if poses is not None:
      self.poses[self.used].copy_(poses)
    self.used += 1
    if self.used == self.chunk:
      self.flush()

  def flush(self):
    if not self.used:
      return
    self.host_values.append(self.values[: self.used].cpu().numpy().copy())
    self.host_valid.append(self.valid[: self.used].cpu().numpy().copy())
    self.host_poses.append(self.poses[: self.used].cpu().numpy().copy())
    self.used = 0

  def result(self, rows, length):
    self.flush()
    values = np.concatenate([v[:, rows] for v in self.host_values])[:length]
    valid = np.concatenate([v[:, rows] for v in self.host_valid])[:length]
    values = np.where(valid[..., None], values, 0)

    def mean(v, ok):
      out = v.sum(0) / np.where(ok.any(0), ok.sum(0), np.nan)[:, None]
      for i, name in enumerate(self.names):
        if "rmse" in name:
          out[:, i] = np.sqrt(np.maximum(out[:, i], 0))
      return out

    bins = [
      mean(values[i : i + self.width], valid[i : i + self.width])
      for i in range(0, length, self.width)
    ]
    return dict(
      full=mean(values, valid),
      bins=np.stack(bins),
      bin_counts=np.array(
        [min(self.width, length - i) for i in range(0, length, self.width)]
      ),
      frames=np.array(length),
      valid_frame_counts=valid.sum(0),
    )


class CapturedFilter:
  """Replay the unchanged square-root/Jacobian kernels with stable CUDA buffers.

  Capture happens during explicit compilation checks/warmup, before timing.
  Inputs are copied each call; returned buffers remain valid until the next call.
  """

  def __init__(self, estimator):
    self.estimator = estimator
    self.graph = None

  def __call__(self, mean, lower, action, context, measurement, *, measurement_std):
    incoming = [getattr(mean, f.name) for f in fields(mean)] + [
      lower,
      action,
      context,
      measurement,
    ]
    if self.graph is None:
      self.buffers = [value.clone() for value in incoming]
      # Graph inputs allocated outside capture must remain owned for its lifetime.
      self.measurement_std = measurement_std.clone()
      static_mean = LieState(*self.buffers[:6])

      def run():
        return self.estimator.predict_correct_kernel(
          static_mean, *self.buffers[6:], measurement_std=self.measurement_std
        )

      stream = torch.cuda.Stream(device=lower.device)
      stream.wait_stream(torch.cuda.current_stream(lower.device))
      with torch.cuda.stream(stream):
        for _ in range(3):
          run()
      torch.cuda.current_stream(lower.device).wait_stream(stream)
      self.graph = torch.cuda.CUDAGraph()
      with torch.cuda.graph(self.graph, stream=stream):
        self.result = run()
    for target, value in zip(self.buffers, incoming, strict=True):
      target.copy_(value)
    self.graph.replay()
    return self.result


class TrackingRuntime:
  """One environment and one model/kernel bundle per group, reused across tasks."""

  def __init__(self, cfg, motions, table, saved):
    self.cfg, self.motions, self.table, self.saved = cfg, motions, table, saved
    if cfg.compile_kernels and (
      not str(cfg.device).startswith("cuda") or not torch.cuda.is_available()
    ):
      raise RuntimeError(
        "Compiled tracking requires an available CUDA GPU; no eager/CPU fallback"
      )
    env_cfg, failure_terms = make_env_config(cfg, motions[0]["path"])
    env_cfg.commands["motion"].motion_file = [m["path"] for m in motions]
    env_cfg.scene.env_spacing = 0.0  # Independent worlds, identical world coordinates.
    env_cfg.episode_length_s = (max(m["frames"] for m in motions) + 1) * 0.02
    self.env = create_env(env_cfg, cfg.num_envs, cfg.device)
    self.raw = raw = self.env.unwrapped
    if set(raw.event_manager.available_modes) & {"step", "interval"}:
      self.close()
      raise ValueError("Full-clip fast path requires no step/interval events")
    self.command = raw.command_manager.get_term("motion")
    self.monitor = TerminationManager(failure_terms, raw)
    self.offsets = dict(
      zip(
        [m["sha256"] for m in motions],
        np.cumsum([0] + [m["frames"] for m in motions[:-1]]).tolist(),
        strict=True,
      )
    )
    self.bundles = {}

  def close(self):
    self.env.close()

  def bundle(self, variant):
    if variant in self.bundles:
      return self.bundles[variant]
    cfg = self.cfg
    local = variant.startswith(("mean-", "belief-"))
    estimator_key, actor_key = (variant, variant) if local else ("C", variant[0])
    wm, actor, ecfg, acfg = load_split_models(
      self.env, self.saved[estimator_key], self.saved[actor_key], cfg.device
    )
    filter_cfg = replace(
      wm.filter_config,
      validate_numerics=False,
      jacobian_batch_chunk_size=cfg.jacobian_batch_chunk_size,
    )
    estimator = LieBeliefFilter(
      wm.dynamics,
      ecfg.estimator_inputs.measurement_noise,
      wm.representation._default_joint_pos,
      config=filter_cfg,
      compile_kernels=cfg.compile_kernels,
    )
    actor_inputs = (
      replace(acfg.estimator_inputs, mode="train")
      if variant == "A-truth"
      else acfg.estimator_inputs
    )
    constants = covariance_encoding_constants(
      torch.empty(0, device=cfg.device),
      actor_inputs.covariance_scales,
      actor_inputs.covariance_std_floor,
    )

    def features(truth, belief, reference, context):
      return build_actor_critic_goals_kernel(
        actor_inputs,
        RpvState(truth.rotation, truth.position, truth.world_velocity),
        belief.to_actor_input(),
        reference,
        constants,
        context,
      )

    def actor_forward(goal, uncertainty, history):
      return actor.forward_with_aux(
        build_actor_obs_td({"motion": goal}, history, task_uncertainty=uncertainty)
      )[0]

    compile_fn = compile_eit_kernel if cfg.compile_kernels else lambda f: f
    from supermimic.validation.metrics import tracking_values_kernel

    def odometry(belief, truth):
      scores = posterior_values_kernel(belief, truth)
      return (
        torch.cat((scores, local_posterior_values(belief, truth)), -1)
        if local
        else scores
      )

    bundle = dict(
      wm=wm,
      estimator_key=estimator_key,
      actor=actor,
      ecfg=ecfg,
      acfg=acfg,
      estimator=estimator,
      predict=CapturedFilter(estimator)
      if cfg.compile_kernels
      else estimator.predict_correct_kernel,
      inputs=actor_inputs,
      features=compile_fn(features),
      metrics=compile_fn(tracking_values_kernel),
      odometry=compile_fn(odometry),
      actor_forward=compile_fn(actor_forward),
      features_eager=features,
      actor_eager=actor_forward,
    )
    self.bundles[variant] = bundle
    return bundle

  @torch.no_grad()
  def prepare(self, tasks, variant, limit=0):
    cfg, raw, command = self.cfg, self.raw, self.command
    bundle = self.bundle(variant)
    scenarios, lengths, starts = [], [], []
    for _, motion, _, ids in tasks:
      scenarios += [(motion["sha256"], i) for i in ids]
      lengths += [min(motion["frames"], limit) if limit else motion["frames"]] * len(
        ids
      )
      starts += [self.offsets[motion["sha256"]]] * len(ids)
    valid_count = len(scenarios)
    scenarios += [scenarios[0]] * (cfg.num_envs - valid_count)
    starts += [starts[0]] * (cfg.num_envs - valid_count)
    lengths += [0] * (cfg.num_envs - valid_count)
    command.cfg.fixed_start_frames = starts
    contexts = apply_domains(
      raw, bundle["wm"].sim_context, [self.table["values"][i] for _, i in scenarios]
    )
    for row, (sha, context) in enumerate(scenarios):
      torch.manual_seed(scenario_seed(cfg.evaluation_seed, sha, context, "reset"))
      raw.reset(env_ids=torch.tensor([row], device=cfg.device))
    obs = raw.get_observations()
    torch.testing.assert_close(
      command.time_steps, torch.tensor(starts, device=cfg.device)
    )
    truth = LieState.from_full(obs["full_state"])
    # Prior stream is keyed by clip/context, not by tensor slot or group.
    priors = [
      paired_prior(
        cfg,
        sha,
        [context],
        truth.index(slice(row, row + 1)),
        bundle["wm"].initial_prior,
      )
      for row, (sha, context) in enumerate(scenarios)
    ]
    belief = BeliefState(
      LieState.from_full(torch.cat([p.mean.to_full() for p in priors])),
      covariance_sqrt=torch.cat([p.covariance_sqrt for p in priors]),
    )
    belief.validate()
    reference = bundle["wm"].representation.reference_window(
      command.motion, command.time_steps, command.reference_alignment
    )
    build_actor_features(
      bundle["inputs"],
      RpvState(truth.rotation, truth.position, truth.world_velocity),
      belief.to_actor_input(),
      reference,
      torch.as_tensor(contexts, device=cfg.device),
    )
    dt = raw.step_dt
    if not np.isclose(
      dt,
      self.saved[bundle["estimator_key"]]["world_model"]["dynamics"]["metadata"]["dt"],
    ):
      raise ValueError("Simulation and estimator control periods differ")
    width = round(cfg.time_bin_s / dt)
    if width < 1 or not np.isclose(width * dt, cfg.time_bin_s):
      raise ValueError("time_bin_s must be a multiple of control_dt")
    noise = noise_block(
      cfg,
      scenarios,
      max(lengths),
      bundle["ecfg"].estimator_inputs.measurement_noise,
      cfg.device,
    )
    return dict(
      bundle=bundle,
      obs=obs,
      belief=belief,
      contexts=torch.as_tensor(contexts, device=cfg.device),
      lengths=torch.tensor(lengths, device=cfg.device),
      lengths_cpu=lengths,
      scenarios=scenarios,
      noise=noise,
      starts=torch.tensor(starts, device=cfg.device),
      width=width,
      parked_qpos=raw.sim.data.qpos.clone(),
      valid_count=valid_count,
    )

  @torch.no_grad()
  def check_compile(self, prepared):
    """Fixed inputs: full filter mean/factor, features and actions, no closed-loop comparison."""
    b = prepared["bundle"]
    belief, obs, contexts = prepared["belief"], prepared["obs"], prepared["contexts"]
    truth = LieState.from_full(obs["full_state"])
    reference = b["wm"].representation.reference_window(
      self.command.motion, self.command.time_steps, self.command.reference_alignment
    )
    sensor = torch.cat(
      (obs["actor"][:, :64] + prepared["noise"][0], obs["actor"][:, 64:]), -1
    )
    history = sensor[:, None].repeat(1, b["acfg"].history_length, 1)
    estimator = b["estimator"]
    std = estimator.measurement_config.standard_deviations(
      29, device=contexts.device, dtype=contexts.dtype
    )
    results = []
    for stance in ("force_eager", "default"):
      with torch.compiler.set_stance(stance):
        posterior = estimator._correction_kernel(belief, sensor[:, :64], std)[0]
        predict = (
          estimator.predict_correct_kernel if stance == "force_eager" else b["predict"]
        )
        mean, lower, invalid = predict(
          posterior.mean,
          posterior.covariance_sqrt,
          torch.zeros_like(sensor[:, :29]),
          contexts,
          sensor[:, :64],
          measurement_std=std,
        )
        candidate = BeliefState(mean, covariance_sqrt=lower)
        if stance == "force_eager":
          scoring_input = candidate.clone()
        goal, uncertainty, _, feature_invalid = b["features"](
          truth, candidate, reference, contexts
        )
        actions = b["actor_forward"](goal, uncertainty, history)
        ee = [
          self.command.cfg.body_names.index(name) for name in self.cfg.ee_body_names
        ]
        scores = b["metrics"](
          truth,
          reference,
          self.command.robot_body_pos_w,
          self.command.body_pos_w,
          self.command.robot_body_quat_w[:, ee],
          self.command.body_quat_w[:, ee],
          self.command.joint_pos,
          self.command.joint_vel,
          ee,
        )
        if bool((invalid | feature_invalid).any()):
          raise ValueError("Invalid fixed-input compilation check")
        results.append(
          (
            posterior.mean.to_full(),
            posterior.covariance_sqrt,
            mean.to_full(),
            lower,
            goal,
            uncertainty,
            actions,
            scores,
            # Use the identical posterior in both stances. NLL is sensitive to
            # small upstream changes to the square-root factor.
            b["odometry"](scoring_input, truth),
          )
        )
    torch.testing.assert_close(results[1], results[0], atol=2e-5, rtol=1e-3)
    return dict(passed=True, atol=2e-5, rtol=1e-3, device=str(contexts.device))

  @torch.no_grad()
  def execute(self, tasks, variant, prepared):
    cfg, raw, command = self.cfg, self.raw, self.command
    b = prepared["bundle"]
    belief, obs, contexts = (
      prepared["belief"].clone(),
      prepared["obs"],
      prepared["contexts"],
    )
    fallback = belief.clone()
    state = DeviceStatus(prepared["lengths"])
    rows = [
      i
      for i, (_, context) in enumerate(prepared["scenarios"][: prepared["valid_count"]])
      if context in cfg.replay_ids()
    ]
    row_tensor = torch.tensor(rows, device=cfg.device, dtype=torch.long)
    metrics = ChunkMetrics(
      cfg.num_envs, prepared["width"], cfg.transfer_chunk_steps, rows, cfg.device
    )
    local = b["inputs"].local_control
    odometry_names = ODOMETRY + LOCAL_METRICS if local else ODOMETRY
    lag = round(1 / raw.step_dt)
    poses_window = deque(maxlen=lag)
    relative_sums = contexts.new_zeros(cfg.num_envs, 2, dtype=torch.float64)
    relative_count = torch.zeros_like(prepared["lengths"])
    odometry = ChunkMetrics(
      cfg.num_envs,
      prepared["width"],
      cfg.transfer_chunk_steps,
      [],
      cfg.device,
      odometry_names,
    )
    ee = [command.cfg.body_names.index(name) for name in cfg.ee_body_names]
    lower, upper = normalized_action_limits(
      self.env, b["wm"].num_actions, device=cfg.device
    )
    estimator = b["estimator"]
    std = estimator.measurement_config.standard_deviations(
      29, device=contexts.device, dtype=contexts.dtype
    )
    first_failure = torch.full_like(prepared["lengths"], -1)
    first_terms = torch.zeros(
      cfg.num_envs, len(self.monitor.active_terms), dtype=torch.bool, device=cfg.device
    )
    initial_truth = obs["full_state"].cpu().numpy().copy()
    initial_prior = belief.mean.to_full().cpu().numpy()
    frames = max(prepared["lengths_cpu"])
    history, previous = None, None
    if str(cfg.device).startswith("cuda"):
      torch.cuda.synchronize(cfg.device)
    started = last_progress = monotonic()
    effective = torch.zeros((), dtype=torch.int64, device=cfg.device)
    for frame in range(frames):
      active = state.active(frame)
      truth_bad = ~torch.isfinite(obs["full_state"]).all(-1) | ~torch.isfinite(
        obs["actor"]
      ).all(-1)
      state.reject(truth_bad, frame)
      active = state.active(frame)
      truth = select_state(active, LieState.from_full(obs["full_state"]), fallback.mean)
      sensor = torch.cat(
        (obs["actor"][:, :64] + prepared["noise"][frame], obs["actor"][:, 64:]), -1
      )
      sensor = torch.where(active[:, None], sensor, 0)
      history = (
        sensor[:, None].repeat(1, b["acfg"].history_length, 1)
        if history is None
        else torch.cat((history[:, 1:], sensor[:, None]), 1)
      )
      filter_active = active & (state.posterior_invalid < 0)
      if previous is None:
        candidate = estimator._correction_kernel(belief, sensor[:, :64], std)[0]
        bad = belief_invalid(candidate)
      else:
        mean, factor, bad = b["predict"](
          belief.mean,
          belief.covariance_sqrt,
          previous,
          contexts,
          sensor[:, :64],
          measurement_std=std,
        )
        candidate = BeliefState(mean, covariance_sqrt=factor)
      state.reject(bad & filter_active, frame, posterior=True)
      if variant != "A-truth":
        state.reject(bad & filter_active, frame)
      belief = isolate_belief(
        state.active(frame) & (state.posterior_invalid < 0), candidate, fallback
      )
      reference = b["wm"].representation.reference_window(
        command.motion, command.time_steps, command.reference_alignment
      )
      goal, uncertainty, _, feature_bad = b["features"](
        truth, belief, reference, contexts
      )
      actions = b["actor_forward"](goal, uncertainty, history).clamp(
        min=lower, max=upper
      )
      action_bad = feature_bad | ~torch.isfinite(actions).all(-1)
      # The last frame is scored but consumes no actor action.
      state.reject(action_bad & (frame + 1 < state.lengths), frame)
      values = b["metrics"](
        truth,
        reference,
        command.robot_body_pos_w,
        command.body_pos_w,
        command.robot_body_quat_w[:, ee],
        command.body_quat_w[:, ee],
        command.joint_pos,
        command.joint_vel,
        ee,
      )
      state.reject(~torch.isfinite(values).all(-1), frame)
      active = state.active(frame)
      effective += active.sum()
      scoring = active & (first_failure < 0)
      ref_pose = torch.cat(
        (
          reference.position[:, 0],
          quat_from_matrix(reference.rotation[:, 0]),
          command.joint_pos,
        ),
        -1,
      )
      posterior = torch.where(
        (state.posterior_invalid < 0)[:, None],
        belief.mean.to_full()[:, :36],
        float("nan"),
      )
      poses = torch.stack((obs["full_state"][:, :36], posterior, ref_pose))[
        :, row_tensor
      ]
      metrics.update(values, scoring, poses)
      odom_values = b["odometry"](belief, truth)
      if local:
        # Compare posterior odometry against the realized physical trajectory.
        pose_pair = (
          torch.stack((belief.mean.rotation, truth.rotation)).double(),
          torch.stack((belief.mean.position, truth.position)).double(),
        )
        if len(poses_window) == lag:
          errors = relative_pose_squared(poses_window[0], pose_pair)
          valid = scoring & (state.posterior_invalid < 0)
          relative_sums += torch.where(valid[:, None], errors, 0)
          relative_count += valid.long()
        poses_window.append(pose_pair)
      odometry.update(
        odom_values,
        scoring & (state.posterior_invalid < 0) & torch.isfinite(odom_values).all(-1),
      )
      if (frame + 1) % cfg.host_check_interval_steps == 0 or frame + 1 == frames:
        # One batched read, preserving per-scenario first failure information.
        status_host = (
          torch.stack((state.first_invalid, state.posterior_invalid)).cpu().numpy()
        )
        if cfg.debug_checks:
          belief.validate()
          expected = torch.minimum(
            prepared["starts"] + frame, command.motion.last_frame(prepared["starts"])
          )
          torch.testing.assert_close(command.time_steps, expected)
      if monotonic() - last_progress >= cfg.progress_interval_s:
        print(
          f"  {variant}: {frame + 1}/{frames} steps, capacity={cfg.num_envs}, elapsed={monotonic() - started:.1f}s",
          flush=True,
        )
        last_progress = monotonic()
      if frame + 1 < frames:
        active = state.active(frame) & (frame + 1 < state.lengths)
        actions = torch.where(active[:, None], actions, 0)
        obs = full_clip_step(
          raw, command, actions, active, prepared["parked_qpos"], self.monitor
        )
        first = self.monitor.terminated & active & (first_failure < 0)
        first_failure.copy_(torch.where(first, frame + 1, first_failure))
        for i, name in enumerate(self.monitor.active_terms):
          first_terms[:, i] |= first & self.monitor.get_term(name)
        previous = actions
    metrics.flush()
    odometry.flush()
    if str(cfg.device).startswith("cuda"):
      torch.cuda.synchronize(cfg.device)
    elapsed = monotonic() - started
    failure = first_failure.cpu().numpy()
    terms = first_terms.cpu().numpy()
    replay = np.concatenate(metrics.host_poses)
    outputs, offset = [], 0
    for key, motion, _, ids in tasks:
      count = len(ids)
      selection = slice(offset, offset + count)
      length = prepared["lengths_cpu"][offset]
      result = metrics.result(selection, length)
      odom = odometry.result(selection, length)
      for field in ("full", "bins", "valid_frame_counts"):
        result["odometry_" + field] = odom[field]
      if local:
        counts = relative_count[selection]
        result["relative_pose_1s"] = (
          (relative_sums[selection] / counts[:, None].clamp_min(1))
          .sqrt()
          .masked_fill(counts[:, None] == 0, float("nan"))
          .cpu()
          .numpy()
        )
        result["relative_pose_1s_count"] = counts.cpu().numpy()
      selected_replay = [
        j for j, row in enumerate(rows) if offset <= row < offset + count
      ]
      result.update(
        context_ids=np.array(ids),
        context_normalized=contexts[selection].cpu().numpy(),
        initial_truth=initial_truth[selection],
        initial_prior=initial_prior[selection],
        first_failure_s=np.where(
          failure[selection] < 0, np.nan, failure[selection] * raw.step_dt
        ),
        first_invalid_frame=status_host[0, selection],
        posterior_invalid_frame=status_host[1, selection],
        replay_context_ids=np.array(
          [prepared["scenarios"][rows[j]][1] for j in selected_replay], dtype=np.int64
        ),
      )
      if selected_replay:
        for i, name in enumerate(("truth", "posterior", "reference")):
          result["replay_" + name] = replay[:length, i][:, selected_replay]
      expected_frames = np.where(failure[selection] < 0, length, failure[selection])
      failed = (result["valid_frame_counts"] != expected_frames).any()
      outputs.append(
        (
          key,
          result,
          dict(
            status="invalid"
            if failed
            else ("complete" if length == motion["frames"] else "censored"),
            control_dt=raw.step_dt,
            clip_frames=motion["frames"],
            scored_frames=length,
            metric_names=list(TRACKING),
            odometry_metric_names=list(odometry_names),
            odometry_status=(
              "invalid"
              if (odom["valid_frame_counts"] != expected_frames).any()
              else ("complete" if length == motion["frames"] else "censored")
            ),
            failure_terms=[
              [name for i, name in enumerate(self.monitor.active_terms) if row[i]]
              for row in terms[selection]
            ],
            posterior_status="ok"
            if (status_host[1, selection] < 0).all()
            else "partial",
            elapsed_s=elapsed,
          ),
        )
      )
      offset += count
    return outputs, dict(
      elapsed_s=elapsed,
      effective_frames=int(effective.cpu()),
    )
