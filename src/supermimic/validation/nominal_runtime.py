"""Five residual controllers sharing one batch of independent MuJoCo worlds."""

# PyTorch 2.14 dynamically re-exports public tensor factories and operations.
# pyright: reportPrivateImportUsage=false

from __future__ import annotations

from dataclasses import asdict, replace
from itertools import groupby
from math import ceil
from time import perf_counter
from typing import cast

import numpy as np
import torch
from mjlab.managers.termination_manager import TerminationManager
from mjlab.utils.lab_api.math import matrix_from_quat

from supermimic.belief import BeliefState
from supermimic.estimator import LieBeliefFilter, belief_invalid
from supermimic.estimator_inputs import local_actor_features
from supermimic.local_control import (
  VECH_DIM,
  local_covariance_features,
  local_state,
  motion_reference,
)
from supermimic.motion_command import MotionCommand, MotionCommandCfg
from supermimic.motion_loader import MotionLoader
from supermimic.representations import GenCoordRepresentation
from supermimic.runners.eit_inference import evaluation_config, load_evaluation_models
from supermimic.scripts.train_utils import create_env
from supermimic.tasks.task_spec import build_actor_obs_td
from supermimic.utils.helpers import normalized_action_limits
from supermimic.utils.lie_state import LieState
from supermimic.validation.batched import CapturedFilter, isolate_belief, noise_block
from supermimic.validation.config import StudyConfig
from supermimic.validation.domains import apply_domains
from supermimic.validation.nominal import ROUTES
from supermimic.validation.nominal_metrics import frame_scores, reward_terms
from supermimic.validation.nominal_posterior import (
  MODE,
  PosteriorBuffer,
  posterior_sample,
)
from supermimic.validation.runtime import full_clip_step, make_env_config, paired_prior


def evaluation_noise(prior, measurement, kappa_b, kappa_o):
  """Scale standard deviations for the paired prior, data generation and correction."""
  return (
    replace(prior, std=tuple(kappa_b * s for s in prior.std)),
    replace(
      measurement, **{name: kappa_o * s for name, s in asdict(measurement).items()}
    ),
  )


def contiguous_groups(rows, key):
  offset = 0
  for name, group in groupby(rows, key):
    stop = offset + sum(1 for _ in group)
    if name is not None:
      yield name, slice(offset, stop)
    offset = stop


class NominalRuntime:
  @torch.no_grad()
  def __init__(
    self, paths, manifest, saved, device, capacity, largest_cohort, compile_filter=True
  ):
    self.cfg = cfg = StudyConfig(device=device, num_envs=capacity, evaluation_seed=0)
    self.device = device
    self.motions = {m["name"]: m for m in manifest["motions"]}
    env_cfg, failures = make_env_config(cfg, paths[0])
    # Reserve the standalone G1 playback contact budget for the largest
    # synchronized clip cohort while retaining the task's pooled capacity.
    env_cfg.sim.nconmax = max(
      env_cfg.sim.nconmax or 0, ceil(256 * largest_cohort / capacity)
    )
    motion = cast(MotionCommandCfg, env_cfg.commands["motion"])
    motion.motion_file = [str(p) for p in paths]
    motion.pose_range = {}
    motion.velocity_range = {}
    motion.joint_position_range = (0.0, 0.0)
    motion.exact_reference_joint_reset = True
    env_cfg.scene.env_spacing = 0.0
    self.env = create_env(env_cfg, capacity, device)
    self.raw = self.env.unwrapped
    self.cmd = cast(MotionCommand, self.raw.command_manager.get_term("motion"))
    self.motion = cast(MotionLoader, self.cmd.motion)
    self.monitor = TerminationManager(failures, self.raw)
    self.terms = reward_terms(self.raw)
    if len(self.cmd.cfg.body_names) != 14 or self.cmd.cfg.body_names[0] != "pelvis":
      raise ValueError("Nominal MPJPE requires the G1 pelvis-first 14-body skeleton")
    self.offsets = dict(
      zip((p.name for p in paths), self.motion.motion_starts.tolist(), strict=True)
    )
    self.actors, self.filters, self.wms, self.predictors = {}, {}, {}, {}
    self.inputs, self.checked_modes = {}, set()
    self.priors = {}
    kappa_b = manifest.get("kappa_b", 1)
    kappa_o = manifest.get("kappa_o", 1)
    self.use_graph = device.startswith("cuda")
    for key, checkpoint in saved.items():
      cfg_model = evaluation_config(checkpoint)
      assert cfg_model.estimator_inputs is not None
      wm, actor = load_evaluation_models(self.env, cfg_model, checkpoint, device)
      self.actors[key], self.wms[key] = actor, wm
      self.inputs[key] = cfg_model.estimator_inputs
      if key in ("mean", "belief"):
        self.priors[key], measurement = evaluation_noise(
          wm.initial_prior,
          cfg_model.estimator_inputs.measurement_noise,
          kappa_b,
          kappa_o,
        )
        self.filters[key] = LieBeliefFilter(
          wm.dynamics,
          measurement,
          cast(GenCoordRepresentation, wm.representation)._default_joint_pos,
          config=replace(
            wm.filter_config, validate_numerics=False, jacobian_batch_chunk_size=32
          ),
          compile_kernels=self.use_graph and compile_filter,
        )
    self.measurement = self.filters["mean"].measurement_config
    self.history_length = saved["mean"]["runner_config"]["history_length"]
    self.limits = normalized_action_limits(self.env, 29, device=device)
    self.context = torch.as_tensor(
      apply_domains(
        self.raw, self.wms["mean"].sim_context, [[0.75] + [0.0] * 32] * capacity
      ),
      device=device,
    )
    self.warmup_s = 0.0

  def close(self):
    self.env.close()

  def check_first_actions(self, modes, filters, beliefs, truth, history, actions):
    """Compare each batched route with the checkpoint's standard actor interface."""
    if not torch.is_inference_mode_enabled() or actions.grad_fn is not None:
      raise RuntimeError("Nominal actor inference must run without a gradient graph")
    slices = dict(filters)
    for mode in modes.keys() - self.checked_modes:
      key, estimator = ROUTES[mode]
      i = modes[mode].start
      row = slice(i, i + 1)
      inputs = self.inputs[key]
      posterior = None
      if estimator:
        offset = i - slices[estimator].start
        posterior = beliefs[estimator].index(slice(offset, offset + 1)).to_actor_input()
        inputs = self.inputs[estimator]
      reference = (
        self.wms[key]
        .representation.reference_window(
          self.cmd.motion, self.cmd.time_steps, self.cmd.reference_alignment
        )
        .index(row)
      )
      goal, condition = local_actor_features(
        inputs, truth.index(row), posterior, reference, self.context[row]
      )
      expected = self.actors[key].forward_with_aux(
        build_actor_obs_td({"motion": goal}, history[row], task_uncertainty=condition)
      )[0]
      torch.testing.assert_close(actions[row], expected, atol=2e-5, rtol=1e-4)
      self.checked_modes.add(mode)

  def reset(self, rows):
    cmd = self.cmd
    starts = [self.offsets[s.clip] for s in rows]
    cmd.cfg.fixed_start_frames = starts + [starts[0]] * (self.cfg.num_envs - len(rows))
    raw_obs, _ = self.raw.reset()
    obs = cast(dict[str, torch.Tensor], raw_obs)
    truth = LieState.from_full(obs["full_state"])
    m, t = self.motion, cmd.time_steps
    torch.testing.assert_close(t, t.new_tensor(cmd.cfg.fixed_start_frames))
    for actual, expected in (
      (truth.position, m.root_pos_w[t]),
      (truth.rotation, matrix_from_quat(m.root_quat_w[t])),
      (truth.joint_pos, m.joint_pos[t]),
      (truth.joint_vel, m.joint_vel[t]),
      (truth.world_velocity, m.root_lin_vel_w[t]),
      ((truth.rotation @ truth.omega.unsqueeze(-1)).squeeze(-1), m.root_ang_vel_w[t]),
    ):
      torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-4)
    return obs, truth

  def update_filter(self, key, belief, action, context, measurement, std):
    estimator = self.filters[key]
    if action is None:
      result = estimator._correction_kernel(belief, measurement, std)[0]
      return result, belief_invalid(result)
    # Only eager evaluation and the first CUDA capture need ordinary tensors
    # for jacrev. Steady-state replay runs entirely in inference mode.
    cache_key = (key, len(action))
    replay = self.use_graph and cache_key in self.predictors
    with torch.inference_mode(replay), torch.no_grad():
      started = None
      if self.use_graph:
        if not replay:
          started = perf_counter()
          self.predictors[cache_key] = CapturedFilter(estimator)
        predict = self.predictors[cache_key]
      else:
        predict = estimator.predict_correct_kernel
        belief = belief.clone()
        action, context, measurement, std = (
          x.clone() for x in (action, context, measurement, std)
        )
      mean, factor, bad = predict(
        belief.mean,
        belief.covariance_sqrt,
        action,
        context,
        measurement,
        measurement_std=std,
      )
      if started is not None:
        torch.cuda.synchronize(self.device)
        self.warmup_s += perf_counter() - started
      return BeliefState(mean, covariance_sqrt=factor), bad

  @torch.inference_mode()
  def run(self, batch, persist, persist_posterior=None, audit=None):
    order = list(ROUTES)
    rows = sorted(batch, key=lambda s: order.index(s.mode))
    n, cap = len(rows), self.cfg.num_envs
    actors = list(contiguous_groups(rows, lambda s: ROUTES[s.mode][0]))
    filters = list(contiguous_groups(rows, lambda s: ROUTES[s.mode][1]))
    modes = dict(contiguous_groups(rows, lambda s: s.mode))
    obs, truth = self.reset(rows)
    parked = self.raw.sim.data.qpos.clone()
    plans = [self.motions[s.clip]["steps"] for s in rows]
    lengths = torch.tensor(plans + [0] * (cap - n), device=self.device)
    active = lengths > 0
    invalid = torch.zeros_like(active)
    failed = torch.zeros_like(active)
    term_hits = torch.zeros(
      cap, len(self.monitor.active_terms), device=self.device, dtype=torch.bool
    )
    counts = torch.zeros_like(lengths)
    scenarios = [(self.motions[s.clip]["sha256"], s.seed) for s in rows]
    unique = list(dict.fromkeys(scenarios))
    noise_ids = torch.tensor([unique.index(s) for s in scenarios], device=self.device)
    belief_slice = modes.get(MODE)
    recorder = (
      PosteriorBuffer(plans[belief_slice])
      if persist_posterior is not None and belief_slice is not None
      else None
    )
    posterior_episodes = []
    audit_actions = []
    noise = noise_block(
      self.cfg,
      unique,
      max(plans) + int(recorder is not None),
      self.measurement,
      self.device,
    )
    std = self.measurement.standard_deviations(
      29, device=self.device, dtype=truth.position.dtype
    )
    beliefs, priors = {}, {}
    cpu_truth = truth.to("cpu")
    for key, sl in filters:
      starts = [
        paired_prior(
          self.cfg,
          sha,
          [seed],
          cpu_truth.index(slice(i, i + 1)),
          self.priors[key],
        )
        for i, (sha, seed) in enumerate(scenarios[sl], start=sl.start)
      ]
      priors[key] = BeliefState(
        LieState.from_full(torch.cat([p.mean.to_full() for p in starts])),
        covariance_sqrt=torch.cat([p.covariance_sqrt for p in starts]),
      ).to(self.device)
      beliefs[key] = priors[key]
    history = previous = None
    # Transfer short frame chunks. Completed trajectories become resumable
    # even when another slot in the same batch survives the entire clip.
    recorded = []
    pieces = [[] for _ in rows]
    completed = np.zeros(n, dtype=bool)
    errors = []
    write_s = 0.0
    live = np.ones(n, dtype=bool)
    max_steps = max(plans)
    for frame in range(max_steps):
      current = LieState.from_full(obs["full_state"][:n])
      sensor = obs["actor"][:n].clone()
      sensor[:, :64] += noise[frame, noise_ids]
      bad = ~(
        torch.isfinite(obs["full_state"][:n]).all(-1) & torch.isfinite(sensor).all(-1)
      )
      sensor = torch.nan_to_num(sensor)
      history = (
        sensor[:, None].repeat(1, self.history_length, 1)
        if history is None
        else torch.cat((history[:, 1:], sensor[:, None]), 1)
      )
      for key, sl in filters:
        capture = recorder is not None and key == "belief"
        if not live[sl].any() and not capture:
          continue
        result, filter_bad = self.update_filter(
          key,
          beliefs[key],
          None if previous is None else previous[sl],
          self.context[sl],
          sensor[sl, :64],
          std,
        )
        bad[sl] |= filter_bad
        if recorder is not None and key == "belief":
          # Inactive rows on their first tick after termination still carry the
          # outgoing posterior/action and the untouched post-physics observation.
          # Capture the raw correction before dead rows are replaced by priors.
          sample = posterior_sample(current.index(sl), result)
          recorder.append(sample.masked_fill(bad[sl, None], float("nan")))
        beliefs[key] = isolate_belief(active[sl] & ~bad[sl], result, priors[key])
      times = self.cmd.time_steps[:n]
      ref = motion_reference(self.motion, times, self.motion.last_frame(times))
      y = torch.zeros_like(ref.y)
      covariance = y.new_zeros(n, VECH_DIM)
      if "GT-Oracle" in modes:
        sl = modes["GT-Oracle"]
        y[sl] = local_state(current.index(sl))
      for key, sl in filters:
        if not live[sl].any():
          continue
        y[sl] = local_state(beliefs[key].mean)
        if key == "belief":
          covariance[sl] = local_covariance_features(
            beliefs[key].mean, beliefs[key].covariance_sqrt
          )
      error = y - ref.y
      if "Prop.+KP" in modes:
        error[modes["Prop.+KP"]] = 0
      condition = torch.cat(
        (y, error, self.context[:n], covariance, ref.next_joint), -1
      )
      actions = sensor.new_zeros(cap, 29)
      for key, sl in actors:
        if not live[sl].any():
          continue
        actor = self.actors[key]
        actions[sl] = actor.decode_goal(
          actor.encode_goal(ref.goal[sl]), history[sl], condition[sl]
        )[0]
      if frame == 0:
        self.check_first_actions(modes, filters, beliefs, current, history, actions)
      bad |= ~torch.isfinite(actions[:n]).all(-1)
      invalid[:n] |= active[:n] & bad
      executing = active & ~invalid
      actions = torch.where(executing[:, None], actions.clamp(*self.limits), 0)
      if audit is not None:
        audit_actions.append(actions[:n].cpu().numpy())
      obs = cast(
        dict[str, torch.Tensor],
        full_clip_step(self.raw, self.cmd, actions, executing, parked, self.monitor),
      )
      scores = frame_scores(self.raw, self.cmd, self.terms)
      invalid |= executing & ~(
        torch.isfinite(scores).all(-1) & torch.isfinite(obs["full_state"]).all(-1)
      )
      valid = executing & ~invalid
      counts += valid
      end = valid & self.monitor.terminated
      for i, name in enumerate(self.monitor.active_terms):
        term_hits[:, i] |= end & self.monitor.get_term(name)
      failed |= end
      recorded.append(scores[:n].float())
      active = valid & ~end & (frame + 1 < lengths)
      previous = actions
      if (frame + 1) % 16 == 0 or frame + 1 == max_steps:
        live = active[:n].cpu().numpy()
      if (frame + 1) % 256 == 0 or not live.any() or frame + 1 == max_steps:
        block = torch.stack(recorded, dim=1).cpu().numpy()
        first = frame + 1 - len(recorded)
        recorded.clear()
        steps = counts[:n].tolist()
        ended = ~active[:n].cpu().numpy()
        bad_rows = invalid[:n].tolist()
        failed_rows = failed[:n].tolist()
        hits = term_hits[:n].tolist()
        for i in np.flatnonzero(~completed):
          used = min(block.shape[1], max(steps[i] - first, 0))
          if used:
            pieces[i].append(block[i, :used].copy())
          if ended[i]:
            values = np.concatenate(pieces[i]) if pieces[i] else block[i, :0]
            pieces[i].clear()
            status = (
              "invalid" if bad_rows[i] else ("failure" if failed_rows[i] else "success")
            )
            terms = [
              k
              for k, hit in zip(self.monitor.active_terms, hits[i], strict=True)
              if hit
            ]
            if bad_rows[i]:
              errors.append(rows[i])
            t_write = perf_counter()
            persist(rows[i], status, plans[i], values, terms)
            if recorder is not None and rows[i].mode == MODE:
              posterior_episodes.append((i, status, values))
            write_s += perf_counter() - t_write
            completed[i] = True
        if ended.all():
          break
      if (frame + 1) % 500 == 0:
        print(
          f"  step {frame + 1}: {int(active.sum())}/{n} trajectories active", flush=True
        )
    if recorder is not None:
      # There is no next control tick for the batch's final action.
      assert (
        belief_slice is not None
        and persist_posterior is not None
        and previous is not None
      )
      sl = belief_slice
      t = recorder.start + len(recorder.pending)
      sensor = obs["actor"][sl, :64] + noise[t, noise_ids[sl]]
      terminal, bad = self.update_filter(
        "belief", beliefs["belief"], previous[sl], self.context[sl], sensor, std
      )
      sample = posterior_sample(LieState.from_full(obs["full_state"][sl]), terminal)
      recorder.append(sample.masked_fill(bad[:, None], float("nan")))
      started = perf_counter()
      for i, status, values in posterior_episodes:
        fields = recorder.episode(i - sl.start, len(values))
        persist_posterior(rows[i], status, values, self.raw.step_dt, fields)
      write_s += perf_counter() - started
    if audit is not None:
      controls = np.stack(audit_actions, axis=1)
      for i, count in enumerate(counts[:n].tolist()):
        audit(rows[i], controls[i, :count], noise[:count, noise_ids[i]].cpu().numpy())
    return int(counts.sum()), errors, write_s
