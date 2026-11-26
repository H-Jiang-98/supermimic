"""Pathwise policy gradients through full belief and stochastic imagined states.

The world model and critic weights are fixed during the actor backward pass;
their input derivatives remain live. Replay prefixes are detached boundaries.
"""

from __future__ import annotations

import copy
import math
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass

import torch
from torch import nn

from supermimic.belief_provider import BeliefProvider, belief_rollout
from supermimic.imagined_rollout import (
  RolloutResult,
  imagine_step,
  observe,
  prepare_rollout,
)
from supermimic.learners.eit_policy_graph import CapturedPolicyUpdate
from supermimic.learners.learner_utils import (
  build_optimizers,
  fit_critic,
  validate_optimizer_state,
)
from supermimic.learners.policy_learner import discounted_bootstrap, gae_returns
from supermimic.utils.compile import compile_eit_kernel
from supermimic.utils.profiling import StreamTimer


@dataclass(frozen=True)
class BeliefPolicyLearnerCfg:
  batch_size: int = 4096
  num_samples: int = 1
  microbatch_size: int = 16
  horizon: int = 32
  num_batches: int = 1
  noise: float = 0.1
  lr: float = 2e-4
  critic_lr: float = 2e-4
  critic_epochs: int = 5
  critic_minibatches: int = 4
  critic_microbatch_size: int | None = None
  grad_clip_value: float = 0.1
  gamma: float = 0.99
  lam: float = 0.95
  max_grad_norm: float = 1.0
  value_clipping: bool = True
  batch_motion_encoder: bool = True

  def __post_init__(self):
    for key in (
      "batch_size",
      "num_samples",
      "microbatch_size",
      "horizon",
      "num_batches",
      "critic_epochs",
      "critic_minibatches",
    ):
      if type(getattr(self, key)) is not int or getattr(self, key) < 1:
        raise ValueError(f"Invalid policy {key}")
    if self.critic_microbatch_size is not None and (
      type(self.critic_microbatch_size) is not int or self.critic_microbatch_size < 1
    ):
      raise ValueError("Invalid critic_microbatch_size")
    if not math.isfinite(self.grad_clip_value) or self.grad_clip_value < 0:
      raise ValueError("Invalid grad_clip_value")
    for key in ("lr", "critic_lr", "max_grad_norm"):
      if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
        raise ValueError(f"Invalid policy {key}")
    if not math.isfinite(self.noise) or self.noise < 0:
      raise ValueError("Invalid policy collection noise")
    if not 0 <= self.gamma < 1 or not 0 <= self.lam <= 1:
      raise ValueError("Policy requires 0 <= gamma < 1 and 0 <= lambda <= 1")
    if type(self.value_clipping) is not bool:
      raise ValueError("value_clipping must be boolean")


def policy_metrics_for_json(metrics):
  result = {name: float(value) for name, value in metrics.items()}
  if math.isnan(result["Policy/explained_variance"]):
    result["Policy/explained_variance"] = None
  return result


@contextmanager
def _module_modes(modules):
  """Restore even child normalization modes and original gradient permissions."""
  modes = {m: m.training for root in modules for m in root.modules()}
  flags = {p: p.requires_grad for root in modules for p in root.parameters()}
  try:
    yield
  finally:
    for module, training in modes.items():
      module.training = training
    for parameter, enabled in flags.items():
      parameter.requires_grad_(enabled)


class BeliefActorCriticPolicyLearner:
  def __init__(
    self,
    actor,
    critic,
    wm_env,
    cfg=None,
    device=None,
    *,
    compile_kernels=False,
    cuda_graphs=False,
  ):
    self.actor: nn.Module = actor
    self._actor_forward = (
      compile_eit_kernel(actor.forward) if compile_kernels else actor
    )
    forward_with_aux = getattr(actor, "forward_with_aux", None)
    self._actor_with_aux = (
      compile_eit_kernel(forward_with_aux)
      if compile_kernels and forward_with_aux
      else forward_with_aux
    )
    self._encode = getattr(actor, "encode_goal", None)
    self._decode = getattr(actor, "decode_goal", None)
    if compile_kernels and self._encode is not None:
      self._encode = compile_eit_kernel(self._encode)
      self._decode = compile_eit_kernel(self._decode)
    self.critic: nn.Module = critic
    self.wm_env = wm_env
    self.wm_env.eit_critic = True
    self.wm_env.critic_input_config = getattr(critic, "config", None)
    self.cfg = cfg or BeliefPolicyLearnerCfg()
    device = str(next(actor.parameters()).device)
    self.actor_optimizers = build_optimizers(
      [actor], self.cfg.lr, betas=(0.9, 0.95), device=device
    )
    self.critic_optimizers = build_optimizers(
      [critic], self.cfg.critic_lr, device=device
    )
    self.actor_steps = self.critic_steps = 0
    parameter = next(actor.parameters())
    self.generators = {}
    for offset, name in enumerate(("sampling", "process", "measurement", "critic")):
      generator = torch.Generator(device=parameter.device)
      generator.manual_seed((torch.initial_seed() + offset) % (2**63 - 1))
      self.generators[name] = generator
    self._v_max = wm_env.max_step_reward / (1 - self.cfg.gamma)
    self.cuda_graphs = cuda_graphs
    self._policy_graph = None
    self._policy_graph_signature = None
    self._policy_graph_stream = None
    self._graph_reason = "disabled"

  def clear_policy_graph(self):
    """Discard process-local capture state after restore/failure/config changes."""
    self._policy_graph = self._policy_graph_signature = None

  def clear_actor_graph(self):
    """Compatibility name for callers that previously cleared actor-only capture."""
    self.clear_policy_graph()

  @property
  def graph_status(self):
    return {
      "requested": self.cuda_graphs,
      "scope": "actor_critic",
      "reason": self._graph_reason,
      **(self._policy_graph.stats if self._policy_graph is not None else {}),
    }

  def _policy_graph_state(self):
    env = self.wm_env
    state = []
    for root in (self.actor, self.critic, env.dynamics, env.representation):
      state.extend(root.parameters())
      state.extend(root.buffers())
    # Only read dependencies: collection replaces actor._aux_loss, which the
    # functional training forward intentionally does not consume.
    state.extend(
      (
        env.representation._default_joint_pos,
        env.representation._soft_joint_pos_limits,
        env._action_lower,
        env._action_upper,
      )
    )
    state.append(env.posterior_provider.estimator.default_joint_pos)
    for module in (self.actor, self.critic):
      state.extend(p.grad for p in module.parameters() if p.grad is not None)
    for optimizer in (*self.actor_optimizers, *self.critic_optimizers):
      state.extend(t for values in optimizer.state.values() for t in values.values())
      state.extend(
        group["lr"] for group in optimizer.param_groups if torch.is_tensor(group["lr"])
      )
    return state

  def _select_policy_graph(self, count):
    env, cfg = self.wm_env, self.cfg
    parameter = next(self.actor.parameters())
    if (
      self.cuda_graphs
      and parameter.is_cuda
      and self._uses_tensor_rollout
      and self._policy_graph_stream is None
    ):
      # AccumulateGrad may survive cache destruction. Reuse this stream through
      # checkpoint restores, graph rebuilds, and whole-update eager fallbacks.
      self._policy_graph_stream = torch.cuda.Stream(device=parameter.device)
    reason = "enabled"
    if not self.cuda_graphs:
      reason = "disabled"
    elif not next(self.actor.parameters()).is_cuda:
      reason = "cpu"
    elif not self._uses_tensor_rollout:
      reason = "custom_environment"
    elif env.posterior_provider.estimator.config.jacobian_backend not in (
      "increment",
      "vmap",
    ):
      reason = "jacobian_backend"
    elif count % cfg.microbatch_size:
      reason = "tail_microbatch"
    if reason != self._graph_reason:
      if self.cuda_graphs:
        print(f"[INFO] EIT policy CUDA graph: {reason}", flush=True)
      self._graph_reason = reason
    if reason != "enabled":
      # A retained capture pool must not inflate the fallback's peak allocation.
      self.clear_policy_graph()
      return None

    estimator, rep = env.posterior_provider.estimator, env.representation
    signature = (
      count,
      asdict(cfg),
      tuple(
        tuple(
          (key, value)
          for key, value in group.items()
          if key != "params" and not torch.is_tensor(value)
        )
        for optimizer in (*self.actor_optimizers, *self.critic_optimizers)
        for group in optimizer.param_groups
      ),
      asdict(env.estimator_input_config),
      asdict(env._eit_critic_config()),
      asdict(estimator.config),
      env.dynamics.process_noise_config,
      self._v_max,
      env._anchor_index,
      env._anchor_pos_threshold,
      env._anchor_ori_threshold,
      env._termination_softness,
      rep.dt,
      rep.substep_coeff,
      env.dynamics.dt,
      self.actor.latent_bc_weight,
      tuple(self.actor._slot_fractions),
      tuple(self.actor._task_names),
      self.actor._anchor,
      tuple(rep.fk._links),
      tuple(rep.fk._tracked),
      id(estimator),
      id(self._actor_with_aux),
      getattr(self._rollout_kernel, "__func__", self._rollout_kernel),
    )
    if self._policy_graph is not None and signature != self._policy_graph_signature:
      self.clear_policy_graph()
    if self._policy_graph is None:
      for module in (self.actor, self.critic):
        for parameter in module.parameters():
          parameter.grad = torch.zeros_like(parameter)
      self._policy_graph = CapturedPolicyUpdate(
        self._policy_update, self._policy_graph_state, self._policy_graph_stream
      )
      self._policy_graph_signature = signature
    else:
      self._policy_graph.check_state()
    return self._policy_graph

  def _actor_backward(
    self, prepared, process_noise, measurement_noise, count, *, starts=None
  ):
    """Capture body: accumulate into stable gradients; return detached outputs."""
    result = self._rollout_kernel(
      prepared, process_noise, measurement_noise, self.cfg.horizon, starts=starts
    )
    size = process_noise.shape[1]
    (-result.returns.sum() / count + result.auxiliary_loss * size / count).backward()
    return (
      tuple(value.detach() for value in result.critic_data),
      {name: value.detach() for name, value in result.metrics.items()},
      result.invalid.detach(),
    )

  @property
  def steps(self):
    return self.actor_steps

  def draw_noises(self, starts):
    parameter = next(self.actor.parameters())
    shape = (self.cfg.horizon, len(starts), self.cfg.num_samples)
    # Keep the fixed tensor interface without consuming the process RNG for K=1.
    process = (
      torch.zeros(*shape, 70, dtype=parameter.dtype, device=parameter.device)
      if self.cfg.num_samples == 1
      else torch.randn(
        *shape,
        70,
        dtype=parameter.dtype,
        device=parameter.device,
        generator=self.generators["process"],
      )
    )
    measurement = torch.randn(
      *shape,
      64,
      dtype=parameter.dtype,
      device=parameter.device,
      generator=self.generators["measurement"],
    )
    return process, measurement

  def sample_start_keys(self, episode_ids):
    """Draw the logical starts for every actor update, in optimizer-step order."""
    episodes, times = self.wm_env.sample_policy_keys(
      self.cfg.batch_size,
      episode_ids,
      generator=self.generators["sampling"],
      num_batches=self.cfg.num_batches,
    )
    return torch.stack((episodes, times), dim=1)

  def sample_starts(self, episode_ids=None):
    return self.wm_env.sample_policy_starts(
      self.cfg.batch_size,
      episode_ids,
      generator=self.generators["sampling"],
      num_batches=self.cfg.num_batches,
    )

  def update(self, episode_ids):
    with self.wm_env.readonly_policy_motion():
      return self.update_prepared(self.sample_starts(episode_ids))

  def update_prepared(self, starts):
    """Reuse detached initial conditions while each new actor generates its own rollout."""
    if len(starts) != self.cfg.num_batches * self.cfg.batch_size:
      raise ValueError("Expected one batch of prepared starts per actor update")
    logs = []
    profiles = []
    with self.wm_env.readonly_policy_motion():
      for offset in range(0, len(starts), self.cfg.batch_size):
        batch = starts.index(slice(offset, offset + self.cfg.batch_size))
        logs.append(self.update_from(batch, *self.draw_noises(batch)))
        profiles.append(self.last_profile)
    self.last_profile = {
      name: sum(profile[name] for profile in profiles) for name in profiles[0]
    }
    means = {}
    for name in logs[0]:
      values = torch.stack([log[name] for log in logs])
      means[name] = (
        values.nanmean() if name == "Policy/explained_variance" else values.mean()
      )
    return means

  def _validate_noises(self, starts, process_noise, measurement_noise):
    if len(starts) < 1:
      raise ValueError("Policy needs at least one prepared start")
    parameter = next(self.actor.parameters())
    for name, noise, width in (
      ("process", process_noise, 70),
      ("measurement", measurement_noise, 64),
    ):
      if (
        noise.shape != (self.cfg.horizon, len(starts), self.cfg.num_samples, width)
        or noise.dtype != parameter.dtype
        or noise.device != parameter.device
        or not torch.isfinite(noise).all()
      ):
        raise ValueError(f"Invalid policy {name} noise")

  @property
  def _uses_tensor_rollout(self):
    return isinstance(getattr(self.wm_env, "posterior_provider", None), BeliefProvider)

  def _clear_rollout(self):
    # Production rollouts own only local tensors, never the provider's state.
    if not self._uses_tensor_rollout:
      self.wm_env.clear_rollout()

  def _rollout(
    self, starts, process_noise, measurement_noise, horizon, *, return_components=False
  ):
    """Compatibility tuple for diagnostic callers; state is returned by _rollout_result."""
    result = self._rollout_result(starts, process_noise, measurement_noise, horizon)
    if result.invalid.any():
      raise FloatingPointError("Invalid imagined rollout")
    return result.legacy_tuple(return_components)

  def _rollout_result(self, starts, process_noise, measurement_noise, horizon):
    prepared = (
      prepare_rollout(self.wm_env, starts, horizon, process_noise, measurement_noise)
      if self._uses_tensor_rollout
      else None
    )
    result = self._rollout_kernel(
      prepared, process_noise, measurement_noise, horizon, starts=starts
    )
    return result

  def _rollout_kernel(
    self, prepared, process_noise, measurement_noise, horizon, *, starts=None
  ):
    """Fixed tensor rollout, including bootstrap/GAE, with explicit result/status.

    ``prepared`` is required for production belief training. The None branch
    preserves the custom analytical environment protocol outside CUDA capture.
    No preparation, state commits or device-to-host checks occur in the tensor path.
    """
    cfg, env = self.cfg, self.wm_env
    count = process_noise.shape[1]
    state = prepared.state if prepared is not None else None
    if prepared is None:
      obs = env.reset_prepared(starts)
      invalid = torch.zeros(count, device=process_noise.device, dtype=torch.bool)
    else:
      obs, invalid = observe(env, prepared, state, 0)
    survival = process_noise.new_ones(count)
    returns = torch.zeros_like(survival)
    observations, rewards, dones, weights = [], [], [], []
    aux_loss = 0.0
    saturation = process_noise.new_zeros(())
    film_scale = process_noise.new_zeros(2)
    scope = (
      nullcontext()
      if prepared is not None
      else belief_rollout(getattr(env, "posterior_provider", None))
    )
    latents = None
    if prepared is not None and self._encode is not None and cfg.batch_motion_encoder:
      goals = prepared.reference.local.goal[: horizon * count]
      latents = self._encode(goals).reshape(horizon, count, -1)
    with scope:
      for t in range(horizon):
        observations.append(obs["critic"].detach())
        weights.append(survival.detach())
        if self._encode is not None:
          latent = (
            latents[t] if latents is not None else self._encode(obs["goal_motion"])
          )
          actions, scale, shift = self._decode(
            latent, obs["history"], obs["task_uncertainty"]
          )
          aux = 0.0
        elif prepared is not None:
          assert self._actor_with_aux is not None
          actions, aux = self._actor_with_aux(obs)
        else:
          actions = self._actor_forward(obs)
          aux = getattr(self.actor, "auxiliary_loss", lambda: 0.0)()
        aux_loss = aux_loss + (aux.mean() if isinstance(aux, torch.Tensor) else aux)
        with torch.no_grad():
          saturation += (env.limit_actions(actions) != actions).float().mean()
          if getattr(self.actor, "film", None) is not None:
            if self._decode is None:
              scale, shift = self.actor.film(obs["task_uncertainty"]).chunk(2, -1)
            film_scale += torch.stack((scale.abs().mean(), shift.abs().mean()))
        if prepared is None:
          obs, reward, done, _ = env.step(
            actions,
            # Custom analytical environments retain their tensor-noise protocol.
            process_noise=(
              torch.zeros_like(process_noise[t])
              if cfg.num_samples == 1
              else process_noise[t]
            ),
            measurement_noise=measurement_noise[t],
          )
        else:
          state, reward, done, failed = imagine_step(
            env,
            prepared,
            state,
            actions,
            # None selects Dynamics.forward directly, bypassing process sampling.
            None if cfg.num_samples == 1 else process_noise[t],
            measurement_noise[t],
            t,
          )
          obs, bad_features = observe(env, prepared, state, t + 1)
          invalid = invalid | failed | bad_features
        returns = returns + cfg.gamma**t * survival * reward
        survival = survival * (1 - done)
        rewards.append(reward.detach())
        dones.append(done.detach())
    bootstrap = discounted_bootstrap(
      self.critic(obs["critic"]).squeeze(-1),
      survival,
      cfg.gamma,
      horizon,
      self._v_max if cfg.value_clipping else None,
    )
    reward_return = returns
    returns = returns + bootstrap
    observations.append(obs["critic"].detach())
    observations = torch.stack(observations)
    step_rewards = torch.stack(rewards)
    with torch.no_grad():
      raw_values = self.critic(observations).squeeze(-1)
      targets = gae_returns(
        raw_values[:-1],
        raw_values[-1],
        step_rewards,
        torch.stack(dones),
        cfg.gamma,
        cfg.lam,
      )
      if cfg.value_clipping:
        targets.clamp_(max=self._v_max)
    data = observations[:-1], targets
    metrics = {
      "return": returns.detach().sum(),
      "raw_reward_sum": step_rewards.sum(),
      "bootstrap": bootstrap.detach().sum(),
      "final_survival": survival.detach().sum(),
      "valid_steps": returns.new_full((), horizon * count),
      "survival_weighted_steps": torch.stack(weights).sum(),
      "action_saturation": saturation * count,
      "film_scale": film_scale * count,
      "value_clamp_fraction": (raw_values > self._v_max).float().mean() * count,
    }
    with torch.no_grad():
      invalid = invalid | ~torch.isfinite(returns) | ~torch.isfinite(targets).all(dim=0)
      invalid = invalid | ~torch.isfinite(raw_values).all(dim=0)
      if isinstance(aux_loss, torch.Tensor):
        invalid = invalid | ~torch.isfinite(aux_loss).all()
    return RolloutResult(
      returns, data, metrics, aux_loss, state, invalid, reward_return, bootstrap
    )

  @staticmethod
  def _check_gradients(module):
    gradients = [p.grad for p in module.parameters() if p.grad is not None]
    if (
      not gradients
      or not torch.stack([torch.isfinite(g).all() for g in gradients]).all()
    ):
      raise FloatingPointError("Missing or non-finite policy gradient")

  def update_from(self, starts, process_noise, measurement_noise):
    """One actor step and configured critic epochs, with fixed caller-owned noise."""
    try:
      self._validate_noises(starts, process_noise, measurement_noise)
      if self._uses_tensor_rollout:
        # Preparation and input checks run before any captured optimizer step.
        self.wm_env.validate_policy_start(starts)
        if self._actor_with_aux is None:
          raise TypeError("Belief training actor requires forward_with_aux")
      count = len(starts) * self.cfg.num_samples
      graph = self._select_policy_graph(count)
      if graph is not None:
        return self._update_with_graph(graph, starts, process_noise, measurement_noise)
      stream = self._policy_graph_stream
      if stream is None:
        return self._update_from_eager(starts, process_noise, measurement_noise)
      # Keep critic as well as actor AccumulateGrad on its original stream after
      # restore, recapture or an eager fallback.
      current = torch.cuda.current_stream(stream.device)
      stream.wait_stream(current)
      try:
        with torch.cuda.stream(stream):
          return self._update_from_eager(starts, process_noise, measurement_noise)
      finally:
        current.wait_stream(stream)
    except Exception:
      self.clear_policy_graph()
      for optimizer in (*self.actor_optimizers, *self.critic_optimizers):
        optimizer.zero_grad(set_to_none=True)
      raise

  def _update_from_eager(self, starts, process_noise, measurement_noise):
    timer = StreamTimer(next(self.actor.parameters()).device)
    cfg, env = self.cfg, self.wm_env
    starts = starts.repeat_interleave(cfg.num_samples)
    process_noise = process_noise.detach().flatten(1, 2)
    measurement_noise = measurement_noise.detach().flatten(1, 2)
    count = len(starts)
    modules = (self.actor, self.critic, env.dynamics, env.representation)
    data, totals = [], {}
    # Analytical test environments and custom rollout environments may have no motion.
    with (
      getattr(env, "readonly_policy_motion", nullcontext)(),
      getattr(env.dynamics, "readonly", nullcontext)(),
      _module_modes(modules),
    ):
      self.actor.train().requires_grad_(True)
      for module in (self.critic, env.dynamics, env.representation):
        module.eval().requires_grad_(False)
        module.zero_grad(set_to_none=True)
      try:
        with timer.measure("policy_rollout_backward_seconds"):
          for optimizer in self.actor_optimizers:
            optimizer.zero_grad(set_to_none=True)
          for offset in range(0, count, cfg.microbatch_size):
            selected = slice(offset, offset + cfg.microbatch_size)
            batch = starts.index(selected)
            with timer.measure("policy_prepare_seconds"):
              prepared = None
              if self._uses_tensor_rollout:
                prepared = prepare_rollout(
                  env,
                  batch,
                  cfg.horizon,
                  process_noise[:, selected],
                  measurement_noise[:, selected],
                )
            with timer.measure("policy_backward_seconds"):
              arguments = (
                prepared,
                process_noise[:, selected],
                measurement_noise[:, selected],
                count,
              )
              regression, metrics, failed = self._actor_backward(
                *arguments, starts=batch
              )
            if failed.any():
              raise FloatingPointError("Invalid imagined rollout; actor step rejected")
            data.append(regression)
            for name, value in metrics.items():
              totals[name] = totals.get(name, 0) + value
            del prepared, arguments, regression, metrics, failed
            self._clear_rollout()

          if self._uses_tensor_rollout:
            env.check_policy_start_identity(starts)
          self._check_gradients(self.actor)
          actor_norm = torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(), cfg.max_grad_norm, error_if_nonfinite=True
          )
          if cfg.grad_clip_value:
            torch.nn.utils.clip_grad_value_(
              self.actor.parameters(), cfg.grad_clip_value
            )
          for optimizer in self.actor_optimizers:
            optimizer.step()
          self.actor_steps += 1
        with timer.measure("critic_regression_seconds"):
          self.actor.requires_grad_(False)
          self.critic.requires_grad_(True)
          # Join the trajectory axis before flattening, preserving time-major sample order.
          features, targets = (
            torch.cat(chunks, dim=1).flatten(0, 1) for chunks in zip(*data, strict=True)
          )
          del data
          critic_loss, critic_norm, updates = fit_critic(
            self.critic,
            self.critic_optimizers,
            features,
            targets,
            epochs=cfg.critic_epochs,
            minibatches=cfg.critic_minibatches,
            microbatch_size=cfg.critic_microbatch_size,
            generator=self.generators["critic"],
            max_norm=cfg.max_grad_norm,
            check_finite=True,
          )
          self.critic_steps += updates
          with torch.no_grad():
            variance = targets.var(unbiased=False)
            if variance > 0:
              # The fitted targets are private and no longer needed. Reuse their
              # storage for residuals and keep each critic forward bounded.
              residuals = targets
              size = cfg.critic_microbatch_size or len(targets)
              for start in range(0, len(targets), size):
                chunk = slice(start, start + size)
                residuals[chunk].sub_(self.critic(features[chunk]).squeeze(-1))
              explained = 1 - residuals.var(unbiased=False) / variance
            else:
              explained = targets.new_tensor(float("nan"))
        self.last_profile = timer.seconds()
        return self._policy_metrics(
          totals, count, actor_norm, critic_loss, critic_norm, explained
        )
      except Exception:
        self.clear_policy_graph()
        for optimizer in (*self.actor_optimizers, *self.critic_optimizers):
          optimizer.zero_grad(set_to_none=True)
        raise
      finally:
        self._clear_rollout()
        if not self._uses_tensor_rollout and hasattr(self.actor, "_aux_loss"):
          self.actor._aux_loss = next(self.actor.parameters()).new_empty(0)

  def _update_with_graph(self, graph, starts, process_noise, measurement_noise):
    cfg, env = self.cfg, self.wm_env
    timer = StreamTimer(process_noise.device)
    modules = (self.actor, self.critic, env.dynamics, env.representation)
    with env.readonly_policy_motion(), env.dynamics.readonly(), _module_modes(modules):
      for module in (env.dynamics, env.representation):
        module.eval().requires_grad_(False)
        module.zero_grad(set_to_none=True)
      with timer.measure("policy_prepare_seconds"):
        starts = starts.repeat_interleave(cfg.num_samples)
        process_noise = process_noise.detach().flatten(1, 2)
        measurement_noise = measurement_noise.detach().flatten(1, 2)
        count = len(starts)
        prepared = []
        for offset in range(0, count, cfg.microbatch_size):
          selected = slice(offset, offset + cfg.microbatch_size)
          prepared.append(
            prepare_rollout(
              env,
              starts.index(selected),
              cfg.horizon,
              process_noise[:, selected],
              measurement_noise[:, selected],
            )
          )
        env.check_policy_start_identity(starts)
        samples = cfg.horizon * count
        minibatches = min(cfg.critic_minibatches, samples)
        permutation = torch.randperm(
          samples // minibatches * minibatches,
          device=process_noise.device,
          generator=self.generators["critic"],
        )
      with timer.measure("policy_update_seconds"):
        metrics, invalid = graph(
          tuple(prepared), process_noise, measurement_noise, count, permutation
        )
      # The complete update has executed, even if its results fail validation.
      self.actor_steps += 1
      self.critic_steps += cfg.critic_epochs * minibatches
      with timer.measure("policy_check_seconds"):
        graph.check_state()
        env.check_policy_start_identity(starts)
        finite = torch.stack(
          [
            torch.isfinite(value).all()
            for name, value in metrics.items()
            if name != "Policy/explained_variance"
          ]
        ).all()
        if invalid or not finite:
          raise FloatingPointError(
            "Invalid EIT policy update after execution; parameters may have changed"
          )
        metrics = {name: value.clone() for name, value in metrics.items()}
      self.last_profile = timer.seconds()
      return metrics

  def _policy_update(
    self, prepared, process_noise, measurement_noise, count, permutation
  ):
    """Single capture body: all actor microbatches and all critic optimizer steps."""
    cfg = self.cfg
    self.actor.train().requires_grad_(True)
    self.critic.eval().requires_grad_(False)
    for optimizer in self.actor_optimizers:
      optimizer.zero_grad(set_to_none=False)
    data, totals = [], {}
    invalid = process_noise.new_zeros((), dtype=torch.bool)
    for index, batch in enumerate(prepared):
      offset = index * cfg.microbatch_size
      selected = slice(offset, offset + cfg.microbatch_size)
      regression, metrics, failed = self._actor_backward(
        batch, process_noise[:, selected], measurement_noise[:, selected], count
      )
      data.append(regression)
      invalid.logical_or_(failed.any())
      for name, value in metrics.items():
        totals[name] = totals.get(name, 0) + value
    actor_norm = torch.nn.utils.clip_grad_norm_(
      self.actor.parameters(), cfg.max_grad_norm, error_if_nonfinite=False
    )
    if cfg.grad_clip_value:
      torch.nn.utils.clip_grad_value_(self.actor.parameters(), cfg.grad_clip_value)
    for optimizer in self.actor_optimizers:
      optimizer.step()
    self.actor.requires_grad_(False)
    self.critic.requires_grad_(True)
    features, targets = (
      torch.cat(chunks, dim=1).flatten(0, 1) for chunks in zip(*data, strict=True)
    )
    del data
    critic_loss, critic_norm, _ = fit_critic(
      self.critic,
      self.critic_optimizers,
      features,
      targets,
      epochs=cfg.critic_epochs,
      minibatches=cfg.critic_minibatches,
      microbatch_size=cfg.critic_microbatch_size,
      max_norm=cfg.max_grad_norm,
      permutation=permutation,
      zero_grad_set_to_none=False,
    )
    with torch.no_grad():
      variance = targets.var(unbiased=False)
      size = cfg.critic_microbatch_size or len(targets)
      for start in range(0, len(targets), size):
        chunk = slice(start, start + size)
        targets[chunk].sub_(self.critic(features[chunk]).squeeze(-1))
      invalid.logical_or_(~torch.isfinite(targets).all())
      explained = torch.where(
        variance > 0,
        1 - targets.var(unbiased=False) / variance,
        variance.new_full((), float("nan")),
      )
    return self._policy_metrics(
      totals, count, actor_norm, critic_loss, critic_norm, explained
    ), invalid

  @staticmethod
  def _policy_metrics(totals, count, actor_norm, critic_loss, critic_norm, explained):
    return {
      "Policy/return": totals["return"] / count,
      "Policy/raw_reward": totals["raw_reward_sum"] / totals["valid_steps"],
      "Policy/bootstrap": totals["bootstrap"] / count,
      "Policy/bootstrap_fraction": totals["bootstrap"].abs()
      / totals["return"].abs().clamp_min(1e-8),
      "Policy/final_survival": totals["final_survival"] / count,
      "Policy/valid_steps": totals["valid_steps"],
      "Policy/survival_weighted_steps": totals["survival_weighted_steps"],
      "Policy/action_saturation": totals["action_saturation"] / totals["valid_steps"],
      "Policy/film_gamma": totals["film_scale"][0] / totals["valid_steps"],
      "Policy/film_beta": totals["film_scale"][1] / totals["valid_steps"],
      "Policy/value_clamp_fraction": totals["value_clamp_fraction"] / count,
      "Policy/critic_loss": critic_loss,
      "Policy/explained_variance": explained,
      "Grad/actor_norm": actor_norm.detach(),
      "Grad/critic_norm": critic_norm.detach(),
    }

  def _contract(self):
    return {
      "actor_parameters": [
        (n, tuple(p.shape)) for n, p in self.actor.named_parameters()
      ],
      "critic": self.critic.contract()
      if hasattr(self.critic, "contract")
      else [(n, tuple(p.shape)) for n, p in self.critic.named_parameters()],
    }

  def state_dict(self):
    return copy.deepcopy(
      {
        "version": "belief_policy_learner_v2",
        "config": asdict(self.cfg),
        "contract": self._contract(),
        "model_version": self.wm_env.dynamics.model_version,
        "bootstrap_value_cap": self._v_max if self.cfg.value_clipping else None,
        "actor_steps": self.actor_steps,
        "critic_steps": self.critic_steps,
        "actor_optimizers": [opt.state_dict() for opt in self.actor_optimizers],
        "critic_optimizers": [opt.state_dict() for opt in self.critic_optimizers],
        "rng": {
          name: generator.get_state() for name, generator in self.generators.items()
        },
      }
    )

  def validate_state_dict(self, state):
    if (
      not isinstance(state, dict)
      or state.get("version") != "belief_policy_learner_v2"
      or state.get("config") != asdict(self.cfg)
      or state.get("contract") != self._contract()
      or state.get("model_version") != self.wm_env.dynamics.model_version
    ):
      raise ValueError("Belief policy checkpoint contract mismatch")
    for name, optimizers in (
      ("actor", self.actor_optimizers),
      ("critic", self.critic_optimizers),
    ):
      saved = state.get(f"{name}_optimizers", [])
      if len(saved) != len(optimizers):
        raise ValueError("Policy optimizer count mismatch")
      for optimizer, supplied in zip(optimizers, saved, strict=True):
        validate_optimizer_state(optimizer, supplied, state.get(f"{name}_steps"))
    rng = state.get("rng")
    if not isinstance(rng, dict) or set(rng) != set(self.generators):
      raise ValueError("Policy RNG streams mismatch")
    for name, generator in self.generators.items():
      value = rng[name]
      if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.uint8
        or value.ndim != 1
      ):
        raise ValueError("Invalid policy RNG state")
      torch.Generator(device=generator.device).set_state(value.cpu())

  def load_state_dict(self, state):
    self.validate_state_dict(state)
    self.clear_policy_graph()
    for name, optimizers in (
      ("actor", self.actor_optimizers),
      ("critic", self.critic_optimizers),
    ):
      for optimizer, supplied in zip(
        optimizers, state[f"{name}_optimizers"], strict=True
      ):
        optimizer.load_state_dict(copy.deepcopy(supplied))
        optimizer.zero_grad(set_to_none=True)
    for name, generator in self.generators.items():
      generator.set_state(state["rng"][name].cpu())
    self.actor_steps, self.critic_steps = state["actor_steps"], state["critic_steps"]
