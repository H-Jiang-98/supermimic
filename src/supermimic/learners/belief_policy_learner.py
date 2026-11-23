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

from supermimic.belief_provider import belief_rollout
from supermimic.learners.learner_utils import validate_optimizer_state
from supermimic.utils.profiling import PhaseTimer


@dataclass(frozen=True)
class BeliefPolicyLearnerCfg:
  batch_size: int = 4
  num_samples: int = 2
  microbatch_size: int = 2
  horizon: int = 4
  num_batches: int = 1
  noise: float = 0.1
  lr: float = 2e-4
  critic_lr: float = 2e-4
  critic_epochs: int = 2
  gamma: float = 0.99
  lam: float = 0.95
  max_grad_norm: float = 1.0
  value_clipping: bool = True

  def __post_init__(self):
    for key in (
      "batch_size",
      "num_samples",
      "microbatch_size",
      "horizon",
      "num_batches",
      "critic_epochs",
    ):
      if type(getattr(self, key)) is not int or getattr(self, key) < 1:
        raise ValueError(f"Invalid policy {key}")
    for key in ("lr", "critic_lr", "max_grad_norm"):
      if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
        raise ValueError(f"Invalid policy {key}")
    if not math.isfinite(self.noise) or self.noise < 0:
      raise ValueError("Invalid policy collection noise")
    if not 0 <= self.gamma < 1 or not 0 <= self.lam <= 1:
      raise ValueError("Policy requires 0 <= gamma < 1 and 0 <= lambda <= 1")
    if type(self.value_clipping) is not bool:
      raise ValueError("value_clipping must be boolean")


def lambda_returns(rewards, continuation, values, gamma: float, lam: float):
  """Time-major lambda returns; values includes the last bootstrap value.

  continuation is a *single-step* multiplier, not accumulated survival. An
  ending transition still receives its immediate reward.
  """
  if rewards.shape != continuation.shape or values.shape != (
    rewards.shape[0] + 1,
    *rewards.shape[1:],
  ):
    raise ValueError("Lambda-return shapes require T rewards and T+1 values")
  following = values[-1]
  targets = []
  for t in reversed(range(len(rewards))):
    following = rewards[t] + gamma * continuation[t] * (
      (1 - lam) * values[t + 1] + lam * following
    )
    targets.append(following)
  return torch.stack(targets[::-1])


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
  def __init__(self, actor, critic, wm_env, cfg=None, device=None):
    del device  # Modules and prepared starts own the actual device/dtype.
    self.actor: nn.Module = actor
    self.critic: nn.Module = critic
    self.wm_env = wm_env
    self.wm_env.eit_critic = True
    self.wm_env.critic_input_config = getattr(critic, "config", None)
    self.cfg = cfg or BeliefPolicyLearnerCfg()
    self.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=self.cfg.lr)
    self.critic_optimizer = torch.optim.Adam(critic.parameters(), lr=self.cfg.critic_lr)
    self.actor_steps = self.critic_steps = 0
    parameter = next(actor.parameters())
    self.generators = {}
    for offset, name in enumerate(("sampling", "process", "measurement")):
      generator = torch.Generator(device=parameter.device)
      generator.manual_seed((torch.initial_seed() + offset) % (2**63 - 1))
      self.generators[name] = generator
    self._v_max = wm_env.max_step_reward / (1 - self.cfg.gamma)

  @property
  def steps(self):
    return self.actor_steps

  def _value(self, inputs):
    value = self.critic(inputs).squeeze(-1)
    return value.clamp(max=self._v_max) if self.cfg.value_clipping else value

  def draw_noises(self, starts):
    parameter = next(self.actor.parameters())
    shape = (self.cfg.horizon, len(starts), self.cfg.num_samples)
    return tuple(
      torch.randn(
        *shape,
        width,
        dtype=parameter.dtype,
        device=parameter.device,
        generator=self.generators[name],
      )
      for name, width in (("process", 70), ("measurement", 64))
    )

  def sample_start_keys(self, episode_ids):
    """Draw the logical starts for every actor update, in optimizer-step order."""
    episodes, times = self.wm_env.sample_policy_keys(
      self.cfg.batch_size,
      episode_ids,
      generator=self.generators["sampling"],
      num_batches=self.cfg.num_batches,
    )
    return torch.stack((episodes, times), dim=1)

  def update(self, episode_ids):
    with self.wm_env.readonly_policy_motion():
      keys = self.sample_start_keys(episode_ids)
      starts = self.wm_env.prepare_policy_starts(*keys.unbind(dim=1))
      return self.update_prepared(starts)

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

  def _rollout(
    self, starts, process_noise, measurement_noise, horizon, *, return_components=False
  ):
    """One equal-length subset; returns live returns and detached regression data."""
    cfg, env = self.cfg, self.wm_env
    obs = env.reset_prepared(starts)
    survival = process_noise.new_ones(len(starts))
    returns = torch.zeros_like(survival)
    observations, rewards, continuations, weights = [], [], [], []
    saturation = process_noise.new_zeros(())
    film_scale = process_noise.new_zeros(2)
    with belief_rollout(getattr(env, "posterior_provider", None)):
      for t in range(horizon):
        observations.append(obs["critic"].detach())
        weights.append(survival.detach())
        actions = self.actor(obs)
        with torch.no_grad():
          saturation += (env.limit_actions(actions) != actions).float().mean()
          if getattr(self.actor, "film", None) is not None:
            scale, shift = self.actor.film(obs["task_uncertainty"]).chunk(2, -1)
            film_scale += torch.stack((scale.abs().mean(), shift.abs().mean()))
        obs, reward, done, _ = env.step(
          actions,
          process_noise=process_noise[t],
          measurement_noise=measurement_noise[t],
        )
        continuation = (1 - done) * (starts.remaining_steps > t + 1)
        returns = returns + cfg.gamma**t * survival * reward
        survival = survival * continuation
        rewards.append(reward.detach())
        continuations.append(continuation.detach())
    bootstrap = cfg.gamma**horizon * survival * self._value(obs["critic"])
    reward_return = returns
    returns = returns + bootstrap
    observations.append(obs["critic"].detach())
    observations = torch.stack(observations)
    with torch.no_grad():
      raw_values = self.critic(observations).squeeze(-1)
      values = raw_values.clamp(max=self._v_max) if cfg.value_clipping else raw_values
      targets = lambda_returns(
        torch.stack(rewards), torch.stack(continuations), values, cfg.gamma, cfg.lam
      )
      if cfg.value_clipping:
        targets.clamp_(max=self._v_max)
    data = (
      observations[:-1].flatten(0, 1),
      targets.flatten(),
      torch.stack(weights).flatten(),
    )
    metrics = {
      "return": returns.detach().sum(),
      "raw_reward_sum": torch.stack(rewards).sum(),
      "bootstrap": bootstrap.detach().sum(),
      "final_survival": survival.detach().sum(),
      "valid_steps": returns.new_tensor(horizon * len(starts)),
      "survival_weighted_steps": torch.stack(weights).sum(),
      "action_saturation": saturation * len(starts),
      "film_scale": film_scale * len(starts),
      "value_clamp_fraction": (raw_values > self._v_max).float().mean() * len(starts),
    }
    if return_components:
      return returns, data, metrics, (reward_return, bootstrap)
    return returns, data, metrics

  @staticmethod
  def _check_gradients(module):
    gradients = [p.grad for p in module.parameters() if p.grad is not None]
    if not gradients or any(not torch.isfinite(g).all() for g in gradients):
      raise FloatingPointError("Missing or non-finite policy gradient")

  def update_from(self, starts, process_noise, measurement_noise):
    """One actor step and configured critic epochs, with fixed caller-owned noise."""
    self._validate_noises(starts, process_noise, measurement_noise)
    timer = PhaseTimer(next(self.actor.parameters()).device)
    self.last_profile = timer.seconds
    cfg, env = self.cfg, self.wm_env
    starts = starts.repeat_interleave(cfg.num_samples)
    process_noise = process_noise.detach().flatten(1, 2)
    measurement_noise = measurement_noise.detach().flatten(1, 2)
    lengths = starts.remaining_steps.clamp(max=cfg.horizon)
    if (lengths < 1).any():
      raise ValueError("Prepared policy starts must have a remaining scoring action")
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
      self.actor_optimizer.zero_grad(set_to_none=True)
      actor_start = timer.now()
      try:
        for horizon in lengths.unique(sorted=True).tolist():
          indices = (lengths == horizon).nonzero().flatten()
          for selected in indices.split(cfg.microbatch_size):
            returns, regression, metrics = self._rollout(
              starts.index(selected),
              process_noise[:, selected],
              measurement_noise[:, selected],
              horizon,
            )
            if not torch.isfinite(returns).all():
              raise FloatingPointError(
                "Non-finite imagined return; actor step rejected"
              )
            (-returns.sum() / count).backward()
            data.append(regression)
            for name, value in metrics.items():
              totals[name] = totals.get(name, 0) + value
            del returns
            env.clear_rollout()

        self._check_gradients(self.actor)
        actor_norm = torch.nn.utils.clip_grad_norm_(
          self.actor.parameters(), cfg.max_grad_norm, error_if_nonfinite=True
        )
        self.actor_optimizer.step()
        self.actor_steps += 1
        timer.seconds["policy_rollout_backward_seconds"] = timer.now() - actor_start
        critic_start = timer.now()
        self.actor.requires_grad_(False)
        self.critic.requires_grad_(True)
        features, targets, weights = (
          torch.cat([record[i] for record in data]) for i in range(3)
        )
        denominator = weights.sum()
        if not torch.isfinite(denominator) or denominator <= 0:
          raise FloatingPointError("Invalid critic survival weights")
        critic_losses, critic_norms = [], []
        for _ in range(cfg.critic_epochs):
          self.critic_optimizer.zero_grad(set_to_none=True)
          loss_sum = targets.new_zeros(())
          for offset in range(0, len(targets), cfg.microbatch_size):
            selection = slice(offset, offset + cfg.microbatch_size)
            prediction = self.critic(features[selection]).squeeze(-1)
            loss = (
              weights[selection] * (prediction - targets[selection]).square()
            ).sum()
            if not torch.isfinite(loss):
              raise FloatingPointError("Non-finite critic loss; critic step rejected")
            (loss / denominator).backward()
            loss_sum += loss.detach()
          self._check_gradients(self.critic)
          critic_norms.append(
            torch.nn.utils.clip_grad_norm_(
              self.critic.parameters(), cfg.max_grad_norm, error_if_nonfinite=True
            )
          )
          self.critic_optimizer.step()
          self.critic_steps += 1
          critic_losses.append(loss_sum / denominator)
        with torch.no_grad():
          errors = self.critic(features).squeeze(-1) - targets
          variance = targets.var(unbiased=False)
          explained = (
            1 - errors.var(unbiased=False) / variance
            if variance > 0
            else targets.new_tensor(float("nan"))
          )
        timer.seconds["critic_regression_seconds"] = timer.now() - critic_start
        return {
          "Policy/return": totals["return"] / count,
          "Policy/raw_reward": totals["raw_reward_sum"] / totals["valid_steps"],
          "Policy/bootstrap": totals["bootstrap"] / count,
          "Policy/bootstrap_fraction": totals["bootstrap"].abs()
          / totals["return"].abs().clamp_min(1e-8),
          "Policy/final_survival": totals["final_survival"] / count,
          "Policy/valid_steps": totals["valid_steps"],
          "Policy/survival_weighted_steps": totals["survival_weighted_steps"],
          "Policy/action_saturation": totals["action_saturation"]
          / totals["valid_steps"],
          "Policy/film_gamma": totals["film_scale"][0] / totals["valid_steps"],
          "Policy/film_beta": totals["film_scale"][1] / totals["valid_steps"],
          "Policy/value_clamp_fraction": totals["value_clamp_fraction"] / count,
          "Policy/critic_loss": torch.stack(critic_losses).mean(),
          "Policy/explained_variance": explained,
          "Grad/actor_norm": actor_norm.detach(),
          "Grad/critic_norm": torch.stack(critic_norms).mean().detach(),
        }
      except Exception:
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        raise
      finally:
        env.clear_rollout()
        if hasattr(self.actor, "_aux_loss"):
          self.actor._aux_loss = next(self.actor.parameters()).new_empty(0)

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
        "version": "belief_policy_learner_v1",
        "config": asdict(self.cfg),
        "contract": self._contract(),
        "model_version": self.wm_env.dynamics.model_version,
        "bootstrap_value_cap": self._v_max if self.cfg.value_clipping else None,
        "actor_steps": self.actor_steps,
        "critic_steps": self.critic_steps,
        "actor_optimizer": self.actor_optimizer.state_dict(),
        "critic_optimizer": self.critic_optimizer.state_dict(),
        "rng": {
          name: generator.get_state() for name, generator in self.generators.items()
        },
      }
    )

  def validate_state_dict(self, state):
    if (
      not isinstance(state, dict)
      or state.get("version") != "belief_policy_learner_v1"
      or state.get("config") != asdict(self.cfg)
      or state.get("contract") != self._contract()
      or state.get("model_version") != self.wm_env.dynamics.model_version
    ):
      raise ValueError("Belief policy checkpoint contract mismatch")
    for name, optimizer in (
      ("actor", self.actor_optimizer),
      ("critic", self.critic_optimizer),
    ):
      validate_optimizer_state(
        optimizer, state.get(f"{name}_optimizer"), state.get(f"{name}_steps")
      )
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
    for name, optimizer in (
      ("actor", self.actor_optimizer),
      ("critic", self.critic_optimizer),
    ):
      supplied = copy.deepcopy(state[f"{name}_optimizer"])
      for values in supplied["state"].values():
        values["step"] = values["step"].cpu()
      optimizer.load_state_dict(supplied)
      optimizer.zero_grad(set_to_none=True)
    for name, generator in self.generators.items():
      generator.set_state(state["rng"][name].cpu())
    self.actor_steps, self.critic_steps = state["actor_steps"], state["critic_steps"]
