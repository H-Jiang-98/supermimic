"""Decompose the training actor objective on fixed starts and caller-owned noise."""

from __future__ import annotations

import hashlib
import math
from contextlib import nullcontext

import torch

from supermimic.learners.belief_policy_learner import _module_modes


def _tensor_digest(value):
  value = value.detach().cpu().contiguous()
  digest = hashlib.sha256(str((value.dtype, tuple(value.shape))).encode())
  digest.update(value.numpy().tobytes())
  return digest.hexdigest()


def _gradient_summary(reward, bootstrap):
  """Accumulate scalar products in double precision without a giant flatten/cat."""
  rr = sum(float(g.double().square().sum()) for g in reward)
  bb = sum(float(g.double().square().sum()) for g in bootstrap)
  rb = sum(
    float((a.double() * b.double()).sum())
    for a, b in zip(reward, bootstrap, strict=True)
  )
  cosine = max(-1.0, min(1.0, rb / math.sqrt(rr * bb))) if rr and bb else None
  return {
    "reward_gradient_norm": math.sqrt(rr),
    "bootstrap_gradient_norm": math.sqrt(bb),
    "total_gradient_norm": math.sqrt(max(0.0, rr + bb + 2 * rb)),
    "gradient_dot_product": rb,
    "gradient_cosine": cosine,
    "gradient_angle_degrees": math.degrees(math.acos(cosine))
    if cosine is not None
    else None,
    "bootstrap_to_reward_gradient_norm": math.sqrt(bb / rr) if rr else None,
  }


@torch.enable_grad()
def actor_objective_gradients(learner, starts, process_noise, measurement_noise):
  """Report gradients of the mean positive return, before clipping or optimization.

  Uses the learner's actual rollout, fixed horizon, soft survival, value cap,
  sample replication and microbatch schedule. Both components come from the same
  graph. No optimizer, generator, parameter gradient, or normalizer is updated.
  The caller's transient imagined rollout is cleared on exit.
  """
  learner._validate_noises(starts, process_noise, measurement_noise)
  cfg, env = learner.cfg, learner.wm_env
  n_starts = len(starts)
  starts = starts.repeat_interleave(cfg.num_samples)
  process = process_noise.detach().flatten(1, 2)
  measurement = measurement_noise.detach().flatten(1, 2)
  modules = (learner.actor, learner.critic, env.dynamics, env.representation)
  parameters = tuple(learner.actor.parameters())
  if not parameters:
    raise ValueError("Actor has no parameters")
  gradients = [[torch.zeros_like(p) for p in parameters] for _ in range(2)]
  totals = [0.0, 0.0]
  buffers = {
    buffer: buffer.detach().clone() for module in modules for buffer in module.buffers()
  }
  auxiliary = getattr(learner.actor, "_aux_loss", None)
  device = parameters[0].device
  devices = (
    [device.index if device.index is not None else torch.cuda.current_device()]
    if device.type == "cuda"
    else []
  )
  with (
    torch.random.fork_rng(devices=devices),
    getattr(env, "readonly_policy_motion", nullcontext)(),
    _module_modes(modules),
  ):
    learner.actor.train().requires_grad_(True)
    for module in modules[1:]:
      module.eval().requires_grad_(False)
    try:
      for offset in range(0, len(starts), cfg.microbatch_size):
        selected = slice(offset, offset + cfg.microbatch_size)
        returns, _, _, components = learner._rollout(
          starts.index(selected),
          process[:, selected],
          measurement[:, selected],
          cfg.horizon,
          return_components=True,
        )
        if not torch.isfinite(returns).all():
          raise FloatingPointError("Non-finite imagined diagnostic return")
        for i, component in enumerate(components):
          objective = component.sum() / len(starts)
          totals[i] += float(objective.detach())
          if objective.requires_grad:
            values = torch.autograd.grad(
              objective,
              parameters,
              allow_unused=True,
              retain_graph=i == 0,
            )
            for accumulated, value in zip(gradients[i], values, strict=True):
              if value is not None:
                if not torch.isfinite(value).all():
                  raise FloatingPointError("Non-finite actor diagnostic gradient")
                accumulated.add_(value.detach())
        del returns, components
        env.clear_rollout()
    finally:
      env.clear_rollout()
      with torch.no_grad():
        for buffer, original in buffers.items():
          if not torch.equal(buffer, original):
            buffer.copy_(original)
      if auxiliary is not None:
        learner.actor._aux_loss = auxiliary
  return {
    "objective": "mean_positive_discounted_soft_survival_return",
    "gradient_scope": "all_actor_parameters_before_clipping",
    "starts": n_starts,
    "samples_per_start": cfg.num_samples,
    "horizon": cfg.horizon,
    "microbatch_size": cfg.microbatch_size,
    "effective_imagined_steps": len(starts) * cfg.horizon,
    "gamma": cfg.gamma,
    "bootstrap_value_cap": learner._v_max if cfg.value_clipping else None,
    "reward_return": totals[0],
    "bootstrap_return": totals[1],
    "total_return": sum(totals),
    "process_noise_sha256": _tensor_digest(process_noise),
    "measurement_noise_sha256": _tensor_digest(measurement_noise),
    **_gradient_summary(*gradients),
  }
