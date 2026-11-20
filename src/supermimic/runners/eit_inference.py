"""Shared online EIT policy adapter for tracking evaluation and playback.

The caller owns physics and reset scheduling. This adapter owns one measured
frame, its history and the corresponding posterior; it has no replay or learner.
"""

from __future__ import annotations

from dataclasses import replace

import torch
from tensordict import TensorDict

from supermimic.belief import GaussianStatePrior
from supermimic.belief_provider import BeliefProvider
from supermimic.estimator import LieBeliefFilter
from supermimic.estimator_inputs import build_actor_features
from supermimic.runners.run_config import load_runner_cfg
from supermimic.runners.supermimic_runner import (
  SuperMimicRunnerCfg,
  WorldModel,
  build_actor,
  build_world_model,
  load_actor_checkpoint,
)
from supermimic.tasks.task_spec import build_actor_obs_td
from supermimic.utils.helpers import normalized_action_limits
from supermimic.utils.lie_state import LieState


def evaluation_config(saved: dict) -> SuperMimicRunnerCfg:
  """Decode a complete EIT checkpoint without importing training state."""
  if saved.get("version") != "eit_training_v1":
    raise ValueError("EIT evaluation requires an eit_training_v1 checkpoint")
  cfg = load_runner_cfg({"runner_cfg": saved["runner_config"]})
  if cfg.estimator_inputs is None or cfg.process_noise is None:
    raise ValueError("EIT evaluation requires estimator inputs and process noise")
  cfg.estimator_inputs = replace(cfg.estimator_inputs, mode="eval")
  cfg.policy.noise = 0.0
  cfg.wandb_run_path = None
  cfg.cuda_graphs = False
  cfg.validate_estimator_inputs()
  return cfg


def load_evaluation_models(env, cfg, saved: dict, device: str):
  """Build private inference models using the existing versioned loaders.

  As with ``build_world_model``, environment construction/shape discovery resets
  physics. Call the adapter's reset after the final evaluation reset or clip seek.
  Motion identity is deliberately not a weight compatibility constraint here.
  """
  if saved.get("version") != "eit_training_v1":
    raise ValueError("EIT evaluation requires an eit_training_v1 checkpoint")
  if cfg.estimator_inputs is None or cfg.estimator_inputs.mode != "eval":
    raise ValueError("Evaluation models require estimator_inputs.mode='eval'")
  wm = build_world_model(cfg, env, device)
  wm.load_bundle(saved["world_model"])
  if (
    wm.filter_config != cfg.filter_config
    or wm.initial_prior != cfg.initial_prior
    or wm.representation.measurement_config != cfg.estimator_inputs.measurement_noise
  ):
    raise ValueError("Checkpoint estimator settings disagree with runner configuration")
  actor = build_actor(cfg, wm, device)
  load_actor_checkpoint(actor, cfg, saved["actor"], for_evaluation=True)
  wm.dynamics.eval().requires_grad_(False)
  actor.eval().requires_grad_(False)
  return wm, actor


class EITPolicyAdapter:
  """Stateful evaluation inputs shared by headless evaluation and viewers.

  ``reset`` consumes the current reset observation, ``act`` is idempotent for
  that frame, and ``advance`` consumes exactly one next observation after a
  physics step. Auto-reset rows receive a new episode and repeated first frame.
  A reference-only scrub refreshes the action while reusing the same posterior.
  Calling ``reset`` never resets the physical environment or visualization.
  """

  def __init__(self, env, actor, wm: WorldModel, cfg: SuperMimicRunnerCfg):
    if cfg.estimator_inputs is None or cfg.estimator_inputs.mode != "eval":
      raise ValueError("EIT policy adapter requires estimator_inputs.mode='eval'")
    if wm.dynamics.process_noise_config is None:
      raise ValueError("EIT policy adapter requires stochastic dynamics")
    cfg.validate_estimator_inputs()
    if (
      wm.representation.measurement_config != cfg.estimator_inputs.measurement_noise
      or wm.filter_config != cfg.filter_config
      or wm.initial_prior != cfg.initial_prior
    ):
      raise ValueError("Adapter estimator settings disagree with runner configuration")
    self.env, self.actor, self.wm, self.cfg = env, actor, wm, cfg
    self.actor.eval()
    self.wm.dynamics.eval()
    self.command = env.unwrapped.command_manager.get_term("motion")
    self.device = next(actor.parameters()).device
    self.num_envs = env.num_envs
    self.provider = BeliefProvider(
      LieBeliefFilter(
        wm.dynamics,
        cfg.estimator_inputs.measurement_noise,
        wm.representation._default_joint_pos,
        config=wm.filter_config,
      ),
      wm.initial_prior,
      differentiable=False,
    )
    self._lower, self._upper = normalized_action_limits(
      env, wm.num_actions, device=self.device
    )
    self.history: torch.Tensor | None = None
    self.episode_id: torch.Tensor | None = None
    self.time_index: torch.Tensor | None = None
    self.last_inputs: TensorDict | None = None
    self._reset_mask: torch.Tensor | None = None
    self._obs: TensorDict | None = None
    self._actions: torch.Tensor | None = None
    self._reference_key: tuple[torch.Tensor, ...] | None = None

  def _observation(self, obs=None) -> TensorDict:
    if obs is None:
      obs = self.env.get_observations()
    values = {}
    parameter = next(self.actor.parameters())
    for name, width in (("full_state", 71), ("actor", 93)):
      value = obs[name].to(device=self.device)
      if (
        value.shape != (self.num_envs, width)
        or value.dtype != parameter.dtype
        or not torch.isfinite(value).all()
      ):
        raise ValueError(f"Invalid EIT {name} observation")
      values[name] = value.detach().clone()
    if not torch.equal(
      values["actor"][:, 64:], self.env.unwrapped.action_manager.action
    ):
      raise ValueError("Measured previous action differs from executed action")
    return TensorDict(values, batch_size=[self.num_envs])

  @torch.no_grad()
  def reset(self, obs=None) -> None:
    current = self._observation(obs)
    sensor = self.wm.representation.add_proprio_noise(current["actor"])
    self.provider.clear()
    self.history = sensor[:, None].repeat(1, self.cfg.history_length, 1)
    slots = torch.arange(self.num_envs, device=self.device)
    self.episode_id = (
      slots if self.episode_id is None else self.episode_id + self.num_envs
    )
    self.time_index = torch.zeros_like(slots)
    self._reset_mask = torch.ones_like(slots, dtype=torch.bool)
    self._obs, self._actions, self.last_inputs = current, None, None
    self._reference_key = None

  @torch.no_grad()
  def act(self, obs=None) -> torch.Tensor:
    if self._obs is None:
      self.reset(obs)
    elif obs is not None:
      current = self._observation(obs)
      if any(not torch.equal(current[key], self._obs[key]) for key in current.keys()):
        raise ValueError("New observation requires advance() or reset() before act()")
    assert self._obs is not None and self.history is not None
    assert self._reset_mask is not None
    representation = self.wm.representation
    context = self.wm.sim_context.read(self.env.unwrapped)
    # Even a cached action requires the same model and calibrated context.
    belief = self.provider.lookup(self.episode_id, self.time_index, context)
    alignment = self.command.reference_alignment
    reference_key = (
      self.command.time_steps,
      alignment.rotation,
      alignment.translation,
    )
    if (
      self._actions is not None
      and belief is not None
      and self._reference_key is not None
      and all(
        torch.equal(value, previous)
        for value, previous in zip(reference_key, self._reference_key, strict=True)
      )
    ):
      return self._actions.clone()
    state = self._obs["full_state"]
    rep, aux = representation.represent(state), representation.extract_aux_state(state)
    candidate = None
    if belief is None:
      sensor = self.history[:, -1]
      candidate = self.provider.prepare(
        measurement=sensor[:, :64],
        previous_action=sensor[:, 64:],
        reset_mask=self._reset_mask,
        context=context,
        episode_id=self.episode_id,
        time_index=self.time_index,
        reset_truth=(
          LieState.from_rep(aux[self._reset_mask], rep[self._reset_mask])
          if isinstance(self.provider.prior_factory, GaussianStatePrior)
          and self._reset_mask.any()
          else None
        ),
      )
      belief = candidate.state.belief
    # Scrubbing the reference changes goals, not the physical measured frame.
    # Reuse Q for that frame instead of fusing the same measurement a second time.
    reference = representation.reference_window(
      self.command.motion,
      self.command.time_steps,
      self.command.reference_alignment,
    )
    goal, uncertainty = build_actor_features(
      self.cfg.estimator_inputs,
      representation.true_rpv(aux, rep),
      belief.to_actor_input(),
      reference,
    )
    inputs = build_actor_obs_td(
      {"motion": goal}, self.history, task_uncertainty=uncertainty
    )
    actions = self.actor(inputs).clamp(min=self._lower, max=self._upper)
    if not torch.isfinite(actions).all():
      raise ValueError("EIT actor produced non-finite actions")
    if candidate is not None:
      self.provider.commit(candidate)
    self.last_inputs = inputs.detach().clone()
    self._actions = actions.detach().clone()
    self._reference_key = tuple(value.clone() for value in reference_key)
    return self._actions.clone()

  @torch.no_grad()
  def advance(self, obs, dones: torch.Tensor) -> None:
    if self._actions is None or self.history is None:
      raise RuntimeError("advance() requires one act() and physical step first")
    current = self._observation(obs)
    if dones.shape != (self.num_envs,) or dones.device != self.device:
      raise ValueError("dones must match the environment batch and device")
    reset = dones.bool()
    sensor = self.wm.representation.add_proprio_noise(current["actor"])
    shifted = torch.cat((self.history[:, 1:], sensor[:, None]), dim=1)
    self.history = torch.where(reset[:, None, None], sensor[:, None], shifted)
    self.episode_id = self.episode_id + self.num_envs * reset.long()
    self.time_index = torch.where(reset, 0, self.time_index + 1)
    self._reset_mask = reset.clone()
    self._obs, self._actions, self.last_inputs = current, None, None
    self._reference_key = None
