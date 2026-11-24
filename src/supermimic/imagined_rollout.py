"""Prepared, functional imagination: no provider transactions or environment writes.

Preparation is an eager boundary. ``observe`` and ``imagine_step`` consume only
local tensors and read-only model/configuration dependencies. Their device-side
status is checked by the caller: before eager optimization, or after the complete
actor/critic update when CUDA graph replay is enabled.
"""

from dataclasses import dataclass

import torch

from supermimic.belief import BeliefState
from supermimic.critic_inputs import (
  build_critic_features_kernel,
  critic_feature_constants,
)
from supermimic.estimator import LieBeliefFilter
from supermimic.estimator_inputs import (
  EpisodeAlignment,
  ReferenceWindow,
  _validate_reference,
  build_actor_critic_goals_kernel,
  covariance_encoding_constants,
)
from supermimic.tasks.task_spec import build_actor_obs_td
from supermimic.utils.lie_state import LieState
from supermimic.utils.validation import rotation_valid


@dataclass(frozen=True)
class ImaginedState:
  belief: BeliefState
  aux: torch.Tensor
  rep: torch.Tensor
  history: torch.Tensor
  last_action: torch.Tensor
  motion_time: torch.Tensor


@dataclass(frozen=True)
class PreparedRollout:
  state: ImaginedState
  estimator: LieBeliefFilter
  context: torch.Tensor
  references: tuple[ReferenceWindow, ...]
  bodies: tuple[tuple[torch.Tensor, ...], ...]
  motion_times: tuple[torch.Tensor, ...]
  measurement_std: torch.Tensor
  actor_encoding: tuple | None
  critic_encoding: tuple | None
  critic_group: str


@dataclass(frozen=True)
class RolloutResult:
  returns: torch.Tensor
  critic_data: tuple[torch.Tensor, torch.Tensor]
  metrics: dict[str, torch.Tensor]
  auxiliary_loss: torch.Tensor | float
  final_state: ImaginedState | None
  invalid: torch.Tensor
  reward_return: torch.Tensor
  bootstrap: torch.Tensor

  def legacy_tuple(self, return_components=False):
    values = (self.returns, self.critic_data, self.metrics)
    if return_components:
      return (*values, (self.reward_return, self.bootstrap))
    return values


@torch.inference_mode(False)
@torch.no_grad()
def prepare_rollout(env, start, horizon, process_noise, measurement_noise):
  """Validate/copy detached initial conditions; precompute clip-clamped references."""
  batch, belief = env.validate_policy_start(start)
  if type(horizon) is not int or horizon < 1:
    raise ValueError("Rollout horizon must be positive")
  for name, noise, width in (
    ("process", process_noise, 70),
    ("measurement", measurement_noise, 64),
  ):
    if (
      noise.shape != (horizon, len(start), width)
      or noise.device != belief.covariance_sqrt.device
      or noise.dtype != belief.covariance_sqrt.dtype
      or not torch.isfinite(noise).all()
    ):
      raise ValueError(f"Invalid rollout {name} noise")
  aux, rep = env._unpack_initial_states(batch)
  alignment = EpisodeAlignment(
    batch["alignment_rotation"], batch["alignment_translation"]
  )
  times = batch["motion_time"]
  ends = env.motion.motion_ends
  if (
    ends.ndim != 1
    or not ends.numel()
    or ends.dtype != torch.long
    or ends.device != times.device
    or (ends <= 0).any()
    or (ends[1:] <= ends[:-1]).any()
  ):
    raise ValueError("Invalid reference clip ends")
  last = ends[torch.searchsorted(ends, times, right=True)] - 1
  references, bodies, motion_times = [], [], []
  truth = env.representation.true_rpv(aux, rep)
  for step in range(horizon + 1):
    current = torch.minimum(times + step, last)
    reference = env.representation.reference_window(env.motion, current, alignment)
    _validate_reference(truth, reference)
    body = env._aligned_reference_bodies(env.motion.body_state(current), alignment)
    if not all(torch.isfinite(value).all() for value in body):
      raise ValueError("Reference bodies must be finite")
    references.append(reference)
    bodies.append(body)
    motion_times.append(current)
  config = env.estimator_input_config
  critic_config = env._eit_critic_config()
  if critic_config.group != config.group:
    raise ValueError("Critic group and input configuration disagree")
  std = config.measurement_noise.standard_deviations(
    29, device=rep.device, dtype=rep.dtype
  )
  return PreparedRollout(
    ImaginedState(
      belief, aux, rep, batch["sensor_window"], batch["prev_action"], times
    ),
    env.posterior_provider.estimator,
    batch["sim_ctx"],
    tuple(references),
    tuple(bodies),
    tuple(motion_times),
    std,
    covariance_encoding_constants(
      rep, config.covariance_scales, config.covariance_std_floor
    )
    if config.uses_uncertainty
    else None,
    critic_feature_constants(rep, critic_config)
    if critic_config.group != "A"
    else None,
    critic_config.group,
  )


@torch.no_grad()
def rows_invalid(*values):
  invalid = torch.zeros(values[0].shape[0], dtype=torch.bool, device=values[0].device)
  for value in values:
    invalid = invalid | ~torch.isfinite(value.reshape(value.shape[0], -1)).all(-1)
  return invalid


def observe(env, prepared, state, step):
  truth = env.representation.true_rpv(state.aux, state.rep)
  actor, uncertainty, goal, invalid = build_actor_critic_goals_kernel(
    env.estimator_input_config,
    truth,
    state.belief.to_actor_input(),
    prepared.references[step],
    prepared.actor_encoding,
  )
  critic = build_critic_features_kernel(
    LieState.from_rep(state.aux, state.rep),
    goal,
    state.history,
    prepared.context,
    state.belief,
    group=prepared.critic_group,
    constants=prepared.critic_encoding,
  )
  obs = build_actor_obs_td(
    {"motion": actor}, state.history, critic=critic, task_uncertainty=uncertainty
  )
  invalid = invalid | rows_invalid(*obs.values()) | ~rotation_valid(truth.rotation)
  return obs, invalid


def imagine_step(env, prepared, state, actions, process_noise, measurement_noise, step):
  """Advance one fixed rollout frame, retaining every pathwise derivative."""
  executed = env.limit_actions(actions)
  aux, rep, joints, done, reward = env._wm_step_kernel(
    state.rep,
    state.aux,
    executed,
    state.last_action,
    prepared.context,
    prepared.bodies[step],
    process_noise=process_noise,
  )
  sensor = env.representation.compute_actor_sensor(
    rep,
    prepared.context,
    aux_state=aux,
    joint_state=joints,
    last_action=executed,
    noisy=False,
  )
  sensor = torch.cat(
    (sensor[..., :64] + measurement_noise * prepared.measurement_std, sensor[..., 64:]),
    -1,
  )
  mean, lower, invalid = prepared.estimator.predict_correct_kernel(
    state.belief.mean,
    state.belief.covariance_sqrt,
    executed,
    prepared.context,
    sensor[..., :64],
    measurement_std=prepared.measurement_std,
  )
  history = torch.cat((state.history[:, 1:], sensor.unsqueeze(1)), dim=1)
  next_state = ImaginedState(
    BeliefState(mean, covariance_sqrt=lower),
    aux,
    rep,
    history,
    executed,
    prepared.motion_times[step + 1],
  )
  invalid = invalid | rows_invalid(actions, aux, rep, joints, done, reward, sensor)
  return next_state, reward, done, invalid
