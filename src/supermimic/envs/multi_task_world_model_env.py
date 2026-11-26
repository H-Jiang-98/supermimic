"""Multi-task differentiable world model env for concurrent encoder training."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from tensordict import TensorDict

from supermimic.envs.world_model_env import WorldModelEnv
from supermimic.estimator_inputs import EITInputConfig, PosteriorProvider
from supermimic.tasks.task_spec import build_actor_obs_td

if TYPE_CHECKING:
  from supermimic.models import Dynamics
  from supermimic.replay_buffer import ReplayBuffer
  from supermimic.representations import StateRepresentation
  from supermimic.tasks.task_spec import TaskSpec


class MultiTaskWorldModelEnv(WorldModelEnv):
  """WM env for motion tracking with per-slot goal modalities.

  Reward is motion-tracking for every slot; tasks differ only in what
  ``goal_<name>`` key the actor's encoder consumes. ``get_observations``
  emits all per-task goal keys full-batch and the actor slices its own
  slot range. The obs-history ring (history_length > 1) lives in the base.
  """

  _supports_estimator_inputs = True

  def __init__(
    self,
    dynamics: Dynamics,
    representation: StateRepresentation,
    env: Any,
    batch_size: int,
    replay_buffer: ReplayBuffer,
    task_specs: list[TaskSpec],
    history_length: int,
    termination_softness: float,
    device: str = "cpu",
    estimator_input_config: EITInputConfig | None = None,
    posterior_provider: PosteriorProvider | None = None,
  ) -> None:
    self.task_specs = task_specs
    assert task_specs[0].name == "motion", "task_specs[0] must be the motion task"
    if estimator_input_config is not None and (
      len(task_specs) != 1 or task_specs[0].goal_dim != estimator_input_config.goal_dim
    ):
      raise ValueError(
        f"Expected a single {estimator_input_config.goal_dim}-dimensional motion task"
      )
    super().__init__(
      dynamics=dynamics,
      representation=representation,
      env=env,
      batch_size=batch_size,
      replay_buffer=replay_buffer,
      history_length=history_length,
      termination_softness=termination_softness,
      device=device,
      estimator_input_config=estimator_input_config,
      posterior_provider=posterior_provider,
    )

  def _build_task_goals(self) -> None:
    # The anchor task's goal is the actor goal already built in _update_goal; reuse it.
    anchor, *rest = self.task_specs
    self._task_goal_obs: dict[str, torch.Tensor] = {anchor.name: self._actor_goal_obs}
    for ts in rest:
      self._task_goal_obs[ts.name] = ts.extract_training_goal(
        self.motion, self.motion_times, self.aux_state
      ).clone()

  def get_observations(self) -> TensorDict:
    self._check_belief_frame()
    history = self._sensor_history()
    return self._observations_with_critic(history, self._critic_obs(history))

  def _observations_with_critic(self, history, critic) -> TensorDict:
    return build_actor_obs_td(
      self._task_goal_obs,
      history,
      critic=critic,
      task_uncertainty=self._task_uncertainty,
    )
