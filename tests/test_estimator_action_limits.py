"""Executed-command consistency across wrapper and physical affine clipping."""

from types import SimpleNamespace

import pytest
import torch
from mjlab.envs.mdp.actions import JointPositionAction

from supermimic.envs.motion_tracking_env import MotionTrackingEnv
from supermimic.estimator_inputs import EITInputConfig
from supermimic.utils.helpers import normalized_action_limits


def _wrapped_action(*, wrapper=2.0):
  term = object.__new__(JointPositionAction)
  term._action_dim = 29
  term._scale = torch.ones(2, 29)
  term._scale[:, 0] = -2.0
  term._scale[:, 1] = 0.5
  term._offset = torch.full((2, 29), 0.4)
  term.cfg = SimpleNamespace(clip={".*": (-0.6, 2.4)})
  term._clip = torch.tensor([-0.6, 2.4]).expand(2, 29, 2).clone()
  term._raw_actions = torch.zeros(2, 29)
  term._processed_actions = torch.zeros(2, 29)
  manager = SimpleNamespace(active_terms=["joint"], get_term=lambda _: term)
  env = SimpleNamespace(
    unwrapped=SimpleNamespace(action_manager=manager), clip_actions=wrapper
  )
  return env, term


def test_normalized_limits_match_actual_joint_position_processing():
  env, term = _wrapped_action()
  lower, upper = normalized_action_limits(env, 29)
  torch.testing.assert_close(lower[:3], torch.tensor([-1.0, -2.0, -1.0]))
  torch.testing.assert_close(upper[:3], torch.tensor([0.5, 2.0, 2.0]))
  actions = torch.linspace(-8.0, 8.0, 58).reshape(2, 29)
  term.process_actions(actions.clamp(-env.clip_actions, env.clip_actions))
  expected_physical = term._processed_actions.clone()
  limited = actions.clamp(min=lower, max=upper)
  term.process_actions(limited)
  torch.testing.assert_close(term._processed_actions, expected_physical)
  assert torch.equal(term.raw_action, limited)


@pytest.mark.parametrize("name", ["_scale", "_offset", "_clip"])
def test_per_environment_action_mapping_is_rejected(name):
  env, term = _wrapped_action()
  getattr(term, name)[1] += 0.1
  with pytest.raises(ValueError, match="identical across environments"):
    normalized_action_limits(env, 29)


def test_non_affine_or_multiple_action_terms_are_rejected():
  env, _ = _wrapped_action()
  env.unwrapped.action_manager.get_term = lambda _: object()
  with pytest.raises(ValueError, match="absolute 29-joint"):
    normalized_action_limits(env, 29)
  env.unwrapped.action_manager.active_terms = ["joint", "other"]
  with pytest.raises(ValueError, match="one affine"):
    normalized_action_limits(env, 29)


def test_empty_intersection_and_zero_scale_fail_explicitly():
  env, term = _wrapped_action(wrapper=0.1)
  term._clip[:] = torch.tensor([5.0, 6.0])
  with pytest.raises(ValueError, match="empty intersection"):
    normalized_action_limits(env, 29)
  term._scale.zero_()
  with pytest.raises(ValueError, match="nonzero"):
    normalized_action_limits(env, 29)


def test_adapter_without_manager_can_expose_wrapper_only_or_unlimited_actions():
  lower, upper = normalized_action_limits(SimpleNamespace(clip_actions=1.5), 29)
  torch.testing.assert_close(lower, torch.full((29,), -1.5))
  torch.testing.assert_close(upper, torch.full((29,), 1.5))
  lower, upper = normalized_action_limits(SimpleNamespace(), 29)
  assert torch.isneginf(lower).all() and torch.isposinf(upper).all()


def test_single_actor_adapter_rejects_new_inputs_in_constructor():
  with pytest.raises(ValueError, match="MultiTaskWorldModelEnv"):
    MotionTrackingEnv(
      dynamics=None,
      representation=SimpleNamespace(),
      env=SimpleNamespace(),
      batch_size=2,
      replay_buffer=None,
      history_length=5,
      termination_softness=0.05,
      estimator_input_config=EITInputConfig(),
    )
