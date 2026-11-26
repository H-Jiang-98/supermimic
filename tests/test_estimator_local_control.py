from dataclasses import replace
from types import SimpleNamespace

import torch

from supermimic.belief import GaussianStatePrior
from supermimic.estimator_inputs import EITInputConfig
from supermimic.local_control import (
  OFFSETS,
  local_covariance_features,
  local_state,
  motion_reference,
  projected_factor,
)
from supermimic.models import LocalMotionActor
from supermimic.tasks.task_spec import make_estimator_motion_task
from supermimic.utils.helpers import matrix_to_quat
from supermimic.utils.lie_state import boxplus, exp, local_error, so3_exp


def test_local_covariance_jacobian_and_independent_prior():
  torch.manual_seed(47)
  state = exp(torch.randn(70, dtype=torch.float64))
  zero = state.velocity.new_zeros(70)
  jac = torch.func.jacrev(lambda e: local_state(boxplus(state, e)))(zero)
  factor = torch.randn(70, 70, dtype=zero.dtype).tril()
  projected = jac @ factor
  torch.testing.assert_close(projected_factor(state, factor), projected)
  rows, cols = torch.tril_indices(68, 68)
  torch.testing.assert_close(
    local_covariance_features(state, factor), (projected @ projected.T)[rows, cols]
  )
  truth = state.to_full().clone()
  prior = GaussianStatePrior()

  def draw(factory):
    return factory.sample(state, generator=torch.Generator().manual_seed(25))

  first = draw(prior)
  torch.testing.assert_close(draw(prior).mean.to_full(), first.mean.to_full())
  wider = draw(GaussianStatePrior(tuple(2 * s for s in prior.std)))
  torch.testing.assert_close(wider.covariance_sqrt, 2 * first.covariance_sqrt)
  torch.testing.assert_close(
    local_error(state, wider.mean), 2 * local_error(state, first.mean)
  )
  torch.testing.assert_close(state.to_full(), truth)


def test_reference_heading_invariance_offsets_and_clip_end():
  torch.manual_seed(39)
  state = exp(torch.randn(60, 70, dtype=torch.double) * 0.4)

  def motion(s):
    return SimpleNamespace(
      root_pos_w=s.position,
      root_quat_w=matrix_to_quat(s.rotation),
      root_lin_vel_w=s.world_velocity,
      root_ang_vel_w=(s.rotation @ s.omega.unsqueeze(-1)).squeeze(-1),
      joint_pos=s.joint_pos,
      joint_vel=s.joint_vel,
    )

  t, last = torch.tensor([1, 49, 50]), torch.tensor([49, 49, 59])
  reference = motion_reference(motion(state), t, last)
  idx = torch.minimum(t[:, None] + torch.tensor(OFFSETS), last[:, None])
  frames = reference.goal.reshape(3, 10, 73)
  torch.testing.assert_close(frames[..., 9:12], state.velocity[idx])
  torch.testing.assert_close(frames[..., 12:15], state.omega[idx])
  torch.testing.assert_close(frames[..., 15:44], state.joint_pos[idx])
  torch.testing.assert_close(reference.next_joint, state.joint_pos[idx[:, 0]])
  torch.testing.assert_close(reference.y, local_state(state.index(t)))
  rotation = so3_exp(t.new_tensor([0, 0, 1]).double() * 1.3)
  transformed = replace(
    state,
    rotation=rotation @ state.rotation,
    position=(rotation @ state.position.unsqueeze(-1)).squeeze(-1)
    + t.new_tensor([4, -2, 0]),
  )
  actual = motion_reference(motion(transformed), t, last)
  torch.testing.assert_close(actual.goal, reference.goal)
  torch.testing.assert_close(local_state(transformed), local_state(state))


def test_residual_pd_and_actor_covariance_permissions():
  torch.manual_seed(31)
  q0 = torch.linspace(-0.6, 0.3, 29)
  scale = torch.linspace(0.1, 1.7, 29)
  args = dict(
    obs_dim=93,
    action_dim=29,
    task_specs=[make_estimator_motion_task(730)],
    slot_mix={"motion": 1.0},
    latent_dim=32,
    hidden_dims=(32, 16),
    latent_bc_weight=0.0,
    history_length=5,
    context_dim=33,
    action_scale=scale,
    action_offset=q0,
  )
  mean = LocalMotionActor(
    inputs=EITInputConfig(group="mean", residual_actor=True), **args
  )
  belief = LocalMotionActor(
    inputs=EITInputConfig(group="belief", residual_actor=True), **args
  )
  goal, history = torch.randn(2, 730), torch.randn(2, 5, 93)
  condition = torch.randn(2, 68 * 2 + 33 + 2346 + 29)
  other = condition.clone()
  other[:, 169:-29] += 1
  with torch.no_grad():
    mean.film[-1].weight.normal_(std=0.1)
    belief.load_state_dict(mean.state_dict())

    def act(actor, x):
      return actor.decode_goal(actor.encode_goal(goal), history, x)[0]

    torch.testing.assert_close(act(mean, condition), act(mean, other), rtol=0, atol=0)
    assert not torch.allclose(act(belief, condition), act(belief, other))
    mean.decoder[-1].weight.zero_()
    mean.decoder[-1].bias.zero_()
    action = act(mean, condition)
    torch.testing.assert_close(action * scale + q0, condition[:, -29:])
    torch.testing.assert_close(
      action.clamp(-1, 1), ((condition[:, -29:] - q0) / scale).clamp(-1, 1)
    )
