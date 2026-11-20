"""Real collector checkpoint seeds and episode-head reconstruction."""

from unittest.mock import Mock

import pytest
import torch

from supermimic.belief_provider import BeliefProvider
from supermimic.replay_buffer import ReplayBuffer
from supermimic.runners.supermimic_runner import InputStagePolicy, build_actor
from tests.test_estimator_belief_integration import BATCH, setup_env
from tests.test_estimator_runner import _PhysicalEnv, _runner


def test_restore_real_heads_before_collection_and_after_partial_reset():
  runner = _runner("B")
  stochastic, _, template = setup_env("B")
  runner._dynamics = runner.wm_env.dynamics = runner._wm.dynamics = stochastic.dynamics
  runner.cfg.process_noise = stochastic.dynamics.process_noise_config
  real, imagined = runner.configure_belief_filter(
    Mock(side_effect=AssertionError("Restored frames must not redraw a prior"))
  )
  runner.env = _PhysicalEnv(runner)
  runner.action_manager = runner.env.action_manager
  runner.replay_buffer = runner.wm_env.replay_buffer = ReplayBuffer(
    runner._buffer_schema(), 8, BATCH
  )
  runner.logger = Mock()
  runner._sampler = None
  runner._pending_norm_updates = []
  runner.cfg.policy.noise = 0
  runner.policy_learner = InputStagePolicy(
    build_actor(runner.cfg, runner._wm, "cpu"), runner.cfg.policy
  )
  runner.reset_collection()

  def draw_prior(*, measurement, context, env_ids):
    prior = template.index(env_ids).clone()
    prior.mean.position.add_(torch.rand_like(prior.mean.position))
    return prior

  source = BeliefProvider(real.estimator, draw_prior)
  for step, expected_head_rows in enumerate(([0, 1, 2], [0])):
    sensor = runner._sim_history[:, -1]
    frame = dict(
      measurement=sensor[:, :64],
      previous_action=sensor[:, 64:],
      reset_mask=runner._measurement_reset,
      episode_id=runner._episode_id,
      time_index=runner._time_index,
      context=runner._sim_context.read(runner.env.unwrapped),
    )
    source(**frame)
    snapshot = source.state_dict()
    if step == 0:
      # A v2 snapshot can restore the corrected belief, but cannot supply its
      # realized prior. Even the same keys can belong to a different prior draw.
      other = BeliefProvider(real.estimator, draw_prior)
      other(**frame)
      real.load_state_dict(other.state_dict())
      assert not torch.equal(
        real.state.belief.mean.position, source.state.belief.mean.position
      )
      old = dict(snapshot, version="right_local_provider_v2")
      old.pop("head_priors")
      real.load_state_dict(old)
      assert real.state.head_priors is None
    else:
      bad_heads = dict(
        snapshot["head_priors"], episode_id=snapshot["episode_id"][:1] + BATCH
      )
      before = real.state
      with pytest.raises(ValueError, match="frame-zero episodes"):
        real.load_state_dict(dict(snapshot, head_priors=bad_heads))
      assert real.state is before
      real.load_state_dict(snapshot)
    # The actor looks up this already corrected frame; no new candidate is made.
    runner.step()
    rows = torch.tensor(expected_head_rows)
    episodes, times = snapshot["episode_id"], snapshot["time_index"]
    assert torch.equal(snapshot["head_priors"]["episode_id"], episodes[rows])
    for i, row in enumerate(expected_head_rows):
      if step == 0:
        with pytest.raises(ValueError, match="head or initial prior unavailable"):
          runner.replay_buffer.episode_prefix(episodes[row].item(), 0)
        continue
      recorded, _ = runner.replay_buffer.episode_prefix(episodes[row].item(), 0)
      torch.testing.assert_close(
        recorded.mean.to_full(), snapshot["head_priors"]["mean"][i]
      )
    runner.wm_env.reset(episode_id=episodes, time_index=times)
    for row in expected_head_rows:
      prior, frames = runner.replay_buffer.training_prefix(episodes[row].item(), 0)
      expected = real.estimator.correct(prior, frames["actor_obs"][0, :64])
      torch.testing.assert_close(
        imagined.state.belief.mean.to_full()[row], expected.mean.to_full()
      )
      torch.testing.assert_close(
        imagined.state.belief.covariance[row], expected.covariance
      )
  assert len(runner.env.actions) == 2


def test_invalid_head_metadata_leaves_replay_unchanged():
  env, _, prior = setup_env("B")
  buffer = env.replay_buffer
  frame = buffer._storage[:, 0].clone()
  frame["episode_id"] += BATCH
  for heads in (
    None,
    (frame["episode_id"].flip((0,)), prior),
    (frame["episode_id"], prior.index([0])),
  ):
    with pytest.raises(ValueError, match="head"):
      buffer.insert(frame, head_priors=heads, require_head_priors=True)
  assert buffer.num_transitions == 1
  retained, _ = buffer.episode_prefix(0, 0)
  torch.testing.assert_close(retained.covariance, prior.covariance[0])
  buffer.insert(frame, require_head_priors=False)
  assert buffer.num_transitions == 2
  with pytest.raises(ValueError, match="head or initial prior unavailable"):
    buffer.episode_prefix(BATCH, 0)
