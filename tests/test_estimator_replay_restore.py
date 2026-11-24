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
      other = BeliefProvider(real.estimator, draw_prior)
      other(**frame)
      real.load_state_dict(other.state_dict())
      assert not torch.equal(
        real.state.belief.mean.position, source.state.belief.mean.position
      )
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


@pytest.mark.parametrize(
  ("fault", "message"),
  [
    ("prev_action", "Executed action"),
    ("actor_obs", "Recorded sensor action"),
    ("sim_ctx", "simulation context"),
  ],
)
def test_insert_rejects_inconsistent_replay_across_wrap(fault, message):
  env, _, _ = setup_env("B")
  replay = env.replay_buffer
  frame = replay._storage[:, 0].clone()
  for t in (1, 2, 3):
    frame["time_index"].fill_(t)
    replay.insert(frame)
  replay.finish_collection_round()
  heads = replay._training_heads

  # Frame four overwrites physical slot zero, directly after slot three.
  frame["time_index"].fill_(4)
  if fault == "prev_action":
    frame["prev_action"][1, 0] += 0.5
    frame["actor_obs"][1, 64:] = frame["prev_action"][1]
  elif fault == "actor_obs":
    frame["actor_obs"][1, 64] += 0.5
  else:
    frame["sim_ctx"][1, 0] += 0.5
  before = replay._storage.clone()
  count, step = replay.insert_count, replay._step
  rng = torch.random.get_rng_state()
  with pytest.raises(ValueError, match=message):
    replay.insert(frame)
  torch.testing.assert_close(replay._storage, before)
  assert (replay.insert_count, replay._step) == (count, step)
  assert replay._training_heads is heads
  assert torch.equal(torch.random.get_rng_state(), rng)


@pytest.mark.parametrize("boundary", ["new_episode", "gap", "terminated", "truncated"])
def test_sealing_compares_actions_and_context_only_across_recorded_successors(boundary):
  env, _, _ = setup_env("B")
  replay = env.replay_buffer
  frame = replay._storage[:, 0].clone()
  frame["time_index"].fill_(1)
  if boundary in ("terminated", "truncated"):
    frame[boundary][0] = True
  replay.insert(frame)

  frame["time_index"].fill_(2)
  if boundary == "new_episode":
    frame["episode_id"][0] += BATCH
    frame["time_index"][0] = 0
  elif boundary == "gap":
    frame["time_index"][0] += 1
  frame["prev_action"][0, 0] += 0.5
  frame["actor_obs"][0, 64:] = frame["prev_action"][0]
  frame["sim_ctx"][0, 0] += 0.5
  replay.insert(frame)
  starts = replay.model_window_starts([0, BATCH], 2)
  assert starts.tolist() == [[0, 0]]
