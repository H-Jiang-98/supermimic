"""Episode-ordered policy starts reuse the EKF prefix, including ring cuts."""

from unittest.mock import Mock

import pytest
import torch

from supermimic.envs.world_model_env import BeliefRolloutStart
from tests.test_estimator_belief_integration import setup_env


def test_presampled_batches_preserve_rng_and_scan_replay_keys_once(monkeypatch):
  env, _, _ = setup_env()
  generator = torch.Generator().manual_seed(9)
  old_generator = generator.clone_state()
  eligible = env.replay_buffer.retained_frames([0, 2])
  selected = torch.cat(
    [torch.randint(len(eligible), (3,), generator=old_generator) for _ in range(4)]
  )
  gather = Mock(wraps=env.replay_buffer.retained_frames)
  monkeypatch.setattr(env.replay_buffer, "retained_frames", gather)
  keys = env.sample_policy_keys(3, [0, 2], generator=generator, num_batches=4)
  gather.assert_called_once_with(
    episode_ids=[0, 2], fields=("episode_id", "time_index")
  )
  assert torch.equal(keys[0], eligible["episode_id"][selected])
  assert torch.equal(keys[1], eligible["time_index"][selected])
  assert torch.equal(generator.get_state(), old_generator.get_state())


def test_streamed_policy_starts_match_independent_rebuilds_after_wrap(monkeypatch):
  env, _, _ = setup_env()
  replay = env.replay_buffer
  frame = replay._storage[:, 0].clone()
  for t in range(1, 6):
    frame["time_index"].fill_(t)
    frame["actor_obs"][:, 6] += 0.01
    replay.insert(frame)
  replay.finish_collection_round()

  # Physical times 2..5 survive. Query gaps, duplicates and two episodes.
  episodes = torch.tensor([1, 0, 0, 1, 0, 0])
  times = torch.tensor([4, 5, 3, 2, 3, 2])
  estimator = env.posterior_provider.estimator
  with monkeypatch.context() as patch:
    prefix = Mock(wraps=replay.training_prefix)
    predict = Mock(wraps=estimator.predict)
    correct = Mock(wraps=estimator.correct)
    patch.setattr(replay, "training_prefix", prefix)
    patch.setattr(estimator, "predict", predict)
    patch.setattr(estimator, "correct", correct)
    batches = []
    for start in env.iter_policy_starts(episodes, times, batch_size=2):
      assert torch.is_grad_enabled()
      assert not start.belief.covariance.requires_grad
      assert len(start) <= 2
      batches.append(start)
    assert prefix.call_count == 2
    assert predict.call_count == 5
    assert correct.call_count == 7
  starts = BeliefRolloutStart.cat(batches)
  assert starts.batch["episode_id"].tolist() == [0, 0, 0, 1, 1]
  assert starts.batch["time_index"].tolist() == [2, 3, 5, 2, 4]
  assert env.posterior_provider.state is None

  for i in range(len(starts)):
    record = starts.batch[i]
    prior, frames = replay.training_prefix(
      int(record["episode_id"]), int(record["time_index"])
    )
    with torch.no_grad():
      belief = estimator.correct(prior, frames["actor_obs"][0, :64])
      for k in range(1, len(frames)):
        belief = estimator.predict(
          belief, frames["action"][k - 1], frames["sim_ctx"][k - 1]
        )
        belief = estimator.correct(belief, frames["actor_obs"][k, :64])
    torch.testing.assert_close(starts.belief.mean.to_full()[i], belief.mean.to_full())
    torch.testing.assert_close(starts.belief.covariance[i], belief.covariance)
    history = (
      torch.arange(env.history_length) + len(frames) - env.history_length
    ).clamp_min(0)
    assert torch.equal(record["sensor_window"], frames["actor_obs"][history])
    assert torch.equal(record["state"], frames["state"][-1])

  ordered = env.prepare_policy_starts(episodes, times)
  assert torch.equal(ordered.batch["episode_id"], episodes)
  assert torch.equal(ordered.batch["time_index"], times)
  torch.testing.assert_close(
    ordered.belief.covariance, starts.belief.covariance[[4, 2, 1, 3, 1, 0]]
  )
  with pytest.raises(ValueError, match="unavailable"):
    list(
      env.iter_policy_starts(torch.tensor([0, 0]), torch.tensor([1, 4]), batch_size=1)
    )
