"""Sequence boundaries, joint posterior learning, and exact optimizer continuation."""

import copy
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch
from tensordict import TensorDict

from supermimic.belief import GaussianStatePrior
from supermimic.estimator import LieBeliefFilter, posterior_nll
from supermimic.estimator_inputs import GaussianMeasurementConfig
from supermimic.learners.belief_wm_learner import (
  BeliefWorldModelLearner,
  BeliefWorldModelLearnerCfg,
)
from supermimic.models import Dynamics, ProcessNoiseConfig
from supermimic.replay_buffer import FieldSpec, ReplayBuffer
from supermimic.representations import GenCoordRepresentation
from supermimic.utils.lie_state import LieState


def fixture(
  *, device="cpu", dtype=torch.float64, buffer_size=12, episodes=(5, 4), seed=31
):
  torch.manual_seed(seed)
  representation = GenCoordRepresentation(
    29, torch.zeros(29, device=device, dtype=dtype)
  )
  representation.dt = 0.02
  representation.substep_coeff = 0.5
  dynamics = (
    Dynamics(
      representation,
      29,
      2,
      torch.ones(29, device=device, dtype=dtype),
      torch.zeros(29, device=device, dtype=dtype),
      (12,),
      ProcessNoiseConfig(hidden_dims=(8,)),
    )
    .to(device=device, dtype=dtype)
    .eval()
  )
  estimator = LieBeliefFilter(
    dynamics, GaussianMeasurementConfig(), torch.zeros(29, device=device, dtype=dtype)
  )
  schema = {
    "state": FieldSpec(71, dtype),
    "actor_obs": FieldSpec(93, dtype),
    "action": FieldSpec(29, dtype),
    "prev_action": FieldSpec(29, dtype),
    "sim_ctx": FieldSpec(2, dtype),
    "episode_id": FieldSpec(dtype=torch.long),
    "time_index": FieldSpec(dtype=torch.long),
    "terminated": FieldSpec(dtype=torch.bool),
    "truncated": FieldSpec(dtype=torch.bool),
  }
  replay = ReplayBuffer(
    schema, buffer_size, 1, device, initial_prior=GaussianStatePrior()
  )
  for episode, length in enumerate(episodes):
    previous = torch.zeros(1, 29, device=device, dtype=dtype)
    for index in range(length):
      state = torch.zeros(1, 71, device=device, dtype=dtype)
      state[:, 2] = 0.8
      state[:, 3] = 1
      state[:, 0] = index * 0.002
      state[:, 36] = 0.1
      state[:, 7:36] = 0.02 * torch.sin(
        torch.arange(29, device=device, dtype=dtype) + index * 0.01
      )
      action = torch.full_like(previous, index * 0.03 + 0.01)
      observation = estimator.observation(LieState.from_full(state))
      noise = torch.randn_like(
        observation
      ) * estimator.measurement_config.standard_deviations(
        29, device=device, dtype=dtype
      )
      frame = TensorDict(
        {
          "state": state,
          "actor_obs": torch.cat((observation + noise, previous), -1),
          "action": action,
          "prev_action": previous,
          "sim_ctx": torch.full((1, 2), episode * 0.1, device=device, dtype=dtype),
          "episode_id": torch.tensor([episode], device=device),
          "time_index": torch.tensor([index], device=device),
          "terminated": torch.tensor([index == length - 1], device=device),
          "truncated": torch.zeros(1, device=device, dtype=torch.bool),
        },
        [1],
      )
      replay.insert(frame)
      previous = action
  replay.finish_collection_round()
  return estimator, replay


def reference(estimator, sequence):
  with torch.no_grad():
    q = estimator.correct(sequence.prior, sequence.prefix["actor_obs"][0, :64])
    for t in range(len(sequence.prefix) - 1):
      q = estimator.predict(
        q, sequence.prefix["action"][t], sequence.prefix["sim_ctx"][t]
      )
      q = estimator.correct(q, sequence.prefix["actor_obs"][t + 1, :64])
  q = q.detach()
  losses = []
  for t in range(sequence.transitions):
    q = estimator.predict(
      q, sequence.frames["action"][t], sequence.frames["sim_ctx"][t]
    )
    q = estimator.correct(q, sequence.frames["actor_obs"][t + 1, :64])
    losses.append(
      posterior_nll(
        q, LieState.from_full(sequence.frames["state"][t + 1]), estimator.config
      )
    )
  return torch.stack(losses), q


def test_replay_window_has_t_plus_one_frames_and_no_terminal_successor():
  _, replay = fixture(episodes=(4, 1, 3))
  assert replay.training_episode_lengths() == {0: 4, 1: 1, 2: 3}
  sequence = replay.belief_sequence(0, 1, 8)
  assert sequence.transitions == 2
  assert sequence.prefix["time_index"].tolist() == [0, 1]
  assert sequence.frames["time_index"].tolist() == [1, 2, 3]
  assert sequence.frames["terminated"].tolist() == [False, False, True]
  assert torch.equal(sequence.frames["action"][:-1], sequence.frames["prev_action"][1:])
  samples = replay.sample_belief_sequences(8, 30)
  assert all(s.episode != 1 and s.transitions > 0 for s in samples)
  assert all(s.frames["episode_id"].unique().numel() == 1 for s in samples)
  with pytest.raises(ValueError, match="successor"):
    replay.belief_sequence(0, 3, 2)
  with pytest.raises(ValueError, match="successor"):
    replay.sample_belief_sequences(2, 1, episode_ids=[1])


def test_retained_segment_prior_reused_and_round_filter_excludes_validation():
  _, replay = fixture(episodes=(7, 3), buffer_size=7)
  assert replay.training_episode_lengths() == {0: 4, 1: 3}
  sequence = replay.belief_sequence(0, 4, 2)
  assert sequence.prefix["time_index"].tolist() == [3, 4]
  again = replay.belief_sequence(0, 4, 2)
  torch.testing.assert_close(
    sequence.prior.mean.to_full(), again.prior.mean.to_full(), atol=0, rtol=0
  )
  selected = replay.retained_frames([1], newest_count=4)
  assert selected["episode_id"].tolist() == [1, 1, 1]
  assert replay.retained_frames([0], newest_count=4)["time_index"].tolist() == [6]
  replay._storage["actor_obs"].add_(1)
  assert not torch.equal(
    sequence.frames["actor_obs"][0], replay._storage["actor_obs"][0, 4]
  )
  torch.testing.assert_close(
    sequence.frames["actor_obs"], again.frames["actor_obs"], atol=0, rtol=0
  )


def test_sampler_rejects_inconsistent_recorded_episode():
  _, replay = fixture(episodes=(4,))
  replay._storage["terminated"][0, 1] = True
  with pytest.raises(ValueError, match="boundary"):
    replay.sample_belief_sequences(2, 1)


def test_sampler_validates_each_sealed_fragment_once_until_overwrite():
  _, replay = fixture(episodes=(3, 3), buffer_size=6)
  with patch.object(replay, "training_prefix", wraps=replay.training_prefix) as prefix:
    lengths = replay.training_episode_lengths()
    assert prefix.call_count == 2
    lengths.clear()
    assert replay.training_episode_lengths() == {0: 3, 1: 3}
    replay.sample_belief_sequences(2, 3)
    # Subsequent samples only gather the three selected windows.
    assert prefix.call_count == 5
    frame = replay._storage[:, 3].clone()
    frame["episode_id"].fill_(2)
    replay.insert(frame)
    assert replay.training_episode_lengths() == {0: 2, 1: 3, 2: 1}
    assert prefix.call_count == 8


def test_boundary_and_ragged_nll_match_independent_recurrence_without_double_fusion():
  estimator, replay = fixture()
  learner = BeliefWorldModelLearner(estimator)
  sequences = [replay.belief_sequence(0, 1, 2), replay.belief_sequence(1, 2, 2)]
  with patch.object(estimator, "correct", wraps=estimator.correct) as correct:
    boundary = learner.boundary_belief(sequences[0])
    assert correct.call_count == len(sequences[0].prefix)
  assert not boundary.covariance.requires_grad
  assert not boundary.mean.velocity.requires_grad
  expected = torch.cat([reference(estimator, s)[0] for s in sequences]).mean()
  actual, diagnostics = learner.loss(sequences)
  torch.testing.assert_close(actual, expected, atol=2e-9, rtol=1e-11)
  torch.testing.assert_close(actual, diagnostics["Loss/posterior_nll"])
  assert learner.evaluate(sequences)["valid_transitions"] == 3


def test_two_heads_update_and_noise_hidden_receives_gradients_after_first_step():
  estimator, replay = fixture(episodes=(4,))
  learner = BeliefWorldModelLearner(estimator)
  sequences = [replay.belief_sequence(0, 0, 2)]
  mean_before = estimator.dynamics.mlp[-1].weight.detach().clone()
  noise_before = estimator.dynamics.noise_head[-1].weight.detach().clone()
  # Collection may freeze all parameters; the model learner enables them again.
  estimator.dynamics.requires_grad_(False)
  first = learner.update(sequences)
  assert first["Grad/mean_norm"] > 0 and first["Grad/noise_norm"] > 0
  assert not torch.equal(mean_before, estimator.dynamics.mlp[-1].weight)
  assert not torch.equal(noise_before, estimator.dynamics.noise_head[-1].weight)
  assert estimator.dynamics.noise_head[0].weight.grad.abs().sum() == 0
  learner.update(sequences)
  assert estimator.dynamics.noise_head[0].weight.grad.abs().sum() > 0
  assert learner.steps == 2


def test_prefix_rebuilt_with_current_parameters_and_no_retained_graph():
  estimator, replay = fixture(episodes=(5,))
  learner = BeliefWorldModelLearner(estimator)
  sequence = replay.belief_sequence(0, 2, 2)
  before = learner.boundary_belief(sequence)
  prior = sequence.prior.mean.to_full().clone()
  learner.update([sequence])
  after = learner.boundary_belief(sequence)
  assert not torch.equal(before.mean.to_full(), after.mean.to_full())
  torch.testing.assert_close(sequence.prior.mean.to_full(), prior, atol=0, rtol=0)
  learner.update([sequence])
  assert learner.steps == 2


@pytest.mark.parametrize("head", ["mlp", "noise_head"])
def test_two_step_posterior_nll_parameter_direction_matches_finite_difference(head):
  estimator, replay = fixture(episodes=(4,))
  learner = BeliefWorldModelLearner(estimator)
  # At segment frame zero the detached boundary is parameter-independent, so a
  # finite difference of the entire call measures precisely the training graph.
  sequence = replay.belief_sequence(0, 0, 2)
  parameter = getattr(estimator.dynamics, head)[-1].weight
  direction = torch.randn_like(parameter)
  direction /= direction.norm()
  loss, _ = learner.loss([sequence])
  derivative = (torch.autograd.grad(loss, parameter)[0] * direction).sum()
  values = []
  epsilon = 2e-6
  original = parameter.detach().clone()
  with torch.no_grad():
    for sign in (1, -1):
      parameter.copy_(original + sign * epsilon * direction)
      values.append(learner.loss([sequence])[0])
    parameter.copy_(original)
  numerical = (values[0] - values[1]) / (2 * epsilon)
  torch.testing.assert_close(derivative, numerical, rtol=2e-5, atol=2e-5)


def test_microbatch_accumulation_is_weighted_by_transition_count():
  estimator, replay = fixture()
  other = copy.deepcopy(estimator)
  first = BeliefWorldModelLearner(
    estimator, BeliefWorldModelLearnerCfg(microbatch_size=1)
  )
  second = BeliefWorldModelLearner(other, BeliefWorldModelLearnerCfg(microbatch_size=2))
  sequences = [replay.belief_sequence(0, 0, 2), replay.belief_sequence(1, 2, 2)]
  stats_a, stats_b = first.update(sequences), second.update(sequences)
  for name in stats_a:
    torch.testing.assert_close(stats_a[name], stats_b[name], atol=1e-8, rtol=1e-10)
  for left, right in zip(
    estimator.dynamics.parameters(), other.dynamics.parameters(), strict=True
  ):
    torch.testing.assert_close(left, right, atol=1e-12, rtol=1e-10)


def test_fixed_small_dataset_nll_decreases_and_normalization_is_fixed_inside_updates():
  estimator, replay = fixture(episodes=(5,))
  learner = BeliefWorldModelLearner(estimator)
  data = replay.retained_frames([0])
  version = estimator.dynamics.model_version
  learner.update_normalization(data["state"], data["action"])
  assert estimator.dynamics.model_version != version
  normalization = [
    copy.deepcopy(n.state_dict()) for n in estimator.dynamics.normalizers()
  ]
  sequences = [replay.belief_sequence(0, 0, 2), replay.belief_sequence(0, 1, 2)]
  start = learner.evaluate(sequences)["Loss/posterior_nll"]
  for _ in range(12):
    learner.update(sequences)
  end = learner.evaluate(sequences)["Loss/posterior_nll"]
  assert end < start - 1.0, (start, end)
  for module, saved in zip(
    estimator.dynamics.normalizers(), normalization, strict=True
  ):
    for name, value in module.state_dict().items():
      torch.testing.assert_close(value, saved[name], atol=0, rtol=0)


@pytest.mark.parametrize(
  ("invalid", "message"),
  [
    ("zero_quaternion", "quaternion"),
    ("action_feature_overflow", "input features"),
    ("action_variance_overflow", "running statistics"),
  ],
)
def test_invalid_normalization_update_preserves_both_statistics(invalid, message):
  estimator, replay = fixture(dtype=torch.float32, episodes=(3,))
  learner = BeliefWorldModelLearner(estimator)
  data = replay.retained_frames()
  states, actions = data["state"], data["action"]
  learner.update_normalization(states, actions)
  if invalid == "zero_quaternion":
    states[1, 3:7] = 0
  elif invalid == "action_feature_overflow":
    estimator.dynamics.action_scale[0] = 2
    actions[1, 0] = torch.finfo(actions.dtype).max
  else:
    actions[:2, 0] = torch.tensor([1e30, -1e30])
  model_version = estimator.dynamics.model_version
  with pytest.raises(ValueError, match=message):
    learner.update_normalization(states, actions)
  assert estimator.dynamics.model_version == model_version
  assert all(not n.training for n in estimator.dynamics.normalizers())


def test_restore_matches_next_update_and_invalid_optimizer_is_atomic():
  estimator, replay = fixture(episodes=(4,))
  learner = BeliefWorldModelLearner(estimator)
  sequences = [replay.belief_sequence(0, 0, 2)]
  learner.update(sequences)
  checkpoint = learner.state_dict()
  restored_estimator = copy.deepcopy(estimator)
  restored = BeliefWorldModelLearner(restored_estimator)
  restored.load_state_dict(checkpoint)
  before = copy.deepcopy(restored.optimizer.state_dict())
  malformed = copy.deepcopy(checkpoint)
  malformed["optimizer"]["state"][0]["exp_avg"].flatten()[0] = torch.nan
  with pytest.raises(ValueError, match="finite"):
    restored.load_state_dict(malformed)
  for index, values in before["state"].items():
    for key, value in values.items():
      torch.testing.assert_close(
        restored.optimizer.state_dict()["state"][index][key], value, atol=0, rtol=0
      )
  stats = learner.update(sequences)
  restored_stats = restored.update(sequences)
  for key in stats:
    torch.testing.assert_close(stats[key], restored_stats[key], atol=0, rtol=0)
  for first, second in zip(
    estimator.dynamics.parameters(), restored.dynamics.parameters(), strict=True
  ):
    torch.testing.assert_close(first, second, atol=0, rtol=0)
  assert learner.steps == restored.steps == 2


def test_reject_nonfinite_loss_or_gradient_before_step():
  estimator, replay = fixture(episodes=(3,))
  learner = BeliefWorldModelLearner(estimator)
  sequences = [replay.belief_sequence(0, 0, 1)]
  before = copy.deepcopy(estimator.dynamics.state_dict())
  handle = estimator.dynamics.mlp[-1].weight.register_hook(lambda g: g * torch.nan)
  with pytest.raises(FloatingPointError, match="gradient"):
    learner.update(sequences)
  handle.remove()
  assert learner.steps == 0
  for name, value in estimator.dynamics.state_dict().items():
    torch.testing.assert_close(value, before[name], atol=0, rtol=0)
  assert all(p.grad is None for p in estimator.dynamics.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_float32_joint_training_and_restore(tmp_path):
  estimator, replay = fixture(device="cuda", dtype=torch.float32, episodes=(4,))
  learner = BeliefWorldModelLearner(estimator)
  sequences = [replay.belief_sequence(0, 0, 2)]
  for _ in range(2):
    stats = learner.update(sequences)
    assert all(torch.isfinite(value) for value in stats.values())
  restored = BeliefWorldModelLearner(copy.deepcopy(estimator))
  path = tmp_path / "learner.pt"
  torch.save(learner.state_dict(), path)
  mapped = torch.load(path, map_location="cuda:0", weights_only=False)
  assert mapped["optimizer"]["state"][0]["step"].is_cuda
  restored.load_state_dict(mapped)
  assert all(
    values["step"].device.type == "cpu" for values in restored.optimizer.state.values()
  )
  first, second = learner.update(sequences), restored.update(sequences)
  for key in first:
    torch.testing.assert_close(first[key], second[key])
  for p, q in zip(
    learner.dynamics.parameters(), restored.dynamics.parameters(), strict=True
  ):
    torch.testing.assert_close(p, q)


@pytest.mark.parametrize(
  "value", [torch.tensor([1.0]), torch.tensor(1), torch.tensor(True)]
)
def test_adam_checkpoint_requires_floating_scalar_clock(value):
  estimator, replay = fixture(episodes=(3,))
  learner = BeliefWorldModelLearner(estimator)
  learner.update([replay.belief_sequence(0, 0, 1)])
  checkpoint = learner.state_dict()
  checkpoint["optimizer"]["state"][0]["step"] = value
  with pytest.raises(ValueError, match="Adam step"):
    learner.load_state_dict(checkpoint)


def test_sequence_rejects_prefix_overlap_or_action_mismatch():
  estimator, replay = fixture(episodes=(4,))
  learner = BeliefWorldModelLearner(estimator)
  sequence = replay.belief_sequence(0, 1, 2)
  broken = sequence.frames.clone()
  broken["state"][0, 0] += 1
  with pytest.raises(ValueError, match="exact first"):
    learner.update([replace(sequence, frames=broken)])
  broken = sequence.frames.clone()
  broken["action"][0, 0] += 1
  with pytest.raises(ValueError, match="misaligned"):
    learner.update([replace(sequence, frames=broken)])
