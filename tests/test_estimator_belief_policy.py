"""SVG gradients, termination indexing, global accumulation and atomic restore."""

from copy import deepcopy
from dataclasses import dataclass, replace

import pytest
import torch
from torch import nn

from supermimic.learners.belief_policy_learner import (
  BeliefActorCriticPolicyLearner,
  BeliefPolicyLearnerCfg,
)


@dataclass
class Start:
  x: torch.Tensor

  def __len__(self):
    return len(self.x)

  def index(self, selected):
    return Start(self.x[selected])

  def repeat_interleave(self, repeats):
    return Start(self.x.repeat_interleave(repeats, 0))


class Dynamics(nn.Linear):
  @property
  def model_version(self):
    return "toy-fixed-model"


class Actor(nn.Module):
  def __init__(self):
    super().__init__()
    self.linear = nn.Linear(2, 1)

  def forward(self, obs):
    return self.linear(obs["x"])


class Env:
  max_step_reward = 2.0

  def __init__(self):
    self.dynamics = Dynamics(1, 1, bias=False).double()
    self.representation = nn.Identity()
    self.steps_taken = 0
    self.clear_rollout()

  def reset_prepared(self, start):
    self.state = start.x.clone()
    self.mean = start.x.clone() + 0.03
    self.covariance = torch.full_like(start.x, 0.02)
    self.previous = torch.zeros_like(start.x)
    return self.obs()

  def obs(self):
    return {
      "x": torch.cat((self.mean, self.previous), -1),
      "critic": torch.cat((self.state, self.mean, self.covariance, self.previous), -1),
    }

  def limit_actions(self, actions):
    return actions.clamp(-5, 5)

  def step(self, actions, *, process_noise, measurement_noise):
    actions = self.limit_actions(actions)
    self.state = (
      self.state
      + self.dynamics(actions)
      + (0.1 + 0.01 * actions.square()) * process_noise[:, :1]
    )
    prediction = self.mean + 0.7 * actions
    variance = 0.9 * self.covariance + 0.03 + 0.005 * actions.square()
    measurement = self.state + 0.05 * measurement_noise[:, :1]
    gain = variance / (variance + 0.1)
    self.mean = prediction + gain * (measurement - prediction)
    self.covariance = (1 - gain) * variance
    reward = (
      1
      - 0.05 * (self.state - 0.4).square()
      - 0.01 * actions.square()
      - 0.02 * (actions - self.previous).square()
    )
    done = torch.sigmoid(self.state - 2.0)
    self.previous = actions
    self.steps_taken += len(actions)
    return self.obs(), reward.squeeze(-1), done.squeeze(-1), {}

  def clear_rollout(self):
    self.state = self.mean = self.covariance = self.previous = None


def setup(microbatch=2):
  torch.manual_seed(41)
  actor = Actor().double()
  critic = nn.Linear(4, 1).double()
  env = Env()
  cfg = BeliefPolicyLearnerCfg(
    batch_size=3, num_samples=2, microbatch_size=microbatch, horizon=3
  )
  learner = BeliefActorCriticPolicyLearner(actor, critic, env, cfg)
  starts = Start(torch.tensor([[0.1], [0.2], [0.3]], dtype=torch.float64))
  noises = learner.draw_noises(starts)
  return learner, starts, noises


@pytest.mark.parametrize(
  "field,value",
  [
    ("gamma", 1.0),
    ("lam", -1.0),
    ("num_samples", 0),
    ("noise", float("nan")),
    ("microbatch_size", True),
  ],
)
def test_configuration_rejects_invalid_values(field, value):
  with pytest.raises(ValueError):
    BeliefPolicyLearnerCfg(**{field: value})


def test_microbatch_accumulation_matches_full_gradients_and_updates():
  small, starts, noises = setup(4)
  full, _, _ = setup(12)
  model_before = deepcopy(small.wm_env.dynamics.state_dict())
  stats_small = small.update_from(starts, *noises)
  stats_full = full.update_from(starts, *noises)
  for name in stats_small:
    torch.testing.assert_close(
      stats_small[name], stats_full[name], rtol=1e-10, atol=1e-12
    )
  for module_a, module_b in ((small.actor, full.actor), (small.critic, full.critic)):
    for a, b in zip(module_a.parameters(), module_b.parameters(), strict=True):
      torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)
      torch.testing.assert_close(a.grad, b.grad, rtol=1e-10, atol=1e-12)
  assert small.wm_env.steps_taken == 2 * 3 * 3
  assert stats_small["Policy/valid_steps"] == 18
  assert small.wm_env.state is None
  assert small.actor_steps == 1 and small.critic_steps == 20
  for key, before in model_before.items():
    assert torch.equal(before, small.wm_env.dynamics.state_dict()[key])
  assert all(p.grad is None for p in small.wm_env.dynamics.parameters())


@pytest.mark.parametrize(
  "horizon,first_done,expected_bootstrap,expected_return,expected_targets",
  [
    (1, 0.25, 1.5, 3.5, [3.5]),
    (2, 0.25, 0.375, 3.875, [3.78125, 5.0]),
    (2, 1.0, 0.0, 2.0, [2.0, 5.0]),
  ],
)
def test_actor_endpoint_and_gae_match_fixed_horizon_reference(
  horizon,
  first_done,
  expected_bootstrap,
  expected_return,
  expected_targets,
):
  from supermimic.learners.policy_learner import (
    ActorCriticPolicyLearner,
    ActorCriticPolicyLearnerCfg,
  )

  learner, starts, noises = setup()
  learner.cfg = replace(
    learner.cfg, gamma=0.5, lam=0.75, horizon=horizon, value_clipping=False
  )
  baseline = ActorCriticPolicyLearner(
    learner.actor,
    learner.critic,
    learner.wm_env,
    ActorCriticPolicyLearnerCfg(gamma=0.5, horizon=horizon),
    cuda_graphs=False,
  )
  with torch.no_grad():
    learner.critic.weight.zero_()
    learner.critic.bias.fill_(4.0)
  env = learner.wm_env
  transitions = iter([(2.0, first_done), (4.0, 0.5)])

  def step(actions, **kwargs):
    reward, done = next(transitions)
    return env.obs(), actions.new_tensor([reward]), actions.new_tensor([done]), {}

  env.step = step
  returns, (_, targets), metrics = learner._rollout(
    starts.index(slice(0, 1)), *(noise[:, 0, :1] for noise in noises), horizon
  )
  torch.testing.assert_close(targets[:, 0], targets.new_tensor(expected_targets))
  assert returns.item() == pytest.approx(expected_return)
  assert metrics["bootstrap"].item() == pytest.approx(expected_bootstrap)
  endpoint = baseline._bootstrap(
    env.obs(),
    metrics["final_survival"],
    returns.new_tensor([first_done if horizon == 1 else 0.5]),
  )
  torch.testing.assert_close(endpoint, metrics["bootstrap"].reshape(1))


def test_fixed_noise_actor_gradient_includes_filter_bootstrap_and_survival():
  learner, starts, noises = setup()
  starts = starts.index(torch.tensor([2]))
  process, measurement = (noise[:, 2, :1] for noise in noises)
  learner.critic.requires_grad_(False)
  learner.wm_env.dynamics.requires_grad_(False)
  parameter = learner.actor.linear.weight
  returns, _, _ = learner._rollout(starts, process, measurement, 3)
  gradient = torch.autograd.grad(returns.sum(), parameter)[0]
  direction = torch.tensor([[0.4, -0.7]], dtype=torch.float64)
  original = parameter.detach().clone()
  epsilon = 1e-5
  values = []
  for sign in (1, -1):
    with torch.no_grad():
      parameter.copy_(original + sign * epsilon * direction)
    value, _, _ = learner._rollout(starts, process, measurement, 3)
    values.append(value.detach().sum())
  with torch.no_grad():
    parameter.copy_(original)
  torch.testing.assert_close(
    (values[0] - values[1]) / (2 * epsilon),
    (gradient * direction).sum(),
    rtol=1e-7,
    atol=1e-9,
  )
  assert gradient.norm() > 1e-4


def test_optimizer_restore_reproduces_next_fixed_batch_update():
  learner, starts, noises = setup()
  learner.update_from(starts, *noises)
  checkpoint = learner.state_dict()
  actor, critic = (
    deepcopy(learner.actor.state_dict()),
    deepcopy(learner.critic.state_dict()),
  )
  learner.update_from(starts, *noises)
  expected_actor, expected_critic = (
    deepcopy(learner.actor.state_dict()),
    deepcopy(learner.critic.state_dict()),
  )
  expected_optimizer = learner.state_dict()
  learner.actor.load_state_dict(actor)
  learner.critic.load_state_dict(critic)
  learner.load_state_dict(checkpoint)
  learner.update_from(starts, *noises)
  for actual, expected in (
    (learner.actor.state_dict(), expected_actor),
    (learner.critic.state_dict(), expected_critic),
  ):
    for name in actual:
      assert torch.equal(actual[name], expected[name])
  actual = learner.state_dict()
  for name in ("actor_optimizers", "critic_optimizers"):
    for index, value in actual[name][0]["state"].items():
      for key in value:
        assert torch.equal(value[key], expected_optimizer[name][0]["state"][index][key])


def test_rng_streams_restore_independently_of_global_rng():
  learner, starts, _ = setup()
  checkpoint = learner.state_dict()
  expected = learner.draw_noises(starts)
  torch.manual_seed(999)
  learner.load_state_dict(checkpoint)
  for actual, target in zip(learner.draw_noises(starts), expected, strict=True):
    assert torch.equal(actual, target)


@pytest.mark.parametrize(
  "malformation", ["critic_nan", "actor_shape", "rng", "group", "counter"]
)
def test_restore_preflight_is_atomic(malformation):
  learner, starts, noises = setup()
  learner.update_from(starts, *noises)
  checkpoint = learner.state_dict()
  corrupted = deepcopy(checkpoint)
  if malformation == "critic_nan":
    corrupted["critic_optimizers"][0]["state"][0]["exp_avg"].fill_(float("nan"))
  elif malformation == "actor_shape":
    corrupted["actor_optimizers"][0]["state"][0]["exp_avg"] = torch.zeros(1)
  elif malformation == "rng":
    corrupted["rng"]["measurement"] = torch.zeros(7, dtype=torch.uint8)
  elif malformation == "group":
    corrupted["critic_optimizers"][0]["param_groups"][0]["lr"] = 3
  else:
    corrupted["critic_steps"] = -1
  with pytest.raises((ValueError, RuntimeError)):
    learner.load_state_dict(corrupted)
  actual = learner.state_dict()
  assert actual["actor_steps"] == checkpoint["actor_steps"]
  for name in ("actor_optimizers", "critic_optimizers"):
    for index, value in actual[name][0]["state"].items():
      for key in value:
        assert torch.equal(value[key], checkpoint[name][0]["state"][index][key])
  for name in actual["rng"]:
    assert torch.equal(actual["rng"][name], checkpoint["rng"][name])


def test_invalid_noise_rejected_before_any_optimizer_change():
  learner, starts, noises = setup()
  original = deepcopy(learner.actor.state_dict())
  noises[0][0, 0, 0, 0] = float("nan")
  with pytest.raises(ValueError, match="process noise"):
    learner.update_from(starts, *noises)
  assert learner.steps == 0
  for name, value in original.items():
    assert torch.equal(value, learner.actor.state_dict()[name])


def real_setup(group="C"):
  from supermimic.critic_inputs import BeliefCritic
  from supermimic.models import MultiEncoderAE
  from tests.test_estimator_belief_integration import setup_env

  env, _, _ = setup_env(group)
  actor = MultiEncoderAE(
    93,
    29,
    env.task_specs,
    {"motion": 1.0},
    8,
    (16, 8),
    0.0,
    5,
    uncertainty_dim=450 if group == "C" else 0,
    film_hidden_dims=(12, 8),
  )
  critic = BeliefCritic(group, 1, hidden_dims=(16, 8))
  learner = BeliefActorCriticPolicyLearner(
    actor,
    critic,
    env,
    BeliefPolicyLearnerCfg(
      batch_size=3, num_samples=2, horizon=2, microbatch_size=2, value_clipping=False
    ),
  )
  return learner


@pytest.mark.parametrize("group", ["A", "B", "C"])
def test_real_belief_policy_two_updates_freeze_model_and_normalizers(group):
  learner = real_setup(group)
  before_model = deepcopy(learner.wm_env.dynamics.state_dict())
  before_actor = deepcopy(learner.actor.state_dict())
  before_normalizers = {
    name: value.clone() for name, value in learner.actor.named_buffers()
  }
  for _ in range(2):
    log = learner.update(episode_ids=[0, 1, 2])
    assert all(
      torch.isfinite(value).all()
      for name, value in log.items()
      if name != "Policy/explained_variance"
    )
  assert learner.actor_steps == 2 and learner.critic_steps == 40
  assert any(
    not torch.equal(value, learner.actor.state_dict()[name])
    for name, value in before_actor.items()
  )
  assert all(
    torch.equal(value, learner.wm_env.dynamics.state_dict()[name])
    for name, value in before_model.items()
  )
  assert all(
    torch.equal(value, dict(learner.actor.named_buffers())[name])
    for name, value in before_normalizers.items()
  )
  assert learner.wm_env.posterior_provider.state is None
  assert all(p.grad is None for p in learner.wm_env.dynamics.parameters())


def test_real_covariance_film_and_bootstrap_fixed_noise_directional_derivative():
  learner = real_setup()
  env = learner.wm_env
  env.dynamics.requires_grad_(False)
  learner.critic.requires_grad_(False)
  with torch.no_grad():
    learner.actor.film[-1].weight.normal_(std=0.01)
    # Bootstrap must exercise its input derivative, beyond zero output init.
    list(learner.critic.parameters())[-2].normal_(std=0.01)
  starts = env.prepare_policy_starts(torch.tensor([0]), torch.tensor([0]))
  process, measurement = (noise[:, :, 0] for noise in learner.draw_noises(starts))
  parameter = learner.actor.decoder[-1].bias
  returns, _, _ = learner._rollout(starts, process, measurement, 2)
  derivative = torch.autograd.grad(returns.sum(), parameter)[0]
  original = parameter.detach().clone()
  direction = torch.randn_like(parameter)
  direction /= direction.norm()
  epsilon = 0.002
  values = []
  for sign in (1, -1):
    with torch.no_grad():
      parameter.copy_(original + sign * epsilon * direction)
    value, _, _ = learner._rollout(starts, process, measurement, 2)
    values.append(value.detach().sum())
  with torch.no_grad():
    parameter.copy_(original)
  torch.testing.assert_close(
    (values[0] - values[1]) / (2 * epsilon),
    derivative @ direction,
    rtol=0.02,
    atol=3e-4,
  )
  assert derivative.norm() > 1e-6
  env.clear_rollout()


def test_critic_uses_raw_values_for_gae_and_only_clips_final_targets():
  learner, starts, noises = setup()
  learner.cfg = replace(learner.cfg, lam=0.0)
  learner._v_max = 1.0
  with torch.no_grad():
    learner.critic.weight.zero_()
    learner.critic.bias.fill_(100.0)
  learner.critic.requires_grad_(False)
  original = learner.wm_env.step
  rewards, continuations = [], []

  def step(*args, **kwargs):
    obs, reward, done, extras = original(*args, **kwargs)
    reward = reward - 50.0
    rewards.append(reward.detach())
    continuations.append((1 - done).detach())
    return obs, reward, done, extras

  learner.wm_env.step = step
  start = starts.index(torch.tensor([2]))
  _, data, _ = learner._rollout(start, noises[0][:, 2, :1], noises[1][:, 2, :1], 2)
  expected = (
    torch.stack(rewards) + learner.cfg.gamma * torch.stack(continuations) * 100
  ).clamp(max=1)
  clipped_values_first = (
    torch.stack(rewards) + learner.cfg.gamma * torch.stack(continuations)
  ).clamp(max=1)
  torch.testing.assert_close(data[1], expected)
  assert not torch.allclose(data[1], clipped_values_first.flatten())


@pytest.mark.parametrize("microbatch", [1, 2, 7])
def test_critic_microbatch_is_independent_and_preserves_logical_updates(microbatch):
  full, starts, noises = setup(12)
  split, _, _ = setup(1)
  split.cfg = replace(split.cfg, critic_microbatch_size=microbatch)
  scored_sizes = []

  def record_scoring_batch(module, args):
    if not torch.is_grad_enabled() and next(module.parameters()).requires_grad:
      scored_sizes.append(len(args[0]))

  with split.critic.register_forward_pre_hook(record_scoring_batch):
    full_metrics = full.update_from(starts, *noises)
    split_metrics = split.update_from(starts, *noises)
  assert sum(scored_sizes) == 18 and max(scored_sizes) <= microbatch
  torch.testing.assert_close(
    split_metrics["Policy/explained_variance"],
    full_metrics["Policy/explained_variance"],
    rtol=1e-12,
    atol=1e-12,
  )
  for a, b in zip(full.critic.parameters(), split.critic.parameters(), strict=True):
    torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)
  assert full.critic_steps == split.critic_steps == 20


def test_muon_and_adam_policy_state_restore_and_clipping():
  torch.manual_seed(93)
  actor = Actor().double()
  actor.linear = nn.Sequential(
    nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 8), nn.Tanh(), nn.Linear(8, 1)
  ).double()
  critic = nn.Sequential(
    nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 8), nn.Tanh(), nn.Linear(8, 1)
  ).double()
  learner = BeliefActorCriticPolicyLearner(
    actor,
    critic,
    Env(),
    BeliefPolicyLearnerCfg(batch_size=3, num_samples=2, microbatch_size=1, horizon=3),
  )
  _, starts, noises = setup()
  assert isinstance(learner.actor_optimizers[1], torch.optim.Muon)
  assert learner.actor_optimizers[0].param_groups[0]["betas"] == (0.9, 0.95)
  assert learner.critic_optimizers[0].param_groups[0]["betas"] == (0.9, 0.999)
  learner.update_from(starts, *noises)
  assert all(p.grad.abs().max() <= 0.1 for p in actor.parameters())
  saved = learner.state_dict()
  weights = deepcopy((actor.state_dict(), critic.state_dict()))
  learner.update_from(starts, *noises)
  expected = deepcopy((actor.state_dict(), critic.state_dict()))
  actor.load_state_dict(weights[0])
  critic.load_state_dict(weights[1])
  learner.load_state_dict(saved)
  learner.update_from(starts, *noises)
  for actual, target in zip(
    (actor.state_dict(), critic.state_dict()), expected, strict=True
  ):
    for name in actual:
      torch.testing.assert_close(actual[name], target[name], rtol=0, atol=0)
  assert learner.actor_steps == 2 and learner.critic_steps == 40


@pytest.mark.parametrize("num_samples", [1, 2])
def test_process_rng_is_only_used_for_multiple_samples(num_samples):
  learner, starts, _ = setup()
  learner.cfg = replace(learner.cfg, num_samples=num_samples)
  process_before = learner.generators["process"].get_state().clone()
  measurement_before = learner.generators["measurement"].get_state().clone()
  process, measurement = learner.draw_noises(starts)
  assert process.shape == (3, 3, num_samples, 70)
  assert measurement.shape == (3, 3, num_samples, 64)
  assert torch.equal(process_before, learner.generators["process"].get_state()) == (
    num_samples == 1
  )
  assert bool(process.count_nonzero()) == (num_samples > 1)
  assert not torch.equal(
    measurement_before, learner.generators["measurement"].get_state()
  )


def test_single_sample_custom_rollout_ignores_supplied_process_noise():
  learner, starts, _ = setup()
  learner.cfg = replace(learner.cfg, num_samples=1)
  process, measurement = (x.flatten(1, 2) for x in learner.draw_noises(starts))
  expected = learner._rollout_result(starts, process, measurement, 3)
  actual = learner._rollout_result(starts, torch.ones_like(process), measurement, 3)
  torch.testing.assert_close(actual.returns, expected.returns)
  torch.testing.assert_close(actual.critic_data, expected.critic_data)
