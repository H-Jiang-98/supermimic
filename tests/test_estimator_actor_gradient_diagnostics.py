"""Independent decomposition matches the actual actor update and preserves state."""

from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from supermimic.actor_gradient_diagnostics import actor_objective_gradients
from supermimic.scripts.check_eit_training import equal_tree
from tests.test_estimator_belief_policy import setup


def test_inference_device_view_moves_calibration_without_mutating_or_copying_optimizers():
  from supermimic.imagined_diagnostics import _inference_device_view

  calibration = torch.arange(29, dtype=torch.float32)
  dynamics = {"state_dict": {"weights": torch.ones(2)}}
  actor = {"actor": {"weights": torch.ones(3)}}
  optimizers = {"exp_avg": torch.ones(5)}
  saved = {
    "world_model": {
      "estimator": {"default_joint_pos": calibration},
      "dynamics": dynamics,
    },
    "actor": actor,
    "policy": optimizers,
  }
  # The meta device exercises the device transfer without requiring a GPU test host.
  view = _inference_device_view(saved, "meta")
  assert view["world_model"]["estimator"]["default_joint_pos"].device.type == "meta"
  assert saved["world_model"]["estimator"]["default_joint_pos"] is calibration
  assert calibration.device.type == "cpu"
  assert view["world_model"]["dynamics"] is dynamics
  assert view["actor"] is actor
  assert view["policy"] is optimizers
  assert optimizers["exp_avg"].device.type == "cpu"


def test_gradient_components_sum_to_training_gradient_and_microbatches_agree():
  learner, starts, noises = setup(1)
  result = actor_objective_gradients(learner, starts, *noises)
  full, _, _ = setup(24)
  other = actor_objective_gradients(full, starts, *noises)
  for key in (
    "reward_gradient_norm",
    "bootstrap_gradient_norm",
    "total_gradient_norm",
    "gradient_cosine",
    "gradient_angle_degrees",
    "reward_return",
    "bootstrap_return",
    "total_return",
  ):
    assert result[key] == pytest.approx(other[key], rel=1e-12, abs=1e-12)
  training = learner.update_from(starts, *noises)
  assert result["total_gradient_norm"] == pytest.approx(
    float(training["Grad/actor_norm"]), rel=1e-12
  )
  assert result["total_return"] == pytest.approx(
    float(training["Policy/return"]), rel=1e-12
  )
  assert result["bootstrap_return"] == pytest.approx(
    float(training["Policy/bootstrap"]), rel=1e-12
  )
  assert result["effective_imagined_steps"] == 18
  assert result["reward_gradient_norm"] > 0
  assert result["bootstrap_gradient_norm"] > 0


def test_diagnostic_preserves_weights_buffers_gradients_modes_and_rng():
  learner, starts, noises = setup()
  learner.actor.register_buffer("diagnostic_buffer", torch.tensor([3.0]))
  learner.actor.eval()
  learner.actor.linear.train()
  learner.critic.requires_grad_(False)
  learner.actor._aux_loss = torch.tensor([42.0])
  modules = (
    learner.actor,
    learner.critic,
    learner.wm_env.dynamics,
    learner.wm_env.representation,
  )
  parameters = [p for module in modules for p in module.parameters()]
  for i, parameter in enumerate(parameters):
    parameter.grad = torch.full_like(parameter, i + 0.125)
  gradients = [p.grad.clone() for p in parameters]
  flags = [p.requires_grad for p in parameters]
  modes = [m.training for module in modules for m in module.modules()]
  weights = [deepcopy(module.state_dict()) for module in modules]
  state = learner.state_dict()
  rng = torch.random.get_rng_state().clone()
  first = actor_objective_gradients(learner, starts, *noises)
  second = actor_objective_gradients(learner, starts, *noises)
  assert first == second
  assert torch.equal(rng, torch.random.get_rng_state())
  equal_tree(state, learner.state_dict())
  for module, expected in zip(modules, weights, strict=True):
    equal_tree(module.state_dict(), expected)
  assert flags == [p.requires_grad for p in parameters]
  assert modes == [m.training for module in modules for m in module.modules()]
  for parameter, expected in zip(parameters, gradients, strict=True):
    assert torch.equal(parameter.grad, expected)
  assert learner.actor._aux_loss.tolist() == [42.0]
  assert learner.wm_env.state is None


def test_each_component_gradient_matches_fixed_noise_finite_difference():
  learner, starts, noises = setup()
  starts = starts.index(slice(2, 3))
  noises = tuple(noise[:, 2:3] for noise in noises)
  result = actor_objective_gradients(learner, starts, *noises)
  repeated = starts.repeat_interleave(learner.cfg.num_samples)
  flat_noise = [noise.flatten(1, 2) for noise in noises]
  differences = []
  epsilon = 1e-5
  for parameter in learner.actor.parameters():
    for i in range(parameter.numel()):
      original = parameter.detach().clone()
      values = []
      for sign in (1, -1):
        with torch.no_grad():
          parameter.copy_(original)
          parameter.view(-1)[i] += sign * epsilon
          _, _, _, components = learner._rollout(
            repeated, *flat_noise, learner.cfg.horizon, return_components=True
          )
          values.append(torch.stack([value.mean() for value in components]))
      with torch.no_grad():
        parameter.copy_(original)
      differences.append((values[0] - values[1]) / (2 * epsilon))
  reward, bootstrap = torch.stack(differences).T
  assert result["reward_gradient_norm"] == pytest.approx(float(reward.norm()), rel=1e-7)
  assert result["bootstrap_gradient_norm"] == pytest.approx(
    float(bootstrap.norm()), rel=1e-7
  )
  assert result["gradient_cosine"] == pytest.approx(
    float(reward @ bootstrap / (reward.norm() * bootstrap.norm())), rel=1e-7
  )


def test_certain_failure_has_zero_bootstrap_gradient_and_undefined_angle():
  learner, starts, noises = setup()
  step = learner.wm_env.step

  def fail(*args, **kwargs):
    obs, reward, done, info = step(*args, **kwargs)
    return obs, reward, torch.ones_like(done), info

  learner.wm_env.step = fail
  result = actor_objective_gradients(learner, starts, *noises)
  assert result["bootstrap_return"] == 0
  assert result["bootstrap_gradient_norm"] == 0
  assert result["gradient_angle_degrees"] is None
  assert result["gradient_cosine"] is None
  assert result["total_gradient_norm"] == result["reward_gradient_norm"]


def test_zero_discount_removes_bootstrap_and_later_rewards():
  learner, starts, noises = setup()
  learner.cfg = replace(learner.cfg, gamma=0)
  result = actor_objective_gradients(learner, starts, *noises)
  assert result["bootstrap_gradient_norm"] == 0
  assert result["reward_gradient_norm"] > 0


def test_real_record_runs_training_horizon_gradient_diagnostic(tmp_path):
  from supermimic.runners.eit_training import EITTraining
  from supermimic.scripts.diagnose_eit import diagnose
  from supermimic.scripts.eval_eit import evaluate
  from tests.test_estimator_eit_inference import physical_runner

  checkpoint, recording, motion = (
    tmp_path / name for name in ("training.pt", "record.pt", "motion.npz")
  )
  with physical_runner(tmp_path, "C") as runner:
    runner.cfg.estimator_inputs = replace(runner.cfg.estimator_inputs, mode="train")
    EITTraining(runner).save(checkpoint)
  # A saved training cap must win over a newly constructed environment's bound.
  saved = torch.load(checkpoint, weights_only=True)
  saved["policy"]["bootstrap_value_cap"] = 0.25
  torch.save(saved, checkpoint)
  evaluate(checkpoint, motion, steps=2, num_envs=1, seed=76, record_path=recording)
  with torch.compiler.set_stance("force_eager"):
    report = diagnose(
      checkpoint,
      recording,
      mode="critic",
      actor_gradients=True,
      gradient_seed=83,
      motion_file=motion,
    )
  gradient = report["actor_gradients"]
  assert gradient["horizon"] == 32
  assert gradient["samples_per_start"] == 1
  assert gradient["starts"] == 1
  assert gradient["effective_imagined_steps"] == 32
  assert gradient["actor_state_source"] == "posterior"
  assert gradient["bootstrap_value_cap"] == 0.25
  assert gradient["total_gradient_norm"] > 0
  assert gradient["episode_ids"] == [0]
  assert len(gradient["process_noise_sha256"]) == 64
