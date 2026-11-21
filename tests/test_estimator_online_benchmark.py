"""Online timing must preserve production actions, beliefs and random streams."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from supermimic.runners.eit_inference import EITPolicyAdapter
from supermimic.scripts.benchmark_eit_online import (
  PhaseTimer,
  latency_statistics,
  run_timing_pass,
)
from tests.test_estimator_eit_inference import physical_runner


@pytest.mark.parametrize("group", ["A", "B", "C"])
def test_timing_hook_does_not_change_actions_posteriors_or_rng(tmp_path, group):
  with physical_runner(tmp_path, group) as runner:
    args = (runner.env, runner.policy_learner.actor, runner._wm, runner.cfg)
    plain = EITPolicyAdapter(*args)
    timer = PhaseTimer("cpu")
    measured = EITPolicyAdapter(*args, phase_timer=timer)
    actor_before = deepcopy(runner.policy_learner.actor.state_dict())
    obs, _ = runner.env.reset()
    dones = None
    for step in range(3):
      timer.values.clear()
      rng = torch.random.get_rng_state()
      if step == 0:
        plain.reset(obs)
      else:
        plain.advance(obs, dones)
      expected = plain.act()
      expected_rng = torch.random.get_rng_state()
      torch.random.set_rng_state(rng)
      if step == 0:
        measured.reset(obs)
      else:
        measured.advance(obs, dones)
      actual = measured.act()
      assert torch.equal(actual, expected)
      assert torch.equal(torch.random.get_rng_state(), expected_rng)
      torch.testing.assert_close(
        measured.provider.state.belief.covariance,
        plain.provider.state.belief.covariance,
        atol=0,
        rtol=0,
      )
      torch.testing.assert_close(
        measured.provider.state.belief.mean.to_full(),
        plain.provider.state.belief.mean.to_full(),
        atol=0,
        rtol=0,
      )
      assert set(timer.values) == {
        "measurement_history",
        "filter_and_jacobian",
        "features",
        "actor",
        "clamp_and_commit",
      }
      assert all(value > 0 for value in timer.values.values())
      if step == 1:
        runner.env.unwrapped.episode_length_buf[0] = (
          runner.env.unwrapped.max_episode_length - 1
        )
      if step != 2:
        obs, _, dones, _ = runner.env.step(expected)
    assert measured.time_index.tolist() == [0, 2]
    for name, value in actor_before.items():
      assert torch.equal(value, runner.policy_learner.actor.state_dict()[name])


def test_latency_quantiles_and_period_headroom():
  result = latency_statistics([10, 20, 30, 40, 50])
  assert result["mean_ms"] == 30
  assert result["p95_ms"] == 48
  assert result["p99_ms"] == 49.6
  assert result["p99_headroom_ms"] == pytest.approx(-29.6)
  assert result["deadline_miss_fraction"] == 0.6
  assert latency_statistics([])["p99_ms"] is None
  with pytest.raises(ValueError, match="finite"):
    latency_statistics([float("nan")])


def test_timing_pass_excludes_warmup_and_separates_reset_samples():
  class Env:
    unwrapped = SimpleNamespace(step_dt=0.02)

    def reset(self):
      self.steps = 0
      return None, None

    def step(self, actions):
      self.steps += 1
      return None, None, torch.tensor([self.steps == 3]), None

  class Policy:
    device = "cpu"
    phase_timer = None
    time_index = torch.tensor([0])
    calls = 0

    def reset(self, obs):
      self.time_index = torch.tensor([0])

    def advance(self, obs, dones):
      self.time_index = torch.where(dones, 0, self.time_index + 1)

    def act(self):
      self.calls += 1
      return torch.zeros(1, 29)

  policy = Policy()
  result = run_timing_pass(Env(), policy, seed=9, warmup=2, samples=4)
  assert policy.calls == 7  # Initial act, two warmup transitions, four samples.
  assert result["warmup_reset_count"] == 0
  assert result["sample_reset_count"] == 1
  assert [row["episode_step"] for row in result["samples"]] == [0, 1, 2, 3]
  assert result["statistics"]["all"]["total"]["count"] == 4
  assert result["statistics"]["continuous_full_filter"]["total"]["count"] == 3
  assert result["statistics"]["reset_prior"]["total"]["count"] == 1
  assert set(result["first_reset"]) == {
    "environment_reset",
    "adapter_reset",
    "first_act",
  }
  assert result["memory"] is None


def test_timing_pass_restores_existing_hook_when_physics_fails():
  def fail():
    raise RuntimeError("reset failed")

  previous = object()
  policy = SimpleNamespace(device="cpu", phase_timer=previous)
  with pytest.raises(RuntimeError, match="reset failed"):
    run_timing_pass(
      SimpleNamespace(reset=fail), policy, seed=9, warmup=0, samples=1, stages=True
    )
  assert policy.phase_timer is previous


def test_benchmark_releases_loaded_checkpoint_before_warmup(monkeypatch, tmp_path):
  import weakref

  from supermimic.scripts import benchmark_eit_online as module

  class Checkpoint(dict):
    pass

  references = []

  def load(*args):
    saved = Checkpoint(optimizer=torch.ones(32))
    references.append(weakref.ref(saved))
    return saved, "checkpoint-digest"

  actor = torch.nn.Linear(1, 1)
  cfg = SimpleNamespace(estimator_inputs=SimpleNamespace(group="A"))
  env = SimpleNamespace(unwrapped=SimpleNamespace(step_dt=0.02), close=lambda: None)
  monkeypatch.setattr(module, "load_hashed_checkpoint", load)
  monkeypatch.setattr(module, "evaluation_config", lambda saved: cfg)
  monkeypatch.setattr(module, "evaluation_env_config", lambda *args: None)
  monkeypatch.setattr(module, "create_env", lambda *args: env)
  monkeypatch.setattr(module, "load_evaluation_models", lambda *args: (None, actor))
  monkeypatch.setattr(
    module,
    "EITPolicyAdapter",
    lambda *args: SimpleNamespace(command=SimpleNamespace(motion=None)),
  )
  monkeypatch.setattr(module, "motion_content_digest", lambda motion: "motion-digest")
  monkeypatch.setattr(module, "file_sha256", lambda path: "motion-sha")

  def timing(*args, **kwargs):
    assert references[0]() is None
    return {"statistics": {}}

  monkeypatch.setattr(module, "run_timing_pass", timing)
  result = module.benchmark(tmp_path / "checkpoint.pt", tmp_path / "motion.npz")
  assert result["checkpoint_released_before_warmup"] is True
  assert "Non-PyTorch allocations" in result["memory_scope"]
