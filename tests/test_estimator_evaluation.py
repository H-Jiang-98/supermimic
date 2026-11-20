"""The tracking evaluator scores each first episode with the original metrics."""

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from supermimic.runners.eit_training import EITTraining
from supermimic.scripts import eval_eit
from tests.test_estimator_eit_inference import physical_runner


class _TrackingEnv:
  """A scripted physical stream with visibly different post-reset states."""

  def __init__(self, endings, *, displacement=None):
    self.endings = endings
    self.num_envs = len(endings)
    self.device = "cpu"
    self.unwrapped = self
    self.step_dt = 0.02
    self.steps = 0
    self.finished = torch.zeros(self.num_envs, dtype=torch.bool)
    self.displacement = torch.tensor(
      displacement or list(range(1, self.num_envs + 1)), dtype=torch.float32
    )
    self.scene = SimpleNamespace(env_origins=torch.zeros(self.num_envs, 3))
    quaternion = torch.zeros(self.num_envs, 2, 4)
    quaternion[..., 0] = 1
    self.command = SimpleNamespace(
      num_envs=self.num_envs,
      device=self.device,
      cfg=SimpleNamespace(body_names=("root", "foot")),
      time_steps=torch.zeros(self.num_envs, dtype=torch.long),
      motion=SimpleNamespace(
        last_frame=lambda times: torch.full_like(times, 9),
        motion_ends=torch.tensor([10]),
      ),
      body_pos_w=torch.zeros(self.num_envs, 2, 3),
      body_quat_w=quaternion.clone(),
      robot_body_pos_w=torch.zeros(self.num_envs, 2, 3),
      robot_body_quat_w=quaternion.clone(),
      joint_vel=torch.zeros(self.num_envs, 29),
      robot_joint_vel=torch.zeros(self.num_envs, 29),
    )
    self.command.body_pos_w[:, 0, 2] = 1
    self.command.body_pos_w[:, 1, 0] = 1
    self.command.robot_body_pos_w.copy_(self.command.body_pos_w)
    self.command.robot_anchor_pos_w = self.command.robot_body_pos_w[:, 0]
    self.command.robot_anchor_quat_w = self.command.robot_body_quat_w[:, 0]
    self.command_manager = SimpleNamespace(get_term=lambda name: self.command)
    self.termination_manager = SimpleNamespace(
      terminated=torch.zeros(self.num_envs, dtype=torch.bool),
      time_outs=torch.zeros(self.num_envs, dtype=torch.bool),
    )
    self.clip_end = torch.zeros(self.num_envs, dtype=torch.bool)
    self.termination_manager.get_term = lambda name: self.clip_end

  def get_observations(self):
    full = torch.zeros(self.num_envs, 71)
    full[:, :3] = self.command.robot_body_pos_w[:, 0]
    full[:, 3] = 1
    return {"full_state": full}

  def step(self, actions):
    assert actions.shape == (self.num_envs, 29)
    self.steps += 1
    # These robot states correspond to the pre-step reference, not the command
    # frame advanced below. Root-relative error is zero despite global drift.
    self.command.robot_body_pos_w.copy_(self.command.body_pos_w)
    self.command.robot_body_pos_w[..., 0] += self.displacement[:, None]
    self.command.robot_joint_vel.copy_(self.command.joint_vel + 3)
    self.termination_manager.terminated.zero_()
    self.termination_manager.time_outs.zero_()
    self.clip_end.zero_()
    dones = torch.zeros(self.num_envs, dtype=torch.bool)
    rewards = torch.arange(1, self.num_envs + 1, dtype=torch.float32)
    rewards[self.finished] = 1000  # Subsequent episodes must not enter totals.
    for row, (at_step, reason) in enumerate(self.endings):
      if self.steps == at_step:
        dones[row] = True
        self.termination_manager.terminated[row] = reason in {"failure", "both"}
        self.clip_end[row] = reason in {"clip_complete", "both"}
        self.termination_manager.time_outs[row] = reason in {
          "clip_complete",
          "timeout",
          "both",
        }
    self.finished |= dones
    # An auto-reset makes terminal measurements unsuitable for scoring.
    self.command.robot_body_pos_w[dones] = 10000
    self.command.robot_joint_vel[dones] = 10000
    self.command.body_pos_w[..., 0] += 10
    self.command.joint_vel += 10
    self.command.time_steps += 1
    return (
      self.get_observations(),
      rewards,
      dones,
      {"time_outs": self.termination_manager.time_outs.clone()},
    )


class _Policy:
  def __init__(self, env):
    self.env = env
    self.resets = 0
    self.acts = 0
    self.advances = []

  def reset(self, obs=None):
    self.resets += 1

  def act(self, obs=None):
    self.acts += 1
    return torch.zeros(self.env.num_envs, 29)

  def advance(self, obs, dones):
    self.advances.append(dones.clone())


@pytest.fixture
def _fixed_belief_diagnostics(monkeypatch):
  # Test episode bookkeeping and actual tracking metric functions independently
  # of the filter, whose streaming/gradient contracts have their own tests.
  def diagnostics(policy, obs):
    size = policy.env.num_envs
    return {
      "error_squared": torch.zeros(size, 70),
      "variance": torch.ones(size, 70),
      "tracking_squared": torch.zeros(size, 3),
      "jitter": torch.zeros(size),
      "innovation_norm": torch.zeros(size),
    }

  monkeypatch.setattr(eval_eit, "_belief_diagnostics", diagnostics)


def _run(endings, *, steps=None, displacement=None):
  env = _TrackingEnv(endings, displacement=displacement)
  policy = _Policy(env)
  result = eval_eit.run_evaluation(env, policy, steps=steps, ee_body_names=("foot",))
  return env, policy, result


def test_first_episode_completion_failure_and_timeout_are_distinct(
  _fixed_belief_diagnostics,
):
  env, policy, result = _run(
    [(1, "failure"), (3, "clip_complete"), (2, "timeout"), (2, "both")]
  )
  episodes = sorted(result["episodes"], key=lambda episode: episode["env_id"])
  assert [episode["reason"] for episode in episodes] == [
    "failure",
    "clip_complete",
    "timeout",
    "failure",
  ]
  assert [episode["length"] for episode in episodes] == [1, 3, 2, 2]
  assert [episode["return"] for episode in episodes] == [1, 6, 6, 8]
  assert result["completed_episodes"] == 4
  assert result["failure_count"] == 2
  assert result["clip_complete_count"] == 1
  assert result["success_rate"] == 0.25
  assert result["mean_episode_return"] == pytest.approx(5.25)
  assert not any(episode["censored"] for episode in episodes)
  assert env.steps == policy.acts == len(policy.advances) == 3
  assert policy.resets == 1


def test_metrics_use_scoring_frame_current_root_and_exclude_reset_states(
  _fixed_belief_diagnostics,
):
  _, _, result = _run(
    [(1, "failure"), (3, "clip_complete"), (2, "timeout"), (2, "both")]
  )
  episodes = sorted(result["episodes"], key=lambda episode: episode["env_id"])
  assert [episode["metric_steps"] for episode in episodes] == [0, 2, 1, 1]
  assert result["metric_episode_count"] == 3
  assert episodes[0]["metrics"]["mpkpe"] is None
  assert result["mpkpe"] == pytest.approx(3)
  assert result["r_mpkpe"] == pytest.approx(0, abs=1e-6)
  assert result["joint_vel_error"] == pytest.approx(3)
  assert result["ee_pos_error"] == pytest.approx(0, abs=1e-6)
  assert result["ee_ori_error"] == pytest.approx(0, abs=1e-6)


def test_tracking_metrics_average_episodes_not_all_surviving_frames(
  _fixed_belief_diagnostics,
):
  _, _, result = _run([(2, "failure"), (4, "clip_complete")], displacement=[1, 9])
  assert result["mpkpe"] == pytest.approx(5)
  assert [e["metric_steps"] for e in result["episodes"]] == [1, 3]


def test_step_budget_retains_unfinished_episode_and_marks_censoring(
  _fixed_belief_diagnostics,
):
  env, policy, result = _run([(1, "failure"), (4, "clip_complete")], steps=2)
  episodes = sorted(result["episodes"], key=lambda episode: episode["env_id"])
  assert [episode["reason"] for episode in episodes] == [
    "failure",
    "evaluation_truncated",
  ]
  assert [episode["return"] for episode in episodes] == [1, 4]
  assert [episode["length"] for episode in episodes] == [1, 2]
  assert result["completed_episodes"] == 1
  assert result["success_rate"] is None
  assert not episodes[0]["censored"]
  assert episodes[1]["censored"]
  assert not episodes[1]["timeout"]
  assert env.steps == policy.acts == 2


def test_no_scored_states_report_missing_metrics_instead_of_perfect_tracking(
  _fixed_belief_diagnostics,
):
  _, _, result = _run([(1, "failure"), (1, "timeout")])
  assert result["metric_episode_count"] == 0
  for key in (
    "mpkpe",
    "r_mpkpe",
    "joint_vel_error",
    "ee_pos_error",
    "ee_ori_error",
  ):
    assert result[key] is None


def test_invalid_end_effector_name_fails_instead_of_reporting_zero_error(
  _fixed_belief_diagnostics,
):
  env = _TrackingEnv([(2, "clip_complete")])
  with pytest.raises(ValueError, match="tracked"):
    eval_eit.run_evaluation(env, _Policy(env), ee_body_names=("missing",))


def test_real_checkpoint_evaluates_new_motion_until_every_first_episode_finishes(
  tmp_path,
):
  torch.manual_seed(2501)
  checkpoint = tmp_path / "training.pt"
  with physical_runner(tmp_path, "C") as runner:
    runner.cfg.estimator_inputs = replace(runner.cfg.estimator_inputs, mode="train")
    EITTraining(runner).save(checkpoint)
  before = checkpoint.read_bytes()
  motion = tmp_path / "held_out.npz"
  with np.load(tmp_path / "motion.npz") as original:
    arrays = {key: original[key].copy() for key in original.files}
  arrays["joint_pos"][:, 0] += 0.001
  np.savez(motion, **arrays)

  # No explicit steps: default evaluation must finish each first episode and
  # use the real filter diagnostics, not the toy-stream fixture above.
  result = eval_eit.evaluate(checkpoint, motion, num_envs=2, seed=808)
  assert result["training_motion_digest"] != result["evaluation_motion_digest"]
  assert result["evaluation_complete"]
  assert result["step_budget"] is None
  assert result["completed_episodes"] == len(result["episodes"]) == 2
  assert result["censored_episodes"] == 0
  assert {episode["env_id"] for episode in result["episodes"]} == {0, 1}
  assert all(episode["start_frame"] == 0 for episode in result["episodes"])
  assert all(episode["length"] > 0 for episode in result["episodes"])
  assert all(
    episode["reason"] in {"failure", "clip_complete", "timeout"}
    for episode in result["episodes"]
  )
  assert result["group"] == "C" and result["mode"] == "eval"
  assert len(result["posterior_rmse"]) == 70
  assert json.loads(json.dumps(result, allow_nan=False)) == result
  assert checkpoint.read_bytes() == before
