"""Independent fixed-episode process, calibration and hard-return diagnostics."""

import pytest
import torch

from supermimic.critic_inputs import BeliefCritic
from supermimic.scripts.diagnose_eit import (
  _validate_record,
  _white_statistics,
  discounted_returns,
  score_frames,
  score_process,
)
from tests.test_estimator_two_stage_wm import fixture


def _episode(length, x=0, reason="failure"):
  state = torch.zeros(length, 71)
  state[:, 0] = torch.arange(length) + x
  state[:, 3] = 1
  sensor = torch.zeros(length, 93)
  sensor[:, 2] = -1
  return {
    "env_id": int(x),
    "reason": reason,
    "time_index": torch.arange(length),
    "full_state": state,
    "prior_mean": state[0].clone(),
    "prior_covariance": torch.eye(70) * 0.01,
    "sensor": sensor,
    "true_goal": torch.zeros(length, 700),
    "reward": torch.arange(1, length + 1).float(),
    "action": torch.zeros(length, 29),
    "sim_ctx": torch.zeros(length, 2),
  }


def test_independent_pairs_have_signed_bias_and_multistep_physical_losses():
  estimator, _ = fixture(dtype=torch.float32)
  dynamics = estimator.dynamics
  representation = dynamics._apply_prediction.__self__
  with torch.no_grad():
    for parameter in dynamics.mlp.parameters():
      parameter.zero_()
  episodes = [_episode(3), _episode(2, 90), _episode(1, 300)]
  for size in (1, 16):
    result = score_process(dynamics, representation, episodes, 3, size)
    assert result["transitions"] == 3
    assert result["windows"] == 1
    assert result["window_scored_successors"] == 2
    assert result["multistep_mse"]["root_pos"] == pytest.approx((1 + 4) / 6)
    assert result["multistep_mse"]["root_ori"] == 0
    assert result["rpv_bias_truth_minus_prediction_right_local"] == [
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      0,
    ]
  short = score_process(dynamics, representation, episodes[1:], 8, 16)
  assert short["transitions"] == 1 and short["process_nll"] is not None
  assert short["windows"] == 0 and short["multistep_mse"]["root_pos"] is None
  isolated = score_process(dynamics, representation, episodes[-1:], 8, 16)
  assert isolated["process_nll"] is None and isolated["whitened_second_moment"] is None


def test_correlated_process_factor_whitens_signed_gaussian_cross_moments():
  gen = torch.Generator().manual_seed(2026)
  z = torch.randn(20000, 9, dtype=torch.float64, generator=gen)
  L = torch.eye(9, dtype=z.dtype)
  L[1, 0], L[1, 1] = 2.0, 0.5
  y = z @ L.T
  _, bias, mean, second = _white_statistics(y, L.expand(len(y), -1, -1))
  torch.testing.assert_close(mean / len(z), z.mean(0))
  torch.testing.assert_close(second / len(z), z.T @ z / len(z))
  torch.testing.assert_close(bias / len(z), y.mean(0))
  assert (second / len(z) - torch.eye(9)).abs().max() < 0.03
  _, _, _, unwhitened = _white_statistics(
    y, torch.eye(9).double().expand(len(y), -1, -1)
  )
  assert unwhitened[0, 1] / len(z) > 1.9


def test_hard_return_uses_terminal_reward_and_separates_censored_values():
  assert discounted_returns(torch.tensor([1.0, 2.0, 3.0]), 0.5).tolist() == [
    2.75,
    3.5,
    3.0,
  ]
  estimator, _ = fixture(dtype=torch.float32)
  critic = BeliefCritic("B", 2, (8,)).eval()
  with torch.no_grad():
    critic.mlp[-1].bias.fill_(8)
  record = {
    "actor_state_source": "posterior",
    "episodes": [
      _episode(3),
      _episode(2, 1, "timeout"),
      _episode(2, 3, "clip_complete"),
      _episode(1, 2, "evaluation_truncated"),
    ],
  }
  result = score_frames(estimator, record, "cpu", None, (critic, 0.5, 4.0))["critic"]
  assert result["episodes"][0]["discounted_return"] == [2.75, 3.5, 3.0]
  assert result["episodes"][0]["raw_value"] == [8.0] * 3
  assert result["episodes"][0]["bootstrap_value"] == [4.0] * 3
  assert result["episodes"][1]["discounted_return"] is None
  assert result["censored_episodes"] == 3 and result["censored_samples"] == 5
  assert result["bootstrap_value_error"]["samples"] == 3
  assert result["bootstrap_value_error"]["bias"] == pytest.approx((1.25 + 0.5 + 1) / 3)
  assert result["raw_value_error"]["bias"] == pytest.approx((5.25 + 4.5 + 5) / 3)


def test_record_rejects_cross_reset_and_missing_successors():
  ctx = {"scale": torch.ones(2)}
  saved = {"world_model": {"sim_context": ctx, "dynamics": {"metadata": {"dt": 0.02}}}}
  ep = _episode(3)
  record = {
    "version": "eit_diagnostic_episodes_v1",
    "source": "mujoco_evaluation",
    "split": "diagnostic",
    "episodes": [ep],
    "sim_context": ctx,
    "control_dt": 0.02,
  }
  _validate_record(record, saved)
  ep["time_index"][2] = 3
  with pytest.raises(ValueError, match="continuous"):
    _validate_record(record, saved)
  ep["time_index"][2] = 2
  ep["sensor"][-1, 64] = 5
  with pytest.raises(ValueError, match="Executed action"):
    _validate_record(record, saved)


def test_real_record_checkpoint_identity_and_optional_imagined_rollout(tmp_path):
  from dataclasses import replace

  from supermimic.runners.belief_training import BeliefModelTraining
  from supermimic.runners.eit_training import EITTraining
  from supermimic.scripts.diagnose_eit import diagnose
  from supermimic.scripts.eval_eit import evaluate
  from supermimic.scripts.record_eit_drift import summarize_evidence
  from tests.test_estimator_eit_inference import physical_runner

  checkpoint = tmp_path / "training.pt"
  recording = tmp_path / "record.pt"
  motion = tmp_path / "motion.npz"
  model_checkpoint = tmp_path / "model.pt"
  with physical_runner(tmp_path, "mean") as runner:
    runner.cfg.estimator_inputs = replace(runner.cfg.estimator_inputs, mode="train")
    BeliefModelTraining(runner).save(model_checkpoint)
    EITTraining(runner).save(checkpoint)
  online = evaluate(
    checkpoint,
    motion,
    steps=3,
    num_envs=1,
    seed=65,
    posterior_bin_steps=2,
    record_path=recording,
  )
  with torch.compiler.set_stance("force_eager"):
    model_report = diagnose(model_checkpoint, recording, window_frames=3)
    report = diagnose(
      checkpoint,
      recording,
      bin_steps=2,
      window_frames=3,
      imagined_horizon=3,
      imagined_samples=2,
      motion_file=motion,
    )
  assert model_report["round"] == model_report["collected_frames"] == 0
  assert model_report["process"]["transitions"] == 2
  assert model_report["fixed_record_posterior"] and "critic" not in model_report
  evidence = summarize_evidence(report, report)
  assert (
    evidence["collected_steps"]
    == evidence["update_groups"]
    == evidence["collected_frames"]
    == 0
  )
  assert "round" not in evidence
  assert report["process"]["transitions"] == 2
  assert report["process"]["windows"] == 1
  assert report["critic"]["censored_samples"] == 3
  assert report["critic"]["bootstrap_value_error"]["samples"] == 0
  assert len(report["imagined"]["episodes"][0]["return_samples"]) == 2
  for replayed, live in zip(
    report["fixed_record_posterior"], online["posterior_time_bins"], strict=True
  ):
    for key in (
      "position_rmse_m",
      "rotation_rmse_rad",
      "velocity_rmse_m_s",
      "coverage_95",
    ):
      assert replayed[key] == pytest.approx(live[key], rel=2e-4, abs=2e-6)
  record = torch.load(recording, weights_only=True)
  record["checkpoint_sha256"] = "different"
  torch.save(record, recording)
  with pytest.raises(ValueError, match="from this checkpoint"):
    diagnose(checkpoint, recording, mode="critic")

  reference = evaluate(
    checkpoint,
    motion,
    steps=2,
    num_envs=1,
    seed=66,
    record_path=recording,
    reference_controller=True,
  )
  assert reference["actor_state_source"] == "reference_joint_positions"
  with pytest.raises(ValueError, match="behavior policy"):
    diagnose(checkpoint, recording, mode="critic")


@pytest.mark.parametrize(
  (
    "steps",
    "tail",
    "target",
    "same_policy",
    "prefix",
    "prefix_error",
    "full_error",
    "critic_error",
  ),
  [
    (2, 1, [2.75], True, 2.0, -1.0, None, None),
    (5, 0, [2.75], True, 2.75, -1.75, -1.75, 7.0),
    (5, 0, [2.75], False, 2.75, None, None, 7.0),
    (2, 1, None, True, 2.0, -1.0, None, None),
    (5, 0, None, True, None, None, None, 7.0),
  ],
)
def test_imagined_errors_compare_matched_horizons_and_policies(
  steps, tail, target, same_policy, prefix, prefix_error, full_error, critic_error
):
  from supermimic.imagined_diagnostics import _compare_returns

  imagined = {"steps": steps, "samples_with_unscored_tail": tail, "return_mean": 1.0}
  ep = _episode(3, reason="failure" if target else "evaluation_truncated")
  physical = {"discounted_return": target, "bootstrap_value": [8.0]}
  row = _compare_returns(imagined, ep, physical, 0.5, same_policy)
  assert row["physical_prefix_return"] == prefix
  assert row["model_minus_physical_prefix"] == prefix_error
  assert row["model_minus_physical"] == full_error
  assert row["critic_minus_model"] == critic_error


def test_diagnostic_source_checks_training_seed_and_expanded_motion_files():
  from supermimic.scripts.diagnose_eit import _training_overlap

  record = {"seed": 8, "motion_digest": "loaded clip", "motion_sha256": "raw clip"}
  saved = {"training_seed": 8, "motion_digest": "loaded clip"}
  with pytest.raises(ValueError, match="distinct seed"):
    _training_overlap(saved, record)
  saved.update(
    motion_digest="combined digest",
    training_motion_files=[{"sha256": "other clip"}, {"sha256": "raw clip"}],
  )
  with pytest.raises(ValueError, match="distinct seed"):
    _training_overlap(saved, record)
  record["seed"] = 13
  assert _training_overlap(saved, record)["status"] == "distinct_seed"
  record.update(seed=8, motion_sha256="holdout clip")
  assert _training_overlap(saved, record)["status"] == "distinct_motion_file"
  del saved["training_motion_files"]
  assert (
    _training_overlap(saved, record)["status"]
    == "unverified_missing_training_file_manifest"
  )
  del saved["training_seed"]
  assert (
    _training_overlap(saved, record)["status"] == "unverified_missing_training_seed"
  )


def test_positive_soft_tail_survives_numeric_underflow():
  from types import SimpleNamespace

  from supermimic.imagined_diagnostics import _rollout

  env = SimpleNamespace(
    device="cpu",
    clear_rollout=lambda: None,
    reset_prepared=lambda _: None,
    motion=SimpleNamespace(last_frame=lambda times: times + 499),
    step=lambda *_args, **_kwargs: (None, torch.ones(1), torch.tensor([0.9]), {}),
  )
  start = SimpleNamespace(
    motion_times=torch.zeros(1, dtype=torch.long), repeat_interleave=lambda _: None
  )
  report = _rollout(
    lambda _: torch.zeros(1, 29), env, start, 400, 1, 0.9, torch.Generator()
  )
  assert report["discounted_tail_weight_mean"] == 0.0
  assert report["samples_with_unscored_tail"] == 1
  assert report["steps"] == 400
  assert not report["clip_horizon_reached"]
