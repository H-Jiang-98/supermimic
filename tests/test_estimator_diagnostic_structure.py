"""Malformed cached diagnostic JSON must trigger recomputation, never a crash."""

from copy import deepcopy

import pytest

from supermimic.scripts.pilot_report_diagnostics import diagnostic_structure_errors


def valid_diagnostic_score():
  return {
    "version": "eit_independent_diagnostics_v2",
    "training_overlap_check": {"status": "distinct_seed"},
    "episodes": 2,
    "process": {
      "windows": 0,
      "window_frames": 8,
      "window_scored_successors": 0,
      "transitions": 2,
      "process_nll": 1.0,
      "multistep_mse": dict.fromkeys(
        ("root_pos", "root_ori", "joint_pos", "joint_vel")
      ),
      "rpv_bias_truth_minus_prediction_right_local": [0.0] * 9,
      "whitened_mean": [0.0] * 70,
      "whitened_second_moment": [[float(i == j) for j in range(70)] for i in range(70)],
    },
    "fixed_record_posterior": [
      {
        "frames": 4,
        "surviving_episodes": 2,
        "start_s": 0.0,
        "end_s": 1.0,
        "position_rmse_m": 0.1,
        "rotation_rmse_rad": 0.1,
        "velocity_rmse_m_s": 0.1,
        "mean_innovation_norm": 1.0,
        "coverage_95": 1.0,
        "null_direction_violations": 0,
      }
    ],
    "critic": {
      "gamma": 0.99,
      "bootstrap_value_cap": 10.0,
      "actor_state_source": "posterior",
      "censored_samples": 0,
      "censored_episodes": 0,
      "raw_value_error": {"samples": 4, "bias": 1.0, "mae": 1.0, "rmse": 1.0},
      "bootstrap_value_error": {"samples": 4, "bias": 1.0, "mae": 1.0, "rmse": 1.0},
      "episodes": [
        {
          "env_id": i,
          "reason": "failure",
          "raw_value": [1.0, 1.0],
          "bootstrap_value": [1.0, 1.0],
          "discounted_return": [2.0, 1.0],
        }
        for i in range(2)
      ],
    },
  }


def valid_gradient_score():
  return {
    "seed": 2028,
    "starts": 2,
    "samples_per_start": 2,
    "horizon": 4,
    "microbatch_size": 2,
    "effective_imagined_steps": 16,
    "episode_ids": [0, 1],
    "start_frames": [0, 0],
    "reward_gradient_norm": 1.0,
    "bootstrap_gradient_norm": 0.0,
    "total_gradient_norm": 1.0,
    "reward_return": 1.0,
    "bootstrap_return": 0.0,
    "total_return": 1.0,
    "gradient_dot_product": 0.0,
    "process_noise_sha256": "e" * 64,
    "measurement_noise_sha256": "f" * 64,
    "gradient_cosine": None,
    "gradient_angle_degrees": None,
  }


def test_complete_diagnostic_and_zero_bootstrap_are_structurally_valid():
  result = valid_diagnostic_score()
  assert diagnostic_structure_errors(result) == []
  result["actor_gradients"] = valid_gradient_score()
  assert diagnostic_structure_errors(result, require_gradients=True) == []


@pytest.mark.parametrize("root", [None, [], "incomplete", 0])
def test_invalid_json_root_returns_a_single_error(root):
  assert len(diagnostic_structure_errors(root)) == 1


@pytest.mark.parametrize("component", ["process", "fixed_record_posterior", "critic"])
@pytest.mark.parametrize("malformed", [None, "missing", 0, {}])
def test_invalid_required_blocks_return_a_single_error(component, malformed):
  result = valid_diagnostic_score()
  result[component] = malformed
  assert len(diagnostic_structure_errors(result)) == 1


def test_empty_posterior_and_nonfinite_deep_fields_are_rejected():
  result = valid_diagnostic_score()
  result["fixed_record_posterior"] = []
  assert diagnostic_structure_errors(result)
  result = valid_diagnostic_score()
  result["process"]["whitened_second_moment"][5][9] = float("nan")
  errors = diagnostic_structure_errors(result)
  assert len(errors) == 1 and "[5][9]" in errors[0]


def test_corrupted_matrix_and_critic_array_lengths_are_rejected():
  result = valid_diagnostic_score()
  result["process"]["whitened_second_moment"] = [[None] * 70 for _ in range(70)]
  assert len(diagnostic_structure_errors(result)) == 1
  result = valid_diagnostic_score()
  result["critic"]["episodes"][0]["discounted_return"] = [1.0]
  assert diagnostic_structure_errors(result)


def test_zero_pair_and_censored_scores_keep_explicit_null_statistics():
  result = valid_diagnostic_score()
  process = result["process"]
  process["transitions"] = 0
  for key in (
    "process_nll",
    "rpv_bias_truth_minus_prediction_right_local",
    "whitened_mean",
    "whitened_second_moment",
  ):
    process[key] = None
  critic = result["critic"]
  critic.update(censored_samples=4, censored_episodes=2)
  for key in ("raw_value_error", "bootstrap_value_error"):
    critic[key] = {"samples": 0, "bias": None, "mae": None, "rmse": None}
  for ep in critic["episodes"]:
    ep.update(reason="evaluation_truncated", discounted_return=None)
  assert diagnostic_structure_errors(result) == []
  malformed = deepcopy(result)
  malformed["critic"]["episodes"][0]["discounted_return"] = [0, 0]
  assert diagnostic_structure_errors(malformed)


def test_clip_completion_is_censored_only_for_infinite_horizon_scores():
  result = valid_diagnostic_score()
  episode = result["critic"]["episodes"][0]
  episode["reason"] = "clip_complete"
  assert diagnostic_structure_errors(result)
  result["version"] = "eit_independent_diagnostics_v1"
  assert diagnostic_structure_errors(result) == []
  result["version"] = "eit_independent_diagnostics_v2"
  episode["discounted_return"] = None
  assert diagnostic_structure_errors(result) == []
