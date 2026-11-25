"""A damaged loader must not be able to manufacture both sides of resume proof."""

import os
import subprocess
import sys
from copy import deepcopy
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from supermimic.runners.eit_resume_reference import (
  REFERENCE_SOURCE,
  REFERENCE_VERSION,
  _private_components,
  build_live_reference,
  compare_reference_update,
  equal_tree,
  fixed_update_execution,
  write_live_reference,
)
from tests.test_estimator_eit_training import advance, training


def test_private_reference_discards_live_graph_without_resetting_it():
  source = training()
  source.refresh_collection()
  source.policy.cuda_graphs = True
  source.policy._policy_graph = graph = object()
  source.policy._policy_graph_signature = signature = object()
  source.policy._policy_graph_stream = stream = object()
  _, _, private, _ = _private_components(source)
  assert private.cuda_graphs is False
  assert private._policy_graph is None
  assert private._policy_graph_signature is None
  assert private._policy_graph_stream is None
  assert private._graph_reason == "disabled"
  assert source.policy.cuda_graphs is True
  assert source.policy._policy_graph is graph
  assert source.policy._policy_graph_signature is signature
  assert source.policy._policy_graph_stream is stream


def test_live_reference_is_generated_without_any_loader_and_preserves_live_state(
  tmp_path,
):
  source = training()
  advance(source)
  path = tmp_path / "training.pt"
  source.save(path)
  state = torch.load(path, weights_only=True)
  before = source.state_dict()
  rng = torch.random.get_rng_state().clone()
  with (
    patch.object(source, "load_state_dict", side_effect=AssertionError("loader used")),
    patch(
      "torch.optim.Adam.load_state_dict",
      side_effect=AssertionError("optimizer loader used"),
    ),
    patch(
      "torch.optim.Optimizer.load_state_dict",
      side_effect=AssertionError("base optimizer loader used"),
    ),
    patch(
      "torch.nn.Module.load_state_dict",
      side_effect=AssertionError("module loader used"),
    ),
  ):
    reference = build_live_reference(source, path, state)
  assert reference["version"] == REFERENCE_VERSION
  assert reference["source"] == REFERENCE_SOURCE
  equal_tree(source.state_dict(), before)
  equal_tree(torch.random.get_rng_state(), rng)
  target = training(81)
  target.load_state_dict(state)
  details = compare_reference_update(target, reference)
  assert details["mean_windows"] == 2
  assert len(details["policy_starts"]) == 2


def test_sparse_reference_uses_training_and_model_validation_singletons(
  tmp_path,
):
  loop = training()
  runner = loop.runner
  loop.cfg = replace(loop.cfg, posterior_save_stride=8)
  runner.posterior_save_stride = 8
  runner.replay_buffer.enable_posterior_cache(8, runner.cfg.history_length)
  loop.refresh_collection()
  runner.step(record_episode_stats=False)
  loop.refresh_collection()

  checkpoint, evidence = tmp_path / "training.pt", tmp_path / "reference.pt"
  loop.before_checkpoint_commit = lambda path, state: write_live_reference(
    loop, path, state, evidence
  )
  advance(loop, checkpoint)
  reference = torch.load(evidence, weights_only=True)
  assert loop.episode_split()[0] == [3]
  assert reference["inputs"]["policy_keys"].tolist() == [[0, 0], [1, 0]]
  restored = training()
  restored.load_state_dict(torch.load(checkpoint, weights_only=True))
  compare_reference_update(restored, reference)


@pytest.mark.parametrize("optimizer", ["actor", "critic", "covariance", "mean"])
def test_losing_a_nonzero_restored_optimizer_moment_fails_against_live_reference(
  tmp_path, monkeypatch, optimizer
):
  source = training()
  advance(source)
  checkpoint = tmp_path / "training.pt"
  source.save(checkpoint)
  state = torch.load(checkpoint, weights_only=True)
  reference = build_live_reference(source, checkpoint, state)
  target = training(81)
  selected = {
    "actor": target.policy.actor_optimizers[0],
    "critic": target.policy.critic_optimizers[0],
    "covariance": target.learner.cov_optimizer,
    "mean": target.learner.mean.optimizers[0],
  }[optimizer]
  original = selected.load_state_dict
  deleted = []

  def broken_loader(value):
    original(value)
    for moments in selected.state.values():
      for key, moment in moments.items():
        if key != "step" and isinstance(moment, torch.Tensor) and moment.any():
          moment.zero_()
          deleted.append(key)
          return

  monkeypatch.setattr(selected, "load_state_dict", broken_loader)
  target.load_state_dict(deepcopy(state))
  assert deleted, "negative test must actually remove a nonzero optimizer moment"
  with pytest.raises(AssertionError, match="model_update|policy_update"):
    compare_reference_update(target, reference)


def test_precommit_reference_matches_committed_bytes_and_failure_keeps_old_checkpoint(
  tmp_path,
):
  loop = training()
  checkpoint, evidence = tmp_path / "training.pt", tmp_path / "reference.pt"
  advance(loop, checkpoint)
  old = checkpoint.read_bytes()

  def fail_reference(path, state):
    assert path.read_bytes() != old
    raise RuntimeError("reference generation failed")

  loop.before_checkpoint_commit = fail_reference
  with pytest.raises(RuntimeError, match="reference generation failed"):
    advance(loop, checkpoint)
  assert checkpoint.read_bytes() == old
  # A new live round from the last committed checkpoint can capture a reference.
  recovered = training()
  recovered.load_state_dict(torch.load(checkpoint, weights_only=True))
  recovered.before_checkpoint_commit = lambda path, state: write_live_reference(
    recovered, path, state, evidence
  )
  advance(recovered, checkpoint)
  reference = torch.load(evidence, weights_only=True)
  assert reference["collected_steps"] == 8
  from supermimic.runners.eit_snapshots import file_sha256

  assert reference["checkpoint_sha256"] == file_sha256(checkpoint)


def test_exact_comparison_rejects_signed_zero_nonfinite_and_type_changes():
  with pytest.raises(AssertionError, match="tensor bytes differ"):
    equal_tree(torch.tensor([0.0]), torch.tensor([-0.0]))
  with pytest.raises(AssertionError, match="dtype/shape"):
    equal_tree(torch.zeros(1), torch.zeros(1, dtype=torch.float64))
  with pytest.raises(AssertionError, match="non-finite"):
    equal_tree(torch.tensor([float("nan")]), torch.tensor([float("nan")]))
  with pytest.raises(AssertionError, match="root.count"):
    equal_tree({"count": 1}, {"count": 1.0})


def test_fixed_execution_restores_determinism_flags_after_failure():
  enabled = torch.are_deterministic_algorithms_enabled()
  warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
  try:
    torch.use_deterministic_algorithms(False, warn_only=True)
    with pytest.raises(RuntimeError, match="test failure"):
      with fixed_update_execution():
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
        raise RuntimeError("test failure")
    assert not torch.are_deterministic_algorithms_enabled()
    assert torch.is_deterministic_algorithms_warn_only_enabled()
  finally:
    torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


@pytest.mark.parametrize(
  "relative",
  [
    "metrics.jsonl",
    ".belief-training.lock",
    ".eit-initializing",
    ".eit-initializing/ref.pt",
    ".eit-publication/actor.pt",
    ".eit-log-interval.json",
    "snapshots/frames_0.pt",
    "wandb_run.json",
    "wandb/offline-run-example/run-example.wandb",
    "wandb-data/artifacts/staging/model.pt",
    "wandb-cache/model.pt",
  ],
)
def test_reference_output_cannot_replace_training_artifacts(tmp_path, relative):
  from supermimic.scripts.train_eit import _reference_path

  with pytest.raises(ValueError, match="separate from training"):
    _reference_path(tmp_path / relative, tmp_path)


def test_real_cpu_checkpoint_live_reference_and_independent_restore(tmp_path):
  from supermimic.learners.belief_policy_learner import BeliefPolicyLearnerCfg
  from supermimic.learners.two_stage_wm import (
    CovarianceFitCfg,
    MeanFitCfg,
    TwoStageWorldModelLearner,
  )
  from supermimic.runners.eit_training import EITTraining, EITTrainingCfg
  from supermimic.scripts.check_eit_resume import verify
  from tests.test_estimator_eit_inference import physical_runner

  checkpoint, reference = tmp_path / "training.pt", tmp_path / "live-reference.pt"
  with physical_runner(tmp_path, "C") as runner:
    runner.cfg.estimator_inputs = replace(runner.cfg.estimator_inputs, mode="train")
    runner.cfg.mean = MeanFitCfg(num_batches=1, batch_size=2, window_frames=3)
    runner.cfg.cov = CovarianceFitCfg(batch_size=2, validation_pairs=4)
    runner.world_model_learner = TwoStageWorldModelLearner(
      runner._dynamics, runner._representation, runner.cfg.mean, runner.cfg.cov, "cpu"
    )
    loop = EITTraining(
      runner,
      EITTrainingCfg(validation_stride=2),
      BeliefPolicyLearnerCfg(
        batch_size=2, num_samples=2, microbatch_size=2, horizon=2, critic_epochs=1
      ),
    )
    loop.before_checkpoint_commit = lambda path, state: write_live_reference(
      loop, path, state, reference
    )
    advance(loop, checkpoint)
  report = verify(checkpoint, tmp_path / "motion.npz", "cpu", reference)
  assert report["version"] == "eit_fixed_input_resume_v2"
  assert report["live_optimizer_reference"] is True
  assert report["expected_source"] == REFERENCE_SOURCE
  assert report["fixed_input_policy_resume_exact"] is True
  assert report["fixed_input_model_resume_exact"] is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA live graph reference")
def test_captured_policy_live_reference_and_independent_restore(tmp_path):
  # Establish deterministic cuBLAS settings before the child initializes CUDA,
  # independently of tests that already used other streams in this process.
  result = subprocess.run(
    [
      sys.executable,
      "-c",
      "from pathlib import Path; import sys; "
      "from tests.test_estimator_live_resume_reference import _cuda_graph_reference; "
      "_cuda_graph_reference(Path(sys.argv[1]))",
      str(tmp_path),
    ],
    env={
      **os.environ,
      "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
      "PYTORCH_JIT": "0",
      "OMP_NUM_THREADS": "1",
      "MUJOCO_GL": "egl",
    },
    capture_output=True,
    text=True,
    timeout=180,
  )
  assert result.returncode == 0, result.stdout + result.stderr


def _cuda_graph_reference(directory):
  from supermimic.scripts.check_eit_resume import verify
  from tests.test_estimator_eit_graph import graph_policy

  torch.set_num_threads(1)
  fixtures = graph_policy.__wrapped__(directory)
  make = next(fixtures)
  try:
    loop, _ = make(return_training=True, integrated=True)
    for _ in range(11):
      loop.run_step()
    policy = loop.policy
    captured, stream = policy._policy_graph, policy._policy_graph_stream
    assert captured.captures == 1 and captured.replay_calls > 0
    _, _, private, _ = _private_components(loop)
    assert private._policy_graph is None and private._policy_graph_stream is None
    before = loop.state_dict()
    checkpoint, evidence = directory / "training.pt", directory / "live-reference.pt"
    loop.before_checkpoint_commit = lambda path, state: write_live_reference(
      loop, path, state, evidence
    )
    loop.save(checkpoint)
    equal_tree(loop.state_dict(), before)
    assert policy._policy_graph is captured and policy._policy_graph_stream is stream
    report = verify(checkpoint, directory / "stationary.npz", "cuda:0", evidence)
    assert report["fixed_input_model_resume_exact"] is True
    assert report["fixed_input_policy_resume_exact"] is True
    # Publishing/verification must also leave the original capture reusable.
    replays = captured.replay_calls
    for _ in range(2):
      loop.run_step()
    assert policy._policy_graph is captured and captured.replay_calls > replays
  finally:
    next(fixtures, None)
