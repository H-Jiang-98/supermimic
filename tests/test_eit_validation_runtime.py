"""Real MuJoCo context injection and frame-complete online C evaluation."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from supermimic.runners.eit_training import EITTraining
from supermimic.validation.config import StudyConfig
from supermimic.validation.domains import apply_domains, domain_table
from supermimic.validation.runtime import make_env_config, run_batch
from tests.test_estimator_eit_inference import physical_runner


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
  root = tmp_path_factory.mktemp("study")
  with physical_runner(root, "C") as runner:
    runner.cfg.estimator_inputs = replace(runner.cfg.estimator_inputs, mode="train")
    path = root / "training.pt"
    EITTraining(runner).save(path)
  return root, torch.load(path, map_location="cpu", weights_only=True)


def test_physical_domains_and_fullclip_continuity(checkpoint, monkeypatch):
  root, saved = checkpoint
  cfg = StudyConfig(
    device="cpu",
    num_domain_contexts=2,
    num_envs=2,
    time_bin_s=0.04,
    boxplot_interval_s=0.04,
  )
  env_cfg, _ = make_env_config(cfg, root / "motion.npz")
  table = domain_table(env_cfg, 2, cfg.sobol_seed)
  # Force the monitor to report failure at every frame; no simulation reset follows.
  from mjlab.managers.termination_manager import TerminationManager

  original = TerminationManager.compute

  def fail(self):
    value = original(self)
    if self.active_terms:
      self._terminated_buf[:] = True
    return value

  monkeypatch.setattr(TerminationManager, "compute", fail)
  from supermimic.validation import runtime

  applied = []

  def checked(env, context, values):
    result = apply_domains(env, context, values)
    com = context.find("base_com")
    # Absolute COM includes nominal torso z; it must not be clamped to [-1,1].
    assert (result[:, com][:, 2] > 1).all()
    applied.append(result)
    return result

  monkeypatch.setattr(runtime, "apply_domains", checked)
  with torch.compiler.set_stance("force_eager"):
    arrays, meta = run_batch(
      cfg,
      "odometry",
      root / "motion.npz",
      "fixture",
      [0, 1],
      table,
      "C-posterior",
      saved,
      saved,
    )
  assert meta["status"] == "complete" and meta["scored_frames"] == 8
  assert arrays["bin_counts"].tolist() == [2, 2, 2, 2]
  assert arrays["first_failure_s"].tolist() == [0.0, 0.0]
  assert np.isfinite(arrays["full"]).all()
  assert len(applied) == 1


def test_batch_partition_preserves_initial_state_and_metrics(checkpoint):
  root, saved = checkpoint
  cfg = StudyConfig(device="cpu", num_domain_contexts=2, num_envs=2)
  env_cfg, _ = make_env_config(cfg, root / "motion.npz")
  table = domain_table(env_cfg, 2, cfg.sobol_seed)
  with torch.compiler.set_stance("force_eager"):
    paired, _ = run_batch(
      cfg,
      "odometry",
      root / "motion.npz",
      "fixture",
      [0, 1],
      table,
      "C-posterior",
      saved,
      saved,
    )
    solo, _ = run_batch(
      cfg,
      "odometry",
      root / "motion.npz",
      "fixture",
      [1],
      table,
      "C-posterior",
      saved,
      saved,
    )
  np.testing.assert_allclose(
    paired["initial_truth"][1], solo["initial_truth"][0], atol=1e-6
  )
  np.testing.assert_allclose(
    paired["initial_prior"][1], solo["initial_prior"][0], atol=1e-6
  )
  np.testing.assert_allclose(paired["full"][1], solo["full"][0], atol=2e-3, rtol=2e-3)
