"""Scene identity, chunk boundaries, isolation and heterogeneous clip execution."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from supermimic.estimator_inputs import GaussianMeasurementConfig, sample_proprio
from supermimic.validation.batched import (
  ChunkMetrics,
  DeviceStatus,
  TrackingRuntime,
  batches,
  noise_block,
)
from supermimic.validation.config import StudyConfig, scenario_seed
from supermimic.validation.domains import domain_table
from supermimic.validation.metrics import TRACKING, BinnedMetrics
from supermimic.validation.runtime import make_env_config, run_batch
from tests.test_eit_validation_runtime import checkpoint  # noqa: F401


def test_noise_blocks_exactly_match_framewise_streams_and_capacity():
  cfg = StudyConfig()
  measurement = GaussianMeasurementConfig()
  scenes = [("one", 0), ("two", 8), ("one", 15)]
  block = noise_block(cfg, scenes, 267, measurement, "cpu")
  for row, (sha, context) in enumerate(scenes):
    g = torch.Generator().manual_seed(
      scenario_seed(cfg.evaluation_seed, sha, context, "sensor")
    )
    old = torch.stack(
      [
        sample_proprio(torch.zeros(1, 93), measurement, generator=g)[0, :64]
        for _ in range(267)
      ]
    )
    torch.testing.assert_close(block[:, row], old, atol=0, rtol=0)
  solo = noise_block(cfg, [scenes[1]], 267, measurement, "cpu")
  torch.testing.assert_close(block[:, 1], solo[:, 0], atol=0, rtol=0)


def test_device_status_stops_at_tail_and_preserves_first_anomaly():
  status = DeviceStatus(torch.tensor([4, 2, 0, 4]))
  status.reject(torch.tensor([False, True, True, False]), 1)
  status.reject(torch.tensor([False, True, False, False]), 2)
  status.reject(torch.tensor([True, False, False, False]), 2, posterior=True)
  assert status.first_invalid.tolist() == [-1, 1, -1, -1]
  assert status.posterior_invalid.tolist() == [2, -1, -1, -1]
  assert status.active(2).tolist() == [True, False, False, True]
  assert not status.active(4).any()


def test_chunk_metrics_match_legacy_across_bins_and_exclude_padding():
  old = BinnedMetrics(TRACKING, 2, 3, "cpu")
  new = ChunkMetrics(4, 3, 4, [], "cpu")
  torch.manual_seed(4)
  values = torch.rand(7, 4, len(TRACKING))
  for frame, v in enumerate(values):
    old.update(v[:2])
    new.update(v, torch.tensor([True, True, frame < 3, False]), torch.empty(3, 0, 36))
  expected = old.result()
  actual = new.result(slice(0, 2), 7)
  for key in ("full", "bins", "bin_counts", "frames"):
    np.testing.assert_allclose(actual[key], expected[key])
  assert new.result(slice(2, 4), 7)["valid_frame_counts"].tolist() == [3, 0]


def test_length_bucketing_never_changes_task_identities():
  work = [
    (str(i), dict(frames=n, name=str(i)), "A-truth", list(range(16)))
    for i, n in enumerate([9, 1, 8, 2, 7])
  ]
  groups = list(batches(work, 64))
  assert [len(g) for g in groups] == [4, 1]
  assert sorted(t[0] for g in groups for t in g) == [str(i) for i in range(5)]
  assert [[t[1]["frames"] for t in g] for g in groups] == [[9, 8, 7, 2], [1]]


def test_multiclip_runtime_matches_legacy_and_padding(checkpoint):  # noqa: F811
  root, saved = checkpoint
  original = root / "motion.npz"
  short = root / "short.npz"
  with np.load(original) as a:
    np.savez(
      short,
      **{k: (a[k][:5] if a[k].ndim and len(a[k]) == 8 else a[k]) for k in a.files},
    )
  cfg = StudyConfig(
    device="cpu",
    compile_kernels=False,
    num_envs=4,
    num_domain_contexts=1,
    transfer_chunk_steps=3,
    host_check_interval_steps=3,
    time_bin_s=0.04,
    boxplot_interval_s=0.04,
  )
  env_cfg, _ = make_env_config(cfg, original)
  table = domain_table(env_cfg, 1, cfg.sobol_seed)
  motions = [
    dict(path=str(original), sha256="long", frames=8, name=original.name),
    dict(path=str(short), sha256="short", frames=5, name=short.name),
  ]
  tasks = [(m["sha256"], m, "C-posterior", [0]) for m in motions]
  with torch.compiler.set_stance("force_eager"):
    runtime = TrackingRuntime(cfg, motions, table, {"C": saved})
    try:
      p = runtime.prepare(tasks, "C-posterior")
      out, timing = runtime.execute(tasks, "C-posterior", p)
      for (_, motion, _, ids), (_, actual, meta) in zip(tasks, out, strict=True):
        old, old_meta = run_batch(
          cfg,
          "tracking",
          motion["path"],
          motion["sha256"],
          ids,
          table,
          "C-posterior",
          saved,
          saved,
        )
        assert meta["status"] == "complete"
        assert meta["scored_frames"] == motion["frames"]
        assert actual["valid_frame_counts"].tolist() == [motion["frames"]]
        for name in (
          "initial_truth",
          "initial_prior",
          "full",
          "bins",
          "replay_truth",
          "replay_posterior",
          "replay_reference",
        ):
          np.testing.assert_allclose(
            actual[name], old[name], atol=2e-5, rtol=1e-3, err_msg=name
          )
        assert meta["odometry_status"] == "complete"
        assert actual["odometry_valid_frame_counts"].tolist() == [motion["frames"]]
        assert actual["odometry_full"].shape == (1, 11)
        pos_error = (
          actual["replay_posterior"][:, 0, :3] - actual["replay_truth"][:, 0, :3]
        )
        np.testing.assert_allclose(
          actual["odometry_full"][0, 1],
          np.sqrt(np.mean(np.sum(pos_error**2, axis=1))),
          atol=1e-6,
        )
      assert timing["effective_frames"] == 13
      assert runtime.command.time_steps.tolist()[:2] == [7, 12]
    finally:
      runtime.close()


def test_compile_requires_cuda():
  cfg = replace(StudyConfig(), device="cpu")
  with pytest.raises(RuntimeError, match="CUDA"):
    TrackingRuntime(cfg, [], {}, {})


def test_numerical_failure_isolated_per_scene(checkpoint):  # noqa: F811
  root, saved = checkpoint
  cfg = StudyConfig(
    device="cpu",
    compile_kernels=False,
    num_envs=3,
    num_domain_contexts=2,
    transfer_chunk_steps=3,
    host_check_interval_steps=3,
  )
  env_cfg, _ = make_env_config(cfg, root / "motion.npz")
  table = domain_table(env_cfg, 2, cfg.sobol_seed)
  motion = dict(
    path=str(root / "motion.npz"), sha256="fixture", frames=8, name="motion.npz"
  )
  task = [("key", motion, "C-posterior", [0, 1])]
  with torch.compiler.set_stance("force_eager"):
    runtime = TrackingRuntime(cfg, [motion], table, {"C": saved})
    try:
      p = runtime.prepare(task, "C-posterior")
      baseline, _ = runtime.execute(task, "C-posterior", p)
      predict = runtime.bundle("C-posterior")["predict"]

      def fail_one(*args, **kwargs):
        mean, lower, bad = predict(*args, **kwargs)
        lower = lower.clone()
        lower[0] = float("nan")
        bad = bad.clone()
        bad[0] = True
        return mean, lower, bad

      runtime.bundle("C-posterior")["predict"] = fail_one
      p = runtime.prepare(task, "C-posterior")
      failed, timing = runtime.execute(task, "C-posterior", p)
      arrays, meta = failed[0][1:]
      assert meta["status"] == "invalid"
      assert arrays["first_invalid_frame"].tolist() == [1, -1]
      assert arrays["valid_frame_counts"].tolist() == [1, 8]
      assert timing["effective_frames"] == 9
      np.testing.assert_allclose(
        arrays["full"][1], baseline[0][1]["full"][1], atol=1e-6
      )
      assert torch.isfinite(runtime.raw.sim.data.qpos).all()
    finally:
      runtime.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA compilation regression")
def test_cuda_filter_capture_and_actor_parity(checkpoint):  # noqa: F811
  root, _ = checkpoint
  saved = torch.load(root / "training.pt", map_location="cuda:0", weights_only=True)
  cfg = StudyConfig(device="cuda:0", num_envs=64, num_domain_contexts=2)
  env_cfg, _ = make_env_config(cfg, root / "motion.npz")
  table = domain_table(env_cfg, 2, cfg.sobol_seed)
  motion = dict(
    path=str(root / "motion.npz"), sha256="fixture", frames=8, name="motion.npz"
  )
  task = [("key", motion, "C-posterior", [0, 1])]
  runtime = TrackingRuntime(cfg, [motion], table, {"C": saved})
  try:
    prepared = runtime.prepare(task, "C-posterior")
    assert runtime.check_compile(prepared)["passed"]
    # Mutate all static-input values and verify capture does not replay stale data.
    prepared["noise"].add_(0.001)
    prepared["belief"].mean.position.add_(0.002)
    assert runtime.check_compile(prepared)["passed"]
    prepared = runtime.prepare(task, "C-posterior")
    outputs, timing = runtime.execute(task, "C-posterior", prepared)
    assert timing["effective_frames"] == 16
    assert outputs[0][2]["status"] == "complete"
  finally:
    runtime.close()


def test_truth_control_survives_posterior_failure(checkpoint, tmp_path):  # noqa: F811
  from supermimic.runners.belief_training import BeliefModelTraining
  from tests.test_estimator_eit_inference import physical_runner

  root, saved = checkpoint
  with physical_runner(tmp_path, "A") as runner:
    runner.cfg.estimator_inputs = replace(runner.cfg.estimator_inputs, mode="train")
    BeliefModelTraining(runner).save(tmp_path / "a.pt")
  actor_saved = torch.load(tmp_path / "a.pt", map_location="cpu", weights_only=True)
  cfg = StudyConfig(
    device="cpu", compile_kernels=False, num_envs=2, num_domain_contexts=2
  )
  env_cfg, _ = make_env_config(cfg, root / "motion.npz")
  table = domain_table(env_cfg, 2, cfg.sobol_seed)
  motion = dict(
    path=str(root / "motion.npz"), sha256="fixture", frames=8, name="motion.npz"
  )
  task = [("key", motion, "A-truth", [0, 1])]
  with torch.compiler.set_stance("force_eager"):
    runtime = TrackingRuntime(cfg, [motion], table, {"C": saved, "A": actor_saved})
    try:
      p = runtime.prepare(task, "A-truth")
      baseline, _ = runtime.execute(task, "A-truth", p)
      predict = runtime.bundle("A-truth")["predict"]

      def fail_one(*args, **kwargs):
        mean, lower, bad = predict(*args, **kwargs)
        lower = lower.clone()
        lower[0] = float("nan")
        bad = bad.clone()
        bad[0] = True
        return mean, lower, bad

      runtime.bundle("A-truth")["predict"] = fail_one
      p = runtime.prepare(task, "A-truth")
      failed, timing = runtime.execute(task, "A-truth", p)
      arrays, meta = failed[0][1:]
      assert meta["status"] == "complete"
      assert arrays["first_invalid_frame"].tolist() == [-1, -1]
      assert meta["odometry_status"] == "invalid"
      assert arrays["odometry_valid_frame_counts"].tolist() == [1, 8]
      assert arrays["posterior_invalid_frame"].tolist() == [1, -1]
      assert timing["effective_frames"] == 16
      np.testing.assert_array_equal(
        arrays["replay_truth"], baseline[0][1]["replay_truth"]
      )
      np.testing.assert_array_equal(arrays["full"], baseline[0][1]["full"])
      assert np.isnan(arrays["replay_posterior"][1:, 0]).all()
    finally:
      runtime.close()


def test_first_physical_failure_censors_both_scores(checkpoint):  # noqa: F811
  root, saved = checkpoint
  cfg = StudyConfig(
    device="cpu",
    compile_kernels=False,
    num_envs=2,
    num_domain_contexts=2,
    time_bin_s=0.04,
  )
  env_cfg, _ = make_env_config(cfg, root / "motion.npz")
  table = domain_table(env_cfg, 2, cfg.sobol_seed)
  motion = dict(
    path=str(root / "motion.npz"), sha256="fixture", frames=8, name="motion.npz"
  )
  task = [("key", motion, "C-posterior", [0, 1])]
  with torch.compiler.set_stance("force_eager"):
    runtime = TrackingRuntime(cfg, [motion], table, {"C": saved})
    try:
      p = runtime.prepare(task, "C-posterior")
      calls = 0

      def monitor():
        nonlocal calls
        assert runtime.raw.episode_length_buf.tolist() == [calls + 1] * 2
        assert runtime.command.time_steps.tolist() == [calls] * 2
        runtime.monitor._terminated_buf[:] = torch.tensor([calls in (2, 5), calls >= 4])
        calls += 1

      runtime.monitor.compute = monitor
      outputs, timing = runtime.execute(task, "C-posterior", p)
      arrays, meta = outputs[0][1:]
      assert meta["status"] == meta["odometry_status"] == "complete"
      np.testing.assert_allclose(arrays["first_failure_s"], [0.06, 0.10])
      for prefix in ("", "odometry_"):
        assert arrays[prefix + "valid_frame_counts"].tolist() == [3, 5]
        np.testing.assert_array_equal(
          np.isfinite(arrays[prefix + "bins"][:, :, 0]),
          [[True, True], [True, True], [False, True], [False, False]],
        )
      for row, end in enumerate([3, 5]):
        truth = arrays["replay_truth"][:end, row, :3]
        for prefix, key in (
          ("", "replay_reference"),
          ("odometry_", "replay_posterior"),
        ):
          delta = truth - arrays[key][:end, row, :3]
          squared = (delta**2).sum(axis=1)
          assert arrays[prefix + "full"][row, 1] == pytest.approx(
            np.sqrt(squared.mean()), abs=1e-6
          )
          populated = [np.sqrt(squared[i : i + 2].mean()) for i in range(0, end, 2)]
          np.testing.assert_allclose(
            arrays[prefix + "bins"][: len(populated), row, 1], populated, atol=1e-6
          )
      assert timing["effective_frames"] == 16
    finally:
      runtime.close()


def test_large_capacity_keeps_small_tail_and_sample_count():
  from supermimic.validation.tracking import batch_capacity

  work = [
    (str(i), dict(frames=i + 1, name=str(i)), "A-truth", list(range(16)))
    for i in range(40)
  ]
  groups = list(batches(work, 512))
  assert [batch_capacity(g, 512) for g in groups] == [512, 128]
  assert sum(len(t[3]) for g in groups for t in g) == 640
  assert batch_capacity(list(work), 640) == 640
