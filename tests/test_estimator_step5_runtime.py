"""Coverage, episode recording, and ONNX parity on actual adapter inputs."""

from dataclasses import replace

import pytest
import torch

from supermimic.evaluation_diagnostics import PosteriorBins, rpv_ellipsoid
from supermimic.runners.eit_inference import EITPolicyAdapter
from supermimic.runners.eit_training import EITTraining
from supermimic.runners.exporter import export_eit_actor, load_eit_backend
from supermimic.scripts.eval_eit import evaluate
from tests.test_estimator_eit_inference import physical_runner


def test_correlated_and_degenerate_ellipsoids():
  covariance = torch.zeros(3, 9, 9)
  covariance[:, :2, :2] = torch.tensor([[2.0, 1.0], [1.0, 2.0]])
  error = torch.zeros(3, 9)
  error[:, :2] = torch.tensor([2.0, 1.0])
  error[1, 5] = 1.0
  covariance[2] = 0
  error[2] = 0
  distance, rank, violation, covered = rpv_ellipsoid(error, covariance)
  torch.testing.assert_close(
    distance, torch.tensor([2.0, 2.0, 0.0], dtype=torch.double)
  )
  assert rank.tolist() == [2, 2, 0]
  assert violation.tolist() == [False, True, False]
  assert covered.tolist() == [True, False, True]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("correlation", [0.0, 0.8])
def test_anisotropic_gaussian_ellipsoids_keep_all_positive_variances(
  dtype, correlation
):
  noise = torch.randn(
    10000, 9, dtype=dtype, generator=torch.Generator().manual_seed(34)
  )
  factor = torch.eye(9, dtype=dtype)
  factor[1, 0] = correlation
  factor[1, 1] = (1 - correlation**2) ** 0.5
  scales = torch.tensor([1.0] + [1e-4] * 8, dtype=dtype)
  factor = scales[:, None] * factor
  error = noise @ factor.T
  covariance = (factor @ factor.T).expand(len(noise), -1, -1)

  distance, rank, violation, covered = rpv_ellipsoid(error, covariance)

  assert (rank == 9).all()
  assert not violation.any()
  expected_distance = noise.double().square().sum(-1)
  torch.testing.assert_close(distance, expected_distance, atol=1e-5, rtol=1e-6)
  assert torch.equal(covered, expected_distance <= 16.918978)
  assert covered.double().mean().item() == pytest.approx(0.95, abs=0.015)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_strongly_correlated_gaussian_ellipsoids_keep_positive_modes(dtype):
  covariance = torch.full((9, 9), 1 - 5e-6, dtype=dtype)
  covariance.diagonal().fill_(1)
  factor = torch.linalg.cholesky(covariance.double())
  noise = torch.randn(
    10000, 9, dtype=torch.double, generator=torch.Generator().manual_seed(42)
  )
  error = (noise @ factor.T).to(dtype)
  # An independent full-rank solve also accounts for error rounded to float32.
  white = torch.linalg.solve_triangular(factor, error.double().T, upper=False)
  expected_distance = white.square().sum(0)

  distance, rank, violation, covered = rpv_ellipsoid(
    error, covariance.expand(len(error), -1, -1)
  )

  assert (rank == 9).all()
  assert not violation.any()
  torch.testing.assert_close(distance, expected_distance, atol=1e-8, rtol=1e-8)
  assert torch.equal(covered, expected_distance <= 16.918978)
  assert covered.double().mean().item() == pytest.approx(0.95, abs=0.015)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_small_positive_modes_and_true_null_directions_in_one_batch(dtype):
  covariance = torch.full((4, 9, 9), 1 - 5e-6, dtype=dtype)
  covariance.diagonal(dim1=-2, dim2=-1).fill_(1)
  covariance[1:3, 8, :] = 0
  covariance[1:3, :, 8] = 0
  covariance[3] = 0
  error = torch.zeros(4, 9, dtype=dtype)
  error[:3, 0], error[:3, 1] = 0.002, -0.002
  error[2, 8] = 1e-30
  # [a, -a, 0, ...] is an eigenvector with eigenvalue 1 - correlation.
  expected_distance = 2 * error[:, 0].double().square() / (
    1 - covariance[:, 0, 1].double()
  )

  distance, rank, violation, covered = rpv_ellipsoid(error, covariance)

  torch.testing.assert_close(distance, expected_distance, atol=1e-8, rtol=1e-8)
  assert rank.tolist() == [9, 8, 8, 0]
  assert violation.tolist() == [False, False, True, False]
  assert covered.tolist() == [True, True, False, True]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_anisotropic_degenerate_support_and_zero_variance_errors(dtype):
  direction = torch.tensor([1.0, 1e-4] + [0.0] * 7, dtype=dtype)
  covariance = torch.outer(direction, direction).expand(4, -1, -1)
  error = 1.5 * direction.expand(4, -1).clone()
  error[1, 1] *= -1
  error[2, 5] = 1e-30
  error[3] *= 2

  distance, rank, violation, covered = rpv_ellipsoid(error, covariance)

  assert rank.tolist() == [1, 1, 1, 1]
  assert violation.tolist() == [False, True, True, False]
  assert covered.tolist() == [True, False, False, False]
  torch.testing.assert_close(
    distance[[0, 3]], torch.tensor([2.25, 9.0], dtype=torch.double)
  )


@pytest.mark.parametrize("kind", ["negative_variance", "zero_variance", "correlation"])
def test_ellipsoids_reject_indefinite_covariance_at_small_scales(kind):
  covariance = torch.eye(9)
  if kind == "negative_variance":
    covariance[1, 1] = -1e-8
  elif kind == "zero_variance":
    covariance[1, 1] = 0
    covariance[0, 1] = covariance[1, 0] = 1e-8
  else:
    covariance[1, 1] = 1e-8
    covariance[0, 1] = covariance[1, 0] = 2e-4
  with pytest.raises(ValueError, match="Indefinite"):
    rpv_ellipsoid(torch.zeros(9), covariance)


def test_seeded_gaussian_coverage_and_streaming_bins():
  generator = torch.Generator().manual_seed(1709)
  size = 4000
  error = torch.randn(size, 9, generator=generator)
  covariance = torch.eye(9).expand(size, -1, -1)
  indices = torch.arange(size)
  data = (
    indices // 100,
    indices % 100,
    error,
    covariance,
    torch.zeros(size),
    error.norm(dim=-1),
    error[:, 3:6] * 0.5,
    error[:, 6:9] * 0.5,
  )
  bins = PosteriorBins(0.02, 25)
  bins.update(*data)
  streaming = PosteriorBins(0.02, 25)
  for start in range(0, size, 67):
    streaming.update(*(value[start : start + 67] for value in data))
  for together, chunks in zip(bins.result(), streaming.result(), strict=True):
    for key in together:
      assert chunks[key] == pytest.approx(together[key])
    assert together["position_rmse_m"] == pytest.approx(
      together["right_local_rho_rmse_m"] * 0.5
    )
    assert together["velocity_rmse_m_s"] == pytest.approx(
      together["right_local_nu_rmse_m_s"] * 0.5
    )
    assert together["surviving_episodes"] == 40
    assert together["coverage_95"] == pytest.approx(0.95, abs=0.025)


def test_recording_and_diagnostics_preserve_physical_frames_and_rng(
  tmp_path, monkeypatch
):
  checkpoint = tmp_path / "training.pt"
  with physical_runner(tmp_path, "B") as runner:
    runner.cfg.estimator_inputs = replace(runner.cfg.estimator_inputs, mode="train")
    EITTraining(runner).save(checkpoint)
  captures = []
  original = EITPolicyAdapter.act

  def capture(adapter, obs):
    actions = original(adapter, obs)
    state = adapter.provider.state
    captures.append(
      tuple(
        t.clone()
        for t in (
          actions,
          state.belief.mean.to_full(),
          state.belief.covariance,
          adapter.history,
          state.context,
          adapter.actor.obs_normalizer._std,
          torch.random.get_rng_state(),
        )
      )
    )
    return actions

  monkeypatch.setattr(EITPolicyAdapter, "act", capture)
  plain = evaluate(
    checkpoint, tmp_path / "motion.npz", steps=3, seed=201, posterior_diagnostics=False
  )
  expected = captures.copy()
  captures.clear()
  result = evaluate(
    checkpoint,
    tmp_path / "motion.npz",
    steps=3,
    seed=201,
    record_path=tmp_path / "episodes.pt",
    posterior_bin_steps=2,
  )
  assert plain["episodes"] == result["episodes"]
  assert "posterior_rmse" not in plain
  for left, right in zip(expected, captures, strict=True):
    assert all(torch.equal(a, b) for a, b in zip(left, right, strict=True))
  record = torch.load(tmp_path / "episodes.pt", weights_only=True)
  for episode in record["episodes"]:
    assert episode["time_index"].tolist() == list(range(episode["length"]))
    assert episode["reason"] == result["episodes"][episode["env_id"]]["reason"]
    assert episode["reward"].sum().item() == pytest.approx(episode["return"])
    for t in episode["time_index"]:
      assert torch.equal(episode["action"][t], expected[t][0][episode["env_id"]])


@pytest.mark.parametrize("group", ["A", "B", "C"])
def test_export_parity_on_recorded_reset_and_live_inputs(tmp_path, group):
  with physical_runner(tmp_path, group) as runner, torch.no_grad():
    actor = runner.policy_learner.actor
    if group == "C":
      actor.film[-1].weight.normal_(std=0.05)
    actor.decoder[-1].bias[0] = 2.0
    runner.env.clip_actions = 0.5
    graph = tmp_path / "actor.onnx"
    export_eit_actor(actor, runner.env, runner._wm, runner.cfg, "fixture", graph)
    backend = load_eit_backend(graph, "fixture", group, "cpu", 1)
    with pytest.raises(ValueError, match="disagree"):
      load_eit_backend(graph, "different", group, "cpu", 1)
    adapter = EITPolicyAdapter(runner.env, actor, runner._wm, runner.cfg)
    obs, _ = runner.env.reset()
    adapter.reset(obs)
    for step in range(3):
      limited = adapter.act(obs)
      inputs = adapter.last_inputs[:1]
      raw = actor(inputs)
      torch.testing.assert_close(backend(inputs), raw, atol=1e-5, rtol=1e-5)
      torch.testing.assert_close(
        backend(inputs).clamp(-0.5, 0.5), limited[:1], atol=1e-5, rtol=1e-5
      )
      assert (raw.abs() > 0.5).any()
      if group == "C":
        stressed = inputs.clone()
        # Increase all nine log standard deviations in each reference block.
        stressed["task_uncertainty"].reshape(1, 10, 45)[:, :, :9] += 8
        torch.testing.assert_close(
          backend(stressed), actor(stressed), atol=1e-5, rtol=1e-5
        )
      if step == 0:
        runner.env.unwrapped.episode_length_buf[0] = (
          runner.env.unwrapped.max_episode_length - 1
        )
      obs, _, dones, _ = runner.env.step(limited)
      if step == 0:
        assert dones[0]
      adapter.advance(obs, dones)
