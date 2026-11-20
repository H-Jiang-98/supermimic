"""Small real-MuJoCo and stochastic-imagination step-2 acceptance run.

Run with OMP_NUM_THREADS=1 and a writable MPLCONFIGDIR. The generated stationary
clip exercises collection, retained training segments and stochastic imagination.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import mjlab.tasks  # noqa: F401
import mujoco
import numpy as np
import torch
import warp as wp
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML
from mjlab.tasks.registry import load_env_cfg

import supermimic.tasks  # noqa: F401
from supermimic.models import ProcessNoiseConfig
from supermimic.runners.supermimic_runner import SuperMimicRunner, SuperMimicRunnerCfg
from supermimic.scripts.train_utils import create_env
from supermimic.tasks.env_cfgs import configure_estimator_env_cfg


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--device", default="cpu")
  args = parser.parse_args()
  torch.set_num_threads(1)
  torch.manual_seed(2026)
  torch.compiler.set_stance("force_eager")
  wp.config.kernel_cache_dir = str(
    Path(tempfile.gettempdir()) / "supermimic-step2-warp"
  )
  start = time.perf_counter()
  with tempfile.TemporaryDirectory(prefix="supermimic-step2-") as directory:
    path = Path(directory) / "motion.npz"
    model = mujoco.MjModel.from_xml_path(str(G1_XML))
    joints = np.flatnonzero(model.jnt_type == int(mujoco.mjtJoint.mjJNT_HINGE))
    frames = 4
    np.savez(
      path,
      joint_pos=np.tile(model.qpos0[model.jnt_qposadr[joints]], (frames, 1)),
      joint_vel=np.zeros((frames, len(joints))),
      root_pos_w=np.tile(model.qpos0[:3], (frames, 1)),
      root_quat_w=np.tile(model.qpos0[3:7], (frames, 1)),
      root_lin_vel_w=np.zeros((frames, 3)),
      root_ang_vel_w=np.zeros((frames, 3)),
      fps=50.0,
    )
    cfg = load_env_cfg("SuperMimic-GenCoord-G1")
    cfg.commands["motion"].motion_file = str(path)
    cfg.commands["motion"].sampling_mode = "start"
    configure_estimator_env_cfg(cfg)
    env = create_env(cfg, 2, args.device)
    try:
      runner_cfg = SuperMimicRunnerCfg.for_estimator_inputs(
        "C",
        dynamics_hidden_dims=(32, 32),
        actor_hidden_dims=(32, 16),
        process_noise=ProcessNoiseConfig(hidden_dims=(16,)),
        sampling_mode="start",
        buffer_size=2,
      )
      runner_cfg.policy.batch_size = 2
      runner_cfg.policy.noise = 0.0
      runner = SuperMimicRunner(
        env, runner_cfg, "SuperMimic-GenCoord-G1", device=args.device
      )
      real, imagined = runner.configure_belief_filter()
      runner.reset_collection()
      version = runner._dynamics.model_version
      reset_slots, jitter_count = 0, 0
      collection_start = time.perf_counter()
      for _ in range(8):
        reset_slots += int(runner._measurement_reset.sum())
        runner.step()
        jitter_count += int((real.diagnostics["jitter"] > 0).sum())
        assert real.state.belief.covariance.grad_fn is None
      collection_seconds = time.perf_counter() - collection_start
      assert version == runner._dynamics.model_version
      assert reset_slots > 2, "Short clip must exercise episode resets"

      runner.finish_collection_round()
      runner._dynamics.requires_grad_(False)
      obs = runner.wm_env.reset(
        episode_id=real.state.episode_id, time_index=real.state.time_index
      )
      oldest_steps = []
      for row, (episode, t) in enumerate(
        zip(real.state.episode_id.tolist(), real.state.time_index.tolist(), strict=True)
      ):
        seed, records = runner.replay_buffer.training_prefix(episode, t)
        oldest_steps.append(int(records["time_index"][0]))
        with torch.no_grad():
          expected = real.estimator.correct(seed, records["actor_obs"][0, :64])
          for i in range(1, len(records)):
            expected = real.estimator.predict(
              expected, records["action"][i - 1], records["sim_ctx"][i - 1]
            )
            expected = real.estimator.correct(expected, records["actor_obs"][i, :64])
        torch.testing.assert_close(
          expected.mean.to_full(), imagined.state.belief.mean.to_full()[row]
        )
        torch.testing.assert_close(
          expected.covariance, imagined.state.belief.covariance[row]
        )
      rewards = []
      imagination_start = time.perf_counter()
      for _ in range(2):
        action = runner.policy_learner.actor(obs)
        obs, reward, _, _ = runner.wm_env.step(
          action,
          process_noise=torch.randn(2, 70, device=args.device),
          measurement_noise=torch.randn(2, 64, device=args.device),
        )
        rewards.append(reward)
      torch.stack(rewards).sum().backward()
      gradients = [
        p.grad for p in runner.policy_learner.actor.parameters() if p.grad is not None
      ]
      assert gradients and all(torch.isfinite(g).all() for g in gradients)
      assert any(g.abs().sum() > 0 for g in gradients)
      covariance = imagined.state.belief.covariance.detach()
      snapshot = runner.wm_env.snapshot()
      restored = runner.wm_env.restore(snapshot)
      for key in obs.keys():
        torch.testing.assert_close(restored[key], obs[key])
      summary = {
        "device": args.device,
        "dtype": str(covariance.dtype),
        "batch": 2,
        "collection_steps": 8,
        "segment_start_time": oldest_steps,
        "imagined_snapshot_restored": True,
        "imagined_horizon": 2,
        "reset_slots": reset_slots,
        "collection_seconds": collection_seconds,
        "imagined_forward_backward_seconds": time.perf_counter() - imagination_start,
        "total_seconds": time.perf_counter() - start,
        "history": list(obs["history"].shape),
        "goal": list(obs["goal_motion"].shape),
        "uncertainty": list(obs["task_uncertainty"].shape),
        "covariance_min_eigenvalue": float(torch.linalg.eigvalsh(covariance).min()),
        "covariance_max_variance": float(covariance.diagonal(dim1=-2, dim2=-1).max()),
        "collection_jitter_slots": jitter_count,
        "innovation_norm_max": float(real.diagnostics["innovation_norm"].max()),
        "actor_gradients_finite": True,
      }
      print("ESTIMATOR_SMOKE " + json.dumps(summary, sort_keys=True))
    finally:
      env.close()


if __name__ == "__main__":
  main()
