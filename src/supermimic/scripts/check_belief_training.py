"""Real MuJoCo acceptance: two learning rounds and exact fixed-batch resume."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
import time
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import mjlab.tasks  # noqa: F401
import mujoco
import numpy as np
import torch
import warp as wp
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML
from mjlab.tasks.registry import load_env_cfg

import supermimic.tasks  # noqa: F401
from supermimic.learners.belief_wm_learner import BeliefWorldModelLearnerCfg
from supermimic.models import ProcessNoiseConfig
from supermimic.runners.belief_training import BeliefModelTraining, BeliefTrainingCfg
from supermimic.runners.supermimic_runner import SuperMimicRunner, SuperMimicRunnerCfg
from supermimic.scripts.train_utils import create_env
from supermimic.tasks.env_cfgs import configure_estimator_env_cfg


def stationary_motion(path: Path):
  model = mujoco.MjModel.from_xml_path(str(G1_XML))
  joints = np.flatnonzero(model.jnt_type == int(mujoco.mjtJoint.mjJNT_HINGE))
  frames = 8
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


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--output", type=Path)
  args = parser.parse_args()
  torch.set_num_threads(1)
  torch.manual_seed(2026)
  torch.compiler.set_stance("force_eager")
  wp.config.kernel_cache_dir = str(
    Path(tempfile.gettempdir()) / "supermimic-step3-warp"
  )
  start = time.perf_counter()
  with tempfile.TemporaryDirectory(prefix="supermimic-step3-") as temporary:
    directory = Path(temporary)
    stationary_motion(directory / "motion.npz")
    cfg = load_env_cfg("SuperMimic-GenCoord-G1")
    cfg.commands["motion"].motion_file = str(directory / "motion.npz")
    cfg.commands["motion"].sampling_mode = "start"
    configure_estimator_env_cfg(cfg)
    env = create_env(cfg, 2, args.device)
    try:
      runner_cfg = SuperMimicRunnerCfg.for_estimator_inputs(
        "C",
        dynamics_hidden_dims=(32, 32),
        actor_hidden_dims=(32, 16),
        process_noise=ProcessNoiseConfig(hidden_dims=(16,)),
        belief_wm=BeliefWorldModelLearnerCfg(num_batches=2, batch_size=4),
        sampling_mode="start",
        buffer_size=12,
      )
      runner_cfg.policy.batch_size = 2
      runner_cfg.policy.noise = 0.0
      runner = SuperMimicRunner(
        env, runner_cfg, "SuperMimic-GenCoord-G1", device=args.device
      )
      training = BeliefModelTraining(runner, BeliefTrainingCfg(8, 2, 2))
      model = runner._dynamics
      mean_initial = deepcopy(model.mlp.state_dict())
      noise_initial = deepcopy(model.noise_head.state_dict())
      actor_initial = deepcopy(runner.policy_learner.actor.state_dict())
      logs = training.run(2, directory / "run")
      assert not runner.logger.ep_extras
      for log in logs:
        count = log["train"]["valid_transitions"]
        assert type(count) is int and 8 <= count <= 16
      assert logs[1]["collection_model_version"] == logs[0]["trained_model_version"]
      assert logs[0]["collection_model_version"] != logs[0]["trained_model_version"]
      assert runner.posterior_provider.state is None
      assert all(
        torch.equal(v, runner.policy_learner.actor.state_dict()[k])
        for k, v in actor_initial.items()
      )
      assert any(
        not torch.equal(v, model.mlp.state_dict()[k]) for k, v in mean_initial.items()
      )
      assert any(
        not torch.equal(v, model.noise_head.state_dict()[k])
        for k, v in noise_initial.items()
      )
      train_ids, _ = training.episode_split()
      batch = runner.replay_buffer.sample_belief_sequences(2, 2, episode_ids=train_ids)
      saved = training.state_dict()
      before = training.learner.evaluate(batch)["Loss/posterior_nll"]
      first_logs = training.learner.update(batch)
      expected = deepcopy(model.state_dict())
      expected_optimizer = deepcopy(training.learner.optimizer.state_dict())
      # Load through disk as well as exercising preflight and provider reset.
      torch.save(saved, directory / "resume.pt")
      training.load_state_dict(
        torch.load(directory / "resume.pt", map_location=args.device, weights_only=True)
      )
      torch.testing.assert_close(
        training.learner.evaluate(batch)["Loss/posterior_nll"], before, rtol=0, atol=0
      )
      second_logs = training.learner.update(batch)
      for key, value in expected.items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
      for key in first_logs:
        torch.testing.assert_close(first_logs[key], second_logs[key], rtol=0, atol=0)
      for key, value in expected_optimizer["state"].items():
        for field, tensor in value.items():
          torch.testing.assert_close(
            training.learner.optimizer.state_dict()["state"][key][field],
            tensor,
            rtol=0,
            atol=0,
          )
      training.refresh_collection()
      runner.step()
      assert runner.posterior_provider.state.model_version == model.model_version
      repository = Path(__file__).resolve().parents[3]
      summary = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "source_commit": subprocess.check_output(
          ["git", "rev-parse", "HEAD"], text=True, cwd=repository
        ).strip(),
        "source_files_sha256": {
          name: hashlib.sha256((repository / name).read_bytes()).hexdigest()
          for name in (
            "src/supermimic/scripts/check_belief_training.py",
            "src/supermimic/runners/belief_training.py",
            "src/supermimic/runners/supermimic_runner.py",
            "src/supermimic/learners/belief_wm_learner.py",
            "src/supermimic/replay_buffer.py",
            "src/supermimic/models.py",
            "src/supermimic/estimator.py",
            "src/supermimic/representations.py",
          )
        },
        "device": args.device,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(args.device)
        if str(args.device).startswith("cuda")
        else None,
        "dtype": "float32",
        "seed": 2026,
        "threads": 1,
        "mean_hidden_dims": [32, 32],
        "noise_hidden_dims": [16],
        "rounds": 2,
        "updates_per_round": 2,
        "collection_steps_per_round": 8,
        "batch_size": 4,
        "microbatch_size": 2,
        "horizon": 2,
        "both_heads_updated": True,
        "actor_unchanged": True,
        "next_round_uses_updated_model": True,
        "fixed_batch_resume_update_bitwise_equal": True,
        "episode_extras_buffer_empty": True,
        "total_seconds": time.perf_counter() - start,
        "metrics": logs,
      }
      if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
      print("BELIEF_TRAINING_SMOKE " + json.dumps(summary, sort_keys=True))
    finally:
      env.close()


if __name__ == "__main__":
  main()
