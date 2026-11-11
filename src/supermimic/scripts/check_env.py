"""Check both public-mjlab tasks with generated motion data and no W&B calls."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--num-envs", type=int, default=2)
  parser.add_argument("--steps", type=int, default=10)
  args = parser.parse_args()
  if args.num_envs < 1 or args.steps < 1:
    parser.error("--num-envs and --steps must be positive")

  import mjlab.tasks  # noqa: F401
  import mujoco
  import numpy as np
  import torch
  from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML
  from mjlab.tasks.registry import load_env_cfg

  import supermimic.tasks  # noqa: F401
  from supermimic.runners.supermimic_runner import BaseRunnerCfg, build_world_model
  from supermimic.scripts.train_utils import create_env, setup_device

  device = setup_device(args.device)
  if device.startswith("cuda") and not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable; verify nvidia-smi and GPU access first")
  with tempfile.TemporaryDirectory(prefix="supermimic-check-") as directory:
    path = Path(directory) / "motion.npz"
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
    for representation, task in (
      ("gencoord", "SuperMimic-GenCoord-G1"),
      ("maxcoord", "SuperMimic-MaxCoord-G1"),
    ):
      cfg = load_env_cfg(task)
      cfg.commands["motion"].motion_file = str(path)
      cfg.commands["motion"].sampling_mode = "external"
      env = create_env(cfg, args.num_envs, device)
      try:
        wm = build_world_model(
          BaseRunnerCfg(representation=representation, dynamics_hidden_dims=(32, 32)),
          env,
          device,
        )
        command = env.unwrapped.command_manager.get_term("motion")
        command.set_sampling_distribution(torch.ones(command.bin_count, device=device))
        for _ in range(args.steps):
          obs, reward, _, _ = env.step(
            torch.zeros(args.num_envs, env.num_actions, device=device)
          )
          if not torch.isfinite(reward).all() or not all(
            torch.isfinite(v).all() for v in obs.values()
          ):
            raise RuntimeError(f"{task}: non-finite simulator output")
        state = obs["full_state"].detach().clone()
        rep = wm.representation.represent(state)
        aux = wm.representation.extract_aux_state(state)
        context = wm.sim_context.read(env.unwrapped).detach().clone()
        action = torch.zeros(
          args.num_envs, env.num_actions, device=device, requires_grad=True
        )
        next_aux, next_rep = wm.dynamics(rep, action, aux, context)
        bodies = wm.representation.world_frame_bodies(next_aux, next_rep)
        loss = sum(value.square().mean() for value in bodies)
        loss.backward()
        if action.grad is None or not torch.isfinite(action.grad).all():
          raise RuntimeError(f"{task}: invalid gradient through dynamics/kinematics")
        if device.startswith("cuda"):
          torch.cuda.synchronize()
        print(
          f"PASS {task}: {args.steps} simulation steps and dynamics backward on {device}"
        )
      finally:
        env.close()


if __name__ == "__main__":
  main()
