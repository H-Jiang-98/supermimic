"""Compare physical and imagined G1 rewards for one recorded control interval."""

from contextlib import closing
from unittest.mock import patch

import numpy as np
import pytest
import torch

from supermimic.runners.supermimic_runner import SuperMimicRunner, SuperMimicRunnerCfg
from supermimic.scripts.train_utils import create_env
from supermimic.tasks.env_cfgs import configure_estimator_env_cfg, gencoord_env_cfg


@pytest.mark.parametrize("timeout", [False, True], ids=["normal", "auto-reset"])
def test_real_and_imagined_rewards_use_20ms_successor(tmp_path, timeout):
  path = tmp_path / "moving.npz"
  frames = 4
  np.savez(
    path,
    joint_pos=np.zeros((frames, 29)),
    joint_vel=np.zeros((frames, 29)),
    root_pos_w=np.column_stack(
      (0.02 * np.arange(frames), np.zeros(frames), np.full(frames, 0.8))
    ),
    root_quat_w=np.tile([1.0, 0.0, 0.0, 0.0], (frames, 1)),
    root_lin_vel_w=np.tile([1.0, 0.0, 0.0], (frames, 1)),
    root_ang_vel_w=np.zeros((frames, 3)),
    fps=50.0,
  )
  cfg = configure_estimator_env_cfg(gencoord_env_cfg())
  cfg.seed = 2026
  cfg.commands["motion"].motion_file = str(path)
  cfg.commands["motion"].sampling_mode = "start"
  runner_cfg = SuperMimicRunnerCfg.for_estimator_inputs(
    "A",
    dynamics_hidden_dims=(16,),
    actor_hidden_dims=(16,),
    buffer_size=2,
    sampling_mode="start",
  )
  runner_cfg.policy.batch_size = 1
  runner_cfg.policy.noise = 0.0
  with (
    torch.compiler.set_stance("force_eager"),
    closing(create_env(cfg, 1, "cpu")) as env,
  ):
    runner = SuperMimicRunner(env, runner_cfg, "SuperMimic-GenCoord-G1", device="cpu")
    runner.reset_collection()
    real = env.unwrapped
    assert real.step_dt == 0.020
    start_time = real.sim.data.time.item()
    command = runner.motion_cmd
    reference_frame = command.time_steps.clone()
    action = torch.full((1, 29), 0.9)
    manager = real.reward_manager

    def check_endpoint():
      assert real.sim.data.time.item() - start_time == pytest.approx(0.020)
      for name in manager.active_terms:
        term = manager.get_term_cfg(name)
        expected = term.func(real, **term.params) * term.weight * real.step_dt
        torch.testing.assert_close(manager.get_term(name), expected, msg=name)
      return real.observation_manager.compute_group("full_state").clone()

    terminal_states = []

    def record_terminal_state(env_ids):
      assert env_ids.tolist() == [0]
      assert torch.equal(command.time_steps, reference_frame)
      # The reward has already been recorded. Refresh kinematics independently
      # and save the endpoint before the auto-reset replaces this robot state.
      real.sim.forward()
      terminal_states.append(check_endpoint())

    if timeout:
      real.episode_length_buf[:] = real.max_episode_length - 1
    with (
      patch.object(runner.policy_learner.actor, "forward", return_value=action),
      patch.object(real.recorder_manager, "record_pre_reset", record_terminal_state),
    ):
      runner.step(record_episode_stats=False)
    assert bool(real.reset_buf.item()) == timeout
    if timeout:
      (successor,) = terminal_states
      assert real.episode_length_buf.item() == 0
      assert not torch.allclose(successor, env.get_observations()["full_state"])
    else:
      assert not terminal_states
      assert torch.equal(command.time_steps, reference_frame + 1)
      # Keep the reference at the frame for which the action was chosen.
      with patch.object(command, "time_steps", reference_frame):
        successor = check_endpoint()

    # Replay the same transition through the imagined step, supplying the
    # observed successor so model prediction error cannot change the reward.
    imagine = runner.wm_env
    imagine.reset()
    representation = imagine.representation
    predicted = (
      representation.extract_aux_state(successor),
      representation.represent(successor),
    )
    with patch.object(imagine.dynamics, "forward", return_value=predicted):
      _, imagined_reward, _, _ = imagine.step(action)
    torch.testing.assert_close(imagined_reward, real.reward_buf, atol=2e-6, rtol=1e-5)
