"""Existing viewers use the same EIT policy lifecycle as headless evaluation."""

from contextlib import closing
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from supermimic.critic_inputs import BeliefCritic
from supermimic.models import ProcessNoiseConfig
from supermimic.runners.eit_inference import EITPolicyAdapter
from supermimic.runners.run_config import EIT_TRAINING_VERSION
from supermimic.runners.supermimic_runner import (
  SuperMimicRunnerCfg,
  actor_checkpoint,
  build_actor,
  build_world_model,
)
from supermimic.scripts import play
from supermimic.scripts.check_estimator import stationary_motion
from supermimic.scripts.train_utils import create_env
from supermimic.scripts.viewers import (
  SuperMimicNativeViewer,
  SuperMimicViserViewer,
  create_wm_vis_model,
)
from supermimic.tasks.env_cfgs import configure_estimator_env_cfg, gencoord_env_cfg

TASK = "SuperMimic-GenCoord-G1"
RUN = "example/supermimic/91lwm1v7"


@pytest.fixture
def playback(tmp_path):
  first, second = tmp_path / "first.npz", tmp_path / "second.npz"
  stationary_motion(first)
  stationary_motion(second)
  env_cfg = configure_estimator_env_cfg(gencoord_env_cfg())
  env_cfg.commands["motion"].motion_file = [str(first), str(second)]
  env_cfg.commands["motion"].sampling_mode = "start"
  # Keep the test focused on clip/explicit reset, independent of random policy falls.
  env_cfg.terminations = {"motion_end": env_cfg.terminations["motion_end"]}
  cfg = SuperMimicRunnerCfg.for_estimator_inputs(
    "C",
    dynamics_hidden_dims=(16,),
    actor_hidden_dims=(16,),
    process_noise=ProcessNoiseConfig(hidden_dims=(8,)),
    sampling_mode="start",
  )
  cfg.estimator_inputs = replace(cfg.estimator_inputs, mode="eval")
  with (
    torch.compiler.set_stance("force_eager"),
    closing(create_env(env_cfg, 1, "cpu")) as env,
  ):
    wm = build_world_model(cfg, env, "cpu")
    actor = build_actor(cfg, wm, "cpu").eval().requires_grad_(False)
    wm.dynamics.eval().requires_grad_(False)
    adapter = EITPolicyAdapter(env, actor, wm, cfg)
    options = dict(
      env=env,
      actor=actor,
      dynamics=wm.dynamics,
      representation=wm.representation,
      wm_vis_model=create_wm_vis_model(env.unwrapped.sim.mj_model),
      num_steps_rollout=1,
      sim_context=wm.sim_context,
      task_specs=actor.task_specs,
      eit_adapter=adapter,
    )
    yield options, adapter


def test_native_online_ghost_tracks_current_posterior_without_visualization_resets(
  playback,
):
  options, adapter = playback
  viewer = SuperMimicNativeViewer(**options)
  viewer.reset_environment()
  for expected_time in range(1, 4):
    assert viewer.step_and_predict(), viewer._last_error
    state = adapter.provider.state
    assert state.time_index.tolist() == [expected_time]
    full = state.belief.mean.to_full()[0].numpy()
    np.testing.assert_allclose(viewer.wm_vis_qpos[:7], full[:7])
    np.testing.assert_allclose(viewer.wm_vis_qpos[viewer.joint_q_adr], full[7:36])
    assert not state.belief.covariance.requires_grad

  # A visualization reset must not reset the filter or its measurement history.
  state = adapter.provider.state
  history = adapter.history.clone()
  viewer.reset_wm_visualizer()
  assert adapter.provider.state is state
  assert torch.equal(adapter.history, history)
  assert viewer.step_and_predict(), viewer._last_error
  assert adapter.provider.state.time_index.tolist() == [4]

  drawing = Mock()
  drawing.get_env_indices.return_value = []
  viewer.env.unwrapped.update_visualizers(drawing)
  assert any(
    call.kwargs.get("label") == "online_posterior"
    for call in drawing.add_ghost_mesh.call_args_list
  )


def test_native_pause_single_step_and_real_reset_use_adapter(playback):
  options, adapter = playback
  viewer = SuperMimicNativeViewer(**options)
  viewer.reset_environment()
  viewer.pause()
  assert not viewer.step_and_predict()
  assert adapter.provider.state is None
  assert viewer.step_and_predict(force=True), viewer._last_error
  previous_episode = adapter.episode_id.clone()
  assert adapter.provider.state.time_index.tolist() == [1]
  viewer.reset_environment()
  assert adapter.provider.state is None
  assert torch.equal(
    adapter.history, adapter.history[:, -1:].expand_as(adapter.history)
  )
  assert torch.all(adapter.episode_id > previous_episode)
  assert viewer.step_and_predict(force=True), viewer._last_error
  assert adapter.provider.state.time_index.tolist() == [1]


def test_auto_reset_refreshes_filter_once_at_the_clip_boundary(playback):
  options, adapter = playback
  viewer = SuperMimicNativeViewer(**options)
  viewer.reset_environment()
  initial_episode = adapter.episode_id.clone()
  for _ in range(8):
    assert viewer.step_and_predict(), viewer._last_error
  assert torch.equal(adapter.episode_id, initial_episode + 1)
  assert adapter.provider.state.time_index.tolist() == [0]
  assert torch.equal(
    adapter.history, adapter.history[:, -1:].expand_as(adapter.history)
  )
  assert viewer.step_and_predict(), viewer._last_error
  assert adapter.provider.state.time_index.tolist() == [1]


def test_viser_selected_clip_seeds_filter_after_the_final_reset_frame(playback):
  options, adapter = playback
  viewer = SuperMimicViserViewer(
    **options, motion_names=["first", "second"], viser_server=Mock()
  )
  viewer._selected_clip_idx = 1
  reset = Mock(wraps=adapter.reset)
  adapter.reset = reset
  viewer.reset_environment()
  reset.assert_called_once()
  command = viewer.env.unwrapped.command_manager.get_term("motion")
  assert command.time_steps.item() >= 8
  # The very first measurement must describe the selected clip's final pose,
  # not the random intermediate reset executed by the base viewer.
  torch.testing.assert_close(
    reset.call_args.args[0]["full_state"], viewer.env.get_observations()["full_state"]
  )
  assert viewer.step_and_predict(), viewer._last_error
  assert viewer._ghost_label == "online_posterior"


def test_eit_actor_without_adapter_still_rejects_truth_input_fallback(playback):
  options, _ = playback
  options = dict(options, eit_adapter=None)
  with pytest.raises(NotImplementedError, match="posterior adapter"):
    SuperMimicNativeViewer(**options)


def test_viser_start_here_resets_after_gui_pose_and_refreshes_observations(playback):
  options, adapter = playback
  viewer = SuperMimicViserViewer(
    **options, motion_names=["first", "second"], viser_server=Mock()
  )
  viewer._pending_update_reasons = set()
  viewer._sync_ui_state = Mock()
  command = viewer.env.unwrapped.command_manager.get_term("motion")
  command._scrubber_handles = (SimpleNamespace(value=10),)
  viewer.reset_environment()
  assert viewer.step_and_predict(), viewer._last_error
  previous_episode = adapter.episode_id.clone()
  reset = Mock(wraps=adapter.reset)
  adapter.reset = reset

  viewer._handle_gui_reset(all_envs=True)

  reset.assert_called_once()
  assert command.time_steps.item() == 10
  assert adapter.provider.state is None
  assert adapter.time_index.tolist() == [0]
  assert torch.all(adapter.episode_id > previous_episode)
  assert torch.equal(
    adapter.history, adapter.history[:, -1:].expand_as(adapter.history)
  )
  torch.testing.assert_close(
    reset.call_args.args[0]["full_state"], viewer.env.get_observations()["full_state"]
  )
  assert viewer.wm_vis_qpos is None
  assert viewer.step_and_predict(), viewer._last_error


def test_reference_scrub_reuses_posterior_and_measurement_but_refreshes_goal(playback):
  options, adapter = playback
  viewer = SuperMimicNativeViewer(**options)
  viewer.reset_environment()
  assert viewer.step_and_predict(), viewer._last_error
  command = viewer.env.unwrapped.command_manager.get_term("motion")
  previous_state = adapter.provider.state
  previous_history = adapter.history.clone()
  previous_goal = adapter.last_inputs["goal_motion"].clone()
  # Give the target frame a different joint reference so cached goal reuse is visible.
  command.motion.joint_pos[5] += 0.2
  command.time_steps[:] = 5
  rng = torch.random.get_rng_state()
  actions = adapter.act(viewer.env.get_observations())
  assert torch.isfinite(actions).all()
  assert adapter.provider.state is previous_state
  assert torch.equal(adapter.history, previous_history)
  assert torch.equal(torch.random.get_rng_state(), rng)
  assert not torch.equal(adapter.last_inputs["goal_motion"], previous_goal)


def test_local_eit_play_runs_native_viewer_step_without_wandb(
  playback, monkeypatch, tmp_path
):
  options, adapter = playback
  checkpoint = tmp_path / "training.pt"
  torch.save(
    {
      "version": EIT_TRAINING_VERSION,
      "stage": "policy",
      "replay_mode": "full",
      "critic": BeliefCritic(
        adapter.cfg.estimator_inputs.group,
        adapter.wm.sim_context.output_dim,
        adapter.cfg.critic_hidden_dims,
      ).checkpoint(),
      "runner_config": asdict(adapter.cfg),
      "world_model": adapter.wm.bundle(),
      "actor": actor_checkpoint(adapter.actor, adapter.cfg),
    },
    checkpoint,
  )
  completed = []

  def make_viewer(**kwargs):
    viewer = SuperMimicNativeViewer(**kwargs)

    def run():
      viewer.reset_environment()
      assert viewer.step_and_predict(), viewer._last_error
      assert viewer.eit_adapter.provider.state is not None
      completed.append(viewer._ghost_label)

    return SimpleNamespace(run=run)

  monkeypatch.setattr(play, "SuperMimicNativeViewer", make_viewer)
  with torch.compiler.set_stance("force_eager"):
    play.play(
      "SuperMimic-GenCoord-G1",
      play.PlayConfig(
        eit_checkpoint=checkpoint,
        motion_file=str(tmp_path / "first.npz"),
        device="cpu",
        viewer="native",
      ),
    )
  assert completed == ["online_posterior"]


def test_local_eit_cli_needs_no_wandb():
  task, cfg = play.parse_play_args(
    [TASK, "--eit-checkpoint", "logs/C/training.pt", "--motion-file", "walk.npz"]
  )
  assert task == TASK
  assert cfg.eit_checkpoint == Path("logs/C/training.pt")
  assert not cfg.wandb_path


def test_local_eit_play_rejects_old_checkpoint_before_starting_viewer(tmp_path):
  checkpoint = tmp_path / "training.pt"
  torch.save({"version": "eit_training_v2", "stage": "policy"}, checkpoint)
  cfg = play.PlayConfig(eit_checkpoint=checkpoint, motion_file="walk.npz", device="cpu")
  with pytest.raises(ValueError, match=EIT_TRAINING_VERSION):
    play.play(TASK, cfg)


@pytest.mark.parametrize(
  ("extra", "message"),
  [
    ([], "requires motion_file"),
    (["--motion-file", "walk.npz", "--wandb-path", RUN], "Choose either"),
    (["--motion-file", "walk.npz", "--initial-task", "velocity"], "motion task"),
  ],
)
def test_local_eit_cli_rejects_incompatible_options(extra, message, capsys):
  with pytest.raises(SystemExit):
    play.parse_play_args([TASK, "--eit-checkpoint", "training.pt", *extra])
  assert message in capsys.readouterr().err


def test_local_eit_play_uses_shared_adapter_and_closes_on_viewer_error(
  monkeypatch, tmp_path
):
  cfg = play.PlayConfig(
    eit_checkpoint=tmp_path / "training.pt", motion_file="walk.npz", device="cpu"
  )
  saved = {"version": EIT_TRAINING_VERSION, "stage": "policy", "replay_mode": "full"}
  runner_cfg = SimpleNamespace(representation="gencoord")
  actor, wm, env = Mock(), Mock(), Mock()
  api = Mock(side_effect=AssertionError("Local playback must not contact W&B"))
  monkeypatch.setattr("wandb.Api", api)
  monkeypatch.setattr(play, "configure_torch_backends", Mock())
  monkeypatch.setattr(play, "evaluation_config", Mock(return_value=runner_cfg))
  load = Mock(return_value=saved)
  monkeypatch.setattr(play.torch, "load", load)
  make_env = Mock()
  monkeypatch.setattr(play, "ManagerBasedRlEnv", make_env)
  monkeypatch.setattr(play, "RslRlVecEnvWrapper", Mock(return_value=env))
  models = Mock(return_value=(wm, actor))
  monkeypatch.setattr(play, "load_evaluation_models", models)
  adapter = Mock()
  make_adapter = Mock(return_value=adapter)
  monkeypatch.setattr(play, "EITPolicyAdapter", make_adapter)
  monkeypatch.setattr(play, "create_wm_vis_model", Mock())
  viewer = Mock()
  viewer.return_value.run.side_effect = RuntimeError("viewer stopped")
  monkeypatch.setattr(play, "SuperMimicViserViewer", viewer)

  with pytest.raises(RuntimeError, match="viewer stopped"):
    play.play(TASK, cfg)

  api.assert_not_called()
  load.assert_called_once_with(
    cfg.eit_checkpoint, map_location="cpu", weights_only=True
  )
  models.assert_called_once_with(env, runner_cfg, saved, "cpu")
  make_adapter.assert_called_once_with(env, actor, wm, runner_cfg)
  env_cfg = make_env.call_args.kwargs["cfg"]
  assert env_cfg.observations["actor"].enable_corruption is False
  assert env_cfg.commands["motion"].fixed_reference_alignment
  assert viewer.call_args.kwargs["eit_adapter"] is adapter
  env.close.assert_called_once_with()
