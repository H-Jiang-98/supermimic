"""Evaluation/playback share the production measurement and posterior semantics."""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, replace

import pytest
import torch

from supermimic.critic_inputs import CRITIC_INPUT_VERSION, BeliefCritic
from supermimic.models import ProcessNoiseConfig
from supermimic.posterior_replay import POSTERIOR_REPLAY_VERSION
from supermimic.runners.eit_inference import (
  EITPolicyAdapter,
  evaluation_config,
  load_evaluation_models,
)
from supermimic.runners.run_config import (
  EIT_POLICY_TRAINING_VERSION,
  EIT_TRAINING_VERSION,
  validate_eit_checkpoint_contract,
)
from supermimic.runners.supermimic_runner import (
  SuperMimicRunner,
  SuperMimicRunnerCfg,
  actor_checkpoint,
)
from supermimic.scripts.check_estimator import stationary_motion
from supermimic.scripts.eval_eit import evaluate
from supermimic.scripts.train_utils import create_env
from supermimic.tasks.env_cfgs import configure_estimator_env_cfg, gencoord_env_cfg


@contextmanager
def physical_runner(tmp_path, group="C"):
  motion = tmp_path / "motion.npz"
  stationary_motion(motion)
  env_cfg = configure_estimator_env_cfg(gencoord_env_cfg())
  env_cfg.commands["motion"].motion_file = str(motion)
  env_cfg.commands["motion"].sampling_mode = "start"
  cfg = SuperMimicRunnerCfg.for_estimator_inputs(
    group,
    dynamics_hidden_dims=(12,),
    actor_hidden_dims=(12,),
    critic_hidden_dims=(12,),
    process_noise=ProcessNoiseConfig(hidden_dims=(8,)),
    buffer_size=8,
    sampling_mode="start",
  )
  cfg.estimator_inputs = replace(cfg.estimator_inputs, mode="eval")
  cfg.policy.noise = 0.0
  with torch.compiler.set_stance("force_eager"):
    env = create_env(env_cfg, 2, "cpu")
    try:
      runner = SuperMimicRunner(env, cfg, "SuperMimic-GenCoord-G1", device="cpu")
      runner.policy_learner.actor.eval()
      runner.configure_belief_filter()
      yield runner
    finally:
      env.close()


def runner_action(runner, obs):
  representation = runner._representation
  inputs = runner._build_sim_obs_td(
    representation.represent(obs["full_state"]),
    representation.extract_aux_state(obs["full_state"]),
    obs["actor"],
    runner.motion_cmd.time_steps,
    runner._sim_context.read(runner.env.unwrapped),
  )
  return inputs, runner.wm_env.limit_actions(runner.policy_learner.actor(inputs))


@pytest.mark.parametrize("group", ["A", "B", "C"])
def test_adapter_matches_collection_features_posterior_and_actions_across_reset(
  tmp_path, group
):
  with physical_runner(tmp_path, group) as runner, torch.inference_mode():
    actor = runner.policy_learner.actor
    adapter = EITPolicyAdapter(runner.env, actor, runner._wm, runner.cfg)
    before = deepcopy(actor.state_dict())
    obs, _ = runner.env.reset()
    rng = torch.random.get_rng_state()
    runner._seed_sim_history()
    expected_inputs, expected_actions = runner_action(runner, obs)
    expected_rng = torch.random.get_rng_state()
    torch.random.set_rng_state(rng)
    adapter.reset(obs)
    actions = adapter.act(obs)

    for step in range(3):
      assert torch.equal(torch.random.get_rng_state(), expected_rng)
      assert torch.equal(actions, expected_actions)
      assert adapter.last_inputs.keys() == expected_inputs.keys()
      for key in expected_inputs.keys():
        torch.testing.assert_close(
          adapter.last_inputs[key], expected_inputs[key], atol=0, rtol=0
        )
      expected = runner.posterior_provider.state
      actual = adapter.provider.state
      torch.testing.assert_close(
        actual.belief.mean.to_full(), expected.belief.mean.to_full(), atol=0, rtol=0
      )
      torch.testing.assert_close(
        actual.belief.covariance, expected.belief.covariance, atol=0, rtol=0
      )
      assert not actual.belief.covariance.requires_grad
      # Repeated UI requests at the same frame consume neither noise nor updates.
      generation = adapter.provider.generation
      assert torch.equal(adapter.act(obs), actions)
      assert torch.equal(torch.random.get_rng_state(), expected_rng)
      assert adapter.provider.generation == generation
      if step == 2:
        break
      if step == 0:
        base = runner.env.unwrapped
        base.episode_length_buf[0] = base.max_episode_length - 1
      obs, _, dones, _ = runner.env.step(actions)
      if step == 0:
        assert dones[0] and not dones[1]
      rng = torch.random.get_rng_state()
      runner._advance_sim_history(obs["actor"], dones)
      expected_inputs, expected_actions = runner_action(runner, obs)
      expected_rng = torch.random.get_rng_state()
      torch.random.set_rng_state(rng)
      adapter.advance(obs, dones)
      if step == 0:
        assert adapter.episode_id.tolist() == [2, 1]
        assert adapter.time_index.tolist() == [0, 1]
        assert torch.equal(
          adapter.history[0], adapter.history[0, :1].expand_as(adapter.history[0])
        )
      actions = adapter.act(obs)
    for key, value in before.items():
      assert torch.equal(value, actor.state_dict()[key])


def test_adapter_reset_does_not_reset_physics_and_rejects_skipped_frames(tmp_path):
  with physical_runner(tmp_path) as runner, torch.no_grad():
    adapter = EITPolicyAdapter(
      runner.env, runner.policy_learner.actor, runner._wm, runner.cfg
    )
    obs, _ = runner.env.reset()
    state = obs["full_state"].clone()
    adapter.reset(obs)
    assert torch.equal(runner.env.get_observations()["full_state"], state)
    with pytest.raises(RuntimeError, match="act"):
      adapter.advance(obs, torch.zeros(2, dtype=torch.bool))
    actions = adapter.act()
    next_obs, _, dones, _ = runner.env.step(actions)
    with pytest.raises(ValueError, match="advance"):
      adapter.act(next_obs)
    adapter.advance(next_obs, dones)
    with pytest.raises(RuntimeError, match="act"):
      adapter.advance(next_obs, dones)
    adapter.act()
    generation = adapter.provider.generation
    adapter.reset(next_obs)
    assert adapter.provider.state is None and adapter.provider.generation > generation
    assert torch.equal(
      runner.env.get_observations()["full_state"], next_obs["full_state"]
    )
    adapter.act()
    assert adapter.time_index.tolist() == [0, 0]


@pytest.mark.parametrize("legacy", [False, True])
def test_evaluation_loader_is_private_and_checks_estimator_contract(tmp_path, legacy):
  with physical_runner(tmp_path, "B") as runner:
    saved = {
      "version": EIT_POLICY_TRAINING_VERSION,
      "stage": "policy",
      "replay_mode": "full",
      "posterior_replay_protocol": POSTERIOR_REPLAY_VERSION,
      "loop_config": {"posterior_save_stride": 1},
      "model_revision": runner._dynamics.model_revision,
      "critic": BeliefCritic(
        "B", runner._sim_context.output_dim, runner.cfg.critic_hidden_dims
      ).checkpoint(),
      "runner_config": asdict(runner.cfg),
      "world_model": deepcopy(runner._wm.bundle()),
      "actor": deepcopy(actor_checkpoint(runner.policy_learner.actor, runner.cfg)),
    }
    if legacy:
      saved["version"] = "eit_training_v5"
      contract = saved["critic"]["contract"]
      contract["version"] = contract["config"]["version"] = "eit_rpv_belief_critic_v2"
      contract["config"]["remaining_time_scale"] = 1.0
      contract["input_dim"] += 1
      weights = saved["critic"]["state_dict"]
      weights["mlp.0.weight"] = torch.nn.functional.pad(weights["mlp.0.weight"], (0, 1))
      with pytest.raises(ValueError, match=EIT_POLICY_TRAINING_VERSION):
        validate_eit_checkpoint_contract(saved, stage="policy")
    saved["runner_config"]["wandb_run_path"] = "unused/source/run"
    saved["runner_config"]["estimator_inputs"]["mode"] = "train"
    saved["actor"]["input_config"]["mode"] = "train"
    cfg = evaluation_config(saved)
    assert cfg.estimator_inputs.mode == "eval" and cfg.policy.noise == 0
    assert cfg.wandb_run_path is None
    assert saved["runner_config"]["estimator_inputs"]["mode"] == "train"
    wm, actor = load_evaluation_models(runner.env, cfg, saved, "cpu")
    assert wm.dynamics.model_version == runner._dynamics.model_version
    for name, tensor in runner.policy_learner.actor.state_dict().items():
      assert torch.equal(actor.state_dict()[name], tensor)
      assert actor.state_dict()[name].data_ptr() != tensor.data_ptr()
    assert all(not p.requires_grad for p in actor.parameters())
    assert all(not p.requires_grad for p in wm.dynamics.parameters())
    malformed = deepcopy(saved)
    prior = malformed["world_model"]["estimator"]["initial_prior"]
    prior["std"] = tuple(2 * value for value in prior["std"])
    with pytest.raises(ValueError, match="estimator settings"):
      load_evaluation_models(runner.env, cfg, malformed, "cpu")


@pytest.mark.parametrize("version", ["eit_training_v1", "eit_training_v2"])
@pytest.mark.parametrize("stage", ["model", "policy"])
def test_evaluation_rejects_old_checkpoint_before_starting_environment(
  tmp_path, version, stage
):
  checkpoint = tmp_path / "training.pt"
  torch.save({"version": version, "stage": stage}, checkpoint)
  with pytest.raises(ValueError, match=EIT_TRAINING_VERSION):
    evaluate(checkpoint, tmp_path / "motion.npz")


@pytest.mark.parametrize("replay_mode", ["model", "full"])
def test_model_checkpoint_contract_accepts_training_or_diagnostic_replay(replay_mode):
  validate_eit_checkpoint_contract(
    {"version": EIT_TRAINING_VERSION, "stage": "model", "replay_mode": replay_mode}
  )


@pytest.mark.parametrize(
  ("stage", "replay_mode"),
  [("model", None), ("model", "other"), ("policy", None), ("policy", "model")],
)
def test_inference_rejects_missing_or_invalid_replay_contract(stage, replay_mode):
  version = EIT_POLICY_TRAINING_VERSION if stage == "policy" else EIT_TRAINING_VERSION
  saved = {"version": version, "stage": stage}
  if replay_mode is not None:
    saved["replay_mode"] = replay_mode
  with pytest.raises(ValueError, match="Invalid replay mode"):
    evaluation_config(saved)


@pytest.mark.parametrize("group", ["A", "B", "C"])
@pytest.mark.parametrize(
  "fault",
  [
    "wrong_version",
    "missing_contract",
    "wrong_group",
    "wrong_encoding",
    "wrong_context",
    "wrong_weights",
  ],
)
def test_inference_rejects_incompatible_critic_before_building_models(group, fault):
  cfg = SuperMimicRunnerCfg.for_estimator_inputs(
    group, process_noise=ProcessNoiseConfig(), critic_hidden_dims=(8,)
  )
  cfg.estimator_inputs = replace(cfg.estimator_inputs, mode="eval")
  context_dim = 33
  critic = BeliefCritic(group, context_dim, cfg.critic_hidden_dims).checkpoint()
  contract = critic["contract"]
  saved = {
    "version": EIT_POLICY_TRAINING_VERSION,
    "stage": "policy",
    "replay_mode": "full",
    "posterior_replay_protocol": POSTERIOR_REPLAY_VERSION,
    "loop_config": {"posterior_save_stride": 1},
    "model_revision": 0,
    "runner_config": asdict(cfg),
    "critic": critic,
    "world_model": {
      "sim_context": {"scale": torch.ones(context_dim)},
      "dynamics": {"model_revision": 0},
    },
  }
  evaluation_config(saved)
  error = CRITIC_INPUT_VERSION
  if fault == "wrong_version":
    contract["version"] = contract["config"]["version"] = "unknown_critic_version"
  elif fault == "missing_contract":
    del saved["critic"]["contract"]
  elif fault == "wrong_group":
    contract["config"]["group"] = "B" if group == "A" else "A"
  elif fault == "wrong_encoding":
    contract["config"]["covariance_encoding"] = "unknown_encoding"
  elif fault == "wrong_context":
    saved["critic"] = BeliefCritic(
      group, context_dim + 1, cfg.critic_hidden_dims
    ).checkpoint()
    error = f"expects {context_dim + 1} context channels"
  else:
    state = critic["state_dict"]
    state["mlp.0.weight"] = state["mlp.0.weight"][:, :-1]
    error = f"expects {contract['input_dim']} inputs"
  with pytest.raises(ValueError, match=error):
    evaluation_config(saved)
  with pytest.raises(ValueError, match=error):
    load_evaluation_models(None, cfg, saved, "cpu")


@pytest.mark.parametrize("reference_change", ["time", "alignment"])
def test_reference_scrub_refreshes_goal_without_refusing_the_measurement(
  tmp_path, reference_change
):
  with physical_runner(tmp_path) as runner, torch.no_grad():
    adapter = EITPolicyAdapter(
      runner.env, runner.policy_learner.actor, runner._wm, runner.cfg
    )
    obs, _ = runner.env.reset()
    # Give this synthetic clip a distinct future frame so a time scrub changes
    # the goal while the robot and its measured frame remain unchanged.
    runner.motion.joint_pos[1] += 0.2
    actions = adapter.act(obs)
    inputs = adapter.last_inputs.clone()
    belief = adapter.provider.state
    generation = adapter.provider.generation
    rng = torch.random.get_rng_state()
    if reference_change == "time":
      runner.motion_cmd.time_steps.add_(1)
    else:
      runner.motion_cmd.reference_alignment.translation[:, 0] += 0.3
    changed = adapter.act(obs)
    assert adapter.provider.state is belief
    assert adapter.provider.generation == generation
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert torch.equal(adapter.last_inputs["history"], inputs["history"])
    assert not torch.equal(adapter.last_inputs["goal_motion"], inputs["goal_motion"])
    assert not torch.equal(changed, actions)
    assert torch.equal(adapter.act(obs), changed)
    assert torch.equal(torch.random.get_rng_state(), rng)


@pytest.mark.parametrize("change", ["model", "context"])
def test_cached_action_rejects_changed_model_or_context(tmp_path, change, monkeypatch):
  with physical_runner(tmp_path) as runner, torch.no_grad():
    adapter = EITPolicyAdapter(
      runner.env, runner.policy_learner.actor, runner._wm, runner.cfg
    )
    adapter.act()
    if change == "model":
      next(runner._dynamics.parameters()).add_(0.1)
    else:
      context = runner._sim_context.read(runner.env.unwrapped) + 0.1
      monkeypatch.setattr(runner._sim_context, "read", lambda _: context)
    with pytest.raises(ValueError, match="version or context"):
      adapter.act()
