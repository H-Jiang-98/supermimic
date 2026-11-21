"""Reproducible shared initialization and paired evaluation scenarios."""

from contextlib import closing

import torch

from supermimic.models import ProcessNoiseConfig
from supermimic.runners.eit_training import EITTraining
from supermimic.runners.supermimic_runner import SuperMimicRunner, SuperMimicRunnerCfg
from supermimic.scripts.check_eit_training import parse_args
from supermimic.scripts.check_estimator import stationary_motion
from supermimic.scripts.eval_eit import evaluate
from supermimic.scripts.train_utils import create_env
from supermimic.tasks.env_cfgs import configure_estimator_env_cfg, gencoord_env_cfg
from tests.test_estimator_eit_training import training


def test_acceptance_policy_horizon_flags_select_the_same_option():
  assert parse_args(["--horizon", "2"]).policy_horizon == 2
  assert parse_args(["--policy-horizon", "2"]).policy_horizon == 2
  assert parse_args([]).policy_horizon == 4


def test_creating_and_reusing_shared_initialization_leave_same_training_rng():
  first = training(31)
  initial = first.initialization_bundle()
  first.load_initialization(initial)
  created = torch.random.get_rng_state()

  second = training(31)
  second.load_initialization(initial)
  assert torch.equal(created, torch.random.get_rng_state())


def test_eval_seed_pairs_first_reset_and_measurement_across_actor_architectures(
  tmp_path,
):
  motion = tmp_path / "motion.npz"
  stationary_motion(motion)
  results = []
  with torch.compiler.set_stance("force_eager"):
    for group in ("B", "C"):
      env_cfg = configure_estimator_env_cfg(gencoord_env_cfg())
      env_cfg.seed = 2026
      env_cfg.commands["motion"].motion_file = str(motion)
      env_cfg.commands["motion"].sampling_mode = "uniform"
      cfg = SuperMimicRunnerCfg.for_estimator_inputs(
        group,
        dynamics_hidden_dims=(16,),
        actor_hidden_dims=(16,),
        critic_hidden_dims=(16,),
        process_noise=ProcessNoiseConfig(hidden_dims=(8,)),
        buffer_size=4,
        sampling_mode="uniform",
      )
      checkpoint = tmp_path / f"{group}.pt"
      with closing(create_env(env_cfg, 2, "cpu")) as env:
        runner = SuperMimicRunner(env, cfg, "SuperMimic-GenCoord-G1", device="cpu")
        EITTraining(runner).save(checkpoint)
      results.append(evaluate(checkpoint, motion, steps=1, num_envs=2, seed=808))
  first, second = results
  assert first["pre_action_tracking_rmse"] == second["pre_action_tracking_rmse"]
  assert first["posterior_rmse"] == second["posterior_rmse"]
