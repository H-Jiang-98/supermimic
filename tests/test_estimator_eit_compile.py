"""Fixed-input CUDA parity for EIT policy gradients through the compiled filter."""

from pathlib import Path

import pytest
import torch
from torch._dynamo.utils import get_compilation_metrics

from supermimic.estimator import FilterConfig
from supermimic.learners.belief_policy_learner import BeliefPolicyLearnerCfg
from supermimic.models import ProcessNoiseConfig
from supermimic.runners.eit_training import EITTraining
from supermimic.runners.supermimic_runner import SuperMimicRunner, SuperMimicRunnerCfg
from supermimic.scripts.check_estimator import stationary_motion
from supermimic.scripts.train_eit import (
  configure_estimator_env_cfg,
  create_env,
  load_env_cfg,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Inductor regression")
@pytest.mark.parametrize("group", ["A", "B", "C"])
@pytest.mark.parametrize("backend", ["vmap", "increment"])
def test_fixed_policy_rollout_compiled(group, backend, tmp_path):
  device = "cuda:0"
  torch.manual_seed(2027)
  motion = tmp_path / "stationary.npz"
  stationary_motion(motion)
  cfg = load_env_cfg("SuperMimic-GenCoord-G1")
  cfg.seed = 2027
  cfg.commands["motion"].motion_file = str(motion)
  cfg.commands["motion"].sampling_mode = "start"
  configure_estimator_env_cfg(cfg)
  env = create_env(cfg, 2, device)
  try:
    cfg = SuperMimicRunnerCfg.for_estimator_inputs(
      group,
      process_noise=ProcessNoiseConfig(hidden_dims=(8,)),
      filter_config=FilterConfig(jacobian_backend=backend),
      dynamics_hidden_dims=(16,),
      actor_hidden_dims=(16, 8),
      critic_hidden_dims=(16, 8),
      replay_ratio=0,
      buffer_size=8,
      sampling_mode="start",
    )
    cfg.policy.batch_size = 2
    runner = SuperMimicRunner(
      env, cfg, "SuperMimic-GenCoord-G1", device=device, compile_eit=True
    )
    training = EITTraining(
      runner,
      policy_cfg=BeliefPolicyLearnerCfg(batch_size=2, horizon=2, microbatch_size=2),
    )
    training.run_step()
    world, policy = runner.wm_env, training.policy
    runner._dynamics.requires_grad_(False)
    policy.critic.requires_grad_(False)
    if policy.actor.film is not None:
      # Exercise the covariance-to-action derivative even at initialization.
      with torch.no_grad():
        policy.actor.film[-1].weight.normal_(std=0.01)
    start = world.prepare_policy_starts(
      torch.arange(2, device=device), torch.zeros(2, device=device, dtype=torch.long)
    )
    tail = start.index(slice(0, 1))
    tail.batch["motion_time"] = world.motion.last_frame(tail.motion_times)

    for inputs in (start, tail):
      noises = [noise[:, :, 0] for noise in policy.draw_noises(inputs)]
      results = []
      for stance in ("force_eager", "default"):
        with torch.compiler.set_stance(stance):
          returns, _, _ = policy._rollout(inputs, *noises, policy.cfg.horizon)
          posterior = world.posterior_provider.state.belief
          gradients = torch.autograd.grad(
            returns.sum(), tuple(policy.actor.parameters()), allow_unused=True
          )
          results.append(
            (
              returns.detach(),
              posterior.mean.to_full().detach(),
              posterior.covariance.detach(),
              [g.detach() if g is not None else None for g in gradients],
            )
          )
          world.clear_rollout()
      torch.testing.assert_close(results[1], results[0], atol=2e-5, rtol=1e-3)
    graphs = {
      (Path(metric.co_filename).name, metric.co_name)
      for metric in get_compilation_metrics()
      if metric.co_filename and metric.graph_op_count
    }
    assert {
      ("models.py", "forward"),
      ("estimator.py", "_vector_jacobian"),
      ("estimator.py", "_propagate_covariance"),
    } <= graphs
  finally:
    env.close()
