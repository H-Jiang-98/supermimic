"""Production graph dispatch, accumulation, ownership and rejection boundaries."""

from copy import deepcopy
from dataclasses import replace
from unittest.mock import Mock

import pytest
import torch

from supermimic.estimator import FilterConfig
from supermimic.learners.belief_policy_learner import (
  BeliefActorCriticPolicyLearner,
  BeliefPolicyLearnerCfg,
)
from supermimic.models import ProcessNoiseConfig
from supermimic.runners.eit_training import EITTraining
from supermimic.runners.supermimic_runner import SuperMimicRunner, SuperMimicRunnerCfg
from supermimic.scripts.check_estimator import stationary_motion
from supermimic.scripts.train_eit import (
  configure_estimator_env_cfg,
  create_env,
  load_env_cfg,
)


@pytest.fixture
def graph_policy(tmp_path):
  if not torch.cuda.is_available():
    pytest.skip("CUDA actor replay regression")
  environments = []

  def make(group="C", backend="increment", compiled=False, return_training=False):
    torch.manual_seed(2027)
    motion = tmp_path / "stationary.npz"
    stationary_motion(motion)
    cfg = load_env_cfg("SuperMimic-GenCoord-G1")
    cfg.seed = 2027
    cfg.commands["motion"].motion_file = str(motion)
    cfg.commands["motion"].sampling_mode = "start"
    configure_estimator_env_cfg(cfg)
    env = create_env(cfg, 2, "cuda:0")
    environments.append(env)
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
      env, cfg, "SuperMimic-GenCoord-G1", device="cuda:0", compile_eit=compiled
    )
    training = EITTraining(
      runner,
      policy_cfg=BeliefPolicyLearnerCfg(
        batch_size=2,
        num_samples=2,
        microbatch_size=2,
        horizon=2,
        critic_epochs=1,
        critic_minibatches=2,
      ),
      policy_cuda_graphs=True,
    )
    training.run_step()
    policy = training.policy
    starts = runner.wm_env.prepare_policy_starts(
      torch.arange(2, device="cuda:0"),
      torch.zeros(2, device="cuda:0", dtype=torch.long),
    )
    return (training if return_training else policy), starts

  yield make
  for env in environments:
    env.close()


def reference_for(policy, compiled=False):
  reference = BeliefActorCriticPolicyLearner(
    deepcopy(policy.actor),
    deepcopy(policy.critic),
    policy.wm_env,
    policy.cfg,
    compile_kernels=compiled,
  )
  reference.load_state_dict(policy.state_dict())
  return reference


def assert_policy_close(actual, expected):
  assert_tree_close(
    (actual.actor.state_dict(), actual.critic.state_dict(), actual.state_dict()),
    (expected.actor.state_dict(), expected.critic.state_dict(), expected.state_dict()),
    atol=2e-5,
    rtol=1e-3,
  )
  torch.testing.assert_close(
    [p.grad for p in actual.actor.parameters()],
    [p.grad for p in expected.actor.parameters()],
    atol=2e-5,
    rtol=1e-3,
  )


def assert_tree_close(actual, expected, *, atol, rtol, equal_nan=False):
  left, left_spec = torch.utils._pytree.tree_flatten(actual)
  right, right_spec = torch.utils._pytree.tree_flatten(expected)
  assert left_spec == right_spec
  for a, b in zip(left, right, strict=True):
    if isinstance(a, torch.Tensor):
      torch.testing.assert_close(a, b, atol=atol, rtol=rtol, equal_nan=equal_nan)
    else:
      assert a == b


@pytest.mark.parametrize("group", ["A", "B", "C"])
@pytest.mark.parametrize("backend", ["vmap", "increment"])
def test_graph_updates_match_eager_and_stop_python_rollout(
  graph_policy, group, backend
):
  policy, starts = graph_policy(group, backend)
  reference = reference_for(policy)
  policy._rollout_kernel = Mock(wraps=policy._rollout_kernel)
  model_before = deepcopy(policy.wm_env.dynamics.state_dict())
  provider = policy.wm_env.posterior_provider
  provider_before = provider.state, provider.generation
  for step in range(4):
    # Valid changing inputs must be copied into fixed graph storage.
    starts.batch["sensor_window"][:, :, 0].add_(0.001)
    noises = policy.draw_noises(starts)
    torch.testing.assert_close(noises, reference.draw_noises(starts))
    actual = policy.update_from(starts, *noises)
    expected = reference.update_from(starts, *noises)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-3, equal_nan=True)
    assert_policy_close(policy, reference)
    if step == 1:
      assert policy._actor_graph.captures == 1
      addresses = [p.grad.data_ptr() for p in policy.actor.parameters()]
    if step > 1:
      assert addresses == [p.grad.data_ptr() for p in policy.actor.parameters()]
    # These live buffers can change between updates without recapture.
    with torch.no_grad():
      for learner in (policy, reference):
        learner.actor.obs_normalizer._mean.add_(0.0001)
        next(learner.critic.parameters()).add_(0.00001)
  assert policy._rollout_kernel.call_count == 4  # 3 warmups + capture, no Python replay
  assert policy._actor_graph.replay_calls == 5
  assert policy.actor_steps == 4 and policy.critic_steps == 8
  torch.testing.assert_close(policy.wm_env.dynamics.state_dict(), model_before)
  assert (
    provider.state is provider_before[0] and provider.generation == provider_before[1]
  )


@pytest.mark.filterwarnings("error:The AccumulateGrad node")
def test_compiled_graph_restore_and_tail_fallback(graph_policy):
  policy, starts = graph_policy(compiled=True)
  reference = reference_for(policy, compiled=True)
  for _ in range(3):
    noises = policy.draw_noises(starts)
    reference.draw_noises(starts)
    policy.update_from(starts, *noises)
    reference.update_from(starts, *noises)
  assert_policy_close(policy, reference)
  saved = policy.state_dict()
  captured = policy._actor_graph
  broken = deepcopy(saved)
  broken["actor_steps"] = -1
  with pytest.raises(ValueError):
    policy.load_state_dict(broken)
  assert policy._actor_graph is captured
  del captured  # Do not keep the old graph/AccumulateGrad alive across restoration.
  policy.load_state_dict(saved)
  assert policy._actor_graph is None
  for _ in range(3):
    noises = policy.draw_noises(starts)
    reference.draw_noises(starts)
    policy.update_from(starts, *noises)
    reference.update_from(starts, *noises)
  assert_policy_close(policy, reference)
  assert policy._actor_graph.captures == 1
  # Expanded N=4 has a tail for M=3: release the graph and use the original path.
  policy.cfg = reference.cfg = replace(policy.cfg, microbatch_size=3)
  noises = policy.draw_noises(starts)
  reference.draw_noises(starts)
  policy.update_from(starts, *noises)
  reference.update_from(starts, *noises)
  assert policy.graph_status["reason"] == "tail_microbatch"
  assert policy._actor_graph is None
  assert_policy_close(policy, reference)


@pytest.mark.parametrize("failure", ["rollout", "gradient", "storage", "stale"])
def test_replay_failure_precedes_optimizer_and_drops_cache(graph_policy, failure):
  policy, starts = graph_policy()
  gate = torch.ones((), device="cuda:0")
  handle = next(policy.actor.parameters()).register_hook(
    lambda gradient: gradient * gate
  )
  for _ in range(3):
    policy.update_from(starts, *policy.draw_noises(starts))
  assert policy._actor_graph.captures == 1
  if failure == "rollout":
    with torch.no_grad():
      policy.actor.decoder[-1].bias.fill_(float("inf"))
  elif failure == "gradient":
    gate.fill_(float("inf"))
  elif failure == "storage":
    policy.actor.obs_normalizer._mean = policy.actor.obs_normalizer._mean.clone()
  else:
    starts = replace(starts, insert_count=starts.insert_count - 1)
  before = deepcopy(
    (policy.actor.state_dict(), policy.critic.state_dict(), policy.state_dict())
  )
  with pytest.raises((FloatingPointError, RuntimeError, ValueError)):
    policy.update_from(starts, *policy.draw_noises(starts))
  after = policy.actor.state_dict(), policy.critic.state_dict(), policy.state_dict()
  # Drawing caller-owned noises is allowed to advance only their RNG streams.
  before[-1]["rng"] = after[-1]["rng"]
  assert_tree_close(after, before, atol=0, rtol=0, equal_nan=True)
  assert policy._actor_graph is None
  assert all(p.grad is None for p in policy.actor.parameters())
  handle.remove()


def test_cpu_request_keeps_existing_update_path():
  from tests.test_estimator_belief_policy import setup

  policy, starts, noises = setup()
  policy.cuda_graphs = True
  policy.update_from(starts, *noises)
  assert policy.graph_status == {"requested": True, "reason": "cpu"}


def test_collection_and_normalization_between_updates_reuse_graph(graph_policy):
  training, starts = graph_policy(return_training=True)
  policy = training.policy
  for _ in range(3):
    policy.update_from(starts, *policy.draw_noises(starts))
  captured = policy._actor_graph
  aux_before = policy.actor._aux_loss
  training.run_step()
  assert policy.actor._aux_loss is not aux_before
  model = policy.wm_env.dynamics
  model.begin_write()
  with torch.no_grad():
    next(model.parameters()).add_(0.0001)
  model.finish_write()
  fresh = policy.wm_env.prepare_policy_starts(
    torch.arange(2, device="cuda:0"), torch.zeros(2, device="cuda:0", dtype=torch.long)
  )
  reference = reference_for(policy)
  noises = policy.draw_noises(fresh)
  reference.draw_noises(fresh)
  policy.update_from(fresh, *noises)
  reference.update_from(fresh, *noises)
  assert_policy_close(policy, reference)
  assert policy._actor_graph is captured and captured.captures == 1
