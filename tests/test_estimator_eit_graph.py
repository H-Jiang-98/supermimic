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
from supermimic.learners.two_stage_wm import CovarianceFitCfg, MeanFitCfg
from supermimic.models import ProcessNoiseConfig
from supermimic.runners.eit_training import EITTraining, EITTrainingCfg
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

  def make(
    group="belief",
    backend="increment",
    compiled=False,
    return_training=False,
    integrated=False,
  ):
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
      actor_hidden_dims=(16, 12, 8),
      critic_hidden_dims=(16, 12, 8),
      replay_ratio=3 if integrated else 0,
      mean=MeanFitCfg(num_batches=1, batch_size=2, window_frames=3),
      cov=CovarianceFitCfg(batch_size=4, pair_budget=4, validation_pairs=4),
      buffer_size=8,
      sampling_mode="start",
    )
    cfg.policy.batch_size = 2
    runner = SuperMimicRunner(
      env, cfg, "SuperMimic-GenCoord-G1", device="cuda:0", compile_eit=compiled
    )
    training = EITTraining(
      runner,
      EITTrainingCfg(
        validation_stride=2, validation_batch_size=2, posterior_save_stride=1
      ),
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


@pytest.mark.parametrize("group", ["mean", "belief"])
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
  policy._policy_update = Mock(wraps=policy._policy_update)
  optimizer_calls = []
  for optimizer in (*policy.actor_optimizers, *policy.critic_optimizers):
    optimizer.step = Mock(wraps=optimizer.step)
    optimizer_calls.append(optimizer.step)
  previous_metrics = None
  for step in range(6):
    # Valid changing inputs must be copied into fixed graph storage.
    starts.batch["sensor_window"][:, :, 0].add_(0.001)
    noises = policy.draw_noises(starts)
    torch.testing.assert_close(noises, reference.draw_noises(starts))
    actual = policy.update_from(starts, *noises)
    expected = reference.update_from(starts, *noises)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-3, equal_nan=True)
    assert_policy_close(policy, reference)
    if previous_metrics is not None:
      torch.testing.assert_close(
        previous_metrics[0], previous_metrics[1], equal_nan=True
      )
    previous_metrics = actual, deepcopy(actual)
    if step == 3:
      assert policy._policy_graph.captures == 1
      addresses = [p.grad.data_ptr() for p in policy.actor.parameters()]
    if step > 3:
      assert addresses == [p.grad.data_ptr() for p in policy.actor.parameters()]
    # These live buffers can change between updates without recapture.
    with torch.no_grad():
      for learner in (policy, reference):
        learner.actor.obs_normalizer._mean.add_(0.0001)
        next(learner.critic.parameters()).add_(0.00001)
  assert policy._policy_update.call_count == 4  # Three whole updates plus capture.
  assert policy._rollout_kernel.call_count == 8  # Two microbatches per Python call.
  assert [call.call_count for call in optimizer_calls] == [4, 4, 8, 8]
  assert policy._policy_graph.replay_calls == 3
  assert policy.actor_steps == 6 and policy.critic_steps == 12
  torch.testing.assert_close(policy.wm_env.dynamics.state_dict(), model_before)
  assert (
    provider.state is provider_before[0] and provider.generation == provider_before[1]
  )


@pytest.mark.filterwarnings("error:The AccumulateGrad node")
def test_compiled_graph_restore_and_tail_fallback(graph_policy):
  policy, starts = graph_policy(compiled=True)
  reference = reference_for(policy, compiled=True)
  for _ in range(4):
    noises = policy.draw_noises(starts)
    reference.draw_noises(starts)
    policy.update_from(starts, *noises)
    reference.update_from(starts, *noises)
  assert_policy_close(policy, reference)
  saved = policy.state_dict()
  captured = policy._policy_graph
  broken = deepcopy(saved)
  broken["actor_steps"] = -1
  with pytest.raises(ValueError):
    policy.load_state_dict(broken)
  assert policy._policy_graph is captured
  del captured  # Do not keep the old graph/AccumulateGrad alive across restoration.
  policy.load_state_dict(saved)
  assert policy._policy_graph is None
  for _ in range(4):
    noises = policy.draw_noises(starts)
    reference.draw_noises(starts)
    policy.update_from(starts, *noises)
    reference.update_from(starts, *noises)
  assert_policy_close(policy, reference)
  assert policy._policy_graph.captures == 1
  # Expanded N=4 has a tail for M=3: release the graph and use the original path.
  policy.cfg = reference.cfg = replace(policy.cfg, microbatch_size=3)
  noises = policy.draw_noises(starts)
  reference.draw_noises(starts)
  policy.update_from(starts, *noises)
  reference.update_from(starts, *noises)
  assert policy.graph_status["reason"] == "tail_microbatch"
  assert policy._policy_graph is None
  assert_policy_close(policy, reference)

  policy.cfg = replace(policy.cfg, microbatch_size=2)
  # Isolate re-entry from accumulated float32 drift in the preceding nine updates.
  reference = reference_for(policy, compiled=True)
  for _ in range(4):
    noises = policy.draw_noises(starts)
    reference.draw_noises(starts)
    policy.update_from(starts, *noises)
    reference.update_from(starts, *noises)
  assert policy._policy_graph.captures == 1
  assert_policy_close(policy, reference)


@pytest.mark.parametrize(
  "failure",
  [
    "rollout",
    "gradient",
    "critic_gradient",
    "overflow",
    "storage",
    "optimizer_storage",
    "stale",
    "noise",
  ],
)
def test_replay_failure_boundaries_and_cache_cleanup(graph_policy, failure):
  policy, starts = graph_policy()
  gate = torch.ones((), device="cuda:0")
  module = policy.critic if failure == "critic_gradient" else policy.actor
  handle = next(module.parameters()).register_hook(
    (lambda gradient: torch.where(gate > 1, torch.full_like(gradient, 1e20), gradient))
    if failure == "overflow"
    else (lambda gradient: gradient * gate)
  )
  for _ in range(4):
    policy.update_from(starts, *policy.draw_noises(starts))
  assert policy._policy_graph.captures == 1
  if failure == "rollout":
    with torch.no_grad():
      policy.actor.decoder[-1].bias.fill_(float("inf"))
  elif failure in ("gradient", "critic_gradient"):
    gate.fill_(float("inf"))
  elif failure == "overflow":
    gate.fill_(2)
  elif failure == "storage":
    policy.actor.obs_normalizer._mean = policy.actor.obs_normalizer._mean.clone()
  elif failure == "optimizer_storage":
    state = next(iter(policy.actor_optimizers[0].state.values()))
    state["exp_avg"] = state["exp_avg"].clone()
  elif failure == "stale":
    starts = replace(starts, insert_count=starts.insert_count - 1)
  noises = policy.draw_noises(starts)
  if failure == "noise":
    noises[1].fill_(float("nan"))
  before = deepcopy(
    (policy.actor.state_dict(), policy.critic.state_dict(), policy.state_dict())
  )
  with pytest.raises((FloatingPointError, RuntimeError, ValueError)):
    policy.update_from(starts, *noises)
  after = policy.actor.state_dict(), policy.critic.state_dict(), policy.state_dict()
  # Drawing caller-owned noises is allowed to advance only their RNG streams.
  before[-1]["rng"] = after[-1]["rng"]
  if failure in ("rollout", "gradient", "critic_gradient", "overflow"):
    assert after[-1]["actor_steps"] == before[-1]["actor_steps"] + 1
    assert after[-1]["critic_steps"] == before[-1]["critic_steps"] + 2
    adam = after[-1]["actor_optimizers"][0]["state"]
    assert all(float(state["step"]) == policy.actor_steps for state in adam.values())
  else:
    assert_tree_close(after, before, atol=0, rtol=0, equal_nan=True)
  assert policy._policy_graph is None
  assert all(p.grad is None for p in policy.actor.parameters())
  handle.remove()


def test_cpu_request_keeps_existing_update_path():
  from tests.test_estimator_belief_policy import setup

  policy, starts, noises = setup()
  policy.cuda_graphs = True
  policy.update_from(starts, *noises)
  assert policy.graph_status == {
    "requested": True,
    "reason": "cpu",
    "scope": "actor_critic",
  }


def test_collection_and_normalization_between_updates_reuse_graph(graph_policy):
  training, starts = graph_policy(return_training=True)
  policy = training.policy
  for _ in range(4):
    policy.update_from(starts, *policy.draw_noises(starts))
  captured = policy._policy_graph
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
  assert policy._policy_graph is captured and captured.captures == 1


@pytest.mark.parametrize(
  "samples,microbatch,critic_microbatch,minibatches",
  [(1, 2, None, 4), (2, 2, 3, 2), (1, 2, 1, 20)],
)
def test_single_graph_batch_layouts(
  graph_policy, samples, microbatch, critic_microbatch, minibatches
):
  policy, starts = graph_policy()
  policy.cfg = replace(
    policy.cfg,
    num_samples=samples,
    microbatch_size=microbatch,
    critic_microbatch_size=critic_microbatch,
    critic_minibatches=minibatches,
    critic_epochs=2,
  )
  reference = reference_for(policy)
  process_rng = policy.generators["process"].get_state().clone()
  for _ in range(5):
    noise = policy.draw_noises(starts)
    torch.testing.assert_close(noise, reference.draw_noises(starts))
    actual = policy.update_from(starts, *noise)
    expected = reference.update_from(starts, *noise)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-3, equal_nan=True)
    assert_policy_close(policy, reference)
  assert policy._policy_graph.captures == 1 and policy._policy_graph.replay_calls == 2
  if samples == 1:
    assert torch.equal(policy.generators["process"].get_state(), process_rng)


def test_optimizer_learning_rates_are_live_or_recaptured(graph_policy):
  policy, starts = graph_policy()
  reference = reference_for(policy)

  def update_both():
    noise = policy.draw_noises(starts)
    reference.draw_noises(starts)
    policy.update_from(starts, *noise)
    reference.update_from(starts, *noise)
    assert_policy_close(policy, reference)

  for _ in range(4):
    update_both()
  captured_id = id(policy._policy_graph)
  for learner in (policy, reference):
    learner.actor_optimizers[0].param_groups[0]["lr"].mul_(0.8)
  update_both()
  assert id(policy._policy_graph) == captured_id
  for learner in (policy, reference):
    learner.actor_optimizers[1].param_groups[0]["lr"] *= 0.8
  update_both()
  assert policy._policy_graph.warmup_calls == 1
  assert policy._policy_graph.captures == 0
  for _ in range(3):
    update_both()
  assert policy._policy_graph.captures == 1


@pytest.mark.filterwarnings("error:The AccumulateGrad node")
def test_online_model_policy_graph_save_and_resume(graph_policy, tmp_path):
  training, _ = graph_policy(return_training=True, integrated=True)
  model_before = deepcopy(training.runner._dynamics.state_dict())
  for _ in range(11):
    training.run_step()
  policy = training.policy
  assert training.learner.mean_steps > 0 and training.learner.cov_steps > 0
  assert policy.actor_steps >= 4 and policy._policy_graph.captures == 1
  assert any(
    not torch.equal(value, model_before[name])
    for name, value in training.runner._dynamics.state_dict().items()
  )
  assert (
    training.last_profile["runtime"]["policy_cuda_graph"]["scope"] == "actor_critic"
  )
  assert policy.last_profile.keys() == {
    "policy_prepare_seconds",
    "policy_update_seconds",
    "policy_check_seconds",
  }
  path = tmp_path / "training.pt"
  training.save(path)
  saved = torch.load(path, weights_only=True, map_location="cuda:0")
  before = policy.actor_steps
  training.load_state_dict(saved)
  assert policy._policy_graph is None
  for _ in range(8):
    training.run_step()
  assert policy.actor_steps >= before + 4
  assert policy._policy_graph.captures == 1
