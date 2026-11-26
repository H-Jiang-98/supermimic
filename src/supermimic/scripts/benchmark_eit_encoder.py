"""Compare batched and stepwise motion encoding on identical full policy updates."""

import argparse
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from statistics import median
from time import perf_counter

import torch

from supermimic.estimator import FilterConfig
from supermimic.learners.belief_policy_learner import BeliefPolicyLearnerCfg
from supermimic.models import ProcessNoiseConfig
from supermimic.runners.eit_training import EITTraining
from supermimic.runners.supermimic_runner import SuperMimicRunner, SuperMimicRunnerCfg
from supermimic.scripts.train_eit import (
  configure_estimator_env_cfg,
  create_env,
  load_env_cfg,
)


def compare(policy, starts, noises):
  actor = policy.actor
  actor_weights = deepcopy(actor.state_dict())
  critic_weights = deepcopy(policy.critic.state_dict())
  initial = policy.state_dict()
  encode, decode = policy._encode, policy._decode
  results = {}
  expected = None
  cuda = next(actor.parameters()).is_cuda
  for batched in (False, True):
    policy.cfg = replace(policy.cfg, batch_motion_encoder=batched)
    initial["config"]["batch_motion_encoder"] = batched
    actor.load_state_dict(actor_weights)
    policy.critic.load_state_dict(critic_weights)
    policy.load_state_dict(initial)
    latents, actions = [], []

    def trace_encode(goal, outputs=latents):
      latent = encode(goal)
      outputs.append(latent.detach())
      return latent

    def trace_decode(latent, history, condition, outputs=actions):
      result = decode(latent, history, condition)
      outputs.append(policy.wm_env.limit_actions(result[0]).detach())
      return result

    policy._encode, policy._decode = trace_encode, trace_decode
    try:
      result = policy._rollout_result(
        starts, *(n.squeeze(2) for n in noises), policy.cfg.horizon
      )
      gradients = torch.autograd.grad(-result.returns.mean(), tuple(actor.parameters()))
      trace = dict(
        latent=torch.cat(latents),
        action=torch.stack(actions),
        returns=result.returns.detach(),
      )
      for prefix in ("encoders", "decoder", "film"):
        trace[prefix] = torch.cat(
          [
            g.flatten()
            for (name, _), g in zip(actor.named_parameters(), gradients, strict=True)
            if name.startswith(prefix)
          ]
        )
      if result.invalid.any():
        raise FloatingPointError("Invalid benchmark rollout")
      trace = {key: value.cpu() for key, value in trace.items()}
      del result, gradients
    finally:
      policy._encode, policy._decode = encode, decode
    name = "batched" if batched else "stepwise"
    assert len(latents) == (1 if batched else policy.cfg.horizon)
    results[name] = dict(encoder_calls_per_microbatch=len(latents))
    latents.clear()
    actions.clear()
    if expected is None:
      expected = trace
    else:
      for key in trace:
        torch.testing.assert_close(trace[key], expected[key], atol=2e-5, rtol=1e-3)
      results["max_absolute_difference"] = {
        k: (v - expected[k]).abs().max().item() for k, v in trace.items()
      }
    # Warm graph capture and replay as well as eager kernels before timing.
    for _ in range(5):
      policy.update_from(starts, *noises)
    warm_replays = policy.graph_status.get("replay_calls", 0)
    if cuda:
      torch.cuda.synchronize()
      torch.cuda.reset_peak_memory_stats()
    timings = []
    for _ in range(5):
      start = perf_counter()
      policy.update_from(starts, *noises)
      if cuda:
        torch.cuda.synchronize()
      timings.append((perf_counter() - start) * 1000)
    results[name].update(
      median_full_update_ms=median(timings),
      samples_ms=timings,
      peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if cuda else None,
    )
    if policy.cuda_graphs:
      status = policy.graph_status
      if status.get("captures") != 1 or status["replay_calls"] - warm_replays != len(
        timings
      ):
        raise RuntimeError(
          f"Timed policy updates did not all replay the CUDA graph: {status}"
        )
      results[name]["cuda_graph"] = status
    # The timed paths include actor/critic backward, optimizer state and graph replay.
    state = {k: v.detach().to("cpu", copy=True) for k, v in actor.named_parameters()}
    state.update(
      {
        "critic." + k: v.detach().to("cpu", copy=True)
        for k, v in policy.critic.named_parameters()
      }
    )
    if not batched:
      stepwise_weights = state
    else:
      for k in state:
        torch.testing.assert_close(state[k], stepwise_weights[k], atol=2e-4, rtol=2e-3)
  return results


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--motion-file", required=True)
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--horizon", type=int, default=8)
  parser.add_argument("--batch", type=int, default=4)
  parser.add_argument("--compile", action="store_true")
  parser.add_argument("--cuda-graphs", action="store_true")
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  if args.cuda_graphs and torch.device(args.device).type != "cuda":
    parser.error("--cuda-graphs requires a CUDA device")
  torch.manual_seed(2027)
  torch.set_num_threads(1)
  task = "SuperMimic-GenCoord-G1"
  cfg = load_env_cfg(task)
  cfg.seed = 2027
  cfg.commands["motion"].motion_file = args.motion_file
  cfg.commands["motion"].sampling_mode = "start"
  configure_estimator_env_cfg(cfg)
  env = create_env(cfg, args.batch, args.device)
  try:
    cfg = SuperMimicRunnerCfg.for_estimator_inputs(
      "belief",
      process_noise=ProcessNoiseConfig(hidden_dims=(16,)),
      filter_config=FilterConfig(jacobian_backend="increment"),
      dynamics_hidden_dims=(16,),
      actor_hidden_dims=(64, 32),
      replay_ratio=0,
      buffer_size=8,
      sampling_mode="start",
    )
    cfg.policy.batch_size = args.batch
    runner = SuperMimicRunner(
      env, cfg, task, device=args.device, compile_eit=args.compile
    )
    policy_cfg = BeliefPolicyLearnerCfg(
      batch_size=args.batch,
      microbatch_size=args.batch,
      horizon=args.horizon,
      critic_minibatches=1,
      critic_epochs=1,
      noise=0,
    )
    training = EITTraining(
      runner, policy_cfg=policy_cfg, policy_cuda_graphs=args.cuda_graphs
    )
    training.run_step()
    policy = training.policy
    with torch.no_grad():
      policy.actor.film[-1].weight.normal_(std=0.01)
    runner._dynamics.requires_grad_(False)
    policy.critic.requires_grad_(False)
    starts = policy.sample_starts()
    results = compare(policy, starts, policy.draw_noises(starts))
    args.output.write_text(
      json.dumps(
        dict(
          device=args.device,
          torch_version=torch.__version__,
          horizon=args.horizon,
          microbatch=args.batch,
          compile=args.compile,
          cuda_graphs=args.cuda_graphs,
          actor_hidden_dims=cfg.actor_hidden_dims,
          critic_hidden_dims=cfg.critic_hidden_dims,
          **results,
        ),
        indent=2,
      )
      + "\n"
    )
  finally:
    env.close()


if __name__ == "__main__":
  main()
