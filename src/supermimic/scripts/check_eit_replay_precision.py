"""Compare candidate posterior storage formats on one physical EIT replay (CPU)."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import cast

import mjlab.tasks  # noqa: F401
import torch
from mjlab.tasks.registry import load_env_cfg

import supermimic.tasks  # noqa: F401
from supermimic.belief import BeliefState
from supermimic.critic_inputs import BeliefCritic
from supermimic.learners.belief_policy_learner import BeliefPolicyLearnerCfg
from supermimic.motion_command import MotionCommandCfg
from supermimic.posterior_replay import pack_mean, unpack_belief, unpack_mean
from supermimic.runners.eit_snapshots import file_sha256
from supermimic.runners.eit_training import EITTraining, EITTrainingCfg
from supermimic.runners.run_config import (
  load_runner_cfg,
  validate_eit_checkpoint_contract,
)
from supermimic.runners.supermimic_runner import SuperMimicRunner, load_actor_checkpoint
from supermimic.scripts.train_utils import create_env
from supermimic.tasks.env_cfgs import configure_estimator_env_cfg
from supermimic.utils.validation import check_covariance, validate_rotation


def factor_round_trip(covariance, dtype):
  """Store a scaled triangular factor, using the FP32 triangle for failed rows."""
  rows, cols = torch.tril_indices(70, 70, device=covariance.device)
  std = covariance.diagonal(dim1=-2, dim2=-1).sqrt()
  scale = torch.where(std > 0, std, 1)
  correlation = covariance / scale.unsqueeze(-1) / scale.unsqueeze(-2)
  factors, info = torch.linalg.cholesky_ex(correlation)
  keep = info == 0
  packed = factors[keep][:, rows, cols].to(dtype)
  stored_std = std[keep]
  fallback = covariance[~keep][:, rows, cols]
  lower = covariance.new_zeros(len(packed), 70, 70)
  lower[:, rows, cols] = packed.to(lower.dtype)
  lower = lower * stored_std.unsqueeze(-1)
  decoded = torch.empty_like(covariance)
  decoded[keep] = lower @ lower.mT
  exact = covariance.new_zeros(len(fallback), 70, 70)
  exact[:, rows, cols] = fallback
  exact[:, cols, rows] = fallback
  decoded[~keep] = exact
  payload = keep.nbytes + packed.nbytes + stored_std.nbytes + fallback.nbytes
  return decoded, payload, int((~keep).sum())


def rejected_frames(values, validate):
  count = 0
  for frame in values:
    try:
      validate(frame)
    except ValueError:
      count += 1
  return count


def policy_probe(policy, starts, noises, horizon):
  """Compare the stochastic value-gradient objective on identical paths and noises."""
  try:
    returns, _, _ = policy._rollout(starts, *noises, horizon)
    gradient = torch.autograd.grad(
      returns.mean(), policy.actor.parameters(), allow_unused=True
    )
    flat = torch.cat([g.detach().flatten() for g in gradient if g is not None])
    if not torch.isfinite(flat).all() or not torch.isfinite(returns).all():
      raise FloatingPointError("Non-finite policy return or gradient")
    return returns.detach(), flat
  finally:
    policy.wm_env.clear_rollout()


def relative_norm(difference, reference):
  denominator = reference.norm().item()
  if denominator:
    relative = difference.norm().item() / denominator
    if math.isfinite(relative):
      return relative
  return None


def cast_error(restored, original):
  error = (restored - original).abs().max().item()
  return error if math.isfinite(error) else None


def audit(training, steps):
  runner, policy = training.runner, training.policy
  training.refresh_collection()
  with runner._dynamics.readonly():
    for _ in range(steps):
      runner.step(record_episode_stats=False)
  runner.finish_collection_round()
  replay = runner.replay_buffer
  cache = replay.posterior_cache
  locations = cache.locations(replay)
  # A single evaluation shares one sealed ring and model version across all codecs.
  # Keep the same ordering between data used for matrix checks and policy starts.
  starts = next(
    runner.wm_env.iter_cached_policy_starts(locations, batch_size=len(locations[0]))
  )
  covariance = starts.belief.covariance
  mean = pack_mean(starts.belief.mean)
  rows, cols = torch.tril_indices(70, 70)
  lower = covariance[:, rows, cols]
  episode_ids = starts.batch["episode_id"]
  horizon = policy.cfg.horizon
  candidates = (episode_ids % training.cfg.validation_stride != 0).nonzero().flatten()
  if not len(candidates):
    raise ValueError("Need a retained training frame")
  chosen = candidates[torch.randperm(len(candidates))[: policy.cfg.batch_size]]
  scored = starts.index(chosen).repeat_interleave(policy.cfg.num_samples)
  noises = tuple(v.flatten(1, 2) for v in policy.draw_noises(starts.index(chosen)))
  policy.actor.eval()
  policy.critic.eval().requires_grad_(False)
  ref_return, ref_grad = policy_probe(policy, scored, noises, horizon)
  reference_rtol = starts.belief.covariance_rtol
  filter_config = runner.posterior_provider.estimator.config

  def validate_cov(cov):
    check_covariance(cov, "cached covariance", rtol=reference_rtol)

  def validate_filter_cov(cov):
    check_covariance(
      cov,
      "filter covariance",
      rtol=filter_config.covariance_rtol,
      atol=filter_config.covariance_atol,
    )

  def score(decoded_mean, decoded_cov, payload, fallback=0):
    belief = BeliefState(unpack_mean(decoded_mean), decoded_cov)
    row = {
      "mean_cov_payload_bytes": payload,
      "fp32_factor_fallback_frames": fallback,
      "invalid_rotations": rejected_frames(belief.mean.rotation, validate_rotation),
      "nonfinite_mean_entries": int((~torch.isfinite(decoded_mean)).sum()),
      "invalid_covariances": rejected_frames(decoded_cov, validate_cov),
      "invalid_filter_covariances": rejected_frames(decoded_cov, validate_filter_cov),
      "cov_entries_nonzero_to_zero": int(
        ((covariance != 0) & (decoded_cov == 0)).sum()
      ),
      "cov_relative_frobenius": relative_norm(decoded_cov - covariance, covariance),
      "mean_max_abs_error": cast_error(decoded_mean, mean),
    }
    # A format that has already failed the replay contract cannot be used for training.
    if not (
      row["invalid_rotations"]
      or row["invalid_covariances"]
      or row["invalid_filter_covariances"]
      or row["nonfinite_mean_entries"]
    ):
      scored_belief = belief.index(chosen.repeat_interleave(policy.cfg.num_samples))
      try:
        ret, grad = policy_probe(
          policy, replace(scored, belief=scored_belief), noises, horizon
        )
      except (ValueError, FloatingPointError) as exc:
        row["policy_error"] = str(exc)
      else:
        row.update(
          max_abs_return_error=(ret - ref_return).abs().max().item(),
          relative_actor_gradient_error=relative_norm(grad - ref_grad, ref_grad),
        )
    return row

  formats = {}
  main_round_trips = {}
  for dtype in (torch.float64, torch.float32, torch.bfloat16, torch.float16):
    name = str(dtype).removeprefix("torch.")
    qmean, qlower = mean.to(dtype), lower.to(dtype)
    cov = torch.empty_like(covariance)
    cov[:, rows, cols] = qlower.float()
    cov[:, cols, rows] = qlower.float()
    formats[name] = score(qmean.float(), cov, qmean.nbytes + qlower.nbytes)
    fields = {}
    for key, stored in replay._storage.items():
      if stored.is_floating_point():
        values = stored[locations[0], locations[1]]
        restored = values.to(dtype).to(values.dtype)
        fields[key] = {
          "max_abs_error": cast_error(restored, values),
          "nonfinite_entries": int((~torch.isfinite(restored)).sum()),
          "allocated_bytes": stored.numel() * dtype.itemsize,
        }
        if key == "alignment_rotation":
          fields[key]["invalid_rotations"] = rejected_frames(
            restored, validate_rotation
          )
    main_round_trips[name] = fields
    if dtype in (torch.bfloat16, torch.float16):
      # Widening through the normal belief API reprojects rotations and repairs
      # covariance roundoff with a full eigendecomposition on the read path.
      try:
        repaired = unpack_belief(qmean, qlower, dtype=mean.dtype)
      except ValueError as exc:
        formats[name + "_with_cast_repair"] = {"decode_error": str(exc)}
      else:
        formats[name + "_with_cast_repair"] = score(
          pack_mean(repaired.mean), repaired.covariance, qmean.nbytes + qlower.nbytes
        )
      formats["fp32_mean_" + name + "_cov"] = score(
        mean, cov, mean.nbytes + qlower.nbytes
      )
      rebuilt, size, fallback = factor_round_trip(covariance, dtype)
      formats["fp32_mean_" + name + "_factor"] = score(
        mean, rebuilt, mean.nbytes + size, fallback
      )
  return {
    "retained_frames": len(starts),
    "source_dtype": str(mean.dtype),
    "posterior_save_stride": cache.stride,
    "validation_stride": training.cfg.validation_stride,
    "main_replay_bytes_by_field": {k: v.nbytes for k, v in replay._storage.items()},
    "current_cache_bytes": cache.storage_bytes(),
    "policy_starts": len(chosen),
    "policy_samples_per_start": policy.cfg.num_samples,
    "policy_horizon": horizon,
    "formats": formats,
    "main_replay_field_casts": main_round_trips,
  }


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--motion-file", type=Path, required=True)
  parser.add_argument("--steps", type=int, default=32)
  parser.add_argument("--num-envs", type=int, default=4)
  parser.add_argument("--seed", type=int, default=2026)
  parser.add_argument("--output", type=Path, help="Write a standalone JSON report")
  args = parser.parse_args()
  if min(args.steps, args.num_envs) < 1:
    parser.error("steps and num-envs must be positive")
  if args.output and args.output.resolve() in (
    args.motion_file.resolve(),
    args.checkpoint.resolve(),
  ):
    parser.error("output must differ from the input files")
  torch.set_num_threads(1)
  torch.manual_seed(args.seed)
  torch.compiler.set_stance("force_eager")
  saved = torch.load(args.checkpoint, weights_only=True, map_location="cpu", mmap=True)
  validate_eit_checkpoint_contract(saved, stage="policy")
  cfg = load_runner_cfg({"runner_cfg": saved["runner_config"]})
  inputs = cfg.estimator_inputs
  assert inputs is not None
  cfg.sampling_mode = "uniform"
  task = "SuperMimic-GenCoord-G1"
  env_cfg = load_env_cfg(task)
  env_cfg.seed = args.seed
  motion = cast(MotionCommandCfg, env_cfg.commands["motion"])
  motion.motion_file = str(args.motion_file.resolve())
  motion.sampling_mode = "uniform"
  configure_estimator_env_cfg(env_cfg)
  env = create_env(env_cfg, args.num_envs, "cpu")
  try:
    runner = SuperMimicRunner(env, cfg, task, device="cpu")
    # Retain every collected posterior for this comparison, and select policy
    # probes using the checkpoint's actual train/validation episode split.
    loop_cfg = replace(EITTrainingCfg(**saved["loop_config"]), posterior_save_stride=1)
    training = EITTraining(
      runner, loop_cfg, BeliefPolicyLearnerCfg(**saved["policy_config"])
    )
    runner._wm.load_bundle(saved["world_model"])
    load_actor_checkpoint(training.actor, cfg, saved["actor"])
    cast(BeliefCritic, training.policy.critic).load_checkpoint(saved["critic"])
    with runner.wm_env.readonly_policy_motion():
      result = audit(training, args.steps)
  finally:
    env.close()
  result.update(
    checkpoint_sha256=file_sha256(args.checkpoint),
    motion_sha256=file_sha256(args.motion_file),
    group=inputs.group,
    seed=args.seed,
    steps=args.steps,
    num_envs=args.num_envs,
    device="cpu",
    torch_version=str(torch.__version__),
  )
  if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
  print("EIT_REPLAY_PRECISION " + json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
  main()
