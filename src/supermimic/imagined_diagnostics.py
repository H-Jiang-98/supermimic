"""Independent soft-survival returns at the heads of recorded physical episodes."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch
from tensordict import TensorDict

from supermimic.belief import BeliefState
from supermimic.belief_provider import BeliefProvider
from supermimic.critic_inputs import CriticInputConfig
from supermimic.envs.multi_task_world_model_env import MultiTaskWorldModelEnv
from supermimic.envs.world_model_env import BeliefRolloutStart
from supermimic.estimator import LieBeliefFilter
from supermimic.motion_loader import motion_content_digest
from supermimic.replay_buffer import FieldSpec, ReplayBuffer
from supermimic.runners.eit_inference import load_evaluation_models
from supermimic.runners.eit_snapshots import file_sha256
from supermimic.scripts.eval_eit import evaluation_env_config
from supermimic.scripts.train_utils import create_env
from supermimic.utils.lie_state import LieState


def _head(ep, env, estimator, motion_digest):
  device = env.device
  ids = torch.tensor([ep["env_id"]], device=device)
  sensor = ep["sensor"][:1].to(device)
  batch = TensorDict(
    {
      "state": ep["full_state"][:1],
      "sensor_window": sensor[:, None].repeat(1, 5, 1),
      "sim_ctx": ep["sim_ctx"][:1],
      "prev_action": sensor[:, 64:],
      "alignment_rotation": ep["alignment_rotation"][None],
      "alignment_translation": ep["alignment_translation"][None],
      "motion_time": torch.tensor([ep["start_frame"]]),
      "time_index": ids * 0,
      "segment_time": ids * 0,
      "episode_id": ids,
    },
    batch_size=[1],
    device=device,
  )
  prior = BeliefState(
    LieState.from_full(ep["prior_mean"][None].to(device)),
    ep["prior_covariance"][None].to(device),
  )
  belief = estimator.correct(prior, sensor[:, :64])
  remaining = env.motion.last_frame(batch["motion_time"]) - batch["motion_time"] + 1
  return BeliefRolloutStart(
    batch,
    belief,
    env.dynamics.model_version,
    env.replay_buffer,
    env.replay_buffer.insert_count,
    motion_digest,
    remaining,
  )


@torch.no_grad()
def _rollout(actor, env, start, horizon, samples, gamma, generator):
  env.clear_rollout()
  obs = env.reset_prepared(start.repeat_interleave(samples))
  remaining = int(start.remaining_steps.item())
  steps = min(horizon, remaining)
  survival = torch.ones(samples, device=env.device, dtype=torch.float64)
  live_tail = torch.full((samples,), gamma > 0, device=env.device, dtype=torch.bool)
  returns = torch.zeros_like(survival)
  for t in range(steps):
    obs, reward, done, _ = env.step(
      actor(obs),
      process_noise=torch.randn(samples, 70, device=env.device, generator=generator),
      measurement_noise=torch.randn(
        samples, 64, device=env.device, generator=generator
      ),
    )
    returns += gamma**t * survival * reward
    continuation = (1 - done) * (remaining > t + 1)
    survival *= continuation
    # A product of positive continuations can underflow before a path ends.
    live_tail &= continuation > 0
  return {
    "steps": steps,
    "clip_horizon_reached": steps == remaining,
    "return_samples": returns.tolist(),
    "return_mean": float(returns.mean()),
    "samples_with_unscored_tail": int(live_tail.sum()),
    "discounted_tail_weight_mean": float(gamma**steps * survival.mean()),
  }


def _compare_returns(imagined, ep, physical, gamma, same_policy):
  steps = imagined["steps"]
  estimate = imagined["return_mean"]
  complete = imagined["samples_with_unscored_tail"] == 0
  targets = physical["discounted_return"]
  full = targets[0] if targets is not None else None
  rewards = ep["reward"][:steps].double()
  prefix = None
  # Hard termination gives a known zero tail. A shorter censored trace does not.
  if len(rewards) == steps or full is not None:
    weights = torch.pow(
      gamma, torch.arange(len(rewards), device=rewards.device, dtype=rewards.dtype)
    )
    prefix = float(weights @ rewards)
  return {
    "env_id": ep["env_id"],
    "physical_reason": ep["reason"],
    "physical_return": full,
    "physical_prefix_return": prefix,
    "model_minus_physical_prefix": estimate - prefix
    if same_policy and prefix is not None
    else None,
    "model_minus_physical": estimate - full
    if complete and same_policy and full is not None
    else None,
    "critic_minus_model": physical["bootstrap_value"][0] - estimate
    if complete
    else None,
  }


@torch.no_grad()
def compare_episode_heads(
  saved, cfg, record, critic_scores, horizon, samples, seed, device, *, motion_file=None
):
  motion_file = Path(motion_file or record["motion_file"])
  if file_sha256(motion_file) != record["motion_sha256"]:
    raise ValueError("Recording's reference motion changed")
  env_cfg = evaluation_env_config(motion_file, record["seed"])
  env = create_env(env_cfg, 1, device)
  try:
    wm, actor = load_evaluation_models(env, cfg, saved, device)
    estimator = LieBeliefFilter(
      wm.dynamics,
      cfg.estimator_inputs.measurement_noise,
      saved["world_model"]["estimator"]["default_joint_pos"].to(device),
      config=cfg.filter_config,
    )
    # An empty, private replay seals externally reconstructed starts for reset_prepared.
    replay = ReplayBuffer(
      {k: FieldSpec(dtype=torch.bool) for k in ("terminated", "truncated")},
      1,
      1,
      device,
    )
    imagined = MultiTaskWorldModelEnv(
      wm.dynamics,
      wm.representation,
      env,
      samples,
      replay,
      actor.task_specs,
      cfg.history_length,
      cfg.termination_softness,
      device,
      replace(cfg.estimator_inputs, mode="train"),
      BeliefProvider(estimator, differentiable=True),
    )
    imagined.eit_critic = True
    imagined.critic_input_config = CriticInputConfig(
      **saved["critic"]["contract"]["config"]
    )
    digest = motion_content_digest(imagined.motion)
    if digest != record["motion_digest"]:
      raise ValueError("Loaded reference differs from recording")
    generator = torch.Generator(device=device).manual_seed(seed)
    actor_source = "truth" if cfg.estimator_inputs.group == "A" else "posterior"
    same_policy = actor_source == record["actor_state_source"]
    rows = []
    with imagined.readonly_policy_motion():
      for ep, physical in zip(
        record["episodes"], critic_scores["episodes"], strict=True
      ):
        start = _head(ep, imagined, estimator, digest)
        values = _rollout(
          actor, imagined, start, horizon, samples, critic_scores["gamma"], generator
        )
        values.update(
          _compare_returns(values, ep, physical, critic_scores["gamma"], same_policy)
        )
        rows.append(values)
    return {
      "horizon": horizon,
      "samples_per_episode": samples,
      "seed": seed,
      "return_scope": "soft_survival_discounted_prefix_without_bootstrap",
      "actor_state_source": actor_source,
      "behavior_policies_match": same_policy,
      "physical_actor_state_source": record["actor_state_source"],
      "episodes": rows,
    }
  finally:
    env.close()
