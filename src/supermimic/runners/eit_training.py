"""Two-stage world model fitting and stochastic value-gradient policy training.

Round checkpoints reuse the model-only loop's atomic file and output-directory
protocol. Optimizer continuation is exact for fixed raw inputs/noises; simulator
and replay state are deliberately restarted after restoration.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from supermimic.critic_inputs import BeliefCritic
from supermimic.envs.world_model_env import BeliefRolloutStart
from supermimic.learners.belief_policy_learner import (
  BeliefActorCriticPolicyLearner,
  BeliefPolicyLearnerCfg,
  policy_metrics_for_json,
)
from supermimic.posterior_replay import POSTERIOR_REPLAY_VERSION
from supermimic.runners.belief_training import (
  BeliefModelTraining,
  BeliefTrainingCfg,
  parameter_digest,
)
from supermimic.runners.run_config import (
  EIT_INITIALIZATION_VERSION,
  EIT_POLICY_TRAINING_VERSION,
  validate_eit_checkpoint_contract,
)
from supermimic.runners.supermimic_runner import (
  SuperMimicRunner,
  actor_checkpoint,
  build_ablation_actors,
  build_actor,
  load_actor_checkpoint,
)
from supermimic.utils.profiling import StreamTimer

if TYPE_CHECKING:
  from supermimic.utils.eit_wandb import EITOfflineWandb


@dataclass(frozen=True)
class EITTrainingCfg(BeliefTrainingCfg):
  normalization_batch_size: int = 16
  posterior_save_stride: int = 1

  def __post_init__(self):
    super().__post_init__()
    if type(self.posterior_save_stride) is not int or self.posterior_save_stride < 1:
      raise ValueError("posterior_save_stride must be positive")
    if (
      type(self.normalization_batch_size) is not int
      or self.normalization_batch_size < 1
    ):
      raise ValueError("normalization_batch_size must be positive")


def initialization_fingerprint(
  bundle: dict, *, include_model_revision: bool = True
) -> str:
  digest = hashlib.sha256()
  skip_revision_for = (
    None
    if include_model_revision
    else bundle.get("world_model", bundle).get("dynamics")
  )

  def hash_value(value):
    if isinstance(value, torch.Tensor):
      digest.update(str((value.dtype, tuple(value.shape))).encode())
      digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    elif isinstance(value, dict):
      for name in sorted(value):
        if value is skip_revision_for and name == "model_revision":
          continue
        digest.update(name.encode())
        hash_value(value[name])
    elif isinstance(value, (tuple, list)):
      for item in value:
        hash_value(item)
    else:
      digest.update(json.dumps(value, sort_keys=True).encode())

  hash_value(bundle)
  return digest.hexdigest()


class EITTraining(BeliefModelTraining):
  """Train the model, actor and critic in separate, versioned phases."""

  cfg: EITTrainingCfg

  def __init__(
    self,
    runner: SuperMimicRunner,
    cfg: EITTrainingCfg | None = None,
    policy_cfg: BeliefPolicyLearnerCfg | None = None,
  ):
    if runner.replay_buffer.replay_mode != "full":
      raise ValueError("EIT policy starts require full replay")
    # Reuse all model-loop preconditions, then explicitly undo its fixed-actor mode.
    super().__init__(runner, cfg or EITTrainingCfg())
    runner.replay_buffer.enable_posterior_cache(
      self.cfg.posterior_save_stride, runner.cfg.history_length
    )
    runner.posterior_save_stride = self.cfg.posterior_save_stride
    self.continuous_collection = True
    actor = runner.policy_learner.actor
    actor.requires_grad_(True)
    inputs = runner.cfg.estimator_inputs
    assert inputs is not None
    parameter = next(actor.parameters())
    critic = BeliefCritic(
      inputs.group, runner._sim_context.output_dim, runner.cfg.critic_hidden_dims
    ).to(device=parameter.device, dtype=parameter.dtype)
    runner.wm_env.eit_critic = True
    runner.policy_learner = BeliefActorCriticPolicyLearner(
      actor,
      critic,
      runner.wm_env,
      policy_cfg or BeliefPolicyLearnerCfg(),
      runner.device,
    )
    self.initialization_digest: str | None = None

  @property
  def policy(self) -> BeliefActorCriticPolicyLearner:
    return self.runner.policy_learner

  @torch.no_grad()
  def update_input_normalization(
    self, frames=None, policy_keys: torch.Tensor | None = None
  ) -> BeliefRolloutStart | None:
    """Normalize this round's saved training slots or an explicit frame selection."""
    timer = StreamTimer(self.runner.device)
    with timer.measure("posterior_cache_normalization_seconds"):
      starts = self._normalize_saved_posteriors(frames, policy_keys, timer)
    self._normalization_profile = timer.seconds()
    return starts

  def _normalize_saved_posteriors(self, frames, policy_keys, timer):
    env = self.runner.wm_env
    replay = self.runner.replay_buffer
    cache = replay.posterior_cache
    if cache is None:
      raise ValueError("EIT normalization requires posterior replay")
    with timer.measure("posterior_cache_read_seconds"):
      locations = cache.locations(replay)
      rows, slots, posterior_slots = locations
      training = (
        replay._storage["episode_id"][rows, slots] % self.cfg.validation_stride != 0
      )
      if frames is None:
        first = replay.insert_count - self.cfg.collection_steps
        selected = cache.sequences[posterior_slots] >= first
      else:
        selected = cache.frame_mask(replay, locations, frames)
      selected &= training
      fresh_locations = tuple(index[selected] for index in locations)
      starts = (
        env.prepare_policy_starts(*policy_keys.unbind(dim=1))
        if policy_keys is not None and len(policy_keys)
        else None
      )
    count = len(fresh_locations[0])
    gaps = (
      self.runner._dynamics.model_revision
      - cache.storage["posterior_revision"][rows, posterior_slots]
      if len(rows)
      else rows
    )
    self._cache_budget = {
      "posterior_save_stride": cache.stride,
      "effective_posterior_count": len(rows),
      "available_policy_starts": int(training.sum()),
      "normalization_samples": count,
      "posterior_revision_gap_min": int(gaps.min()) if len(gaps) else 0,
      "posterior_revision_gap_max": int(gaps.max()) if len(gaps) else 0,
      "posterior_revision_gap_mean": float(gaps.float().mean()) if len(gaps) else 0,
      **cache.storage_bytes(),
      "episode_head_storage_bytes": replay.episode_head_storage_bytes,
      "main_replay_storage_bytes": sum(
        t.numel() * t.element_size() for t in replay._storage.values()
      ),
    }
    if not count:
      return starts

    # Stage statistics so an invalid later chunk cannot partially update live models.
    candidate_actor = deepcopy(self.actor)
    candidate_critic = deepcopy(self.policy.critic)
    candidate_actor.train()
    candidate_critic.train()
    batch_size = self.cfg.normalization_batch_size
    batches = env.iter_cached_policy_starts(fresh_locations, batch_size=batch_size)
    try:
      for _ in range(0, count, batch_size):
        with timer.measure("posterior_cache_read_seconds"):
          chunk = next(batches)
        obs = env.reset_prepared(chunk)
        candidate_actor.update_normalization(obs)
        candidate_critic.update_normalization(obs["critic"])
      for module in (candidate_actor.obs_normalizer, candidate_actor.goal_normalizers):
        if not all(torch.isfinite(t).all() for t in module.buffers()):
          raise ValueError("Non-finite actor normalization statistics")
      self.actor.obs_normalizer.load_state_dict(
        candidate_actor.obs_normalizer.state_dict()
      )
      self.actor.goal_normalizers.load_state_dict(
        candidate_actor.goal_normalizers.state_dict()
      )
      # Critic candidates differ only in their explicit normalizer buffers.
      self.policy.critic.load_checkpoint(candidate_critic.checkpoint())
      return starts
    finally:
      env.clear_rollout()
      self.actor.eval()
      self.policy.critic.eval()

  def policy_episode_ids(self) -> list[int]:
    """Include every retained training episode, even a single-frame fragment."""
    return [
      episode
      for episode in self.runner.replay_buffer.training_episode_lengths()
      if episode % self.cfg.validation_stride
    ]

  def _update_policy(self) -> dict:
    runner = self.runner
    actor_before, critic_before = self.policy.actor_steps, self.policy.critic_steps
    policy_ids = self.policy_episode_ids()
    with runner._dynamics.readonly(), runner.wm_env.readonly_policy_motion():
      policy_keys = self.policy.sample_start_keys(policy_ids)
      starts = self.update_input_normalization(None, policy_keys)
      assert starts is not None
      policy_logs = self.policy.update_prepared(starts)
    self._policy_profile = {**self._normalization_profile, **self.policy.last_profile}
    self._policy_budget = {
      **getattr(self, "_cache_budget", {}),
      "actor_updates": self.policy.actor_steps - actor_before,
      "critic_updates": self.policy.critic_steps - critic_before,
      "policy_starts_drawn": len(starts),
      "imagined_steps": float(policy_logs["Policy/valid_steps"])
      * self.policy.cfg.num_batches,
      "survival_weighted_imagined_steps": float(
        policy_logs["Policy/survival_weighted_steps"]
      )
      * self.policy.cfg.num_batches,
      "policy_n": self.policy.cfg.batch_size,
      "policy_k": self.policy.cfg.num_samples,
      "policy_h": self.policy.cfg.horizon,
      "policy_microbatch": self.policy.cfg.microbatch_size,
    }
    runner.wm_env.clear_rollout()
    return {
      "critic_version": parameter_digest(self.policy.critic),
      "policy": policy_metrics_for_json(policy_logs),
    }

  def state_dict(self) -> dict:
    state = super().state_dict()
    state.update(
      stage="policy",
      version=EIT_POLICY_TRAINING_VERSION,
      posterior_replay_protocol=POSTERIOR_REPLAY_VERSION,
      model_revision=self.runner._dynamics.model_revision,
      critic=deepcopy(self.policy.critic.checkpoint()),
      policy=deepcopy(self.policy.state_dict()),
      policy_config=asdict(self.policy.cfg),
      initialization_digest=self.initialization_digest,
      actor_version=parameter_digest(self.actor),
      critic_version=parameter_digest(self.policy.critic),
    )
    return state

  @staticmethod
  def _check_checkpoint_version(state: dict) -> None:
    validate_eit_checkpoint_contract(state, stage="policy")
    if state.get("version") != EIT_POLICY_TRAINING_VERSION:
      raise ValueError(
        "EIT incremental training requires v4; v3 remains evaluation/export only"
      )

  def load_state_dict(self, state: dict) -> None:
    self._check_checkpoint_version(state)
    if state.get("policy_config") != asdict(self.policy.cfg):
      raise ValueError("EIT policy configuration mismatch")
    candidate_critic = deepcopy(self.policy.critic)
    candidate_critic.load_checkpoint(state["critic"])
    candidate_actor = deepcopy(self.actor)
    load_actor_checkpoint(candidate_actor, self.runner.cfg, state["actor"])
    if parameter_digest(candidate_actor) != state.get(
      "actor_version"
    ) or parameter_digest(candidate_critic) != state.get("critic_version"):
      raise ValueError("EIT actor/critic fingerprint mismatch")
    initialization = state.get("initialization_digest")
    if initialization is not None and (
      not isinstance(initialization, str) or len(initialization) != 64
    ):
      raise ValueError("Invalid initialization digest")
    # Validate policy optimizers against the checkpoint's model before touching
    # the active model; the candidate environment is shallow, with a private model.
    from copy import copy

    candidate_env = copy(self.runner.wm_env)
    candidate_wm = deepcopy(self.runner._wm)
    candidate_wm.load_bundle(state["world_model"])
    candidate_env.dynamics = candidate_wm.dynamics
    candidate_policy = BeliefActorCriticPolicyLearner(
      candidate_actor,
      candidate_critic,
      candidate_env,
      self.policy.cfg,
      self.runner.device,
    )
    candidate_policy.load_state_dict(state["policy"])
    # Release the policy preflight copies before the model loader creates its own.
    del candidate_policy, candidate_env, candidate_wm, candidate_actor, candidate_critic
    # The inherited loader preflights model, model optimizer, actor, metadata and
    # RNG before any mutation. All additional EIT components passed above.
    saved_cfg = replace(
      self.cfg, posterior_save_stride=state["loop_config"]["posterior_save_stride"]
    )
    old_cfg, old_stride = self.cfg, self.runner.posterior_save_stride
    try:
      self.cfg = saved_cfg
      self.runner.posterior_save_stride = saved_cfg.posterior_save_stride
      super().load_state_dict(state)
    except Exception:
      self.cfg, self.runner.posterior_save_stride = old_cfg, old_stride
      raise
    self.policy.critic.load_checkpoint(state["critic"])
    self.policy.load_state_dict(state["policy"])
    self.initialization_digest = initialization
    self.actor.eval().requires_grad_(True)
    self.policy.critic.eval().requires_grad_(True)

  def run(
    self,
    rounds: int,
    output: Path,
    *,
    resume: Path | None = None,
    snapshot_interval: int | None = None,
    logger: EITOfflineWandb | None = None,
  ):
    checkpoint = Path(output).resolve() / "training.pt"
    if resume is None and self._output_checkpoint == checkpoint and checkpoint.exists():
      saved = torch.load(checkpoint, map_location=self.runner.device, weights_only=True)
      if parameter_digest(self.actor) != saved.get("actor_version") or parameter_digest(
        self.policy.critic
      ) != saved.get("critic_version"):
        raise ValueError("Active actor/critic differs from output checkpoint")
    return super().run(
      rounds, output, resume=resume, snapshot_interval=snapshot_interval, logger=logger
    )

  def initialization_bundle(self) -> dict:
    """Export explicit, independently allocated common A/B/C initial weights."""
    if (
      self.rounds
      or self.learner.mean_steps
      or self.learner.cov_steps
      or self.policy.actor_steps
      or self.policy.critic_steps
    ):
      raise ValueError("Common initialization can only be exported before training")
    runner = self.runner
    source = self.actor
    parameter = next(source.parameters())
    # Creating the shared file must leave the same collection RNG as loading it.
    with torch.random.fork_rng(devices=[parameter.device] if parameter.is_cuda else []):
      actors = build_ablation_actors(runner.cfg, runner._wm, runner.device)
      for actor in actors.values():
        actor.to(parameter)
        actor.copy_common_initialization_from(source)
      critics = {}
      for group in ("A", "B", "C"):
        critic = BeliefCritic(
          group, runner._sim_context.output_dim, runner.cfg.critic_hidden_dims
        ).to(parameter)
        critic.copy_common_initialization_from(self.policy.critic)
        critics[group] = critic
    # B/C include identical extra belief columns even when the source was A.
    critics["C"].copy_common_initialization_from(critics["B"])
    world_model = runner._wm.bundle()
    # Shared initial weights retain their v3 identity across training protocols.
    world_model["dynamics"].pop("model_revision", None)
    return deepcopy(
      {
        "version": EIT_INITIALIZATION_VERSION,
        "generation_group": runner.cfg.estimator_inputs.group,
        "generation_seed": self.training_seed,
        "world_model": world_model,
        "actors": {
          group: actor_checkpoint(
            actor,
            replace(
              runner.cfg,
              estimator_inputs=replace(runner.cfg.estimator_inputs, group=group),
            ),
          )
          for group, actor in actors.items()
        },
        "critics": {group: critic.checkpoint() for group, critic in critics.items()},
      }
    )

  def load_initialization(self, bundle: dict) -> None:
    if (
      self.rounds
      or self.learner.mean_steps
      or self.learner.cov_steps
      or self.policy.actor_steps
      or self.policy.critic_steps
    ):
      raise ValueError("Initialization cannot replace an active training run")
    if bundle.get("version") != EIT_INITIALIZATION_VERSION:
      raise ValueError("Unsupported EIT initialization")
    if bundle.get("generation_seed") not in (None, self.training_seed):
      raise ValueError("Initialization seed differs from the training seed")
    runner = self.runner
    group = runner.cfg.estimator_inputs.group
    wm = deepcopy(runner._wm)
    wm.load_bundle(bundle["world_model"])
    if (
      wm.filter_config != runner.cfg.filter_config
      or wm.initial_prior != runner.cfg.initial_prior
      or wm.representation.measurement_config
      != runner.cfg.estimator_inputs.measurement_noise
    ):
      raise ValueError(
        "Initialization estimator settings disagree with runner configuration"
      )
    if set(bundle.get("actors", {})) != {"A", "B", "C"} or set(
      bundle.get("critics", {})
    ) != {"A", "B", "C"}:
      raise ValueError("Common initialization requires all three actors and critics")
    device = torch.device(runner.device)
    devices = (
      [device.index if device.index is not None else torch.cuda.current_device()]
      if device.type == "cuda"
      else []
    )
    # Constructors draw fresh weights; preserve caller RNG even on failed preflight.
    with torch.random.fork_rng(devices=devices):
      for key in ("A", "B", "C"):
        config = replace(
          runner.cfg, estimator_inputs=replace(runner.cfg.estimator_inputs, group=key)
        )
        actor = build_actor(config, wm, runner.device).to(next(self.actor.parameters()))
        load_actor_checkpoint(actor, config, bundle["actors"][key])
        critic = BeliefCritic(
          key, runner._sim_context.output_dim, runner.cfg.critic_hidden_dims
        ).to(next(self.actor.parameters()))
        critic.load_checkpoint(bundle["critics"][key])
    # Hash every group's weights/config and the context, before mutating anything.
    identity = initialization_fingerprint(bundle)
    runner._wm.load_bundle(bundle["world_model"])
    load_actor_checkpoint(self.actor, runner.cfg, bundle["actors"][group])
    self.policy.critic.load_checkpoint(bundle["critics"][group])
    self.initialization_digest = identity
    runner.wm_env.clear_rollout()
    self._ready = False
