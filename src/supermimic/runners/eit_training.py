"""Alternating posterior-NLL and stochastic value-gradient training.

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

import torch

from supermimic.critic_inputs import BeliefCritic
from supermimic.learners.belief_policy_learner import (
  BeliefActorCriticPolicyLearner,
  BeliefPolicyLearnerCfg,
  policy_metrics_for_json,
)
from supermimic.runners.belief_training import (
  BeliefModelTraining,
  BeliefTrainingCfg,
  _round_training_metrics,
)
from supermimic.runners.supermimic_runner import (
  SuperMimicRunner,
  actor_checkpoint,
  build_ablation_actors,
  build_actor,
  load_actor_checkpoint,
)


@dataclass(frozen=True)
class EITTrainingCfg(BeliefTrainingCfg):
  normalization_batch_size: int = 16

  def __post_init__(self):
    super().__post_init__()
    if (
      type(self.normalization_batch_size) is not int
      or self.normalization_batch_size < 1
    ):
      raise ValueError("normalization_batch_size must be positive")


def parameter_digest(module: torch.nn.Module) -> str:
  digest = hashlib.sha256()
  for name, value in module.state_dict().items():
    digest.update(name.encode())
    digest.update(value.detach().cpu().contiguous().numpy().tobytes())
  return digest.hexdigest()


class EITTraining(BeliefModelTraining):
  """Train the model, actor and critic in separate, versioned phases."""

  def __init__(
    self,
    runner: SuperMimicRunner,
    cfg: EITTrainingCfg | None = None,
    policy_cfg: BeliefPolicyLearnerCfg | None = None,
  ):
    # Reuse all model-loop preconditions, then explicitly undo its fixed-actor mode.
    super().__init__(runner, cfg or EITTrainingCfg())
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
  def update_input_normalization(self, frames) -> None:
    """Use each fresh training frame once, with current-model posterior goals."""
    if len(frames) == 0:
      return
    # Stage all small statistics before copying; malformed later batches must
    # not leave half-updated actor/critic normalization.
    candidate_actor = deepcopy(self.actor)
    candidate_critic = deepcopy(self.policy.critic)
    candidate_actor.train()
    candidate_critic.train()
    env = self.runner.wm_env
    try:
      for offset in range(0, len(frames), self.cfg.normalization_batch_size):
        batch = frames[offset : offset + self.cfg.normalization_batch_size]
        start = env.prepare_policy_starts(batch["episode_id"], batch["time_index"])
        obs = env.reset_prepared(start)
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
    finally:
      env.clear_rollout()
      self.actor.eval()
      self.policy.critic.eval()

  def run_round(self, checkpoint_path: Path | None = None) -> dict:
    runner, learner = self.runner, self.learner
    if not self._ready:
      self.refresh_collection()
    self.actor.eval()
    self.policy.critic.eval()
    version_before = runner._dynamics.model_version
    actor_before = parameter_digest(self.actor)
    for _ in range(self.cfg.collection_steps):
      runner.step(record_episode_stats=False)
    if (
      runner._dynamics.model_version != version_before
      or parameter_digest(self.actor) != actor_before
    ):
      raise RuntimeError("Collection model or actor changed within a round")
    runner.finish_collection_round()
    self._ready = False
    train_ids, valid_ids = self.episode_split()
    fresh = runner.replay_buffer.retained_frames(newest_count=self.cfg.collection_steps)
    train_rows = fresh["episode_id"] % self.cfg.validation_stride != 0
    train_frames = fresh[train_rows]
    if len(train_frames):
      learner.update_normalization(train_frames["state"], train_frames["action"])
    logs = []
    for _ in range(learner.cfg.num_batches):
      sequences = runner.replay_buffer.sample_belief_sequences(
        learner.cfg.horizon, learner.cfg.batch_size, episode_ids=train_ids
      )
      logs.append(learner.update(sequences))
    validation = learner.evaluate(
      runner.replay_buffer.sample_belief_sequences(
        learner.cfg.horizon, self.cfg.validation_batch_size, episode_ids=valid_ids
      )
    )
    self.update_input_normalization(train_frames)
    model_before_policy = runner._dynamics.model_version
    policy_logs = self.policy.update(episode_ids=train_ids)
    if runner._dynamics.model_version != model_before_policy:
      raise RuntimeError("Policy update modified the frozen world model")
    self.rounds += 1
    self.collected_frames += self.cfg.collection_steps * runner.env.num_envs
    metrics = {
      "round": self.rounds,
      "optimizer_steps": learner.steps,
      "collected_frames": self.collected_frames,
      "training_episodes": train_ids,
      "validation_episodes": valid_ids,
      "collection_model_version": version_before,
      "trained_model_version": runner._dynamics.model_version,
      "collection_actor_version": actor_before,
      "trained_actor_version": parameter_digest(self.actor),
      "critic_version": parameter_digest(self.policy.critic),
      "train": _round_training_metrics(logs),
      "validation": {key: float(value) for key, value in validation.items()},
      "policy": policy_metrics_for_json(policy_logs),
    }
    if checkpoint_path is not None:
      self.save(checkpoint_path, metrics=metrics)
    runner.wm_env.clear_rollout()
    self.refresh_collection()
    return metrics

  def state_dict(self) -> dict:
    state = super().state_dict()
    state.update(
      version="eit_training_v1",
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
    if state.get("version") != "eit_training_v1":
      raise ValueError(
        "EIT continuation requires eit_training_v1; model-only checkpoints require explicit weight import"
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
    # The inherited loader preflights model, model optimizer, actor, metadata and
    # RNG before any mutation. All additional EIT components passed above.
    super().load_state_dict(state)
    self.policy.critic.load_checkpoint(state["critic"])
    self.policy.load_state_dict(state["policy"])
    self.initialization_digest = initialization
    self.actor.eval().requires_grad_(True)
    self.policy.critic.eval().requires_grad_(True)

  def run(self, rounds: int, output: Path, *, resume: Path | None = None):
    checkpoint = Path(output).resolve() / "training.pt"
    if resume is None and self._output_checkpoint == checkpoint and checkpoint.exists():
      saved = torch.load(checkpoint, map_location=self.runner.device, weights_only=True)
      if parameter_digest(self.actor) != saved.get("actor_version") or parameter_digest(
        self.policy.critic
      ) != saved.get("critic_version"):
        raise ValueError("Active actor/critic differs from output checkpoint")
    return super().run(rounds, output, resume=resume)

  def initialization_bundle(self) -> dict:
    """Export explicit, independently allocated common A/B/C initial weights."""
    if (
      self.rounds
      or self.learner.steps
      or self.policy.actor_steps
      or self.policy.critic_steps
    ):
      raise ValueError("Common initialization can only be exported before training")
    runner = self.runner
    actors = build_ablation_actors(runner.cfg, runner._wm, runner.device)
    source = self.actor
    for actor in actors.values():
      actor.to(next(source.parameters()))
      actor.copy_common_initialization_from(source)
    critics = {}
    for group in ("A", "B", "C"):
      critic = BeliefCritic(
        group, runner._sim_context.output_dim, runner.cfg.critic_hidden_dims
      ).to(next(source.parameters()))
      critic.copy_common_initialization_from(self.policy.critic)
      critics[group] = critic
    # B/C include identical extra belief columns even when the source was A.
    critics["C"].copy_common_initialization_from(critics["B"])
    return deepcopy(
      {
        "version": "eit_initialization_v1",
        "world_model": runner._wm.bundle(),
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
      or self.learner.steps
      or self.policy.actor_steps
      or self.policy.critic_steps
    ):
      raise ValueError("Initialization cannot replace an active training run")
    if bundle.get("version") != "eit_initialization_v1":
      raise ValueError("Unsupported EIT initialization")
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
    digest = hashlib.sha256()

    def hash_value(value):
      if isinstance(value, torch.Tensor):
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
      elif isinstance(value, dict):
        for name in sorted(value):
          digest.update(name.encode())
          hash_value(value[name])
      elif isinstance(value, (tuple, list)):
        for item in value:
          hash_value(item)
      else:
        digest.update(json.dumps(value, sort_keys=True).encode())

    hash_value(bundle)
    identity = digest.hexdigest()
    runner._wm.load_bundle(bundle["world_model"])
    load_actor_checkpoint(self.actor, runner.cfg, bundle["actors"][group])
    self.policy.critic.load_checkpoint(bundle["critics"][group])
    self.initialization_digest = identity
    runner.wm_env.clear_rollout()
    self._ready = False
