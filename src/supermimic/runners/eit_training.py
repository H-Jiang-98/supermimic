"""Two-stage world model fitting and stochastic value-gradient policy training.

Step checkpoints reuse common weight files and the model loop's atomic state
writer. Optimizer continuation is exact for fixed raw inputs/noises; simulator
and replay state are deliberately restarted after restoration.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import cast

import torch
import wandb
from filelock import FileLock
from tensordict import TensorDict

from supermimic.critic_inputs import BeliefCritic, critic_state_features
from supermimic.envs.world_model_env import BeliefRolloutStart
from supermimic.estimator_inputs import (
  EpisodeAlignment,
  RpvState,
  normalization_motion_goals,
)
from supermimic.learners.belief_policy_learner import (
  BeliefActorCriticPolicyLearner,
  BeliefPolicyLearnerCfg,
  policy_metrics_for_json,
)
from supermimic.models import _CudaGraphNormalization
from supermimic.posterior_replay import POSTERIOR_REPLAY_VERSION, unpack_mean
from supermimic.representations import GenCoordRepresentation
from supermimic.runners.belief_training import (
  BeliefModelTraining,
  parameter_digest,
)
from supermimic.runners.eit_snapshots import file_sha256
from supermimic.runners.run_config import (
  EIT_INITIALIZATION_VERSION,
  EIT_POLICY_TRAINING_VERSION,
  validate_eit_checkpoint_contract,
)
from supermimic.runners.supermimic_runner import (
  Ratio,
  SuperMimicRunner,
  actor_checkpoint,
  build_ablation_actors,
  build_actor,
  load_actor_checkpoint,
)
from supermimic.utils.lie_state import LieState
from supermimic.utils.profiling import PhaseTimer, StreamTimer, runtime_metadata
from supermimic.utils.validation import validate_rotation
from supermimic.utils.wandb_artifacts import log_model_artifact

# The checkpoint rename commits a prepared set. A process interrupted during the
# remaining renames leaves the committed files here for the next locked recovery.
_PUBLICATION = ".eit-publication"
_INITIALIZING = ".eit-initializing"
_INTERVAL = ".eit-log-interval.json"
_WEIGHT_FILES = ("world_model.pt", "actor.pt", "critic.pt")


def _finish_publication(output: Path) -> None:
  staged = output / _PUBLICATION
  if staged.is_symlink():
    raise ValueError("Training publication directory must be local")
  if not staged.exists():
    return
  if any(staged.iterdir()) and not (staged / "staging").is_file():
    raise ValueError("Unrecognized training publication directory")
  if (staged / "prepared").is_file() and not (staged / "training.pt").exists():
    for name in (*_WEIGHT_FILES, _INTERVAL):
      source = staged / name
      if source.exists():
        source.replace(output / name)
  # Clear the commit marker before discarding any uncommitted staged weights.
  (staged / "prepared").unlink(missing_ok=True)
  # Remove the ownership marker last so an interrupted cleanup is repeatable.
  for path in staged.iterdir():
    if path.name != "staging":
      path.unlink()
  (staged / "staging").unlink(missing_ok=True)
  staged.rmdir()


@dataclass(frozen=True)
class EITTrainingCfg:
  validation_stride: int = 5
  validation_batch_size: int = 4
  normalization_batch_size: int = 16
  posterior_save_stride: int = 1

  def __post_init__(self):
    if type(self.validation_stride) is not int or self.validation_stride < 2:
      raise ValueError("validation_stride must be at least two")
    if type(self.validation_batch_size) is not int or self.validation_batch_size < 1:
      raise ValueError("validation_batch_size must be positive")
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
    *,
    policy_cuda_graphs: bool = False,
  ):
    if runner.replay_buffer.replay_mode != "full":
      raise ValueError("EIT policy starts require full replay")
    if (
      runner.cfg.replay_ratio != 0
      and runner.replay_buffer.buffer_size < runner.cfg.mean.window_frames
    ):
      raise ValueError(
        f"buffer_size ({runner.replay_buffer.buffer_size}) must cover "
        f"mean.window_frames ({runner.cfg.mean.window_frames}) when training is enabled"
      )
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
      compile_kernels=getattr(runner, "compile_eit", False),
      cuda_graphs=policy_cuda_graphs,
    )
    self.initialization_digest: str | None = None
    self.collected_steps = self.update_groups = self.skipped_update_groups = 0
    self._posterior_normalized_until = 0
    self._last_metrics = None
    self._wandb_run = None
    mean = self.learner.mean_cfg
    self.should_train = Ratio(
      runner.env.num_envs
      * runner.cfg.replay_ratio
      / (mean.num_batches * mean.batch_size * mean.window_frames)
    )
    if not math.isfinite(self.should_train._ratio):
      raise ValueError("replay_ratio must be finite")
    if runner.cfg.cuda_graphs:
      raise ValueError("EIT does not support explicit learner CUDA graphs")

  @property
  def policy(self) -> BeliefActorCriticPolicyLearner:
    return self.runner.policy_learner

  @torch.no_grad()
  def update_input_normalization(
    self, frames=None, policy_keys: torch.Tensor | None = None
  ) -> BeliefRolloutStart | None:
    """Normalize newly saved training slots or an explicit frame selection."""
    timer = StreamTimer(self.runner.device)
    with timer.measure("posterior_cache_normalization_seconds"):
      starts = self._normalize_saved_posteriors(frames, policy_keys, timer)
      if frames is None:
        self._posterior_normalized_until = self.runner.replay_buffer.insert_count
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
        first = self._posterior_normalized_until
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

    replay.finish_collection_round()
    inputs = env.estimator_input_config
    assert inputs is not None
    live = (
      self.actor.obs_normalizer,
      cast(_CudaGraphNormalization, self.actor.goal_normalizers["motion"]),
      *cast(BeliefCritic, self.policy.critic).normalizers(),
    )
    staged = deepcopy(live)
    for module in staged:
      module.train()
    storage = replay._storage.select(
      "state", "actor_obs", "motion_time", "alignment_rotation", "alignment_translation"
    )
    batch_size = self.cfg.normalization_batch_size
    try:
      for offset in range(0, count, batch_size):
        selection = slice(offset, offset + batch_size)
        rows, slots, pslots = (index[selection] for index in fresh_locations)
        with timer.measure("posterior_cache_read_seconds"):
          batch = cast(TensorDict, storage[rows, slots])
          posterior = None
          if inputs.uses_posterior:
            mean = unpack_mean(cache.storage["posterior_mean"][rows, pslots]).to(
              dtype=batch["state"].dtype
            )
            posterior = RpvState(mean.rotation, mean.position, mean.velocity)
        values = self._normalization_inputs(batch, posterior)
        for module, value in zip(staged, values, strict=True):
          module.update(value)
      if not torch.stack(
        [value.isfinite().all() for module in staged for value in module.buffers()]
      ).all():
        raise ValueError("Non-finite actor or critic normalization statistics")
      for destination, source in zip(live, staged, strict=True):
        destination.load_state_dict(source.state_dict())
      return starts
    finally:
      env.clear_rollout()
      self.actor.eval()
      self.policy.critic.eval()

  def _normalization_inputs(self, batch, posterior):
    """Construct the five fitted channels in actor/critic normalizer order."""
    env = self.runner.wm_env
    representation, motion = (
      cast(GenCoordRepresentation, env.representation),
      env.motion,
    )
    times = batch["motion_time"]
    if ((times < 0) | (times >= motion.motion_ends[-1])).any():
      raise ValueError("Invalid normalization reference frame")
    state, sensor = batch["state"], batch["actor_obs"]
    alignment = EpisodeAlignment(
      batch["alignment_rotation"], batch["alignment_translation"]
    )
    validate_rotation(alignment.rotation)
    if not torch.stack(
      [v.isfinite().all() for v in (state, sensor, alignment.translation)]
    ).all():
      raise ValueError("Non-finite replay input to actor/critic normalization")
    reference = representation.reference_window(motion, times, alignment)
    aux = representation.extract_aux_state(state)
    rep = representation.represent(state)
    actor_goal, critic_goal = normalization_motion_goals(
      representation.true_rpv(aux, rep), posterior, reference
    )
    critic_state = critic_state_features(LieState.from_rep(aux, rep))
    # Both history normalizers count only the latest physical measurement.
    return sensor, actor_goal, critic_state, critic_goal, sensor

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
      starts = runner.wm_env.prepare_policy_starts(*policy_keys.unbind(dim=1))
      assert starts is not None
      policy_logs = self.policy.update_prepared(starts)
    self._policy_profile = self.policy.last_profile.copy()
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
    sampler = getattr(self.runner, "_sampler", None)
    state.update(
      stage="policy",
      collected_steps=self.collected_steps,
      skipped_update_groups=self.skipped_update_groups,
      scheduler=self.should_train.state_dict(),
      return_sampler=sampler.state_dict() if sampler is not None else None,
      wandb_run=self._wandb_run,
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
      raise ValueError("EIT step training requires the current checkpoint protocol")

  def load_state_dict(self, state: dict) -> None:
    self._check_checkpoint_version(state)
    scheduler = deepcopy(self.should_train)
    scheduler.load_state_dict(state.get("scheduler"))
    for name in ("collected_steps", "skipped_update_groups"):
      if type(state.get(name)) is not int or state[name] < 0:
        raise ValueError(f"Invalid checkpoint {name}")
    if state["collected_steps"] * self.runner.env.num_envs != state["collected_frames"]:
      raise ValueError("Checkpoint collection counters disagree")
    if scheduler._prev is not None and scheduler._prev > state["collected_steps"]:
      raise ValueError("Scheduler is ahead of collection")
    sampler = getattr(self.runner, "_sampler", None)
    if (sampler is None) != (state.get("return_sampler") is None):
      raise ValueError("Checkpoint ReturnSampler configuration mismatch")
    if sampler is not None:
      deepcopy(sampler).load_state_dict(state["return_sampler"])
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
    self.should_train = scheduler
    self.collected_steps = state["collected_steps"]
    self.skipped_update_groups = state["skipped_update_groups"]
    self._posterior_normalized_until = 0
    self._last_metrics = state.get("last_metrics")
    self._wandb_run = state.get("wandb_run")
    self.actor.eval().requires_grad_(True)
    self.policy.critic.eval().requires_grad_(True)

  def _restart_after_restore(self, state):
    sampler = getattr(self.runner, "_sampler", None)
    if sampler is not None:
      sampler.load_state_dict(state["return_sampler"])
      self.runner.motion_cmd.set_sampling_distribution(sampler.weights)
    self.refresh_collection()

  def run_round(self, *args, **kwargs):
    raise TypeError("EIT uses run_step() or run(max_iterations, output)")

  @staticmethod
  def _repair_logs(output, state):
    """Keep committed step records, preserving torn/ahead bytes for inspection."""
    interval_path = output / _INTERVAL
    if interval_path.is_symlink():
      raise ValueError("Training interval record must be local")
    interval = json.loads(interval_path.read_bytes()) if interval_path.exists() else {}
    for name in ("metrics.jsonl", "profiling.jsonl"):
      pending = interval.get(name)
      if pending is not None and pending.get("step") != state["collected_steps"] - 1:
        raise ValueError("Training interval record disagrees with checkpoint")
      path = output / name
      if path.is_symlink():
        raise ValueError("Training logs must be local files")
      previous = -1
      if path.exists():
        with path.open("r+b") as stream:
          end = 0
          for line in stream:
            if not line.endswith(b"\n"):
              break
            row = json.loads(line)
            step = row.get("step")
            if type(step) is not int or step <= previous:
              raise ValueError("Training log step order is invalid")
            if step >= state["collected_steps"]:
              break
            end += len(line)
            previous = step
          stream.seek(end)
          tail = stream.read()
          if tail:
            archive = path.with_name(
              f"{name}.interrupted.{hashlib.sha256(tail).hexdigest()}"
            )
            if archive.is_symlink():
              raise ValueError("Training recovery archive must be local")
            archive.write_bytes(tail)
            stream.truncate(end)
      if pending is not None and previous < pending["step"]:
        # The logger may not have appended either row before interruption. Each
        # file is repaired independently so a crash between appends is retryable.
        with path.open("a") as stream:
          stream.write(json.dumps(pending, allow_nan=False) + "\n")

  @classmethod
  def reconcile_output(cls, output, *, validate_state=None):
    output = Path(output).resolve()
    checkpoint, lock = output / "training.pt", output / ".belief-training.lock"
    if any(
      p.is_symlink()
      for p in (
        checkpoint,
        lock,
        output / _INTERVAL,
        *(output / n for n in _WEIGHT_FILES),
      )
    ):
      raise ValueError("Training recovery requires local files")
    with FileLock(lock, timeout=0):
      state = torch.load(checkpoint, map_location="cpu", weights_only=True)
      cls._check_checkpoint_version(state)
      if validate_state is not None:
        validate_state(state)
      _finish_publication(output)
      # Before staged publication, a failed save could leave weights from an
      # uncommitted step. Install the authoritative copies once for legacy sets.
      if not (output / _INTERVAL).exists() and any(
        (output / name).exists() for name in _WEIGHT_FILES
      ):
        staged = output / _PUBLICATION
        staged.mkdir()
        try:
          (staged / "staging").touch()
          for name in _WEIGHT_FILES:
            torch.save(state[Path(name).stem], staged / name)
          (staged / _INTERVAL).write_text("{}")
          (staged / "prepared").touch()
        finally:
          _finish_publication(output)
      cls._repair_logs(output, state)
      return {
        "collected_steps": state["collected_steps"],
        "collected_frames": state["collected_frames"],
        "checkpoint_sha256": file_sha256(checkpoint),
      }

  def run_step(self) -> dict:
    """One physical collection step and every group due under the common Ratio."""
    runner, learner = self.runner, self.learner
    timer = PhaseTimer(runner.device)
    if not self._ready:
      self.refresh_collection()
    with timer.measure("physical_collection_seconds"):
      runner.step()
    step = self.collected_steps
    self.collected_steps += 1
    self.collected_frames += runner.env.num_envs
    runner.finish_collection_round()
    # Count new physical inputs exactly once, including during warmup and steps
    # with zero training groups. Multiple groups below see frozen normalization.
    with timer.measure("model_normalization_seconds"):
      fresh = cast(TensorDict, self._fresh_training_frames(1, ("state", "action")))
      if len(fresh):
        learner.update_normalization(fresh["state"], fresh["action"])
    self.update_input_normalization()
    due = self.should_train(step)
    logs = []
    if due:
      lengths = runner.replay_buffer.training_episode_lengths()
      train_ids = [e for e in lengths if e % self.cfg.validation_stride]
      valid_ids = [e for e in lengths if not e % self.cfg.validation_stride]
      windows, _, _, held_out = runner.replay_buffer.model_fit_starts(
        train_ids, valid_ids, learner.mean_cfg.window_frames
      )
      ready = bool(
        len(windows) and len(held_out) and self._cache_budget["available_policy_starts"]
      )
      if not ready:
        self.skipped_update_groups += due
      else:
        for _ in range(due):
          with timer.measure("world_model_seconds"):
            train, validation = learner.fit(
              runner.replay_buffer,
              train_ids,
              valid_ids,
              fresh[:0],
              self.cfg.validation_batch_size,
            )
          for name, seconds in learner.last_profile.items():
            timer.seconds[name] = timer.seconds.get(name, 0.0) + seconds
          with timer.measure("policy_seconds"):
            policy = self._update_policy()
          for name, seconds in self._policy_profile.items():
            timer.seconds[name] = timer.seconds.get(name, 0.0) + seconds
          self.update_groups += 1
          logs.append(
            {
              "train": train,
              "validation": validation,
              **policy,
              "budget": {
                **self._policy_budget,
                "mean_updates": train["mean_updates"],
                "covariance_updates": train["cov_updates"],
                "available_training_pairs": train["cov_pairs"],
                "available_validation_pairs": validation["cov_pairs"],
                "covariance_pair_draws": train["cov_pair_draws"],
                "covariance_validation_draws": validation["cov_pair_draws"],
                "mean_windows": train["mean_windows"],
                "available_training_windows": train["available_mean_windows"],
              },
            }
          )
    metrics = {
      "step": step,
      "collected_steps": self.collected_steps,
      "collected_frames": self.collected_frames,
      "update_groups": self.update_groups,
      "skipped_update_groups": self.skipped_update_groups,
      "mean_steps": learner.mean_steps,
      "cov_steps": learner.cov_steps,
      "actor_steps": self.policy.actor_steps,
      "critic_steps": self.policy.critic_steps,
      "groups_this_step": len(logs),
      "updates": logs,
      "normalization_samples": self._cache_budget["normalization_samples"],
      "profiling": {
        "seconds": {**timer.seconds, **self._normalization_profile},
        "runtime": runtime_metadata(next(self.actor.parameters()))
        | {"policy_cuda_graph": self.policy.graph_status},
      },
    }
    json.dumps(metrics, allow_nan=False)
    self._last_metrics = metrics
    self.last_profile = metrics["profiling"]
    return metrics

  def save(self, path: Path, *, metrics: dict | None = None) -> None:
    super().save(path, metrics=self._last_metrics if metrics is None else metrics)

  def run(self, max_iterations: int, output: Path, *, resume: Path | None = None):
    """Collect to a cumulative target, using common W&B, Logger and weight files."""
    if type(max_iterations) is not int or max_iterations < 0:
      raise ValueError("max_iterations must be nonnegative")
    runner = self.runner
    for name in ("save_interval", "log_interval"):
      if type(getattr(runner.cfg, name)) is not int or getattr(runner.cfg, name) < 1:
        raise ValueError(f"{name} must be positive")
    os.environ.setdefault("WANDB_MODE", "offline")
    if os.environ["WANDB_MODE"] not in ("offline", "disabled"):
      raise ValueError("EIT supports WANDB_MODE=offline or disabled")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "training.pt"
    lock = output / ".belief-training.lock"
    initializing = output / _INITIALIZING
    protected = (
      checkpoint,
      lock,
      initializing,
      *(
        output / name
        for name in (
          "metrics.jsonl",
          "profiling.jsonl",
          _PUBLICATION,
          _INTERVAL,
          "world_model.pt",
          "actor.pt",
          "critic.pt",
        )
      ),
    )
    if any(path.is_symlink() for path in protected):
      raise ValueError("Training output requires local files")
    with FileLock(lock, timeout=0):
      entries = set(output.iterdir()) - {lock}
      occupied = bool(entries)
      source = Path(resume).resolve() if resume is not None else self._output_checkpoint
      # Only a previously empty output claimed before W&B initialization can be
      # retried without a checkpoint. Preserve SDK offline runs across attempts.
      abandoned_initial = (
        not checkpoint.exists()
        and initializing.is_file()
        and initializing.stat().st_size == 0
        and all(
          p.name in (_PUBLICATION, "wandb", "wandb-data", "wandb-cache")
          and p.is_dir()
          and not p.is_symlink()
          for p in entries - {initializing}
        )
      )
      if (
        occupied
        and not abandoned_initial
        and (source != checkpoint or not checkpoint.is_file())
      ):
        raise ValueError(
          "Nonempty output requires resuming its own current training.pt"
        )
      if resume is not None:
        self.load_state_dict(
          torch.load(resume, map_location=runner.device, weights_only=True)
        )

      elif occupied and not abandoned_initial:
        saved = torch.load(checkpoint, map_location=runner.device, weights_only=True)
        if (
          saved["collected_steps"] != self.collected_steps
          or saved["learner"]["model_version"] != runner._dynamics.model_version
          or saved["actor_version"] != parameter_digest(self.actor)
          or saved["critic_version"] != parameter_digest(self.policy.critic)
        ):
          raise ValueError("Active training state differs from the output checkpoint")
      if max_iterations < self.collected_steps:
        raise ValueError("max_iterations is below the restored cumulative step")
      if occupied:
        _finish_publication(output)
      if checkpoint.is_file():
        self._repair_logs(output, {"collected_steps": self.collected_steps})
        initializing.unlink(missing_ok=True)
      elif not abandoned_initial:
        initializing.touch(exist_ok=False)
      runner.cfg.max_iterations = max_iterations
      self._output_checkpoint = checkpoint
      provenance = {
        "source_checkpoint": str(source) if source is not None else None,
        "starting_collected_steps": self.collected_steps,
        "starting_collected_frames": self.collected_frames,
        "parent_run": self._wandb_run,
      }
      os.environ.setdefault("WANDB_CACHE_DIR", str(output / "wandb-cache"))
      os.environ.setdefault("WANDB_DATA_DIR", str(output / "wandb-data"))
      try:
        runner._init_wandb(
          output,
          extra_config={
            "continuation": provenance,
            "eit_loop": asdict(self.cfg),
            "eit_policy": asdict(self.policy.cfg),
          },
          fresh_run=True,
        )
      except Exception:
        # A failed SDK initialization may already have created log files. Leave
        # a resumable initial state instead of an occupied, unrecoverable output.
        try:
          if not checkpoint.exists():
            staged = output / _PUBLICATION
            staged.mkdir()
            try:
              (staged / "staging").touch()
              runner.save_weights(staged)
              self.save(staged / "training.pt")
              (staged / _INTERVAL).write_text("{}")
              (staged / "prepared").touch()
              (staged / "training.pt").replace(checkpoint)
              initializing.unlink()
            finally:
              _finish_publication(output)
        finally:
          runner._cleanup()
        raise
      run = wandb.run
      assert run is not None
      self._wandb_run = {"id": run.id, "name": run.name, "project": run.project}
      runner.logger.tot_timesteps = self.collected_frames
      runtime_metadata(next(self.actor.parameters()), reset_peak=True)
      upload = None
      last_saved = None
      collected = 0
      seconds, budget, latest = {}, {}, {}
      training_peaks = {}
      history = []
      log_count = 0
      interval = runner.cfg.log_interval
      total_logs = 0
      if max_iterations > self.collected_steps:
        total_logs = (
          (max_iterations - 1) // interval
          - (self.collected_steps - 1) // interval
          + int((max_iterations - 1) % interval != 0)
        )
      mean_loss_keys = tuple(
        f"Loss/{name}" for name in ("state", *self.learner.mean.loss_group_weights)
      )
      console_keys = (
        *mean_loss_keys,
        "Policy/reward",
        "Policy/grad_norm",
        "Policy/value_function",
      )

      def publish():
        nonlocal upload, last_saved
        key = (self.collected_steps, self.update_groups)
        if key == last_saved:
          return
        if upload is not None:
          upload.result()  # Finish staging the previous file versions before overwrite.
        timer = PhaseTimer(runner.device)
        staged = output / _PUBLICATION
        staged.mkdir()
        try:
          (staged / "staging").touch()
          with timer.measure("checkpoint_save_seconds"):
            runner.save_weights(staged)
            self.save(staged / "training.pt")
          reference_seconds = self._last_checkpoint_hook_seconds
          if reference_seconds:
            seconds["resume_reference_seconds"] = (
              seconds.get("resume_reference_seconds", 0.0) + reference_seconds
            )
            # The disposable reference execution resets CUDA's peak counters.
            # Keep the preceding training high-water marks across that reset.
            assert self._checkpoint_hook_runtime_before is not None
            for name in ("cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes"):
              peak = self._checkpoint_hook_runtime_before[name]
              if peak is not None:
                training_peaks[name] = max(peak, training_peaks.get(name, 0))
          seconds["checkpoint_save_seconds"] = (
            seconds.get("checkpoint_save_seconds", 0.0)
            + timer.seconds["checkpoint_save_seconds"]
            - reference_seconds
          )
          if collected:
            rows = log_rows(self.collected_steps - 1)
            (staged / _INTERVAL).write_text(json.dumps(rows, allow_nan=False))
          elif not (output / _INTERVAL).exists():
            (staged / _INTERVAL).write_text("{}")
          (staged / "prepared").touch()
          (staged / "training.pt").replace(checkpoint)
        finally:
          _finish_publication(output)
        initializing.unlink(missing_ok=True)
        paths = [str(output / name) for name in _WEIGHT_FILES]
        upload = uploader.submit(
          log_model_artifact,
          [*paths, str(checkpoint)],
          run=run,
          metadata={
            "collected_steps": self.collected_steps,
            "collected_frames": self.collected_frames,
            "update_groups": self.update_groups,
          },
        )
        last_saved = key

      def log_rows(step):
        counters = {
          "Counters/" + name: getattr(self, name)
          for name in (
            "collected_steps",
            "collected_frames",
            "update_groups",
            "skipped_update_groups",
          )
        }
        counters.update(
          {
            "Counters/" + name: value
            for name, value in (
              ("mean_updates", self.learner.mean_steps),
              ("covariance_updates", self.learner.cov_steps),
              ("actor_updates", self.policy.actor_steps),
              ("critic_updates", self.policy.critic_steps),
            )
          }
        )
        losses = {
          **latest,
          **counters,
          **{"Budget/" + k: v for k, v in budget.items()},
          **{"Perf/" + k: v for k, v in seconds.items()},
        }
        # Keep original EIT diagnostics for reports and checkpoint log recovery.
        for key in mean_loss_keys:
          source = "Model/" + key
          if source in losses:
            losses[key] = losses[source]
        for key, source in (
          ("Policy/grad_norm", "Grad/actor_norm"),
          ("Policy/value_function", "Policy/critic_loss"),
        ):
          if source in losses:
            losses[key] = losses[source]
        if "Policy/return" in losses:
          losses["Policy/reward"] = losses["Policy/return"] / (
            self.policy.cfg.horizon * runner.wm_env.representation.dt
          )
        runtime = runtime_metadata(next(self.actor.parameters()))
        runtime["policy_cuda_graph"] = self.policy.graph_status
        runtime["eit_execution"] = "eager"
        if getattr(runner, "compile_eit", False):
          from torch._dynamo.utils import calculate_time_spent, counters

          runtime["eit_execution"] = "inductor"
          runtime["compile_seconds_total"] = calculate_time_spent()["total_wall_time"]
          runtime["compiled_graphs_total"] = counters["stats"]["unique_graphs"]
        for name, peak in training_peaks.items():
          runtime[name] = max(runtime[name], peak)
        losses.update(
          {
            "Perf/" + k: v
            for k, v in runtime.items()
            if k.startswith("cuda_peak") and v is not None
          }
        )
        return {
          "metrics.jsonl": {"step": step, **losses},
          "profiling.jsonl": {"step": step, "seconds": seconds, "runtime": runtime},
        }

      def log(step):
        nonlocal collected, log_count
        rows = log_rows(step)
        row = rows["metrics.jsonl"]
        runner.logger.log(
          it=log_count,
          total_it=total_logs,
          collect_time=seconds.get("physical_collection_seconds", 0.0),
          wm_learn_time=seconds.get("world_model_seconds", 0.0),
          policy_learn_time=seconds.get("policy_seconds", 0.0),
          loss_dict={k: v for k, v in row.items() if k != "step"},
          num_collected_steps=collected,
          console_keys=console_keys,
        )
        for name, record in rows.items():
          with (output / name).open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        history.append(row)
        log_count += 1
        collected = 0
        seconds.clear()
        budget.clear()

      try:
        with ThreadPoolExecutor(max_workers=1) as uploader:
          publish()
          for step in range(self.collected_steps, max_iterations):
            metrics = self.run_step()
            collected += runner.env.num_envs
            for name, value in metrics["profiling"]["seconds"].items():
              seconds[name] = seconds.get(name, 0.0) + value
            for update in metrics["updates"]:
              for section, prefix in (
                ("train", "Model/"),
                ("validation", "ModelValidation/"),
                ("policy", ""),
              ):
                latest.update(
                  {prefix + k: v for k, v in update[section].items() if v is not None}
                )
              for name, value in update["budget"].items():
                if name in (
                  "mean_updates",
                  "covariance_updates",
                  "actor_updates",
                  "critic_updates",
                  "covariance_pair_draws",
                  "covariance_validation_draws",
                  "mean_windows",
                  "policy_starts_drawn",
                  "imagined_steps",
                  "survival_weighted_imagined_steps",
                ):
                  budget[name] = budget.get(name, 0) + value
                else:
                  latest["Budget/" + name] = value
            budget["normalization_samples"] = (
              budget.get("normalization_samples", 0) + metrics["normalization_samples"]
            )
            if step % runner.cfg.save_interval == 0 or step == max_iterations - 1:
              publish()
            if step % runner.cfg.log_interval == 0:
              log(step)
          publish()
          if collected:
            log(self.collected_steps - 1)
          if upload is not None:
            upload.result()
      finally:
        runner._cleanup()
      return history

  def initialization_bundle(self) -> dict:
    """Export explicit, independently allocated common A/B/C initial weights."""
    if (
      self.update_groups
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
      self.update_groups
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
