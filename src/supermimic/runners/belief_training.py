"""Round-based mean and covariance fitting with a fixed collection policy.

Checkpoints resume model optimization at a round boundary. Physical simulator
state is deliberately reset; replay and online posteriors are rebuilt from new
measurements. They are never restored under a changed model.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from shutil import copyfileobj
from tempfile import NamedTemporaryFile
from typing import cast

import torch
from filelock import FileLock

from supermimic.belief_provider import belief_rollout
from supermimic.learners.two_stage_wm import TwoStageWorldModelLearner
from supermimic.models import MultiEncoderAE
from supermimic.motion_loader import motion_content_digest
from supermimic.runners.eit_snapshots import (
  checkpoint_snapshot_interval,
  file_sha256,
  publish_snapshot,
)
from supermimic.runners.run_config import (
  EIT_TRAINING_VERSION,
  validate_eit_checkpoint_contract,
)
from supermimic.runners.supermimic_runner import (
  SuperMimicRunner,
  actor_checkpoint,
  load_actor_checkpoint,
)
from supermimic.utils.profiling import PhaseTimer, runtime_metadata


@dataclass(frozen=True)
class BeliefTrainingCfg:
  collection_steps: int = 64
  validation_stride: int = 5
  validation_batch_size: int = 4

  def __post_init__(self):
    for name in ("collection_steps", "validation_stride", "validation_batch_size"):
      value = getattr(self, name)
      if type(value) is not int or value < (
        2 if name != "validation_batch_size" else 1
      ):
        raise ValueError(f"Invalid {name}: {value}")


def parameter_digest(module: torch.nn.Module) -> str:
  digest = hashlib.sha256()
  for name, value in module.state_dict().items():
    digest.update(name.encode())
    digest.update(value.detach().cpu().contiguous().numpy().tobytes())
  return digest.hexdigest()


class BeliefModelTraining:
  """Collect, normalize training episodes, optimize, validate, and reset.

  Episode IDs divisible by validation_stride are permanently held out. The
  split never depends on which fragments survive buffer overwrite. Both sets
  must contain a valid transition; increase collection length/env count if not.
  """

  def __init__(self, runner: SuperMimicRunner, cfg: BeliefTrainingCfg | None = None):
    if not isinstance(runner.world_model_learner, TwoStageWorldModelLearner):
      raise ValueError("Belief training requires estimator_inputs and process_noise")
    if (
      runner.cfg.estimator_inputs is None or runner.cfg.estimator_inputs.mode != "train"
    ):
      raise ValueError("Belief model collection requires estimator_inputs.mode='train'")
    if not runner.cfg.freeze_estimator_normalization:
      raise ValueError("Belief training requires fixed normalization during collection")
    if runner.cfg.buffer_size < 2:
      raise ValueError("Belief training needs at least two retained frames")
    self.runner = runner
    self.cfg = cfg or BeliefTrainingCfg()
    # Historical snapshots belong only to the round-based model-only entry point.
    self.snapshot_interval = 1
    self.rounds = 0
    self.collected_frames = 0
    self.training_seed = getattr(getattr(runner.env, "cfg", None), "seed", None)
    self.training_motion_files = [
      {"path": str(path), "sha256": file_sha256(path)}
      for path in (Path(f).resolve() for f in runner.motion.files)
    ]
    self._ready = False
    self.continuous_collection = False
    self.before_checkpoint_commit: Callable[[Path, dict], None] | None = None
    # This entry point does not optimize the policy or change its normalizers.
    runner.policy_learner.actor.eval().requires_grad_(False)

  @property
  def learner(self) -> TwoStageWorldModelLearner:
    return cast(TwoStageWorldModelLearner, self.runner.world_model_learner)

  @property
  def actor(self) -> MultiEncoderAE:
    return cast(MultiEncoderAE, self.runner.policy_learner.actor)

  def refresh_collection(self):
    self.runner.configure_belief_filter()
    self.runner.reset_collection()
    self._ready = True

  def episode_split(self):
    lengths = self.runner.replay_buffer.training_episode_lengths()
    train = [e for e, n in lengths.items() if n > 1 and e % self.cfg.validation_stride]
    valid = [
      e for e, n in lengths.items() if n > 1 and not e % self.cfg.validation_stride
    ]
    if not train or not valid:
      raise ValueError(
        "Need valid transitions in both training and held-out episodes; "
        "increase collection_steps, buffer_size or num_envs"
      )
    return train, valid

  def _fresh_training_frames(self, count: int, fields: tuple[str, ...]):
    frames = self.runner.replay_buffer.retained_frames(
      newest_count=count, fields=("episode_id", *fields)
    )
    return frames[frames["episode_id"] % self.cfg.validation_stride != 0]

  def _update_policy(self) -> dict:
    return {}

  def run_round(self, checkpoint_path: Path | None = None) -> dict:
    runner, learner = self.runner, self.learner
    timer = PhaseTimer(runner.device)
    round_start = timer.now()
    runtime_metadata(next(self.actor.parameters()), reset_peak=True)
    if not self._ready:
      with timer.measure("initial_reset_seconds"):
        self.refresh_collection()
    version_before = self._model_identity()
    actor_before = parameter_digest(self.actor)
    replay = runner.replay_buffer
    # Fold each chunk into private statistics before the next ring overwrite.
    staged = None
    steps = self.cfg.collection_steps
    terminated = truncated = completed = 0

    def count_endings(count):
      nonlocal terminated, truncated, completed
      flags = replay.retained_frames(
        newest_count=count, fields=("terminated", "truncated")
      )
      terminated += int(flags["terminated"].sum())
      truncated += int(flags["truncated"].sum())
      completed += int((flags["terminated"] | flags["truncated"]).sum())

    collect_start = timer.now()
    with runner._dynamics.readonly(), belief_rollout(runner.posterior_provider):
      staged = self._collect_round_steps(steps, replay, learner, count_endings, staged)
    remaining = (steps - 1) % replay.buffer_size + 1
    count_endings(remaining)
    timer.seconds["physical_collection_seconds"] = timer.now() - collect_start
    if (
      self._model_identity() != version_before
      or parameter_digest(self.actor) != actor_before
    ):
      raise RuntimeError("Collection model or actor changed within a round")
    runner.finish_collection_round()
    self._ready = self.continuous_collection

    train_ids, valid_ids = self.episode_split()
    train, validation = learner.fit(
      replay,
      train_ids,
      valid_ids,
      self._fresh_training_frames(remaining, ("state", "action")),
      self.cfg.validation_batch_size,
      staged_normalizers=staged,
    )
    trained_version = self._model_identity()
    policy = self._update_policy()
    if self._model_identity() != trained_version:
      raise RuntimeError("Policy update modified the frozen world model")
    metrics = {
      "round": self.rounds + 1,
      "mean_steps": learner.mean_steps,
      "cov_steps": learner.cov_steps,
      "collected_frames": self.collected_frames
      + self.cfg.collection_steps * runner.env.num_envs,
      "training_episodes": train_ids,
      "validation_episodes": valid_ids,
      "model_identity_kind": "revision" if self.continuous_collection else "sha256",
      "collection_model_version": version_before,
      "trained_model_version": trained_version,
      "collection_actor_version": actor_before,
      "trained_actor_version": parameter_digest(self.actor),
      "train": train,
      "validation": validation,
      **policy,
      "budget": {
        "collected_frames": steps * runner.env.num_envs,
        "available_training_windows": train["available_mean_windows"],
        "sampled_training_windows": train["mean_windows"],
        "available_validation_windows": validation["available_mean_windows"],
        "sampled_validation_windows": validation["mean_windows"],
        "training_pairs": train["cov_pairs"],
        "validation_pairs": validation["cov_pairs"],
        "covariance_pair_draws": train["cov_pair_draws"],
        "mean_updates": train["mean_updates"],
        "covariance_updates": train["cov_updates"],
        "covariance_batch_size": learner.cov_cfg.batch_size,
        "covariance_pair_budget": learner.cov_cfg.training_pairs(learner.mean_cfg),
        "covariance_validation_draws": validation["cov_pair_draws"],
        "physical_terminations": terminated,
        "physical_truncations": truncated,
        "physical_episode_endings": completed,
        "physical_termination_frequency": terminated / (steps * runner.env.num_envs),
        "physical_episode_ending_frequency": completed / (steps * runner.env.num_envs),
        **getattr(self, "_policy_budget", {}),
      },
    }
    metrics["profiling"] = {
      "version": "eit_training_profile_v1",
      "runtime": runtime_metadata(next(self.actor.parameters())),
      "warmup_rounds_in_process": getattr(self, "_profiled_rounds", 0),
      "seconds": {
        **timer.seconds,
        **learner.last_profile,
        **getattr(self, "_policy_profile", {}),
      },
    }
    # Verify the entire row before advancing the round or replacing its checkpoint.
    json.dumps(metrics, allow_nan=False)
    self.rounds = metrics["round"]
    self.collected_frames = metrics["collected_frames"]
    if checkpoint_path is not None:
      with timer.measure("checkpoint_save_seconds"):
        self.save(checkpoint_path, metrics=metrics)
      reference_seconds = getattr(self, "_last_checkpoint_hook_seconds", 0.0)
      if reference_seconds:
        timer.seconds["checkpoint_save_seconds"] -= reference_seconds
        timer.seconds["resume_reference_seconds"] = reference_seconds
    if not self.continuous_collection:
      with timer.measure("refresh_reset_seconds"):
        self.refresh_collection()
    after_save = runtime_metadata(next(self.actor.parameters()))
    before_hook = getattr(self, "_checkpoint_hook_runtime_before", None)
    if before_hook is not None:
      for key in ("cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes"):
        if before_hook.get(key) is not None:
          after_save[key] = max(before_hook[key], after_save[key])
    self.last_profile = {
      "round": self.rounds,
      "collected_frames": self.collected_frames,
      **metrics["profiling"],
      "runtime_after_save": after_save,
      "seconds": {**metrics["profiling"]["seconds"], **timer.seconds},
    }
    elapsed = timer.now() - round_start
    reference_seconds = timer.seconds.get("resume_reference_seconds", 0.0)
    self.last_profile["seconds"]["round_seconds"] = elapsed - reference_seconds
    if reference_seconds:
      self.last_profile["seconds"]["round_seconds_including_resume_reference"] = elapsed
      self.last_profile["resume_reference_runtime"] = self._checkpoint_hook_runtime
    self._profiled_rounds = getattr(self, "_profiled_rounds", 0) + 1
    return metrics

  def _model_identity(self):
    model = self.runner._dynamics
    return model.model_revision if self.continuous_collection else model.model_version

  def _collect_round_steps(self, steps, replay, learner, count_endings, staged):
    runner = self.runner
    for step in range(1, steps + 1):
      runner.step(record_episode_stats=False)
      if step % replay.buffer_size == 0 and step < steps:
        count_endings(replay.buffer_size)
        frames = self._fresh_training_frames(replay.buffer_size, ("state", "action"))
        if len(frames):
          staged = learner.stage_normalization(
            frames["state"], frames["action"], staged
          )
        del frames
    return staged

  def state_dict(self) -> dict:
    runner = self.runner
    if runner._pending_collection is not None:
      raise RuntimeError("Finish the pending collection transaction before saving")
    return deepcopy(
      {
        "version": EIT_TRAINING_VERSION,
        "stage": "model",
        "replay_mode": runner.replay_buffer.replay_mode,
        "motion_digest": motion_content_digest(runner.motion),
        "training_motion_files": self.training_motion_files,
        "training_seed": self.training_seed,
        "loop_config": asdict(self.cfg),
        **(
          {}
          if self.continuous_collection
          else {"snapshot_interval": self.snapshot_interval}
        ),
        "runner_config": asdict(runner.cfg),
        "world_model": runner._wm.bundle(),
        "actor": actor_checkpoint(self.actor, runner.cfg),
        "learner": self.learner.state_dict(),
        **(
          {"update_groups": self.update_groups}
          if self.continuous_collection
          else {"rounds": self.rounds}
        ),
        "collected_frames": self.collected_frames,
        "num_envs": runner.env.num_envs,
        "episode_id": getattr(runner, "_episode_id", None),
        "rng_cpu": torch.random.get_rng_state(),
        "rng_cuda": (
          torch.cuda.get_rng_state(runner.device)
          if torch.device(runner.device).type == "cuda"
          else None
        ),
      }
    )

  @staticmethod
  def _check_checkpoint_version(state: dict) -> None:
    validate_eit_checkpoint_contract(state, stage="model")

  def load_state_dict(self, state: dict) -> None:
    """Preflight every component on private copies, then refresh the live stream."""
    runner = self.runner
    self._check_checkpoint_version(state)
    if state.get("replay_mode") != runner.replay_buffer.replay_mode:
      raise ValueError("Training checkpoint replay mode differs from the runner")
    if state.get("motion_digest") != motion_content_digest(runner.motion):
      raise ValueError("Belief training checkpoint motion content mismatch")
    if state.get("training_seed") not in (None, self.training_seed):
      raise ValueError("Checkpoint training seed differs from the environment seed")
    if state.get("training_motion_files") and [
      f["sha256"] for f in state["training_motion_files"]
    ] != [f["sha256"] for f in self.training_motion_files]:
      raise ValueError(
        "Checkpoint training motion file digests differ from this runner"
      )
    if state.get("loop_config") != asdict(self.cfg):
      raise ValueError("Belief training loop configuration mismatch")
    operational = (
      {
        "max_iterations",
        "log_interval",
        "save_interval",
        "run_name",
        "wandb_project",
        "log_root",
      }
      if self.continuous_collection
      else set()
    )
    saved_runner = dict(state.get("runner_config", {}))
    # Historical v7/v8 checkpoints used the fixed [128, 64] FiLM network.
    saved_runner.setdefault("film_hidden_dims", (128, 64))
    if (
      isinstance(saved_runner.get("estimator_inputs"), dict)
      and "policy_version" not in saved_runner["estimator_inputs"]
    ):
      # New v8-only fields are defaults for historical v7 training checkpoints.
      saved_runner = dict(saved_runner)
      saved_runner["estimator_inputs"] = dict(
        saved_runner["estimator_inputs"],
        policy_version="v7",
        gt_pose_std=(0.03, 0.03),
        gt_velocity_std=0.2,
      )
      for name in ("heading_dropout_probability", "heading_rate_hz", "heading_std"):
        saved_runner.setdefault(name, getattr(type(runner.cfg)(), name))
    if {k: v for k, v in saved_runner.items() if k not in operational} != {
      k: v for k, v in asdict(runner.cfg).items() if k not in operational
    }:
      raise ValueError("Belief training runner configuration mismatch")
    if runner._pending_collection is not None:
      raise RuntimeError("Cannot restore during a pending collection transaction")
    progress = "update_groups" if self.continuous_collection else "rounds"
    for key in (progress, "collected_frames"):
      if type(state.get(key)) is not int or state[key] < 0:
        raise ValueError(f"Invalid checkpoint {key}")
    if (
      type(state.get("num_envs")) is not int or state["num_envs"] != runner.env.num_envs
    ):
      raise ValueError("Checkpoint environment count mismatch")
    ids = state["episode_id"]
    if ids is None and state[progress]:
      raise ValueError("Completed rounds require checkpoint episode IDs")
    if ids is not None and (
      not isinstance(ids, torch.Tensor)
      or ids.shape != (runner.env.num_envs,)
      or ids.dtype != torch.long
      or (ids < 0).any()
      or not torch.equal(
        ids.cpu() % runner.env.num_envs, torch.arange(runner.env.num_envs)
      )
    ):
      raise ValueError("Invalid checkpoint episode IDs")
    cpu_rng = state["rng_cpu"].cpu()
    torch.Generator().set_state(cpu_rng)
    cuda_rng = state["rng_cuda"]
    if cuda_rng is not None and (
      not isinstance(cuda_rng, torch.Tensor)
      or cuda_rng.dtype != torch.uint8
      or cuda_rng.ndim != 1
    ):
      raise ValueError("Invalid checkpoint CUDA RNG state")
    if torch.device(runner.device).type == "cuda" and cuda_rng is not None:
      cuda_rng = cuda_rng.cpu()
      torch.Generator(device=runner.device).set_state(cuda_rng)
    candidate_wm = deepcopy(runner._wm)
    candidate_wm.load_bundle(state["world_model"])
    inputs = runner.cfg.estimator_inputs
    if (
      inputs is None
      or candidate_wm.filter_config != runner.cfg.filter_config
      or candidate_wm.initial_prior != runner.cfg.initial_prior
      or candidate_wm.representation.measurement_config != inputs.measurement_noise
    ):
      raise ValueError(
        "World model settings disagree with the saved runner configuration"
      )
    candidate_learner = TwoStageWorldModelLearner(
      candidate_wm.dynamics,
      candidate_wm.representation,
      runner.cfg.mean,
      runner.cfg.cov,
      runner.device,
    )
    candidate_learner.load_state_dict(state["learner"])
    candidate_actor = deepcopy(self.actor)
    load_actor_checkpoint(candidate_actor, runner.cfg, state["actor"])
    del candidate_learner, candidate_actor, candidate_wm
    # A fresh EIT process already owns an empty raw ring. Preflight the
    # checkpoint layout while retaining the active replay if allocation fails.
    replay = runner.replay_buffer
    if replay.posterior_cache is not None and not replay.insert_count:
      cache = replay.prepare_posterior_cache(
        runner.posterior_save_stride, runner.cfg.history_length
      )
    else:
      replay = runner._make_replay_buffer()
      cache = replay.posterior_cache
    runner._wm.load_bundle(state["world_model"])
    runner.cfg.filter_config = runner._wm.filter_config
    runner.cfg.initial_prior = runner._wm.initial_prior
    self.learner.load_state_dict(state["learner"])
    load_actor_checkpoint(self.actor, runner.cfg, state["actor"])
    setattr(self, progress, state[progress])
    self.collected_frames = state["collected_frames"]
    if not self.continuous_collection:
      self.snapshot_interval = checkpoint_snapshot_interval(state)
    # Older v3 checkpoints do not document the original collection seed/files.
    self.training_seed = state.get("training_seed")
    if not state.get("training_motion_files"):
      self.training_motion_files = []
    # Existing replay may contain later episodes with reused IDs. Restore begins
    # a fresh replay and physical trajectory, while preserving optimizer state.
    replay.posterior_cache = cache
    if replay.replay_mode == "full":
      replay.initial_prior = runner._wm.initial_prior
    runner.replay_buffer = runner.wm_env.replay_buffer = replay
    runner._episode_id = ids.to(runner.device).clone() if ids is not None else None
    runner._pending_norm_updates.clear()
    torch.random.set_rng_state(cpu_rng)
    if torch.device(runner.device).type == "cuda" and cuda_rng is not None:
      torch.cuda.set_rng_state(cuda_rng, runner.device)
    self._output_checkpoint = None
    self._ready = False
    self._restart_after_restore(state)

  def _restart_after_restore(self, state):
    self.refresh_collection()

  def save(self, path: Path, *, metrics: dict | None = None) -> None:
    """Atomic replacement prevents interruption from truncating the last snapshot."""
    if metrics is not None:
      json.dumps(metrics, allow_nan=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = self.state_dict()
    state["last_metrics"] = metrics
    self._last_checkpoint_hook_seconds = 0.0
    self._checkpoint_hook_runtime_before = None
    temporary = NamedTemporaryFile(dir=path.parent, delete=False)
    temporary_path = Path(temporary.name)
    try:
      with temporary:
        torch.save(state, temporary)
      if self.before_checkpoint_commit is not None:
        self._checkpoint_hook_runtime_before = runtime_metadata(
          next(self.actor.parameters())
        )
        timer = PhaseTimer(self.runner.device)
        with timer.measure("reference"):
          self.before_checkpoint_commit(temporary_path, state)
        self._last_checkpoint_hook_seconds = timer.seconds["reference"]
        self._checkpoint_hook_runtime = runtime_metadata(next(self.actor.parameters()))
        # The disposable reference clone is a separate acceptance workload. Keep
        # its peak separately, then combine pre-hook and post-hook training peaks.
        runtime_metadata(next(self.actor.parameters()), reset_peak=True)
      temporary_path.replace(path)
    finally:
      temporary_path.unlink(missing_ok=True)

  _output_checkpoint: Path | None = None

  @staticmethod
  def _repair_profile_tail(path: Path, state: dict) -> dict:
    """Seal a complete last row or archive a torn append, never invent timings.

    Call only while holding the output directory's training lock. Complete but
    malformed rows are rejected; only the final unterminated JSON fragment can
    be removed, after its exact bytes have been preserved separately.
    """
    content = path.read_bytes() if path.exists() else b""
    lines = content.splitlines(keepends=True)
    prefix, rounds = b"", []
    fragment = None
    repaired = False
    for index, line in enumerate(lines):
      tail = index == len(lines) - 1 and not line.endswith(b"\n")
      try:
        row = json.loads(line)
      except (ValueError, UnicodeDecodeError):
        if not tail:
          raise ValueError("Malformed complete profiling row") from None
        fragment = line
        repaired = True
        break
      if (
        not isinstance(row, dict)
        or type(row.get("round")) is not int
        or row["round"] < 1
        or row["round"] > state["rounds"]
        or (rounds and row["round"] <= rounds[-1])
      ):
        raise ValueError("Profiling rounds disagree with the current checkpoint")
      rounds.append(row["round"])
      prefix += line + (b"\n" if tail else b"")
      repaired |= tail
    archive = None
    if fragment is not None:
      digest = hashlib.sha256(fragment).hexdigest()
      archive = path.with_name(f"profiling.interrupted.{digest}.fragment")
      if archive.is_symlink():
        raise ValueError("Profiling recovery archive must be local")
      if archive.exists():
        if archive.read_bytes() != fragment:
          raise ValueError("Profiling recovery archive content conflict")
      else:
        with NamedTemporaryFile(dir=path.parent, delete_on_close=False) as stream:
          stream.write(fragment)
          stream.flush()
          stream.close()
          os.link(stream.name, archive)
    if repaired:
      with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
          stream.write(prefix)
          stream.close()
          temporary.replace(path)
        finally:
          temporary.unlink(missing_ok=True)
    result = {
      "profiling_tail_repaired": repaired,
      "archived_profile_fragment": str(archive) if archive is not None else None,
      "missing_profile_rounds": sorted(
        set(range(1, state["rounds"] + 1)) - set(rounds)
      ),
      "missing_timing_policy": "Unavailable post-commit timings remain missing; no zero or reconstructed duration is substituted.",
    }
    if repaired or result["missing_profile_rounds"]:
      note = path.with_name("profiling_recovery.json")
      if note.is_symlink():
        raise ValueError("Profiling recovery note must be local")
      evidence = {
        "version": "eit_profiling_recovery_v1",
        "checkpoint_round": state["rounds"],
        **result,
        "preserved_fragments": [
          item.name
          for item in sorted(path.parent.glob("profiling.interrupted.*.fragment"))
        ],
      }
      with NamedTemporaryFile(dir=path.parent, mode="w", delete=False) as stream:
        temporary = Path(stream.name)
        try:
          json.dump(evidence, stream, indent=2, allow_nan=False)
          stream.write("\n")
          stream.close()
          temporary.replace(note)
        finally:
          temporary.unlink(missing_ok=True)
    return result

  @classmethod
  def reconcile_output(
    cls, output: Path, *, validate_state: Callable[[dict], object] | None = None
  ) -> dict:
    """Validate and repair one locked checkpoint without taking a training step.

    The optional caller preflight runs on the same loaded state before any
    snapshot or log mutation, allowing pilot identity/budget checks to be atomic
    with recovery rather than reopening a potentially different checkpoint.
    """
    output = Path(output).resolve()
    checkpoint, metrics = output / "training.pt", output / "metrics.jsonl"
    profiles, lock = output / "profiling.jsonl", output / ".belief-training.lock"
    if any(path.is_symlink() for path in (checkpoint, metrics, profiles, lock)):
      raise ValueError("Training recovery requires local output files")
    with FileLock(lock, timeout=0):
      state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
      cls._check_checkpoint_version(state)
      if validate_state is not None:
        validate_state(state)
      missing = cls._missing_metric_bytes(metrics, state)
      profile = cls._repair_profile_tail(profiles, state)
      publish_snapshot(checkpoint, state)
      if missing:
        with metrics.open("ab") as stream:
          stream.write(missing)
      return {
        "version": "eit_output_recovery_v1",
        "round": state["rounds"],
        "collected_frames": state["collected_frames"],
        "checkpoint_sha256": file_sha256(checkpoint),
        "repaired_metric_bytes": len(missing),
        **profile,
      }

  def run(
    self,
    rounds: int,
    output: Path,
    *,
    resume: Path | None = None,
    snapshot_interval: int | None = None,
  ) -> list[dict]:
    if type(rounds) is not int or rounds < 1:
      raise ValueError("rounds must be positive")
    if snapshot_interval is not None and (
      type(snapshot_interval) is not int or snapshot_interval < 1
    ):
      raise ValueError("Snapshot interval must be a positive integer")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "training.pt"
    metrics_path = output / "metrics.jsonl"
    profile_path = output / "profiling.jsonl"
    lock_path = output / ".belief-training.lock"

    def check_destination():
      if lock_path.is_symlink():
        raise ValueError("Training lock must be local to the output directory")
      occupied = any(p != lock_path for p in output.iterdir())
      source = Path(resume).resolve() if resume is not None else self._output_checkpoint
      if occupied and (
        source != checkpoint
        or not checkpoint.is_file()
        or checkpoint.is_symlink()
        or metrics_path.is_symlink()
        or profile_path.is_symlink()
      ):
        raise ValueError(
          "Nonempty output requires resuming its own current training.pt"
        )
      return occupied

    check_destination()
    with FileLock(lock_path, timeout=0):
      occupied = check_destination()
      if resume is not None or occupied:
        source = resume if resume is not None else checkpoint
        with Path(source).open("rb") as stream:
          state = torch.load(stream, map_location=self.runner.device, weights_only=True)
          self._check_checkpoint_version(state)
          missing = self._missing_metric_bytes(metrics_path, state)
          if resume is not None:
            self.load_state_dict(state)
          elif (
            self.rounds != state["rounds"]
            or self.runner._dynamics.model_version != state["learner"]["model_version"]
            or (
              self.continuous_collection
              and self.runner._dynamics.model_revision != state["model_revision"]
            )
          ):
            raise ValueError("Active training state differs from the output checkpoint")
          if not occupied:
            # Fork the exact bytes just loaded even if the source trainer advances.
            with NamedTemporaryFile(dir=output, delete_on_close=False) as temporary:
              stream.seek(0)
              copyfileobj(stream, temporary)
              temporary.close()
              os.link(temporary.name, checkpoint)
        publish_snapshot(checkpoint, state)
        self._repair_profile_tail(profile_path, state)
        if missing:
          with metrics_path.open("ab") as stream:
            stream.write(missing)
        del state
      elif self.rounds == 0:
        if snapshot_interval is not None:
          self.snapshot_interval = snapshot_interval
        self.save(checkpoint)
        publish_snapshot(checkpoint)
      # Reconcile the old checkpoint under its recorded policy first. Overrides
      # apply only to newly committed rounds and never rewrite published bytes.
      if snapshot_interval is not None:
        self.snapshot_interval = snapshot_interval
      self._output_checkpoint = checkpoint
      logs = []
      for _ in range(rounds):
        metrics = self.run_round(checkpoint)
        timer = PhaseTimer(self.runner.device)
        publication_start = timer.now()
        publish_snapshot(checkpoint)
        row = json.dumps(metrics, sort_keys=True, allow_nan=False)
        with metrics_path.open("ab") as stream:
          stream.write((row + "\n").encode())
        publication_seconds = timer.now() - publication_start
        self.last_profile["seconds"]["snapshot_and_metrics_seconds"] = (
          publication_seconds
        )
        self.last_profile["seconds"]["round_seconds"] += publication_seconds
        if "round_seconds_including_resume_reference" in self.last_profile["seconds"]:
          self.last_profile["seconds"]["round_seconds_including_resume_reference"] += (
            publication_seconds
          )
        # Post-commit timings cannot live inside the checkpoint they measure.
        # A missing row after interruption means missing profiling, not zero cost.
        with profile_path.open("a") as stream:
          stream.write(json.dumps(self.last_profile, sort_keys=True, allow_nan=False))
          stream.write("\n")
        print("BELIEF_TRAINING " + row)
        logs.append(metrics)
      return logs

  @staticmethod
  def _missing_metric_bytes(path: Path, state: dict) -> bytes:
    """Check the last completed row before finishing any interrupted append."""
    last = None
    partial = b""
    if path.exists():
      with path.open("rb") as stream:
        for line in stream:
          if line.endswith(b"\n"):
            last = json.loads(line)
          else:
            partial = line
    completed = state.get("last_metrics")
    if last is not None and last["round"] == state["rounds"]:
      if (
        last["trained_model_version"]
        != state.get("model_revision", state["learner"]["model_version"])
        or last["mean_steps"] != state["learner"]["mean"]["steps"]
        or last["cov_steps"] != state["learner"]["cov"]["steps"]
        or last["collected_frames"] != state["collected_frames"]
        # Compare the metric row using the same JSON encoding as the writer.
        or (
          completed is not None
          and json.dumps(last, sort_keys=True) != json.dumps(completed, sort_keys=True)
        )
        or partial
      ):
        raise ValueError("Metrics history disagrees with the current checkpoint")
      return b""
    if completed is not None and (
      completed["round"] == state["rounds"]
      and completed["trained_model_version"]
      == state.get("model_revision", state["learner"]["model_version"])
      and (
        last is None
        or (
          last["round"] == completed["round"] - 1
          and last["trained_model_version"] == completed["collection_model_version"]
        )
      )
    ):
      expected = (json.dumps(completed, sort_keys=True) + "\n").encode()
      if not expected.startswith(partial):
        raise ValueError("Partial metrics row disagrees with the current checkpoint")
      return expected[len(partial) :]
    if last is None and completed is None and not partial:
      return b""
    raise ValueError("Metrics history and checkpoint refer to different rounds")
