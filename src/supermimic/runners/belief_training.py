"""Round-based mean and covariance fitting with a fixed collection policy.

Checkpoints resume model optimization at a round boundary. Physical simulator
state is deliberately reset; replay and online posteriors are rebuilt from new
measurements. They are never restored under a changed model.
"""

from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from shutil import copyfileobj
from tempfile import NamedTemporaryFile
from typing import cast

import torch
from filelock import FileLock

from supermimic.learners.two_stage_wm import TwoStageWorldModelLearner
from supermimic.models import MultiEncoderAE
from supermimic.motion_loader import motion_content_digest
from supermimic.runners.eit_snapshots import file_sha256, publish_snapshot
from supermimic.runners.run_config import (
  EIT_TRAINING_VERSION,
  validate_eit_checkpoint_contract,
)
from supermimic.runners.supermimic_runner import (
  SuperMimicRunner,
  actor_checkpoint,
  load_actor_checkpoint,
)


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
    self.rounds = 0
    self.collected_frames = 0
    self.training_seed = getattr(getattr(runner.env, "cfg", None), "seed", None)
    self.training_motion_files = [
      {"path": str(path), "sha256": file_sha256(path)}
      for path in (Path(f).resolve() for f in runner.motion.files)
    ]
    self._ready = False
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
    if not self._ready:
      self.refresh_collection()
    version_before = runner._dynamics.model_version
    actor_before = parameter_digest(self.actor)
    replay = runner.replay_buffer
    # Fold each chunk into private statistics before the next ring overwrite.
    staged = None
    steps = self.cfg.collection_steps
    for step in range(1, steps + 1):
      runner.step(record_episode_stats=False)
      if step % replay.buffer_size == 0 and step < steps:
        frames = self._fresh_training_frames(replay.buffer_size, ("state", "action"))
        if len(frames):
          staged = learner.stage_normalization(
            frames["state"], frames["action"], staged
          )
        del frames
    if (
      runner._dynamics.model_version != version_before
      or parameter_digest(self.actor) != actor_before
    ):
      raise RuntimeError("Collection model or actor changed within a round")
    runner.finish_collection_round()
    self._ready = False
    train_ids, valid_ids = self.episode_split()
    remaining = (steps - 1) % replay.buffer_size + 1
    train, validation = learner.fit(
      replay,
      train_ids,
      valid_ids,
      self._fresh_training_frames(remaining, ("state", "action")),
      self.cfg.validation_batch_size,
      staged_normalizers=staged,
    )
    trained_version = runner._dynamics.model_version
    policy = self._update_policy()
    if runner._dynamics.model_version != trained_version:
      raise RuntimeError("Policy update modified the frozen world model")
    metrics = {
      "round": self.rounds + 1,
      "mean_steps": learner.mean_steps,
      "cov_steps": learner.cov_steps,
      "collected_frames": self.collected_frames
      + self.cfg.collection_steps * runner.env.num_envs,
      "training_episodes": train_ids,
      "validation_episodes": valid_ids,
      "collection_model_version": version_before,
      "trained_model_version": trained_version,
      "collection_actor_version": actor_before,
      "trained_actor_version": parameter_digest(self.actor),
      "train": train,
      "validation": validation,
      **policy,
    }
    # Verify the entire row before advancing the round or replacing its checkpoint.
    json.dumps(metrics, allow_nan=False)
    self.rounds = metrics["round"]
    self.collected_frames = metrics["collected_frames"]
    if checkpoint_path is not None:
      self.save(checkpoint_path, metrics=metrics)
    self.refresh_collection()
    return metrics

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
        "runner_config": asdict(runner.cfg),
        "world_model": runner._wm.bundle(),
        "actor": actor_checkpoint(self.actor, runner.cfg),
        "learner": self.learner.state_dict(),
        "rounds": self.rounds,
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
    if state.get("runner_config") != asdict(runner.cfg):
      raise ValueError("Belief training runner configuration mismatch")
    if runner._pending_collection is not None:
      raise RuntimeError("Cannot restore during a pending collection transaction")
    for key in ("rounds", "collected_frames"):
      if type(state.get(key)) is not int or state[key] < 0:
        raise ValueError(f"Invalid checkpoint {key}")
    if (
      type(state.get("num_envs")) is not int or state["num_envs"] != runner.env.num_envs
    ):
      raise ValueError("Checkpoint environment count mismatch")
    ids = state["episode_id"]
    if ids is None and state["rounds"]:
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

    runner._wm.load_bundle(state["world_model"])
    runner.cfg.filter_config = runner._wm.filter_config
    runner.cfg.initial_prior = runner._wm.initial_prior
    runner.configure_belief_filter()
    self.learner.load_state_dict(state["learner"])
    load_actor_checkpoint(self.actor, runner.cfg, state["actor"])
    self.rounds, self.collected_frames = state["rounds"], state["collected_frames"]
    # Older v3 checkpoints do not document the original collection seed/files.
    self.training_seed = state.get("training_seed")
    if not state.get("training_motion_files"):
      self.training_motion_files = []
    # Existing replay may contain later episodes with reused IDs. Restore begins
    # a fresh replay and physical trajectory, while preserving optimizer state.
    runner.replay_buffer = runner._make_replay_buffer()
    runner.wm_env.replay_buffer = runner.replay_buffer
    runner._episode_id = ids.to(runner.device).clone() if ids is not None else None
    runner._pending_norm_updates.clear()
    torch.random.set_rng_state(cpu_rng)
    if torch.device(runner.device).type == "cuda" and cuda_rng is not None:
      torch.cuda.set_rng_state(cuda_rng, runner.device)
    self._output_checkpoint = None
    self.refresh_collection()

  def save(self, path: Path, *, metrics: dict | None = None) -> None:
    """Atomic replacement prevents interruption from truncating the last snapshot."""
    if metrics is not None:
      json.dumps(metrics, allow_nan=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = self.state_dict()
    state["last_metrics"] = metrics
    temporary = NamedTemporaryFile(dir=path.parent, delete=False)
    temporary_path = Path(temporary.name)
    try:
      with temporary:
        torch.save(state, temporary)
      temporary_path.replace(path)
    finally:
      temporary_path.unlink(missing_ok=True)

  _output_checkpoint: Path | None = None

  def run(self, rounds: int, output: Path, *, resume: Path | None = None) -> list[dict]:
    if type(rounds) is not int or rounds < 1:
      raise ValueError("rounds must be positive")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "training.pt"
    metrics_path = output / "metrics.jsonl"
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
        if missing:
          with metrics_path.open("ab") as stream:
            stream.write(missing)
        del state
      elif self.rounds == 0:
        self.save(checkpoint)
        publish_snapshot(checkpoint)
      self._output_checkpoint = checkpoint
      logs = []
      for _ in range(rounds):
        metrics = self.run_round(checkpoint)
        publish_snapshot(checkpoint)
        row = json.dumps(metrics, sort_keys=True, allow_nan=False)
        with metrics_path.open("ab") as stream:
          stream.write((row + "\n").encode())
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
        last["trained_model_version"] != state["learner"]["model_version"]
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
      and completed["trained_model_version"] == state["learner"]["model_version"]
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
