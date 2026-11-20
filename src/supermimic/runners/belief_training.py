"""Round-based posterior-NLL learning with a fixed collection policy.

Checkpoints resume model optimization at a round boundary. Physical simulator
state is deliberately reset; replay and online posteriors are rebuilt from new
measurements. They are never restored under a changed model.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import cast

import torch
from filelock import FileLock

from supermimic.estimator import LieBeliefFilter
from supermimic.learners.belief_wm_learner import BeliefWorldModelLearner
from supermimic.models import MultiEncoderAE
from supermimic.motion_loader import motion_content_digest
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


def _round_training_metrics(updates: list[dict[str, torch.Tensor]]) -> dict:
  counts = torch.stack([log["valid_transitions"] for log in updates]).long()
  total = counts.sum()
  metrics: dict[str, float | int] = {"valid_transitions": int(total)}
  for name in updates[0]:
    if name == "valid_transitions":
      continue
    values = torch.stack([log[name] for log in updates]).double()
    # Each optimizer update contributes one gradient norm, while its loss and
    # diagnostics are means over the transitions actually present in its windows.
    mean = (
      values.mean() if name.startswith("Grad/") else (values * counts).sum() / total
    )
    metrics[name] = float(mean)
  return metrics


class BeliefModelTraining:
  """Collect, normalize training episodes, optimize, validate, and reset.

  Episode IDs divisible by validation_stride are permanently held out. The
  split never depends on which fragments survive buffer overwrite. Both sets
  must contain a valid transition; increase collection length/env count if not.
  """

  def __init__(self, runner: SuperMimicRunner, cfg: BeliefTrainingCfg | None = None):
    if not isinstance(runner.world_model_learner, BeliefWorldModelLearner):
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
    self._ready = False
    # This entry point does not optimize the policy or change its normalizers.
    runner.policy_learner.actor.eval().requires_grad_(False)

  @property
  def learner(self) -> BeliefWorldModelLearner:
    return cast(BeliefWorldModelLearner, self.runner.world_model_learner)

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

  def run_round(self, checkpoint_path: Path | None = None) -> dict:
    runner, learner = self.runner, self.learner
    if not self._ready:
      self.refresh_collection()
    version_before = runner._dynamics.model_version
    for _ in range(self.cfg.collection_steps):
      runner.step(record_episode_stats=False)
    if runner._dynamics.model_version != version_before:
      raise RuntimeError("The collection model changed within a round")
    runner.finish_collection_round()
    self._ready = False
    train_ids, valid_ids = self.episode_split()
    fresh = runner.replay_buffer.retained_frames(newest_count=self.cfg.collection_steps)
    train_rows = fresh["episode_id"] % self.cfg.validation_stride != 0
    # A singleton still supplies a state/action for normalization. A round with
    # only fresh validation frames can train from earlier retained transitions.
    if train_rows.any():
      learner.update_normalization(
        fresh["state"][train_rows], fresh["action"][train_rows]
      )
    # No cached posterior is reused after normalizers or parameters change.
    # The learner rebuilds each prefix with current weights and the same prior.
    train_logs = []
    for _ in range(learner.cfg.num_batches):
      sequences = runner.replay_buffer.sample_belief_sequences(
        learner.cfg.horizon, learner.cfg.batch_size, episode_ids=train_ids
      )
      train_logs.append(learner.update(sequences))
    held_out = runner.replay_buffer.sample_belief_sequences(
      learner.cfg.horizon, self.cfg.validation_batch_size, episode_ids=valid_ids
    )
    validation = learner.evaluate(held_out)
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
      "train": _round_training_metrics(train_logs),
      "validation": {key: float(value) for key, value in validation.items()},
    }
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
        "version": "belief_model_training_v2",
        "motion_digest": motion_content_digest(runner.motion),
        "loop_config": asdict(self.cfg),
        "runner_config": asdict(runner.cfg),
        "world_model": runner._wm.bundle(),
        "actor": actor_checkpoint(self.actor, runner.cfg),
        "learner": self.learner.state_dict(),
        "rounds": self.rounds,
        "collected_frames": self.collected_frames,
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
    if state.get("version") == "belief_model_training_v1":
      raise ValueError(
        "Legacy belief training checkpoint has no verified motion identity"
      )
    if state.get("version") != "belief_model_training_v2":
      raise ValueError("Unsupported belief training checkpoint")

  def load_state_dict(self, state: dict) -> None:
    """Preflight every component on private copies, then refresh the live stream."""
    runner = self.runner
    self._check_checkpoint_version(state)
    if state.get("motion_digest") != motion_content_digest(runner.motion):
      raise ValueError("Belief training checkpoint motion content mismatch")
    if state.get("loop_config") != asdict(self.cfg):
      raise ValueError("Belief training loop configuration mismatch")
    if state.get("runner_config") != asdict(runner.cfg):
      raise ValueError("Belief training runner configuration mismatch")
    if runner._pending_collection is not None:
      raise RuntimeError("Cannot restore during a pending collection transaction")
    for key in ("rounds", "collected_frames"):
      if type(state.get(key)) is not int or state[key] < 0:
        raise ValueError(f"Invalid checkpoint {key}")
    ids = state.get("episode_id")
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
    candidate_estimator = LieBeliefFilter(
      candidate_wm.dynamics,
      candidate_wm.representation.measurement_config,
      cast(torch.Tensor, candidate_wm.representation._default_joint_pos),
      candidate_wm.filter_config,
    )
    candidate_learner = BeliefWorldModelLearner(candidate_estimator, self.learner.cfg)
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
    # Existing replay may contain later episodes with reused IDs. Restore begins
    # a fresh replay and physical trajectory, while preserving optimizer state.
    from supermimic.replay_buffer import ReplayBuffer

    runner.replay_buffer = ReplayBuffer(
      runner._buffer_schema(),
      runner.cfg.buffer_size,
      runner.env.num_envs,
      runner.device,
      initial_prior=runner._wm.initial_prior,
    )
    runner.wm_env.replay_buffer = runner.replay_buffer
    if ids is not None:
      runner._episode_id = ids.to(runner.device).clone()
    runner._pending_norm_updates.clear()
    torch.random.set_rng_state(cpu_rng)
    if torch.device(runner.device).type == "cuda" and cuda_rng is not None:
      torch.cuda.set_rng_state(cuda_rng, runner.device)
    self._output_checkpoint = None
    self.refresh_collection()

  def save(self, path: Path, *, metrics: dict | None = None) -> None:
    """Atomic replacement prevents interruption from truncating the last snapshot."""
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
        state = torch.load(source, map_location=self.runner.device, weights_only=True)
        self._check_checkpoint_version(state)
        missing = self._missing_metric_bytes(metrics_path, state) if occupied else b""
        if resume is not None:
          self.load_state_dict(state)
        elif (
          self.rounds != state["rounds"]
          or self.runner._dynamics.model_version != state["learner"]["model_version"]
        ):
          raise ValueError("Active training state differs from the output checkpoint")
        if missing:
          with metrics_path.open("ab") as stream:
            stream.write(missing)
        del state
      self._output_checkpoint = checkpoint
      logs = []
      for _ in range(rounds):
        metrics = self.run_round(checkpoint)
        row = json.dumps(metrics, sort_keys=True)
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
        or last["optimizer_steps"] != state["learner"]["steps"]
        or last["collected_frames"] != state["collected_frames"]
        # Older EIT rows wrote NaN for undefined explained variance. Compare
        # their serialized values so a faithfully restored NaN matches itself.
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
