"""Posterior-NLL training on retained, measured belief sequences.

Every optimizer step reconstructs window boundary beliefs with the current
model and the same realized segment priors. No posterior survives an update.
Normalization updates are explicit round-boundary operations.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, fields
from typing import TYPE_CHECKING, cast

import torch

from supermimic.belief import BeliefState
from supermimic.estimator import (
  CorrectionDiagnostics,
  LieBeliefFilter,
  NllDiagnostics,
  posterior_nll,
)
from supermimic.replay_buffer import BeliefTrainingSequence
from supermimic.utils.lie_state import LieState

if TYPE_CHECKING:
  from collections.abc import Sequence


@dataclass(frozen=True)
class BeliefWorldModelLearnerCfg:
  lr: float = 2e-4
  batch_size: int = 4
  microbatch_size: int = 2
  horizon: int = 2
  num_batches: int = 10
  max_grad_norm: float = 1.0

  def __post_init__(self):
    if not math.isfinite(self.lr) or self.lr <= 0:
      raise ValueError("Belief learner lr must be finite and positive")
    if not math.isfinite(self.max_grad_norm) or self.max_grad_norm <= 0:
      raise ValueError("Belief learner max_grad_norm must be finite and positive")
    for name in ("batch_size", "microbatch_size", "horizon", "num_batches"):
      value = getattr(self, name)
      if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"Belief learner {name} must be a positive integer")


def _stack_beliefs(beliefs: list[BeliefState]) -> BeliefState:
  return BeliefState(
    LieState(
      *(
        torch.stack([getattr(b.mean, f.name) for b in beliefs])
        for f in fields(LieState)
      )
    ),
    torch.stack([b.covariance for b in beliefs]),
  )


def _validate_sequence(sequence: BeliefTrainingSequence) -> None:
  if not isinstance(sequence, BeliefTrainingSequence):
    raise TypeError("Belief learner requires BeliefTrainingSequence records")
  sequence.prior.validate()
  if sequence.prior.mean.position.shape != (3,):
    raise ValueError("Each sequence must carry one unbatched segment prior")
  prefix, window = sequence.prefix, sequence.frames
  required = {
    "state",
    "actor_obs",
    "action",
    "prev_action",
    "sim_ctx",
    "episode_id",
    "time_index",
    "terminated",
    "truncated",
  }
  for frames in (prefix, window):
    if frames.ndim != 1 or len(frames) < 1 or not required <= frames.keys():
      raise ValueError(
        "Belief sequence frames require a nonempty one-dimensional schema"
      )
    if frames["state"].shape != (len(frames), 71):
      raise ValueError("Belief sequence truth must have 71 components")
    if frames["actor_obs"].shape != (len(frames), 93):
      raise ValueError("Belief sequence actor_obs must have 93 components")
    if frames["action"].shape != (len(frames), 29):
      raise ValueError("Belief sequence actions must have 29 components")
    if not torch.equal(frames["actor_obs"][:, 64:], frames["prev_action"]):
      raise ValueError("Belief sequence sensor action suffix is inconsistent")
    if len(frames) > 1 and (
      (frames["episode_id"][1:] != frames["episode_id"][:-1]).any()
      or not torch.equal(frames["time_index"][1:], frames["time_index"][:-1] + 1)
      or (frames["terminated"][:-1] | frames["truncated"][:-1]).any()
      or not torch.equal(frames["prev_action"][1:], frames["action"][:-1])
    ):
      raise ValueError("Belief sequence crosses a reset or has misaligned actions")
    if not torch.equal(
      frames["sim_ctx"], frames["sim_ctx"][:1].expand_as(frames["sim_ctx"])
    ):
      raise ValueError("Belief sequence context must remain fixed within an episode")
    for name in ("state", "actor_obs", "action", "sim_ctx"):
      value = frames[name]
      if (
        value.dtype != sequence.prior.covariance.dtype
        or value.device != sequence.prior.covariance.device
      ):
        raise ValueError("Belief sequence values must share prior dtype and device")
      if not torch.isfinite(value).all():
        raise ValueError(f"Belief sequence {name} must be finite")
  if len(window) < 2:
    raise ValueError("Belief sequence needs at least one recorded successor")
  if any(not torch.equal(prefix[name][-1], window[name][0]) for name in required):
    raise ValueError("Belief sequence prefix must end at the exact first window frame")


class BeliefWorldModelLearner:
  """Jointly learn mean and process-noise heads from corrected posterior NLL.

  ``update`` makes one optimizer step over a list of windows. Microbatches
  accumulate sums divided by the *total valid transition count*, so ragged
  windows receive the same per-transition weight regardless of chunking.
  The deterministic WorldModelLearner retains its existing data contract.
  """

  def __init__(
    self, estimator: LieBeliefFilter, cfg: BeliefWorldModelLearnerCfg | None = None
  ) -> None:
    self.estimator = estimator
    self.dynamics = estimator.dynamics
    self.cfg = cfg or BeliefWorldModelLearnerCfg()
    if self.dynamics.noise_head is None:
      raise ValueError("BeliefWorldModelLearner requires a stochastic noise head")
    self._all_params = list(self.dynamics.parameters())
    self.optimizer = torch.optim.Adam(self._all_params, lr=self.cfg.lr)
    self.steps = 0
    self.dynamics.eval()

  @torch.no_grad()
  def update_normalization(self, states: torch.Tensor, actions: torch.Tensor) -> None:
    """Update only with fresh training-episode frames at a collection boundary."""
    if (
      states.ndim < 1
      or actions.ndim < 1
      or states.shape[:-1] != actions.shape[:-1]
      or states.shape[-1] != 71
      or actions.shape[-1] != 29
    ):
      raise ValueError("Normalization needs aligned 71D states and 29D actions")
    if (
      states.numel() == 0
      or not states.is_floating_point()
      or not actions.is_floating_point()
      or not torch.isfinite(states).all()
      or not torch.isfinite(actions).all()
    ):
      raise ValueError("Normalization requires nonempty finite collected data")
    quaternion_norm_sq = states[..., 3:7].square().sum(-1)
    if not (torch.isfinite(quaternion_norm_sq) & (quaternion_norm_sq > 0)).all():
      raise ValueError("Normalization requires nonzero, finite quaternion norms")
    state = LieState.from_full(states)
    state.validate()
    _, rep = state.to_rep()
    values = (rep.reshape(-1, 68), self.dynamics.proc(actions.reshape(-1, 29)))
    # Stage both small sets of statistics. Even finite values may overflow a
    # batch variance, including when only the second normalizer is affected.
    normalizers = self.dynamics.normalizers()
    candidates = copy.deepcopy(normalizers)
    for candidate, value in zip(candidates, values, strict=True):
      if not torch.isfinite(value).all():
        raise ValueError("Normalization produced non-finite input features")
      candidate.train()
      candidate.update(value)
      if candidate.count < 0 or not all(
        torch.isfinite(buffer).all() for buffer in candidate.buffers()
      ):
        raise ValueError("Normalization produced invalid running statistics")
    for normalizer, candidate in zip(normalizers, candidates, strict=True):
      normalizer.load_state_dict(candidate.state_dict())
      normalizer.eval()

  @torch.no_grad()
  def boundary_belief(self, sequence: BeliefTrainingSequence) -> BeliefState:
    """Replay the full retained prefix and detach only the window boundary."""
    _validate_sequence(sequence)
    frames = sequence.prefix
    belief = cast(
      BeliefState,
      self.estimator.correct(sequence.prior, frames["actor_obs"][0, :64]),
    )
    for index in range(len(frames) - 1):
      belief = cast(
        BeliefState,
        self.estimator.predict(
          belief, frames["action"][index], frames["sim_ctx"][index]
        ),
      )
      belief = cast(
        BeliefState,
        self.estimator.correct(belief, frames["actor_obs"][index + 1, :64]),
      )
    return belief.detach()

  def _loss_sum(self, sequences: Sequence[BeliefTrainingSequence]):
    if not sequences:
      raise ValueError("Belief learner requires at least one sequence")
    belief = _stack_beliefs([self.boundary_belief(s) for s in sequences])
    active = list(sequences)
    terms, stats = [], {}
    block_slices = {
      "rotation": slice(0, 3),
      "position": slice(3, 6),
      "velocity": slice(6, 9),
      "omega": slice(9, 12),
      "joint_pos": slice(12, 41),
      "joint_vel": slice(41, 70),
    }

    def record(name, value):
      detached = value.detach().sum()
      stats[name] = stats.get(name, 0) + detached

    for step in range(max(s.transitions for s in active)):
      keep = [index for index, s in enumerate(active) if step < s.transitions]
      active = [active[index] for index in keep]
      belief = belief.index(keep)
      actions = torch.stack([s.frames["action"][step] for s in active])
      context = torch.stack([s.frames["sim_ctx"][step] for s in active])
      measurement = torch.stack([s.frames["actor_obs"][step + 1, :64] for s in active])
      truth = LieState.from_full(
        torch.stack([s.frames["state"][step + 1] for s in active])
      )
      predicted = cast(BeliefState, self.estimator.predict(belief, actions, context))
      belief, correction = cast(
        tuple[BeliefState, CorrectionDiagnostics],
        self.estimator.correct(predicted, measurement, return_diagnostics=True),
      )
      nll, diagnostic = cast(
        tuple[torch.Tensor, NllDiagnostics],
        posterior_nll(belief, truth, self.estimator.config, return_diagnostics=True),
      )
      terms.append(nll.sum())
      # Diagnostics do not retain a second prediction graph or contribute a loss.
      with torch.no_grad():
        prediction_nll, prediction_diagnostic = cast(
          tuple[torch.Tensor, NllDiagnostics],
          posterior_nll(
            predicted.detach(), truth, self.estimator.config, return_diagnostics=True
          ),
        )
        variance = belief.covariance.detach().diagonal(dim1=-2, dim2=-1)
        record("Loss/posterior_nll", nll)
        record("Diagnostics/mahalanobis", diagnostic.mahalanobis)
        record("Diagnostics/log_determinant", diagnostic.log_determinant)
        record("Diagnostics/prediction_nll", prediction_nll)
        record("Diagnostics/prediction_mahalanobis", prediction_diagnostic.mahalanobis)
        record(
          "Diagnostics/prediction_mse", prediction_diagnostic.residual.square().mean(-1)
        )
        record("Diagnostics/covariance_mean", variance.mean(-1))
        record(
          "Diagnostics/marginal_95_coverage",
          (diagnostic.residual.abs() <= 1.96 * variance.sqrt())
          .to(variance.dtype)
          .mean(-1),
        )
        record(
          "Diagnostics/jitter_fraction",
          ((diagnostic.jitter > 0) | (correction.jitter > 0)).to(variance.dtype),
        )
        record(
          "Diagnostics/jitter", torch.maximum(diagnostic.jitter, correction.jitter)
        )
        for name, block in block_slices.items():
          record(
            f"Diagnostics/{name}_mse", diagnostic.residual[..., block].square().mean(-1)
          )
          record(f"Diagnostics/{name}_variance", variance[..., block].mean(-1))
    return torch.stack(terms).sum(), stats, sum(s.transitions for s in sequences)

  def loss(self, sequences: Sequence[BeliefTrainingSequence]):
    """Differentiable mean loss and detached, per-transition diagnostics."""
    total, stats, count = self._loss_sum(sequences)
    return total / count, {name: value / count for name, value in stats.items()}

  @torch.no_grad()
  def evaluate(
    self, sequences: Sequence[BeliefTrainingSequence]
  ) -> dict[str, torch.Tensor]:
    self.dynamics.eval()
    count, totals = 0, {}
    for offset in range(0, len(sequences), self.cfg.microbatch_size):
      _, stats, n = self._loss_sum(
        sequences[offset : offset + self.cfg.microbatch_size]
      )
      count += n
      for key, value in stats.items():
        totals[key] = totals.get(key, 0) + value
    if not count:
      raise ValueError("Belief learner evaluation requires at least one transition")
    return {
      **{name: value / count for name, value in totals.items()},
      "valid_transitions": next(iter(totals.values())).new_tensor(count),
    }

  def update(
    self, sequences: Sequence[BeliefTrainingSequence]
  ) -> dict[str, torch.Tensor]:
    if not sequences:
      raise ValueError("Belief learner update requires at least one sequence")
    for sequence in sequences:
      _validate_sequence(sequence)
    count = sum(s.transitions for s in sequences)
    self.dynamics.train()
    self.dynamics.requires_grad_(True)
    # Frozen statistics throughout prefix reconstruction and differentiable windows.
    for normalizer in self.dynamics.normalizers():
      normalizer.eval()
    self.optimizer.zero_grad(set_to_none=True)
    totals = {}
    try:
      for offset in range(0, len(sequences), self.cfg.microbatch_size):
        loss, stats, _ = self._loss_sum(
          sequences[offset : offset + self.cfg.microbatch_size]
        )
        if not torch.isfinite(loss):
          raise FloatingPointError("Non-finite posterior NLL; optimizer step rejected")
        (loss / count).backward()
        for key, value in stats.items():
          totals[key] = totals.get(key, 0) + value
      for parameter in self._all_params:
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
          raise FloatingPointError(
            "Missing or non-finite belief model gradient; optimizer step rejected"
          )

      def grad_norm(module):
        return (
          torch.stack([p.grad.detach().square().sum() for p in module.parameters()])
          .sum()
          .sqrt()
        )

      mean_norm = grad_norm(self.dynamics.mlp)
      noise_norm = grad_norm(self.dynamics.noise_head)
      norm = torch.nn.utils.clip_grad_norm_(
        self._all_params, self.cfg.max_grad_norm, error_if_nonfinite=True
      )
      self.optimizer.step()
      self.steps += 1
    except Exception:
      self.optimizer.zero_grad(set_to_none=True)
      raise
    finally:
      self.dynamics.eval()
    return {
      **{name: value / count for name, value in totals.items()},
      "Grad/mean_norm": mean_norm,
      "Grad/noise_norm": noise_norm,
      "Grad/total_norm": norm.detach(),
      "valid_transitions": mean_norm.new_tensor(count),
    }

  def _contract(self):
    return {
      "filter": asdict(self.estimator.config),
      "measurement": asdict(self.estimator.measurement_config),
      "default_joint_pos": self.estimator.default_joint_pos.tolist(),
      "parameter_names": [name for name, _ in self.dynamics.named_parameters()],
    }

  def state_dict(self) -> dict:
    """Save alongside the full model bundle; normalization lives in that bundle."""
    return {
      "version": "belief_world_model_learner_v1",
      "cfg": asdict(self.cfg),
      "contract": self._contract(),
      "model_version": self.dynamics.model_version,
      "steps": self.steps,
      "optimizer": copy.deepcopy(self.optimizer.state_dict()),
    }

  def validate_state_dict(self, state: dict) -> None:
    """Preflight optimizer and metadata without modifying any live parameter."""
    if (
      not isinstance(state, dict)
      or state.get("version") != "belief_world_model_learner_v1"
    ):
      raise ValueError("Unsupported belief learner checkpoint")
    if (
      state.get("cfg") != asdict(self.cfg) or state.get("contract") != self._contract()
    ):
      raise ValueError("Belief learner checkpoint configuration mismatch")
    if state.get("model_version") != self.dynamics.model_version:
      raise ValueError("Belief learner checkpoint requires the matching model bundle")
    steps = state.get("steps")
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 0:
      raise ValueError("Belief learner checkpoint steps must be a nonnegative integer")
    supplied = state.get("optimizer")
    expected = self.optimizer.state_dict()
    if (
      not isinstance(supplied, dict)
      or set(supplied) != {"state", "param_groups"}
      or supplied["param_groups"] != expected["param_groups"]
    ):
      raise ValueError("Belief learner optimizer groups mismatch")
    optimizer_state = supplied["state"]
    ids = expected["param_groups"][0]["params"]
    if not isinstance(optimizer_state, dict) or set(optimizer_state) != (
      set(ids) if steps else set()
    ):
      raise ValueError("Belief learner optimizer parameter states mismatch")
    for index, parameter in zip(ids, self._all_params, strict=True):
      if not steps:
        continue
      values = optimizer_state[index]
      if not isinstance(values, dict) or set(values) != {
        "step",
        "exp_avg",
        "exp_avg_sq",
      }:
        raise ValueError("Invalid Adam state fields")
      for name, value in values.items():
        if (
          not isinstance(value, torch.Tensor)
          or value.layout != torch.strided
          or value.device.type == "meta"
          or not torch.isfinite(value).all()
        ):
          raise ValueError("Adam state must contain finite tensors")
        if name == "step":
          if (
            value.ndim != 0
            or value.dtype not in (torch.float32, torch.float64)
            or float(value) != steps
          ):
            raise ValueError("Adam step does not match learner step")
        elif value.shape != parameter.shape or value.dtype != parameter.dtype:
          raise ValueError("Adam moment shape/dtype mismatch")
        elif name == "exp_avg_sq" and (value < 0).any():
          raise ValueError("Adam second moment must be nonnegative")

  def load_state_dict(self, state: dict) -> None:
    self.validate_state_dict(state)
    optimizer = copy.deepcopy(state["optimizer"])
    # torch.load(map_location="cuda") also moves the scalar Adam clock. Our
    # noncapturable Adam keeps that clock on CPU during ordinary initialization;
    # restore the same device contract instead of introducing a CUDA step tensor.
    for values in optimizer["state"].values():
      values["step"] = values["step"].cpu()
    self.optimizer.load_state_dict(optimizer)
    self.steps = state["steps"]
    self.optimizer.zero_grad(set_to_none=True)
