"""Autoregressive mean fitting followed by conditional process covariance fitting."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import asdict, dataclass

import torch
from tensordict import TensorDict

from supermimic.learners.learner_utils import validate_optimizer_state
from supermimic.learners.wm_learner import WorldModelLearner, WorldModelLearnerCfg
from supermimic.models import Dynamics
from supermimic.replay_buffer import ReplayBuffer
from supermimic.representations import StateRepresentation
from supermimic.utils.lie_state import LieState, local_error
from supermimic.utils.profiling import PhaseTimer


@dataclass(frozen=True)
class MeanFitCfg:
  lr: float = 2e-4
  num_batches: int = 10
  batch_size: int = 2048
  window_frames: int = 8

  def __post_init__(self):
    _check_sizes(self.lr, self.num_batches, self.batch_size, self.window_frames)
    if self.window_frames < 2:
      raise ValueError("mean window_frames must include a successor")


@dataclass(frozen=True)
class CovarianceFitCfg:
  lr: float = 2e-4
  batch_size: int = 8192
  pair_budget: int | None = None
  validation_pairs: int = 8192

  def __post_init__(self):
    _check_sizes(self.lr, self.batch_size, self.validation_pairs)
    if self.pair_budget is not None:
      _check_sizes(self.lr, self.pair_budget)

  def training_pairs(self, mean: MeanFitCfg) -> int:
    return (
      self.pair_budget
      if self.pair_budget is not None
      else (mean.num_batches * mean.batch_size * (mean.window_frames - 1))
    )


def _check_sizes(lr, *sizes):
  if not math.isfinite(lr) or lr <= 0:
    raise ValueError("Learning rate must be finite and positive")
  if any(type(n) is not int or n < 1 for n in sizes):
    raise ValueError("Fit sizes must be positive integers")


def process_nll(residual: torch.Tensor, lower: torch.Tensor) -> torch.Tensor:
  """Per-transition negative log density in the right-local tangent coordinates."""
  whitened = torch.linalg.solve_triangular(lower, residual.unsqueeze(-1), upper=False)
  log_scale = lower.diagonal(dim1=-2, dim2=-1).log().sum(-1)
  return log_scale + 0.5 * (
    whitened.square().sum((-2, -1)) + residual.shape[-1] * math.log(2 * math.pi)
  )


class TwoStageWorldModelLearner:
  def __init__(
    self,
    dynamics: Dynamics,
    representation: StateRepresentation,
    mean_cfg: MeanFitCfg,
    cov_cfg: CovarianceFitCfg,
    device: str,
    *,
    compile_kernels: bool = False,
  ):
    if dynamics.noise_head is None:
      raise ValueError("Two-stage fitting requires a process noise head")
    self.dynamics = dynamics
    self.noise_head = dynamics.noise_head
    self.mean_cfg, self.cov_cfg = mean_cfg, cov_cfg
    self.mean = WorldModelLearner(
      dynamics,
      representation,
      WorldModelLearnerCfg(
        lr=mean_cfg.lr,
        num_batches=mean_cfg.num_batches,
        batch_size=mean_cfg.batch_size,
        horizon=mean_cfg.window_frames,
      ),
      device,
      cuda_graphs=False,
      compile_kernels=compile_kernels,
      check_finite=True,
    )
    self.cov_optimizer = torch.optim.Adam(self.noise_head.parameters(), lr=cov_cfg.lr)
    self.mean_steps = self.cov_steps = 0
    self.generators = {}
    for offset, name in enumerate(
      ("mean_train", "mean_validation", "cov_train", "cov_validation")
    ):
      self.generators[name] = torch.Generator(device=device).manual_seed(
        (torch.initial_seed() + 10 + offset) % (2**63 - 1)
      )
    self.dynamics.eval().requires_grad_(False)

  @torch.no_grad()
  def stage_normalization(
    self, states: torch.Tensor, actions: torch.Tensor, staged=None
  ):
    """Accumulate a chunk into private statistics while collection uses the live model."""
    if (
      states.ndim < 1
      or actions.ndim < 1
      or states.shape[:-1] != actions.shape[:-1]
      or states.shape[-1] != 71
      or actions.shape[-1] != 29
      or not states.numel()
      or not torch.isfinite(states).all()
      or not torch.isfinite(actions).all()
    ):
      raise ValueError(
        "Normalization requires aligned finite training states and actions"
      )
    norm_sq = states[..., 3:7].square().sum(-1)
    if not (torch.isfinite(norm_sq) & (norm_sq > 0)).all():
      raise ValueError("Invalid training state quaternion")
    if staged is None:
      staged = deepcopy(self.dynamics.normalizers())
    self.mean.update_normalization(states, actions, normalizers=staged)
    if not all(torch.isfinite(v).all() for n in staged for v in n.buffers()):
      raise ValueError("Non-finite model normalization statistics")
    return staged

  def _commit_normalization(self, staged) -> None:
    self.dynamics.begin_write()
    for live, candidate in zip(self.dynamics.normalizers(), staged, strict=True):
      live.load_state_dict(candidate.state_dict())
      live.eval()
    self.dynamics.finish_write()

  def update_normalization(self, states: torch.Tensor, actions: torch.Tensor) -> None:
    self._commit_normalization(self.stage_normalization(states, actions))

  def update_mean(self, windows: TensorDict):
    self.noise_head.requires_grad_(False)
    self.noise_head.zero_grad(set_to_none=True)
    # Eager scripted quaternion kernels otherwise switch arithmetic after profiling
    # a new batch shape, which changes the first update after a process restart.
    try:
      with torch.jit.optimized_execution(False):
        logs = self.mean.update([windows])
    except Exception:
      self.dynamics.zero_grad(set_to_none=True)
      self.dynamics.eval().requires_grad_(False)
      raise
    self.mean_steps += 1
    return logs

  @torch.no_grad()
  def residual_data(self, pairs: TensorDict) -> TensorDict:
    current = LieState.from_full(pairs["state"][:, 0])
    successor = LieState.from_full(pairs["state"][:, 1])
    action, context = pairs["action"][:, 0], pairs["sim_ctx"][:, 0]
    predicted = self.dynamics.mean_lie(current, action, context)
    return TensorDict(
      {
        "inputs": self.dynamics.lie_inputs(current, action, context),
        "residual": local_error(predicted, successor),
      },
      [len(pairs)],
    )

  def covariance_loss(self, batch: TensorDict) -> torch.Tensor:
    return process_nll(batch["residual"], self.dynamics.noise_factor(batch["inputs"]))

  def update_covariance(self, batch: TensorDict) -> torch.Tensor:
    self.dynamics.begin_write()
    self.dynamics.eval()
    self.dynamics.mlp.requires_grad_(False)
    self.dynamics.mlp.zero_grad(set_to_none=True)
    self.noise_head.requires_grad_(True)
    self.cov_optimizer.zero_grad(set_to_none=True)
    loss = self.covariance_loss(batch).mean()
    if not torch.isfinite(loss):
      raise FloatingPointError("Non-finite process covariance loss")
    loss.backward()
    # Adam stores squared gradients in parameter precision. A finite gradient
    # can still overflow that second moment and produce an unusable checkpoint.
    if not all(
      torch.isfinite(p.grad.square()).all() for p in self.noise_head.parameters()
    ):
      self.cov_optimizer.zero_grad(set_to_none=True)
      raise FloatingPointError("Non-finite squared process covariance gradient")
    self.cov_optimizer.step()
    self.dynamics.finish_write()
    self.cov_steps += 1
    return loss.detach()

  def fit(
    self,
    replay: ReplayBuffer,
    train_ids,
    valid_ids,
    fresh_train: TensorDict,
    validation_batch_size: int,
    *,
    staged_normalizers=None,
  ) -> tuple[dict, dict]:
    cfg, cov = self.mean_cfg, self.cov_cfg
    timer = PhaseTimer(next(self.dynamics.parameters()).device)
    self.last_profile = timer.seconds
    mean_before, cov_before = self.mean_steps, self.cov_steps
    windows, valid_windows, pairs, held_out = replay.model_fit_starts(
      train_ids, valid_ids, cfg.window_frames
    )
    if not len(windows):
      raise ValueError(
        f"No complete {cfg.window_frames}-frame training window in replay"
      )
    if not len(held_out):
      raise ValueError("No held-out transition with a recorded successor in replay")

    def sample(starts, batch_size, frames, stream):
      indices = torch.randint(
        len(starts),
        (batch_size,),
        device=starts.device,
        generator=self.generators[stream],
      )
      return replay.model_windows(starts[indices], frames)

    if len(fresh_train):
      staged_normalizers = self.stage_normalization(
        fresh_train["state"], fresh_train["action"], staged_normalizers
      )
    if staged_normalizers is not None:
      self._commit_normalization(staged_normalizers)
    try:
      with timer.measure("mean_rollout_update_seconds"):
        updates = [
          self.update_mean(
            sample(windows, cfg.batch_size, cfg.window_frames, "mean_train")
          )
          for _ in range(cfg.num_batches)
        ]
      train = {
        key: float(torch.stack([u[key] for u in updates]).mean()) for key in updates[0]
      }
      train["mean_windows"] = cfg.num_batches * cfg.batch_size
      train["available_mean_windows"] = len(windows)
      train["mean_updates"] = self.mean_steps - mean_before
      self.dynamics.eval().requires_grad_(False)
      self.dynamics.zero_grad(set_to_none=True)

      # Only the sampled batch's detached residuals exist at any one time.
      budget = cov.training_pairs(cfg)
      total = torch.zeros((), dtype=torch.float64, device=pairs.device)
      for offset in range(0, budget, cov.batch_size):
        size = min(cov.batch_size, budget - offset)
        with timer.measure("residual_extraction_seconds"):
          data = self.residual_data(sample(pairs, size, 2, "cov_train"))
        with timer.measure("covariance_fit_seconds"):
          total += self.update_covariance(data).double() * size
        del data
      train["cov_pairs"] = len(pairs)
      train["cov_pair_draws"] = budget
      train["cov_updates"] = self.cov_steps - cov_before
      train["Loss/process_nll"] = float(total / budget)

      with torch.no_grad(), timer.measure("model_validation_seconds"):
        validation = {
          "mean_windows": validation_batch_size if len(valid_windows) else 0,
          "available_mean_windows": len(valid_windows),
        }
        if len(valid_windows):
          loss, groups = self.mean.batch_loss(
            sample(
              valid_windows, validation_batch_size, cfg.window_frames, "mean_validation"
            )
          )
          validation["Loss/state"] = float(loss)
          validation.update(
            {
              f"Loss/{k}": float(v)
              for k, v in zip(self.mean.loss_group_weights, groups, strict=True)
            }
          )
        total.zero_()
        for offset in range(0, cov.validation_pairs, cov.batch_size):
          size = min(cov.batch_size, cov.validation_pairs - offset)
          batch = self.residual_data(sample(held_out, size, 2, "cov_validation"))
          total += self.covariance_loss(batch).double().sum()
          del batch
        validation["cov_pairs"] = len(held_out)
        validation["cov_pair_draws"] = cov.validation_pairs
        validation["Loss/process_nll"] = float(total / cov.validation_pairs)
      for phase, metrics in (("training", train), ("validation", validation)):
        for key, value in metrics.items():
          if not math.isfinite(value):
            raise FloatingPointError(f"Non-finite {phase} metric {key}")
      return train, validation
    finally:
      self.dynamics.zero_grad(set_to_none=True)
      self.dynamics.eval().requires_grad_(False)

  def state_dict(self):
    return deepcopy(
      {
        "model_version": self.dynamics.model_version,
        "rng": {
          name: generator.get_state() for name, generator in self.generators.items()
        },
        "mean": {
          "config": asdict(self.mean_cfg),
          "steps": self.mean_steps,
          "optimizers": [opt.state_dict() for opt in self.mean.optimizers],
        },
        "cov": {
          "config": asdict(self.cov_cfg),
          "steps": self.cov_steps,
          "optimizer": self.cov_optimizer.state_dict(),
        },
      }
    )

  def load_state_dict(self, state):
    if state.get("model_version") != self.dynamics.model_version:
      raise ValueError("Optimizer checkpoint requires matching model weights")
    if set(state.get("rng", {})) != set(self.generators):
      raise ValueError("World model RNG streams mismatch")
    for name, generator in self.generators.items():
      torch.Generator(device=generator.device).set_state(state["rng"][name].cpu())
    mean, cov = state["mean"], state["cov"]
    if mean.get("config") != asdict(self.mean_cfg) or cov.get("config") != asdict(
      self.cov_cfg
    ):
      raise ValueError("World model fitting configuration mismatch")
    if len(mean["optimizers"]) != len(self.mean.optimizers):
      raise ValueError("Mean optimizer count mismatch")
    items = [
      *zip(self.mean.optimizers, mean["optimizers"], strict=True),
      (self.cov_optimizer, cov["optimizer"]),
    ]
    for opt, saved in items:
      validate_optimizer_state(
        opt, saved, cov["steps"] if opt is self.cov_optimizer else mean["steps"]
      )
    for opt, saved in items:
      saved = deepcopy(saved)
      if opt is self.cov_optimizer:
        for moments in saved["state"].values():
          moments["step"] = moments["step"].cpu()
      opt.load_state_dict(saved)
      opt.zero_grad(set_to_none=True)
    self.mean_steps, self.cov_steps = mean["steps"], cov["steps"]
    for name, generator in self.generators.items():
      generator.set_state(state["rng"][name].cpu())
