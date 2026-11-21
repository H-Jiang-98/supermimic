"""World model learner (supervised training)."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

from supermimic.learners.learner_utils import (
  CapturedUpdate,
  _optimize_step,
  build_optimizers,
  graph_state,
)

if TYPE_CHECKING:
  from supermimic.models import Dynamics
  from supermimic.representations import StateRepresentation


@dataclass
class WorldModelLearnerCfg:
  lr: float = 2e-4
  num_batches: int = 10
  batch_size: int = 2048
  horizon: int = 8

  # Loss Weights (Bryson's Rule) — integrated root pose, both representations
  nominal_error_root_pose_pos: float = 0.01
  nominal_error_root_pose_ori: float = 0.02

  # Loss Weights — gencoord representation
  nominal_error_joint_pos: float = 0.1
  nominal_error_joint_vel: float = 3.14

  # Loss Weights — maxcoord representation
  nominal_error_body_pos: float = 0.05
  nominal_error_body_rot: float = 0.1
  nominal_error_body_vel: float = 1.0
  nominal_error_body_avel: float = 3.14


class WorldModelLearner:
  """Trainer for World Model (Supervised)."""

  def __init__(
    self,
    dynamics: Dynamics,
    representation: StateRepresentation,
    cfg: WorldModelLearnerCfg,
    device: str = "cpu",
    *,
    cuda_graphs: bool = True,
    compile_kernels: bool = True,
    check_finite: bool = False,
  ) -> None:
    self.dynamics = dynamics
    self.representation = representation
    self.cfg = cfg
    self.device = device
    self._check_finite = check_finite

    mean_model = dynamics.mlp if dynamics.noise_head is not None else dynamics
    self._transition = (
      dynamics.mean_rep if dynamics.noise_head is not None else dynamics
    )
    self._all_params = list(mean_model.parameters())
    self.optimizers = build_optimizers([mean_model], self.cfg.lr, device=self.device)

    # Bryson's rule: each named channel group is weighted 1 / nominal_error^2.
    self.loss_group_weights = {
      name: 1.0 / getattr(cfg, attr) ** 2
      for name, attr in representation.LOSS_WEIGHT_CONFIG.items()
    }

    self._compiled_rollout = torch.compile(self._rollout, disable=not compile_kernels)
    self._compiled_compute_losses = torch.compile(
      self._compute_losses, disable=not compile_kernels
    )
    self._compiled_represent = torch.compile(
      self.representation.represent, disable=not compile_kernels
    )
    self._captured_update = CapturedUpdate(
      self._update_from,
      lambda: graph_state([dynamics, representation], self.optimizers),
      enabled=cuda_graphs,
    )

  def update_normalization(
    self, states: torch.Tensor, actions: torch.Tensor, *, normalizers=None
  ) -> None:
    """Update the dynamics state/action normalizers from fresh collected data.

    ``actions`` are raw policy outputs (what the replay buffer stores); the
    action normalizer sees proc (matching what ``forward`` feeds the MLP after
    ``Dynamics.proc``).
    """
    rep_states = self.representation.represent(states)
    rep_states_flat = rep_states.reshape(-1, rep_states.shape[-1])
    actions_flat = actions.reshape(-1, actions.shape[-1])

    # EmpiricalNormalization.update is a no-op in eval mode; toggle to train
    # for the updates and restore afterward.
    if normalizers is None:
      normalizers = self.dynamics.normalizers()
    for n in normalizers:
      n.train()
    state_normalizer, action_normalizer = normalizers
    try:
      state_normalizer.update(rep_states_flat)
      action_normalizer.update(self.dynamics.proc(actions_flat))
    finally:
      for n in normalizers:
        n.eval()

  def update(self, data: Iterable[TensorDict]) -> dict[str, torch.Tensor]:
    self.dynamics.train()
    for normalizer in self.dynamics.normalizers():
      normalizer.eval()
    for p in self._all_params:
      p.requires_grad_(True)
    batches = torch.stack([td.select("state", "action", "sim_ctx") for td in data])
    return self._captured_update(batches)

  def _update_from(self, batches: TensorDict) -> dict[str, torch.Tensor]:
    stats = []

    for td in batches.unbind(0):
      loss_state, group_mse = self.batch_loss(td)

      _optimize_step(
        self.optimizers,
        loss_state,
        self._all_params,
        max_norm=1.0,
        check_finite=self._check_finite,
      )

      stats.append(torch.cat([loss_state.unsqueeze(0), group_mse]).detach())

    names = ["state", *self.loss_group_weights]
    means = torch.stack(stats).mean(0)
    return {f"Loss/{name}": m for name, m in zip(names, means, strict=True)}

  def batch_loss(self, td: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
    """Score predictions after the real first frame against the recorded successors."""
    states = self._compiled_represent(td["state"])
    aux = self.representation.extract_aux_state(td["state"])
    pred, pred_aux = self._compiled_rollout(
      states[:, 0], td["action"], aux[:, 0], td["sim_ctx"][:, 0]
    )
    return self._compiled_compute_losses(pred, pred_aux, states[:, 1:], aux[:, 1:])

  def _rollout(
    self,
    init_rep_state: torch.Tensor,
    actions: torch.Tensor,
    init_aux: torch.Tensor,
    sim_ctx: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Autoregressive rollout over the window's actions.

    The prediction from the last action is never scored, so the rollout stops
    one step short.

    Returns:
      (pred_rep_states, pred_aux_states), stacked on dim 1.
    """
    rep_state, aux = init_rep_state, init_aux
    pred_rep_states: list[torch.Tensor] = []
    pred_aux_states: list[torch.Tensor] = []
    for k in range(actions.shape[1] - 1):
      aux, rep_state = self._transition(rep_state, actions[:, k], aux, sim_ctx)
      pred_rep_states.append(rep_state)
      pred_aux_states.append(aux)
    return torch.stack(pred_rep_states, dim=1), torch.stack(pred_aux_states, dim=1)

  def _compute_losses(
    self,
    pred_rep_states: torch.Tensor,
    pred_aux_states: torch.Tensor,
    gt_rep_states: torch.Tensor,
    gt_aux_states: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    # Squared error per named channel group. Bryson's rule scales each group by
    # its weight; the state loss is the plain mean over every channel.
    loss_blocks = self.representation.loss_blocks
    pred_blocks = loss_blocks(pred_rep_states, pred_aux_states)
    gt_blocks = loss_blocks(gt_rep_states, gt_aux_states)
    sq_errs = {name: (pred_blocks[name] - gt_blocks[name]) ** 2 for name in pred_blocks}
    group_mse = torch.stack([sq.mean() for sq in sq_errs.values()])
    # Filled on the device: a captured update cannot copy from the host.
    widths = torch.stack(
      [group_mse.new_full((), sq.shape[-1]) for sq in sq_errs.values()]
    )
    weights = torch.stack(
      [group_mse.new_full((), self.loss_group_weights[name]) for name in sq_errs]
    )
    loss_state = (group_mse * weights * widths).sum() / widths.sum()
    return loss_state, group_mse
