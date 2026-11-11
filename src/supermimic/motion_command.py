"""SuperMimic motion extensions on top of public mjlab's tracking command."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
from mjlab.tasks.tracking.mdp.commands import MotionCommand as BaseMotionCommand
from mjlab.tasks.tracking.mdp.commands import MotionCommandCfg as BaseMotionCommandCfg

from supermimic.motion_loader import MotionLoader
from supermimic.utils.forward_kinematics import ForwardKinematics

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class MotionCommand(BaseMotionCommand):
  cfg: MotionCommandCfg

  def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRlEnv):
    if not hasattr(BaseMotionCommand, "_load_motion") or not hasattr(
      BaseMotionCommand, "_sample_time_steps"
    ):
      raise RuntimeError(
        "Use the sibling public mjlab checkout with the SuperMimic motion hooks"
      )
    super().__init__(cfg, env)
    self._just_reset = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self._sampling_weights = torch.full(
      (self.bin_count,), 1 / self.bin_count, device=self.device
    )

  def _load_motion(self) -> MotionLoader:
    if int(self.body_indexes[0]) != 0:
      raise ValueError("The first tracked body must be the robot root")
    fk = ForwardKinematics(
      self._env.sim.mj_model,
      self.robot.indexing.body_ids,
      self.robot.indexing.joint_q_adr,
      self.body_indexes,
      self.device,
    )
    return MotionLoader(
      self.cfg.motion_file,
      fk,
      self.robot.num_joints,
      self.device,
      expected_fps=1 / self._env.step_dt,
    )

  def set_sampling_distribution(self, weights: torch.Tensor) -> None:
    if self.cfg.sampling_mode != "external":
      raise ValueError("set_sampling_distribution requires sampling_mode='external'")
    weights = weights.detach().to(self._sampling_weights)
    if weights.shape != self._sampling_weights.shape:
      raise ValueError(
        f"Expected {self.bin_count} sampling weights, got {tuple(weights.shape)}"
      )
    total = weights.sum()
    if (
      not torch.isfinite(weights).all()
      or not torch.isfinite(total)
      or (weights < 0).any()
      or total <= 0
    ):
      raise ValueError(
        "Sampling weights must be finite, nonnegative and have positive sum"
      )
    self._sampling_weights.copy_(weights / total)

  def _sample_time_steps(self, env_ids: torch.Tensor) -> None:
    if self.cfg.sampling_mode == "external":
      bins = torch.multinomial(self._sampling_weights, len(env_ids), replacement=True)
      # Exact inverse of ReturnSampler's floor(frame * bin_count / frame_count).
      count = self.motion.time_step_total
      low = (bins * count + self.bin_count - 1) // self.bin_count
      high = ((bins + 1) * count + self.bin_count - 1) // self.bin_count
      self.time_steps[env_ids] = (
        low + (torch.rand(len(env_ids), device=self.device) * (high - low)).long()
      )
      p = self._sampling_weights
      entropy = -(p * p.clamp_min(1e-12).log()).sum()
      self.metrics["sampling_entropy"][env_ids] = (
        entropy / torch.log(p.new_tensor(self.bin_count)) if self.bin_count > 1 else 1.0
      )
      self.metrics["sampling_top1_prob"][env_ids] = p.max()
      self.metrics["sampling_top1_bin"][env_ids] = p.argmax().float() / self.bin_count
    elif self.cfg.sampling_mode == "start":
      self.time_steps[env_ids] = 0
    else:
      super()._sample_time_steps(env_ids)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    self._just_reset[env_ids] = True

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    ids = slice(None) if env_ids is None else env_ids
    frames = self.time_steps[ids]
    # A reset must expose the sampled frame, including during an auto-reset.
    # At clip end the termination manager resets the episode. With auto_reset
    # disabled, hold the last reference instead of jumping into another clip.
    self.time_steps[ids] = torch.minimum(
      frames + (~self._just_reset[ids]).long(), self.motion.last_frame(frames)
    )
    self._just_reset[ids] = False
    if self._pending_forward:
      self._env.sim.forward()
      self._pending_forward = False
    self.update_relative_body_poses()
    if env_ids is None and self.cfg.sampling_mode == "adaptive":
      self.bin_failed_count.lerp_(self._current_bin_failed, self.cfg.adaptive_alpha)
      self._current_bin_failed.zero_()


def motion_end(env: ManagerBasedRlEnv, command_name: str = "motion") -> torch.Tensor:
  command = env.command_manager.get_term(command_name)
  return command.time_steps >= command.motion.last_frame(command.time_steps)


@dataclass(kw_only=True)
class MotionCommandCfg(BaseMotionCommandCfg):
  motion_file: str | list[str]
  sampling_mode: Literal["adaptive", "uniform", "start", "external"] = "adaptive"

  def build(self, env: ManagerBasedRlEnv) -> MotionCommand:
    return MotionCommand(self, env)
