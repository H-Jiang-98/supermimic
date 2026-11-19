"""SuperMimic motion extensions on top of public mjlab's tracking command."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
from mjlab.tasks.tracking.mdp.commands import MotionCommand as BaseMotionCommand
from mjlab.tasks.tracking.mdp.commands import MotionCommandCfg as BaseMotionCommandCfg
from mjlab.utils.lab_api.math import matrix_from_quat, quat_from_matrix, quat_mul

from supermimic.estimator_inputs import (
  EpisodeAlignment,
  RpvState,
  create_episode_alignment,
)
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
    self.reference_alignment = EpisodeAlignment.identity(
      self.num_envs, device=self.device, dtype=self.motion.root_pos_w.dtype
    )
    self._just_reset = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self._sampling_weights = torch.full(
      (self.bin_count,), 1 / self.bin_count, device=self.device
    )

  def align_reference_positions(self, positions: torch.Tensor) -> torch.Tensor:
    """Transform raw motion points into fixed episode coordinates, without origins."""
    if not self.cfg.fixed_reference_alignment:
      return positions
    rotation = self.reference_alignment.rotation[:, None]
    return (rotation @ positions.unsqueeze(-1)).squeeze(-1) + (
      self.reference_alignment.translation[:, None]
    )

  def align_reference_quaternions(self, quaternions: torch.Tensor) -> torch.Tensor:
    if not self.cfg.fixed_reference_alignment:
      return quaternions
    rotation = quat_from_matrix(self.reference_alignment.rotation)[:, None]
    return quat_mul(rotation.expand_as(quaternions), quaternions)

  def _align_reference_vectors(self, vectors: torch.Tensor) -> torch.Tensor:
    if not self.cfg.fixed_reference_alignment:
      return vectors
    return (self.reference_alignment.rotation[:, None] @ vectors.unsqueeze(-1)).squeeze(
      -1
    )

  @property
  def body_pos_w(self) -> torch.Tensor:
    return (
      self.align_reference_positions(self.motion.body_pos_w[self.time_steps])
      + (self._env.scene.env_origins[:, None])
    )

  @property
  def body_quat_w(self) -> torch.Tensor:
    return self.align_reference_quaternions(self.motion.body_quat_w[self.time_steps])

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    return self._align_reference_vectors(self.motion.body_lin_vel_w[self.time_steps])

  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    return self._align_reference_vectors(self.motion.body_ang_vel_w[self.time_steps])

  @property
  def anchor_pos_w(self) -> torch.Tensor:
    return self.body_pos_w[:, self.motion_anchor_body_index]

  @property
  def anchor_quat_w(self) -> torch.Tensor:
    return self.body_quat_w[:, self.motion_anchor_body_index]

  @property
  def anchor_lin_vel_w(self) -> torch.Tensor:
    return self.body_lin_vel_w[:, self.motion_anchor_body_index]

  @property
  def anchor_ang_vel_w(self) -> torch.Tensor:
    return self.body_ang_vel_w[:, self.motion_anchor_body_index]

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
    # The new episode must start from raw reference data, not the previous
    # episode's transform. _write_reference_state_to_sim captures the new pose.
    if self.cfg.fixed_reference_alignment:
      self.reference_alignment.rotation[env_ids] = torch.eye(
        3, device=self.device, dtype=self.reference_alignment.rotation.dtype
      )
      self.reference_alignment.translation[env_ids] = 0
    super()._resample_command(env_ids)
    self._just_reset[env_ids] = True

  def _write_reference_state_to_sim(
    self,
    env_ids: torch.Tensor,
    root_pos: torch.Tensor,
    root_ori: torch.Tensor,
    root_lin_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
  ) -> None:
    if self.cfg.fixed_reference_alignment:
      frames = self.time_steps[env_ids]
      alignment = create_episode_alignment(
        RpvState(
          matrix_from_quat(root_ori),
          root_pos - self._env.scene.env_origins[env_ids],
          root_lin_vel,
        ),
        RpvState(
          matrix_from_quat(self.motion.root_quat_w[frames]),
          self.motion.root_pos_w[frames],
          self.motion.root_lin_vel_w[frames],
        ),
      )
      self.reference_alignment.rotation[env_ids] = alignment.rotation
      self.reference_alignment.translation[env_ids] = alignment.translation
    super()._write_reference_state_to_sim(
      env_ids,
      root_pos,
      root_ori,
      root_lin_vel,
      root_ang_vel,
      joint_pos,
      joint_vel,
    )

  def update_relative_body_poses(self) -> None:
    # Keep the hybrid reward: fixed episode references for actor/global terms,
    # root-xy/yaw-relative body poses for the body-shape terms. This derives a
    # scoring target without changing the saved reference trajectory, and matches
    # tracking_reward._reanchor_targets in the imagined rollout.
    super().update_relative_body_poses()

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
  fixed_reference_alignment: bool = False

  def build(self, env: ManagerBasedRlEnv) -> MotionCommand:
    return MotionCommand(self, env)
