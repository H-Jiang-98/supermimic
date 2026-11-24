"""RPV-belief privileged critic inputs and explicitly updated normalizers.

The common prefix is [truth68, truth_goal700, history465, context]. B/C append a relative RPV mean (12) and the right-local 9x9 marginal
covariance (45). The provider still owns the full 70-dimensional belief.
A never reads the supplied belief, including during eval.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, replace
from typing import Literal

import torch
from torch import nn

from supermimic.belief import BeliefState
from supermimic.estimator_inputs import covariance_encoding_constants
from supermimic.models import _CudaGraphNormalization, build_mlp
from supermimic.utils.lie_state import TANGENT_VERSION, LieState
from supermimic.utils.validation import check_covariance, validate_state_dict

CRITIC_INPUT_VERSION = "eit_rpv_belief_critic_v3"
COMMON_DIM = 1233
MEAN_DIM = 12
COVARIANCE_DIM = 45


@dataclass(frozen=True)
class CriticInputConfig:
  group: Literal["A", "B", "C"] = "B"
  version: str = CRITIC_INPUT_VERSION
  tangent_version: str = TANGENT_VERSION
  covariance_encoding: str = "rpv9_log_std_corr_v1"
  covariance_scales: tuple[float, ...] = (1.0,) * 3  # theta, rho, nu
  mean_scales: tuple[float, ...] = (1.0,) * 2  # position, velocity; rot6d is unitless
  epsilon_std: float = 1e-6

  def __post_init__(self):
    if self.group not in ("A", "B", "C"):
      raise ValueError("Critic group must be A, B or C")
    if (
      self.version != CRITIC_INPUT_VERSION
      or self.tangent_version != TANGENT_VERSION
      or self.covariance_encoding != "rpv9_log_std_corr_v1"
    ):
      raise ValueError("Unsupported critic input/covariance version")
    for name, length in (("covariance_scales", 3), ("mean_scales", 2)):
      values = getattr(self, name)
      if len(values) != length or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError(f"{name} requires {length} finite positive scales")
    if not math.isfinite(self.epsilon_std) or self.epsilon_std <= 0:
      raise ValueError("epsilon_std must be finite and positive")

  def input_dim(self, context_dim: int) -> int:
    if type(context_dim) is not int or context_dim < 0:
      raise ValueError("context_dim must be a nonnegative integer")
    return (
      COMMON_DIM + context_dim + (0 if self.group == "A" else MEAN_DIM + COVARIANCE_DIM)
    )


def _config(config: CriticInputConfig | None, group: str | None):
  if config is None:
    return CriticInputConfig(group="B" if group is None else group)
  if group is not None and config.group != group:
    raise ValueError("Critic group and input configuration disagree")
  return config


def encode_rpv_covariance(
  covariance: torch.Tensor, config: CriticInputConfig | None = None
) -> torch.Tensor:
  """Encode a theta/rho/nu marginal as nine log stds and 36 correlations."""
  cfg = config or CriticInputConfig()
  if covariance.shape[-2:] != (9, 9) or not covariance.is_floating_point():
    raise ValueError("Critic RPV covariance must have shape (...,9,9)")
  check_covariance(
    covariance, "Critic covariance", rtol=128 * torch.finfo(covariance.dtype).eps
  )
  constants = critic_feature_constants(covariance, cfg)
  features = encode_rpv_covariance_kernel(covariance, constants)
  if not torch.isfinite(features).all():
    raise ValueError("Critic covariance encoding produced non-finite features")
  return features


def critic_feature_constants(reference, config):
  scale, _, rows, cols = covariance_encoding_constants(
    reference, config.covariance_scales, config.epsilon_std
  )
  floor = reference.new_tensor(config.epsilon_std).square()
  mean_scale = reference.new_tensor(
    [1.0] * 6 + [s for s in config.mean_scales for _ in range(3)]
  )
  if not (
    torch.isfinite(mean_scale).all()
    and (mean_scale > 0).all()
    and torch.isfinite(floor)
    and floor > 0
  ):
    raise ValueError("Critic scales/floor are not representable in this dtype")
  return scale, floor, rows, cols, mean_scale


def encode_rpv_covariance_kernel(covariance, constants):
  scale, floor, rows, cols, _ = constants
  scaled = (
    (covariance + covariance.mT) * 0.5 / scale.unsqueeze(-1) / scale.unsqueeze(-2)
  )
  std = scaled.diagonal(dim1=-2, dim2=-1).clamp_min(floor).sqrt()
  correlation = scaled / std.unsqueeze(-1) / std.unsqueeze(-2)
  return torch.cat((std.log(), correlation[..., rows, cols].clamp(-1, 1)), -1)


def critic_state_features(truth: LieState) -> torch.Tensor:
  """The critic's 68 privileged state channels, also used by its normalizer."""
  return torch.cat(
    (
      truth.position[..., 2:3],
      -truth.rotation[..., 2, :],
      (truth.rotation.mT @ truth.velocity.unsqueeze(-1)).squeeze(-1),
      truth.omega,
      truth.joint_pos,
      truth.joint_vel,
    ),
    -1,
  )


def build_critic_features(
  truth: LieState,
  true_goal: torch.Tensor,
  history: torch.Tensor,
  sim_ctx: torch.Tensor,
  belief: BeliefState | None = None,
  *,
  config: CriticInputConfig | None = None,
  group: str | None = None,
) -> torch.Tensor:
  """Build raw critic features; the caller verifies provider frame identity.

  History contains five frames, oldest first,
  with the actually executed previous action in each frame's final 29 channels.
  Validate the provider's complete belief, then project only RPV into B/C inputs.
  Marginal extraction preserves the rollout graph and never conditions on truth.
  """
  cfg = _config(config, group)
  truth.validate()
  batch = truth.position.shape[:-1]
  if sim_ctx.ndim != len(batch) + 1 or sim_ctx.shape[:-1] != batch:
    raise ValueError("Critic context must match truth batch shape")
  for name, value, shape in (
    ("goal", true_goal, (*batch, 700)),
    ("history", history, (*batch, 5, 93)),
    ("context", sim_ctx, (*batch, sim_ctx.shape[-1])),
  ):
    if (
      value.shape != shape
      or value.dtype != truth.position.dtype
      or value.device != truth.position.device
      or not torch.isfinite(value).all()
    ):
      raise ValueError(f"Invalid critic {name} shape, dtype, device or values")
  if cfg.group != "A":
    if not isinstance(belief, BeliefState):
      raise ValueError("B/C critic requires a complete BeliefState")
    belief.validate_structure()
    mean = belief.mean
    if (
      mean.position.shape != truth.position.shape
      or mean.position.dtype != truth.position.dtype
      or mean.position.device != truth.position.device
    ):
      raise ValueError("Critic belief must match truth batch, dtype and device")
    check_covariance(
      belief.covariance_rpv,
      "Critic covariance",
      rtol=128 * torch.finfo(history.dtype).eps,
    )
  features = build_critic_features_kernel(
    truth,
    true_goal,
    history,
    sim_ctx,
    belief,
    group=cfg.group,
    constants=critic_feature_constants(history, cfg) if cfg.group != "A" else None,
  )
  if not torch.isfinite(features).all():
    raise ValueError("Critic encoding produced non-finite features")
  return features


def build_critic_features_kernel(
  truth, true_goal, history, sim_ctx, belief, *, group, constants
):
  """Trusted tensor encoding; belief marginals come from covariance factors."""
  pieces = [critic_state_features(truth), true_goal, history.flatten(-2), sim_ctx]
  if group != "A":
    mean = belief.mean
    inverse = truth.rotation.mT
    rotation = inverse @ mean.rotation
    mean_features = torch.cat(
      (
        rotation[..., :2].flatten(-2),
        (inverse @ (mean.position - truth.position).unsqueeze(-1)).squeeze(-1),
        (inverse @ (mean.velocity - truth.velocity).unsqueeze(-1)).squeeze(-1),
      ),
      -1,
    )
    pieces.extend(
      (
        mean_features / constants[-1],
        encode_rpv_covariance_kernel(belief.covariance_rpv, constants),
      )
    )
  return torch.cat(pieces, -1)


class BeliefCritic(nn.Module):
  """Group-specific value network; forward never updates normalization."""

  def __init__(
    self,
    group: str,
    context_dim: int,
    hidden_dims: tuple[int, ...] = (256, 128),
    *,
    config: CriticInputConfig | None = None,
  ):
    super().__init__()
    self.config = _config(config, group)
    self.group, self.context_dim = group, context_dim
    self.hidden_dims = tuple(hidden_dims)
    if not self.hidden_dims or any(
      type(n) is not int or n < 1 for n in self.hidden_dims
    ):
      raise ValueError("Critic hidden dimensions must be positive integers")
    self.input_dim = self.config.input_dim(context_dim)
    self.state_normalizer = _CudaGraphNormalization(68)
    self.goal_normalizer = _CudaGraphNormalization(700)
    self.history_normalizer = _CudaGraphNormalization(93)
    self.mlp = build_mlp(self.input_dim, self.hidden_dims, 1)
    nn.init.zeros_(self.mlp[-1].weight)
    nn.init.zeros_(self.mlp[-1].bias)

  def normalizers(self):
    return (self.state_normalizer, self.goal_normalizer, self.history_normalizer)

  def _split(self, features):
    if features.ndim < 2 or features.shape[-1] != self.input_dim:
      raise ValueError(f"Critic input must have shape (...,{self.input_dim})")
    if (
      features.dtype != self.mlp[0].weight.dtype
      or features.device != self.mlp[0].weight.device
    ):
      raise ValueError("Critic features must match network dtype and device")
    return (
      features[..., :68],
      features[..., 68:768],
      features[..., 768:1233].reshape(*features.shape[:-1], 5, 93),
    )

  def forward(self, features: torch.Tensor) -> torch.Tensor:
    state, goal, history = self._split(features)
    normalized = torch.cat(
      (
        self.state_normalizer(state),
        self.goal_normalizer(goal),
        self.history_normalizer(history).flatten(-2),
        features[..., 1233:],
      ),
      -1,
    )
    return self.mlp(normalized)

  @torch.no_grad()
  def update_normalization(self, features: torch.Tensor) -> None:
    """Update from fresh real training frames, counting only the newest sensor."""
    values = self._split(features)
    if features.numel() == 0 or not torch.isfinite(features).all():
      raise ValueError("Critic normalization requires nonempty finite features")
    values = (
      values[0].reshape(-1, 68),
      values[1].reshape(-1, 700),
      values[2][..., -1, :].reshape(-1, 93),
    )
    candidates = copy.deepcopy(self.normalizers())
    for module, value in zip(candidates, values, strict=True):
      module.train()
      module.update(value)
      if not all(torch.isfinite(buffer).all() for buffer in module.buffers()):
        raise ValueError("Critic normalization produced non-finite statistics")
    for module, candidate in zip(self.normalizers(), candidates, strict=True):
      module.load_state_dict(candidate.state_dict())
      module.eval()

  def contract(self) -> dict:
    return {
      "version": CRITIC_INPUT_VERSION,
      "config": asdict(self.config),
      "context_dim": self.context_dim,
      "hidden_dims": self.hidden_dims,
      "input_dim": self.input_dim,
    }

  def checkpoint(self) -> dict:
    return {"contract": self.contract(), "state_dict": copy.deepcopy(self.state_dict())}

  def validate_checkpoint(self, checkpoint: dict) -> None:
    contract = checkpoint.get("contract") if isinstance(checkpoint, dict) else None
    version = contract.get("version") if isinstance(contract, dict) else None
    if version != CRITIC_INPUT_VERSION:
      raise ValueError(
        f"Group {self.group} requires critic contract {CRITIC_INPUT_VERSION!r}, "
        f"got {version!r}"
      )
    if contract != self.contract():
      raise ValueError("Critic checkpoint contract mismatch")
    state = checkpoint.get("state_dict")
    validate_state_dict(self, state, "critic")
    for name in ("state_normalizer", "goal_normalizer", "history_normalizer"):
      if (
        state[f"{name}.count"] < 0
        or (state[f"{name}._var"] < 0).any()
        or (state[f"{name}._std"] < 0).any()
        or not torch.allclose(
          state[f"{name}._std"].square(), state[f"{name}._var"], rtol=1e-5, atol=1e-8
        )
      ):
        raise ValueError("Invalid critic checkpoint normalization statistics")

  def load_checkpoint(self, checkpoint: dict) -> None:
    self.validate_checkpoint(checkpoint)
    self.load_state_dict(checkpoint["state_dict"])

  @torch.no_grad()
  def copy_common_initialization_from(self, source: BeliefCritic) -> None:
    if (
      replace(self.config, group="A") != replace(source.config, group="A")
      or self.context_dim != source.context_dim
      or self.hidden_dims != source.hidden_dims
    ):
      raise ValueError("Critic common initialization contracts differ")
    for destination, origin in zip(
      self.normalizers(), source.normalizers(), strict=True
    ):
      destination.load_state_dict(origin.state_dict())
    columns = min(self.input_dim, source.input_dim)
    self.mlp[0].weight[:, :columns].copy_(source.mlp[0].weight[:, :columns])
    self.mlp[0].bias.copy_(source.mlp[0].bias)
    for destination, origin in zip(self.mlp[1:], source.mlp[1:], strict=True):
      destination.load_state_dict(origin.state_dict())
