from __future__ import annotations

import time
import warnings
from concurrent.futures import Future, ThreadPoolExecutor
from copy import copy, deepcopy
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

import torch
import torch.nn as nn
import tyro
import wandb
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from supermimic.belief import BeliefState, GaussianStatePrior
from supermimic.belief_provider import BeliefProvider
from supermimic.envs.motion_tracking_env import MotionTrackingEnv
from supermimic.envs.multi_task_world_model_env import MultiTaskWorldModelEnv
from supermimic.estimator import FilterConfig, LieBeliefFilter
from supermimic.estimator_inputs import (
  EITInputConfig,
  GaussianMeasurementConfig,
  PosteriorInput,
  PosteriorProvider,
  build_actor_features,
)
from supermimic.learners.belief_wm_learner import (
  BeliefWorldModelLearner,
  BeliefWorldModelLearnerCfg,
)
from supermimic.learners.policy_learner import (
  ActorCriticPolicyLearner,
  ActorCriticPolicyLearnerCfg,
  BpttPolicyLearner,
  BpttPolicyLearnerCfg,
)
from supermimic.learners.wm_learner import WorldModelLearner, WorldModelLearnerCfg
from supermimic.models import Dynamics, MultiEncoderAE, ProcessNoiseConfig, build_mlp
from supermimic.replay_buffer import FieldSpec, ReplayBuffer
from supermimic.representations import StateRepresentation, create_representation
from supermimic.runners.exporter import export_actor_as_onnx
from supermimic.runners.reset_sampling import ReturnSampler, ReturnSamplerCfg
from supermimic.sim_context import SimContext
from supermimic.tasks.task_spec import (
  TasksCfg,
  build_actor_obs_td,
  make_estimator_motion_task,
  resolve_task_specs,
)
from supermimic.utils.helpers import (
  extract_action_scale_offset,
  unpack_obs_dims,
)
from supermimic.utils.lie_state import LieState
from supermimic.utils.logger import Logger
from supermimic.utils.validation import validate_state_dict
from supermimic.utils.wandb_artifacts import download_model_dir, log_model_artifact


@dataclass
class BaseRunnerCfg:
  """Shared configuration fields for all runners."""

  wandb_project: str = "supermimic"
  run_name: str | None = None
  log_root: str = "logs"
  """Root directory for run logs and checkpoints; relative to the working directory."""
  wandb_run_path: str | None = None
  headless: bool = True
  cuda_graphs: bool = True
  """Capture learner updates as CUDA graphs after warmup."""
  buffer_size: int = 1000
  representation: str = "gencoord"  # "gencoord" or "maxcoord"
  actor_obs: str = "rep_state"  # "rep_state" or "proprio"
  log_interval: int = 10
  save_interval: int = 1000
  history_length: int = 1
  termination_softness: float = 0.05
  dynamics_hidden_dims: tuple[int, ...] = (2048, 2048, 2048)
  wm: WorldModelLearnerCfg = field(default_factory=WorldModelLearnerCfg)


BpttCfg = Annotated[BpttPolicyLearnerCfg, tyro.conf.subcommand(name="bptt")]
ActorCriticCfg = Annotated[
  ActorCriticPolicyLearnerCfg, tyro.conf.subcommand(name="actor-critic")
]


@dataclass
class SuperMimicRunnerCfg(BaseRunnerCfg):
  estimator_inputs: EITInputConfig | None = None
  """Opt-in estimator inputs/FiLM; policy/critic optimization is a later stage."""
  process_noise: ProcessNoiseConfig | None = None
  """Opt-in full right-local stochastic dynamics and posterior-NLL learning."""
  filter_config: FilterConfig = field(default_factory=FilterConfig)
  initial_prior: GaussianStatePrior = field(default_factory=GaussianStatePrior)
  belief_wm: BeliefWorldModelLearnerCfg = field(
    default_factory=BeliefWorldModelLearnerCfg
  )
  freeze_estimator_normalization: bool = True
  """Keep the collection snapshot fixed; model training updates stats by round."""
  max_iterations: int = 1000000
  replay_ratio: int = 16
  latent_dim: int = 32
  actor_hidden_dims: tuple[int, ...] = (2048, 1024, 512, 256)
  critic_hidden_dims: tuple[int, ...] = (256, 128)
  latent_bc_weight: float = 1e-1
  """Weight on the latent BC loss pulling non-anchor encoders toward the
  anchor encoder's latent on the same state. 0 disables."""
  tasks: TasksCfg = field(default_factory=TasksCfg)
  """Per-task hyperparameters."""
  slot_mix: dict[str, float] = field(
    default_factory=lambda: {"motion": 1.0, "velocity": 0.0}
  )
  """Fractional slot allocation per task. Must sum to 1.0."""
  policy: BpttCfg | ActorCriticCfg = field(default_factory=ActorCriticCfg)

  sampling_mode: Literal["uniform", "adaptive", "start", "return"] = "return"
  """RSI reset distribution: 'return' (concentrate resets on the bins where the
  policy earns low return; see ReturnSampler) or one of mjlab's native modes.
  train.py resolves this into the motion command's cfg before the env is built
  ('return' drives mjlab's 'external' mode)."""
  return_sampler: ReturnSamplerCfg = field(default_factory=ReturnSamplerCfg)

  @classmethod
  def for_estimator_inputs(cls, group: Literal["A", "B", "C"] = "A", **kwargs):
    inputs = kwargs.pop("estimator_inputs", EITInputConfig(group=group))
    return cls(
      estimator_inputs=inputs,
      actor_obs="proprio",
      history_length=inputs.history_length,
      cuda_graphs=False,
      **kwargs,
    )

  def validate_estimator_inputs(self) -> None:
    inputs = self.estimator_inputs
    if inputs is None:
      if self.process_noise is not None:
        raise ValueError("Stochastic dynamics require the estimator input path")
      return
    if self.representation != "gencoord" or self.actor_obs != "proprio":
      raise ValueError(
        "Estimator input v1 requires G1 GenCoord and actor_obs='proprio'"
      )
    if self.history_length != inputs.history_length:
      raise ValueError("runner and estimator_inputs history_length must agree")
    if self.slot_mix.get("motion") != 1.0 or any(
      value != 0.0 for name, value in self.slot_mix.items() if name != "motion"
    ):
      raise ValueError("Estimator input v1 supports only the motion task")
    if self.cuda_graphs:
      raise ValueError("Estimator input v1 must be validated in eager mode")


@dataclass
class WorldModel:
  """The world model plus the env dimensions that sized it."""

  representation: StateRepresentation
  dynamics: Dynamics
  sim_context: SimContext
  num_goals: int
  num_actions: int
  filter_config: FilterConfig = field(default_factory=FilterConfig)
  initial_prior: GaussianStatePrior = field(default_factory=GaussianStatePrior)

  def migrate_bundle(self, bundle: dict) -> dict:
    """Migrate older integrator weights and preserve their filter settings."""
    if self.dynamics.process_noise_config is None:
      raise ValueError("Bundle migration requires stochastic dynamics")
    migrated = deepcopy(bundle)
    metadata = migrated.get("estimator", {})
    if metadata.get("version") != 1:
      raise ValueError("Expected a versioned estimator bundle for migration")
    if migrated["dynamics"]["metadata"]["format_version"] == "stochastic_dynamics_v1":
      old_filter = metadata.get("filter_config", {})
      added = {"jacobian_backend", "jacobian_chunk_size", "jacobian_batch_chunk_size"}
      if set(old_filter) != {f.name for f in fields(FilterConfig)} - added:
        raise ValueError("Expected a complete step-2 filter configuration")
      metadata["filter_config"] = {
        **old_filter,
        **{key: getattr(self.filter_config, key) for key in added},
      }
    migrated["dynamics"] = self.dynamics.migrate_checkpoint(migrated["dynamics"])
    # Validate without modifying this model or any active provider.
    deepcopy(self).load_bundle(migrated)
    return migrated

  def bundle(self) -> dict:
    """The ``world_model.pt`` contents."""
    bundle = {
      "dynamics": (
        self.dynamics.checkpoint()
        if self.dynamics.process_noise_config is not None
        else self.dynamics.state_dict()
      ),
      "sim_context": self.sim_context.state_dict(),
    }
    if self.dynamics.process_noise_config is not None:
      measurement = getattr(self.representation, "measurement_config", None)
      bundle["estimator"] = {
        "version": 1,
        "filter_config": asdict(self.filter_config),
        "initial_prior": asdict(self.initial_prior),
        "measurement_config": asdict(measurement) if measurement is not None else None,
        "default_joint_pos": self.representation._default_joint_pos.detach().clone(),
      }
    return bundle

  def load_bundle(self, bundle: dict) -> None:
    """Validate all components before committing a ``world_model.pt`` bundle."""
    metadata = None
    if self.dynamics.process_noise_config is not None:
      metadata = bundle.get("estimator")
      if metadata is None:
        warnings.warn(
          "Legacy stochastic bundle lacks estimator settings; using the supplied configuration",
          stacklevel=2,
        )
      else:

        def decode(cls, name):
          data = metadata.get(name)
          if not isinstance(data, dict) or data.keys() != {f.name for f in fields(cls)}:
            raise ValueError(f"Incomplete estimator checkpoint {name}")
          return cls(**data)

        if metadata.get("version") != 1 or not torch.equal(
          metadata["default_joint_pos"], self.representation._default_joint_pos
        ):
          raise ValueError("Estimator checkpoint layout or joint calibration mismatch")
        filter_config = decode(FilterConfig, "filter_config")
        prior = decode(GaussianStatePrior, "initial_prior")
        measurement = (
          decode(GaussianMeasurementConfig, "measurement_config")
          if metadata["measurement_config"] is not None
          else None
        )
        if (
          measurement is None
          and getattr(self.representation, "measurement_config", None) is not None
          or measurement is not None
          and self.representation.actor_obs_mode != "proprio"
        ):
          raise ValueError("Estimator checkpoint measurement configuration mismatch")
        if measurement is not None:
          # Configuration validation must not change the live representation.
          copy(self.representation).configure_estimator_measurement(measurement)
      self.dynamics.validate_checkpoint(bundle["dynamics"])
      dynamics_state = bundle["dynamics"]["state_dict"]
    else:
      dynamics_state = bundle["dynamics"]
      validate_state_dict(self.dynamics, dynamics_state, "Dynamics")
    context_state = bundle["sim_context"]
    validate_state_dict(self.sim_context, context_state, "Simulation context")
    if isinstance(self.sim_context, SimContext) and (context_state["scale"] <= 0).any():
      raise ValueError("Simulation context checkpoint scale must be positive")

    self.dynamics.load_state_dict(dynamics_state)
    self.sim_context.load_state_dict(context_state)
    if metadata is not None:
      self.filter_config, self.initial_prior = filter_config, prior
      if measurement is not None:
        self.representation.measurement_config = measurement


def build_world_model(cfg: BaseRunnerCfg, env, device: str) -> WorldModel:
  """Construct the representation and dynamics for ``env`` sized by ``cfg``.

  Resets ``env`` to read its observation dimensions.
  """
  inputs = getattr(cfg, "estimator_inputs", None)
  if isinstance(cfg, SuperMimicRunnerCfg):
    cfg.validate_estimator_inputs()
  if inputs is not None:
    actor_cfg = env.unwrapped.cfg.observations["actor"]
    motion_cfg = env.unwrapped.cfg.commands["motion"]
    if (
      actor_cfg.enable_corruption
      or actor_cfg.terms["joint_pos"].params.get("biased", False)
      or not getattr(motion_cfg, "fixed_reference_alignment", False)
      or not getattr(env.unwrapped.cfg, "forward_before_rl_signals", False)
    ):
      raise ValueError(
        "Call configure_estimator_env_cfg before creating the simulation: "
        "EIT requires single corruption, calibrated proprio, fixed references "
        "and end-of-step kinematics"
      )
  if cfg.history_length > 1 and cfg.actor_obs != "proprio":
    raise RuntimeError(
      "history_length > 1 requires actor_obs='proprio' (--runner.actor-obs proprio)"
    )

  obs, _ = env.reset()
  num_full_state, num_goals = unpack_obs_dims(obs)
  num_actions = env.num_actions

  print(f"[INFO] State Dim: {num_full_state}, Goal Dim: {num_goals}")

  sim_context = SimContext.from_env(env.unwrapped)
  sim_ctx_dim = sim_context.output_dim
  print(
    f"[INFO] Sim Context: {sim_ctx_dim} dim ({[s.name for s in sim_context.specs]})"
  )

  default_joint_pos = env.unwrapped.scene["robot"].data.default_joint_pos[0].to(device)
  representation = create_representation(
    cfg.representation,
    num_full_state,
    num_actions,
    env.unwrapped.step_dt,
    env.unwrapped.cfg.decimation,
    default_joint_pos,
    actor_obs_mode=cfg.actor_obs,
  ).to(device)

  representation.bind(env.unwrapped, sim_context)
  if inputs is not None:
    if num_actions != 29 or abs(env.unwrapped.step_dt - 0.02) > 1e-8:
      raise ValueError("Estimator input v1 requires 29 actions at 50 Hz")
    representation.configure_estimator_measurement(inputs.measurement_noise)
    num_goals = 700

  print(
    f"[INFO] Representation: {cfg.representation} "
    f"(full={representation.full_state_dim}, rep={representation.rep_state_dim}, "
    f"pred={representation.prediction_dim})"
  )

  action_scale, action_offset = extract_action_scale_offset(env)
  dynamics = Dynamics(
    representation,
    num_actions,
    sim_ctx_dim,
    action_scale,
    action_offset,
    hidden_dims=cfg.dynamics_hidden_dims,
    process_noise=getattr(cfg, "process_noise", None),
  ).to(device)

  return WorldModel(
    representation=representation,
    dynamics=dynamics,
    sim_context=sim_context,
    num_goals=num_goals,
    num_actions=num_actions,
    filter_config=getattr(cfg, "filter_config", FilterConfig()),
    initial_prior=getattr(cfg, "initial_prior", GaussianStatePrior()),
  )


def build_actor(
  cfg: SuperMimicRunnerCfg, wm: WorldModel, device: str
) -> MultiEncoderAE:
  """Actor sized by ``cfg``; the resolved task specs live on ``actor.task_specs``."""
  representation = wm.representation
  inputs = cfg.estimator_inputs
  cfg.validate_estimator_inputs()
  if inputs is not None and (
    wm.num_actions != 29 or representation.proprio_obs_dim(wm.num_actions) != 93
  ):
    raise ValueError("Estimator input v1 requires the G1 93-dimensional sensor layout")
  obs_dim = (
    representation.rep_state_dim
    if cfg.actor_obs == "rep_state"
    else representation.proprio_obs_dim(wm.num_actions)
  )
  return MultiEncoderAE(
    obs_dim=obs_dim,
    action_dim=wm.num_actions,
    task_specs=(
      [make_estimator_motion_task()]
      if inputs is not None
      else resolve_task_specs(cfg.tasks, representation, wm.num_goals, device)
    ),
    slot_mix={"motion": 1.0} if inputs is not None else cfg.slot_mix,
    latent_dim=cfg.latent_dim,
    hidden_dims=cfg.actor_hidden_dims,
    latent_bc_weight=cfg.latent_bc_weight,
    history_length=cfg.history_length,
    uncertainty_dim=450 if inputs is not None and inputs.uses_uncertainty else 0,
    input_version=inputs.version if inputs is not None else "legacy",
  ).to(device)


def actor_checkpoint(actor: MultiEncoderAE, cfg: SuperMimicRunnerCfg) -> dict:
  """Store feature semantics as well as weights; legacy checkpoints stay unchanged."""
  if cfg.estimator_inputs is None:
    return actor.state_dict()
  return {
    "input_config": cfg.estimator_inputs.to_dict(),
    "actor": actor.state_dict(),
    "architecture": {
      "hidden_dims": list(cfg.actor_hidden_dims),
      "latent_dim": cfg.latent_dim,
      "film_hidden_dims": [128, 64],
    },
  }


def load_actor_checkpoint(
  actor: MultiEncoderAE,
  cfg: SuperMimicRunnerCfg,
  checkpoint: dict,
  *,
  for_evaluation: bool = False,
) -> None:
  """Reject cross-version or cross-group loads before touching actor parameters."""
  inputs = cfg.estimator_inputs
  if inputs is None:
    if "input_config" in checkpoint:
      raise ValueError("Estimator input checkpoint cannot be loaded as a legacy actor")
    validate_state_dict(actor, checkpoint, "Actor")
    actor.load_state_dict(checkpoint)
    return
  if "input_config" not in checkpoint:
    raise ValueError("Estimator input actor requires a versioned checkpoint")
  saved = EITInputConfig.from_dict(checkpoint["input_config"])
  if for_evaluation:
    if inputs.mode != "eval":
      raise ValueError("for_evaluation requires mode='eval'")
    saved = replace(saved, mode="eval")
  if saved != inputs:
    raise ValueError("Actor input configuration mismatch (including group/mode/noise)")
  expected = {
    "hidden_dims": list(cfg.actor_hidden_dims),
    "latent_dim": cfg.latent_dim,
    "film_hidden_dims": [128, 64],
  }
  if checkpoint.get("architecture") != expected:
    raise ValueError("Actor architecture mismatch")
  validate_state_dict(actor, checkpoint["actor"], "Actor")
  actor.load_state_dict(checkpoint["actor"])


def build_ablation_actors(
  cfg: SuperMimicRunnerCfg, wm: WorldModel, device: str
) -> dict[str, MultiEncoderAE]:
  """Independent A/B/C actors with copied common weights and normalization."""
  if cfg.estimator_inputs is None:
    raise ValueError("build_ablation_actors requires estimator_inputs")
  actors = {}
  for group in ("A", "B", "C"):
    group_cfg = replace(
      cfg, estimator_inputs=replace(cfg.estimator_inputs, group=group)
    )
    actors[group] = build_actor(group_cfg, wm, device)
    if group != "A":
      actors[group].copy_common_initialization_from(actors["A"])
  return actors


@dataclass
class InputStagePolicy:
  """Collection-only actor holder, deliberately not a partial training baseline."""

  actor: MultiEncoderAE
  cfg: BpttPolicyLearnerCfg | ActorCriticPolicyLearnerCfg

  def update(self):
    raise NotImplementedError(
      "Complete estimator and belief critic training is a later stage"
    )


def load_wandb_checkpoint(wandb_run_path: str, wm: WorldModel, device: str) -> None:
  """Warm-start ``wm`` from another W&B run's world-model bundle."""
  print(f"[INFO] Loading World Model from W&B: {wandb_run_path}")
  run = wandb.Api().run(wandb_run_path)
  wm.load_bundle(
    torch.load(download_model_dir(run) / "world_model.pt", map_location=device)
  )


class Ratio:
  def __init__(self, ratio: float) -> None:
    self._ratio = ratio
    self._prev: int | None = None

  def __call__(self, step: int) -> int:
    step = int(step)
    if self._ratio == 0:
      return 0
    if self._ratio < 0:
      return 1
    if self._prev is None:
      self._prev = step
      return 1
    repeats = int((step - self._prev) * self._ratio)
    self._prev += repeats / self._ratio
    return repeats


@dataclass
class _CollectionStep:
  fields: dict[str, torch.Tensor]
  actor_td: TensorDict | None
  head_priors: tuple[torch.Tensor, BeliefState] | None
  result: tuple
  token: object = field(default_factory=object)
  next_sensor: torch.Tensor | None = None
  stored: bool = False
  history_advanced: bool = False
  logged: bool = False
  sampler_updated: bool = False
  sampling_weights: torch.Tensor | None = None


class BaseRunner[CfgT: BaseRunnerCfg]:
  """Base class with shared initialization for all runners.

  Handles: env dimension extraction, representation creation, world model
  setup, replay buffer, action manager, and logger. Subclasses implement
  ``_init_policy()`` for their specific policy setup.
  """

  cfg: CfgT

  def __init__(
    self,
    env,
    cfg: CfgT,
    task_id: str,
    registry_name: str | None = None,
    device: str = "cuda:0",
  ):
    self.env = env
    self.cfg = cfg
    self.task_id = task_id
    self.registry_name = registry_name
    self.device = device

    self._init_common()
    self._init_policy()

    self.logger = Logger(
      run_name=cfg.run_name,
      num_envs=env.num_envs,
      max_episode_length=env.max_episode_length,
      device=device,
    )

  def _init_common(self) -> None:
    wm = build_world_model(self.cfg, self.env, self.device)
    self._wm = wm
    self._representation = wm.representation
    self._dynamics = wm.dynamics
    self._sim_context = wm.sim_context
    self._num_goals = wm.num_goals
    self._num_actions = wm.num_actions

    if self.cfg.wandb_run_path is not None:
      load_wandb_checkpoint(self.cfg.wandb_run_path, wm, self.device)
      if getattr(self.cfg, "process_noise", None) is not None:
        self.cfg.filter_config, self.cfg.initial_prior = (
          wm.filter_config,
          wm.initial_prior,
        )
        self.cfg.estimator_inputs = replace(
          self.cfg.estimator_inputs,
          measurement_noise=wm.representation.measurement_config,
        )

    self.action_manager = self.env.unwrapped.action_manager
    self.motion_cmd = self.env.unwrapped.command_manager.get_term("motion")
    self.motion = self.motion_cmd.motion

    self.replay_buffer = ReplayBuffer(
      schema=self._buffer_schema(),
      buffer_size=self.cfg.buffer_size,
      num_envs=self.env.num_envs,
      device=self.device,
      initial_prior=wm.initial_prior,
    )
    if self._dynamics.process_noise_config is not None:
      self.world_model_learner = BeliefWorldModelLearner(
        LieBeliefFilter(
          self._dynamics,
          self._representation.measurement_config,
          self._representation._default_joint_pos,
          config=wm.filter_config,
        ),
        self.cfg.belief_wm,
      )
    else:
      self.world_model_learner = WorldModelLearner(
        self._dynamics,
        self._representation,
        self.cfg.wm,
        device=self.device,
        cuda_graphs=self.cfg.cuda_graphs,
      )

    self._pending_norm_updates: list[tuple] = []
    self._sim_history: torch.Tensor | None = None

  def _seed_sim_history(self) -> None:
    """Seed the real-sim obs-history ring with the noised current sensor."""
    if (
      self.cfg.history_length == 1
      and getattr(self.cfg, "estimator_inputs", None) is None
    ):
      return
    seed = self._representation.add_proprio_noise(
      self.env.get_observations()["actor"].to(self.device)
    )
    self._sim_history = seed.unsqueeze(1).repeat(1, self.cfg.history_length, 1)
    if getattr(self.cfg, "estimator_inputs", None) is not None:
      self._measurement_reset = torch.ones(
        seed.shape[0], dtype=torch.bool, device=seed.device
      )
      slots = torch.arange(seed.shape[0], device=seed.device)
      previous = getattr(self, "_episode_id", None)
      self._episode_id = slots if previous is None else previous + seed.shape[0]
      self._time_index = torch.zeros_like(slots)
      self._posterior_ready = False

  def _collection_history(self, sensor: torch.Tensor) -> torch.Tensor:
    """The ``(B, history_length, sensor_dim)`` window for a real-sim actor step.
    history_length > 1 needs ``_seed_sim_history()`` first."""
    if self.cfg.history_length == 1:
      return sensor.unsqueeze(1)
    assert self._sim_history is not None
    return self._sim_history

  def _advance_sim_history(
    self, next_actor: torch.Tensor, dones: torch.Tensor, *, next_sensor=None
  ) -> None:
    """Append the noised next sensor; rows whose episode just reset repeat that
    single realization, matching the warm-start padding."""
    if (
      self.cfg.history_length == 1
      and getattr(self.cfg, "estimator_inputs", None) is None
    ):
      return
    assert self._sim_history is not None
    if next_sensor is None:
      next_sensor = self._representation.add_proprio_noise(next_actor)
    shifted = torch.cat([self._sim_history[:, 1:], next_sensor.unsqueeze(1)], dim=1)
    reset = dones.bool().view(-1, 1, 1)
    history = torch.where(reset, next_sensor.unsqueeze(1), shifted)
    if getattr(self.cfg, "estimator_inputs", None) is not None:
      reset_rows = reset.flatten()
      episode_id = self._episode_id + len(dones) * reset_rows.long()
      time_index = torch.where(reset_rows, 0, self._time_index + 1)
      self._episode_id, self._time_index = episode_id, time_index
      self._measurement_reset = reset_rows.clone()
      self._posterior_ready = False
    self._sim_history = history

  def _policy_input_dims(self) -> tuple[int, int]:
    """Compute actor and critic input dimensions from the representation."""
    actor_dim = self._representation.actor_obs_dim(
      self._num_actions, self._num_goals, self.cfg.history_length
    )
    critic_dim = (
      self._representation.rep_state_dim
      + self._num_goals
      + self._sim_context.output_dim
    )
    return actor_dim, critic_dim

  def _build_mlp(
    self,
    name: str,
    input_dim: int,
    output_dim: int,
    hidden_dims: tuple[int, ...],
    **kwargs: object,
  ) -> nn.Module:
    """Build an MLPModel with input obs normalization."""
    obs = TensorDict(
      {name: torch.zeros(1, input_dim, device=self.device)},
      batch_size=[1],
      device=self.device,
    )
    return MLPModel(
      obs=obs,
      obs_groups={name: [name]},
      obs_set=name,
      output_dim=output_dim,
      hidden_dims=hidden_dims,
      obs_normalization=True,
      **kwargs,
    ).to(self.device)

  def _create_wm_env(self, batch_size: int) -> MotionTrackingEnv:
    """Create a :class:`MotionTrackingEnv` bound to this runner's world model and buffer."""
    return MotionTrackingEnv(
      dynamics=self._dynamics,
      representation=self._representation,
      env=self.env,
      batch_size=batch_size,
      replay_buffer=self.replay_buffer,
      history_length=self.cfg.history_length,
      termination_softness=self.cfg.termination_softness,
      device=self.device,
    )

  @property
  def _stores_actor_obs(self) -> bool:
    """The buffer carries the real sensor only for its sole reader, the
    history warm-start (``WorldModelEnv.sample_initial_states``)."""
    return self._representation.actor_obs_mode == "proprio" and (
      self.cfg.history_length > 1
      or getattr(self.cfg, "estimator_inputs", None) is not None
    )

  def _buffer_schema(self) -> dict[str, FieldSpec]:
    """Override to customize the replay buffer schema (e.g. extra fields)."""
    schema = {
      "state": FieldSpec(self._representation.full_state_dim),
      "action": FieldSpec(self._num_actions),
      "prev_action": FieldSpec(self._num_actions),
      "motion_time": FieldSpec(dtype=torch.long),
      "terminated": FieldSpec(dtype=torch.bool),
      "truncated": FieldSpec(dtype=torch.bool),
    }
    if self._stores_actor_obs:
      schema["actor_obs"] = FieldSpec(
        self._representation.proprio_obs_dim(self._num_actions)
      )
    schema["sim_ctx"] = FieldSpec(self._sim_context.output_dim)
    if getattr(self.cfg, "estimator_inputs", None) is not None:
      schema["alignment_rotation"] = FieldSpec((3, 3))
      schema["alignment_translation"] = FieldSpec(3)
      schema["episode_id"] = FieldSpec(dtype=torch.long)
      schema["time_index"] = FieldSpec(dtype=torch.long)
    return schema

  def _init_policy(self) -> None:
    """Subclasses implement this to create their policy learner."""
    raise NotImplementedError

  def _step_and_store(
    self,
    actions: torch.Tensor,
    curr_states: torch.Tensor,
    actor_td: TensorDict | None = None,
    extra_fields: dict[str, torch.Tensor] | None = None,
  ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict[str, object]]:
    """Step the env and write the transition to the replay buffer.

    When given, ``actor_td`` is queued for the actor-normalizer update.
    """
    step = self._execute_collection_step(actions, curr_states, actor_td, extra_fields)
    self._store_collection_step(step)
    return step.result

  def _execute_collection_step(
    self, actions, curr_states, actor_td, extra_fields
  ) -> _CollectionStep:
    motion_cmd = self.env.unwrapped.command_manager.get_term("motion")
    fields = {
      "state": curr_states.detach().clone(),
      "action": actions.detach().clone(),
      "prev_action": self.action_manager.action.clone(),
      "motion_time": motion_cmd.time_steps.clone(),
    }
    if extra_fields:
      fields.update({k: v.detach().clone() for k, v in extra_fields.items()})
    actor_td = actor_td.detach().clone() if actor_td is not None else None
    provider = getattr(self, "posterior_provider", None)
    head_priors = None
    if isinstance(provider, BeliefProvider) and "episode_id" in fields:
      state = provider.state
      if (
        state is None
        or not torch.equal(state.episode_id, fields["episode_id"])
        or not torch.equal(state.time_index, fields["time_index"])
      ):
        raise ValueError("Collection posterior must match the frames being recorded")
      head_priors = state.head_priors
      if head_priors is not None:
        self.replay_buffer.validate_head_priors(
          state.episode_id, state.time_index, head_priors
        )
    return _CollectionStep(fields, actor_td, head_priors, self.env.step(actions))

  def _store_collection_step(self, step: _CollectionStep):
    if step.stored:
      return
    _, _, dones, extras = step.result
    truncated = extras.get(
      "time_outs",
      torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device),
    )
    terminated = dones.bool() & ~truncated

    transition = TensorDict(
      dict(step.fields, terminated=terminated, truncated=truncated),
      batch_size=[self.env.num_envs],
    )
    self.replay_buffer.insert(
      transition, head_priors=step.head_priors, token=step.token
    )
    if step.actor_td is not None:
      self._pending_norm_updates.append(
        (step.fields["state"], step.fields["action"], step.actor_td)
      )
    step.stored = True

  def _flush_normalization(self) -> None:
    """Apply queued normalizer updates. Call outside ``inference_mode()``."""
    if not self._pending_norm_updates:
      return
    if (
      isinstance(getattr(self, "posterior_provider", None), BeliefProvider)
      and self.cfg.freeze_estimator_normalization
    ):
      self._pending_norm_updates.clear()
      return
    states, actions, actor_inputs = zip(*self._pending_norm_updates, strict=True)
    self.world_model_learner.update_normalization(torch.cat(states), torch.cat(actions))
    self._update_actor_normalization(actor_inputs)
    self._pending_norm_updates.clear()

  def _update_actor_normalization(self, actor_inputs: tuple[TensorDict, ...]) -> None:
    """Subclasses override to update the actor obs normalizer."""
    raise NotImplementedError

  def _create_log_dir(self, *extra_parts: str) -> Path:
    """Create a run directory under log_root, using a timestamp if no name is set."""
    name = self.cfg.run_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if Path(name).name != name or name in (".", ".."):
      raise ValueError("runner.run_name must be a single directory name")
    log_dir = (
      Path(self.cfg.log_root).expanduser().joinpath(self.task_id, *extra_parts, name)
    )
    try:
      log_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
      raise FileExistsError(
        f"Log directory already exists: {log_dir}. "
        "Choose a different runner.run_name to preserve existing checkpoints."
      ) from error
    print(f"[INFO] Logging to {log_dir}")
    return log_dir

  def _init_wandb(self, log_dir: Path, name_suffix: str = "") -> None:
    """Initialize the W&B run and link the registry artifacts."""
    wandb_config = {
      "env_cfg": asdict(self.env.cfg),
      "runner_cfg": asdict(self.cfg),
    }

    default_name = f"{self.task_id}{name_suffix}-{log_dir.name}"
    wandb.init(
      project=self.cfg.wandb_project,
      name=self.cfg.run_name or default_name,
      dir=str(log_dir),
      config=wandb_config,
    )

    if self.registry_name:
      api = wandb.Api()
      for name in self.registry_name.split(","):
        name = name.strip()
        if ":" not in name:
          name = name + ":latest"
        print(f"[INFO] Linking registry artifact: {name}")
        artifact = api.artifact(name)
        wandb.run.use_artifact(artifact)

  def _cleanup(self) -> None:
    """Close environment and finish W&B run."""
    env_close_fn = getattr(self.env, "close", None)
    if env_close_fn:
      env_close_fn()
    wandb.finish()


class SuperMimicRunner(BaseRunner[SuperMimicRunnerCfg]):
  """Trains WM + multi-encoder actor end-to-end via BPTT/AC.

  Slots are contiguous blocks of the env batch; each block routes through its
  task's encoder. Motion-tracking reward is the only reward path; tasks differ
  only in what goal modality their encoder consumes.
  """

  _pending_collection: _CollectionStep | None = None

  def __init__(
    self,
    *args,
    posterior_provider: PosteriorProvider | None = None,
    imagined_posterior_provider: PosteriorProvider | None = None,
    **kwargs,
  ):
    if (
      isinstance(posterior_provider, BeliefProvider)
      and posterior_provider is imagined_posterior_provider
    ):
      raise ValueError(
        "Collection and imagination require independent belief provider state"
      )
    self.posterior_provider = posterior_provider
    self.imagined_posterior_provider = imagined_posterior_provider
    self._current_posterior: PosteriorInput | None = None
    self._posterior_ready = False
    super().__init__(*args, **kwargs)

  def configure_belief_filter(
    self, prior_factory=None, config: FilterConfig | None = None
  ):
    """Attach independent real/imagined providers using this runner's model.

    Real resets default to a full noisy-truth prior. An explicit factory can
    override real initialization. Training segments use the configured Gaussian.
    """
    if self.cfg.estimator_inputs is None or self.cfg.process_noise is None:
      raise ValueError(
        "Enable estimator_inputs and process_noise before building runner"
      )
    if config is not None:
      self.cfg.filter_config = self._wm.filter_config = config
    if prior_factory is None:
      prior_factory = self._wm.initial_prior
    estimator = LieBeliefFilter(
      self._dynamics,
      self.cfg.estimator_inputs.measurement_noise,
      self._representation._default_joint_pos,
      config=self._wm.filter_config,
    )
    learner = getattr(self, "world_model_learner", None)
    if isinstance(learner, BeliefWorldModelLearner):
      learner.estimator = estimator
    self.posterior_provider = BeliefProvider(
      estimator, prior_factory, differentiable=False
    )
    self.imagined_posterior_provider = BeliefProvider(estimator, differentiable=True)
    self.wm_env.posterior_provider = self.imagined_posterior_provider
    self._current_posterior = None  # Discard any previous legacy injection.
    self._posterior_ready = False
    return self.posterior_provider, self.imagined_posterior_provider

  def _init_policy(self) -> None:
    actor = build_actor(self.cfg, self._wm, self.device)
    self._active_task_specs = [
      ts for ts in actor.task_specs if self.cfg.slot_mix.get(ts.name, 0.0) > 0.0
    ]

    print("[INFO] Sim slot ranges:")
    for ts, (lo, hi) in zip(
      actor.task_specs, actor.slot_ranges(self.env.num_envs), strict=True
    ):
      print(f"  {ts.name}: [{lo}, {hi}) — goal_dim={ts.goal_dim}")
    policy_cfg = self.cfg.policy

    self.wm_env = MultiTaskWorldModelEnv(
      dynamics=self._dynamics,
      representation=self._representation,
      env=self.env,
      batch_size=policy_cfg.batch_size,
      replay_buffer=self.replay_buffer,
      task_specs=self._active_task_specs,
      history_length=self.cfg.history_length,
      termination_softness=self.cfg.termination_softness,
      device=self.device,
      estimator_input_config=self.cfg.estimator_inputs,
      posterior_provider=self.imagined_posterior_provider,
    )
    if torch.cuda.is_available() and self.cfg.estimator_inputs is None:
      self.wm_env.step = torch.compile(self.wm_env.step)
      self.wm_env.reset = torch.compile(self.wm_env.reset)

    self._configure_reset_sampling()

    if self.cfg.estimator_inputs is not None:
      # No placeholder belief critic is allowed to become a training baseline.
      # Collection and explicit short-sequence autograd checks remain available.
      self.policy_learner = InputStagePolicy(actor=actor, cfg=policy_cfg)
      return

    _, critic_input_dim = self._policy_input_dims()
    critic = build_mlp(
      critic_input_dim,
      self.cfg.critic_hidden_dims,
      output_dim=1,
    ).to(self.device)

    if isinstance(policy_cfg, ActorCriticPolicyLearnerCfg):
      self.policy_learner = ActorCriticPolicyLearner(
        actor,
        critic,
        self.wm_env,
        policy_cfg,
        device=self.device,
        cuda_graphs=self.cfg.cuda_graphs,
      )
    else:
      self.policy_learner = BpttPolicyLearner(
        actor,
        self.wm_env,
        policy_cfg,
        device=self.device,
        cuda_graphs=self.cfg.cuda_graphs,
      )

  @torch.inference_mode()
  def reset_collection(self) -> None:
    """Start real collection with one shared initial measurement per environment.

    Input-stage users call this once, then step(posterior=...) or use the
    constructor's collection posterior provider. Padding is not a filter update.
    Match step()'s inference context for simulator caches created during rollout.
    """
    if self._pending_collection is not None:
      self.step()
    self.env.reset()
    self.replay_buffer.truncate_open_episodes()
    if self._sampler is not None:
      self._sampler.truncate_open_episodes()
    self._seed_sim_history()
    self._current_posterior = None
    self._posterior_ready = False

  def _build_sim_obs_td(
    self,
    rep_state: torch.Tensor,
    aux_state: torch.Tensor,
    sim_actor_obs: torch.Tensor,
    motion_times: torch.Tensor,
    sim_ctx: torch.Tensor,
    *,
    posterior: PosteriorInput | None = None,
  ) -> TensorDict:
    inputs = self.cfg.estimator_inputs
    if inputs is not None:
      if self._sim_history is None:
        raise RuntimeError("Seed the measurement history after reset before collection")
      sensor = self._sim_history[:, -1]
      candidate = None
      if isinstance(self.posterior_provider, BeliefProvider):
        provider = self.posterior_provider
        estimator = provider.estimator
        if (
          estimator.dynamics is not self._dynamics
          or estimator.measurement_config != inputs.measurement_noise
          or not torch.equal(
            estimator.default_joint_pos, self._representation._default_joint_pos
          )
          or provider.differentiable
        ):
          raise ValueError(
            "Collection provider must use this model/measurement and detached state"
          )
        if posterior is not None:
          raise ValueError("A live belief provider requires a keyed posterior")
        current = provider.lookup(self._episode_id, self._time_index, sim_ctx)
        if current is None:
          reset = self._measurement_reset
          candidate = provider.prepare(
            measurement=sensor[..., :64],
            previous_action=sensor[..., 64:],
            reset_mask=self._measurement_reset,
            context=sim_ctx,
            episode_id=self._episode_id,
            time_index=self._time_index,
            reset_truth=(
              LieState.from_rep(aux_state[reset], rep_state[reset])
              if isinstance(provider.prior_factory, GaussianStatePrior) and reset.any()
              else None
            ),
          )
          current = candidate.state.belief
        posterior = current.to_actor_input()
      else:
        # Only legacy injection lacks a keyed provider; consume it once per frame.
        if not self._posterior_ready:
          if posterior is None and self.posterior_provider is not None:
            posterior = self.posterior_provider(
              measurement=sensor[..., :64],
              previous_action=sensor[..., 64:],
              reset_mask=self._measurement_reset,
            )
          self._current_posterior = posterior
        elif posterior is not None:
          raise ValueError("A posterior was already supplied for this measurement")
        posterior = self._current_posterior
      alignment = self.motion_cmd.reference_alignment
      reference = self._representation.reference_window(
        self.motion, motion_times, alignment
      )
      goal, uncertainty = build_actor_features(
        inputs,
        self._representation.true_rpv(aux_state, rep_state),
        posterior,
        reference,
      )
      actor_td = build_actor_obs_td(
        {"motion": goal}, self._sim_history, task_uncertainty=uncertainty
      )
      if candidate is not None:
        self.posterior_provider.commit(candidate)
      self._posterior_ready = True
      return actor_td
    proprio = self.cfg.actor_obs == "proprio"
    sensor = self._representation.compute_actor_sensor(
      rep_state,
      sim_ctx,
      aux_state=aux_state,
      sim_actor_obs=sim_actor_obs if proprio else None,
      noisy=proprio,
    )
    goals = {
      ts.name: ts.extract_training_goal(self.motion, motion_times, aux_state)
      for ts in self._active_task_specs
    }
    return build_actor_obs_td(goals, self._collection_history(sensor))

  def _configure_reset_sampling(self) -> None:
    self._sampler: ReturnSampler | None = None
    env_mode = self.motion_cmd.cfg.sampling_mode
    expected = (
      "external" if self.cfg.sampling_mode == "return" else self.cfg.sampling_mode
    )
    if env_mode != expected:
      raise RuntimeError(
        f"runner sampling_mode='{self.cfg.sampling_mode}' needs the motion command "
        f"in '{expected}' mode but the env was built with '{env_mode}'; resolve it "
        "into the env cfg before env creation ('external' also needs an mjlab with "
        "MotionCommand external-mode support)"
      )
    if self.cfg.sampling_mode != "return":
      return
    self._sampler = ReturnSampler(
      cfg=self.cfg.return_sampler,
      bin_count=self.motion_cmd.bin_count,
      frame_count=self.motion.time_step_total,
      num_envs=self.env.num_envs,
      max_step_reward=self.wm_env.max_step_reward,
      device=self.device,
    )

  def finish_collection_round(self) -> None:
    """Seal training segment heads after collection, preserving the live stream."""
    if self._pending_collection is not None:
      self.step()
    self.replay_buffer.finish_collection_round()

  def _collect_action(self, posterior) -> _CollectionStep:
    obs = self.env.get_observations()
    curr_states = obs["full_state"].to(self.device)
    sim_actor_obs = obs["actor"].to(self.device)

    # Pre-step frames; env.step() advances time_steps in place.
    motion_times = self.motion_cmd.time_steps.to(self.device).clone()

    rep_state = self._representation.represent(curr_states)
    aux_state = self._representation.extract_aux_state(curr_states)
    sim_ctx = self._sim_context.read(self.env.unwrapped)
    actor_td = self._build_sim_obs_td(
      rep_state,
      aux_state,
      sim_actor_obs,
      motion_times,
      sim_ctx,
      posterior=posterior,
    )

    actions = self.policy_learner.actor(actor_td)
    actions = actions + torch.randn_like(actions) * self.policy_learner.cfg.noise
    if self.cfg.estimator_inputs is not None:
      actions = self.wm_env.limit_actions(actions)

    extra_fields = {"sim_ctx": sim_ctx}
    if self._stores_actor_obs:
      extra_fields["actor_obs"] = (
        actor_td["history"][:, -1]
        if self.cfg.estimator_inputs is not None
        else sim_actor_obs
      )
    if self.cfg.estimator_inputs is not None:
      if not torch.equal(actor_td["history"][:, -1, 64:], self.action_manager.action):
        raise RuntimeError("Measured previous action differs from executed action")
      alignment = self.motion_cmd.reference_alignment
      extra_fields["alignment_rotation"] = alignment.rotation
      extra_fields["alignment_translation"] = alignment.translation

      extra_fields.update(episode_id=self._episode_id, time_index=self._time_index)
    return self._execute_collection_step(
      actions,
      curr_states,
      actor_td=actor_td,
      extra_fields=extra_fields,
    )

  def step(
    self, *, posterior: PosteriorInput | None = None, record_episode_stats: bool = True
  ) -> int:
    with torch.inference_mode():
      step = self._pending_collection
      if step is None:
        step = self._pending_collection = self._collect_action(posterior)
      next_obs, rewards, dones, extras = step.result
      if self.cfg.estimator_inputs is not None and step.next_sensor is None:
        step.next_sensor = self._representation.add_proprio_noise(
          next_obs["actor"].to(self.device)
        )
      self._store_collection_step(step)
      if not step.history_advanced:
        self._advance_sim_history(
          next_obs["actor"].to(self.device), dones, next_sensor=step.next_sensor
        )
        step.history_advanced = True
      if not step.logged:
        if record_episode_stats:
          self.logger.process_env_step(
            rewards, self.env.unwrapped.termination_manager.terminated, extras
          )
        step.logged = True

      if self._sampler is not None and not step.sampler_updated:
        # A failure on a step that also times out counts as the time-out.
        manager = self.env.unwrapped.termination_manager
        truncated = manager.time_outs.to(self.device, torch.bool)
        clip_end = manager.get_term("motion_end").to(self.device, torch.bool)
        step.sampling_weights = self._sampler.record(
          step.fields["motion_time"],
          rewards,
          manager.terminated.to(self.device, torch.bool) & ~truncated,
          clip_end,
          truncated & ~clip_end,
        )
        step.sampler_updated = True
      if step.sampling_weights is not None:
        self.motion_cmd.set_sampling_distribution(step.sampling_weights)
        step.sampling_weights = None

    self._flush_normalization()
    self._pending_collection = None
    return self.env.num_envs

  def _update_actor_normalization(self, actor_inputs: tuple[TensorDict, ...]) -> None:
    self.policy_learner.actor.train()
    for actor_td in actor_inputs:
      self.policy_learner.actor.update_normalization(actor_td)

  def run(self) -> None:
    if self.cfg.estimator_inputs is not None:
      raise NotImplementedError(
        "Full estimator policy/critic training is a later stage. Use "
        "python -m supermimic.scripts.train_belief for posterior-NLL model learning."
      )
    log_dir = self._create_log_dir()
    self._init_wandb(log_dir)

    self.env.reset()
    self._seed_sim_history()

    print(f"[INFO] Starting training loop for {self.cfg.max_iterations} iterations...")

    should_train = Ratio(
      self.env.num_envs
      / self.cfg.wm.num_batches
      / self.cfg.wm.batch_size
      / self.cfg.wm.horizon
      * self.cfg.replay_ratio
    )

    num_collected_steps = 0
    start_time = time.time()
    wm_learn_time = 0.0
    policy_learn_time = 0.0
    loss_dict: dict[str, torch.Tensor] = {}
    log_count = 0
    uploader = ThreadPoolExecutor(max_workers=1)
    upload: Future | None = None
    first_update = True

    for step in range(self.cfg.max_iterations):
      num_collected_steps += self.step()

      for _ in range(should_train(step)):
        if self.replay_buffer.num_transitions < self.cfg.wm.horizon:
          continue

        data = (
          self.replay_buffer.sample(self.cfg.wm.horizon, self.cfg.wm.batch_size)
          for _ in range(self.cfg.wm.num_batches)
        )

        start_wm_time = time.time()
        if first_update:
          print(
            f"[INFO] First world-model update at simulation step {step}; "
            "initialization/compilation may take several minutes.",
            flush=True,
          )
        wm_logs = self.world_model_learner.update(data)
        wm_learn_time += time.time() - start_wm_time

        start_policy_time = time.time()
        if first_update:
          print(
            "[INFO] First world-model update returned; starting first policy update.",
            flush=True,
          )
        policy_logs = self.policy_learner.update()
        policy_learn_time += time.time() - start_policy_time
        if first_update:
          print("[INFO] First policy update returned.", flush=True)
          first_update = False

        loss_dict.update(wm_logs)
        loss_dict.update(policy_logs)

      if step % self.cfg.log_interval == 0:
        elapsed_time = time.time() - start_time
        total_train_time = wm_learn_time + policy_learn_time
        collect_time = elapsed_time - total_train_time

        self.logger.log(
          it=log_count,
          total_it=self.cfg.max_iterations // self.cfg.log_interval,
          collect_time=collect_time,
          wm_learn_time=wm_learn_time,
          policy_learn_time=policy_learn_time,
          loss_dict=loss_dict,
          num_collected_steps=num_collected_steps,
        )

        log_count += 1
        start_time = time.time()
        wm_learn_time = 0
        policy_learn_time = 0
        num_collected_steps = 0

      if step % self.cfg.save_interval == 0:
        assert wandb.run is not None  # _init_wandb ran before the loop
        # The previous version must be logged before its files are overwritten.
        if upload is not None:
          upload.result()
        world_model_path = str(log_dir / "world_model.pt")
        actor_path = str(log_dir / "actor.pt")
        torch.save(self._wm.bundle(), world_model_path)
        torch.save(actor_checkpoint(self.policy_learner.actor, self.cfg), actor_path)
        checkpoint_files = [world_model_path, actor_path]
        critic = getattr(self.policy_learner, "critic", None)
        if critic is not None:
          critic_path = str(log_dir / "critic.pt")
          torch.save(critic.state_dict(), critic_path)
          checkpoint_files.append(critic_path)
        # Skip ONNX export on huge datasets.
        if (
          self._representation.actor_obs_mode == "proprio"
          and self.motion.time_step_total <= 15_000
        ):
          checkpoint_files += export_actor_as_onnx(
            actor=self.policy_learner.actor,
            env=self.env.unwrapped,
            path=str(log_dir),
            run_name=wandb.run.name,
            term_widths=self._representation.proprio_term_widths(self._num_actions),
          )
        upload = uploader.submit(log_model_artifact, checkpoint_files)

    uploader.shutdown()
    self._cleanup()
