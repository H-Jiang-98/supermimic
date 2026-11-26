"""Decode serialized runner configs and validate EIT checkpoint metadata."""

from __future__ import annotations

import dataclasses
import math
import types
from typing import Any, Union, get_args, get_origin, get_type_hints

import torch

from supermimic.critic_inputs import CRITIC_INPUT_VERSION, CriticInputConfig
from supermimic.estimator import FilterConfig
from supermimic.estimator_inputs import EITInputConfig, GaussianMeasurementConfig
from supermimic.runners.eit_snapshots import checkpoint_snapshot_interval
from supermimic.runners.supermimic_runner import SuperMimicRunnerCfg

EIT_TRAINING_VERSION = "eit_model_training_v5"
EIT_POLICY_TRAINING_VERSION = "eit_training_v7"
_LEGACY_POLICY_VERSION = "eit_training_v5"
LEGACY_INITIALIZATION_VERSION = "eit_initialization_v5"
EIT_INITIALIZATION_VERSION = "eit_initialization_v6"


def validate_eit_checkpoint_contract(
  saved: dict,
  *,
  stage: str | None = None,
  for_inference: bool = False,
  archived_pilot: bool = False,
) -> None:
  """Validate the EIT format and the joint trainer's critic layout before loading."""
  version = saved.get("version") if isinstance(saved, dict) else None
  legacy = for_inference and version == _LEGACY_POLICY_VERSION
  archive = archived_pilot and version == "eit_training_v6"
  if version not in (EIT_TRAINING_VERSION, EIT_POLICY_TRAINING_VERSION) and not (
    legacy or archive
  ):
    raise ValueError(
      f"Expected {EIT_TRAINING_VERSION} or {EIT_POLICY_TRAINING_VERSION} checkpoint, got {version!r}"
    )
  if saved["version"] == EIT_TRAINING_VERSION:
    checkpoint_snapshot_interval(saved)
  kind = saved.get("stage")
  if kind not in ("model", "policy") or (stage is not None and kind != stage):
    raise ValueError(f"Unexpected EIT training stage: {kind!r}")
  expected_kind = "model" if saved["version"] == EIT_TRAINING_VERSION else "policy"
  if kind != expected_kind:
    raise ValueError("Checkpoint version and training stage disagree")
  replay_mode = saved.get("replay_mode")
  if replay_mode not in ("model", "full") or (
    kind == "policy" and replay_mode != "full"
  ):
    raise ValueError(f"Invalid replay mode {replay_mode!r} for {kind} checkpoint")
  # Batch evaluation can reuse cached results without constructing the filter.
  for component, section, key in (
    ("runner_config", "estimator_inputs", "measurement_noise"),
    ("actor", "input_config", "measurement_noise"),
    ("world_model", "estimator", "measurement_config"),
  ):
    bundle = saved.get(component)
    settings = bundle.get(section) if isinstance(bundle, dict) else None
    if isinstance(settings, dict) and isinstance(noise := settings.get(key), dict):
      GaussianMeasurementConfig(**noise)
  if kind == "policy":
    from supermimic.posterior_replay import POSTERIOR_REPLAY_VERSION

    error = "Invalid incremental posterior replay checkpoint contract"
    try:
      stride = saved["loop_config"]["posterior_save_stride"]
      revision = saved["model_revision"]
      compatible = (
        kind == "policy"
        and (
          saved.get("posterior_replay_protocol")
          == (
            "incremental_posterior_sqrt_tril_f32_v2"
            if archive
            else POSTERIOR_REPLAY_VERSION
          )
          or for_inference
          and saved.get("posterior_replay_protocol")
          == "incremental_posterior_tril_f32_v1"
        )
        and type(stride) is int
        and 1 <= stride <= saved["runner_config"]["buffer_size"]
        and type(revision) is int
        and revision >= 0
        and revision == saved["world_model"]["dynamics"].get("model_revision")
      )
    except (KeyError, TypeError, AttributeError) as exc:
      raise ValueError(error) from exc
    if not compatible:
      raise ValueError(error)
  if kind == "model":
    if "critic" in saved or "policy" in saved:
      raise ValueError("Model-stage checkpoint contains policy training state")
    return
  critic_version = "eit_rpv_belief_critic_v2" if legacy else CRITIC_INPUT_VERSION
  if archive:
    critic_version = "eit_rpv_belief_critic_v3"
  error = f"EIT checkpoint requires the {critic_version} critic contract"
  try:
    contract = saved["critic"]["contract"]
    config_data = dict(contract["config"])
    if legacy:
      # v5 actor inference only. Validate the extra critic column without loading it.
      scale = config_data.pop("remaining_time_scale")
      if (
        not math.isfinite(scale)
        or scale <= 0
        or config_data["version"] != critic_version
      ):
        raise ValueError(error)
      config_data["version"] = CRITIC_INPUT_VERSION
    if archive:
      if (
        config_data["version"] != critic_version
        or config_data["group"] not in ("A", "B", "C")
        or config_data["tangent_version"] != "right_se23_theta_rho_nu_omega_q_qdot_v1"
      ):
        raise ValueError(error)
      # The archived A/B/C critic layout has the same widths. Validate its
      # stored identifiers above before reusing the numeric width/scale checks.
      config_data.update(
        version=CRITIC_INPUT_VERSION,
        tangent_version=CriticInputConfig().tangent_version,
      )
    config = CriticInputConfig(**config_data)
    group = saved["runner_config"]["estimator_inputs"]["group"]
    context_dim = contract["context_dim"]
    input_dim = config.input_dim(context_dim) + int(legacy)
    context_scale = saved["world_model"]["sim_context"]["scale"]
    input_weight = saved["critic"]["state_dict"]["mlp.0.weight"]
    compatible = (
      contract["version"] == critic_version
      and config_data == dataclasses.asdict(config)
      and config.group == group
      and contract["input_dim"] == input_dim
    )
  except (KeyError, TypeError, ValueError) as exc:
    raise ValueError(error) from exc
  if not compatible:
    raise ValueError(error)
  if not isinstance(context_scale, torch.Tensor) or not isinstance(
    input_weight, torch.Tensor
  ):
    raise ValueError(
      f"{error}: world-model context scale and critic input weights must be tensors"
    )
  if context_scale.shape != (context_dim,):
    raise ValueError(
      f"{error}: critic expects {context_dim} context channels, "
      f"world-model context scale has shape {tuple(context_scale.shape)}"
    )
  if input_weight.ndim != 2 or input_weight.shape[1] != input_dim:
    raise ValueError(
      f"{error}: critic expects {input_dim} inputs, "
      f"mlp.0.weight has shape {tuple(input_weight.shape)}"
    )


def load_runner_cfg(run_config: dict[str, Any]) -> SuperMimicRunnerCfg:
  """Reconstruct ``SuperMimicRunnerCfg`` from ``run_config["runner_cfg"]``.

  JSON round-tripping turns tuples into lists and erases the ``policy`` union's
  variant; both are recovered structurally. EIT snapshots require every serialized field.
  Original SuperMimic run configs can omit fields and use current defaults.
  """
  raw = run_config.get("runner_cfg")
  if not raw:
    raise ValueError("run has no serialized runner_cfg; cannot rebuild its config")
  defaulted = None if raw.get("estimator_inputs") is not None else []
  cfg = _rebuild(SuperMimicRunnerCfg, raw, defaulted, "")
  if defaulted:
    print(f"[WARN] run cfg missing fields, using current defaults: {defaulted}")
  return cfg


def check_representation_matches(task_id: str, representation_name: str) -> None:
  """The task fixes the env's obs layout; a run trained with the other
  representation would rebuild — and fail to load — at the wrong shapes."""
  expected = "maxcoord" if "MaxCoord" in task_id else "gencoord"
  if representation_name != expected:
    raise ValueError(
      f"{task_id} is a {expected} task but the run was trained with "
      f"{representation_name}; pass a matching task id"
    )


def _rebuild(
  cls: type, raw: dict[str, Any], defaulted: list[str] | None, prefix: str
) -> Any:
  if cls is FilterConfig:
    return FilterConfig.from_dict(raw)
  hints = get_type_hints(cls)
  kwargs: dict[str, Any] = {}
  if defaulted is None and raw.keys() != {
    f.name for f in dataclasses.fields(cls) if f.init
  }:
    raise ValueError(
      f"Incomplete or unknown EIT configuration fields in {prefix or cls.__name__}"
    )
  for field in dataclasses.fields(cls):
    if not field.init:
      continue
    if field.name not in raw:
      assert defaulted is not None
      defaulted.append(f"{prefix}{field.name}")
      continue
    kwargs[field.name] = _coerce(
      hints[field.name], raw[field.name], defaulted, f"{prefix}{field.name}."
    )
  return cls(**kwargs)


def _coerce(hint: Any, value: Any, defaulted: list[str] | None, prefix: str) -> Any:
  origin = get_origin(hint)
  if origin is tuple and isinstance(value, list):
    # Flat tuple fields only: a tuple nested inside another hint stays a list.
    return tuple(value)
  is_union = origin is Union or origin is types.UnionType
  candidates = get_args(hint) if is_union else (hint,)
  variants = [c for c in candidates if _is_cfg_type(c)]
  if EITInputConfig in variants and isinstance(value, dict):
    # A versioned sensor/goal contract must never be silently filled from new defaults.
    return EITInputConfig.from_dict(value)
  if variants and isinstance(value, dict):
    return _rebuild(_pick_variant(variants, value), value, defaulted, prefix)
  return value


def _is_cfg_type(hint: Any) -> bool:
  return isinstance(hint, type) and dataclasses.is_dataclass(hint)


def _pick_variant(variants: list[type], raw: dict[str, Any]) -> type:
  """Best field overlap with the serialized keys; ties go to the smallest
  variant, which is the exact match when a subclass merely extends it."""

  def score(variant: type) -> tuple[int, int]:
    names = {f.name for f in dataclasses.fields(variant)}
    return (len(names & raw.keys()), -len(names))

  return max(variants, key=score)
