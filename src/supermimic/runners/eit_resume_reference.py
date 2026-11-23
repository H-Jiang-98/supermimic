"""Next-update evidence cloned from live optimizers before checkpoint commit.

The expected side never invokes a checkpoint or optimizer loader. Physics and
motion are read-only inputs; copied modules, parameter-linked optimizers and RNG
streams own every hypothetical update. The live trainer continues unchanged.
"""

from __future__ import annotations

import math
import os
from contextlib import contextmanager
from copy import copy, deepcopy
from dataclasses import fields
from pathlib import Path
from tempfile import NamedTemporaryFile
from types import SimpleNamespace
from typing import Any, cast

import torch
from tensordict import TensorDict

from supermimic.belief import BeliefState
from supermimic.belief_provider import BeliefProvider
from supermimic.envs.world_model_env import BeliefRolloutStart
from supermimic.estimator import LieBeliefFilter
from supermimic.motion_loader import motion_content_digest
from supermimic.runners.eit_resume_contract import REFERENCE_VERSION
from supermimic.runners.eit_snapshots import file_sha256
from supermimic.utils.gencoord_helpers import compute_gencoord_state
from supermimic.utils.lie_state import LieState

REFERENCE_SOURCE = "live_training_before_checkpoint_commit"


def execution_contract() -> dict[str, Any]:
  return {
    "torch_compile": "force_eager",
    "torchscript_optimized_execution": False,
    "torchscript_import_mode": (
      "enabled"
      if isinstance(compute_gencoord_state, torch.jit.ScriptFunction)
      else "disabled_before_import"
    ),
    "deterministic_algorithms": True,
    "fixed_input_assembly": "cpu_contiguous_then_shared_device_constructors_v1",
    "discarded_private_policy_warmup_updates": 1,
    "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    "expected_optimizer_origin": "deepcopy_of_live_parameter_optimizer_graph",
  }


@contextmanager
def fixed_update_execution():
  enabled = torch.are_deterministic_algorithms_enabled()
  warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
  try:
    torch.use_deterministic_algorithms(True, warn_only=False)
    with torch.compiler.set_stance("force_eager"), torch.jit.optimized_execution(False):
      yield
  finally:
    torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def equal_tree(left, right, path="root"):
  """Compare finite tensor bytes exactly, allowing CPU/device relocation only."""
  if isinstance(left, torch.Tensor):
    if not isinstance(right, torch.Tensor):
      raise AssertionError(f"{path}: tensor type differs")
    if left.dtype != right.dtype or left.shape != right.shape:
      raise AssertionError(
        f"{path}: tensor dtype/shape differs: {left.dtype} {tuple(left.shape)} "
        f"!= {right.dtype} {tuple(right.shape)}"
      )
    a, b = left.detach().cpu().contiguous(), right.detach().cpu().contiguous()
    if (a.is_floating_point() or a.is_complex()) and (
      not torch.isfinite(a).all() or not torch.isfinite(b).all()
    ):
      raise AssertionError(f"{path}: non-finite tensor cannot establish exact restore")
    a, b = a.reshape(-1).view(torch.uint8), b.reshape(-1).view(torch.uint8)
    if not torch.equal(a, b):
      different = (a != b).nonzero().flatten()
      raise AssertionError(
        f"{path}: tensor bytes differ at {len(different)} bytes; "
        f"first byte offset {int(different[0])}"
      )
  elif isinstance(left, dict):
    if not isinstance(right, dict) or left.keys() != right.keys():
      raise AssertionError(f"{path}: dictionary keys differ")
    for key in left:
      equal_tree(left[key], right[key], f"{path}.{key}")
  elif isinstance(left, (list, tuple)):
    if type(left) is not type(right) or len(left) != len(right):
      raise AssertionError(f"{path}: sequence type/length differs")
    for index, (a, b) in enumerate(zip(left, right, strict=True)):
      equal_tree(a, b, f"{path}[{index}]")
  else:
    if isinstance(left, float) and not math.isfinite(left):
      raise AssertionError(f"{path}: non-finite scalar cannot establish exact restore")
    if type(left) is not type(right) or left != right:
      raise AssertionError(f"{path}: {left!r} != {right!r}")


def _cpu(value):
  if isinstance(value, torch.Tensor):
    return value.detach().cpu().contiguous().clone()
  if isinstance(value, dict):
    return {key: _cpu(item) for key, item in value.items()}
  if isinstance(value, (tuple, list)):
    return type(value)(_cpu(item) for item in value)
  return value


def _private_components(training):
  # Copy this entire graph at once: optimizer parameters must point at the same
  # cloned Parameters as their modules, and mean/noise learners share dynamics.
  live = training.policy
  cloned = deepcopy(
    {
      "world_model": training.runner._wm,
      "learner": training.learner,
      "actor": live.actor,
      "critic": live.critic,
      "actor_optimizers": live.actor_optimizers,
      "critic_optimizers": live.critic_optimizers,
      "generators": live.generators,
    }
  )
  wm, learner = cloned["world_model"], cloned["learner"]
  if (
    learner.dynamics is not wm.dynamics
    or learner.mean.representation is not wm.representation
  ):
    raise RuntimeError("Private learner lost shared model/representation identity")
  # torch.compile wrappers are functions with closures, which deepcopy cannot
  # rebind. The fixed-input verifier deliberately uses eager arithmetic.
  mean = learner.mean
  mean._compiled_rollout = mean._rollout
  mean._compiled_compute_losses = mean._compute_losses
  mean._compiled_represent = wm.representation.represent
  mean._transition = wm.dynamics.mean_rep
  mean._captured_update._fn = mean._update_from
  mean._captured_update._enabled = False
  env = copy(training.runner.wm_env)
  env.dynamics, env.representation = wm.dynamics, wm.representation
  cfg = training.runner.cfg
  if cfg.estimator_inputs is None:
    raise ValueError("Live continuation evidence requires estimator inputs")
  env.posterior_provider = BeliefProvider(
    LieBeliefFilter(
      wm.dynamics,
      cfg.estimator_inputs.measurement_noise,
      wm.representation._default_joint_pos,
      config=cfg.filter_config,
    ),
    differentiable=True,
    incremental=training.runner.wm_env.posterior_provider.incremental,
  )
  env._readonly_motion = None
  env.clear_rollout()
  policy = copy(live)
  for name in (
    "actor",
    "critic",
    "actor_optimizers",
    "critic_optimizers",
    "generators",
  ):
    setattr(policy, name, cloned[name])
  policy.wm_env = env
  return wm, learner, policy, env


def _model_update(learner, windows, pairs):
  logs = (
    learner.update_mean(windows),
    learner.update_covariance(learner.residual_data(pairs)),
  )
  return logs


def _model_inputs(inputs, device):
  return tuple(
    TensorDict(
      inputs[name],
      batch_size=list(inputs[name]["state"].shape[:2]),
      device=device,
    )
    for name in ("mean_windows", "covariance_pairs")
  )


def _policy_start(payload, replay, device):
  return BeliefRolloutStart(
    TensorDict(
      payload["batch"], batch_size=[len(payload["remaining_steps"])], device=device
    ),
    BeliefState(
      LieState(
        **{key: value.to(device) for key, value in payload["belief_mean"].items()}
      ),
      payload["belief_covariance"].to(device),
    ),
    payload["model_revision"],
    replay,
    replay.insert_count,
    payload["motion_digest"],
    payload["remaining_steps"].to(device),
  )


def _warm_policy(wm, learner, policy, env, cfg, payload, noises):
  """Prime first-backward kernels with one discarded, independent object graph."""
  context = SimpleNamespace(
    runner=SimpleNamespace(_wm=wm, wm_env=env, cfg=cfg),
    learner=learner,
    policy=policy,
  )
  _, _, warm_policy, _ = _private_components(context)
  device = next(warm_policy.actor.parameters()).device
  warm_policy.update_from(
    _policy_start(payload, env.replay_buffer, device),
    *(noise.to(device) for noise in noises),
  )


def build_live_reference(training, checkpoint: Path, state: dict):
  """Capture expected updates from live state; ``state`` is only identity metadata."""
  if state.get("stage") != "policy" or state["update_groups"] < 1:
    raise ValueError(
      "Live continuation evidence requires a completed policy update group"
    )
  if training.update_groups != state["update_groups"]:
    raise ValueError("Live trainer and pending checkpoint update groups differ")
  # Verify that the committed bytes will describe these live components. This is
  # a comparison only: none of the saved values is loaded into the expected side.
  current = training.state_dict()
  for key in ("world_model", "actor", "critic", "learner", "policy"):
    equal_tree(current[key], state[key], f"live_checkpoint.{key}")
  del current
  parameter = next(training.actor.parameters())
  devices = [parameter.device.index] if parameter.device.type == "cuda" else []
  with (
    torch.random.fork_rng(devices=devices),
    fixed_update_execution(),
  ):
    wm, learner, policy, env = _private_components(training)
    replay = training.runner.replay_buffer
    train_ids, _ = training.episode_split()
    length = learner.mean_cfg.window_frames
    candidates = replay.model_window_starts(train_ids, length)
    if not len(candidates):
      raise ValueError("Live resume reference has no complete training window")
    windows = replay.model_windows(
      candidates[
        torch.arange(learner.mean_cfg.batch_size, device=candidates.device)
        % len(candidates)
      ],
      length,
    ).select("state", "action", "sim_ctx")
    pairs = replay.model_windows(
      replay.model_window_starts(train_ids, 2)[: learner.cov_cfg.batch_size],
      2,
    ).select("state", "action", "sim_ctx")
    frames = replay.posterior_frames(
      training.policy_episode_ids(), fields=("episode_id", "time_index")
    )
    if not len(frames):
      raise ValueError("Live resume reference has no saved training posterior")
    indices = torch.arange(policy.cfg.batch_size, device=frames.device) % len(frames)
    keys = torch.stack(
      (frames["episode_id"][indices], frames["time_index"][indices]), 1
    )
    fixed_model_inputs = cast(
      dict[str, Any],
      _cpu(
        {"mean_windows": dict(windows.items()), "covariance_pairs": dict(pairs.items())}
      ),
    )
    windows, pairs = _model_inputs(fixed_model_inputs, parameter.device)
    model_logs = _model_update(learner, windows, pairs)
    model_expected = _cpu(
      {
        "logs": model_logs,
        "world_model": wm.bundle(),
        "learner": learner.state_dict(),
      }
    )
    with env.readonly_policy_motion():
      starts = env.prepare_policy_starts(*keys.unbind(dim=1))
      noises = policy.draw_noises(starts)
      fixed_starts = _cpu(
        {
          "batch": dict(starts.batch.items()),
          # Preserve the native rotation matrix: quaternion round-tripping would
          # add floating-point error to an otherwise bitwise fixed input.
          "belief_mean": {
            field.name: getattr(starts.belief.mean, field.name)
            for field in fields(starts.belief.mean)
          },
          "belief_covariance": starts.belief.covariance,
          "remaining_steps": starts.remaining_steps,
          "model_revision": starts.model_version,
          "model_version": env.dynamics.model_version,
          "motion_digest": starts.motion_digest,
        }
      )
      fixed_noises = _cpu(noises)
      _warm_policy(
        wm, learner, policy, env, training.runner.cfg, fixed_starts, fixed_noises
      )
      starts = _policy_start(fixed_starts, replay, parameter.device)
      policy_logs = policy.update_from(
        starts, *(noise.to(parameter.device) for noise in fixed_noises)
      )
    reference = {
      "version": REFERENCE_VERSION,
      "source": REFERENCE_SOURCE,
      "checkpoint_sha256": file_sha256(checkpoint),
      "collected_steps": state["collected_steps"],
      "update_groups": state["update_groups"],
      "group": state["runner_config"]["estimator_inputs"]["group"],
      "device": str(parameter.device),
      "dtype": str(parameter.dtype),
      "initialization_digest": state.get("initialization_digest"),
      "motion_digest": state["motion_digest"],
      "source_update_counts": {
        "mean": state["learner"]["mean"]["steps"],
        "covariance": state["learner"]["cov"]["steps"],
        "actor": state["policy"]["actor_steps"],
        "critic": state["policy"]["critic_steps"],
      },
      "inputs": _cpu(
        {
          **fixed_model_inputs,
          "starts": fixed_starts,
          "policy_keys": keys,
          "noises": fixed_noises,
        }
      ),
      "expected_model": model_expected,
      "expected_policy": _cpu(
        {
          "logs": policy_logs,
          "actor": policy.actor.state_dict(),
          "critic": policy.critic.state_dict(),
          "policy": policy.state_dict(),
        }
      ),
      "verification_execution": execution_contract(),
    }
  return reference


def write_live_reference(training, checkpoint: Path, state: dict, output: Path):
  """Publish reference before the exact temporary checkpoint bytes are committed."""
  output = Path(output)
  if output.resolve() == Path(checkpoint).resolve():
    raise ValueError("Reference output must differ from checkpoint")
  reference = build_live_reference(training, checkpoint, state)
  output.parent.mkdir(parents=True, exist_ok=True)
  with NamedTemporaryFile(dir=output.parent, delete=False) as stream:
    temporary = Path(stream.name)
    try:
      torch.save(reference, stream)
      stream.flush()
      temporary.replace(output)
    finally:
      temporary.unlink(missing_ok=True)


def compare_reference_update(training, reference) -> dict[str, Any]:
  """The caller has restored once; compare its update against live-only evidence."""
  device = training.runner.device
  if reference["verification_execution"] != execution_contract():
    raise ValueError("Live reference and verifier numeric execution contracts differ")
  inputs = reference["inputs"]
  windows, pairs = _model_inputs(inputs, device)
  with fixed_update_execution():
    model_logs = _model_update(training.learner, windows, pairs)
    equal_tree(
      reference["expected_model"],
      {
        "logs": model_logs,
        "world_model": training.runner._wm.bundle(),
        "learner": training.learner.state_dict(),
      },
      "model_update",
    )
    payload = inputs["starts"]
    replay, env = training.runner.replay_buffer, training.runner.wm_env
    if (
      payload["model_version"] != env.dynamics.model_version
      or payload["model_revision"] != env.dynamics.model_revision
    ):
      raise ValueError("Reference start model version differs from restored update")
    if payload["motion_digest"] != motion_content_digest(env.motion):
      raise ValueError("Reference start motion differs from restored motion")
    starts = _policy_start(payload, replay, device)
    noises = training.policy.draw_noises(starts)
    equal_tree(inputs["noises"], noises, "policy_noises")
    _warm_policy(
      training.runner._wm,
      training.learner,
      training.policy,
      env,
      training.runner.cfg,
      payload,
      inputs["noises"],
    )
    logs = training.policy.update_from(
      starts, *(noise.to(device) for noise in inputs["noises"])
    )
    equal_tree(
      reference["expected_policy"],
      {
        "logs": logs,
        "actor": training.actor.state_dict(),
        "critic": training.policy.critic.state_dict(),
        "policy": training.policy.state_dict(),
      },
      "policy_update",
    )
  return {
    "mean_windows": len(windows),
    "covariance_pairs": len(pairs),
    "policy_starts": inputs["policy_keys"].tolist(),
  }
