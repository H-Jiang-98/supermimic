"""Score frozen EIT dynamics, filter and critic on recorded independent episodes."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import torch

from supermimic.belief import BeliefState
from supermimic.critic_inputs import (
  BeliefCritic,
  CriticInputConfig,
  build_critic_features,
)
from supermimic.estimator import LieBeliefFilter
from supermimic.evaluation_diagnostics import PosteriorBins
from supermimic.learners.two_stage_wm import process_nll
from supermimic.models import Dynamics
from supermimic.representations import GenCoordRepresentation
from supermimic.runners.belief_training import parameter_digest
from supermimic.runners.eit_inference import evaluation_config, load_hashed_checkpoint
from supermimic.utils.lie_state import LieState, local_error

_TERMINAL = {"failure", "clip_complete"}
_CENSORED = {"timeout", "evaluation_truncated"}


def _validate_record(record, saved):
  if (
    record.get("version") != "eit_diagnostic_episodes_v1"
    or record.get("source") != "mujoco_evaluation"
    or record.get("split") != "diagnostic"
    or not record.get("episodes")
  ):
    raise ValueError("Expected independent MuJoCo diagnostic episodes")
  context = saved["world_model"]["sim_context"]
  if any(not torch.equal(v, record["sim_context"][k]) for k, v in context.items()):
    raise ValueError(
      "Record and checkpoint have different simulation context calibration"
    )
  if record["control_dt"] != saved["world_model"]["dynamics"]["metadata"]["dt"]:
    raise ValueError("Recorded control_dt differs from dynamics time step")
  ids = set()
  for ep in record["episodes"]:
    if ep["env_id"] in ids or ep["reason"] not in _TERMINAL | _CENSORED:
      raise ValueError("Duplicate episode slot or unknown termination reason")
    ids.add(ep["env_id"])
    n = len(ep["reward"])
    if (
      n < 1
      or ep["time_index"].dtype != torch.long
      or not torch.equal(ep["time_index"], torch.arange(n))
    ):
      raise ValueError("Each diagnostic episode must start at zero and be continuous")
    for key, shape in {
      "full_state": (n, 71),
      "sensor": (n, 93),
      "action": (n, 29),
      "sim_ctx": (n, context["scale"].numel()),
      "true_goal": (n, 700),
      "remaining_time": (n,),
      "reward": (n,),
      "prior_mean": (71,),
      "prior_covariance": (70, 70),
    }.items():
      value = ep[key]
      if (
        value.shape != shape
        or value.dtype != torch.float32
        or not torch.isfinite(value).all()
      ):
        raise ValueError(f"Invalid recorded {key}")
    if not torch.equal(ep["sensor"][1:, 64:], ep["action"][:-1]):
      raise ValueError("Executed action differs from next sensor's previous action")
    if not torch.equal(ep["sim_ctx"], ep["sim_ctx"][:1].expand_as(ep["sim_ctx"])):
      raise ValueError("Simulation context changed within an episode")


def _training_overlap(saved, record):
  """Check the evaluated model's training inputs against the recorded scene."""
  training_seed = saved.get("training_seed")
  files = saved.get("training_motion_files") or []
  matches = saved["motion_digest"] == record["motion_digest"] or any(
    f["sha256"] == record["motion_sha256"] for f in files
  )
  if training_seed is None:
    status = "unverified_missing_training_seed"
  elif training_seed != record["seed"]:
    status = "distinct_seed"
  elif matches:
    raise ValueError("Diagnostic records must use a distinct seed for training motion")
  elif files:
    status = "distinct_motion_file"
  else:
    status = "unverified_missing_training_file_manifest"
  return {
    "status": status,
    "training_seed": training_seed,
    "recording_seed": record["seed"],
    "training_motion_match": matches if files or matches else None,
  }


def _load_model(saved, cfg, device):
  bundle = saved["world_model"]
  checkpoint = bundle["dynamics"]
  meta, buffers = checkpoint["metadata"], checkpoint["state_dict"]
  joint_pos = bundle["estimator"]["default_joint_pos"].to(device)
  representation = GenCoordRepresentation(29, joint_pos)
  representation.dt = meta["dt"]
  representation.substep_coeff = meta["integration"]["substep_coeff"]
  dynamics = Dynamics(
    representation,
    29,
    len(bundle["sim_context"]["scale"]),
    buffers["action_scale"],
    buffers["action_offset"],
    cfg.dynamics_hidden_dims,
    cfg.process_noise,
  ).to(device)
  dynamics.load_checkpoint(checkpoint)
  if saved["learner"]["model_version"] != dynamics.model_version:
    raise ValueError("Saved learner and dynamics fingerprints disagree")
  estimator_meta = bundle["estimator"]
  if estimator_meta["measurement_config"] != asdict(
    cfg.estimator_inputs.measurement_noise
  ) or estimator_meta["filter_config"] != asdict(cfg.filter_config):
    raise ValueError("Saved filter and runner configuration disagree")
  dynamics.eval().requires_grad_(False)
  return dynamics, representation


def _white_statistics(residual, lower):
  residual, lower = residual.double(), lower.double()
  white = torch.linalg.solve_triangular(
    lower, residual.unsqueeze(-1), upper=False
  ).squeeze(-1)
  return (
    process_nll(residual, lower).sum(),
    residual[:, :9].sum(0),
    white.sum(0),
    white.T @ white,
  )


@torch.no_grad()
def score_process(dynamics, representation, episodes, window_frames, batch_size):
  """Use every within-episode start with a recorded successor (or full window)."""
  if window_frames < 2 or batch_size < 1:
    raise ValueError("window_frames must be >= 2 and batch_size positive")
  device = dynamics.action_scale.device
  pairs = windows = 0
  totals = None
  mse = {name: 0.0 for name in representation.LOSS_WEIGHT_CONFIG}
  offsets = torch.arange(window_frames, device=device)

  for ep in episodes:
    state, action, context = (
      ep[key].to(device) for key in ("full_state", "action", "sim_ctx")
    )
    n = len(state)
    for start in range(0, n - 1, batch_size):
      stop = min(n - 1, start + batch_size)
      selection = slice(start, stop)
      mean, factor = dynamics.distribution_lie(
        LieState.from_full(state[selection]), action[selection], context[selection]
      )
      residual = local_error(mean, LieState.from_full(state[start + 1 : stop + 1]))
      values = _white_statistics(residual, factor)
      totals = (
        values
        if totals is None
        else tuple(a + b for a, b in zip(totals, values, strict=True))
      )
      pairs += stop - start

    for start in range(0, n - window_frames + 1, batch_size):
      stop = min(n - window_frames + 1, start + batch_size)
      index = torch.arange(start, stop, device=device)[:, None] + offsets
      states = state[index]
      rep = representation.represent(states)
      aux = representation.extract_aux_state(states)
      target = representation.loss_blocks(rep[:, 1:], aux[:, 1:])
      next_rep, next_aux = rep[:, 0], aux[:, 0]
      for t in range(window_frames - 1):
        current = index[:, t]
        next_aux, next_rep = dynamics.mean_rep(
          next_rep, action[current], next_aux, context[current]
        )
        predicted = representation.loss_blocks(next_rep, next_aux)
        for name, value in predicted.items():
          mse[name] += float(
            (value - target[name][:, t]).double().square().mean(-1).sum()
          )
      windows += stop - start
  n_scored = windows * (window_frames - 1)
  nll, bias, white_mean, second = (x / pairs for x in totals) if totals else (None,) * 4
  return {
    "windows": windows,
    "window_frames": window_frames,
    "window_scored_successors": n_scored,
    "multistep_mse": {k: v / n_scored if n_scored else None for k, v in mse.items()},
    "transitions": pairs,
    "process_nll": float(nll) if nll is not None else None,
    "rpv_bias_truth_minus_prediction_right_local": bias.tolist()
    if bias is not None
    else None,
    "whitened_mean": white_mean.tolist() if white_mean is not None else None,
    "whitened_second_moment": second.tolist() if second is not None else None,
  }


def discounted_returns(rewards, gamma):
  """Full hard-terminal returns-to-go, including the terminating action."""
  if rewards.ndim != 1 or not 0 <= gamma < 1:
    raise ValueError("Expected one episode of rewards and 0 <= gamma < 1")
  result = torch.empty_like(rewards, dtype=torch.float64)
  tail = 0.0
  for t in reversed(range(len(rewards))):
    tail = float(rewards[t]) + gamma * tail
    result[t] = tail
  return result


def _errors(errors):
  if not errors:
    return {"samples": 0, "bias": None, "mae": None, "rmse": None}
  values = torch.cat(errors).double()
  return {
    "samples": values.numel(),
    "bias": float(values.mean()),
    "mae": float(values.abs().mean()),
    "rmse": float(values.square().mean().sqrt()),
  }


def _load_critic(saved, cfg, record, digest, device):
  if saved["stage"] != "policy":
    raise ValueError("Critic scoring requires a policy-stage checkpoint")
  group = cfg.estimator_inputs.group
  if record["checkpoint_sha256"] != digest or record["group"] != group:
    raise ValueError("Critic Monte Carlo returns require episodes from this checkpoint")
  if record["actor_state_source"] not in (
    ("truth", "posterior") if group == "A" else ("posterior",)
  ):
    raise ValueError("Incompatible behavior policy input source")
  critic = BeliefCritic(
    group,
    len(saved["world_model"]["sim_context"]["scale"]),
    cfg.critic_hidden_dims,
    config=CriticInputConfig(**saved["critic"]["contract"]["config"]),
  )
  critic.load_checkpoint(saved["critic"])
  if saved["critic_version"] != parameter_digest(critic):
    raise ValueError("Invalid critic fingerprint")
  if saved["policy_config"] != saved["policy"]["config"]:
    raise ValueError("Saved critic and learner configuration disagree")
  gamma = saved["policy_config"]["gamma"]
  value_cap = saved["policy"].get("bootstrap_value_cap")
  if saved["policy_config"]["value_clipping"]:
    if value_cap is None:
      value_cap = record["max_step_reward"] / (1 - gamma)
    if not math.isfinite(value_cap) or value_cap <= 0:
      raise ValueError("Invalid bootstrap value cap")
  else:
    value_cap = None
  return critic.eval().requires_grad_(False).to(device), gamma, value_cap


def _posterior_stream(estimator, ep, device):
  prior = BeliefState(
    LieState.from_full(ep["prior_mean"][None].to(device)),
    ep["prior_covariance"][None].to(device),
  )
  for t in range(len(ep["reward"])):
    belief, diagnostic = estimator.correct(
      prior, ep["sensor"][t : t + 1, :64].to(device), return_diagnostics=True
    )
    yield belief, diagnostic
    if t + 1 < len(ep["reward"]):
      prior = estimator.predict(
        belief, ep["action"][t : t + 1].to(device), ep["sim_ctx"][t : t + 1].to(device)
      )


@torch.no_grad()
def score_frames(estimator, record, device, bins, critic_info):
  critic, gamma, cap = critic_info if critic_info else (None, None, None)
  episodes = []
  errors, raw_errors = [], []
  censored_samples = censored_episodes = 0
  for ep in record["episodes"]:
    n = len(ep["reward"])
    values = []
    sensor = ep["sensor"].to(device)
    for t, (belief, diagnostic) in enumerate(_posterior_stream(estimator, ep, device)):
      truth = LieState.from_full(ep["full_state"][t : t + 1].to(device))
      if bins is not None:
        bins.update(
          ep["time_index"].new_tensor([ep["env_id"]]).to(device),
          ep["time_index"][t : t + 1].to(device),
          local_error(belief.mean, truth)[:, :9],
          belief.covariance[:, :9, :9],
          diagnostic.jitter,
          diagnostic.innovation.norm(dim=-1),
          truth.position - belief.mean.position,
          truth.velocity - belief.mean.velocity,
        )
      if critic is not None:
        # Reset pads with five copies of the first measured frame.
        history = sensor[torch.arange(t - 4, t + 1, device=device).clamp_min(0)][None]
        features = build_critic_features(
          truth,
          ep["true_goal"][t : t + 1].to(device),
          history,
          ep["sim_ctx"][t : t + 1].to(device),
          ep["remaining_time"][t : t + 1].to(device),
          belief,
          config=critic.config,
        )
        values.append(critic(features).reshape(()).cpu())
    if critic is not None:
      raw = torch.stack(values).double()
      used = raw if cap is None else raw.clamp(max=cap)
      returns = None
      if ep["reason"] in _TERMINAL:
        returns = discounted_returns(ep["reward"], gamma)
        errors.append(used - returns)
        raw_errors.append(raw - returns)
      else:
        censored_samples += n
        censored_episodes += 1
      episodes.append(
        {
          "env_id": ep["env_id"],
          "reason": ep["reason"],
          "raw_value": raw.tolist(),
          "bootstrap_value": used.tolist(),
          "discounted_return": returns.tolist() if returns is not None else None,
        }
      )
  result = {}
  if bins is not None:
    result["fixed_record_posterior"] = bins.result()
  if critic is not None:
    result["critic"] = {
      "gamma": gamma,
      "bootstrap_value_cap": cap,
      "actor_state_source": record["actor_state_source"],
      "raw_value_error": _errors(raw_errors),
      "bootstrap_value_error": _errors(errors),
      "censored_samples": censored_samples,
      "censored_episodes": censored_episodes,
      "episodes": episodes,
    }
  return result


@torch.no_grad()
def diagnose(
  checkpoint,
  recording,
  *,
  mode="all",
  device="cpu",
  batch_size=128,
  window_frames=8,
  bin_steps=50,
  imagined_horizon=0,
  imagined_samples=4,
  imagined_seed=2027,
  actor_gradients=False,
  gradient_seed=2028,
  gradient_episodes=4,
  motion_file=None,
):
  if mode not in ("all", "process", "posterior", "critic"):
    raise ValueError("Unknown diagnostic mode")
  saved, digest = load_hashed_checkpoint(checkpoint, "cpu")
  record, record_digest = load_hashed_checkpoint(recording, "cpu")
  cfg = evaluation_config(saved)
  assert cfg.estimator_inputs is not None
  _validate_record(record, saved)
  independence = _training_overlap(saved, record)
  dynamics, representation = _load_model(saved, cfg, device)
  result = {
    "version": "eit_independent_diagnostics_v1",
    "checkpoint_sha256": digest,
    "recording_sha256": record_digest,
    "recording_source": record["source"],
    "split": record["split"],
    "training_overlap_check": independence,
    "motion_digest": record["motion_digest"],
    "recording_seed": record["seed"],
    "recording_checkpoint_sha256": record["checkpoint_sha256"],
    "actor_state_source": record["actor_state_source"],
    "stage": saved["stage"],
    "group": cfg.estimator_inputs.group,
    "model_version": dynamics.model_version,
    "round": saved["rounds"],
    "episodes": len(record["episodes"]),
  }
  if mode in ("process", "all"):
    result["process"] = score_process(
      dynamics, representation, record["episodes"], window_frames, batch_size
    )
  if mode != "process":
    bins = (
      PosteriorBins(record["control_dt"], bin_steps)
      if mode in ("all", "posterior")
      else None
    )
    critic = (
      _load_critic(saved, cfg, record, digest, device)
      if mode == "critic" or (mode == "all" and saved["stage"] == "policy")
      else None
    )
    estimator = LieBeliefFilter(
      dynamics,
      cfg.estimator_inputs.measurement_noise,
      representation._default_joint_pos,
      config=cfg.filter_config,
    )
    result.update(score_frames(estimator, record, device, bins, critic))
  if imagined_horizon:
    if "critic" not in result or imagined_horizon < 1 or imagined_samples < 1:
      raise ValueError(
        "Imagined return comparison requires critic scoring and positive horizon/samples"
      )
    from supermimic.imagined_diagnostics import compare_episode_heads

    result["imagined"] = compare_episode_heads(
      saved,
      cfg,
      record,
      result["critic"],
      imagined_horizon,
      imagined_samples,
      imagined_seed,
      device,
      motion_file=motion_file,
    )
  if actor_gradients:
    if "critic" not in result:
      raise ValueError("Actor gradient diagnostics require critic scoring")
    from supermimic.imagined_diagnostics import episode_head_actor_gradients

    result["actor_gradients"] = episode_head_actor_gradients(
      saved,
      cfg,
      record,
      critic[0],
      gradient_seed,
      device,
      motion_file=motion_file,
      max_episodes=gradient_episodes,
    )
  return result


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--recording", type=Path, required=True)
  parser.add_argument(
    "--mode", choices=("all", "process", "posterior", "critic"), default="all"
  )
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--batch-size", type=int, default=128)
  parser.add_argument("--window-frames", type=int, default=8)
  parser.add_argument("--bin-steps", type=int, default=50)
  parser.add_argument(
    "--motion-file",
    type=Path,
    help="Location of the original motion for imagined returns",
  )
  parser.add_argument("--imagined-horizon", type=int, default=0)
  parser.add_argument("--imagined-samples", type=int, default=4)
  parser.add_argument("--imagined-seed", type=int, default=2027)
  parser.add_argument(
    "--actor-gradients",
    action="store_true",
    help="Separate reward/bootstrap actor gradients on fixed episode heads and noise",
  )
  parser.add_argument("--gradient-seed", type=int, default=2028)
  parser.add_argument("--gradient-episodes", type=int, default=4)
  parser.add_argument("--output", type=Path, required=True)
  kwargs = vars(parser.parse_args())
  output = kwargs.pop("output")
  protected_inputs = (kwargs["checkpoint"], kwargs["recording"], kwargs["motion_file"])
  if any(path and path.resolve() == output.resolve() for path in protected_inputs):
    raise ValueError("Diagnostic output must be distinct from the inputs")
  with torch.compiler.set_stance("force_eager"):
    result = diagnose(**kwargs)
  content = json.dumps(result, allow_nan=False)
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(content + "\n")


if __name__ == "__main__":
  main()
