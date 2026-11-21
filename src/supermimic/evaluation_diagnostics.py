"""Read-only calibration and recording at the evaluator's pre-action boundary."""

import math
from collections import defaultdict

import torch

from supermimic.estimator_inputs import RpvState, build_motion_goal
from supermimic.utils.lie_state import LieState

# 0.95 quantiles of chi-square with 0..9 degrees of freedom.
_CHI95 = (
  0.0,
  3.841459,
  5.991465,
  7.814728,
  9.487729,
  11.070498,
  12.591587,
  14.067140,
  15.507313,
  16.918978,
)


def rpv_ellipsoid(error, covariance):
  """Mahalanobis distance on standardized PSD support, rank, and violations."""
  if covariance.shape != (*error.shape, 9) or error.shape[-1] != 9:
    raise ValueError("RPV ellipsoids require matching 9D errors and 9x9 marginals")
  if not torch.isfinite(error).all() or not torch.isfinite(covariance).all():
    raise ValueError("Non-finite RPV error or covariance")
  eps = torch.finfo(covariance.dtype).eps
  matrix = covariance.double()
  matrix = (matrix + matrix.mT) * 0.5
  variances = matrix.diagonal(dim1=-2, dim2=-1)
  zero_variance = variances == 0
  if (variances < 0).any() or (zero_variance & (matrix != 0).any(dim=-1)).any():
    raise ValueError("Indefinite RPV covariance")
  # Detect support in correlation space: physical units and unequal marginal
  # scales must not turn small positive variances into deterministic directions.
  scale = torch.where(zero_variance, 1, variances).sqrt()
  matrix = matrix / scale.unsqueeze(-1) / scale.unsqueeze(-2)
  standardized_error = (error.double() / scale).masked_fill(zero_variance, 0)
  values, vectors = torch.linalg.eigh(matrix)
  spectral_scale = values.abs().amax(-1, keepdim=True)
  # Allow negative roundoff from the stored covariance's precision, but resolve
  # positive modes at the float64 working precision. A small positive variance
  # is not a deterministic constraint, even for strongly correlated float32 data.
  negative_tolerance = spectral_scale * (9 * eps)
  if (values < -negative_tolerance).any():
    raise ValueError("Indefinite RPV covariance")
  rank_tolerance = spectral_scale * (9 * torch.finfo(values.dtype).eps)
  positive = values > rank_tolerance
  rank = positive.sum(-1)
  coords = (vectors.mT @ standardized_error.unsqueeze(-1)).squeeze(-1)
  distance = (coords.square() / torch.where(positive, values, 1) * positive).sum(-1)
  null_error = coords.masked_fill(positive, 0).norm(dim=-1)
  violation = (null_error > 9 * eps * standardized_error.norm(dim=-1)) | (
    zero_variance & (error != 0)
  ).any(dim=-1)
  covered = (~violation) & (distance <= values.new_tensor(_CHI95)[rank])
  return distance, rank, violation, covered


class PosteriorBins:
  """Frame-weighted statistics by elapsed control steps, across surviving episodes."""

  def __init__(self, control_dt, bin_steps=50):
    if (
      not math.isfinite(control_dt)
      or control_dt <= 0
      or type(bin_steps) is not int
      or bin_steps < 1
    ):
      raise ValueError("Positive control_dt and bin_steps are required")
    self.dt, self.width = control_dt, bin_steps
    self.bins = {}

  def update(
    self,
    episode_id,
    time_index,
    error,
    covariance,
    jitter,
    innovation_norm,
    position_error_w,
    velocity_error_w,
  ):
    distance, rank, violation, covered = rpv_ellipsoid(error, covariance)
    features = torch.cat(
      (
        error.double().reshape(-1, 3, 3).square().sum(-1),
        torch.stack((distance, violation, covered, jitter > 0, innovation_norm), -1),
        position_error_w.double().square().sum(-1, keepdim=True),
        velocity_error_w.double().square().sum(-1, keepdim=True),
      ),
      -1,
    )
    indices = time_index // self.width
    for index in indices.unique().tolist():
      selection = indices == index
      if index not in self.bins:
        self.bins[index] = (
          set(),
          torch.zeros(10, dtype=torch.double),
          torch.zeros(10, dtype=torch.long),
        )
      ids, totals, ranks = self.bins[index]
      ids.update(episode_id[selection].tolist())
      totals += features[selection].sum(0).cpu()
      ranks += torch.bincount(rank[selection], minlength=10).cpu()

  def result(self):
    records = []
    for index, (ids, sums, ranks) in sorted(self.bins.items()):
      count = int(ranks.sum())
      mean = sums / count
      records.append(
        {
          "start_s": index * self.width * self.dt,
          "end_s": (index + 1) * self.width * self.dt,
          "surviving_episodes": len(ids),
          "frames": count,
          "rotation_rmse_rad": float(mean[0].sqrt()),
          "right_local_rho_rmse_m": float(mean[1].sqrt()),
          "position_rmse_m": float(mean[8].sqrt()),
          "right_local_nu_rmse_m_s": float(mean[2].sqrt()),
          "velocity_rmse_m_s": float(mean[9].sqrt()),
          "mean_supported_mahalanobis_squared": float(mean[3]),
          "null_direction_violations": int(sums[4]),
          "coverage_95": float(mean[5]),
          "jitter_frames": int(sums[6]),
          "mean_innovation_norm": float(mean[7]),
          "rank_counts": ranks.tolist(),
        }
      )
    return records


class EpisodeRecorder:
  """Store independent CPU pre-action samples and the reward for the same action."""

  def __init__(self, num_envs):
    self.frames = [defaultdict(list) for _ in range(num_envs)]
    self.priors = {}

  def before(self, policy, obs, actions, active):
    state = policy.provider.state
    command = policy.command
    rep = policy.wm.representation
    truth = LieState.from_full(obs["full_state"])
    reference = rep.reference_window(
      command.motion, command.time_steps, command.reference_alignment
    )
    goal, _, _ = build_motion_goal(
      RpvState(truth.rotation, truth.position, truth.velocity), reference
    )
    remaining = (
      command.motion.last_frame(command.time_steps) - command.time_steps + 1
    ) * rep.dt
    values = {
      "time_index": state.time_index,
      "full_state": obs["full_state"],
      "sensor": policy.history[:, -1],
      "action": actions,
      "sim_ctx": state.context,
      "true_goal": goal,
      "remaining_time": remaining,
    }
    rows = active.nonzero().flatten().tolist()
    for name, tensor in values.items():
      for row, value in zip(rows, tensor[active].detach().cpu(), strict=True):
        self.frames[row][name].append(value)
    if not self.priors:
      ids, prior = state.head_priors
      if not torch.equal(ids, state.episode_id) or state.time_index.any():
        raise ValueError("Diagnostic recording must begin at episode heads")
      means, covs = prior.mean.to_full().cpu(), prior.covariance.cpu()
      for row in rows:
        self.priors[row] = {
          "prior_mean": means[row],
          "prior_covariance": covs[row],
          "alignment_rotation": command.reference_alignment.rotation[row]
          .detach()
          .cpu()
          .clone(),
          "alignment_translation": command.reference_alignment.translation[row]
          .detach()
          .cpu()
          .clone(),
        }

  def after(self, reward, active):
    for row in active.nonzero().flatten().tolist():
      self.frames[row]["reward"].append(reward[row].detach().cpu().clone())

  def episodes(self, records):
    return [
      {
        **record,
        **self.priors[row],
        **{key: torch.stack(values) for key, values in self.frames[row].items()},
      }
      for row, record in enumerate(records)
    ]
