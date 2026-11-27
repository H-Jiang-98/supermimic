"""Versioned Belief-in-Loop state samples and sparse right-local projections."""

# PyTorch dynamically re-exports public tensor factories and operations.
# pyright: reportPrivateImportUsage=false

from __future__ import annotations

import zipfile

import numpy as np

from supermimic.validation.nominal import atomic_npz, tracking_digest

VERSION = "nominal_belief_posterior_v1"
MODE = "Belief-in-Loop"
SINGULAR_COS = 1e-6
CONVENTION = (
  "SI; rotation maps body to world; ZYX Euler; gravity=-R[2,:]; "
  "basis u=normalize(g cross least-aligned Cartesian axis), v=g cross u; "
  "covariance lower triangle in row order (world height,u,v,body vx,vy,vz); "
  "Euler marginal variances in rad^2; final state is post-action and pre-reset"
)
FIELDS = {
  "truth_position": (3,),
  "truth_rotation": (3, 3),
  "truth_velocity": (3,),
  "mean_height": (1,),
  "mean_rotation": (3, 3),
  "mean_velocity": (3,),
  "intrinsic_cov_lower": (21,),
  "euler_variance": (2,),
}


def posterior_path(output, scenario):
  return output / "posteriors" / scenario.clip / f"{scenario.seed}.npz"


def posterior_sample(truth, belief):
  import torch

  from supermimic.utils.lie_state import skew

  mean = belief.mean
  rotation = mean.rotation.double()
  gravity = -rotation[..., 2, :]
  axis = torch.eye(3, dtype=rotation.dtype, device=rotation.device)[
    gravity.abs().argmin(-1)
  ]
  u = torch.nn.functional.normalize(torch.linalg.cross(gravity, axis), dim=-1)
  v = torch.linalg.cross(gravity, u)
  basis = torch.stack((u, v), -2)
  # The first nine rows of a lower-triangular factor have support in columns 0:9.
  lower = belief.covariance_sqrt[..., :9, :9].double()
  factor = torch.cat(
    (
      -gravity.unsqueeze(-2) @ lower[..., 3:6, :],
      basis @ skew(gravity) @ lower[..., :3, :],
      lower[..., 6:9, :],
    ),
    -2,
  )
  covariance = factor @ factor.mT
  i, j = torch.tril_indices(6, 6, device=rotation.device)
  # Body angular perturbation -> ZYX roll/pitch. Both charts lose a
  # differentiable roll/pitch pair at vertical pitch.
  a, b, c = rotation[..., 2, :].unbind(-1)
  horizontal = torch.hypot(b, c)
  h = horizontal.clamp_min(SINGULAR_COS)
  sin_roll, cos_roll = b / h, c / h
  tan_pitch = -a / h
  jacobian = torch.stack(
    (
      torch.stack((torch.ones_like(a), sin_roll * tan_pitch, cos_roll * tan_pitch), -1),
      torch.stack((torch.zeros_like(a), cos_roll, -sin_roll), -1),
    ),
    -2,
  )
  variances = (jacobian @ lower[..., :3, :]).square().sum(-1)
  variances = variances.masked_fill(horizontal[..., None] < SINGULAR_COS, float("nan"))
  return torch.cat(
    (
      truth.position,
      truth.rotation.flatten(-2),
      truth.velocity,
      mean.position[..., 2:],
      mean.rotation.flatten(-2),
      mean.velocity,
      covariance[..., i, j],
      variances,
    ),
    -1,
  )


class PosteriorBuffer:
  """Transfer fixed-size time chunks. Only executed prefixes are written to disk."""

  def __init__(self, plans):
    width = sum(int(np.prod(shape)) for shape in FIELDS.values())
    self.rows = [np.empty((n + 1, width), dtype=np.float64) for n in plans]
    self.pending = []
    self.start = 0

  def append(self, sample):
    self.pending.append(sample)
    if len(self.pending) == 128:
      self.flush()

  def flush(self):
    import torch

    if self.pending:
      block = torch.stack(self.pending, dim=1).cpu().numpy()
      for row, values in zip(self.rows, block, strict=True):
        dest = row[self.start : self.start + len(values)]
        dest[:] = values[: len(dest)]
      self.start += block.shape[1]
      self.pending.clear()

  def episode(self, i, executed):
    self.flush()
    if self.start < executed + 1:
      raise ValueError("Missing terminal posterior sample")
    values = self.rows[i][: executed + 1]
    fields = {}
    offset = 0
    for name, shape in FIELDS.items():
      width = int(np.prod(shape))
      fields[name] = values[:, offset : offset + width].reshape(-1, *shape)
      if name not in ("intrinsic_cov_lower", "euler_variance"):
        fields[name] = fields[name].astype(np.float32)
      offset += width
    return fields


def metadata(manifest, scenario, status, frames):
  return dict(
    version=VERSION,
    convention=CONVENTION,
    identity=manifest["identity"],
    checkpoint_sha256=manifest["checkpoints"]["belief"]["sha256"],
    clip=scenario.clip,
    mode=scenario.mode,
    seed=scenario.seed,
    status=status,
    tracking_sha256=tracking_digest(frames),
  )


def save_posterior(output, manifest, scenario, status, frames, dt, fields):
  atomic_npz(
    posterior_path(output, scenario),
    **metadata(manifest, scenario, status, frames),
    dt=dt,
    step=np.arange(len(frames) + 1, dtype=np.int32),
    **fields,
  )


def intrinsic_cholesky(packed):
  i, j = np.tril_indices(6)
  covariance = np.empty((len(packed), 6, 6))
  covariance[:, i, j] = covariance[:, j, i] = packed
  return np.linalg.cholesky(covariance)


def load_posterior(output, manifest, scenario, status, frames):
  """Verify provenance, state indices and shape before accepting paired artifacts."""
  n = len(frames) + 1
  try:
    with np.load(posterior_path(output, scenario), allow_pickle=False) as z:
      for key, expected in metadata(manifest, scenario, status, frames).items():
        if z[key].item() != expected:
          raise ValueError(f"{key} mismatch")
      data = {k: z[k] for k in (*FIELDS, "dt", "step")}
  except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as e:
    raise ValueError(f"{scenario}: {e}") from e
  if (
    not np.array_equal(data["step"], np.arange(n))
    or data["dt"].shape != ()
    or not np.isfinite(data["dt"])
    or data["dt"] <= 0
  ):
    raise ValueError(f"{scenario}: invalid state timebase")
  for key, shape in FIELDS.items():
    if data[key].dtype.kind != "f" or data[key].shape != (n, *shape):
      raise ValueError(f"{scenario}: invalid {key} dtype or shape")
    if key != "euler_variance" and not np.isfinite(data[key]).all():
      raise ValueError(f"{scenario}: nonfinite {key}")
  for key in ("truth_rotation", "mean_rotation"):
    # Match the float-precision SO(3) checks used by LieState validation.
    matrix = data[key].astype(float)
    tolerance = 128 * np.finfo(data[key].dtype).eps
    gram = matrix.swapaxes(-1, -2) @ matrix
    if not np.allclose(gram, np.eye(3), atol=tolerance, rtol=tolerance) or np.any(
      np.linalg.det(matrix) <= 0
    ):
      raise ValueError(f"{scenario}: {key} is not a proper rotation")
  h = np.linalg.norm(data["mean_rotation"][:, 2, 1:].astype(float), axis=-1)
  ev = data["euler_variance"]
  if not (
    np.isnan(ev[h < SINGULAR_COS]).all()
    and np.isfinite(ev[h >= SINGULAR_COS]).all()
    and (ev[h >= SINGULAR_COS] >= 0).all()
  ):
    raise ValueError(f"{scenario}: invalid Euler variances")
  try:
    intrinsic_cholesky(data["intrinsic_cov_lower"])
  except np.linalg.LinAlgError as e:
    raise ValueError(
      f"{scenario}: intrinsic covariance is not positive definite"
    ) from e
  return data
