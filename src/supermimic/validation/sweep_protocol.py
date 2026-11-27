"""Frozen conditions and checkpoint verification for the 50k uncertainty sweep."""

from supermimic.validation.config import digest
from supermimic.validation.nominal import ROUTES

CONDITIONS = {
  "prior-2": (2, 1),
  "prior-4": (4, 1),
  "sensor-2": (1, 2),
  "sensor-4": (1, 4),
}


def condition_manifest(base, name, pilot=False):
  kb, ko = CONDITIONS[name]
  manifest = {k: v for k, v in base.items() if k not in ("identity", "source_runs")}
  manifest.update(
    protocol="eit_uncertainty_sweep_v1",
    baseline_identity=base["identity"],
    routes={mode: route for mode, route in ROUTES.items() if kb == 1 or route[1]},
    kappa_b=kb,
    kappa_o=ko,
    pilot=pilot,
    record_belief_posterior=True,
  )
  if pilot:
    subset = {"walk1_subject1.npz", "fallAndGetUp1_subject1.npz"}
    manifest["motions"] = [
      dict(m, steps=min(32, m["steps"])) for m in base["motions"] if m["name"] in subset
    ]
    manifest["seeds"] = base["seeds"][:1]
    if len(manifest["motions"]) != 2:
      raise ValueError("Pilot requires both the walk and fall/get-up clips")
  manifest["identity"] = digest(manifest)
  return manifest


def load_models(config, base, device):
  from supermimic.runners.eit_inference import evaluation_config, load_hashed_checkpoint

  saved = {}
  for key, path in config.items():
    checkpoint, sha = load_hashed_checkpoint(path, device)
    expected = base["checkpoints"][key]
    inputs = evaluation_config(checkpoint).estimator_inputs
    if (
      sha != expected["sha256"]
      or checkpoint["collected_steps"] != expected["collected_steps"]
      or checkpoint["training_seed"] != expected["training_seed"]
      or inputs is None
      or inputs.group != key
      or not inputs.residual_actor
    ):
      raise ValueError(f"Checkpoint does not match the frozen nominal study: {path}")
    saved[key] = checkpoint
  return saved
