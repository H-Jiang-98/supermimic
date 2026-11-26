"""Short four-method CUDA training, common-initialization and resume check."""

import argparse
import sys
from pathlib import Path
from unittest.mock import patch

import torch

from supermimic.eit_methods import BASELINES, LOCAL_METHODS
from supermimic.local_control import LOCAL_DIM, VECH_DIM
from supermimic.runners.eit_training import initialization_fingerprint
from supermimic.scripts import train_eit
from supermimic.scripts.check_estimator import stationary_motion


def check_film(actor, method):
  """Probe FiLM itself, including latent independently of the decoder path."""
  parameter = next(actor.parameters())
  width = actor.condition_normalizer._mean.shape[-1]
  condition = parameter.new_zeros((2, width + 29)).requires_grad_()
  history = parameter.new_zeros((2, 5, 93))
  goal = parameter.new_ones((2, 730))
  latent = actor.encode_goal(goal)
  assert not actor.film[-1].weight.any() and not actor.film[-1].bias.any()
  with torch.no_grad():
    actor.film[-1].weight.normal_(std=0.02)
    actor.condition_normalizer._mean.fill_(0.125)
  _, gamma, beta = actor.decode_goal(latent, history, condition)
  dz, dc = torch.autograd.grad((gamma + beta).sum(), (latent, condition))
  y, error, context, covariance, _ = dc.split(
    [LOCAL_DIM, LOCAL_DIM, width - 2 * LOCAL_DIM - VECH_DIM, VECH_DIM, 29], -1
  )
  assert dz.any() and context.any()
  assert bool(y.any()) == (method != "prop_kp")
  assert bool(error.any()) == (method != "prop_kp")
  assert bool(covariance.any()) == (method == "belief")


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  root = args.output.resolve()
  root.mkdir(parents=True, exist_ok=False)
  motion = root / "motion.npz"
  stationary_motion(motion, frames=64)
  init = root / "init.pt"

  def run(*flags):
    argv = [
      "train_eit",
      "--motion-file",
      str(motion),
      "--device",
      "cuda:0",
      "--eager",
      "--seed",
      "2027",
      *flags,
    ]
    with patch.object(sys, "argv", argv):
      train_eit.main()

  options = """
    --num-envs 4 --buffer-size 8 --replay-ratio 1 --mean-updates 1
    --mean-batch-size 2 --mean-window-frames 3 --cov-batch-size 4
    --cov-validation-pairs 4 --validation-stride 2 --validation-batch-size 2
    --policy-batch-size 2 --policy-microbatch-size 2 --policy-horizon 2
    --critic-epochs 1 --critic-minibatches 1 --mean-hidden-dims 12
    --cov-hidden-dims 8 --actor-hidden-dims 12 --critic-hidden-dims 12
  """.split()
  run(
    "--group", "gt", "--prepare-initialization", "--initialization", str(init), *options
  )
  run(
    "--group",
    "mean",
    "--prepare-initialization",
    "--initialization",
    str(init),
    *options,
  )
  bundle = torch.load(init, map_location="cpu", weights_only=True)
  legacy = dict(bundle, version="eit_initialization_v5", generation_group="mean")
  for section in ("actors", "critics"):
    legacy[section] = {key: bundle[section][key] for key in ("mean", "belief")}
  legacy_path = root / "legacy-init.pt"
  torch.save(legacy, legacy_path)
  run("--prepare-initialization", "--initialization", str(legacy_path), *options)
  try:
    run(
      "--group",
      "prop_kp",
      "--prepare-initialization",
      "--initialization",
      str(legacy_path),
      *options,
    )
  except ValueError as error:
    if "four-method initialization" not in str(error):
      raise
  else:
    raise AssertionError("Baseline preparation accepted a two-method bundle")
  digest = initialization_fingerprint(bundle)
  full = bundle["critics"]["mean"]["state_dict"]
  load = train_eit.EITTraining.load_initialization

  def probe(training, initial):
    # Probe before loading, so the actual training always starts from the
    # unmodified zero-initialized FiLM. Preserve random streams for training.
    with torch.random.fork_rng(devices=[0]):
      check_film(training.actor, training.runner.cfg.estimator_inputs.group)
    load(training, initial)
    assert initialization_fingerprint(
      training.runner._wm.bundle(), include_model_revision=False
    ) == initialization_fingerprint(bundle["world_model"])
    assert initialization_fingerprint(
      training.actor.state_dict()
    ) == initialization_fingerprint(bundle["actors"]["mean"]["actor"])

  for group in LOCAL_METHODS:
    critic = bundle["critics"][group]
    width = critic["contract"]["input_dim"]
    assert full["mlp.0.weight"].shape[1] - width == (
      LOCAL_DIM + VECH_DIM if group in BASELINES else 0
    )
    for name, weight in critic["state_dict"].items():
      expected = full[name][:, :width] if name == "mlp.0.weight" else full[name]
      torch.testing.assert_close(weight, expected, atol=0, rtol=0)
    out = root / group
    checkpoint = out / "training.pt"
    with patch.object(train_eit.EITTraining, "load_initialization", probe):
      run(
        "--group",
        group,
        "--residual-actor",
        "--output",
        str(out),
        "--max-iterations",
        "5",
        "--initialization",
        str(init),
        *options,
      )
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    run("--output", str(out), "--resume", str(checkpoint), "--max-iterations", "10")
    resumed = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert resumed["initialization_digest"] == saved["initialization_digest"] == digest
    cfg = resumed["runner_config"]["estimator_inputs"]
    assert (cfg["group"], cfg["residual_actor"]) == (group, True)
    assert saved["collected_steps"] == 5 and resumed["collected_steps"] == 10
    for name in ("actor_steps", "critic_steps"):
      assert resumed["policy"][name] > saved["policy"][name] > 0
    for name in ("mean", "cov"):
      assert resumed["learner"][name]["steps"] > saved["learner"][name]["steps"] > 0
    for name in ("actor", "critic", "world_model", "learner", "policy"):
      for value in torch.utils._pytree.tree_leaves(resumed[name]):
        if isinstance(value, torch.Tensor) and value.is_floating_point():
          assert value.isfinite().all(), (group, name)
    print(
      f"PASS {group} FiLM, initialization {digest}, "
      f"actor steps {saved['policy']['actor_steps']} -> {resumed['policy']['actor_steps']}, "
      f"critic steps {saved['policy']['critic_steps']} -> {resumed['policy']['critic_steps']}, "
      f"finite model and optimizer state",
      flush=True,
    )


if __name__ == "__main__":
  main()
