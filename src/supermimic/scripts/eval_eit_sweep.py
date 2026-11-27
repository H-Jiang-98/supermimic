"""Paired uncertainty sweeps on the frozen 50k nominal study."""

import argparse
import tomllib
from pathlib import Path

from filelock import FileLock

from supermimic.scripts.eval_eit_nominal import run
from supermimic.validation.config import atomic_json, file_digest
from supermimic.validation.nominal_report import load_manifest
from supermimic.validation.sweep_protocol import (
  CONDITIONS,
  condition_manifest,
  load_models,
)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("command", choices=("baseline", "run", "report"))
  parser.add_argument("--config", type=Path, default=Path("eval_eit_sweep_50k.toml"))
  parser.add_argument("--condition", choices=tuple(CONDITIONS), action="append")
  parser.add_argument(
    "--pilot",
    action="store_true",
    help="Exercise 32 actions of two clips and all noise levels in a separate directory",
  )
  parser.add_argument("--num-envs", type=int)
  parser.add_argument("--device")
  args = parser.parse_args()
  if (
    args.condition or args.pilot or args.num_envs or args.device
  ) and args.command != "run":
    parser.error("Execution options require run")
  root = args.config.resolve().parent
  with args.config.open("rb") as stream:
    config = tomllib.load(stream)
  study = config["study"]
  baseline = root / study["baseline"]
  base = load_manifest(baseline)
  if (
    base["identity"] != study["baseline_identity"]
    or set(config["checkpoints"]) != set(base["checkpoints"])
    or any(
      c["collected_steps"] != study["collected_steps"]
      or c["training_seed"] != study["training_seed"]
      for c in base["checkpoints"].values()
    )
  ):
    raise ValueError(
      "Baseline identity or checkpoint budget differs from the requested study"
    )
  output = root / study["output"]
  if args.command != "run":
    from supermimic.validation.sweep_report import build_report

    build_report(baseline, base, output, include_sweep=args.command == "report")
    return
  if args.pilot:
    output = output / "pilot"
  args.layout, args.no_compile = "mixed", False
  args.record_belief_posterior, args.audit_controls = True, args.pilot
  device = args.device or study["device"]
  capacity = args.num_envs if args.num_envs is not None else study["num_envs"]
  saved = load_models(
    {k: root / p for k, p in config["checkpoints"].items()}, base, device
  )
  for name in dict.fromkeys(args.condition or CONDITIONS):
    manifest = condition_manifest(base, name, args.pilot)
    paths = [root / study["motion_root"] / m["name"] for m in manifest["motions"]]
    for p, m in zip(paths, manifest["motions"], strict=True):
      if file_digest(p) != m["sha256"]:
        raise ValueError(f"Motion file differs from the nominal study: {p}")
    folder = output / name
    folder.mkdir(parents=True, exist_ok=True)
    with FileLock(str(folder / ".study.lock"), timeout=0):
      path = folder / "manifest.json"
      if (
        path.exists()
        and load_manifest(folder, allow_pending=True)["identity"]
        != manifest["identity"]
      ):
        raise ValueError(f"Conflicting condition in {folder}")
      atomic_json(path, manifest)
      print(
        f"{name}: {len(paths)} clips x {len(manifest['seeds'])} seeds x {len(manifest['routes'])} modes",
        flush=True,
      )
      run(args, paths, manifest, saved, device, capacity, folder)
  if not args.pilot and not args.condition:
    from supermimic.validation.sweep_report import build_report

    build_report(baseline, base, output, include_sweep=True)


if __name__ == "__main__":
  main()
