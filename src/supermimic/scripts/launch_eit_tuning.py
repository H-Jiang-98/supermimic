"""Train one method C configuration per physical GPU."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING or __package__:
  from . import launch_eit
else:  # Support running the file directly before activating the training environment.
  import launch_eit


WIDTH_OPTIONS = {
  "hidden_dims",
  "mean_hidden_dims",
  "cov_hidden_dims",
  "actor_hidden_dims",
  "critic_hidden_dims",
}


def _trial_value(value, field: str, index: int, count: int):
  if isinstance(value, dict):
    return {
      key: _trial_value(item, f"{field}.{key}", index, count)
      for key, item in value.items()
    }
  widths = field.removeprefix("training.") in WIDTH_OPTIONS
  if widths and (not isinstance(value, list) or not value):
    raise ValueError(f"{field} requires a nonempty list of hidden widths")
  if not isinstance(value, list):
    return value
  if widths and all(type(item) is int for item in value):
    selected = value  # A flat network shape is shared, even if len(shape) == count.
  else:
    if len(value) != count:
      raise ValueError(
        f"{field} requires {count} entries for gpu_index, got {len(value)}"
      )
    selected = value[index]
  if widths:
    if (
      not isinstance(selected, list)
      or not selected
      or any(type(width) is not int or width < 1 for width in selected)
    ):
      raise ValueError(f"{field} requires positive integer widths for each network")
  elif isinstance(selected, (list, dict)):
    raise ValueError(f"{field} requires one scalar per GPU")
  return selected


def plan(config: dict, base: Path):
  required = {"gpu_index", "paths", "training"}
  if not required <= config.keys() or config.keys() - required - {
    "seed",
    "env",
    "wm validation",
  }:
    raise ValueError(
      'Expected gpu_index, [paths], [training] and optional seed, [env.sim.mujoco], ["wm validation"]'
    )
  gpus = launch_eit._indices(config["gpu_index"], "gpu_index")
  sections = {
    k: v for k, v in config.items() if k in ("training", "env", "wm validation")
  }
  if not all(
    isinstance(table, dict) for table in [config["paths"], *sections.values()]
  ):
    raise ValueError("Expected configuration tables for paths and training settings")
  if {"hidden_dims", "mean_hidden_dims"} <= sections["training"].keys():
    raise ValueError("hidden_dims and mean_hidden_dims specify the same network")

  environment, motion, output = launch_eit.resolve_paths(config["paths"], base)
  queues = {}
  for index, gpu in enumerate(gpus):
    settings = {
      name: _trial_value(table, name, index, len(gpus))
      for name, table in sections.items()
    }
    command = launch_eit.worker_command(settings, environment, motion)
    seed = _trial_value(config.get("seed", 2027), "seed", index, len(gpus))
    if type(seed) is not int or seed < 0:
      raise ValueError("seed must be a nonnegative integer for each GPU")
    trial = f"trial_{index + 1:03d}"
    # train_eit creates its initialization in the output's parent directory.
    queues[gpu] = [
      [
        *command,
        "--group",
        "C",
        "--seed",
        str(seed),
        "--run-name",
        f"eit-C-{trial}-seed{seed}",
        "--output",
        str(output / trial / f"seed{seed}" / "C"),
      ]
    ]
  return environment, motion, (queues,)


def main():
  launch_eit.run_cli(plan, "train_eit_tuning.toml", __doc__, wait_for_all=True)


if __name__ == "__main__":
  main()
