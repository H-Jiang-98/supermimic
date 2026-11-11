"""Translate TOML settings to typed command-line inputs."""

from dataclasses import fields, is_dataclass
from typing import Any


def config_args(
  values: dict[str, Any], defaults: Any, prefix: str = "", *, context: str = "training"
) -> list[str]:
  """Keep Tyro's type checking and reject unknown or recursive config keys."""
  if is_dataclass(defaults) and not isinstance(defaults, type):
    known = {field.name: getattr(defaults, field.name) for field in fields(defaults)}
  elif isinstance(defaults, dict):
    known = defaults
  else:
    raise ValueError(f"{prefix.rstrip('.')} is not a settings table")

  args: list[str] = []
  for key, value in values.items():
    name = f"{prefix}{key}"
    if key not in known or name == "config":
      raise ValueError(f"Unknown {context} setting: {name}")
    if isinstance(value, dict):
      args.extend(config_args(value, known[key], f"{name}.", context=context))
      continue
    if isinstance(value, list):
      value = repr(tuple(value) if isinstance(known[key], tuple) else value)
    elif isinstance(value, (str, bool, int, float)):
      value = str(value)
    else:
      raise ValueError(f"Unsupported value for {context} setting: {name}")
    args.extend([f"--{name.replace('_', '-')}", value])
  return args
