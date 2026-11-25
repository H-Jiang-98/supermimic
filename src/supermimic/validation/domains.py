"""Sobol samples in event parameter space, with physical read-back checks."""

import re

import torch
from mjlab.managers.event_manager import RecomputeLevel


def domain_table(env_cfg, count, seed):
  events = env_cfg.events
  friction = events["foot_friction"].params
  com = events["base_com"].params
  bias = events["encoder_bias"].params
  if friction.get("operation") != "abs" or com.get("operation") != "add":
    raise ValueError("Unsupported domain randomization operation")
  if any(p.get("distribution", "uniform") != "uniform" for p in (friction, com, bias)):
    raise ValueError("Sobol protocol requires uniform randomization ranges")
  fr = friction["ranges"]
  ranges = [fr[0] if isinstance(fr, dict) else fr]
  ranges += [com["ranges"][i] for i in range(3)]
  ranges += [bias["bias_range"]] * 29
  bounds = torch.tensor(ranges, dtype=torch.float64)
  if not torch.isfinite(bounds).all() or (bounds[:, 1] < bounds[:, 0]).any():
    raise ValueError("Invalid domain bounds")
  unit = torch.quasirandom.SobolEngine(33, scramble=True, seed=seed).draw_base2(
    (count - 1).bit_length(), dtype=torch.float64
  )[:count]
  samples = bounds[:, 0] + unit * (bounds[:, 1] - bounds[:, 0])
  names = ["foot_friction", "base_com_dx", "base_com_dy", "base_com_dz"] + [
    f"encoder_bias_{i}" for i in range(29)
  ]
  return dict(
    columns=names,
    ranges=bounds.tolist(),
    unit=unit.tolist(),
    values=samples.tolist(),
    seed=seed,
    scramble=True,
  )


def apply_domains(env, sim_context, values):
  """Startup events allocate model fields; overwrite samples before final reset."""
  robot = env.scene["robot"]
  values = torch.as_tensor(values, device=env.device, dtype=torch.float32)
  if values.shape != (env.num_envs, 33):
    raise ValueError("Expected one 33D domain per environment")
  cfg = env.cfg.events["foot_friction"].params["asset_cfg"]
  pattern = cfg.geom_names
  local = [
    i
    for i, name in enumerate(robot.geom_names)
    if (re.match(pattern, name) if isinstance(pattern, str) else name in pattern)
  ]
  if not local:
    raise ValueError("No foot geoms matched")
  ids = robot.indexing.geom_ids[local]
  env.sim.model.geom_friction[:, ids, 0] = values[:, :1]
  body_name = env.cfg.events["base_com"].params["asset_cfg"].body_names[0]
  body = int(robot.indexing.body_ids[robot.body_names.index(body_name)])
  nominal = torch.tensor(
    env.sim.mj_model.body_ipos[body].copy(), device=env.device, dtype=values.dtype
  )
  env.sim.model.body_ipos[:, body, :] = nominal + values[:, 1:4]
  robot.data.encoder_bias[:] = values[:, 4:]
  env.sim.recompute_constants(RecomputeLevel.set_const)
  expected_raw = torch.cat((values[:, :1], nominal + values[:, 1:4], values[:, 4:]), -1)
  actual = sim_context.read(env)
  expected = (expected_raw - sim_context.mean) / sim_context.scale
  torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
  return actual.detach().cpu().numpy()
