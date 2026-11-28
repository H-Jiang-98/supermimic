"""Collector-owned v8 random streams; packet scheduling stays out of imagination."""

import torch


class CollectionObservations:
  def __init__(self, config, device, seed):
    self.config = config
    self.generators = {}
    for offset, name in enumerate(("heading", "gt"), 101):
      generator = torch.Generator(device=device)
      generator.manual_seed((seed + offset) % (2**63 - 1))
      self.generators[name] = generator
    self.rate = self.phase = None

  def packet(self, truth, reset, time_index, dt):
    cfg = self.config
    shape = truth.position.shape[:-1]
    # Fixed-size device draws avoid nonzero() and host reads in the scheduler.
    random = torch.rand(
      *shape,
      3,
      device=truth.position.device,
      dtype=truth.position.dtype,
      generator=self.generators["heading"],
    )
    rate = cfg.heading_rate_hz[0] + random[..., 0] * (
      cfg.heading_rate_hz[1] - cfg.heading_rate_hz[0]
    )
    rate = torch.where(random[..., 1] < cfg.heading_dropout_probability, 0, rate)
    if self.rate is None:
      self.rate, self.phase = rate, random[..., 2]
    else:
      assert self.phase is not None
      self.rate = torch.where(reset, rate, self.rate)
      self.phase = torch.where(reset, random[..., 2], self.phase)
    now = time_index * dt * self.rate + self.phase
    before = (time_index - 1) * dt * self.rate + self.phase
    valid = (self.rate > 0) & (now.floor() > before.floor())
    noise = torch.randn(
      *shape,
      device=random.device,
      dtype=random.dtype,
      generator=self.generators["heading"],
    )
    yaw = torch.atan2(truth.rotation[..., 1, 0], truth.rotation[..., 0, 0])
    yaw = yaw + cfg.heading_std * noise
    yaw = torch.atan2(yaw.sin(), yaw.cos())
    return torch.where(valid, yaw, 0), valid

  def gt_noise(self, reference):
    return torch.randn(
      *reference.shape[:-1],
      9,
      device=reference.device,
      dtype=reference.dtype,
      generator=self.generators["gt"],
    )

  def state_dict(self):
    return {name: generator.get_state() for name, generator in self.generators.items()}

  def validate_state_dict(self, state):
    if not isinstance(state, dict) or state.keys() != self.generators.keys():
      raise ValueError("Collection RNG streams mismatch")
    for name, value in state.items():
      if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.uint8
        or value.ndim != 1
      ):
        raise ValueError("Invalid collection RNG state")
      torch.Generator(device=self.generators[name].device).set_state(value.cpu())

  def load_state_dict(self, state):
    self.validate_state_dict(state)
    for name, value in state.items():
      self.generators[name].set_state(value.cpu())
    # A checkpoint starts new physical episodes, like the replay and provider.
    self.rate = self.phase = None
