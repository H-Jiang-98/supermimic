"""FiLM identity, information permissions and differentiability for step 1."""

from __future__ import annotations

import io

import pytest
import torch
from tensordict import TensorDict

from supermimic.models import MultiEncoderAE
from supermimic.tasks.task_spec import TaskSpec


def _actor(uncertainty_dim: int = 0) -> MultiEncoderAE:
  def no_goal_extraction(*args):
    raise AssertionError("Network test should receive prepared inputs")

  return MultiEncoderAE(
    obs_dim=93,
    action_dim=29,
    task_specs=[TaskSpec("motion", 700, no_goal_extraction)],
    slot_mix={"motion": 1.0},
    latent_dim=32,
    hidden_dims=(64, 32),
    latent_bc_weight=0.0,
    history_length=5,
    uncertainty_dim=uncertainty_dim,
    input_version="eit_rpv_v1",
  )


def _obs(uncertainty: bool = True) -> TensorDict:
  fields = {
    "history": torch.randn(4, 5, 93),
    "goal_motion": torch.randn(4, 700),
  }
  if uncertainty:
    fields["task_uncertainty"] = torch.randn(4, 450)
  return TensorDict(fields, batch_size=[4])


def test_zero_film_and_copied_normalizers_equal_base_without_aliasing():
  torch.manual_seed(3)
  base, modulated = _actor(), _actor(450)
  obs = _obs()
  base.update_normalization(obs)
  missing, unexpected = modulated.load_state_dict(base.state_dict(), strict=False)
  assert not unexpected and all(key.startswith("film.") for key in missing)
  base.eval()
  modulated.eval()
  assert modulated.decoder[0].in_features == 497
  assert modulated.film is not None
  assert modulated.film[-1].out_features == 128
  assert torch.equal(base(obs), modulated(obs))
  common = ("obs_normalizer", "goal_normalizers", "encoders", "decoder")
  for name in common:
    left = getattr(base, name).state_dict()
    right = getattr(modulated, name).state_dict()
    for key in left:
      assert torch.equal(left[key], right[key])
      assert left[key].data_ptr() != right[key].data_ptr()
  with torch.no_grad():
    base.decoder[0].weight.add_(1.0)
  assert not torch.equal(base.decoder[0].weight, modulated.decoder[0].weight)


def test_disabled_film_does_not_read_or_register_uncertainty():
  actor = _actor().eval()
  obs = _obs(uncertainty=False)
  expected = actor(obs)
  obs["task_uncertainty"] = torch.full((4, 1), float("nan"))
  assert torch.equal(actor(obs), expected)
  assert actor.film is None
  assert not any(name.startswith("film.") for name in actor.state_dict())
  assert not any(name.startswith("film.") for name, _ in actor.named_parameters())


def test_film_staged_gradients_and_covariance_sensitivity():
  torch.manual_seed(11)
  actor = _actor(450).eval()
  assert actor.film is not None
  obs = _obs()
  uncertainty = obs["task_uncertainty"].requires_grad_()
  weights = torch.randn(4, 29)
  optimizer = torch.optim.SGD(actor.film.parameters(), lr=0.05)
  (actor(obs) * weights).sum().backward()
  last = actor.film[-1]
  assert torch.isfinite(last.weight.grad).all()
  assert last.weight.grad.abs().sum() > 0
  for layer in actor.film[:-1]:
    for parameter in layer.parameters():
      assert parameter.grad is not None
      assert torch.count_nonzero(parameter.grad) == 0
  assert torch.count_nonzero(uncertainty.grad) == 0

  optimizer.step()
  actor.zero_grad(set_to_none=True)
  uncertainty.grad = None
  actions = actor(obs)
  (actions * weights).sum().backward()
  assert torch.isfinite(uncertainty.grad).all()
  assert uncertainty.grad.abs().sum() > 0
  assert actor.film[0].weight.grad.abs().sum() > 0
  assert torch.isfinite(actor.film[0].weight.grad).all()
  changed = obs.clone()
  changed["task_uncertainty"] = -uncertainty.detach()
  assert not torch.allclose(actor(changed), actions)


def test_film_is_applied_after_first_layernorm_before_gelu():
  torch.manual_seed(1)
  actor = _actor(450).eval()
  assert actor.film is not None
  with torch.no_grad():
    actor.film[-1].bias.copy_(torch.linspace(-0.8, 0.5, 128))
  obs = _obs()
  goal = actor.goal_normalizers["motion"](obs["goal_motion"])
  latent = torch.tanh(actor.encoders["motion"](goal))
  history = actor.obs_normalizer(obs["history"]).flatten(1)
  u = actor.decoder[1](actor.decoder[0](torch.cat([latent, history], -1)))
  gamma, beta = actor.film(obs["task_uncertainty"]).chunk(2, -1)
  expected = actor.decoder[2:]((1.0 + gamma) * u + beta)
  assert torch.equal(actor(obs), expected)


def test_state_roundtrip_preserves_learned_film_and_normalizers():
  torch.manual_seed(19)
  actor = _actor(450)
  obs = _obs()
  actor.update_normalization(obs)
  assert actor.film is not None
  with torch.no_grad():
    actor.film[-1].weight.normal_(std=0.02)
  actor.eval()
  stream = io.BytesIO()
  torch.save(actor.state_dict(), stream)
  stream.seek(0)
  restored = _actor(450).eval()
  restored.load_state_dict(torch.load(stream, weights_only=True))
  assert torch.equal(actor(obs), restored(obs))


@pytest.mark.parametrize("bad_dim", [None, 449])
def test_enabled_film_requires_exact_covariance_features(bad_dim):
  actor = _actor(450)
  obs = _obs(uncertainty=False)
  if bad_dim is not None:
    obs["task_uncertainty"] = torch.zeros(4, bad_dim)
  with pytest.raises(ValueError, match="task_uncertainty"):
    actor(obs)


def test_history_shape_errors_before_flattening():
  actor = _actor()
  obs = _obs(False)
  obs["history"] = torch.randn(4, 1, 465)
  with pytest.raises(ValueError, match="history must have shape"):
    actor(obs)
