import torch
import torch.nn as nn
from tensordict import TensorDict

from supermimic.learners.learner_utils import (
  CapturedUpdate,
  _optimize_step,
  build_optimizers,
)
from supermimic.models import build_mlp


def _ids(params) -> list[int]:
  return [id(p) for p in params]


def test_muon_steps_the_hidden_matrices_and_adam_the_rest():
  torch.manual_seed(0)
  mlp = build_mlp(5, (8, 8, 8), 3)
  linears = [m for m in mlp if isinstance(m, nn.Linear)]
  hidden = [linears[1].weight, linears[2].weight]
  adam, muon = build_optimizers([mlp], 1e-3)
  assert isinstance(muon, torch.optim.Muon)
  assert _ids(muon.param_groups[0]["params"]) == _ids(hidden)
  rest = [p for p in mlp.parameters() if id(p) not in _ids(hidden)]
  assert _ids(adam.param_groups[0]["params"]) == _ids(rest)
  before = [p.detach().clone() for p in mlp.parameters()]
  _optimize_step([adam, muon], mlp(torch.randn(4, 5)).square().mean())
  for param, old in zip(mlp.parameters(), before, strict=True):
    assert not torch.equal(param, old)


def test_disabled_capture_runs_each_update_without_graph_state():
  weight = nn.Parameter(torch.tensor(1.0))
  optimizer = torch.optim.SGD([weight], lr=0.1)

  def update(inputs):
    loss = (weight * inputs["x"]).square().mean()
    _optimize_step([optimizer], loss)
    return {"loss": loss.detach()}

  def state():
    raise AssertionError("disabled capture must not inspect CUDA graph state")

  captured = CapturedUpdate(update, state, warmup=0, enabled=False)
  losses = [
    captured(TensorDict({"x": torch.ones(2)}, batch_size=[2]))["loss"].item()
    for _ in range(5)
  ]
  assert all(
    after < before for before, after in zip(losses[:-1], losses[1:], strict=True)
  )
  assert weight.item() < 1.0


def test_critic_fit_uses_one_permutation_unweighted_mse_and_small_batches(monkeypatch):
  from unittest.mock import Mock

  from supermimic.learners.learner_utils import fit_critic

  critic = nn.Linear(1, 1, bias=False)
  critic.weight.data.zero_()
  optimizer = torch.optim.SGD(critic.parameters(), lr=0.1)
  random = Mock(wraps=torch.randperm)
  monkeypatch.setattr(torch, "randperm", random)
  features, targets = torch.ones(2, 1), torch.tensor([1.0, 3.0])
  # One logical minibatch has the plain arithmetic mean gradient: -4.
  loss, _, steps = fit_critic(
    critic, [optimizer], features, targets, epochs=1, minibatches=1, max_norm=100
  )
  torch.testing.assert_close(loss, torch.tensor(5.0))
  torch.testing.assert_close(critic.weight, torch.tensor([[0.4]]))
  random.reset_mock()
  _, _, steps = fit_critic(
    critic, [optimizer], features, targets, epochs=5, minibatches=4
  )
  assert steps == 10 and random.call_count == 1
