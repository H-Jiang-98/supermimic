"""Standalone CPU prototype for matrix integration and vmap(jacrev).

Run with ``OMP_NUM_THREADS=1 python -m supermimic.scripts.check_matrix_jacobian``.
Assertions check values and outer gradients; stdout is a JSON test record.
The independent prototype is retained alongside the production matrix backend;
the quaternion reference explicitly calls the original integration adapter.
"""

import json
import time
from dataclasses import fields, replace

import torch
from torch.func import jacrev, vmap

from supermimic.belief import BeliefState
from supermimic.estimator import FilterConfig, LieBeliefFilter, posterior_nll
from supermimic.estimator_inputs import GaussianMeasurementConfig
from supermimic.models import Dynamics, ProcessNoiseConfig
from supermimic.representations import GenCoordRepresentation
from supermimic.utils.lie_state import LieState, boxplus, local_error, so3_exp


def mv(matrix, vector):
  return (matrix @ vector.unsqueeze(-1)).squeeze(-1)


def state_fields(state):
  return tuple(getattr(state, f.name) for f in fields(state))


def matrix_rep(state):
  return torch.cat(
    (
      state.position[..., 2:3],
      -state.rotation[..., 2, :],
      mv(state.rotation.mT, state.velocity),
      state.omega,
      state.joint_pos,
      state.joint_vel,
    ),
    dim=-1,
  )


def matrix_step(model, state, action, context, coefficient):
  inputs = model._normalized_inputs(matrix_rep(state), action, context)
  prediction = model.mlp(inputs)
  dv, dw, dj = prediction.split((3, 3, 29), dim=-1)
  increment = so3_exp(model.dt * (state.omega + coefficient * dw))
  return LieState(
    state.rotation @ increment,
    state.position + model.dt * (state.velocity + mv(state.rotation, coefficient * dv)),
    state.velocity + mv(state.rotation, dv),
    mv(increment.mT, state.omega + dw),
    state.joint_pos + model.dt * (state.joint_vel + coefficient * dj),
    state.joint_vel + dj,
  )


def matrix_factor(model, state, action, context):
  values = model.noise_head(
    model._normalized_inputs(matrix_rep(state), action, context)
  )
  values = torch.where(
    model._noise_diagonal,
    torch.nn.functional.softplus(values) + model.process_noise_config.diagonal_floor,
    values,
  )
  return (
    values.new_zeros(*values.shape[:-1], 4900)
    .scatter(-1, model._noise_indices.expand_as(values), values)
    .reshape(*values.shape[:-1], 70, 70)
  )


class MatrixFilter(LieBeliefFilter):
  def __init__(self, model, coefficient, backend):
    super().__init__(
      model, GaussianMeasurementConfig(), model.action_offset.new_zeros(29)
    )
    self.coefficient, self.backend = coefficient, backend

  def mean_step(self, state, action, context):
    return matrix_step(self.dynamics, state, action, context, self.coefficient)

  def transition_jacobian(self, mean, action, context, nominal=None):
    if torch.is_inference_mode_enabled():
      raise RuntimeError("Matrix Jacobians require local autograd; use no_grad()")
    if self.backend == "eager":
      nominal = self.mean_step(mean, action, context)

      def residual(epsilon):
        return local_error(
          nominal, self.mean_step(boxplus(mean, epsilon), action, context)
        )

      return torch.autograd.functional.jacobian(
        residual, action.new_zeros(70), create_graph=torch.is_grad_enabled()
      )

    def residual(
      epsilon, rotation, position, velocity, omega, joint_pos, joint_vel, a, ctx
    ):
      state = LieState(rotation, position, velocity, omega, joint_pos, joint_vel)
      nominal = self.mean_step(state, a, ctx)
      return local_error(nominal, self.mean_step(boxplus(state, epsilon), a, ctx))

    return vmap(jacrev(residual, argnums=0))(
      action.new_zeros(*action.shape[:-1], 70), *state_fields(mean), action, context
    )

  def predict(self, belief, action, context):
    mean = self.mean_step(belief.mean, action, context)
    factor = matrix_factor(self.dynamics, belief.mean, action, context)
    jacobian = self.transition_jacobian(belief.mean, action, context)
    covariance = jacobian @ belief.covariance @ jacobian.mT + factor @ factor.mT
    return BeliefState(mean, (covariance + covariance.mT) * 0.5)


class QuaternionReference(torch.nn.Module):
  """Explicit old adapter, unaffected by the production matrix default."""

  def __init__(self, model):
    super().__init__()
    self.model = model

  def mean_lie(self, state, action, context):
    aux, rep = state.to_rep()
    inputs = self.model._normalized_inputs(rep, action, context)
    return LieState.from_rep(
      *self.model._apply_prediction(aux, rep, self.model.mlp(inputs), self.model.dt)
    )

  def distribution_lie(self, state, action, context):
    _, rep = state.to_rep()
    inputs = self.model._normalized_inputs(rep, action, context)
    return self.mean_lie(state, action, context), self.model._noise_factor(inputs)


def fixture(batch, dtype=torch.float64):
  rep = GenCoordRepresentation(29, torch.zeros(29, dtype=dtype))
  rep.dt, rep.substep_coeff = 0.02, 0.6
  model = (
    Dynamics(
      rep,
      29,
      2,
      torch.full((29,), 0.5, dtype=dtype),
      torch.zeros(29, dtype=dtype),
      (16,),
      process_noise=ProcessNoiseConfig(hidden_dims=(12,)),
    )
    .to(dtype=dtype)
    .eval()
  )
  with torch.no_grad():
    model.noise_head[-1].weight.normal_(std=0.001)
  state = LieState(
    so3_exp(torch.randn(batch, 3, dtype=dtype) * 0.4),
    torch.randn(batch, 3, dtype=dtype) * 0.1,
    torch.randn(batch, 3, dtype=dtype) * 0.2,
    torch.randn(batch, 3, dtype=dtype) * 0.1,
    torch.randn(batch, 29, dtype=dtype) * 0.1,
    torch.randn(batch, 29, dtype=dtype) * 0.1,
  )
  return (
    model,
    rep,
    state,
    torch.randn(batch, 29, dtype=dtype) * 0.1,
    torch.randn(batch, 2, dtype=dtype) * 0.1,
  )


def max_error(actual, expected, *, atol=1e-8, rtol=1e-7):
  torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
  return float((actual - expected).abs().max().detach())


def finite_difference(objective, parameter, gradient, epsilon=2e-6):
  assert torch.isfinite(gradient).all() and gradient.norm() > 1e-10
  direction = gradient.detach() / gradient.detach().norm()
  original = parameter.detach().clone()
  values = []
  try:
    for sign in (1, -1):
      with torch.no_grad():
        parameter.copy_(original + sign * epsilon * direction)
        values.append(objective().detach())
  finally:
    with torch.no_grad():
      parameter.copy_(original)
  ad, fd = (gradient * direction).sum(), (values[0] - values[1]) / (2 * epsilon)
  max_error(ad, fd, atol=2e-5, rtol=3e-4)
  return {
    "autograd": ad.item(),
    "finite_difference": fd.item(),
    "relative_error": ((ad - fd).abs() / fd.abs().clamp_min(1e-12)).item(),
  }


def run():
  torch.manual_seed(81)
  torch.set_num_threads(1)
  result = {
    "device": "cpu",
    "dtype": "float64",
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "mean_hidden": [16],
    "noise_hidden": [12],
    "batch": 2,
    "horizon": 2,
  }
  model, representation, state, action, context = fixture(2)
  vec = MatrixFilter(model, representation.substep_coeff, "vmap")
  eager = MatrixFilter(model, representation.substep_coeff, "eager")
  legacy = LieBeliefFilter(
    QuaternionReference(model),
    GaussianMeasurementConfig(),
    torch.zeros(29, dtype=torch.float64),
    FilterConfig(jacobian_backend="eager"),
  )

  # Integration values over several absolute orientations, including near pi.
  parity = []
  for angle in (0.0, 1e-8, 0.8, 3.14):
    theta = torch.tensor([angle, 0.0, 0.0], dtype=torch.float64).expand(2, 3)
    sample = replace(state, rotation=so3_exp(theta))
    old = legacy.mean_step(sample, action, context)
    new = vec.mean_step(sample, action, context)
    parity.append(
      max(
        max_error(a, b, atol=3e-9)
        for a, b in zip(state_fields(new), state_fields(old), strict=True)
      )
    )
  result["integration_max_abs_error"] = max(parity)

  for _ in range(3):
    vec.transition_jacobian(state, action, context)
    legacy.transition_jacobian(state, action, context)
  av = vec.transition_jacobian(state, action, context)
  ae = eager.transition_jacobian(state, action, context)
  al = legacy.transition_jacobian(state, action, context)
  result["jacobian_vs_matrix_eager_max_abs_error"] = max_error(av, ae)
  result["jacobian_vs_quaternion_eager_max_abs_error"] = max_error(av, al, atol=3e-9)
  finite_columns = []
  with torch.no_grad():
    nominal = vec.mean_step(state, action, context)
    for delta in torch.eye(70, dtype=action.dtype) * 1e-6:
      plus = vec.mean_step(boxplus(state, delta), action, context)
      minus = vec.mean_step(boxplus(state, -delta), action, context)
      finite_columns.append(
        (local_error(nominal, plus) - local_error(nominal, minus)) / 2e-6
      )
  result["jacobian_finite_difference_max_abs_error"] = max_error(
    av, torch.stack(finite_columns, dim=-1), atol=2e-8, rtol=2e-6
  )
  weight = torch.randn_like(av)

  def objective():
    return (vec.transition_jacobian(state, action, context) * weight).sum()

  parameter = model.mlp[0].weight
  mixed = torch.autograd.grad(objective(), parameter)[0]
  eager_mixed = torch.autograd.grad((ae * weight).sum(), parameter)[0]
  result["mixed_derivative_vs_eager_max_abs_error"] = max_error(mixed, eager_mixed)
  result["mixed_parameter_derivative"] = finite_difference(
    objective, parameter, mixed, 1e-5
  )

  initial = BeliefState(
    state, torch.eye(70, dtype=torch.float64).expand(2, 70, 70) * 0.01
  )
  observations = [vec.observation(state).detach() + shift for shift in (0.003, -0.002)]
  truth = boxplus(
    state, torch.linspace(-0.001, 0.002, 70, dtype=torch.float64)
  ).detach()
  action.requires_grad_()

  def nll(estimator=vec):
    belief = initial
    losses = []
    for observation in observations:
      belief = estimator.correct(
        estimator.predict(belief, action, context), observation
      )
      losses.append(posterior_nll(belief, truth).mean())
    return torch.stack(losses).mean()

  names = (
    "mean_first_weight",
    "mean_last_bias",
    "noise_first_weight",
    "noise_last_weight",
    "action",
  )
  params = (
    model.mlp[0].weight,
    model.mlp[-1].bias,
    model.noise_head[0].weight,
    model.noise_head[-1].weight,
    action,
  )
  nll_vec, nll_eager, nll_legacy = nll(), nll(eager), nll(legacy)
  result["nll_vs_eager_max_abs_error"] = max_error(nll_vec, nll_eager)
  result["nll_vs_quaternion_max_abs_error"] = max_error(nll_vec, nll_legacy, atol=2e-5)
  gradients = torch.autograd.grad(nll_vec, params)
  reference = torch.autograd.grad(nll_eager, params)
  old_gradients = torch.autograd.grad(nll_legacy, params)
  result["nll_gradients"] = {}
  for name, param, grad, ref, old_grad in zip(
    names, params, gradients, reference, old_gradients, strict=True
  ):
    result["nll_gradients"][name] = {
      **finite_difference(nll, param, grad),
      "vs_matrix_eager_max_abs_error": max_error(grad, ref, atol=2e-5),
      "vs_quaternion_eager_max_abs_error": max_error(
        grad, old_grad, atol=2e-5, rtol=3e-5
      ),
    }
  model.requires_grad_(False)
  frozen = torch.autograd.grad(nll(), action)[0]
  result["frozen_model_action_derivative"] = finite_difference(nll, action, frozen)
  model.requires_grad_(True)

  # Exact zero angular increment exercises the Taylor branch and its outer AD.
  zero_model, zero_rep, zero_state, zero_action, zero_context = fixture(2)
  with torch.no_grad():
    zero_model.mlp[-1].weight.zero_()
    zero_model.mlp[-1].bias.zero_()
  zero_state = replace(zero_state, omega=torch.zeros_like(zero_state.omega))
  zero_filter = MatrixFilter(zero_model, zero_rep.substep_coeff, "vmap")

  def zero_objective():
    return (
      zero_filter.transition_jacobian(zero_state, zero_action, zero_context) * weight
    ).sum()

  zero_gradient = torch.autograd.grad(zero_objective(), zero_model.mlp[-1].bias)[0]
  result["zero_increment_mixed_derivative"] = finite_difference(
    zero_objective, zero_model.mlp[-1].bias, zero_gradient
  )

  m32, rep32, state32, action32, context32 = fixture(2, torch.float32)
  vec32 = MatrixFilter(m32, rep32.substep_coeff, "vmap")
  eager32 = MatrixFilter(m32, rep32.substep_coeff, "eager")
  a32 = vec32.transition_jacobian(state32, action32, context32)
  ref32 = eager32.transition_jacobian(state32, action32, context32)
  result["float32_jacobian_max_abs_error"] = max_error(a32, ref32, atol=3e-6, rtol=3e-5)
  g32 = torch.autograd.grad((a32 * weight.float()).sum(), m32.mlp[0].weight)[0]
  gr32 = torch.autograd.grad((ref32 * weight.float()).sum(), m32.mlp[0].weight)[0]
  result["float32_mixed_derivative_max_abs_error"] = max_error(
    g32, gr32, atol=5e-5, rtol=5e-4
  )

  result["two_step_forward_backward_seconds"] = {}
  for backend, filt in (("eager", eager), ("vmap", vec)):
    torch.autograd.grad(nll(filt), params)
    start = time.perf_counter()
    for _ in range(2):
      torch.autograd.grad(nll(filt), params)
    result["two_step_forward_backward_seconds"][backend] = (
      time.perf_counter() - start
    ) / 2

  timings = []
  for batch in (1, 2, 4, 8):
    m, rep, s, a, ctx = fixture(batch)
    row = {"batch": batch}
    for backend in ("eager", "vmap"):
      filt = MatrixFilter(m, rep.substep_coeff, backend)
      with torch.no_grad():
        filt.transition_jacobian(s, a, ctx)
        start = time.perf_counter()
        for _ in range(3):
          filt.transition_jacobian(s, a, ctx)
        row[backend + "_seconds"] = (time.perf_counter() - start) / 3
    row["speedup"] = row["eager_seconds"] / row["vmap_seconds"]
    timings.append(row)
  result["jacobian_timings"] = timings
  result["status"] = "passed"
  print(json.dumps(result, indent=2))


if __name__ == "__main__":
  run()
