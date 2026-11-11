import mujoco
import numpy as np
import pytest
import torch

from .conftest import make_fk


def compare_with_mujoco(model, permutation=None):
  fk, joints = make_fk(model, permutation=permutation)
  rng = np.random.default_rng(31)
  args = [[] for _ in range(6)]
  expected = [[] for _ in range(4)]
  for _ in range(6):
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    data.qpos[:3] = rng.normal(size=3)
    quat = rng.normal(size=4)
    data.qpos[3:7] = quat / np.linalg.norm(quat)
    data.qpos[model.jnt_qposadr[joints]] += rng.uniform(-0.6, 0.6, len(joints))
    data.qvel[:] = rng.normal(size=model.nv)
    mujoco.mj_forward(model, data)
    velocities = np.zeros((model.nbody - 1, 6))
    for i in range(1, model.nbody):
      mujoco.mj_objectVelocity(
        model, data, mujoco.mjtObj.mjOBJ_XBODY, i, velocities[i - 1], 0
      )
    values = (
      data.qpos[:3],
      data.qpos[3:7],
      data.qpos[model.jnt_qposadr[joints]],
      velocities[0, 3:],
      velocities[0, :3],
      data.qvel[model.jnt_dofadr[joints]],
    )
    for buf, value in zip(args, values, strict=True):
      buf.append(value.copy())
    for buf, value in zip(
      expected,
      (data.xpos[1:], data.xquat[1:], velocities[:, 3:], velocities[:, :3]),
      strict=True,
    ):
      buf.append(value.copy())
  tensors = [torch.tensor(np.stack(values)).reshape(2, 3, -1) for values in args]
  for actual, reference in zip(fk(*tensors), expected, strict=True):
    reference = torch.tensor(np.stack(reference)).reshape_as(actual)
    torch.testing.assert_close(actual, reference, atol=2e-12, rtol=2e-12)
  return fk, tensors


def test_g1_poses_and_body_origin_velocities(g1_model):
  compare_with_mujoco(g1_model)


def test_entity_joint_order_can_differ_from_model_order(g1_model):
  compare_with_mujoco(g1_model, np.arange(29)[::-1])


def test_offsets_fixed_links_slide_and_multiple_joints():
  model = mujoco.MjModel.from_xml_string("""
  <mujoco><compiler angle="radian"/><worldbody><body><freejoint/>
  <geom type="sphere" size=".1"/><body pos=".2 .1 .3" euler=".3 .2 .1">
  <joint type="hinge" axis="0 1 0" pos=".1 .03 -.02" ref=".4"/>
  <joint type="hinge" axis="1 0 0" pos="-.02 .04 .1" ref="-.2"/>
  <geom type="sphere" size=".05"/><body pos=".2 .1 .3">
  <joint type="slide" axis="0 0 1" ref=".1"/><geom type="sphere" size=".04"/>
  <body pos=".1 .2 .3" euler=".2 .4 .3"><geom type="sphere" size=".03"/>
  </body></body></body></body></worldbody></mujoco>""")
  compare_with_mujoco(model)


def test_gradients_include_positions_rotations_and_velocities(g1_model):
  fk, tensors = compare_with_mujoco(g1_model)
  constants = [t[0, :1].detach() for t in tensors]
  qj, vj = constants[2].requires_grad_(), constants[5].requires_grad_()

  def evaluate(q, v):
    return fk(constants[0], constants[1], q, constants[3], constants[4], v)

  assert torch.autograd.gradcheck(evaluate, (qj, vj), fast_mode=True)


def test_unsupported_ball_joint_fails_clearly():
  model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody><body><freejoint/>
  <geom type="sphere" size=".1"/><body><joint type="ball"/>
  <geom type="sphere" size=".1"/></body></body></worldbody></mujoco>""")
  with pytest.raises(ValueError, match="hinge and slide"):
    make_fk(model)
