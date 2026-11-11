"""Small MuJoCo fixtures; no GPU or motion download required."""

import mujoco
import numpy as np
import pytest
import torch
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML

from supermimic.utils.forward_kinematics import ForwardKinematics


@pytest.fixture
def g1_model():
  return mujoco.MjModel.from_xml_path(str(G1_XML))


def make_fk(model, dtype=torch.float64, permutation=None):
  joints = np.where(np.isin(model.jnt_type, [2, 3]))[0]
  if permutation is not None:
    joints = joints[permutation]
  bodies = torch.arange(1, model.nbody)
  qadr = model.jnt_qposadr[joints]
  return ForwardKinematics(
    model, bodies, torch.tensor(qadr), torch.arange(len(bodies)), dtype=dtype
  ), joints


@pytest.fixture
def fk(g1_model):
  return make_fk(g1_model, dtype=torch.float32)[0]


@pytest.fixture
def motion_file(tmp_path):
  def write(name="motion.npz", frames=6, root_x=0.0, public=False, fps=50.0):
    values = {
      "joint_pos": np.zeros((frames, 29), dtype=np.float32),
      "joint_vel": np.zeros((frames, 29), dtype=np.float32),
      "root_pos_w": np.tile([root_x, 0.0, 0.8], (frames, 1)),
      "root_quat_w": np.tile([1.0, 0.0, 0.0, 0.0], (frames, 1)),
      "root_lin_vel_w": np.zeros((frames, 3)),
      "root_ang_vel_w": np.zeros((frames, 3)),
      "fps": fps,
    }
    if public:
      for field in list(values):
        if field.startswith("root_"):
          values[field.replace("root_", "body_")] = np.repeat(
            values.pop(field)[:, None], 30, axis=1
          )
    path = tmp_path / name
    np.savez(path, **values)
    return str(path)

  return write
