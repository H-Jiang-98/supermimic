import copy

import mujoco
import numpy as np
from mjlab.viewer.native.visualizer import MujocoNativeDebugVisualizer

from supermimic.utils.ghost_model import set_ghost_body_poses


def test_independent_body_poses_keep_public_viewer_model_dimensions(g1_model):
  model = copy.deepcopy(g1_model)
  # Both public viewers reuse MjData allocated from the simulation model.
  data = mujoco.MjData(g1_model)
  body_ids = np.arange(1, model.nbody, 2)
  rng = np.random.default_rng(13)
  original_pos = g1_model.body_pos.copy()
  original_quat = g1_model.body_quat.copy()
  for _ in range(2):
    positions = rng.normal(size=(len(body_ids), 3))
    quaternions = rng.normal(size=(len(body_ids), 4))
    quaternions /= np.linalg.norm(quaternions, axis=-1, keepdims=True)
    data.qpos[:] = set_ghost_body_poses(model, body_ids, positions, quaternions)
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.xpos[body_ids], positions, atol=1e-12)
    alignment = np.abs(np.sum(data.xquat[body_ids] * quaternions, axis=-1))
    np.testing.assert_allclose(alignment, 1, atol=1e-12)
    scene = mujoco.MjvScene(g1_model, maxgeom=2 * model.ngeom)
    visualizer = MujocoNativeDebugVisualizer(scene, g1_model, env_idx=0)
    visualizer.add_ghost_mesh(data.qpos.copy(), model, alpha=0.8)
    assert scene.ngeom > 0
    for geom in scene.geoms[: scene.ngeom]:
      np.testing.assert_allclose(geom.pos, data.geom_xpos[geom.objid], atol=1e-6)
  np.testing.assert_array_equal(g1_model.body_pos, original_pos)
  np.testing.assert_array_equal(g1_model.body_quat, original_quat)
