"""Display independent body predictions through public mjlab's ghost API."""

import mujoco
import numpy as np


def set_ghost_body_poses(
  model: mujoco.MjModel,
  body_ids: np.ndarray,
  positions: np.ndarray,
  quaternions: np.ndarray,
) -> np.ndarray:
  """Set rest transforms on a visualization-only copy and return its qpos.

  Maximal-coordinate predictions need not satisfy joint constraints. Moving
  each tracked body's rest transform lets the public ``add_ghost_mesh`` API
  render those poses while retaining the original model/data dimensions.
  Hinge/slide joints remain at qpos0, where their relative transforms vanish;
  untracked fixed/intermediate links retain their local rest transforms.
  Never pass the simulation model to this function.
  """
  qpos = model.qpos0.copy()
  world_pos = np.zeros((model.nbody, 3))
  world_quat = np.zeros((model.nbody, 4))
  world_quat[0, 0] = 1
  targets = {int(bid): row for row, bid in enumerate(body_ids)}
  inverse_parent = np.empty(4)
  for bid in range(1, model.nbody):
    parent = model.body_parentid[bid]
    joint = model.body_jntadr[bid]
    is_free = joint >= 0 and model.jnt_type[joint] == int(mujoco.mjtJoint.mjJNT_FREE)
    if bid in targets:
      row = targets[bid]
      world_pos[bid] = positions[row]
      world_quat[bid] = quaternions[row]
      if is_free:
        adr = model.jnt_qposadr[joint]
        qpos[adr : adr + 3] = world_pos[bid]
        qpos[adr + 3 : adr + 7] = world_quat[bid]
      else:
        mujoco.mju_negQuat(inverse_parent, world_quat[parent])
        mujoco.mju_rotVecQuat(
          model.body_pos[bid], world_pos[bid] - world_pos[parent], inverse_parent
        )
        mujoco.mju_mulQuat(model.body_quat[bid], inverse_parent, world_quat[bid])
    elif is_free:
      adr = model.jnt_qposadr[joint]
      world_pos[bid] = qpos[adr : adr + 3]
      world_quat[bid] = qpos[adr + 3 : adr + 7]
    else:
      mujoco.mju_rotVecQuat(world_pos[bid], model.body_pos[bid], world_quat[parent])
      world_pos[bid] += world_pos[parent]
      mujoco.mju_mulQuat(world_quat[bid], world_quat[parent], model.body_quat[bid])
  return qpos
