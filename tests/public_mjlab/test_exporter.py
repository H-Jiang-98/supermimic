"""Check the public mjlab metadata contract on real exported ONNX files."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import onnx
import pytest
import torch
from mjlab.envs.mdp.actions import JointPositionAction
from mjlab.managers import ObservationTermCfg
from onnx.reference import ReferenceEvaluator
from tensordict import TensorDict

from supermimic.models import MultiEncoderAE
from supermimic.runners.exporter import export_actor_as_onnx
from supermimic.tasks.task_spec import TaskSpec

SENSOR_WIDTHS = {
  "projected_gravity": 3,
  "base_ang_vel": 3,
  "joint_pos": 2,
  "joint_vel": 2,
  "actions": 2,
}


@pytest.fixture
def export_env():
  # Real public metadata serialization, with only simulation state stubbed out.
  def term(**kwargs):
    return ObservationTermCfg(func=lambda _: None, **kwargs)

  configs = {
    "actor": {name: term() for name in SENSOR_WIDTHS},
    "goal": {
      "command": term(scale=(0.5, 1.0, 1.5, 2.0), clip=(-3.0, 3.0)),
      "motion_anchor_pos_b": term(scale=99.0),
      "motion_anchor_ori_b": term(scale=torch.full((6,), 0.25)),
    },
  }
  configs["actor"]["projected_gravity"] = term(scale=2.0, clip=(-0.7, 0.7))
  configs["actor"]["joint_pos"] = term(
    scale=torch.tensor([0.5, 1.5]), flatten_history_dim=False
  )
  frames, bodies, joints = 5, 2, 2
  positions = torch.randn(frames, bodies, 3)
  quaternions = torch.zeros(frames, bodies, 4)
  quaternions[..., 0] = 1
  velocities = torch.randn(frames, bodies, 3)
  motion = SimpleNamespace(
    time_step_total=frames,
    joint_pos=torch.randn(frames, joints),
    joint_vel=torch.randn(frames, joints),
    root_lin_vel_w=velocities[:, 0],
    body_state=lambda ids: (
      positions[ids],
      quaternions[ids],
      velocities[ids],
      velocities[ids],
    ),
  )
  command = SimpleNamespace(
    motion=motion,
    cfg=SimpleNamespace(anchor_body_name="pelvis", body_names=("pelvis", "hand")),
  )
  joint_action = MagicMock(spec=JointPositionAction)
  joint_action._scale = torch.ones(1, joints)
  robot = SimpleNamespace(
    joint_names=["joint0", "joint1"],
    spec=SimpleNamespace(
      actuators=[SimpleNamespace(target=f"robot/joint{i}", id=i) for i in range(joints)]
    ),
    data=SimpleNamespace(default_joint_pos=torch.zeros(1, joints)),
  )
  return SimpleNamespace(
    scene={"robot": robot},
    sim=SimpleNamespace(
      mj_model=SimpleNamespace(
        actuator_gainprm=np.ones((joints, 10)),
        actuator_biasprm=np.zeros((joints, 10)),
      )
    ),
    action_manager=SimpleNamespace(get_term=lambda _: joint_action),
    command_manager=SimpleNamespace(
      active_terms=["motion"], get_term=lambda _: command
    ),
    observation_manager=SimpleNamespace(
      active_terms={group: list(terms) for group, terms in configs.items()},
      get_term_cfg=lambda group, name: configs[group][name],
    ),
  )


@pytest.mark.parametrize("history_length", [1, 4])
def test_exported_metadata_describes_input_and_preserves_actions(
  export_env, tmp_path, history_length
):
  specs = [
    TaskSpec(name="motion", goal_dim=10, extract_training_goal=lambda *_: None),
    TaskSpec(
      name="velocity",
      goal_dim=3,
      extract_training_goal=lambda *_: None,
      precompute_replay_goal=lambda motion: motion.root_lin_vel_w,
    ),
  ]
  actor = MultiEncoderAE(
    obs_dim=sum(SENSOR_WIDTHS.values()),
    action_dim=2,
    task_specs=specs,
    slot_mix={"motion": 1.0},
    latent_dim=4,
    hidden_dims=(8,),
    latent_bc_weight=0.0,
    history_length=history_length,
  ).eval()
  paths = export_actor_as_onnx(
    actor, export_env, str(tmp_path), "test", list(SENSOR_WIDTHS.values())
  )
  assert {Path(path).stem for path in paths} == {
    "policy_motion",
    "policy_velocity",
    "policy_velocity_replay",
  }
  history = torch.randn(1, history_length, actor.obs_dim)
  flat_history = torch.cat(
    [part.flatten(1) for part in history.split(list(SENSOR_WIDTHS.values()), dim=-1)],
    dim=-1,
  )
  motion_goal, velocity_goal = torch.randn(1, 10), torch.randn(1, 3)
  replay_goal = export_env.command_manager.get_term("motion").motion.root_lin_vel_w[2:3]
  for path in paths:
    model = onnx.load(path)
    onnx.checker.check_model(model)
    metadata = {item.key: item.value for item in model.metadata_props}
    name = Path(path).stem
    is_motion = name == "policy_motion"
    is_replay = name == "policy_velocity_replay"
    goal_names = (
      ["command", "motion_anchor_ori_b"]
      if is_motion
      else ([] if is_replay else ["command"])
    )
    names = metadata["observation_names"].split(",")
    assert names == list(SENSOR_WIDTHS) + goal_names
    for field in (
      "observation_terms_scale",
      "observation_terms_clip",
      "observation_terms_history_length",
      "observation_terms_flatten_history_dim",
      "observation_history_lengths",
    ):
      assert len(metadata[field].split(",")) == len(names)
    histories = [
      int(float(value))
      for value in metadata["observation_terms_history_length"].split(",")
    ]
    assert histories == [history_length] * len(SENSOR_WIDTHS) + [1] * len(goal_names)
    assert (
      metadata["observation_history_lengths"]
      == metadata["observation_terms_history_length"]
    )
    assert all(
      float(v) == 1
      for v in metadata["observation_terms_flatten_history_dim"].split(",")
    )
    widths = SENSOR_WIDTHS | {
      "command": 4 if is_motion else 3,
      "motion_anchor_ori_b": 6,
    }
    input_dim = model.graph.input[0].type.tensor_type.shape.dim[1].dim_value
    assert (
      sum(widths[n] * h for n, h in zip(names, histories, strict=True)) == input_dim
    )
    scales = dict(
      zip(names, metadata["observation_terms_scale"].split(","), strict=True)
    )
    clips = dict(zip(names, metadata["observation_terms_clip"].split(","), strict=True))
    assert scales["projected_gravity"] == "2.000"
    assert scales["joint_pos"] == "0.500;1.500"
    assert clips["projected_gravity"] == "-0.700;0.700"
    if is_motion:
      assert scales["command"] == "0.500;1.000;1.500;2.000"
      assert scales["motion_anchor_ori_b"] == ";".join(["0.250"] * 6)
      assert clips["command"] == "-3.000;3.000"
    elif not is_replay:
      assert metadata["command_names"] == "twist"
      assert scales["command"] == "1.000"
      assert clips["command"] == "-inf;inf"

    goal = motion_goal if is_motion else (replay_goal if is_replay else velocity_goal)
    obs = flat_history if is_replay else torch.cat([flat_history, goal], dim=-1)
    inputs = {"obs": obs.numpy()}
    if is_motion or is_replay:
      inputs["time_step"] = np.array([[2.0]], dtype=np.float32)
    actual = ReferenceEvaluator(model).run(None, inputs)[0]
    actor.set_slot_mix({"motion" if is_motion else "velocity": 1.0})
    td = TensorDict(
      {"history": history, "goal_motion": motion_goal, "goal_velocity": goal},
      batch_size=[1],
    )
    with torch.no_grad():
      expected = actor(td).numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
