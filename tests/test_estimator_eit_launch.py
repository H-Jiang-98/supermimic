import asyncio
import json
import sys

import pytest

from supermimic.scripts.launch_eit import launch, plan


@pytest.fixture
def config():
  return {
    "paths": {
      "environment_root": "env",
      "dataset_root": "data",
      "checkpoint_root": "ckpt",
      "motion_file": "motions/*.npz",
      "experiment": "walk",
    },
    "training": {"mean_lr": 0.003, "hidden_dims": [32, 16]},
    "methods": {
      "A": {"gpus": [3, 1], "seeds": [9, 2, 4]},
      "C": {"gpus": [1], "seeds": [2, 5]},
    },
  }


@pytest.mark.parametrize(
  "allocator_env,from_setup",
  [
    (None, False),
    ("PYTORCH_ALLOC_CONF", False),
    ("PYTORCH_CUDA_ALLOC_CONF", False),
    ("PYTORCH_ALLOC_CONF", True),
    ("PYTORCH_CUDA_ALLOC_CONF", True),
  ],
)
def test_launch_shared_seeds_and_gpu_queues(
  config, tmp_path, monkeypatch, allocator_env, from_setup
):
  for key in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):
    monkeypatch.delenv(key, raising=False)
  if allocator_env is not None and not from_setup:
    monkeypatch.setenv(allocator_env, "expandable_segments:False")
  expected_allocator = {
    allocator_env or "PYTORCH_ALLOC_CONF": (
      "expandable_segments:False" if allocator_env else "expandable_segments:True"
    )
  }
  environment, motion, phases = plan(config, tmp_path)
  assert motion == tmp_path / "data/motions/*.npz"
  python = environment / ".venv/bin/python"
  python.parent.mkdir(parents=True)
  setup = f"export EIT_TEST_DIR={tmp_path}\n"
  if from_setup:
    setup += f"export {allocator_env}=expandable_segments:False\n"
  (environment / "env.sh").write_text(setup)
  # A tiny worker claims its physical card for the life of the process. A
  # second worker on the same card would fail the exclusive file creation.
  python.write_text(
    f"#!{sys.executable}\n"
    + "import os\nassert {k: os.environ[k] for k in "
    + f"('PYTORCH_ALLOC_CONF', 'PYTORCH_CUDA_ALLOC_CONF') if k in os.environ}} == {expected_allocator!r}\n"
    + """
import json, os, sys, time
from pathlib import Path
root = Path(os.environ["EIT_TEST_DIR"])
gpu = os.environ["CUDA_VISIBLE_DEVICES"]
busy = root / f"busy-{gpu}"
with busy.open("x"):
  args = sys.argv[1:]
  value = lambda flag: args[args.index(flag) + 1]
  assert value("--device") == "cuda:0"
  assert "--mean-lr=0.003" in args
  assert args[args.index("--hidden-dims")+1:args.index("--hidden-dims")+3] == ["32", "16"]
  init = Path(value("--initialization"))
  if "--prepare-initialization" in args:
    assert value("--group") == "B"
    init.parent.mkdir(parents=True, exist_ok=True)
    init.touch(exist_ok=False)
  else:
    group, seed = value("--group"), value("--seed")
    assert init.is_file() and init.stem == f"seed{seed}"
    output = Path(value("--output"))
    assert output == root / "ckpt/walk" / f"seed{seed}" / group
    with (root / f"jobs-{gpu}").open("a") as log:
      print(json.dumps([group, int(seed)]), file=log)
  time.sleep(.02)
busy.unlink()
"""
  )
  python.chmod(0o755)
  asyncio.run(launch(phases, environment))
  jobs = {
    f.name: [json.loads(line) for line in f.read_text().splitlines()]
    for f in tmp_path.glob("jobs-*")
  }
  assert jobs == {
    "jobs-3": [["A", 9], ["A", 4]],
    "jobs-1": [["A", 2], ["C", 2], ["C", 5]],
  }
  assert len(list((tmp_path / "ckpt/walk/init").glob("*.pt"))) == 4


@pytest.mark.parametrize(
  "key,value",
  [
    ("seeds", [2, 2]),
    ("seeds", [True]),
    ("gpus", []),
    ("gpus", [-1]),
  ],
)
def test_invalid_method(config, tmp_path, key, value):
  config["methods"]["A"][key] = value
  with pytest.raises(ValueError, match=rf"methods\.A\.{key}"):
    plan(config, tmp_path)


def test_output_cannot_be_overridden_by_training_options(config, tmp_path):
  config["training"]["output"] = "old-records"
  with pytest.raises(ValueError, match="output"):
    plan(config, tmp_path)


@pytest.mark.parametrize(
  "solver,expected",
  [
    (None, (10, 20)),
    ({"iterations": 20, "ls_iterations": 50}, (20, 50)),
    ({"ls_iterations": 35}, (10, 35)),
  ],
)
def test_mujoco_limits_reach_model(config, tmp_path, monkeypatch, solver, expected):
  import mujoco

  from supermimic.scripts import train_eit

  if solver is not None:
    config["env"] = {"sim": {"mujoco": solver}}
  _, _, phases = plan(config, tmp_path)
  flags = [
    f"--env.sim.mujoco.{key.replace('_', '-')}={value}"
    for key, value in (solver or {}).items()
  ]
  for phase in phases:
    for jobs in phase.values():
      for cmd in jobs:
        assert [arg for arg in cmd if arg.startswith("--env.sim.mujoco.")] == flags

  class ModelChecked(Exception):
    pass

  def inspect_model(cfg, num_envs, device):
    model = mujoco.MjModel.from_xml_string("<mujoco/>")
    cfg.sim.mujoco.apply(model)
    assert (model.opt.iterations, model.opt.ls_iterations) == expected
    raise ModelChecked

  command = next(iter(phases[0].values()))[0]
  monkeypatch.setattr(sys, "argv", command[2:])
  monkeypatch.setattr(train_eit, "create_env", inspect_model)
  monkeypatch.setenv("WANDB_MODE", "disabled")
  with pytest.raises(ModelChecked):
    train_eit.main()


@pytest.mark.parametrize("option", ["eager", "policy_cuda_graphs"])
@pytest.mark.parametrize("enabled", [False, True])
def test_execution_flag_is_forwarded_to_both_phases(config, tmp_path, option, enabled):
  config["training"][option] = enabled
  _, _, phases = plan(config, tmp_path)
  for phase in phases:
    for jobs in phase.values():
      for command in jobs:
        assert ("--" + option.replace("_", "-") in command) == enabled


@pytest.mark.parametrize("option", ["eager", "policy_cuda_graphs"])
def test_execution_flag_requires_a_toml_boolean(config, tmp_path, option):
  config["training"][option] = "false"
  with pytest.raises(ValueError, match="boolean"):
    plan(config, tmp_path)


@pytest.mark.parametrize(
  "requested,eager,expected",
  [(False, False, False), (True, False, True), (True, True, False)],
)
def test_worker_graph_flag_and_eager_override(
  tmp_path, monkeypatch, requested, eager, expected
):
  from types import SimpleNamespace

  from supermimic.scripts import train_eit

  class ConfigurationChecked(Exception):
    pass

  def inspect_training(*args, policy_cuda_graphs):
    assert policy_cuda_graphs is expected
    raise ConfigurationChecked

  flags = [
    "train_eit",
    "--motion-file=unused.npz",
    "--device=cpu",
    f"--output={tmp_path}",
  ]
  if requested:
    flags.append("--policy-cuda-graphs")
  if eager:
    flags.append("--eager")
  monkeypatch.setattr(sys, "argv", flags)
  monkeypatch.setenv("WANDB_MODE", "disabled")
  monkeypatch.setattr(
    train_eit, "create_env", lambda *a: SimpleNamespace(close=lambda: None)
  )
  monkeypatch.setattr(train_eit, "SuperMimicRunner", lambda *a, **kw: None)
  monkeypatch.setattr(train_eit, "EITTraining", inspect_training)
  with pytest.raises(ConfigurationChecked):
    train_eit.main()


@pytest.mark.parametrize(
  "validation",
  [
    None,
    {
      "validation_stride": 7,
      "validation_batch_size": 11,
      "cov_validation_pairs": 13,
    },
  ],
)
def test_validation_configuration_reaches_worker(
  config, tmp_path, monkeypatch, validation
):
  from types import SimpleNamespace

  from supermimic.scripts import train_eit

  if validation is not None:
    config["wm validation"] = validation
  expected = validation or dict(
    validation_stride=5, validation_batch_size=4, cov_validation_pairs=8192
  )
  _, _, phases = plan(config, tmp_path)
  for phase in phases:
    for jobs in phase.values():
      for command in jobs:
        for name, value in (validation or {}).items():
          assert f"--{name.replace('_', '-')}={value}" in command

  class ConfigurationChecked(Exception):
    pass

  def inspect_training(runner, loop_cfg, policy_cfg, **kwargs):
    assert loop_cfg.validation_stride == expected["validation_stride"]
    assert loop_cfg.validation_batch_size == expected["validation_batch_size"]
    assert runner.cfg.cov.validation_pairs == expected["cov_validation_pairs"]
    raise ConfigurationChecked

  command = next(iter(phases[0].values()))[0]
  monkeypatch.setattr(sys, "argv", command[2:])
  monkeypatch.setenv("WANDB_MODE", "disabled")
  monkeypatch.setattr(
    train_eit, "create_env", lambda *a: SimpleNamespace(close=lambda: None)
  )
  monkeypatch.setattr(
    train_eit, "SuperMimicRunner", lambda env, cfg, *a, **kw: SimpleNamespace(cfg=cfg)
  )
  monkeypatch.setattr(train_eit, "EITTraining", inspect_training)
  with pytest.raises(ConfigurationChecked):
    train_eit.main()


@pytest.mark.parametrize(
  "validation",
  [
    {"validation_stride": 1},
    {"validation_stride": True},
    {"validation_batch_size": 0},
    {"cov_validation_pairs": -1},
    {"cov_validation_pairs": 2.5},
    {"unknown": 3},
  ],
)
def test_invalid_model_validation_config(config, tmp_path, validation):
  config["wm validation"] = validation
  with pytest.raises(ValueError, match="wm validation"):
    plan(config, tmp_path)


def test_duplicate_validation_option_rejected(config, tmp_path):
  config["training"]["cov_validation_pairs"] = 8
  config["wm validation"] = {"cov_validation_pairs": 16}
  with pytest.raises(ValueError, match="Duplicate cov_validation_pairs"):
    plan(config, tmp_path)
