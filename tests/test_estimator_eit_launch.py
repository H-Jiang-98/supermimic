import asyncio
import json
import sys
import tomllib
from pathlib import Path

import pytest

from supermimic.scripts.launch_eit import launch, plan, run_queue


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
      "mean": {"gpus": [3, 1], "seeds": [9, 2, 4]},
      "belief": {"gpus": [1], "seeds": [2, 5]},
    },
  }


def test_env_setup_can_replace_positional_arguments(tmp_path):
  (tmp_path / "env.sh").write_text(
    "set -- major minor\n"
    'export EIT_TEST_VERSION="$1.$2" CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=FASTEST_FIRST\n'
  )
  output = tmp_path / "worker result.json"
  arguments = ["--motion-file", "motion clips/*.npz", "", "value with spaces"]
  worker = (
    "import json, os, sys\n"
    "from pathlib import Path\n"
    "keys = ('EIT_TEST_VERSION', 'CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER')\n"
    "Path(sys.argv[1]).write_text(json.dumps([[os.environ[k] for k in keys], sys.argv[2:]]))"
  )
  command = [sys.executable, "-c", worker, str(output), *arguments]
  asyncio.run(run_queue(3, [command], tmp_path))
  assert json.loads(output.read_text()) == [
    ["major.minor", "3", "PCI_BUS_ID"],
    arguments,
  ]


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
    assert value("--group") == "mean"
    init.parent.mkdir(parents=True, exist_ok=True)
    init.touch(exist_ok=False)
  else:
    group, seed = value("--group"), value("--seed")
    assert init.is_file() and init.stem == f"seed{seed}"
    output = Path(value("--output"))
    assert output == root / "ckpt/walk" / f"seed{seed}" / f"{group}-direct"
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
    "jobs-3": [["mean", 9], ["mean", 4]],
    "jobs-1": [["mean", 2], ["belief", 2], ["belief", 5]],
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
  config["methods"]["mean"][key] = value
  with pytest.raises(ValueError, match=rf"methods\.mean\.{key}"):
    plan(config, tmp_path)


@pytest.mark.parametrize("group", ["A", "B", "C"])
def test_retired_groups_rejected_by_launcher_and_worker(
  config, tmp_path, monkeypatch, group
):
  from supermimic.scripts import train_eit

  config["methods"] = {group: {"gpus": [0], "seeds": [1]}}
  with pytest.raises(ValueError, match="EIT method"):
    plan(config, tmp_path)
  monkeypatch.setattr(
    sys, "argv", ["train_eit", "--group", group, "--motion-file", "unused.npz"]
  )
  with pytest.raises(SystemExit) as rejected:
    train_eit.main()
  assert rejected.value.code == 2


def test_checked_in_v7_config_has_four_variants_per_seed():
  root = Path(__file__).resolve().parents[1]
  with (root / "train_eit_v7.toml").open("rb") as stream:
    config = tomllib.load(stream)
  _, _, (prepare, train) = plan(config, root)

  def commands(queues):
    return [cmd for jobs in queues.values() for cmd in jobs]

  jobs = commands(train)
  by_seed = {}
  for cmd in jobs:
    seed = cmd[cmd.index("--seed") + 1]
    group = cmd[cmd.index("--group") + 1]
    mode = "residual" if "--residual-actor" in cmd else "direct"
    assert Path(cmd[cmd.index("--output") + 1]).name == f"{group}-{mode}"
    by_seed.setdefault(seed, []).append((group, mode))
  assert by_seed and all(
    sorted(variants)
    == [
      ("belief", "direct"),
      ("belief", "residual"),
      ("mean", "direct"),
      ("mean", "residual"),
    ]
    for variants in by_seed.values()
  )
  assert sorted(cmd[cmd.index("--seed") + 1] for cmd in commands(prepare)) == sorted(
    by_seed
  )


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
