import asyncio
import sys
import tomllib
from pathlib import Path

import pytest

from supermimic.scripts.launch_eit import launch
from supermimic.scripts.launch_eit_tuning import main, plan

TEST_CONFIG = """\
gpu_index = [5, 2, 7]
seed = 42
[paths]
environment_root = "env"
dataset_root = "data"
motion_file = "walk.npz"
checkpoint_root = "runs"
experiment = "widths"
[training]
policy_lr = 2e-4
mean_lr = [2e-4, 3e-4, 4e-4]
hidden_dims = [[512, 512, 512], [1024, 1024, 1024], [2048, 2048, 2048]]
critic_hidden_dims = [64, 32, 16]
eager = [false, true, false]
["wm validation"]
validation_stride = 5
"""


@pytest.fixture
def config():
  return tomllib.loads(TEST_CONFIG)


def test_example_config_builds_one_job_per_gpu():
  path = Path(__file__).parents[1] / "train_eit_tuning.toml"
  with path.open("rb") as stream:
    example = tomllib.load(stream)
  _, _, (queues,) = plan(example, path.parent)
  assert list(queues) == example["gpu_index"]
  assert all(len(jobs) == 1 for jobs in queues.values())


def test_tuning_dispatches_method_c_concurrently(config, tmp_path, monkeypatch):
  environment, _, phases = plan(config, tmp_path)
  commands = {}
  started = asyncio.Event()

  class Worker:
    async def wait(self):
      await asyncio.wait_for(started.wait(), timeout=2)
      return 0

  async def spawn(*args, env):
    commands[int(env["CUDA_VISIBLE_DEVICES"])] = args
    if len(commands) == 3:
      started.set()
    return Worker()

  monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
  asyncio.run(launch(phases, environment, wait_for_all=True))
  for i, gpu in enumerate(config["gpu_index"]):
    args = commands[gpu]
    assert args[args.index("--group") + 1] == "C"
    assert args[args.index("--device") + 1] == "cuda:0"
    assert "--policy-lr=0.0002" in args
    assert f"--mean-lr={config['training']['mean_lr'][i]}" in args
    start = args.index("--hidden-dims") + 1
    assert list(args[start : start + 3]) == [str(512 * 2**i)] * 3
    start = args.index("--critic-hidden-dims") + 1
    assert args[start : start + 3] == ("64", "32", "16")
    assert ("--eager" in args) == (i == 1)
    output = tmp_path / "runs/widths" / f"trial_{i + 1:03d}/seed42/C"
    assert Path(args[args.index("--output") + 1]) == output


def test_cli_finishes_healthy_trial_and_reports_all_failed_gpus(
  tmp_path, monkeypatch, capsys
):
  config_path = tmp_path / "tuning.toml"
  config_path.write_text(TEST_CONFIG)
  data = tmp_path / "data"
  data.mkdir()
  (data / "walk.npz").touch()
  environment = tmp_path / "env"
  python = environment / ".venv/bin/python"
  python.parent.mkdir(parents=True)
  (environment / "env.sh").write_text(
    "export CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=FASTEST_FIRST\n"
  )
  python.write_text(
    f"#!{sys.executable}\n"
    + """
import os, sys, time
from pathlib import Path
gpu = int(os.environ["CUDA_VISIBLE_DEVICES"])
assert os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
if gpu in (2, 5):
  sys.exit(gpu)
time.sleep(0.2)
output = Path(sys.argv[sys.argv.index("--output") + 1])
output.mkdir(parents=True)
(output / "finished").write_text(str(gpu))
"""
  )
  python.chmod(0o755)
  monkeypatch.setattr(sys, "argv", ["launch_eit_tuning", "--config", str(config_path)])
  with pytest.raises(SystemExit) as error:
    main()
  assert error.value.code == 1
  stderr = capsys.readouterr().err
  for gpu in (2, 5):
    assert f"GPU {gpu} exited with status {gpu}" in stderr
  assert (tmp_path / "runs/widths/trial_003/seed42/C/finished").read_text() == "7"


def test_cancelling_tuning_terminates_active_workers(config, tmp_path, monkeypatch):
  environment, _, phases = plan(config, tmp_path)
  ready = asyncio.Event()
  started = set()
  stopped = set()

  class Worker:
    returncode = None

    def __init__(self, gpu):
      self.gpu = gpu

    async def wait(self):
      if self.returncode is None:
        started.add(self.gpu)
        if len(started) == len(config["gpu_index"]):
          ready.set()
        await asyncio.Future()
      return self.returncode

    def terminate(self):
      self.returncode = -15
      stopped.add(self.gpu)

  async def spawn(*args, env):
    return Worker(int(env["CUDA_VISIBLE_DEVICES"]))

  async def interrupt():
    task = asyncio.create_task(launch(phases, environment, wait_for_all=True))
    await asyncio.wait_for(ready.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
      await task

  monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
  asyncio.run(interrupt())
  assert stopped == set(config["gpu_index"])


@pytest.mark.parametrize(
  "section, key, value",
  [
    (None, "gpu_index", []),
    (None, "gpu_index", [1, 0, 1]),
    ("training", "mean_lr", [2e-4]),
    ("training", "mean_lr", [[1], [2], [3]]),
    ("training", "hidden_dims", [[512], [1024]]),
    ("training", "hidden_dims", [[512], [], [2048]]),
    ("training", "hidden_dims", [[512], 1024, [2048]]),
    ("wm validation", "validation_stride", [2, 3]),
  ],
)
def test_reject_invalid_tuning_config(config, tmp_path, section, key, value):
  settings = config if section is None else config[section]
  settings[key] = value
  with pytest.raises(ValueError, match=key):
    plan(config, tmp_path)
