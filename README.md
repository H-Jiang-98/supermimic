# SuperMimic

A [SuperTrack](https://montreal.ubisoft.com/en/supertrack-motion-tracking-for-physically-simulated-characters-using-supervised-learning/)
implementation for training world models and policies to mimic motion data, built on [mjlab_for_supermimic](https://github.com/H-Jiang-98/mjlab_for_supermimic).

## 快速开始：LAFAN1 G1 数据处理与训练

**先完成下方 [Setup](#setup)，再在 `supermimic/` 仓库根目录运行以下 Bash 命令。**
原始 CSV 应放在 `data/lafan1_g1/`；以下示例使用 GPU 0，并通过
`WANDB_MODE=disabled` 关闭 W&B 上传，无需登录 W&B。

### 1. 批量处理 `lafan1_g1` 数据

将所有 CSV 从 30 Hz 重采样为训练所需的 50 Hz，并保存为
`data/motions/<原文件名>.npz`。已存在且非空的 NPZ 会跳过，中断后重新运行即可
继续转换；如需重新生成某条动作，先移走对应的 NPZ。

```bash
(
  set -euo pipefail
  shopt -s nullglob
  motion_csvs=(data/lafan1_g1/*.csv)
  if ((${#motion_csvs[@]} == 0)); then
    echo "未找到 data/lafan1_g1/*.csv，请先准备数据。" >&2
    exit 1
  fi
  mkdir -p data/motions
  for csv in "${motion_csvs[@]}"; do
    motion_name=$(basename "$csv" .csv)
    motion_npz="data/motions/${motion_name}.npz"
    if [[ -s "$motion_npz" ]]; then
      echo "跳过已转换动作：$motion_npz"
      continue
    fi
    echo "转换：$csv -> $motion_npz"
    CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl WANDB_MODE=disabled \
      uv run python -m mjlab.scripts.csv_to_npz \
      --input-file "$csv" \
      --output-name "$motion_name" \
      --input-fps 30 --output-fps 50 \
      --device cuda:0 --render False
    cp /tmp/motion.npz "${motion_npz}.tmp"
    mv "${motion_npz}.tmp" "$motion_npz"
  done
)
```

当前转换器固定将结果写入 `/tmp/motion.npz`，`--output-name` 只设置 W&B
名称；因此循环中每转换一条就复制到目标目录，转换过程需串行运行。
`WANDB_MODE=disabled` 下转换器仍可能打印上传提示，但不会实际上传。

### 2. 选择单条或全部 motion files 开启训练

训练超参数统一在根目录的 [`train_config.toml`](train_config.toml) 中设置，
每项注释都给出代码默认值。文件采用较小规模的起始配置：1024 个仿真环境、
WM batch size 512、policy batch size 1024、100000 个采集步。
`[runner.wm]` 管理世界模型参数，`[runner.policy]` 管理策略参数。
这些是起始设置，尚未验证完整训练的收敛效果。

**训练单条 motion**：修改配置中的 `motion_file`，或用命令行临时覆盖。

```bash
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled \
  uv run src/supermimic/scripts/train.py SuperMimic-GenCoord-G1 \
  --config train_config.toml
```

**使用全部 motion 联合训练一个模型**：加载 `data/motions/` 下的所有 NPZ。

```bash
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled \
  uv run src/supermimic/scripts/train.py SuperMimic-GenCoord-G1 \
  --config train_config.toml \
  --motion-file 'data/motions/*.npz' \
  --runner.run-name lafan1_all
```

通配符路径必须保留引号，由程序展开。若只选择部分 motion，可传入逗号分隔的
路径，例如 `--motion-file 'data/motions/run1_subject5.npz,data/motions/walk1_subject1.npz'`。
两种训练方式都会训练世界模型和策略，并在本地保存 checkpoint。
如需最大坐标表示，将任务名替换为 `SuperMimic-MaxCoord-G1`；
首次编译耗时及短程验证参数见 [Diagnosing startup stalls](#diagnosing-startup-stalls)。

### 3. 修改超参数与短程验证

参数优先级为 **命令行显式参数 > TOML 配置文件 > 代码默认值**。
TOML 使用下划线键名（如 `batch_size`），命令行使用连字符
（如 `--runner.wm.batch-size`）。配置文件和 motion 文件的相对路径均相对于
运行命令时的工作目录；省略 `--config` 时保留原来的代码默认行为。

在配置文件的 `[runner]` 中设置 `log_root` 可以指定日志和 checkpoint 的根目录：

```toml
[runner]
log_root = "/path/to/training_logs" # 默认："logs"；也支持相对路径和 ~。
run_name = "lafan1_single_baseline"
```

实际保存目录为 `<log_root>/<任务名>/<run_name>/`，未设置 `run_name` 时使用时间戳。
例如上面的配置会写入
`/path/to/training_logs/SuperMimic-GenCoord-G1/lafan1_single_baseline/`，
W&B 启用时的本地文件也保存在该运行目录下。
相对 `log_root` 以启动命令的工作目录为基准；也可通过
`--runner.log-root /path/to/training_logs` 临时覆盖。
同名运行目录已存在时会报错，以免覆盖已有 checkpoint。

先用小规模配置验证采样及世界模型、策略更新：

```bash
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled \
  uv run src/supermimic/scripts/train.py SuperMimic-GenCoord-G1 \
  --config train_config.toml \
  --runner.run-name lafan1_smoke \
  --num-envs 128 \
  --runner.wm.batch-size 64 \
  --runner.policy.batch-size 64 \
  --runner.max-iterations 30 \
  --runner.log-interval 1 \
  --eager True
```

正式训练时去掉上述临时覆盖参数。查看配置文件加载后的参数及帮助：

```bash
uv run src/supermimic/scripts/train.py SuperMimic-GenCoord-G1 \
  --config train_config.toml --help
```

## 快速开始：W&B

项目已接入 W&B，可在线记录训练配置、指标和模型 checkpoint。先完成下方
[Setup](#setup)，并按前面的数据处理步骤准备
`data/motions/run1_subject5.npz`。以下命令在 `supermimic/` 仓库根目录执行。

### 1. 登录 W&B

在浏览器登录 [W&B 设置页面](https://wandb.ai/settings)，创建并复制 API Key，
然后在终端执行：

```bash
cd ~/Github_Repos/WBC/supermimic_ws/supermimic
WANDB_MODE=online uv run wandb login
```

按终端提示粘贴 API Key 并回车。更换账号时执行
`WANDB_MODE=online uv run wandb login --relogin`。
登录方式见 [W&B 官方说明](https://github.com/wandb/wandb#quickstart)。

### 2. 启动训练并上传指标

若需要指定 W&B 团队，在启动训练前执行以下命令，将占位内容替换为你有写入权限的
团队标识；不设置时使用账号的默认归属：

```bash
export WANDB_ENTITY="你的团队标识"
```

使用当前配置训练动作，并将指标上传到 `supermimic` 项目：
```bash
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=online \
  uv run src/supermimic/scripts/train.py SuperMimic-GenCoord-G1 \
  --config train_config.toml \
  --runner.wandb-project supermimic \
  --runner.run-name "lafan1_all_$(date +%Y%m%d_%H%M%S)"
```

**在线训练应使用 `WANDB_MODE=online`，不要沿用前面示例的 `WANDB_MODE=disabled`。**
运行名附加时间戳，以免与已有日志目录重名；当前代码会拒绝复用同名目录。

### 3. 日志与训练结果

训练曲线、checkpoint 选择及可视化用法见 [评估与可视化](README_eval.md)。

默认配置下，本地运行目录如下：

```text
logs/SuperMimic-GenCoord-G1/<运行名>/
├── world_model.pt
├── actor.pt
├── critic.pt
└── wandb/
```

修改 `[runner].log_root` 或传入 `--runner.log-root /path/to/training_logs`
可更换根目录；W&B 本地文件也保存在该运行目录下。
第一次世界模型和策略更新需要编译，可能几分钟没有新的训练日志；
终端会打印首次更新的开始和完成提示。排查方式见
[Diagnosing startup stalls](#diagnosing-startup-stalls)。

## Setup

Install [uv](https://docs.astral.sh/uv/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Use Python 3.13 and keep both fork repositories as siblings. The `dev` branch of
`mjlab_for_supermimic` is based on public mjlab 1.6.0 and contains the motion-command
extension hooks required by SuperMimic. For the existing local checkout pair:

```bash
cd ~/Github_Repos/WBC/supermimic_ws
git -C mjlab_for_supermimic switch dev
git -C supermimic switch dev
cd supermimic
uv sync --locked --python 3.13
```

After committing and publishing both `dev` branches, a fresh workspace can be
created with:

```bash
mkdir supermimic_ws && cd supermimic_ws
git clone --branch dev git@github.com:H-Jiang-98/mjlab_for_supermimic.git
git clone --branch dev git@github.com:H-Jiang-98/supermimic.git
cd supermimic
uv sync --locked --python 3.13
```

The editable dependency points to `../mjlab_for_supermimic`. PyTorch 2.14.0
(CUDA 13.0) and Warp 1.18.0 are pinned in the lockfile. Create a separate
`supermimic/.venv` here with `uv sync`; an environment copied from another workspace
can retain editable imports pointing to that old checkout. If you rename or move
the workspace, recreate `.venv` at the new location: installed command scripts
and activation scripts contain absolute paths. A normal `uv sync --locked` can
report all packages installed without repairing those paths. Keep a backup of
the old environment until the verification commands below pass.

`patches/mjlab-1.6.0-motion-hooks.patch` is retained for reproducing the hooks on
the original public checkout at `59b6c720698c032388ba8ab9d0f3582f869d34ff`.
The fork's `dev` branch already contains these hooks; no patch command is needed.

### Public mjlab compatibility

The original code depended on private `mjlab_research`. Public mjlab lacks some
of its motion interfaces, so changing the directory name alone is insufficient.
This checkout implements the required extensions in SuperMimic:

- `motion_loader.py`: single files, lists and globs, exclusive clip boundaries,
  root-state arrays, and `body_state()` computed from joint/root kinematics.
- `utils/forward_kinematics.py`: differentiable floating-base hinge/slide
  kinematics, including world-frame body-origin velocities and entity joint order.
- `motion_command.py`: external sampling distributions used by the default
  return-based sampler, clip-end truncation, and reset updates that preserve the
  sampled frame and do not advance unrelated environments.
- ONNX exports use public mjlab's per-observation metadata, with the actor's
  sensor history length and one sample per goal. The legacy history-length key
  is retained as a matching alias for existing deployment tools.

The mjlab fork adds two overridable methods, `_load_motion()` and
`_sample_time_steps()`. The default public tracking command keeps its original
loader and sampling behavior; the SuperMimic task configurations select the
extended command and add a `motion_end` timeout.

Motion files must contain `joint_pos` and `joint_vel` in entity joint order, plus
either the public mjlab `body_*_w` arrays (root at body index 0) or explicit
`root_pos_w`, `root_quat_w`, `root_lin_vel_w`, and `root_ang_vel_w` arrays. Use
world-frame root velocities and wxyz quaternions. Non-root targets are recomputed
with forward kinematics to agree with the world-model reward. The clip must
already match the control frequency (50 Hz for these tasks); an optional `fps`
field is checked, and mismatches are rejected instead of silently changing speed.

### Verification

```bash
uv sync --locked
nvidia-smi
uv run python -c 'import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
uv run python -m pytest -q
uv run src/supermimic/scripts/check_env.py --device cuda:0
```

`check_env.py` generates a short motion file, runs both tasks with two simulated
robots, checks finite observations/rewards, and backpropagates through the world
model and kinematics. It does not access W&B. Use `--device cpu` for a CPU smoke
check. To inspect the training options:

```bash
uv run src/supermimic/scripts/train.py SuperMimic-GenCoord-G1 --help
```

Training needs either access to the default W&B motion artifact or a local
motion file supplied with `--motion-file /path/to/motion.npz`. For multiple local
clips, pass a quoted glob or comma-separated paths. For local experiments without
W&B uploads, set `WANDB_MODE=disabled` and supply local motion files.

Downloaded datasets, robot assets and converted motion clips belong in the
ignored `data/` directory. They are local inputs, separate from the source
commits; `check_env.py` and the test suite generate their own motion fixtures.
The migrated local example can be used with
`--motion-file data/motions/run1_subject5.npz`.

### Diagnosing startup stalls

The first world-model and policy updates invoke `torch.compile`; initial
compilation can take several minutes before the next learning log appears.
The runner prints the start and return of each first update, and the learner
prints when CUDA graph capture starts and finishes after warmup.

To isolate compilation/capture from simulation and learning, add `--eager True`
to the training command. This disables PyTorch compilation and learner CUDA
graphs, but MuJoCo/Warp still compiles its simulator kernels. For a short check,
also use `--num-envs 128 --runner.wm.batch-size 64
--runner.policy.batch-size 64 --runner.max-iterations 30
--runner.log-interval 1`. Remove `--eager True` to restore accelerated training.

If MuJoCo Warp reports solver or linesearch iteration limits, try
`--env.sim.mujoco.iterations 100 --env.sim.mujoco.ls-iterations 50`.
These increase the solver budget (and can increase simulation cost); they do
not address PyTorch compilation stalls or guarantee convergence for every state.

Checkpoint files are saved locally in all W&B modes. `WANDB_MODE=disabled`
skips artifact uploads and remote pruning entirely. `WANDB_MODE=offline`
queues artifacts for later sync without querying the online registry.

## Usage

Available task IDs:

- `SuperMimic-GenCoord-G1` — generalized coordinates representation
- `SuperMimic-MaxCoord-G1` — maximal coordinates representation

Train a world model and policy:

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> uv run src/supermimic/scripts/train.py \
  <TASK_ID> --runner.run_name <description>
```

## 评估与可视化

播放配置、W&B checkpoint 选择、三个机器人的含义及仿真警告排查，
统一见 [README_eval.md](README_eval.md)。

## Acknowledgements

SuperMimic contains code inspired by these projects:

- [ControlVAE](https://github.com/heyuanYao-pku/Control-VAE) (BSD-3-Clause)
- [robotic_world_model](https://github.com/leggedrobotics/robotic_world_model) (Apache-2.0)
- [DreamerV3](https://github.com/danijar/dreamerv3) (MIT)

## License

Copyright 2026 The SuperMimic Authors.

Released under the [Apache License 2.0](LICENSE).
