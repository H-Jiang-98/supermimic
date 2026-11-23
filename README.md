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
├── world_model.pt  # 世界模型与仿真上下文参数
├── actor.pt        # 策略网络参数
├── critic.pt       # 价值网络参数（Actor-Critic 模式）
└── wandb/          # W&B 运行记录与调试日志
```

修改 `[runner].log_root` 或传入 `--runner.log-root /path/to/training_logs`
可更换根目录；W&B 本地文件也保存在该运行目录下。
模型按 `[runner].save_interval` 保存（默认每 1000 个采集步），同名 `.pt`
文件会被覆盖，保留最新保存点；不包含优化器状态和 replay buffer。
第一次世界模型和策略更新需要编译，可能几分钟没有新的训练日志；
终端会打印首次更新的开始和完成提示。排查方式见
[Diagnosing startup stalls](#diagnosing-startup-stalls)。

### 4. 离线训练与后续同步

使用 `WANDB_MODE=offline` 启动训练时，模型仍按上述规则保存。
`wandb/offline-run-<时间>-<ID>/` 内的 `run-<ID>.wandb` 保存训练指标、配置
和运行记录，`files/requirements.txt` 保存依赖版本，`logs/` 保存 W&B 调试日志。
待上传的模型版本还会暂存到 `$WANDB_DATA_DIR/artifacts/staging/`
（未设置 `WANDB_DATA_DIR` 时使用 W&B 默认数据目录），占用随保存次数增长。

训练结束且网络可用后，在训练所在机器登录并同步；将下面的路径替换为实际的
`offline-run-*` 目录。同步前保留该目录及模型暂存文件：

```bash
WANDB_MODE=online uv run wandb login
WANDB_MODE=online uv run wandb beta sync \
  --entity "你的账号或团队标识" \
  --project supermimic \
  --no-skip-synced \
  --verbose \
  "logs/SuperMimic-GenCoord-G1/<运行名>/wandb/offline-run-<时间>-<ID>"
```

该命令上传离线记录的指标、配置和模型 artifacts。`WANDB_MODE=disabled`
不会生成可供后续同步的 W&B 离线记录。

通过本地 `download_training.sh` 完整下载日志和 artifacts 后，脚本会校验模型
文件的大小及 MD5，并将 `.wandb` 中的服务器暂存路径映射为本地绝对路径，打印
可在本机执行的同步命令。原始记录保留为 `*.wandb.before-path-map`；模型文件
和训练指标保持原样，脚本本身不上传数据。`--logs-only` 不处理这些路径。

已经下载的训练结果可直接处理，无需再次连接服务器：

```bash
bash download_training.sh --map-only --dry-run # 校验文件和待映射路径
bash download_training.sh --map-only           # 映射并打印本机同步命令
```

修改脚本顶部的 `LOCAL_DOWNLOAD_ROOT`、`TASK_ID`、`RUN_NAME` 可选择其他运行；
`PYTHON_BIN` 默认使用项目 `.venv/bin/python`，需已安装 W&B。

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
The fork's `dev` branch already contains the motion hooks.

The estimator task also uses the fork's `forward_before_rl_signals` option.
For a checkout that predates this option, apply the kinematics patch from `supermimic/`.

```bash
git -C ../mjlab_for_supermimic apply ../supermimic/patches/mjlab-1.6.0-post-step-kinematics.patch
```

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

### GenCoord state frames

GenCoord is different from the original supermimic definition, with root angular velocity 
in the root body frame (the pelvis for G1). The packed layout is unchanged:

```text
full_state = [p_w(3), quat_wxyz(4), qj(N), v_w(3), omega_b(3), vj(N)]
rep_state  = [z(1), g_b(3), v_b(3), omega_b(3), qj(N), vj(N)]
prediction = [delta_v_b(3), delta_omega_b(3), delta_vj(N)]
```

The `full_state` observation's `root_ang_vel` term reads `root_link_ang_vel_b`.
Representation and reconstruction pass this angular velocity through directly;
only linear velocity changes frame. Predicted velocity increments are expressed
in the current root frame; integration transports angular velocity to the next
root frame as `omega_b_next = E.T @ (omega_b + delta_omega_b)`, where
`E = Exp(dt * (omega_b + c * delta_omega_b))` and `c = (m + 1) / (2 * m)` for
`m` simulation substeps. FK and tracking rewards still use world-frame velocities,
with `omega_w = R @ omega_b` at the FK boundary. Motion-file fields ending in `_w`
and MaxCoord retain their world-frame convention.

Network inputs, prediction semantics, parameter shapes, and current-architecture
checkpoints are unchanged. Newly collected replay states use body-frame angular
velocity. Any externally saved GenCoord full states from the previous convention
must first convert their angular block using `omega_b = R.T @ omega_w`; the
identical tensor shape cannot identify the old convention automatically.

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

## Estimator 输入第一步

`eit_rpv_v1` 实现 [plan_step1.md](plan_step1.md) 的输入与网络阶段：默认
5×93 本体历史、700 维 motion 目标、C 组 450 维任务 covariance 编码及
decoder 首层 LayerNorm 后的 FiLM。A/B 完全旁路 FiLM。

奖励保留全局骨盆跟踪与相对身体形态跟踪的混合定义。actor 和全局奖励使用
固定 episode 参考；刚体位姿奖励在评分时按当前真值根位姿的水平位置和 yaw
重新对齐参考。两条 rollout 都用动作执行满 20 ms 后的机器人状态，对照选择该动作时的参考帧评分。
真实环境在最后一个物理子步结束后刷新刚体位姿和速度，随后计算终止与奖励，再执行自动 reset 和参考帧更新。

使用 `SuperMimicRunnerCfg.for_estimator_inputs("C")` 创建新配置，也可选择
`"A"` 或 `"B"`。创建仿真环境前调用
`supermimic.tasks.env_cfgs.configure_estimator_env_cfg(env_cfg)`，以统一测量定义、
关闭重复 corruption 并启用固定 episode 参考对齐。motion sampling mode 仍需
与 runner 一致（`return` 对应 `external`）。新路径仅支持 G1 GenCoord、29 个
关节和 50 Hz。输入阶段 runner 默认使用 eager；联合训练入口 `train_eit` 在 CUDA 上默认编译训练热点，可用 `--eager` 切换。

后验通过 `estimator_inputs.PosteriorInput` 注入：`mean` 是矩阵朝向、世界系
位置和线速度组成的 `RpvState`；C 还需要按 `[theta, rho, nu]` 排列的右局部
9×9 `covariance_rpv`。A 训练使用真值，A 评估与 B/C 必须提供后验均值。
`EITInputConfig(mode="eval")` 控制信息来源；`actor.eval()` 仅控制网络模式。

输入测试可通过 `runner.step(posterior=...)` 注入一帧 RPV 后验，也可向想象环境的 `reset(batch, posterior=...)` 和 `step(action, posterior=...)` 传入测试样本。外部回调 `posterior_provider` 接收 `measurement`（64 维）、`previous_action`（29 维）和 `reset_mask`。手动调用 `runner.reset_collection()` 会将未结束的 episode 标记为截断，保留已采集的帧。

EIT 训练使用独立的采集和想象 provider，actor 和滤波器共用同一帧测量。初始化、采集和想象起点的用法见下节。

`build_ablation_actors(cfg, wm, device)` 显式复制三组共同网络权重及归一化状态，
各组参数存储独立。用 `actor_checkpoint(actor, cfg)` 保存包含输入版本、
噪声配置、结构及归一化状态的 checkpoint，再用 `load_actor_checkpoint` 恢复。
从训练配置切换到 `mode="eval"` 时显式传入 `for_evaluation=True`；其余配置
仍严格匹配。世界模型和估计器后续也须从相同初始权重建立独立实例。

固定 actor 拟合动力学使用 `train_belief`，联合更新世界模型、actor 和 critic 使用 `train_eit`。
play/viewer 已通过共享 EIT 适配层支持本地 `training.pt`，用法见
[EIT 评估与播放](README_eval.md#eit-定量评估和播放)，其中包含 EIT ONNX 导出与共享运行时的命令。

CPU 验收测试（无需完整 MuJoCo 闭环）：

```bash
OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/supermimic-mpl \
  .venv/bin/python -m pytest -q tests/test_estimator_*.py
```

## Estimator 第二步：完整 belief 与随机动力学

[plan_step2.md](plan_step2.md) 的三个实现阶段提供完整 belief 几何、随机模型与
可微滤波核心，以及真实采集/想象 provider。G1 状态交换布局为
`[p_world(3), quat_wxyz(4), q(29), v_world(3), omega_body(3), qdot(29)]`；
70 维右局部误差为 `[theta, rho, nu, omega_body, q, qdot]`。`BeliefState`
持有完整均值和 70×70 covariance，`to_actor_input()` 提取前 9 维边缘块供现有
700 维目标与 450 维 FiLM 分支使用。B/C critic 也只编码后验的 R/p/v 均值与
9×9 边缘 covariance；滤波器和 replay 策略起点使用完整 belief。

`Dynamics(..., process_noise=ProcessNoiseConfig(...))` 启用 2485 维下三角噪声头。
`distribution(rep, action, aux, context)` 返回均值及 L；`sample(..., epsilon)`
在预测均值右侧注入 Lε。L 的对角使用 softplus 加显式下界，每行的非对角系数乘以该行的对角尺度。默认初始标准差
按上述六个误差块依次为 `(0.01, 0.001, 0.05, 0.05, 0.005, 0.05)`，单位与
对应物理量一致，表示一个控制步的过程噪声。均值和噪声头共同注册为模型参数，
每轮分别用多步均值误差和一步过程残差更新。`forward` 返回确定性均值；随机模型使用下述矩阵积分核心。

`LieBeliefFilter` 是纯函数核心，提供 `initialize`、`predict`、`correct`、
`observation` 及 `to_actor_input`。64 维观测的四组标准差必须为有限正数，
并且平方后在运行精度下仍须为有限正数。训练、恢复和评估都执行这一约束。
每帧传入的标准正态样本可以取零。滤波器先将创新 covariance 换算到无量纲坐标。
重力三轴共用尺度，各标量传感器使用各自的尺度。默认容差下，若整批创新矩阵的
Gershgorin 谱界能确认数值满秩，直接使用 Cholesky。其余情况执行谱检查，
数值秩亏时使用可观测子空间的伪逆，
随后执行 Joseph covariance 更新与右局部坐标 reset。调用
`predict/correct(..., return_diagnostics=True)` 可读取 A、innovation、增益、
reset Jacobian、effective_rank 和 jitter。增益求解使用的 jitter 同时进入 Joseph 更新的测量 covariance。
满秩 innovation 求解默认从 `1e-8` 开始尝试 jitter，每次乘 10，最多 5 次，上限 `1e-4`。

在 runner 管理的连续实采或想象段内，provider 可复用上一帧已提交后验的校验，
并通过张量身份和 PyTorch 原地修改版本识别变化。新段、恢复后的首帧和外部输入均重新校验。
在 RTX 5090 上以 B 组、8 个环境、每轮 64 步和策略 microbatch 2 对照 `ff7807c`，
普通帧的 provider 谱分解从 5 次降至 2 次，单环境重置帧从 10 次降至 5 次。
每轮实采平均用时从 2.36 s 降至 2.04 s。[采集配置、计时与同步次数](docs/validation/eit_filter_positive_noise.json)

下面以已创建的 G1 EIT 并行训练环境 `env` 为例，参考动作采用 `uniform` 采样。示例执行一轮训练，每隔两个采集步保存一次后验。

```python
from supermimic.models import ProcessNoiseConfig
from supermimic.runners.eit_training import EITTraining, EITTrainingCfg
from supermimic.runners.supermimic_runner import SuperMimicRunner, SuperMimicRunnerCfg

cfg = SuperMimicRunnerCfg.for_estimator_inputs(
    "C", process_noise=ProcessNoiseConfig(), sampling_mode="uniform", buffer_size=128,
)
runner = SuperMimicRunner(env, cfg, "SuperMimic-GenCoord-G1", device=str(env.device))
training = EITTraining(runner, EITTrainingCfg(posterior_save_stride=2))
training.run_round()
```

默认在真实首帧状态右侧施加独立高斯小扰动，以扰动后的完整状态作先验均值，
各维方差为对应标准差的平方。`cfg.initial_prior.std` 依次设置姿态、位置、线速度、
角速度、关节位置、关节速度的标准差，默认值为
`(0.01, 0.01, 0.05, 0.05, 0.005, 0.05)`，单位依次为 rad、m、m/s、rad/s、rad、rad/s。
首帧融合 o₀ 一次，之后使用实际执行动作和记录的测量逐帧预测、修正。
真实采集可通过 `configure_belief_filter(prior_factory=...)` 注入其他已知先验。
部分 reset 只初始化对应槽位。

provider 持有当前完整后验及其 `episode_id`、`time_index`、`model_version`。
同一 replay 生命周期内，各并行环境的 episode 编号唯一；每个 episode 的
`time_index` 从 0 开始，与参考片段的 `motion_time` 分别计数。runner 每次构建
actor 输入都以当前帧标识向 provider 查询，同帧 restore 后立即使用恢复的后验。

想象起点使用物理帧的 `(episode_id, time_index)` 定位。可用 `sample_policy_starts` 从训练 episode 中抽样，也可用 `prepare_policy_starts(episode_ids, time_indices)` 按指定顺序读取已保存的帧。起点包含该帧动作前的后验、实测历史、潜在状态、参考对齐和 ξ。

```python
wm_env = runner.wm_env
starts = wm_env.sample_policy_starts(4, training.policy_episode_ids())
obs = wm_env.reset_prepared(starts)
obs, reward, termination_probability, _ = wm_env.step(training.actor(obs))
saved = wm_env.snapshot()
obs = wm_env.restore(saved)
```

`reset_prepared` 直接使用保存的后验。之后的每个想象步传播后验、融合新测量，并按潜在真值计算奖励。固定噪声实验可向 `step` 传入标准正态样本 `process_noise` 和 `measurement_noise`。需要核对真实采集记录时，只要物理首帧仍在 replay 中，`episode_prefix(episode_id, t)` 就能取出该 episode 实际使用的首帧先验和到 t 为止的记录。

想象环境的 `snapshot()/restore()` 保存和恢复已提交帧的潜在状态、历史、动作、参考时间与对齐、ξ 及完整后验。恢复后可在相同模型和参考运动上从该帧继续。

想象环境快照使用 `version=2`，与 provider 的 v3 是不同的版本标识。
`motion_digest` 是实际加载参考的 SHA-256 指纹，包含 clip 边界、关节和根状态、
FK 刚体位姿与速度，以及各张量的 shape/dtype。保存和恢复时重新计算，
相同长度但内容不同的 clip、clip 顺序变化或原地修改均会被拒绝，校验发生在恢复
provider 与环境状态之前。文件路径和设备不参与指纹；数据相同的文件可迁移位置。
计算指纹时按帧分块传回 CPU。策略更新在只读阶段入口计算一次参考指纹，在准备起点和整个 rollout 中校验同一参考。

`BeliefProvider.prepare/commit` 将滤波更新分成候选与提交；环境完成目标和
后验校验后再提交。想象步先校验动作和显式噪声，并为噪声保留快照。
构建后验或目标失败时，重试可省略噪声以复用同次采样和计算图。再次传入带梯度的动作或噪声时须使用原张量。
若需放弃想象候选或更换动作，显式 reset。真实首帧候选失败时复用同一先验。物理步完成后若测量或 replay 写入失败，下次 `runner.step()` 继续提交已经执行的动作与结果。

provider 的 `prepare()` 和 `prepare_restore()` 仅在 `differentiable=True` 且调用方开启梯度时保留外层计算图。
使用 `torch.no_grad()` 评估想象序列时，提交的后验和 context 均不带图；局部 Jacobian 仍正常计算。
provider 也尊重调用方的 `inference_mode()`，在内部转换为普通张量后求局部导数。
从评估切回可微调用后，动作与模型梯度继续正常传播。

EIT 用整数 `model_revision` 检查采集、准备起点和想象期间的模型身份。真实 provider 的快照分别记录当前模型的修订号和最后一次滤波使用的 `posterior_revision`。世界模型拟合完成后，真实采集从已提交的后验继续，用更新后的模型处理下一帧。策略想象期间固定模型参数，保留预测对动作的梯度。

`Dynamics.checkpoint()` 和 `load_checkpoint()` 使用 SHA-256 内容指纹 `model_version` 校验模型权重与配置。随机模型的 `world_model.pt` 还保存滤波数值设置、测量标定及首帧高斯配置。`BeliefProvider.state_dict()/load_state_dict()` 可单独保存和恢复在线 belief、context、物理帧标识及滤波配置。

第三步的滤波器要求动力学实现 `mean_lie` 和 `distribution_lie`。矩阵积分在每次
姿态乘法后执行一次 Newton–Schulz 正交化，控制 float32 长序列中的旋转漂移。传播 Jacobian 默认使用
`vmap(jacrev)` 并保留高阶计算图。`FilterConfig(jacobian_backend="eager")`
保留普通 autograd 参考；`jacobian_batch_chunk_size` 和 `jacobian_chunk_size`
控制样本和 Jacobian 行的分块，默认均为 `None`。数值与性能验收见下节。

三个阶段可分别运行以下验收，完整回归使用 `pytest -q`：

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_estimator_lie_state.py
OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_estimator_stochastic_dynamics.py tests/test_estimator_belief_filter.py
OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_estimator_belief_provider.py tests/test_estimator_belief_integration.py tests/test_estimator_replay_restore.py
OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/supermimic-step2-mpl \
  .venv/bin/python -m supermimic.scripts.check_estimator
```

MuJoCo smoke 自动生成四帧静止参考，使用 2 个环境、容量为 2 的环形 buffer、8 步真实采集和两步随机想象。脚本核对后验均值和 covariance、想象回报对 actor 的梯度以及想象帧快照恢复，输出 JSON 诊断。使用 `--device cuda:0` 可运行 GPU 验收。策略/模型联合调度和 belief critic 见第四步。

## Estimator 第三步：拟合预测均值和过程 covariance

`train_belief` 固定 actor。每轮采集结束后，用新训练帧更新模型归一化，然后完成两次拟合。验证 episode 默认满足 `episode_id % 5 == 0`。采样只使用同一 episode 中时间连续的真实后继帧。

`train_belief` 在创建 runner 时选择 `replay_mode="model"`。每帧保存 `state`、`action`、`sim_ctx`、`episode_id`、`time_index`、`terminated` 和 `truncated`。采集端用实测历史和在线滤波产生动作，在执行下一步前核对上一执行动作。训练与验证各自使用可恢复的独立 RNG，均值窗口和 covariance 转移均有放回抽样。

| 阶段 | 数据与目标 | 默认配置 |
|---|---|---|
| 预测均值 | 从真实首帧开始回喂预测，优化根位置、根朝向、关节位置和关节速度的加权 MSE | 8 帧窗口、7 步预测，每批 2048 窗，每轮 10 次 Muon/Adam 更新 |
| 过程 covariance | 均值更新后冻结均值和归一化，仅提取当前抽中转移的一步右局部残差，以高斯 NLL 拟合下三角因子 | 独立 Adam，每批 8192 对，每组默认抽 143,360 对，18 次更新，验证另抽 8192 对 |

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m supermimic.scripts.train_belief \
  --motion-file data/motions/walk1_subject1.npz --device cuda:0 \
  --output logs/belief-b --group B --rounds 10 \
  --mean-window-frames 8 --mean-batch-size 2048 --mean-updates 10 --mean-lr 2e-4 \
  --cov-batch-size 8192 --cov-validation-pairs 8192 --cov-lr 2e-4
```

`metrics.jsonl` 记录两个阶段各自的步数、分组 MSE、每对转移的过程 NLL，以及训练和验证样本量。恢复时传入 `--resume logs/belief-b/training.pt` 和原 motion，`--rounds` 指新增轮数。两个阶段的优化器、配置、归一化和 RNG 均从轮次边界恢复，物理环境和 replay 重新开始。实现与实测结果见[第三步说明](plan_step3.md)。

## Estimator 第四步：完整策略与模型交替训练

[plan_step4.md](plan_step4.md) 的完整训练入口是 `train_eit`。每轮依次采集、
更新模型归一化、拟合均值、拟合过程 covariance、读取缓存后验并更新 actor/critic 归一化、
冻结世界模型执行随机想象策略梯度与 critic 回归，然后保存。进程内跨轮保留物理环境、在线 belief 和观测历史，只有真实 episode 结束才 reset。
`train_belief` 继续用于固定 actor 的模型学习，旧确定性训练仍使用原入口。

A 的 critic 不含 belief；B/C 使用相同结构，额外读取 12 维 R/p/v 后验均值编码和
45 维 9×9 边缘 covariance 编码。均值采用
`[rot6d(R_trueᵀ R_mean), R_trueᵀ(p_mean − p_true), R_trueᵀ(v_mean − v_true)]`；
旋转占 6 维，位置、线速度各占 3 维。covariance 提取完整后验中按
`[δθ, δρ, δν]` 排序的前 9×9 块，在后验均值的右局部切空间编码为 9 个 log std
与 36 个上三角相关系数，保留 R/p/v 的交叉项，不做条件化。
三组 critic 共同读取真值、参考、控制历史、剩余 clip 时间和 context；A 的总输入为
`1234 + dim(context)`，B/C 为 `1291 + dim(context)`。后验角速度、关节位置和
关节速度的均值，以及 9×9 块之外的 covariance，不直接进入 B/C critic。
滤波器传播完整 70×70 covariance。上述压缩只发生在
critic 特征入口，保留 R/p/v 均值和边缘 covariance 的梯度。它是任务相关的信息压缩，
长期价值与控制性能需通过实验验证，不能认为与完整 belief critic 等价。
策略想象的 K 条轨迹共享起点与后验，独立采样过程及测量噪声；冻结参数时保留
动作穿过滤波、covariance、FiLM、soft survival 和价值 bootstrap 的导数。
clip 最后一个评分动作后停止回报，普通策略 horizon 截断保留 bootstrap。

真实环境每帧执行在线滤波，并将转移写入主 replay。`--posterior-save-stride S` 控制后验的保存间隔，默认 1，取值范围为 1 到 `buffer_size`。在全局采集步 0、S、2S…，缓存写入已融合当前测量、尚未执行当前动作的后验。采集步跨轮次和单环境 reset 连续计数。

每个环境的后验容量为 ⌈buffer_size / S⌉。每条记录保存 76 个 float32 均值分量、70×70 对称 covariance 的 2,485 个 float32 下三角元素，以及滤波时的整数修订号。控制历史由主 replay 的 `actor_obs` 和最近淘汰的 `history_length−1` 帧观测拼成。真实 episode 开头以首帧补齐，环形覆盖处使用保留的边界观测。

| 训练步骤 | 使用的帧 |
|---|---|
| 世界模型拟合 | 从主 replay 的训练 episode 取有效窗口和相邻转移，包括未保存后验的帧。 |
| actor/critic 归一化 | 直接读取本轮新采、仍保留且已保存后验的训练帧。按原分块顺序用最新传感器、真实状态和目标更新统计，B/C 的 actor 目标还读取后验均值。整轮校验通过后提交统计，每帧计一次。本轮样本数为零时沿用已有统计。 |
| 策略想象 | 从全部仍保留且已保存后验的训练帧均匀有放回抽样，单帧片段也可入选。样本可来自不同模型修订号。若无可用起点则报错。 |

每轮 `budget` 中，`effective_posterior_count` 是全部有效后验数，`available_policy_starts` 是其中属于训练 episode 的数量，`normalization_samples` 是本轮归一化的帧数。`posterior_revision_gap_*` 汇总当前模型与缓存来源的修订号差距。主 replay、后验、边界历史、序号索引和真实首帧先验分别统计存储字节数。

在线预测与校正在同一次传播中共享中间后验的校验，满秩创新协方差复用成功的 Cholesky 分解。想象步在提交前构建 critic 特征，并直接交给本步的策略或价值计算。一次 RTX 5090 的 B 组对照中，两版均用 8 个环境、每轮 64 个采集步、策略 microbatch 2。第 2–4 轮平均耗时从 6.09 s 降至 4.96 s，其中归一化从 0.522 s 降至 0.076 s。

可用已有联合训练 checkpoint 在 CPU 上采一份 replay，比较不同存储精度的旋转及协方差有效性、字节数和固定噪声下的 actor 梯度。下面的命令写出实验报告。正式训练使用 FP32 后验缓存。[精度实验结果](docs/validation/eit_replay_precision.json)

```bash
OMP_NUM_THREADS=1 PYTORCH_JIT=0 .venv/bin/python -m supermimic.scripts.check_eit_replay_precision \
  --checkpoint logs/eit-v5/B/training.pt --motion-file data/motions/walk1_subject1.npz \
  --num-envs 4 --steps 32 --output /tmp/eit-replay-precision.json
```

`posterior_cache_normalization_seconds` 计入缓存读取和归一化的总耗时，`posterior_cache_read_seconds` 是其中的读取耗时。CUDA 计时在当前 stream 上记录事件，归一化结束时统一读取。默认 5 帧历史、4,096 个环境下，边界观测占约 5.81 MiB，另加索引。S=1、2、4 的实际缓存大小和三轮 CPU 耗时见[增量回放验收记录](docs/validation/incremental_posterior_replay.json)。

A/B/C 的训练、评估和播放均要求联合 checkpoint 内嵌 `eit_rpv_belief_critic_v2` critic 契约。
加载时校验组别、输入维度和 `rpv9_log_std_corr_v1` 编码。A 组也执行同一版本检查。
新训练默认在输出目录的父目录创建或读取 `eit_initialization_v3_eit_rpv_belief_critic_v2_seed<seed>.pt`。
同一 seed 的三组使用同一份初始化包，分别写入各自的训练目录。也可用 `--initialization` 指定当前契约的初始化包。
各组独立采集与更新。当前默认使用公共 ReturnSampler，保留 `--sampling-mode uniform`。

在服务器仓库根目录运行 `python3 src/supermimic/scripts/launch_eit.py --dry-run` 可预览多卡计划，去掉 `--dry-run` 即开始训练。
在 [train_eit_config.toml](train_eit_config.toml) 设置 CPFS 路径、训练超参数、各方法的显卡和 seed。[多卡配置说明](docs/train_eit_pro6000.md#使用-toml-分配显卡和-seed)

```bash
WANDB_MODE=offline OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/supermimic-mpl \
  .venv/bin/python -m supermimic.scripts.train_eit \
  --motion-file data/motions/walk1_subject1.npz --device cuda:0 \
  --group B --output logs/eit-v5/B --max-iterations 1000000 \
  --num-envs 1024 --buffer-size 1000 --replay-ratio 16 \
  --mean-updates 10 --mean-batch-size 2048 --mean-window-frames 8 \
  --cov-batch-size 8192 --cov-validation-pairs 8192 \
  --policy-updates 1 --policy-batch-size 4096 --samples 1 \
  --policy-horizon 32 --policy-microbatch-size 16 \
  --critic-epochs 5 --critic-minibatches 4 --save-interval 1000 --log-interval 10
```

每个物理步调用公共 `runner.step()`。公共 Ratio 按
`num_envs × replay_ratio / (mean_updates × mean_batch_size × mean_window_frames)`
调度 **mean → covariance → policy/critic** 更新组，默认稳态每 10 步一组。
warmup 等待完整训练窗口、验证转移和 posterior；跳过的机会不积累。新训练帧和新保存
posterior 各只更新一次归一化，包括同一步多组训练时。

默认每组为 **10 mean、18 covariance、1 actor、20 critic** 次逻辑更新。
covariance 预算为 `10×2048×(8−1)=143,360` 次有放回抽样，分 17 个 8192 整批和一个
4096 尾批；独立验证抽 8192 对。可用 `--cov-pair-budget` 覆盖训练预算。
两个模型训练入口共用此 fitter；buffer、policy horizon、samples 不改变默认预算，
不再缓存全部残差或遍历 covariance epochs。`train_belief` 仍保留分轮采集。

actor/critic 复用 Muon＋Adam，学习率均为 2e-4。actor 完整逻辑 batch 累积后做 norm 1.0
及逐元素 0.1 clipping。critic 使用原始 value 计算 GAE，只裁剪最终 target，拟合不按累计
survival 加权的 MSE；每次 actor 更新后 5 epoch×4 minibatch，复用一次 permutation。
`--critic-microbatch-size` 独立配置，默认关闭。critic 的片尾零 bootstrap 与 ReturnSampler
的片尾均值 bootstrap 各自独立。完整预算、后验缓存容量及 GPU 启动见
[PRO 6000 说明](docs/train_eit_pro6000.md)；当前没有该规模的 PRO 6000 实测。

唯一周期保存参数为公共 `runner.cfg.save_interval`，CLI 对应 `--save-interval`。
按累计零基物理 `step % save_interval == 0` 触发，另保存初始及结束状态，重复状态不重复发布。
每次保存 `world_model.pt`、`actor.pt`、带输入契约的 `critic.pt` 和原子替换的 `training.pt`，
作为同一个离线 model artifact 归档。覆盖前等待上一份归档结束，不再发布独立历史快照。

EIT 默认 `WANDB_MODE=offline`，测试可设 `disabled`，disabled 仍保存本地权重。
复用公共 W&B 初始化、Logger、artifact 与清理入口。每次启动/恢复都新建 run，记录来源、
起始累计步及 parent run。项目和名称使用公共 runner 配置；日志按 `log_interval` 输出。
没有固定 ID 拼接、`--wandb-mode` 或 `--wandb-log-models`。[离线记录说明](docs/eit_wandb.md)

`max_iterations` 是累计目标，默认 1,000,000，替代 EIT 的 rounds/collection_steps：

```bash
WANDB_MODE=offline OMP_NUM_THREADS=1 .venv/bin/python -m supermimic.scripts.train_eit \
  --motion-file data/motions/walk1_subject1.npz --device cuda:0 \
  --output logs/eit-v5/B --resume logs/eit-v5/B/training.pt --max-iterations 2000000
OMP_NUM_THREADS=1 .venv/bin/python -m supermimic.scripts.eval_eit \
  --checkpoint logs/eit-v5/B/training.pt \
  --motion-file data/motions/walk1_subject1.npz --device cuda:0 \
  --num-envs 8 --output logs/eit-v5/B/evaluation.json
```

当前策略 checkpoint 为 `eit_training_v5`，模型专用为 `eit_model_training_v4`；旧 checkpoint
不兼容。完整状态保存优化器、归一化、独立 RNG、Ratio 余额、累计计数和 ReturnSampler 统计。
恢复重建物理轨迹、replay、在线 belief 和轨迹相关窗口；累计调度及日志继续，新 replay 的
posterior 保存序列从 0 开始。共享初始化仍为 `eit_initialization_v3`。

RTX 5090 已完成默认预算的一组 10/18/1/20 更新、完整规模 checkpoint 恢复及离线 artifact 校验，
A/B/C 小批量 CUDA 下一次更新恢复逐位一致，1095 项回归通过。[实测与范围](docs/validation/eit_v5_rtx5090.json)

`--resume-reference PATH` 在最终 checkpoint 提交前生成固定输入、噪声及 live optimizer
下一次更新的参考结果，供 `check_eit_resume` 独立验证。固定输入连续性不意味着恢复物理轨迹。
`eval_eit` 复用 mjlab tracking 的首 episode 评估循环和跟踪指标，从 clip 头开始，
等待所有环境首个 episode 结束。它关闭动作探索，保留传感噪声及在线滤波，三组均使用
后验均值。可用 `--steps` 限制 smoke 预算，未结束轨迹标记截断，成功率记为 null。
兼容的新 motion 可直接用于评估，训练与评估指纹分别记录；续训校验保持严格。
本地 playback 使用 `play --eit-checkpoint ... --motion-file ...`，详见
[评估统计口径及 viewer 说明](README_eval.md#eit-定量评估和播放)。

`check_eit_training` 默认对 A/B/C 各采集 12 步。以下命令在 B 组复验三个保存间隔，每次都核对固定输入下的下一次模型和策略更新，并在独立进程运行 4 步评估。

```bash
for stride in 1 2 4; do
  OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/supermimic-mpl \
    .venv/bin/python -m supermimic.scripts.check_eit_training \
    --device cpu --groups B --max-iterations 12 --posterior-save-stride "$stride"
done
```

GPU 验收可将设备设为 `--device cuda:0` 并加上 `--full-width` 使用完整网络宽度。

## Acknowledgements

SuperMimic contains code inspired by these projects:

- [ControlVAE](https://github.com/heyuanYao-pku/Control-VAE) (BSD-3-Clause)
- [robotic_world_model](https://github.com/leggedrobotics/robotic_world_model) (Apache-2.0)
- [DreamerV3](https://github.com/danijar/dreamerv3) (MIT)

## License

Copyright 2026 The SuperMimic Authors.

Released under the [Apache License 2.0](LICENSE).
