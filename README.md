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
decoder 首层 LayerNorm 后的 FiLM。A/B 完全旁路 FiLM。旧配置保持原有行为。

奖励保留全局骨盆跟踪与相对身体形态跟踪的混合定义。actor 和全局奖励使用
固定 episode 参考；刚体位姿奖励在评分时按当前真值根位姿的水平位置和 yaw
重新对齐参考。两条 rollout 都用动作执行满 20 ms 后的机器人状态，对照选择该动作时的参考帧评分。
真实环境在最后一个物理子步结束后刷新刚体位姿和速度，随后计算终止与奖励，再执行自动 reset 和参考帧更新。

使用 `SuperMimicRunnerCfg.for_estimator_inputs("C")` 创建新配置，也可选择
`"A"` 或 `"B"`。创建仿真环境前调用
`supermimic.tasks.env_cfgs.configure_estimator_env_cfg(env_cfg)`，以统一测量定义、
关闭重复 corruption 并启用固定 episode 参考对齐。motion sampling mode 仍需
与 runner 一致（`return` 对应 `external`）。新路径仅支持 G1 GenCoord、29 个
关节和 50 Hz，默认 eager 执行。

后验通过 `estimator_inputs.PosteriorInput` 注入：`mean` 是矩阵朝向、世界系
位置和线速度组成的 `RpvState`；C 还需要按 `[theta, rho, nu]` 排列的右局部
9×9 `covariance_rpv`。A 训练使用真值，A 评估与 B/C 必须提供后验均值。
`EITInputConfig(mode="eval")` 控制信息来源；`actor.eval()` 仅控制网络模式。

真实采集入口为 `runner.reset_collection()`、`runner.step(posterior=...)`。
再次调用 `reset_collection()` 会将 replay 中尚未结束的 episode 标记为截断，
保留已有数据，同时防止恢复历史或采样训练序列时跨越手动 reset。
也可在构造 runner 时提供 legacy `posterior_provider`，成功采集时每个时间点调用一次，
关键字参数是 `measurement`（64 维）、`previous_action`（29 维）和
`reset_mask`。actor 和 provider 共用同一次高斯测量，完整 replay 保存这份测量用于重建策略历史。
`imagined_posterior_provider` 应为独立实例；想象入口
`runner.wm_env.reset(batch, posterior=...)` 和 `step(action, posterior=...)`
支持逐步注入，不会沿用上一时刻的注入后验。reset 的 batch 可由
`sample_initial_states()` 从 replay 生成，其中保存原测量历史及 episode 对齐。最早保留帧和保留区内各次真实 reset 都是训练片段起点，起点以前的 actor 历史用该帧同一次测量和实际上一动作填充。
想象环境先校验后验和下一帧输入，再提交时间、状态与历史；校验失败时保留
原环境状态，补齐后验后可重试。直接注入无效后验不会消耗测量噪声；外部
provider 自身的状态更新及其调用前的噪声采样不在环境回滚范围内。
这一节的逐步注入接口使用 RPV 视图。完整滤波后验的初始化和恢复见下方第二步。

`build_ablation_actors(cfg, wm, device)` 显式复制三组共同网络权重及归一化状态，
各组参数存储独立。用 `actor_checkpoint(actor, cfg)` 保存包含输入版本、
噪声配置、结构及归一化状态的 checkpoint，再用 `load_actor_checkpoint` 恢复。
从训练配置切换到 `mode="eval"` 时显式传入 `for_evaluation=True`；其余配置
仍严格匹配。世界模型和估计器后续也须从相同初始权重建立独立实例。

输入阶段可运行采集和短序列计算图验证，`run()` / policy learner `update()` 明确拒绝
完整策略训练。第二步接入真实滤波器，第三步通过 `train_belief` 先拟合预测均值，再拟合过程 covariance。第四步通过 `train_eit` 接通 belief critic、策略优化和短程闭环评估。
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
`observation` 及 `to_actor_input`。64 维观测使用第一步的高斯配置。
滤波器先将创新 covariance 换算到无量纲坐标，再判定有效秩。重力三轴共用尺度，
各标量传感器使用各自的尺度。满秩使用 Cholesky，秩亏使用可观测子空间的伪逆，
随后执行 Joseph covariance 更新与右局部坐标 reset。调用
`predict/correct(..., return_diagnostics=True)` 可读取 A、innovation、增益、
reset Jacobian、effective_rank 和 jitter。增益求解使用的 jitter 同时进入 Joseph 更新的测量 covariance。
满秩 innovation 求解默认从 `1e-8` 开始尝试 jitter，每次乘 10，最多 5 次，上限 `1e-4`。

真实采集可通过 runner 绑定同一个模型及独立的两个 provider：

```python
from supermimic.models import ProcessNoiseConfig

cfg = SuperMimicRunnerCfg.for_estimator_inputs(
    "C", process_noise=ProcessNoiseConfig(),
)
# 用上面的 cfg 创建 env 和 runner。
real, imagined = runner.configure_belief_filter()
runner.reset_collection()
for _ in range(collection_steps):
    runner.step()
runner.finish_collection_round()
```

默认在真实首帧状态右侧施加独立高斯小扰动，以扰动后的完整状态作先验均值，
各维方差为对应标准差的平方。`cfg.initial_prior.std` 依次设置姿态、位置、线速度、
角速度、关节位置、关节速度的标准差，默认值为
`(0.01, 0.01, 0.05, 0.05, 0.005, 0.05)`，单位依次为 rad、m、m/s、rad/s、rad、rad/s。
首帧融合 o₀ 一次，之后使用实际执行动作和记录的测量逐帧预测、修正。
真实采集可通过 `configure_belief_filter(prior_factory=...)` 注入其他已知先验。
部分 reset 只初始化对应槽位。EIT 策略回放从保留片段的首帧真值独立抽取配置的高斯先验。

provider 持有当前完整后验及其 `episode_id`、`time_index`、`model_version`。
同一 replay 生命周期内，各并行环境的 episode 编号唯一；每个 episode 的
`time_index` 从 0 开始，与参考片段的 `motion_time` 分别计数。runner 每次构建
actor 输入都以当前帧标识向 provider 查询，同帧 restore 后立即使用恢复的后验。

真实 provider 的 v3 快照记录实际使用的在线首帧先验。完整 replay 另外记录这份先验，在物理首帧仍保留时可通过 `episode_prefix` 核对实际前缀。
完整 replay 在 `finish_collection_round()` 时为每个保留片段建立策略先验。重复读取或更新模型时复用同一份抽样，覆盖片段首帧后才为新的起点抽样。首次按键重放也会自动完成本轮分段。
想象 reset 接收标量或每个槽位各自的索引。`time_index` 定位物理 episode 中的帧，`segment_time` 定位保留片段中的帧。

```python
obs = runner.wm_env.reset(episode_id=episode_ids, segment_time=0)
obs, reward, termination_probability, _ = runner.wm_env.step(
    action, process_noise=epsilon70, measurement_noise=eta64,
)
saved = runner.wm_env.snapshot()
obs = runner.wm_env.restore(saved)
```

reset 在当前模型版本上重建从片段起点到目标帧的完整后验，以及同一目标帧的 actor 历史、潜在状态、参考和 ξ。该重建不保留历史模型学习图，返回的后验作为策略想象的初始条件。均值拟合直接从窗口首帧真值开始。所选物理帧已被覆盖或片段内部断开时，reset 报错并保留原 rollout。
想象环境的 `snapshot()/restore()` 保存和恢复已提交帧的潜在状态、历史、动作、参考时间与对齐、ξ 及完整后验。可在相同模型和参考运动上直接从纯想象帧继续，恢复时不抽样、不重复融合测量。后续两个噪声参数均为标准正态样本，省略时各采样一次。奖励使用潜在真值。

想象环境快照使用 `version=2`，与 provider 的 v3 是不同的版本标识。
`motion_digest` 是实际加载参考的 SHA-256 指纹，包含 clip 边界、关节和根状态、
FK 刚体位姿与速度，以及各张量的 shape/dtype。保存和恢复时重新计算，
相同长度但内容不同的 clip、clip 顺序变化或原地修改均会被拒绝，校验发生在恢复
provider 与环境状态之前。文件路径和设备不参与指纹；数据相同的文件可迁移位置。
计算按帧分块传回 CPU，仅在快照边界执行。旧 v1 或缺少指纹的快照无法验证参考身份，
必须从保留 replay 重新 reset 并生成新快照，不能仅改版本号后加载。

`BeliefProvider.prepare/commit` 将滤波更新分成候选与提交；环境完成目标和
后验校验后再提交。想象步先校验动作和显式噪声，并为噪声保留快照。
构建后验或目标失败时，重试可省略噪声以复用同次采样和计算图。再次传入带梯度的动作或噪声时须使用原张量。
若需放弃想象候选或更换动作，显式 reset。真实首帧候选失败时复用同一先验。物理步完成后若测量或 replay 写入失败，下次 `runner.step()` 继续提交已经执行的动作与结果。

provider 的 `prepare()` 和 `prepare_restore()` 仅在 `differentiable=True` 且调用方开启梯度时保留外层计算图。
使用 `torch.no_grad()` 评估想象序列时，提交的后验和 context 均不带图；局部 Jacobian 仍正常计算。
provider 也尊重调用方的 `inference_mode()`，在内部转换为普通张量后求局部导数。
从评估切回可微调用后，动作与模型梯度继续正常传播。

`Dynamics.model_version` 在 provider 校验时根据当前权重、归一化和过程配置计算内容指纹，
因此 fused Adam 更新后，恢复、推进及查询旧后验都会触发版本检查。真实 provider 默认使用
`freeze_estimator_normalization=True` 固定本阶段采集的模型和 actor 归一化
快照。后续模型更新需要重建活动采集器后验。模型参数
固定时使用 `requires_grad_(False)`，保留预测对动作的梯度。`Dynamics.checkpoint()`
和 `load_checkpoint()` 对保存的权重快照计算并核对相同的内容指纹。随机模型的 `world_model.pt` 一并保存滤波数值设置、测量标定及首帧高斯配置。在线状态另外通过
`BeliefProvider.state_dict()/load_state_dict()` 保存完整 belief、context、episode 序号、帧序号、模型版本
和滤波配置。

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

MuJoCo smoke 自动生成四帧静止参考，使用 2 个环境、容量为 2 的环形 buffer、8 步真实采集和两步随机想象。脚本独立核对从保留片段重建的完整均值和 covariance，再检查想象回报对 actor 的梯度和纯想象帧快照恢复，输出 JSON 诊断。使用 `--device cuda:0` 可运行 GPU 验收。策略/模型联合调度和 belief critic 见第四步。

## Estimator 第三步：拟合预测均值和过程 covariance

`train_belief` 固定 actor。每轮采集结束后，用新训练帧更新模型归一化，然后完成两次拟合。验证 episode 默认满足 `episode_id % 5 == 0`。采样只使用同一 episode 中时间连续的真实后继帧。

`train_belief` 在创建 runner 时选择 `replay_mode="model"`。每帧保存 `state`、`action`、`sim_ctx`、`episode_id`、`time_index`、`terminated` 和 `truncated`。采集端继续用实测历史和在线滤波产生动作，在执行下一步前核对上一执行动作。模型 replay 封存和建立窗口索引均不消耗随机数。均值窗口抽样和 covariance 批次洗牌仍使用训练 RNG，旧版本与新版在相同 seed 下可能产生不同的训练轨迹。

| 阶段 | 数据与目标 | 默认配置 |
|---|---|---|
| 预测均值 | 从真实首帧开始回喂预测，优化根位置、根朝向、关节位置和关节速度的加权 MSE | 8 帧窗口、7 步预测，每批 2048 窗，每轮 10 次 Muon/Adam 更新 |
| 过程 covariance | 按本轮更新后的均值计算全部一步右局部残差，以零均值高斯 NLL 拟合完整下三角因子 | 独立 Adam，每批 256 对，每轮打乱遍历一次，含末批 |

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m supermimic.scripts.train_belief \
  --motion-file data/motions/walk1_subject1.npz --device cuda:0 \
  --output logs/belief-b --group B --rounds 10 \
  --mean-window-frames 8 --mean-batch-size 2048 --mean-updates 10 --mean-lr 2e-4 \
  --cov-batch-size 256 --cov-epochs 1 --cov-lr 2e-4
```

`metrics.jsonl` 记录两个阶段各自的步数、分组 MSE、每对转移的过程 NLL，以及训练和验证样本量。恢复时传入 `--resume logs/belief-b/training.pt` 和原 motion，`--rounds` 指新增轮数。两个阶段的优化器、配置、归一化和 RNG 均从轮次边界恢复，物理环境和 replay 重新开始。实现与实测结果见[第三步说明](plan_step3.md)。

## Estimator 第四步：完整策略与模型交替训练

[plan_step4.md](plan_step4.md) 的完整训练入口是 `train_eit`。每轮依次采集、
更新模型归一化、拟合均值、拟合过程 covariance、重建后验并更新 actor/critic 归一化、
冻结世界模型执行随机想象策略梯度与 critic 回归，然后保存和 reset。
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

每轮模型更新完毕后，先从所有保留的训练帧抽取本轮各次策略更新的起点，单帧片段也可入选。
`train_eit` 使用默认的完整 replay，按 episode 从片段先验顺序滤波，直到最后一个需要的帧。本轮新采集且仍保留的训练帧
参与 actor/critic 归一化，每帧计一次。保存抽中起点的后验与控制历史，归一化更新后
按原抽样顺序启动 rollout。起点准备和策略更新期间，参考 motion 张量及其底层存储只读；
进入该阶段时计算一次内容摘要，随后各 microbatch 共用。

A/B/C 的训练、评估和播放均要求联合 checkpoint 内嵌 `eit_rpv_belief_critic_v2` critic 契约。
加载时校验组别、输入维度和 `rpv9_log_std_corr_v1` 编码。A 组也执行同一版本检查。
新训练默认在输出目录的父目录创建或读取 `eit_initialization_v3_eit_rpv_belief_critic_v2_seed<seed>.pt`。
同一 seed 的三组使用同一份初始化包，分别写入各自的训练目录。也可用 `--initialization` 指定当前契约的初始化包。
各组独立采集与更新。以下是初始工程配置，尚未验证长时控制收敛。

```bash
OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/supermimic-mpl \
  .venv/bin/python -m supermimic.scripts.train_eit \
  --motion-file data/motions/walk1_subject1.npz --device cuda:0 \
  --group B --output logs/eit-v3/B \
  --num-envs 8 --rounds 10 --wandb-mode offline
```

默认 N=4、K=2、H=4、microbatch=2，每轮一次 actor 更新及两轮 critic 回归；
两个训练入口接受相同的 `--mean-*` 和 `--cov-*` 参数，默认配置见第三步。
`--policy-horizon`、`--samples`、`--policy-batch-size`、`--policy-microbatch-size`
分别控制想象长度、噪声轨迹数、起点数与分块。更大规模应先测量完整训练显存。

`train_eit` 每轮原子更新完整的 `training.pt`，用于断点续训；历史快照默认只保留
第 0 轮及累计轮数为 10 的倍数的状态。用 `--snapshot-interval N` 调整历史快照间隔，
单位为轮，N 必须为正整数；设为 1 恢复逐轮保存。结束轮数不是间隔倍数时，最新状态
仍保存在 `training.pt`，不额外发布历史快照。此参数不使用原 `runner.save_interval`。
恢复时省略该参数会沿用 checkpoint 中的间隔，旧 checkpoint 没有该字段时沿用 1；
显式传入可改变后续保存间隔，已发布快照不删除。评估清单中的候选轮次必须有相应快照。
三卡 A/B/C 的完整命令见 [PRO 6000 训练说明](docs/train_eit_pro6000.md)。

EIT 默认启用离线 W&B；`WANDB_MODE=disabled` 或 `--wandb-mode disabled` 可关闭。
每轮记录模型损失、策略/critic 指标、累计实采帧数、预算、耗时与显存，同时保存超参数。
训练期间不登录或上传。续训沿用输出目录内的 run ID，生成新的离线片段；之后按训练
顺序用 `wandb sync --append` 同步。默认只记录指标和配置；加 `--wandb-log-models`
可将按间隔保存的快照作为模型附件暂存，复用原有 artifact 实现。
参数、文件位置及同步命令见 [EIT 离线 W&B 说明](docs/eit_wandb.md)。

恢复时配置和组别从 checkpoint 读取，`--rounds` 表示新增轮数：

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m supermimic.scripts.train_eit \
  --motion-file data/motions/walk1_subject1.npz --device cuda:0 \
  --output logs/eit-v3/B --resume logs/eit-v3/B/training.pt --rounds 10
OMP_NUM_THREADS=1 .venv/bin/python -m supermimic.scripts.eval_eit \
  --checkpoint logs/eit-v3/B/training.pt \
  --motion-file data/motions/walk1_subject1.npz --device cuda:0 \
  --num-envs 8 --output logs/eit-v3/B/evaluation.json
```

`eit_training_v3` 记录本轮使用的 `replay_mode` 及均值 Muon/Adam、covariance Adam 的状态。续训时核对 replay 模式。策略训练另存 actor、critic 及其优化器、归一化和随机数流。两个训练入口、评估与播放均只接受 v3 轮次存档，初始化包只接受 `eit_initialization_v3`。
恢复前在私有副本上校验全部组件。恢复后清空 replay 并重置真实环境；固定相同
原始输入和噪声可以验证优化器续训一致，但不承诺恢复物理模拟轨迹。
`eval_eit` 复用 mjlab tracking 的首 episode 评估循环和跟踪指标，从 clip 头开始，
等待所有环境首个 episode 结束。它关闭动作探索，保留传感噪声及在线滤波，三组均使用
后验均值。可用 `--steps` 限制 smoke 预算，未结束轨迹标记截断，成功率记为 null。
兼容的新 motion 可直接用于评估，训练与评估指纹分别记录；续训校验保持严格。
本地 playback 使用 `play --eit-checkpoint ... --motion-file ...`，详见
[评估统计口径及 viewer 说明](README_eval.md#eit-定量评估和播放)。

可复现两轮真实训练、固定输入恢复和独立评估验收：

```bash
OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/supermimic-mpl \
  .venv/bin/python -m supermimic.scripts.check_eit_training --device cpu
OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/supermimic-mpl \
  .venv/bin/python -m supermimic.scripts.check_eit_training --device cuda:0 --full-width
```

实测的训练规模、恢复检查和推理验证见[第三步验收记录](plan_step3.md#实测验收)。

## Acknowledgements

SuperMimic contains code inspired by these projects:

- [ControlVAE](https://github.com/heyuanYao-pku/Control-VAE) (BSD-3-Clause)
- [robotic_world_model](https://github.com/leggedrobotics/robotic_world_model) (Apache-2.0)
- [DreamerV3](https://github.com/danijar/dreamerv3) (MIT)

## License

Copyright 2026 The SuperMimic Authors.

Released under the [Apache License 2.0](LICENSE).
