# EIT v6 训练

EIT（estimator in the loop）将在线状态估计、随机世界模型和策略训练连接起来。
当前联合训练协议为 `eit_training_v6`，支持 G1 GenCoord、29 个关节、50 Hz 控制。
安装与 motion 数据准备见 [环境安装](docs/setup.md) 和 [数据转换](README.md#1-批量处理-lafan1_g1-数据)。

## 模块设计概览

| 模块 | 设计概览 | 实现说明 |
|---|---|---|
| 输入与 A/B/C 分组 | actor 读取实测历史与参考目标；C 通过 FiLM 使用任务不确定性 | [输入设计](docs/eit_inputs.md) |
| 状态估计 | 在 70 维右局部误差空间传播完整 belief，使用平方根滤波 | [滤波](docs/eit_filter.md) |
| Replay | 保存真实转移与稀疏后验，构造因果历史和可微想象起点 | [Replay](docs/eit_replay.md) |
| 世界模型 | 分阶段拟合多步预测均值和一步过程噪声 covariance | [训练实现](docs/eit_training.md) |
| Actor–critic | 固定时域想象、路径梯度与 GAE；与模型更新交替执行 | [训练实现](docs/eit_training.md) |
| CUDA 执行 | 编译热点；可将完整 actor–critic 更新捕获为一张图 | [CUDA Graph](docs/eit_cuda_graphs.md) |
| 运行与评估 | 保存、恢复、离线日志、pilot 和独立诊断 | [运行说明](docs/eit_operations.md)、[批量评估](docs/eit_batch.md) |

每个物理步先采集并更新在线 belief、replay 和归一化，再由 replay ratio 调度训练组。
每组依次拟合世界模型均值、过程 covariance，随后固定世界模型执行 actor–critic 更新。
策略训练固定展开 H 步；越过参考片尾后保持完整末帧并继续计分，末步使用价值 bootstrap。
真实采集和整段评估仍在片尾结束。

| 组别 | 训练时构造 actor 目标的 R/p/v | actor 不确定性输入 | critic |
|---|---|---|---|
| A | 真值 | 无 | 共同特权输入 |
| B | 在线后验均值 | 无 | 共同输入 + 后验 R/p/v 均值与 covariance |
| C | 在线后验均值 | 450 维任务 covariance，FiLM 调制 | 与 B 相同 |

正式评估时三组都运行滤波器并用后验均值构造 actor 目标。
B−A 同时改变训练目标来源和 critic 输入；C−B 比较加入不确定性与 FiLM 的效果。
同一 seed 共享世界模型、actor 共同部分及 B/C critic 初值，各组独立采集和更新。

## 基于 train_eit_config.toml 训练

以下命令在仓库根目录执行。配置入口为 [`train_eit_config.toml`](train_eit_config.toml)，
由 `launch_eit.py` 读取，再转换为每个 `train_eit` 子进程的命令行参数。
`train_eit` 本身不接收这个 TOML 的 `--config` 参数。

### 1. 设置路径和实验名

仓库中的配置包含服务器路径；在启动前将 `[paths]` 改为自己的环境，例如：

```toml
[paths]
environment_root = "."
dataset_root = "data"
motion_file = "motions/*.npz"
checkpoint_root = "artifacts"
experiment = "eit_v6_lafan_run1"
```

该示例使用仓库内 `.venv/bin/python` 和 `data/motions/*.npz`。
`environment_root` 必须指向包含 `.venv/bin/python` 的目录，环境中需能导入当前代码；
若该目录有 `env.sh`，启动器会先加载它。
三个根目录的相对路径以 **TOML 所在目录** 为基准。
`motion_file` 相对 `dataset_root`，支持单文件或 glob；`experiment` 相对 `checkpoint_root`。
后两项不能是绝对路径，也不能包含 `..`。

每批新实验使用新的 `experiment`。输出目录和初始化文件已存在时，启动器不会自动续训。

### 2. 分配 GPU、方法和 seed

```toml
[methods.A]
gpus = [0]
seeds = [2027]

[methods.B]
gpus = [1]
seeds = [2027]

[methods.C]
gpus = [2]
seeds = [2027]
```

`gpus` 使用物理 GPU 编号。每个方法将 seeds 按顺序轮流分配给这些 GPU；
同一 GPU 上的任务串行，不同 GPU 并行，使用独立进程。
例如 `gpus = [0, 1]`、`seeds = [2027, 2028, 2029]` 会让 GPU 0 依次跑 2027、2029。
多个方法可以共用同一张 GPU；只训练 B 时删除 A、C 两个方法表。
所有方法共用 `[training]`，每个 seed 先由 B 组生成一次共享初始化，再开始训练。

仓库当前配置为 A/B/C 分别使用 GPU **0/6/7**、seed **2027**，请按机器实际情况修改。

### 3. 设置训练预算

下表对应仓库当前 TOML 的值，单次 `train_eit` 的代码默认值可能不同。

| TOML 键 | 当前值 | 含义 |
|---|---|---|
| `num_envs` / `max_iterations` | 2048 / 300000 | 并行环境数 / 每个环境累计物理步数 |
| `buffer_size` / `posterior_save_stride` | 1000 / 1 | 每环境保留帧数 / 后验保存步距 |
| `sampling_mode` / `replay_ratio` | `return` / 16 | 参考起点采样方式 / 更新频率预算 |
| `mean_updates` / `mean_batch_size` / `mean_window_frames` | 10 / 2048 / 8 | 每组均值拟合预算 |
| `cov_batch_size` / `cov_validation_pairs` | 8192 / 8192 | covariance 训练 batch / 验证抽样数 |
| `policy_updates` / `policy_batch_size` / `samples` | 1 / 2048 / 1 | 每组 actor 更新 / 起点数 / 每起点轨迹数 |
| `policy_horizon` / `policy_microbatch_size` | 32 / 1024 | 想象时域 / 每次反传轨迹数 |
| `critic_epochs` / `critic_minibatches` | 5 / 4 | 每次 actor 更新后的 critic 拟合预算 |
| `mean_lr` / `cov_lr` / `policy_lr` / `critic_lr` | 均为 0.0002 | 各优化器学习率 |
| `exploration` | 0.0 | 真实采集动作探索噪声标准差 |
| `log_interval` / `save_interval` | 10 / 1000 | 物理步日志 / 保存间隔 |

未指定 `cov_pair_budget` 时，每组按 `mean_updates × mean_batch_size × (mean_window_frames−1)`
抽样，当前为 143360 对、18 次 covariance 更新。
当前配置稳态每 5 个物理步调度一组，通常为 **10 mean → 18 covariance → 1 actor → 20 critic**。
数据不足时跳过本次机会，实际次数以日志为准。

网络宽度由 `mean_hidden_dims`、`cov_hidden_dims`、`actor_hidden_dims`、
`critic_hidden_dims` 控制。更改结构后使用新的实验名和初始化文件。
`samples = 1` 时想象潜在状态使用模型均值，但仍保留观测噪声与滤波过程 covariance。

### 4. 选择执行模式与物理求解参数

```toml
[training]
eager = false
policy_cuda_graphs = true
jacobian_backend = "increment"
jacobian_batch_chunk_size = 512

[env.sim.mujoco]
iterations = 20
ls_iterations = 50
```

这些是已有表中的字段，修改原表即可。CUDA 默认编译热点；`eager = true` 同时关闭编译和
显式 policy CUDA Graph，便于排查。当前配置启用完整 actor–critic 图；
`policy_batch_size × samples` 不能整除 actor microbatch 时，整次更新回退普通调用。
`increment` 是 Jacobian 后端，与 `eager` 编译开关独立。
求解器参数只影响物理环境，不是训练更新次数。

### 5. 预览并启动

```bash
python3 src/supermimic/scripts/launch_eit.py --config train_eit_config.toml --dry-run
WANDB_MODE=offline python3 src/supermimic/scripts/launch_eit.py --config train_eit_config.toml
```

预览会列出每张 GPU 的初始化及训练命令，不启动工作进程，也不验证路径是否存在。
核对路径、GPU 和预算后执行第二条命令。无需 W&B 记录时使用 `WANDB_MODE=disabled`。
EIT 当前仅支持 offline/disabled。首次模型更新和图捕获可能较慢。

先做短程检查时，可复制 TOML 为 `train_eit_smoke.toml`，修改实验名、GPU 和以下字段：

```toml
# 替换副本 [training] 中的对应字段，其他字段沿用。
num_envs = 8
max_iterations = 40
buffer_size = 32
mean_updates = 1
mean_batch_size = 8
mean_window_frames = 4
cov_batch_size = 16
cov_validation_pairs = 16
policy_batch_size = 8
policy_microbatch_size = 4
policy_horizon = 4
mean_hidden_dims = [32, 32]
cov_hidden_dims = [16]
actor_hidden_dims = [32, 16]
critic_hidden_dims = [16]
log_interval = 1
save_interval = 20
eager = true
policy_cuda_graphs = false
```

用 `--config train_eit_smoke.toml` 预览和启动。检查日志中模型与策略更新计数实际增加、
损失有限且保存成功；短程测试通过只代表训练链路可运行。
恢复正式配置时使用新的实验名，避免复用小网络初始化。

### 6. 查看产物与续训

```text
<checkpoint_root>/<experiment>/
├── init/seed2027.pt
└── seed2027/
    ├── A/
    ├── B/
    └── C/
        ├── training.pt
        ├── world_model.pt
        ├── actor.pt
        ├── critic.pt
        ├── metrics.jsonl
        ├── profiling.jsonl
        └── wandb/
```

每组保存本地最新四文件；`training.pt` 用于续训。日志关注实际更新计数、
模型留出误差、策略回报与阶段耗时，字段含义见 [运行说明](docs/eit_operations.md)。

启动器不自动续训。以下示例接续前面的本地路径配置，并保留 TOML 的求解器与执行选项：

```bash
CUDA_VISIBLE_DEVICES=2 WANDB_MODE=offline OMP_NUM_THREADS=1 \
  PYTORCH_ALLOC_CONF=expandable_segments:True .venv/bin/python -m supermimic.scripts.train_eit \
  --motion-file 'data/motions/*.npz' --device cuda:0 \
  --resume artifacts/eit_v6_lafan_run1/seed2027/C/training.pt \
  --output artifacts/eit_v6_lafan_run1/seed2027/C --max-iterations 400000 \
  --env.sim.mujoco.iterations 20 --env.sim.mujoco.ls-iterations 50 \
  --policy-cuda-graphs
```

`max_iterations` 是包含恢复前步数的累计目标。组别、训练 seed、网络、学习率、
replay/策略配置和 Jacobian 设置从 checkpoint 恢复；motion 必须与原训练匹配。
物理轨迹、replay 和在线 belief 重新建立，先重新积累训练样本。
编译/graph 开关由本次命令决定；自定义 MuJoCo 求解器参数需像示例一样再次传入。
详细恢复边界见 [训练实现](docs/eit_training.md#保存与恢复)。
评估、播放和 ONNX 导出见 [README_eval.md](README_eval.md)。
