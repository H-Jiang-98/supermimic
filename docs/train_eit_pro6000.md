# PRO 6000 上的 EIT 训练、预算与续训

当前联合训练协议为 `eit_training_v6`。EIT 保留在线 belief、后验 replay、随机世界模型与可微 imagined belief rollout；采集调用公共 `runner.step()`，更新由公共 `Ratio` 调度，优化器、critic fitting、权重文件和 W&B 记录复用 SuperMimic。采用 float32。CUDA 默认使用 TorchInductor 编译训练热点，CPU 使用 eager。单次训练加 `--eager` 或在 TOML 的 `[training]` 设置 `eager = true` 可切换回 eager。显式 learner CUDA Graph 独立于该开关，当前关闭。

观测噪声默认使用未截断高斯分布，标准差为原均匀噪声半宽 / 3，使 ±3σ 对齐原范围。策略训练在 `num_samples = 1` 时直接使用 world model 的 mean prediction，不采样过程噪声；`num_samples > 1` 时保留过程噪声采样。两种模式都保留观测噪声和滤波器过程协方差。checkpoint 中显式保存的观测噪声配置仍按保存值恢复。

下表记录服务器上的三组短程对照。长期收敛及 replay 填满后的耗时还需单独测量。

新启动的 `train_eit` 默认使用 `--jacobian-backend increment`，也可在 TOML
的 `[training]` 设置 `jacobian_backend = "increment"`。先计算网络输出
`[Δv_b(3), Δω_b(3), Δv_j(29)]` 相对 70 维右局部扰动的导数 `G`，再对现有
积分函数计算直接状态导数 `A` 和增量导数 `B`，以 `F = A + B @ G` 组装
70×70 Jacobian。积分部分仍通过自动微分计算，包含旋转正交化、新机体坐标系
角速度和 nominal 局部误差，保留策略训练所需的外层梯度。穿过 world model
的反向方向从 70 减为 35；整体耗时仍取决于积分、滤波和其他训练步骤。
`--jacobian-backend vmap` 可切回原完整状态求导，`eager` 为逐行参考后端。
`--eager` 仅控制编译，与 Jacobian 后端独立。续训保持 checkpoint 保存的
后端设置；旧 checkpoint 继续使用原后端。

2026-11-24 的固定输入微基准使用服务器 GPU 2（NVIDIA RPBZZZ6）、float32、
batch/chunk=256、三层 2048 隐藏单元，冻结 world model 参数。每个分支预热
3 次，再取 10 次同步墙钟计时的中位数，不包含编译时间。外层反传同时覆盖
动作和状态各字段。结果如下，显存为相对分支常驻张量的临时 allocated 峰值：

| 编译模式 | 完整状态 vmap → increment 耗时 | 临时显存 vmap → increment |
|---|---:|---:|
| 仅 Jacobian | 7.65 → 5.12 ms（减少 33.0%） | 295.4 → 150.9 MiB |
| Jacobian + 外层反传 | 17.78 → 11.38 ms（减少 36.0%） | 908.8 → 448.5 MiB |

这不是整轮训练计时。未编译 eager 下，新增积分求导的调度开销使本次外层
反传从 61.32 增至 64.57 ms；需要 eager 吞吐时可选回 `vmap`。同一大网络
float32 对照中，编译后 Jacobian 最大绝对差约 1.94e-6，外层梯度约 5.34e-5。
float64 有限差分及混合导数、CUDA 编译、A/B/C 策略 rollout、训练与恢复等
相关回归共 198 项通过。[原始微基准数据](validation/eit_increment_jacobian.json)。

2026-11-23 已在本地 RTX 5090（32 GiB）完成旧版 v5 协议验证：[实测报告](validation/eit_v5_rtx5090.json)。C 组使用下表默认预算及完整网络宽度，Jacobian batch chunk=32、`PYTORCH_JIT=0`；采集 11 步后完成一组 10/18/1/20 更新。世界模型阶段 0.555 s，策略阶段 691.753 s（其中 critic 0.074 s）；PyTorch 峰值 allocated/reserved 为 15.20/17.76 GiB。新进程恢复至累计 14 步，优化器和调度状态一致，两次启动生成五个四文件 artifact，离线记录约 0.90 GiB。A/B/C 小批量 CUDA 下一次更新恢复逐位一致，完整回归 1095 项通过。

该实测只覆盖一个完整默认更新与恢复后的 warmup；1000 帧缓存已分配但尚未填满，未测稳定容量索引成本或收敛，也不能推算为 PRO 6000 性能。阶段计时存在包含关系，不能把父阶段与子阶段相加。

## CUDA 编译对照（2026-11-24）

服务器报告 GPU 为 NVIDIA RPBZZZ6（每卡 97,887 MiB），PyTorch 版本为 2.14.0+cu130。A/B/C 分别使用物理卡 1/2/6，同一 seed 2027 公共初始化、40 个动作文件和 float32。每组设 2048 个环境、4096 个策略起点、horizon 32、actor microbatch 和 Jacobian chunk 均为 256。使用服务器已有 eager 日志，只运行编译侧，参数 `--max-iterations 21` 对齐从零计数的日志 `step=20`。

下表的每组耗时取最后一个日志区间，由两组更新的阶段总时间除以 2 得到。两种模式至此均完成 30 次 mean、54 次 covariance、3 次 actor 和 60 次 critic 更新。该区间的实际想象转移数也逐组吻合（A 262,019，B/C 262,052）。显存取进程最大 allocated，单位为 GiB。

| 组 | rollout 加反传，eager → 编译（秒/组） | 耗时下降 | 世界模型，eager → 编译（秒/组） | 峰值显存，eager → 编译（GiB） |
|---|---:|---:|---:|---:|
| A | 143.98 → 113.00 | 21.5% | 0.873 → 0.442 | 69.78 → 42.72 |
| B | 150.95 → 140.66 | 6.8% | 0.762 → 0.441 | 70.38 → 42.78 |
| C | 149.09 → 146.99 | 1.4% | 0.726 → 0.445 | 70.70 → 43.07 |

首次完整更新所在的 `step=10` 日志中，进程累计编译时间 A/B/C 分别为 80.13、105.09、105.36 秒。到 `step=20` 又分别增加约 4.57、3.32、3.98 秒，三组的累计编译图数均从 23 增至 25。各进程可复用已有的磁盘编译缓存，阶段耗时包含在该阶段触发的编译。C 组最后两次更新的耗时差较小，长期吞吐还需更长区间测量。

原始数据位于服务器 `/mnt/cpfs/zbl-cpfs-new/CKPT/leojiang/supermimic_ws/`。eager 实验目录为 `eit_v5_Lafan`，编译实验目录为 `eit_v5_Lafan_compiled_kiGq1fqz`。组内的 `seed2027/{A,B,C}/metrics.jsonl` 提供预算和计数，`profiling.jsonl` 提供阶段时间、显存及编译累计值。

## A/B/C 与公共初始化

| 组 | 示例 GPU | 训练 actor 的目标来源 | 不确定性输入 | critic |
|---|---:|---|---|---|
| A | 0 | 真值 R、p、v | 无 | 共同特权输入 |
| B | 1 | 在线后验均值 | 无 | 共同输入，加后验均值和协方差编码 |
| C | 2 | 在线后验均值 | 450 维任务 covariance，通过 FiLM 调制 | 与 B 相同 |

正式评估三组 actor 都用在线后验均值，A 也运行完整滤波器。B−A 同时改变训练目标来源和 critic 输入；C−B 比较增加任务 covariance 与 FiLM 的效果。同一 seed 共享世界模型、actor 共同部分和 B/C critic 初值，各组独立采集与更新。

三张卡分别运行独立进程，不使用 DDP。`CUDA_VISIBLE_DEVICES=1` 后，进程内仍传 `--device cuda:0`。使用实际分配的 GPU；用 `nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,driver_version --format=csv` 检查设备。服务器保留相邻的 `mjlab_for_supermimic` 仓库，按 [Setup](../README.md#setup) 使用锁定依赖重新建立环境。

## 默认配置与预算

| 参数 | 默认值 |
|---|---:|
| num_envs / buffer_size | 1024 / 1000 |
| replay_ratio | 16 |
| mean batch / window frames / updates | 2048 / 8 / 10 |
| covariance batch / validation draws | 8192 / 8192 |
| covariance training draws | 自动：143,360 |
| policy batch / horizon / samples | 4096 / 32 / 1 |
| actor updates / microbatch | 1 / 16 |
| critic epochs / minibatches | 5 / 4 |
| critic microbatch | 默认关闭，独立于 actor |
| log_interval / save_interval | 10 / 1000 个物理采集步 |
| max_iterations | 1,000,000 个累计物理采集步 |
| sampling_mode | return，可选 uniform |

每个物理采集步包含 num_envs 个真实帧。更新组频率为：

`num_envs × replay_ratio / (mean_updates × mean_batch_size × mean_window_frames)`。

默认频率为 0.1，即稳态每 10 个物理步一组。首次触发时若缺完整 mean 窗口、验证转移或可用 posterior，就丢弃这次机会；不补欠账。小数余额、零 replay ratio（只采集）、负 replay ratio（每步一组）和单步多组更新均沿用原版 Ratio。恢复保留调度余额，但新 replay 仍需重新 warmup。

每组依次执行 **10 mean → 18 covariance → 1 actor → 20 critic** 更新。mean 的名义窗口帧预算为 163,840，实际预测损失覆盖 143,360 个转移。covariance 预算按 `U × B × (H−1)` 推导，均匀、有放回抽取；默认为 17 个 8192 整批和一个 4096 尾批。阶段 NLL 按样本数加权。训练与验证抽样使用独立、可恢复的 RNG；只计算当前抽中转移的 detached 残差，不缓存全量残差。`--cov-pair-budget` 可覆盖训练预算，`--cov-validation-pairs` 控制验证抽样次数。

buffer、policy horizon、samples 均不改变默认 covariance 预算。增大 buffer 会增加 replay/索引/一致性检查开销；增大 samples 主要增加想象轨迹数。每次 actor 更新默认想象 4096×1×32＝131,072 个转移。actor 完整逻辑 batch 累积后做 norm 1.0、逐元素 0.1 clipping，再更新一次。actor Adam betas 为 (0.9, 0.95)，critic 为 (0.9, 0.999)，二者学习率 2e-4，并复用公共 Muon 权重分组。

critic 用原始 value 计算 GAE，只裁剪最终 target；MSE 不按累计 survival 加权。每次 actor 后用一次固定 permutation 训练 5×4 次，跨 epoch 复用。样本先按时间、轨迹顺序汇总，actor microbatch 不改变分组；不足四条有效样本时减少非空 minibatch 数。想象训练固定展开 H 步。参考到达片尾后，目标窗口和逐步奖励继续使用完整末帧，包括其速度。critic 输入移除剩余时长，actor 在第 H 步按存活权重 bootstrap。真实采集和整段评估仍在片尾结束，ReturnSampler 沿用片尾均值 bootstrap。旧版 v5 快照可用于 actor 推理和整段评估，v6 训练使用新的共享初始化和输出目录。

每个新训练帧、新保存 posterior 只更新一次对应归一化，包含 warmup；同一步多组优化不重复计数。验证 episode 固定为 `episode_id % validation_stride == 0`，不进入训练或归一化。

## 使用 TOML 分配显卡和 seed

服务器配置在仓库根目录的 [train_eit_config.toml](../train_eit_config.toml)。在该目录执行：

```bash
python3 src/supermimic/scripts/launch_eit.py --config train_eit_config.toml --dry-run
python3 src/supermimic/scripts/launch_eit.py --config train_eit_config.toml
```

`[paths]` 指定 CPFS 的三处根目录。启动器使用 `environment_root/.venv/bin/python`，存在 `environment_root/env.sh` 时先加载它。动作路径相对 `dataset_root`，默认使用 `motions/walk1_subject1.npz`，也可指定 `motions/*.npz`。相对根目录按 TOML 所在目录解析。

`[training]` 的键对应单次训练参数，例如 `mean_lr` 对应 `--mean-lr`。三个方法共用这些超参数。每个方法的 `gpus` 填物理卡号，`seeds` 填训练种子。例如 `gpus = [1, 6]`、`seeds = [11, 12, 13]` 会在 GPU 1 上依次运行 11、13，在 GPU 6 上运行 12。多个方法共用同一张卡时也依次执行；不同 GPU 并行。只保留这次要训练的方法表即可。

每个不同的 seed 用 B 组生成一份公共初始化，保存在 `checkpoint_root/experiment/init/`。各组的权重、`training.pt` 和训练日志分别写入 `checkpoint_root/experiment/seed<seed>/<A|B|C>/`。开启新一批实验时修改 `experiment`；接续已有 checkpoint 使用下文的单次训练 `--resume` 命令。

## 单次训练与三卡命令示例

未指定 `--initialization` 时，初始化文件默认位于输出目录的父目录，命名为 `eit_initialization_v4_eit_rpv_belief_critic_v3_noise<fingerprint>_seed<seed>.pt`。`fingerprint` 是实际观测噪声配置 SHA-256 的前 12 位，避免不同噪声配置复用旧初始化文件。显式指定的初始化路径仍按原路径加载并校验。

EIT CLI 不读取根目录 `train_config.toml`。`--save-interval`、`--log-interval`、`--run-name`、`--wandb-project` 写入公共 runner 配置；运行目录仍由 `--output` 指定。

```bash
export OMP_NUM_THREADS=1 PYTORCH_JIT=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
export WANDB_MODE=offline MPLCONFIGDIR=/tmp/supermimic-mpl
EIT_ROOT="$PWD/artifacts/eit_pro6000_v6"
EIT_MOTION="$PWD/data/motions/walk1_subject1.npz"
EIT_PYTHON="$PWD/.venv/bin/python"
mkdir -p "$EIT_ROOT/init"
EIT_ARGS=(--motion-file "$EIT_MOTION" --device cuda:0
  --num-envs 1024 --buffer-size 1000 --replay-ratio 16
  --mean-updates 10 --mean-batch-size 2048 --mean-window-frames 8
  --cov-batch-size 8192 --cov-validation-pairs 8192
  --policy-updates 1 --policy-batch-size 4096 --samples 1
  --policy-horizon 32 --policy-microbatch-size 16
  --critic-epochs 5 --critic-minibatches 4
  --log-interval 10 --save-interval 1000
  --posterior-save-stride 1 --jacobian-batch-chunk-size 32)

for seed in 2027 2028 2029 2030 2031; do
  CUDA_VISIBLE_DEVICES=0 "$EIT_PYTHON" -m supermimic.scripts.train_eit \
    "${EIT_ARGS[@]}" --seed "$seed" --group B --prepare-initialization \
    --initialization "$EIT_ROOT/init/seed$seed.pt"
done

# 每张卡串行运行五个 seed，三张卡之间并行。
for group_gpu in A:0 B:1 C:2; do
  (
    set -euo pipefail
    group=${group_gpu%:*}; gpu=${group_gpu#*:}
    for seed in 2027 2028 2029 2030 2031; do
      CUDA_VISIBLE_DEVICES="$gpu" "$EIT_PYTHON" -m supermimic.scripts.train_eit \
        "${EIT_ARGS[@]}" --seed "$seed" --group "$group" \
        --initialization "$EIT_ROOT/init/seed$seed.pt" \
        --output "$EIT_ROOT/seed$seed/$group" \
        --run-name "eit-$group-seed$seed" --max-iterations 1000000
    done
  ) &
done
wait
```

`jacobian-batch-chunk-size=32` 是上述命令显式设置的测量起点，CLI 默认不分块。显存不足时减小 actor microbatch 或 Jacobian chunk；critic microbatch 通过独立的 `--critic-microbatch-size` 设置，逻辑更新次数保持不变。

后验缓存每条约 10,252 字节；1024 环境、1000 容量、stride=1 时约 9.777 GiB，buffer=2000 时约 19.554 GiB。该内存按容量分配；基础 replay、模型、优化器和反传图另计。buffer 在同一进程采集 1000 步后填满；重启清空 replay。性能测量应覆盖完整训练组和稳定容量下的索引开销。

## 保存、续训与离线记录

`runner.cfg.save_interval` 是唯一周期保存配置，单位为每环境物理步。周期条件为累计零基 `step % save_interval == 0`。例如 interval=1000 时，周期保存的已采步数为 1、1001、2001……；另保存初始和结束状态，同一状态不重复发布。日志同样按累计零基物理 step 触发，结束时刷出剩余记录；终端 iteration 与 W&B 横轴为本次 run 从 0 开始的日志序号，JSONL step 和 Counters 仍保留累计物理进度。

每个保存点写 `world_model.pt`、`actor.pt`、带输入契约的 `critic.pt` 和原子替换的 `training.pt`，共同记录到一个 model artifact。覆盖文件前等待上一份后台归档完成。无 `snapshots/` 目录，也没有按 round 发布历史文件的流程。

完整 checkpoint 包含模型、actor、critic、优化器、归一化、RNG、Ratio 余额、累计计数、ReturnSampler 统计和模型 revision。恢复建立新的物理轨迹、replay、在线 belief 和 ReturnSampler 轨迹窗口；后验保存序列从新 replay 的 0 开始，日志及调度使用累计步。v3/v4 checkpoint 不兼容当前训练与评估。

```bash
# 首段到累计 5000 步，同时生成固定输入的下一次更新参考。
"$EIT_PYTHON" -m supermimic.scripts.train_eit "${EIT_ARGS[@]}" \
  --seed 2026 --group B --output "$EIT_ROOT/pilot/B" --max-iterations 5000 \
  --resume-reference "$EIT_ROOT/pilot/B-reference.pt"
"$EIT_PYTHON" -m supermimic.scripts.check_eit_resume \
  --checkpoint "$EIT_ROOT/pilot/B/training.pt" --motion-file "$EIT_MOTION" \
  --device cuda:0 --reference "$EIT_ROOT/pilot/B-reference.pt" \
  --output "$EIT_ROOT/pilot/resume.json"
# 第二段到累计 10000 步，不是再新增 10000 步。
"$EIT_PYTHON" -m supermimic.scripts.train_eit "${EIT_ARGS[@]}" \
  --output "$EIT_ROOT/pilot/B" --resume "$EIT_ROOT/pilot/B/training.pt" \
  --max-iterations 10000
```

每次启动或恢复都新建离线 W&B run，config 的 `continuation` 记录来源、起始累计步/帧和 parent run。`WANDB_MODE=disabled` 关闭 W&B，但仍保存四个本地文件。没有固定 ID、离线拼接、`--wandb-mode` 或 `--wandb-log-models`。离线模式不查询/清理在线 registry。项目与名称取公共 runner 配置。[离线记录说明](eit_wandb.md)

本地仅保留最新四文件；离线 artifact staging 会保留版本内容，占用随保存次数增长。完整训练包含约 1000 个周期保存点，不能用“只保留一个 training.pt”估算磁盘。用实际 pilot 的 `du -sh "$EIT_ROOT"` 和 `profiling.jsonl` 测量归档量、各阶段时间与 CUDA peak，再安排多 seed 预算。

## 评估与验收

```bash
"$EIT_PYTHON" -m supermimic.scripts.eval_eit \
  --checkpoint "$EIT_ROOT/pilot/B/training.pt" --motion-file "$EIT_MOTION" \
  --device cuda:0 --num-envs 8 --output "$EIT_ROOT/pilot/evaluation.json"
WANDB_MODE=disabled "$EIT_PYTHON" -m supermimic.scripts.check_eit_training \
  --device cpu --groups B --max-iterations 12
```

检查 `Counters/collected_steps`、`Counters/collected_frames`、各优化器累计计数和 `Budget/*` 的实际抽样/更新次数。`Perf/*` 与 `profiling.jsonl` 记录阶段时间和峰值显存；缓存读取时间是归一化时间的子项，不应重复求和。固定输入恢复不等于物理轨迹续接。

批量评估默认读取当前 `training.pt`；历史候选必须在清单 `checkpoints` 中明确指定保留的 artifact 文件路径。训练不重新生成已覆盖的候选。[批量评估](eit_batch.md) · [当前 pilot](eit_pilot.md)
