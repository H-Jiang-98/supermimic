# PRO 6000 上的 EIT 训练、预算与续训

当前联合训练协议为 `eit_training_v5`。EIT 保留在线 belief、后验 replay、随机世界模型与可微 imagined belief rollout；采集调用公共 `runner.step()`，更新由公共 `Ratio` 调度，优化器、critic fitting、权重文件和 W&B 记录复用 SuperMimic。采用 float32、eager 执行。

本文配置尚无 PRO 6000 完整更新的耗时、峰值显存或收敛实测。历史 Step 6 小批量结果不能作为当前配置的性能证据。先测完整更新、保存及恢复，再确定多 seed 的正式运行预算。

2026-11-23 已在本地 RTX 5090（32 GiB）完成当前协议验证：[实测报告](validation/eit_v5_rtx5090.json)。C 组使用下表默认预算及完整网络宽度，Jacobian batch chunk=32、`PYTORCH_JIT=0`；采集 11 步后完成一组 10/18/1/20 更新。世界模型阶段 0.555 s，策略阶段 691.753 s（其中 critic 0.074 s）；PyTorch 峰值 allocated/reserved 为 15.20/17.76 GiB。新进程恢复至累计 14 步，优化器和调度状态一致，两次启动生成五个四文件 artifact，离线记录约 0.90 GiB。A/B/C 小批量 CUDA 下一次更新恢复逐位一致，完整回归 1095 项通过。

该实测只覆盖一个完整默认更新与恢复后的 warmup；1000 帧缓存已分配但尚未填满，未测稳定容量索引成本或收敛，也不能推算为 PRO 6000 性能。阶段计时存在包含关系，不能把父阶段与子阶段相加。

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

buffer、policy horizon、samples 均不改变默认 covariance 预算。增大 buffer 会增加 replay/索引/一致性检查开销；增大 samples 主要增加想象轨迹数。默认每组最多想象 4096×1×32＝131,072 个转移。actor 完整逻辑 batch 累积后做 norm 1.0、逐元素 0.1 clipping，再更新一次。actor Adam betas 为 (0.9, 0.95)，critic 为 (0.9, 0.999)，二者学习率 2e-4，并复用公共 Muon 权重分组。

critic 用原始 value 计算 GAE，只裁剪最终 target；MSE 不按累计 survival 加权。每次 actor 后用一次固定 permutation 训练 5×4 次，跨 epoch 复用。样本先按时间、轨迹顺序汇总，actor microbatch 不改变分组；不足四条有效样本时减少非空 minibatch 数。critic 保留剩余 clip 时间和片尾零 bootstrap；ReturnSampler 独立保留片尾均值 bootstrap。

每个新训练帧、新保存 posterior 只更新一次对应归一化，包含 warmup；同一步多组优化不重复计数。验证 episode 固定为 `episode_id % validation_stride == 0`，不进入训练或归一化。

## 启动与三卡运行

EIT CLI 不读取根目录 `train_config.toml`。`--save-interval`、`--log-interval`、`--run-name`、`--wandb-project` 写入公共 runner 配置；运行目录仍由 `--output` 指定。

```bash
export OMP_NUM_THREADS=1 PYTORCH_JIT=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
export WANDB_MODE=offline MPLCONFIGDIR=/tmp/supermimic-mpl
EIT_ROOT="$PWD/artifacts/eit_pro6000_v5"
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

`runner.cfg.save_interval` 是唯一周期保存配置，单位为每环境物理步。周期条件为累计零基 `step % save_interval == 0`。例如 interval=1000 时，周期保存的已采步数为 1、1001、2001……；另保存初始和结束状态，同一状态不重复发布。日志同样按累计零基 step 触发，结束时刷出剩余记录。

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
