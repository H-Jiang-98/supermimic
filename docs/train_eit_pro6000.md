# A B C 三组 baseline 在 PRO 6000 上的独立训练说明

本文使用估计器在环训练入口 `supermimic.scripts.train_eit`，将 A、B、C 分别放在 GPU 0、1、2 上训练。每张卡依次运行 seed 2027–2031，共 15 条训练线；同一 seed 的三组读取同一份初始化包，随后独立采集和更新。每条训练线运行 100 轮，在第 50 轮保存现场更新参考、验证恢复，再继续 50 轮。

默认参数与 [Step 6 锁定预算](validation/eit_step6_budget.json) 一致，任务为 `SuperMimic-GenCoord-G1`，motion 为 `walk1_subject1.npz`。这是进入正式多 seed 比较的起始预算，尚未验证训练收敛；原 pilot 的闭环完成率为 0。这里的验证和测试使用同一 motion、不同评估 seed，不代表未见动作或受试者的泛化测试。

存储规则在原预算上调整为：每轮覆盖最新 `training.pt`，历史快照只保留第 0 轮及每 10 轮的状态。间隔由 `--snapshot-interval` 配置，不改变实采帧数和学习更新预算。原预算 JSON 作为历史依据保留，新存储设置单独记录到实验目录的 `storage_policy.json`。

**本入口不读取根目录 `train_config.toml`，也不接受 `--config` 或 `--runner.*`。** 那些配置属于另一套 `train.py` 训练入口。下面通过一个 Bash 配置文件统一管理当前 EIT 命令行参数。

## 1 三组分别比较什么

R、p、v 分别表示基座朝向、位置、线速度。三组都有相同的本体历史、完整在线滤波器和世界模型训练流程。

| 组别 | GPU | 训练时 actor 跟踪目标使用的 R p v | actor 显式协方差输入 | 训练 critic |
| --- | --- | --- | --- | --- |
| A | 0 | 仿真或想象中的真值 | 无 | 共同特权输入，不含后验输入 |
| B | 1 | 在线后验均值 | 无 | 共同输入，加 12 维相对 R p v 均值和 45 维协方差编码 |
| C | 2 | 在线后验均值 | 450 维任务误差协方差，通过 FiLM 调制 | 与 B 相同的输入和结构 |

正式评估时，三组 actor 都用在线后验均值构造目标。A 也运行完整滤波器，不能将其解释为“完全不用估计器”的基线。B−A 同时包含训练目标来源和 critic 后验输入的变化；C−B 比较增加任务协方差及 FiLM 后的效果，其中也包含额外参数和参考信息路径。

同一 seed 共享世界模型和 actor 共同部分的初值，B/C 还共享 critic 初值。三组只共享初始权重，不共享训练后的参数、replay 或优化器状态。实现细节见 [总体设计](../plan_sketch.md) 和 [Step 4](../plan_step4.md)。

## 2 服务器环境与显卡检查

以下按三张完整 GPU 分配作业，不使用 DDP 或 `torchrun`。`CUDA_VISIBLE_DEVICES=1` 后，该进程内的这张卡编号为 `cuda:0`，因此三个训练命令都传 `--device cuda:0`。

如果服务器是 **NVIDIA RTX PRO 6000 Blackwell Server Edition**，官方规格为每卡 96 GB GDDR7；实际型号和可用容量以服务器查询结果为准，不将 RTX 6000 Ada 等其他型号视为同一配置。[NVIDIA 官方规格](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/)

将当前 SuperMimic 源码、`uv.lock`、motion 数据以及匹配版本的 `mjlab_for_supermimic` 一起放到服务器。目录关系必须保持为：

```text
supermimic_ws/
├── mjlab_for_supermimic/
└── supermimic/
    ├── pyproject.toml
    ├── uv.lock
    ├── src/
    └── data/motions/walk1_subject1.npz
```

在服务器重新创建 Python 环境，不直接搬运本机 `.venv`。当前仓库锁定 Python 3.13、PyTorch 2.14.0、CUDA 13.0 构建和 Warp 1.18.0；运行时驱动必须支持该构建。mjlab fork 需要 motion hooks 和 `forward_before_rl_signals` 支持，安装与旧版本补丁说明见 [README Setup](../README.md#setup)。

以下所有命令均从服务器的 `supermimic/` 仓库根目录执行：

```bash
uv sync --locked --python 3.13
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,driver_version --format=csv
df -h .

for gpu in 0 1 2; do
  CUDA_VISIBLE_DEVICES="$gpu" .venv/bin/python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA unavailable"
assert torch.cuda.device_count() == 1, "Expected one visible GPU"
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
x = torch.ones((32, 32), device="cuda:0")
print("CUDA matmul:", (x @ x)[0, 0].item())
PY
done

.venv/bin/python -m supermimic.scripts.train_eit --help
```

GPU 编号 0、1、2 是示例，可改为分配给你的卡或 GPU UUID。若通过作业调度器获得 GPU，沿用调度器给出的可见设备映射，不覆盖成未分配的物理卡。

默认间隔 10 时，每条训练线有 11 份历史快照和 1 份最新 checkpoint，15 条共 180 份。按 pilot 文件体积粗估约 30 GiB，加上共享初始化、恢复参考和日志，训练工件建议预留 **40–60 GiB**；诊断、备份及更大配置另计。原 260–300 GiB 估计对应逐轮保留。原 RTX 5090 的约 8.8 小时是 15 条训练线的纯训练累计估计，不能作为 PRO 6000 三卡并发的完工时间；在服务器记录实际轮次耗时后再估算。

## 3 创建统一超参数配置

先建立独立实验目录。以下 `eit_pro6000_v1` 用于本次冻结配置；需要更改训练协议时另建目录，并重新生成初始化包和评估清单。配置文件及启动脚本在启动前写好，运行过程中保持不变。

```bash
mkdir -p artifacts/eit_pro6000_v1/launch
cat > artifacts/eit_pro6000_v1/launch/config.sh <<'BASH'
# 必须从 supermimic 仓库根目录 source 或启动后续脚本。
EIT_REPO="$(pwd -P)"
EIT_ROOT="$EIT_REPO/artifacts/eit_pro6000_v1"
EIT_PYTHON="$EIT_REPO/.venv/bin/python"
EIT_MOTION="$EIT_REPO/data/motions/walk1_subject1.npz"
EIT_SEEDS=(2027 2028 2029 2030 2031)
EIT_FIRST_ROUNDS=50
EIT_SECOND_ROUNDS=50
EIT_SNAPSHOT_INTERVAL=10

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_JIT=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1

EIT_WANDB_ARGS=(
  --wandb-mode "$WANDB_MODE"
  --wandb-project supermimic
  --wandb-group eit_pro6000_v1
  # 如需暂存模型附件，在此增加 --wandb-log-models（额外占磁盘）。
)

EIT_ARGS=(
  "${EIT_WANDB_ARGS[@]}"
  --motion-file "$EIT_MOTION"
  --device cuda:0
  --num-envs 8
  --collection-steps 64
  --buffer-size 128
  --snapshot-interval "$EIT_SNAPSHOT_INTERVAL"
  --mean-lr 2e-4
  --mean-updates 10
  --mean-batch-size 2048
  --mean-window-frames 8
  --cov-lr 2e-4
  --cov-batch-size 256
  --cov-epochs 1
  --policy-updates 1
  --policy-batch-size 4
  --samples 2
  --policy-horizon 4
  --policy-microbatch-size 2
  --critic-epochs 2
  --exploration 0.1
  --hidden-dims 2048 2048 2048
  --actor-hidden-dims 2048 1024 512 256
  --critic-hidden-dims 256 128
)
BASH
```

新训练要调整命令行超参数，编辑 `EIT_ARGS` 对应行。`EIT_FIRST_ROUNDS + EIT_SECOND_ROUNDS` 是每条训练线的总轮数；本次正式实验保持 50＋50。修改 `EIT_SNAPSHOT_INTERVAL` 可调整历史快照频率，例如 5 表示保存第 0、5、10、15……轮，1 表示逐轮保留。actor 学习率、折扣因子等未开放的参数见第 6 节，不能凭空添加命令行选项。

线程数和两个数值环境变量沿用恢复复验设置，必须在 Python 导入 PyTorch 前生效。这里使用 `WANDB_MODE=offline`，无需登录即可记录 W&B 指标和配置，训练时不上传。`EIT_WANDB_ARGS` 同时用于首次训练和续训；将模式改为 `disabled` 可关闭。默认不暂存模型附件，上述 40–60 GiB 估计也不包含附件副本；启用 `--wandb-log-models` 后需要额外空间。详细参数和后续同步方法见 [EIT 离线 W&B 说明](eit_wandb.md)。

## 4 生成共享初始化并冻结评估清单

### 4.1 每个 seed 只生成一份初始化

在三组并发训练前，用 B 组生成每个 seed 的公共初始化包。每个包已经包含 A/B/C 所需的初始模型；生成时不进行训练。

```bash
(
  set -euo pipefail
  source artifacts/eit_pro6000_v1/launch/config.sh
  mkdir -p "$EIT_ROOT/init" "$EIT_ROOT/logs" "$EIT_ROOT/resume"
  cp docs/validation/eit_step6_budget.json "$EIT_ROOT/budget_source.json"
  git rev-parse HEAD > "$EIT_ROOT/source_commit.txt"
  git status --short > "$EIT_ROOT/source_status.txt"
  git diff --binary > "$EIT_ROOT/source_diff.patch"
  uv pip freeze --python "$EIT_PYTHON" > "$EIT_ROOT/packages.txt"
  nvidia-smi -q > "$EIT_ROOT/gpu_info.txt"
  # 包含未提交和未跟踪的 Python 源文件，保存本次实际运行源码。
  tar -czf "$EIT_ROOT/source.tar.gz" src pyproject.toml uv.lock
  sha256sum "$EIT_MOTION" "$EIT_ROOT/source.tar.gz" > "$EIT_ROOT/input_sha256.txt"

  for seed in "${EIT_SEEDS[@]}"; do
    CUDA_VISIBLE_DEVICES=0 "$EIT_PYTHON" -m supermimic.scripts.train_eit \
      "${EIT_ARGS[@]}" --group B --seed "$seed" \
      --prepare-initialization --initialization "$EIT_ROOT/init/seed${seed}.pt" \
      > "$EIT_ROOT/logs/init_seed${seed}.log" 2>&1
  done
)
```

不要用三个不同的初始化包分别启动 A/B/C。同一 seed、同一包才是配对比较。已有初始化包内容与所请求配置不符时，生成入口会拒绝复用。

### 4.2 在训练前冻结验证和测试场景

下面生成服务器本地路径的清单，保留锁定预算中的候选帧数：0、5120、10240、…、51200。第 0 轮只作为初始对照，不参与训练后快照的选模。验证 seed 为 17031，测试 seed 为 17032，每个候选各有 2 条首 episode；冻结时只读取场景和配置，不执行测试集评估。

```bash
(
  set -euo pipefail
  source artifacts/eit_pro6000_v1/launch/config.sh
  "$EIT_PYTHON" - "$EIT_ROOT" "$EIT_MOTION" "$EIT_SNAPSHOT_INTERVAL" "${EIT_SEEDS[@]}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
root, motion = map(Path, sys.argv[1:3])
interval = int(sys.argv[3])
seeds = list(map(int, sys.argv[4:]))
budget = json.loads((root / "budget_source.json").read_text())
assert seeds == budget["training_seeds"], "Changed seeds require a new protocol"
assert hashlib.sha256(motion.read_bytes()).hexdigest() == budget["scenes"]["training"][0]["sha256"], "Motion differs from locked budget"
frames_per_round = budget["collected_frames_per_run"] // budget["rounds_per_run"]
assert interval > 0
assert all(f % (frames_per_round * interval) == 0
           for f in budget["validation_candidate_frames"]), (
    "Snapshot interval must cover all frozen validation candidates; "
    "otherwise define a new candidate protocol before training"
)
spec = {
    "design": "seen", "device": "cuda:0",
    "runs": [
        {"group": group, "training_seed": seed,
         "directory": str(root / f"seed{seed}" / group),
         "frames": budget["validation_candidate_frames"]}
        for seed in seeds for group in ("A", "B", "C")
    ],
    "training": [{"path": str(motion), "subject": "subject1"}],
    "validation": [{"path": str(motion), "subject": "subject1",
                    "seed": 17031, "num_envs": 2}],
    "test": [{"path": str(motion), "subject": "subject1",
              "seed": 17032, "num_envs": 2}],
}
with (root / "evaluation_spec.json").open("x") as f:
    json.dump(spec, f, indent=2)
with (root / "storage_policy.json").open("x") as f:
    json.dump({"source_budget_digest": budget["budget_digest"],
               "latest_checkpoint_interval_rounds": 1,
               "snapshot_interval_rounds": interval,
               "snapshot_interval_frames": frames_per_round * interval,
               "retain_initial_snapshot": True}, f, indent=2)
PY
  CUDA_VISIBLE_DEVICES=0 "$EIT_PYTHON" -m supermimic.scripts.batch_eval_eit freeze \
    --spec "$EIT_ROOT/evaluation_spec.json" --manifest "$EIT_ROOT/manifest.json"
)
```

`budget_source.json` 和 `storage_policy.json` 用于保存预算及存储依据，`train_eit` 不会自动读取或强制执行它们。实际超参数由 `EIT_ARGS`、代码默认值和恢复 checkpoint 决定，必须保持一致。当前候选为每 10 轮，保存间隔 1、2、5、10 都能覆盖；若改成 20 等不能覆盖全部候选的间隔，上面的检查会拒绝继续，需要先制定新的候选清单。更改 motion、seed、轮数或候选间隔时，应生成新的预算与场景清单，不能只改启动命令后继续使用旧清单。

## 5 三张卡分别启动 A B C

创建工作脚本。它在一张 GPU 上顺序完成一个组的五个 seed；每个 seed 先训练 50 轮，再做独立恢复验收，验收成功后续训 50 轮。

```bash
cat > artifacts/eit_pro6000_v1/launch/run_group.sh <<'BASH'
#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/config.sh"
group="${1:?Usage: run_group.sh A|B|C GPU_ID}"
gpu="${2:?GPU_ID is required}"
case "$group" in A|B|C) ;; *) echo "Invalid group: $group" >&2; exit 2 ;; esac
test -f "$EIT_ROOT/manifest.json"
export CUDA_VISIBLE_DEVICES="$gpu"

# 首次启动专用：先检查所有目标，防止误把已有作业当成全新训练。
for seed in "${EIT_SEEDS[@]}"; do
  test -f "$EIT_ROOT/init/seed${seed}.pt"
  if [[ -e "$EIT_ROOT/seed${seed}/$group" ]]; then
    echo "Output exists; use the recovery instructions: seed${seed}/$group" >&2
    exit 1
  fi
done

for seed in "${EIT_SEEDS[@]}"; do
  run="$EIT_ROOT/seed${seed}/$group"
  reference="$EIT_ROOT/resume/seed${seed}/${group}-reference.pt"
  mkdir -p "$(dirname -- "$reference")" "$EIT_ROOT/logs"
  echo "START group=$group seed=$seed gpu=$gpu"

  "$EIT_PYTHON" -m supermimic.scripts.train_eit \
    "${EIT_ARGS[@]}" --group "$group" --seed "$seed" \
    --output "$run" --initialization "$EIT_ROOT/init/seed${seed}.pt" \
    --rounds "$EIT_FIRST_ROUNDS" --resume-reference "$reference" \
    > "$EIT_ROOT/logs/${group}_${seed}_first.log" 2>&1

  "$EIT_PYTHON" -m supermimic.scripts.check_eit_resume \
    --checkpoint "$run/training.pt" --reference "$reference" \
    --motion-file "$EIT_MOTION" --device cuda:0 \
    --output "$EIT_ROOT/resume/seed${seed}/${group}.json" \
    > "$EIT_ROOT/logs/${group}_${seed}_resume_check.log" 2>&1

  "$EIT_PYTHON" -m supermimic.scripts.train_eit \
    "${EIT_WANDB_ARGS[@]}" --motion-file "$EIT_MOTION" --device cuda:0 \
    --output "$run" --resume "$run/training.pt" --rounds "$EIT_SECOND_ROUNDS" \
    > "$EIT_ROOT/logs/${group}_${seed}_second.log" 2>&1
  echo "DONE group=$group seed=$seed"
done
echo "GROUP_DONE group=$group"
BASH

nohup bash artifacts/eit_pro6000_v1/launch/run_group.sh A 0 \
  > artifacts/eit_pro6000_v1/logs/worker_A.log 2>&1 &
nohup bash artifacts/eit_pro6000_v1/launch/run_group.sh B 1 \
  > artifacts/eit_pro6000_v1/logs/worker_B.log 2>&1 &
nohup bash artifacts/eit_pro6000_v1/launch/run_group.sh C 2 \
  > artifacts/eit_pro6000_v1/logs/worker_C.log 2>&1 &
```

`nohup` 适用于允许直接后台运行作业的服务器；有调度系统时，将同一个工作脚本放入各自的一卡作业。无需三个组逐 seed 同步等待，但三组应使用相同 GPU 型号、源码、配置和预算。

日志写在训练目录外，因为首次训练要求输出目录为空。脚本遇到任何训练或恢复验证错误立即停止该组，不会悄悄跳过失败 seed；其他 GPU 上的组可继续完成。

## 6 超参数含义与修改位置

### 6.1 物理采集与训练轮次

| 参数 | 本次值 | 含义与影响 |
| --- | --- | --- |
| `--group` | A / B / C | 选择上述消融组。恢复时从 checkpoint 读取，不能借此把 B 改成 C |
| `--seed` | 2027–2031 | 训练随机种子及初始化包身份。五组配对 seed 不包含 pilot 的 2026 |
| `--rounds` | 每段 50 | 本次调用新增的训练轮数。新训练 50，再恢复 50，累计才是 100 |
| `--snapshot-interval` | 10 | 历史快照保存间隔，单位为累计轮数；第 0 轮始终保留，最新 `training.pt` 每轮更新。必须为正整数 |
| `--num-envs` | 8 | 并行 MuJoCo 环境数。增大会改变每轮实采帧数、replay 和后验重建开销 |
| `--collection-steps` | 64 | 每轮每个环境采集的控制步数；每轮实采帧数为 `num_envs × collection_steps` |
| `--buffer-size` | 128 | 每个环境的 replay 容量，单位为帧；总基础容量为 `num_envs × buffer_size`，还要考虑后验前缀等存储 |
| `--exploration` | 0.1 | 真实采集时 actor 动作上的高斯探索噪声标准差，按策略动作单位计；不是传感器噪声或过程协方差 |
| `--motion-file` | walking NPZ | 参考动作文件；本次按单文件配置，数据须匹配 50 Hz 控制频率 |
| `--output` | 每组每 seed 独立 | 训练 checkpoint、快照和指标的输出目录 |

每轮依次执行物理采集、均值模型拟合、过程协方差拟合、当前模型下的后验重建、actor/critic 更新，然后保存。训练使用 uniform reset；每轮刷新物理采集环境，64 步对应最多 1.28 秒连续在线历史，失败会更早 reset。增大总轮数不会自动延长每段连续历史。

### 6.2 世界模型

| 参数 | 本次值 | 含义与影响 |
| --- | --- | --- |
| `--mean-lr` | `2e-4` | 均值模型 Muon/Adam 优化器学习率 |
| `--mean-updates` | 10 | 每轮均值模型的优化批次数 |
| `--mean-batch-size` | 2048 | 每次均值更新抽取的轨迹窗口数；允许重复抽样，不等于 2048 个独立 episode |
| `--mean-window-frames` | 8 | 一个均值训练窗口含 8 帧，其中首帧是真实初态，后面 7 帧滚动预测并计算加权 MSE；窗口不跨 reset |
| `--cov-lr` | `2e-4` | 过程噪声头 Adam 学习率；协方差阶段固定均值参数和归一化，以一步残差的高斯 NLL 拟合 |
| `--cov-batch-size` | 256 | 每次协方差更新最多使用的一步有效转移对数，最后不足一批也更新 |
| `--cov-epochs` | 1 | 每轮对全部保留的有效训练对完整遍历次数 |
| `--hidden-dims` | 2048 2048 2048 | 均值动力学网络三层隐藏宽度；噪声头另有默认宽度 128、64 |

协方差更新数为每轮 `cov_epochs × ceil(valid_training_pairs / cov_batch_size)`。三组失败时间和保留轨迹不同，有效对数可能不同，因此不要求实际 covariance 更新次数完全相等；要求 batch/epoch 规则相同并报告实际次数。

### 6.3 actor 和 critic

| 参数 | 本次值 | 含义与影响 |
| --- | --- | --- |
| `--policy-updates` | 1 | 每轮 actor 优化次数，每次同时产生供 critic 回归的想象数据 |
| `--policy-batch-size` | 4 | N，每次 actor 更新从 replay 选择的起点数 |
| `--samples` | 2 | K，每个起点的独立过程及测量噪声轨迹数；共有 N×K＝8 条想象轨迹 |
| `--policy-horizon` | 4 | H，每条想象轨迹最多前进 4 个控制步，即 0.08 秒；普通 horizon 结束使用价值 bootstrap，clip 结束按片尾规则截短 |
| `--policy-microbatch-size` | 2 | 对展开后的 N×K 条轨迹分块执行前向和反传，并累积梯度后更新；也用于 critic 数据分块。它不是起点总数 N |
| `--critic-epochs` | 2 | 每次 actor 更新后，critic 对该批想象目标完整回归两遍，每遍一次优化器更新 |
| `--actor-hidden-dims` | 2048 1024 512 256 | actor 隐藏层宽度，三组共同部分保持一致；C 的 FiLM 由组别启用 |
| `--critic-hidden-dims` | 256 128 | critic 隐藏层宽度，B/C 输入及结构相同 |

名义上每次 actor 更新最多计算 `N × K × H = 32` 个想象转移。片尾截短和 soft survival 会改变有效步数及权重，这些都不是物理采集帧数。增加 H 会增加通过动力学、滤波和协方差的反传深度；增加 N 或 K 则增加采样量，不能将三者看作可互换的 batch。

### 6.4 当前未开放命令行的参数

以下均为本次代码默认值。需要修改时，在新实验版本中修改对应 dataclass 默认值，或为 `train_eit.py` 增加明确的解析和构造参数后再运行。仅修改根目录 TOML 不会生效。

| 配置字段 | 默认值 | 含义与源码位置 |
| --- | --- | --- |
| `BeliefPolicyLearnerCfg.lr` | `2e-4` | actor Adam 学习率，见 [belief_policy_learner.py](../src/supermimic/learners/belief_policy_learner.py) |
| `BeliefPolicyLearnerCfg.critic_lr` | `2e-4` | critic Adam 学习率，同上 |
| `gamma` | 0.99 | 每个控制步的回报折扣因子，同上 |
| `lam` | 0.95 | critic 的 λ-return 混合系数，同上 |
| `max_grad_norm` | 1.0 | actor 和 critic 梯度的范数裁剪上限，不是逐元素裁剪值，同上 |
| `value_clipping` | `True` | 将使用的价值限制在 `max_step_reward / (1 - gamma)` 上界；不是 PPO 的策略比率裁剪，同上 |
| `BeliefTrainingCfg.validation_stride` | 5 | 按 episode ID 每 5 个留出一个用于模型验证，不是每 5 轮做闭环评估，见 [belief_training.py](../src/supermimic/runners/belief_training.py) |
| `validation_batch_size` | 4 | 内部世界模型验证的批大小，同上 |
| `EITTrainingCfg.normalization_batch_size` | 16 | 后验前缀准备和输入归一化的处理批大小，见 [eit_training.py](../src/supermimic/runners/eit_training.py) |
| `ProcessNoiseConfig.hidden_dims` | 128 64 | 过程噪声头隐藏层，见 [models.py](../src/supermimic/models.py) |

例如，要另开 actor 学习率为 `1e-4` 的实验，修改 `BeliefPolicyLearnerCfg` 中 `lr: float = 1e-4`，保留 `critic_lr` 或另行明确修改，然后保存新代码版本、重新生成初始化并新建输出目录。当前没有可直接使用的 `--policy-lr`、`--critic-lr`、`--gamma` 或 `--lam` 选项。

**恢复配置以 checkpoint 为准。** `--resume` 会读取保存的 runner、loop、policy、组别、训练 seed、环境数及优化器状态。恢复命令附加新的 `--mean-lr`、`--policy-horizon`、`--num-envs` 等参数不会覆盖已保存配置；恢复时不能同时传 `--initialization`。想修改训练方案，应建立新的实验协议和明确的迁移流程。

`--snapshot-interval` 属于存储设置，可以在恢复时显式覆盖；省略时继承 checkpoint 中的间隔，旧 checkpoint 没有此字段时继承 1。间隔按累计轮数计算，例如在第 23 轮恢复、间隔为 10，下一份历史快照在第 30 轮。已有快照不删除，也不补造已经错过的轮次。若结束在第 103 轮，最新 `training.pt` 为第 103 轮，最后一份按间隔保存的快照仍是第 100 轮。正式实验改变间隔时要同步记录 `storage_policy.json`，并确保冻结的评估候选仍全部可用。

### 6.5 显存相关参数

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--jacobian-batch-chunk-size` | 不传，即 `None` | 对滤波状态转移 Jacobian 的批维度进行 vmap 分块，减小瞬时显存占用 |
| `--jacobian-chunk-size` | 不传，即 `None` | 对 jacrev 的输出 Jacobian 行进行分块，减小瞬时显存占用 |

出现显存不足时，可以在独立排错配置里先减小 policy microbatch，或给上述两个分块参数设置较小正整数，再核对结果及完整轮次耗时。分块和梯度累积可能改变浮点运算次序；不要假定它们与原配置逐位一致。正式比较中三组统一设置，重新记录配置与恢复验收。

当前 EIT 使用 float32、eager 路径，均值训练关闭 CUDA graphs 和编译内核；CLI 没有 AMP/BF16、DDP 或启用编译的开关。大显存本身不会消除逐段后验重建和 Jacobian 的计算成本。

## 7 日志 预算核对与异常恢复

### 7.1 查看进度和预算

```bash
tail -f artifacts/eit_pro6000_v1/logs/worker_A.log
# 另开终端查看具体训练段输出，B/C 同理。
tail -f artifacts/eit_pro6000_v1/logs/A_2027_first.log
nvidia-smi
```

正常输出结构如下：

```text
artifacts/eit_pro6000_v1/
├── launch/                       # 固定配置和工作脚本
├── init/seed2027.pt               # 每个 seed 的公共初始化
├── manifest.json                 # 冻结的评估场景和候选清单
├── storage_policy.json           # 本次历史快照保存间隔
├── logs/                         # 工作脚本及各 Python 进程日志
├── resume/seed2027/               # A/B/C 第 50 轮现场参考和恢复验收 JSON
├── seed2027/
│   ├── A/
│   │   ├── training.pt           # 当前联合训练状态
│   │   ├── metrics.jsonl         # 每轮损失、更新计数、采样预算
│   │   ├── profiling.jsonl       # 保存等阶段及完整轮次计时
│   │   ├── wandb_run.json        # 此组别和 seed 固定的 W&B run ID
│   │   ├── wandb/                # 每次训练/续训各一个 offline-run-* 片段
│   │   └── snapshots/            # 第 0 轮及按间隔保存的不可变快照和索引
│   ├── B/
│   └── C/
└── seed2028/ ... seed2031/
```

每条训练线 100 轮的预期计数如下。它们按组、按 seed 分别计算，不能把三张卡的帧数相加后当成每组的预算。

| 项目 | 每轮 | 100 轮 |
| --- | --- | --- |
| 物理实采帧数 | 8×64＝512 | 51,200 |
| 均值更新 | 10 | 1,000 |
| 均值窗口抽样次数 | 10×2048＝20,480 | 2,048,000，含重复抽样 |
| actor 更新 | 1 | 100 |
| critic 更新 | 1×2＝2 | 200 |
| covariance 更新 | 按有效对数计算 | 汇总实际值，不预设三组相等 |

除了 loss，还应查看物理失败频率、有效窗口/转移对、实际想象步数、独立验证进度和完成率。`metrics.jsonl` 的训练想象回报与从片头开始的独立闭环回报具有不同起点分布，不宜直接相减解释模型误差。`profiling.jsonl` 用于检查后验重建、优化和保存的实际成本。

15 条训练线对应 15 个独立 W&B run，共用 `eit_pro6000_v1` 实验分组。第 50 轮续训沿用各自 run ID 和累计轮数；离线片段需按先后顺序同步才能在同一 run 中查看完整曲线。默认横轴为 `Counters/collected_frames`。`ModelValidation/*` 是训练期间的世界模型留出指标，独立闭环验证/测试结果仍由第 8 节的评估流程生成，当前不会自动导入 W&B。`profiling.jsonl` 和 W&B 的 `Perf/*` 不包含 W&B 记录与退出刷盘开销。

### 7.2 中断后恢复

上述工作脚本用于首次启动，**不是自动重试调度器**。中断后先确认原进程已经退出，再检查该组当前 checkpoint。不要重新运行整组脚本、覆盖日志或从较旧快照回退覆盖已有输出。

下面以 B、seed 2027 为例读取累计轮数；实际恢复仍绑定 B 的 GPU 1：

```bash
source artifacts/eit_pro6000_v1/launch/config.sh
EIT_RUN="$EIT_ROOT/seed2027/B"
"$EIT_PYTHON" - "$EIT_RUN/training.pt" <<'PY'
import sys
import torch
s = torch.load(sys.argv[1], map_location="cpu", weights_only=True, mmap=True)
print("group:", s["runner_config"]["estimator_inputs"]["group"])
print("seed:", s["training_seed"], "rounds:", s["rounds"])
print("frames:", s["collected_frames"])
PY
```

- 若完成 `r < 50` 轮，先恢复 `50-r` 轮，并传回原路径的 `--resume-reference`，保证第 50 轮现场参考被捕获。随后执行第 5 节的恢复验收，成功后再训练 50 轮。
- 若恰好完成 50 轮，先使用第 50 轮的 `training.pt` 和现场参考执行恢复验收，再继续。若现场参考缺失，不可将“重新加载两次相同 checkpoint”代替现场参照验收；应记录异常并复跑该训练线。
- 若完成 `50 < r < 100` 轮，确认第 50 轮验收 JSON 已通过，恢复 `100-r` 轮即可。
- 若已完成 100 轮，不再追加训练；转向该组尚未开始的后续 seed，并逐一按第 5 节三条 Python 命令执行。已有完整结果的 seed 不重跑。

例如 checkpoint 已完成 73 轮，则新增 27 轮：

```bash
CUDA_VISIBLE_DEVICES=1 "$EIT_PYTHON" -m supermimic.scripts.train_eit \
  "${EIT_WANDB_ARGS[@]}" --motion-file "$EIT_MOTION" --device cuda:0 \
  --output "$EIT_RUN" --resume "$EIT_RUN/training.pt" --rounds 27 \
  > "$EIT_ROOT/logs/B_2027_recovery_from73.log" 2>&1
```

恢复会清空 replay、重置真实环境并采集新轨迹。固定输入下的下一次模型和策略更新可以验收一致性，但真实物理轨迹不承诺接续未中断运行。除预定第 50 轮恢复外，任何意外恢复都应记录轮次和原因；它改变 replay 留存和后续数据，不能隐藏。若因此统一调整恢复日程，应重新锁定协议并重跑受影响的配对比较。

## 8 训练结束后的验证和测试

训练每轮保存最新 `training.pt`，默认每 10 轮发布历史快照，`--rounds` 本身不自动触发独立闭环评估。冻结清单将每 10 轮的快照作为验证候选，可以在全部训练完成后统一评估；不需要每 10 轮重启训练进程，这会额外清空 replay。

三组全部训练结束后，在空闲 GPU 0 上运行验证，命令会使用已冻结的每组每 seed 候选清单和整段评估协议：

```bash
(
  set -euo pipefail
  source artifacts/eit_pro6000_v1/launch/config.sh
  CUDA_VISIBLE_DEVICES=0 "$EIT_PYTHON" -m supermimic.scripts.batch_eval_eit run \
    --manifest "$EIT_ROOT/manifest.json" --output "$EIT_ROOT/evaluation" \
    --split validation
)
```

验证依次按完成率、归一化进度、MPKPE、较早快照选模。下面先检查验证是否完整，将 15 条训练线的入选快照封存，再使用同一输出目录执行测试。测试入口会从已有验证任务重新计算入选快照，因此封存后保持验证工件和候选清单不变：

```bash
(
  set -euo pipefail
  source artifacts/eit_pro6000_v1/launch/config.sh
  "$EIT_PYTHON" - "$EIT_ROOT" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
val = json.loads((root / "evaluation/validation.json").read_text())
assert val["complete"] and not val["pending_validation"]
assert len(val["selected"]) == 15, "All five A/B/C pairs must be selected"
seal = {"manifest_digest": val["manifest_digest"], "selected": val["selected"]}
path = root / "evaluation/selection_sealed.json"
if path.exists():
    assert json.loads(path.read_text()) == seal, "Selection changed after sealing"
else:
    with path.open("x") as f:
        json.dump(seal, f, indent=2)
PY
  CUDA_VISIBLE_DEVICES=0 "$EIT_PYTHON" -m supermimic.scripts.batch_eval_eit run \
    --manifest "$EIT_ROOT/manifest.json" --output "$EIT_ROOT/evaluation" \
    --split test
  "$EIT_PYTHON" - "$EIT_ROOT" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1]) / "evaluation"
seal = json.loads((root / "selection_sealed.json").read_text())
test = json.loads((root / "test.json").read_text())
assert test["complete"]
assert {k: test[k] for k in seal} == seal, "Test selection differs from sealed selection"
PY
)
```

正式评估不设置外部 `--steps` 截断，也不改成 10 秒 timeout。默认清单包含 165 个验证任务、330 条验证 episode，选模后有 15 个测试任务、30 条测试 episode；若策略能坚持更长时间，评估耗时会明显增加。最后按训练 seed 配对汇总 B−A、C−B，并保留逐 episode 原始记录，不以五个 seed 中最好的一个代替总体结果。批量评估并不自动完成 Step 7 的全部统计报告，交付要求见 [Step 7](../plan_step7.md)。

## 9 如何利用 PRO 6000 做后续调参

第一轮正式比较保持本说明的统一参数。需要扩展预算或优化吞吐时，在独立实验目录先用 pilot seed 2026 做排错与性能测量，再冻结新协议；不要根据正式测试结果回头调参。

| 目标 | 可考察的参数 | 需要同时核对的影响 |
| --- | --- | --- |
| 增加真实经验 | 总轮数、`num-envs`、`collection-steps` | 重新计算实采帧数和每帧更新量；改变采集长度也改变连续后验历史 |
| 增加想象起点覆盖 | `policy-batch-size`，如试 4→8 | 起点准备、后验重建、显存和整轮耗时；不能只计 actor 反传 |
| 增加同一起点的噪声样本 | `samples`，如试 2→4 | N×K 轨迹数增加，检查梯度方差和模型误差影响 |
| 增加想象时间范围 | `policy-horizon`，如试 4→8 | 反传深度、模型误差累积、真实控制改善及 bootstrap 梯度贡献 |
| 提高 GPU 并行度 | `policy-microbatch-size`，如试 2→4 | N/K/H 保持不变，测完整轮次及显存峰值，复验数值与恢复行为 |
| 改进均值或协方差拟合 | 对应 lr、batch、更新数或 epochs | 留出均值误差、一步 NLL、残差白化与后验覆盖率；不要只看训练 loss |

这些是单因素试验方向，不是已经验证适合 PRO 6000 的最佳参数。每次改动应在三组中保持统一规则，保留默认配置作对照。采集、replay 后验重建和完整滤波 Jacobian 可能限制吞吐，显存未用满并不直接说明需要放大所有 batch。

本说明对应的关键入口为 [train_eit](../src/supermimic/scripts/train_eit.py)、[世界模型参数](../src/supermimic/scripts/train_utils.py)、[策略参数](../src/supermimic/learners/belief_policy_learner.py)、[恢复验证](../src/supermimic/scripts/check_eit_resume.py) 和 [批量评估](../src/supermimic/scripts/batch_eval_eit.py)。实际启动前用服务器当前代码的 `--help` 核对接口，保留源码快照、初始化包、冻结清单和运行日志。
