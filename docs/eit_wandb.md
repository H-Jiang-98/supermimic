# EIT 训练的离线 W&B 记录

`supermimic.scripts.train_eit` 默认使用 W&B 离线模式，训练时无需 API key，不登录或上传。每轮 checkpoint、`metrics.jsonl`、`profiling.jsonl` 写入后，将对应指标写到本地 `.wandb` 文件。训练配置也会保存到 W&B config。三卡启动脚本见 [PRO 6000 说明](train_eit_pro6000.md)。

## 启动和配置

在原训练命令后追加这些参数即可：

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m supermimic.scripts.train_eit \
  --motion-file data/motions/walk1_subject1.npz --device cuda:0 \
  --group B --seed 2027 --output logs/eit-demo/seed2027/B \
  --rounds 10 --snapshot-interval 10 \
  --wandb-mode offline --wandb-project supermimic --wandb-group eit-demo
```

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--wandb-mode` | 环境变量 `WANDB_MODE`；未设置则 `offline` | `offline` 记录到本地，`disabled` 关闭；显式参数优先，不支持 `online` |
| `--wandb-project` | 新 run 使用 runner 的 `supermimic`；续训读取本地元数据 | 后续同步使用的项目 |
| `--wandb-entity` | 新 run 使用 `WANDB_ENTITY`，可留空 | 后续同步的个人账户或团队，留空时可在同步命令中指定 |
| `--wandb-name` | `eit-<组别>-seed<seed>` | 展示名，例如 `eit-B-seed2027` |
| `--wandb-group` | `eit` | 实验分组；同一次实验的 A/B/C 和所有 seed 应相同 |
| `--wandb-log-models` | 不启用 | 除指标外，将已发布历史快照暂存为模型 artifacts，增加磁盘占用 |

以上是记录设置，不影响学习率、训练预算或 checkpoint 恢复配置。已有 run 的项目、账户、名称和实验分组存于 `wandb_run.json`；续训省略时沿用，显式填写时须一致。模型附件开关按本次进程生效，续训需要附件时也要传 `--wandb-log-models`。

若 shell 中沿用旧说明的 `WANDB_MODE=disabled`，必须改为 `offline`，或显式传 `--wandb-mode offline`。`--prepare-initialization`、`check_eit_resume` 不创建 W&B run；pilot 单独关闭 W&B，以保持性能验收口径。

## 记录的内容

| W&B 指标前缀 | 内容 |
| --- | --- |
| `Counters/*` | 累计训练轮数、实采帧数、均值和 covariance 更新次数 |
| `Train/*` | 世界模型训练损失、过程 NLL、窗口与转移样本量 |
| `ModelValidation/*` | 世界模型留出数据上的损失和样本量 |
| `Policy/*`、`Grad/*` | 想象回报、critic loss、explained variance、动作饱和、FiLM 与梯度范数等 |
| `Budget/*` | 当轮采集、训练更新、想象步数、终止频率及有效样本统计 |
| `Perf/*` | 采集、后验重建、训练、保存等阶段耗时 |
| `Runtime/*`、`RuntimeAfterSave/*` | 线程数和可用时的 CUDA 显存统计 |

默认横轴为累计实采帧数 `Counters/collected_frames`，也可在 W&B 页面选择累计轮数 `Counters/round`。`None` 和非有限数值不绘制。config 保存组别、seed、motion 摘要、共享初始化摘要、环境数、runner/loop/policy 参数和快照间隔。

`ModelValidation/*` 不代表独立整段闭环测试；`Policy/return` 是训练想象回报。现有独立验证/测试结果、视频和旧 JSONL 历史不会自动导入。每轮耗时不包含 W&B 回调及退出刷盘时间。

## 文件和续训

```text
<output>/
├── training.pt
├── metrics.jsonl
├── profiling.jsonl
├── snapshots/
├── wandb_run.json                  # 固定 run ID、分组与训练身份
├── wandb/
│   ├── offline-run-<时间>-<ID>/
│   │   ├── run-<ID>.wandb          # SDK 原生离线记录
│   │   ├── eit_session.json       # 起始/最后记录轮数、是否完成刷盘
│   │   ├── files/
│   │   └── logs/
│   └── offline-run-<续训时间>-<ID>/
├── wandb-data/artifacts/staging/   # 可选模型附件副本
└── wandb-cache/                    # SDK 按需创建的缓存
```

在同一个 `--output` 中恢复其 `training.pt`，会读取原 run ID，写一个新的离线片段。比如首次训练 50 轮记录 step 1–50，续训 50 轮记录 51–100。不同组别/seed 的输出目录各有独立 ID；将 checkpoint 复制到新训练目录会创建新 run。不要删除或手工修改 `wandb_run.json`。

W&B SDK 不支持直接续写原离线文件，所以这里使用“固定 ID＋独立片段＋累计 step”。代码不传离线模式下会被忽略的 `resume` 参数；之后使用 CLI 的 `--append` 按训练顺序合并。依据：[W&B 0.24.2 初始化实现](https://github.com/wandb/wandb/blob/v0.24.2/wandb/sdk/wandb_init.py)、[同步命令实现](https://github.com/wandb/wandb/blob/v0.24.2/wandb/cli/cli.py)。

默认只记录指标和配置。启用模型附件时，复用 [已有 artifact helper](../src/supermimic/utils/wandb_artifacts.py)，只暂存第 0 轮及按 `--snapshot-interval` 发布的历史快照，不额外复制每轮覆盖的 `training.pt`。artifact 名为 `model-<run ID>`，包含完整 EIT 快照，附带轮数和帧数元数据；`latest` 表示最后一次暂存的历史快照。它与旧训练入口拆分的 actor/world-model 文件格式不同，EIT 播放仍使用 `--eit-checkpoint`。

默认将附件及缓存放在训练目录下；显式设置 `WANDB_DATA_DIR`、`WANDB_CACHE_DIR` 时尊重环境变量。后续同步模型需要保留 staging 文件。搬到另一台机器同步时也要复制这些文件，并使用已有 `supermimic.scripts.relocate_wandb` 校验/重映射路径（参数见 `--help`）。

W&B 初始化、记录或关闭失败会打印警告，checkpoint 和 JSONL 仍照常保存。检查 `eit_session.json` 中的 `finished`、`logging_ok`；初始化失败可能没有该文件。突然被终止的进程也可能留下未完成片段。续训只记录新轮次，不自动补齐崩溃前缺失的 W&B 点；本地 JSONL 是完整训练记录的依据。

## 训练结束后手动同步

以下命令会上传到 W&B，只有需要在线可视化时才执行。先确认片段已完成刷盘，再登录并按训练顺序逐个同步到同一 entity/project。第一次和后续片段均可使用 `--append`；不要并行同步同一 run 的片段，也不要重复同步已上传的片段。

```bash
WANDB_MODE=online .venv/bin/wandb login

# 将下面两个路径分别替换为同一训练线首次训练、续训的实际片段目录。
WANDB_MODE=online .venv/bin/wandb sync --append \
  --entity YOUR_ENTITY --project supermimic \
  logs/eit-demo/seed2027/B/wandb/offline-run-FIRST-ID
WANDB_MODE=online .venv/bin/wandb sync --append \
  --entity YOUR_ENTITY --project supermimic \
  logs/eit-demo/seed2027/B/wandb/offline-run-SECOND-ID
```

片段顺序以 `eit_session.json` 的 `start_round` 为准。不要使用普通 `wandb sync --sync-all` 来合并续训片段。15 条训练线分别同步后，在 `eit-demo`（或设定的 `--wandb-group`）分组下比较 A/B/C 曲线。
