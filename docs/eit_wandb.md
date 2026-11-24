# EIT 离线 W&B

`train_eit` 与 SuperMimic 共用 `BaseRunner._init_wandb`、`Logger.log`、`log_model_artifact` 和结束清理。未设置时默认 `WANDB_MODE=offline`；测试可设置 `WANDB_MODE=disabled`，当前 EIT 不开放在线模式。

```bash
WANDB_MODE=offline OMP_NUM_THREADS=1 .venv/bin/python -m supermimic.scripts.train_eit \
  --motion-file data/motions/walk1_subject1.npz --device cuda:0 \
  --group B --output logs/eit-v5/B --max-iterations 1000 \
  --wandb-project supermimic --run-name eit-B-seed2026
```

项目和名称写入公共 runner 配置。`--output` 为运行目录；`--log-interval` 默认 10，`--save-interval` 默认 1000，二者均以累计物理采集步为单位。旧的 `--wandb-mode`、`--wandb-log-models`、`--wandb-group` 和固定 run ID 管理已移除。

每次启动/恢复产生新 run。其 config 包含 `runner_cfg`、`eit_loop`、`eit_policy` 和 `continuation`：来源 checkpoint、起始累计步/帧、父 run 的 ID/名称/项目。终端 Learning iteration 与 W&B step 使用本次 run 的日志序号，从 0 开始；续训的新 run 也从 0 开始。分母包含本次运行的周期日志和结束补记。JSONL 的 step 仍为累计零基物理步，用于恢复和报表。恢复后重新 warmup，但累计计数不归零。

记录包含：

- `Counters/*`：累计物理步、帧、训练组、跳过机会及各优化器逻辑更新次数。
- `Budget/*`：日志区间实际 mean 窗口、covariance 抽样/更新、policy 起点及想象转移数；可用转移与缓存大小为最近一次测量。
- `Model/*`、`ModelValidation/*`、`Policy/*`、`Grad/*`：训练、留出与策略指标。
- `Perf/*`：各阶段耗时、公共 Logger 吞吐和 CUDA peak。子阶段时间不可重复相加。

保存入口在初始、周期和结束调用。周期为 `step % save_interval == 0`，例如 interval=1000 对应已采 1、1001、2001……步。每次保存将 `world_model.pt`、`actor.pt`、带契约的 `critic.pt` 和完整 `training.pt` 作为同一个 `model-<run ID>` artifact。写入新版本前等待上一份后台归档完成。四个文件先写入暂存目录，原子替换 `training.pt` 后发布其余权重。发布中断时，恢复流程在训练目录锁内完成已提交的文件替换。

每个有新采集步的保存点也保留当前日志区间的预算、耗时和累计计数。若进程在保存和写日志之间中断，恢复会补齐 `metrics.jsonl` 与 `profiling.jsonl`。即使日志周期和保存周期不同，恢复也能保留截至该 checkpoint 的最后一段数据。

离线模式只暂存 artifact，不查询或清理在线 registry。disabled 仍写本地四个文件，但不创建 artifact。没有 EIT 专用 W&B 管理器、`wandb_run.json` 或 `snapshots/index.json`。

W&B 的 `wandb/offline-run-*`、`wandb-data` 与 `wandb-cache` 会使用磁盘；后两者可由对应 `WANDB_DATA_DIR`、`WANDB_CACHE_DIR` 环境变量指定。离线版本内容不会因覆盖本地权重而消失。先用实际保存点测量容量，再估算长跑存储。日志不含独立评估结果；评估仍使用 `eval_eit`/`batch_eval_eit`。

需要上传时，由用户在有网络的环境使用已安装 W&B CLI 同步各个 `offline-run-*` 目录。每个启动段是独立 run，保留其父 run 信息，不使用 append 或固定 ID 合并。

## 核心日志对齐

终端默认只展开均值模型 Loss/state、状态分组 Loss/*、
Policy/reward、Policy/grad_norm 和 Policy/value_function。
公共 Logger 的 Mean reward、Mean episode length、环境 extras、步数和性能摘要仍显示。
EIT 专用验证指标、Counters、Budget 和细分诊断继续写入 W&B 和现有 JSONL，
原字段保留以兼容报表。未更新的指标沿用最近一次结果；首次更新前不填零。

| 核心字段 | 来源 |
| --- | --- |
| Loss/state 和状态分组 Loss/* | 相应 Model/Loss/* |
| Policy/reward | Policy/return / (policy_horizon × dt) |
| Policy/grad_norm | Grad/actor_norm |
| Policy/value_function | Policy/critic_loss |

Policy/reward 与普通模式使用同样的 return 归一化；训练目标和奖励函数不变。
Policy/return、Policy/raw_reward 等原始指标仍保留。

物理步触发条件仍是 step % log_interval == 0，结束补记独占一个新日志序号。
例如 interval=2、训练到累计 4 步，JSONL step 为 0、2、3，
终端和 W&B 日志序号为 0、1、2。同目录续训到 6 步时，
JSONL 追加 4、5，新 run 日志序号为 0、1。

性能计时范围保持原样：公共 FPS 以采集、模型和策略阶段计时之和计算，
不代表包含所有保存等开销的端到端吞吐。JSONL 仍不是完整 W&B 副本，
公共 Logger 添加的 episode 指标、Train/* 和 total_fps 不补入 metrics.jsonl。
历史日志与 checkpoint 格式不变；新展示规则在下次启动或续训时生效。
