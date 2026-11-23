# EIT 离线 W&B

`train_eit` 与 SuperMimic 共用 `BaseRunner._init_wandb`、`Logger.log`、`log_model_artifact` 和结束清理。未设置时默认 `WANDB_MODE=offline`；测试可设置 `WANDB_MODE=disabled`，当前 EIT 不开放在线模式。

```bash
WANDB_MODE=offline OMP_NUM_THREADS=1 .venv/bin/python -m supermimic.scripts.train_eit \
  --motion-file data/motions/walk1_subject1.npz --device cuda:0 \
  --group B --output logs/eit-v5/B --max-iterations 1000 \
  --wandb-project supermimic --run-name eit-B-seed2026
```

项目和名称写入公共 runner 配置。`--output` 为运行目录；`--log-interval` 默认 10，`--save-interval` 默认 1000，二者均以累计物理采集步为单位。旧的 `--wandb-mode`、`--wandb-log-models`、`--wandb-group` 和固定 run ID 管理已移除。

每次启动/恢复产生新 run。其 config 包含 `runner_cfg`、`eit_loop`、`eit_policy` 和 `continuation`：来源 checkpoint、起始累计步/帧、父 run 的 ID/名称/项目。横轴是累计零基物理 step；不需要离线片段拼接或 append。恢复后重新 warmup，但累计计数不归零。

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
