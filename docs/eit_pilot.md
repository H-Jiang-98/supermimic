# EIT v5 pilot

当前 `run_eit_pilot` 使用 `eit_pilot_v4` 元数据与 `eit_training_v5` checkpoint。它按累计物理步运行 A/B/C，共享初始化；默认 1024 环境、1000 容量、replay ratio 16、mean 10×2048×8、policy 4096×1×32、critic 5×4。covariance 每组抽样 143,360 对，8192 batch，18 次更新；验证独立抽 8192 对。

pilot 的默认目标为累计 100 步，在 50 步保存现场下一次更新参考，独立验证优化器恢复后继续到 100。这个短流程不代表收敛实验，也不能测量满 replay 的稳态索引成本。warmup 及恢复后跳过的机会由实际计数报告，不按 round 推算优化次数。

```bash
WANDB_MODE=offline OMP_NUM_THREADS=1 .venv/bin/python -m supermimic.scripts.run_eit_pilot \
  --motion-file data/motions/walk1_subject1.npz --device cuda:0 \
  --online-device cpu --output artifacts/eit-pilot-v5
.venv/bin/python -m supermimic.scripts.report_eit_pilot \
  --pilot artifacts/eit-pilot-v5 --output artifacts/eit-pilot-v5/report
```

每次训练启动/恢复均创建新离线 run；可用 `WANDB_MODE=disabled` 做无 SDK 的本地训练检查，报告中应保留模式。`--phase train` 只训练和验证恢复，`--phase evaluate` 只评估；已存在的协议必须完全匹配，不迁移旧 round 协议。

保存复用公共 `runner.cfg.save_interval`，初始、周期和结束保存最新四文件和同一 model artifact。每组第二段启动前，pilot 在 `resume/<组别>-midpoint.pt` 保留中点 checkpoint，并用对应现场参考验证恢复。第二段即使在采到新的一步之前中断，重启也可复用中点和验证回执。最终候选为各组 `training.pt`，验证采用独立 seed，保留测试集供后续使用。

独立诊断保留最终策略梯度、A 的 truth-input critic、共同策略记录和 reference-controller 记录。共同记录仅用于 process/posterior 比较，不计为独立闭环表现。记录、评分及 benchmark 各自运行在子进程；失败时保留命令日志和源码摘要。

v4 pilot 报告从 v5 checkpoint 核对累计物理预算、实际训练组及每组更新数，汇总 `Logger` 曲线、阶段时间、CUDA peak 和本地离线存储字节数。报告校验各组的中点参考和回执摘要，并按冻结清单从原始 episode 重算验证分数及入选 checkpoint。可在报告的 `pending` 和 `failures` 中查看缺失的证据和身份不匹配的文件。

[PRO 6000 配置与完整长跑](train_eit_pro6000.md) · [批量评估](eit_batch.md) · [离线 W&B](eit_wandb.md)
