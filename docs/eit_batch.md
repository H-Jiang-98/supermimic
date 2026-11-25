# EIT 候选快照的批量评估

在仓库根目录保存 `study.json`，路径以该文件所在目录为基准。`frames` 为累计实采帧数（物理步×环境数）。默认候选文件是运行目录中的 `training.pt`，其帧数必须与清单一致。

当前使用 v6 策略 checkpoint；推理兼容的 v5 快照也可准入，但需通过初始化、critic 契约和训练数据校验。训练不再维护 `snapshots/index.json`；历史候选须显式指定，例如在某个 run 中加入 `"checkpoints": {"40": "artifacts/retained/training.pt"}`。映射值相对于该 run 的 `directory`，也可使用绝对路径。文件必须已保留或从 model artifact 取得，不能从最新 checkpoint 补造。

以下清单假设候选使用 5 环境、累计 8 物理步（40 帧）、seed 364；请按实际训练记录修改路径、seed 和帧数。后验保存步距仅影响训练 replay，评估每步均运行滤波。

```json
{
  "design": "seen",
  "device": "cpu",
  "training": [
    {"path": "data/motions/walk1_subject1.npz", "subject": "subject1"}
  ],
  "validation": [
    {"path": "data/motions/walk1_subject1.npz", "subject": "subject1", "seed": 365, "num_envs": 2}
  ],
  "test": [
    {"path": "data/motions/walk1_subject1.npz", "subject": "subject1", "seed": 366, "num_envs": 2}
  ],
  "runs": [
    {"directory": "logs/eit-v6/B", "group": "B", "training_seed": 364, "frames": [40]}
  ]
}
```

`seen` 使用训练中见过的 motion，训练、验证、测试各自使用独立 seed。`heldout_clip` 要求训练、验证和测试的片段互不重叠。`heldout_subject` 还要求三个集合的 subject 互不重叠。程序用加载后的参考张量摘要识别同一个 clip，训练文件列表按采集时实际展开的顺序填写。

冻结清单时会在指定设备上加载各文件，登记原始文件 SHA-256、参考张量摘要、完整评估环境配置（含 reset 扰动、域随机化、终止阈值、奖励、物理求解、动作和观察配置）、prior、测量噪声、控制周期及实际超时步数。显示设置不参与冻结，seed 和环境数量单独登记。每个场景从第 0 帧运行，每个环境贡献一个首 episode。使用自定义 prior 或测量噪声时，将存档中的配置填入顶层 `initial_prior` 和 `measurement_noise`。

```bash
.venv/bin/python -m supermimic.scripts.batch_eval_eit freeze --spec study.json --manifest logs/eit-v6/manifest.json
.venv/bin/python -m supermimic.scripts.batch_eval_eit run --manifest logs/eit-v6/manifest.json --output logs/eit-v6/eval --split validation
.venv/bin/python -m supermimic.scripts.batch_eval_eit run --manifest logs/eit-v6/manifest.json --output logs/eit-v6/eval --split test
```

`run` 为每项评估启动独立子进程，输出 `tasks/<任务摘要>.json`，保留场景、快照摘要及逐环境的原始 episode。用 `report` 替换 `run` 可重新汇总已有结果。任务缺失、失败或仍有截断时，命令返回非零退出码，报告列出受影响的组别、训练 seed、候选帧数和场景。再次 `run` 会补跑缺失、失败或缓存损坏的任务，并复用已完成的任务。

执行任务或复用缓存前都会核验当前环境配置与冻结清单一致；配置漂移会使任务失败，避免旧缓存和新配置结果混入同一排名。当前批量清单版本为 `eit_batch_full_clip_v3`。旧清单必须重新执行 `freeze` 写入新的清单路径；新任务摘要不会复用旧缓存。需要汇总旧实验时，应使用对应的原始配置与代码。

`validation.json` 将 actor 更新次数为零的候选列入 `baselines`，包括 0 帧初始策略和已采集帧的 warmup 策略。无法读取或未通过准入检查的候选列入 `unavailable`。同一训练 seed 和组别的验证任务全部完成后，仅更新过 actor 的候选参与选模，依次比较 clip 等权完成率、有效进度、MPKPE 和更早实采帧数。MPKPE 为空时排在有有效误差样本的同分候选之后。若全部候选仍在 warmup，验证可以完成，但没有入选策略，测试保持未完成。

`test.json` 在全部测试场景中使用验证集选出的同一快照。先对每个 episode 的有效帧求均值，再在每个 clip 内等权平均 episode，最后对 clip 等权平均。报告给出计划和已记录的 episode 数、各指标有效 episode/clip 数及跟踪误差有效帧数。存在截断或缺少任务的候选，整段完成率为 null。

聚合回归使用 [`tests/fixtures/eit_clip_aggregation.json`](../tests/fixtures/eit_clip_aggregation.json)，覆盖逐帧 MPKPE、首步失败和不同长度的 clip；运行 `python -m pytest -q tests/test_estimator_batch_eval.py` 可验证。
