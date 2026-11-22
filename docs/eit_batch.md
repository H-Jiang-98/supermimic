# EIT 候选快照的批量评估

在仓库根目录保存 `study.json`。清单中的路径以该 JSON 所在目录为基准。下面的样例对应一个 B 组训练 seed，每轮用 5 个环境采集 8 步。增加组别或 seed 时，在 `runs` 中逐项指定训练输出目录和预定的实采帧数。

本例的第 1 轮候选 `frames_40.pt` 需要训练时指定 `--snapshot-interval 1`，参见
[小网络验收命令](../README_eval.md#共同初始化快照与批量选模)。`train_eit` 新训练默认每 10 轮
发布历史快照；清单中的候选必须与保存间隔匹配。缺失的历史轮次不能从最新 `training.pt` 补造。

批量评估接受 v4 和 v3 策略快照。`frames` 按累计实采帧数填写；训练中的
`posterior_save_stride` 决定各环境后验的保存频率，并影响策略可选起点数。评估时每个控制步都运行滤波。

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
    {"directory": "logs/step5/B", "group": "B", "training_seed": 364, "frames": [0, 40]}
  ]
}
```

`seen` 使用训练中见过的 motion，训练、验证、测试各自使用独立 seed。`heldout_clip` 要求训练、验证和测试的片段互不重叠。`heldout_subject` 还要求三个集合的 subject 互不重叠。程序用加载后的参考张量摘要识别同一个 clip，训练文件列表按采集时实际展开的顺序填写。

冻结清单时会在指定设备上加载各文件，登记原始文件 SHA-256、参考张量摘要、完整评估环境配置（含 reset 扰动、域随机化、终止阈值、奖励、物理求解、动作和观察配置）、prior、测量噪声、控制周期及实际超时步数。显示设置不参与冻结，seed 和环境数量单独登记。每个场景从第 0 帧运行，每个环境贡献一个首 episode。使用自定义 prior 或测量噪声时，将存档中的配置填入顶层 `initial_prior` 和 `measurement_noise`。

```bash
.venv/bin/python -m supermimic.scripts.batch_eval_eit freeze --spec study.json --manifest logs/step5/manifest.json
.venv/bin/python -m supermimic.scripts.batch_eval_eit run --manifest logs/step5/manifest.json --output logs/step5/eval --split validation
.venv/bin/python -m supermimic.scripts.batch_eval_eit run --manifest logs/step5/manifest.json --output logs/step5/eval --split test
```

`run` 为每项评估启动独立子进程，输出 `tasks/<任务摘要>.json`，保留场景、快照摘要及逐环境的原始 episode。用 `report` 替换 `run` 可重新汇总已有结果。任务缺失、失败或仍有截断时，命令返回非零退出码，报告列出受影响的组别、训练 seed、候选帧数和场景。再次 `run` 会补跑缺失、失败或缓存损坏的任务，并复用已完成的任务。

执行任务或复用缓存前都会核验当前环境配置与冻结清单一致；配置漂移会使任务失败，避免旧缓存和新配置结果混入同一排名。当前批量清单版本为 `eit_batch_full_clip_v2`。旧 `v1` 清单缺少完整配置，必须重新执行 `freeze` 写入新的清单路径；新任务摘要不会复用旧缓存。需要汇总旧实验时，应使用对应的原始配置与代码。

`validation.json` 单列第 0 轮基线。训练 seed 和组别下的验证任务全部完成后，程序按 clip 等权完成率、有效进度、MPKPE 和更早实采帧数依次选模。MPKPE 为空时排在有有效误差样本的同分候选之后。`test.json` 在全部测试场景中使用验证集选出的同一快照。先对每个 episode 的有效帧求均值，再在每个 clip 内等权平均 episode，最后对 clip 等权平均。报告给出计划和已记录的 episode 数、各指标有效 episode/clip 数及跟踪误差有效帧数。存在截断或缺少任务的候选，整段完成率为 null。

手算样例包含逐帧 MPKPE、首步失败和不同长度的 clip。真实 CPU 样例保留 v1 协议下小型 B 策略在 walking motion 上的原始 episode 及汇总期望值，仅用于聚合回归，不作为 v2 可执行清单。通过 `python -m pytest -q tests/test_estimator_batch_eval.py` 可复算两份结果。[手算样例](validation/eit_clip_aggregation.json) · [真实评估样例](validation/eit_batch_walk.json)
