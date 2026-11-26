# EIT v7 局部 belief 控制

`dev_observable` 上的 v7 包含 mean-direct、mean-residual、belief-direct 和 belief-residual。四组均运行完整的 70 维平方根滤波器，并使用各自联合训练的 estimator 和 privileged critic。相同 seed 共享初始化文件。

正式训练和评测在 `supermimic/` 目录执行。先在 [train_eit_v7.toml](train_eit_v7.toml) 的 `[paths]` 中填写环境、动作数据与输出根目录，再在 `[methods.mean]` 和 `[methods.belief]` 中设置 GPU、seed 和 `residual_actor`。每个 seed 只生成一份初始化包，四组分别写入 `<experiment>/seed<seed>/<组名>-<direct|residual>/`。

```bash
.venv/bin/python -m supermimic.scripts.launch_eit --config train_eit_v7.toml --dry-run
.venv/bin/python -m supermimic.scripts.launch_eit --config train_eit_v7.toml
.venv/bin/python -m supermimic.scripts.eval_eit_validation run --config eval_eit_v7.toml
```

- 滤波状态保存世界系位姿和机体系速度。71 维交换格式继续使用世界系线速度，本地投影取向下重力、向上高度、机体系线速度和角速度、关节位置与速度。已知的 33 维仿真参数单独输入。
- Motion goal 的原点为当前参考的水平位置，方向为当前参考的 heading，因此每个未来帧保留其离地高度。十个未来帧在各自 clip 末尾截断。当前帧另供后验误差计算，下一帧原始关节角另供 PD 前馈。
- 两个 actor 的 FiLM 具有相同参数形状。归一化统计取自真实 replay 帧；mean 组在归一化之后屏蔽局部协方差，belief 组使用全部 2346 个下三角元素。残差模式在限幅之前加上 PD 前馈。
- `[initial_belief]` 控制真实 reset 扰动之上的独立 prior。恢复训练和推理从 checkpoint 读取六个标准差。恢复训练使用 `train_eit --resume <方法目录>/training.pt --output <方法目录> --motion-file '<训练动作路径>' --max-iterations <累计步数>`。
- 正式配置采用 32 步想象、10 秒物理 episode 和 40 个 clip × 16 个配对场景。评测按首次失败前的帧计分。1 秒相对位姿比较后验和真值各自在起始机体系中的运动增量，仅统计完整的 1 秒窗。局部 NEES 和 NLL 使用 67 维坐标，其中重力使用两个切平面分量。

短程验收每组使用 4 个物理环境，先训练 5 步再恢复到 8 步，累计 32 个真实转移和 6 次联合更新。联合验证使用同一行走片段的前 65 帧与两个配对 context。四组均在片尾前触发失败。轨迹较短，部分方法没有完整的 1 秒窗；对应指标保存为 `null`。[四组训练、推理和评测指标](docs/validation/eit_v7_short_runs.json)

批量编码对照在相同起点、权重和噪声下检查 latent、限幅后动作、return 和梯度，并对预热后的完整 actor–critic 更新计时。用 `python -m supermimic.scripts.benchmark_eit_encoder --motion-file <clip> --output <结果.json>` 复现；集群运行可增加 `--device cuda:0 --compile --cuda-graphs`，记录图回放和峰值显存。[CPU eager 与 Inductor 结果](docs/validation/eit_v7_encoder.json)
