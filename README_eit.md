# EIT v7 四方法训练

四种方法在真实采集时都运行 70 维平方根滤波器，并保存同一格式的 posterior replay。世界模型使用相同的训练流程。策略训练从保存帧恢复本体感知历史、动作、物理参数 ξ 和参考对齐。各方法在想象过程中的区别如下。

| 方法 | FiLM 输入 | 想象过程 | Critic 输入 |
|---|---|---|---|
| Prop.+KP (`prop_kp`) | ξ、goal latent | 仅推进世界模型 | 公共 privileged 输入 |
| GT-Modulation (`gt`) | ξ、goal latent、68 维真值局部状态及参考误差 | 仅推进世界模型，用当前生成状态计算局部特征 | 公共 privileged 输入 |
| `mean` | ξ、goal latent、后验局部均值及参考误差 | 推进世界模型并滤波 | 公共输入、局部后验均值、投影协方差 |
| `belief` | mean 的输入和投影协方差 | 推进世界模型并滤波 | 同 mean |

公共 critic 输入包含真值状态、privileged goal、历史和 ξ。四个 actor 共用相同的网络形状，decoder 都接收 goal latent 和历史。每个 seed 共用初始化包，两个 baseline critic 从完整 critic 复制公共输入列、偏置和后续层。包含任一 baseline 的 seed 在启动训练队列前检查 v6 初始化包。仅运行 mean 或 belief 的 seed 也可复用 v5。任一方法生成的兼容 v6 包都可以复用。

在 `train_eit_v7.toml` 的 `[paths]` 设置环境、数据与输出根目录。默认 seed 为 2027，GPU 0 顺序运行 prop_kp 和 mean，GPU 1 顺序运行 gt 和 belief。每个方法表的 `residual_actor` 默认为 true，可设为 false 运行 direct，或用布尔列表选择多个模式。运行目录为 `<experiment>/seed<seed>/<方法>-<direct|residual>/`。[训练配置](train_eit_v7.toml)

```bash
.venv/bin/python -m supermimic.scripts.launch_eit --config train_eit_v7.toml --dry-run
.venv/bin/python -m supermimic.scripts.launch_eit --config train_eit_v7.toml
```

- 滤波状态保存世界系位姿和机体系速度。71 维交换格式继续使用世界系线速度，本地投影取向下重力、向上高度、机体系线速度和角速度、关节位置与速度。已知的 33 维仿真参数单独输入。
- Motion goal 的原点为当前参考的水平位置，方向为当前参考的 heading，因此每个未来帧保留其离地高度。十个未来帧在各自 clip 末尾截断。当前帧另供后验误差计算，下一帧原始关节角另供 PD 前馈。
- FiLM 输出层初始化为零。归一化统计取自真实 replay 帧，禁用通道在归一化后置零。belief 组使用投影协方差的全部 2346 个下三角元素。残差模式在限幅之前加上 PD 前馈。
- `[initial_belief]` 控制真实 reset 扰动之上的独立 prior。恢复训练和推理从 checkpoint 读取六个标准差。恢复训练使用 `train_eit --resume <方法目录>/training.pt --output <方法目录> --motion-file '<训练动作路径>' --max-iterations <累计步数>`。
- 正式配置采用 32 步想象、10 秒物理 episode 和 40 个 clip × 16 个配对场景。评测按首次失败前的帧计分。1 秒相对位姿比较后验和真值各自在起始机体系中的运动增量，仅统计完整的 1 秒窗。局部 NEES 和 NLL 使用 67 维坐标，其中重力使用两个切平面分量。

本次四方法小批验证在空闲 GPU 上执行，包含 FiLM 通道、后验隔离、checkpoint 续训和两种 baseline 的编译及 CUDA graph 对照。[验证结果](docs/validation/eit_v7_four_methods.json) · [原始日志](docs/validation/eit_v7_four_methods_logs.tar.gz)

此前 mean/belief × direct/residual 的短程验收每组使用 4 个物理环境，先训练 5 步再恢复到 8 步，累计 32 个真实转移和 6 次联合更新。联合验证使用同一行走片段的前 65 帧与两个配对 context。四组均在片尾前触发失败。轨迹较短，部分方法没有完整的 1 秒窗；对应指标保存为 `null`。[四组训练、推理和评测指标](docs/validation/eit_v7_short_runs.json)

批量编码对照在相同起点、权重和噪声下检查 latent、限幅后动作、return 和梯度，并对预热后的完整 actor–critic 更新计时。用 `python -m supermimic.scripts.benchmark_eit_encoder --motion-file <clip> --output <结果.json>` 复现；集群运行可增加 `--device cuda:0 --compile --cuda-graphs`，记录图回放和峰值显存。[CPU eager 与 Inductor 结果](docs/validation/eit_v7_encoder.json)
