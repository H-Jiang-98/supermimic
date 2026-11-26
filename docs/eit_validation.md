# 同一条轨迹上的 Tracking 与 Odometry 评测

在仓库根目录运行下列命令。它读取 `eval_eit_validation.toml`，执行选中的策略，并生成两份报告、CSV 和图表。

```bash
.venv/bin/python -m supermimic.scripts.eval_eit_validation run
```

## 选择方法与续跑

下例选择 C 方法。每次 `run` 执行的策略由所用 TOML 的 `methods` 列表决定。

```toml
[evaluation]
methods = ["C-posterior"]
```

可选值为 `A-posterior`、`B-posterior`、`C-posterior`、`A-truth`。每种策略会产生自己的闭环轨迹，每个 clip/context 在该策略下只做一次正式仿真。两套评分共用每帧的真值和 C 估计器后验。A-truth 使用真值控制，仍在线记录 C 的估计误差。

默认配置使用 40 个 clips、每个 clip 16 个 Sobol contexts，一次并行全部 640 个场景。路径相对于 TOML 文件。可以用 `--config` 选择另一份配置，用 `--output` 指定结果目录。

例如，目录里已经跑完 A 时，可以把 `methods` 改为 `["C-posterior"]`，保留原来的 `output` 并执行 `run`。程序会检查分片身份、校验和及两套评分状态，只补跑 C 缺失或无效的任务。C 完成后，两套报告会同时展示 A 和 C。绘图会读取该目录中所有已完成的方法，包括上次完成、本次未选中的方法。每套报告分别检查该方法的所有 clips 和 contexts 是否都有完整有效的分片。缺失、截断或数值无效的方法会在相应报告中标出。

```bash
# 从现有结果重新生成两套图表
.venv/bin/python -m supermimic.scripts.eval_eit_validation report

# 在 Viser 中同步播放选定场景的参考、真值和后验
.venv/bin/python -m supermimic.scripts.eval_eit_validation view
```

`report` 可以在其余方法尚未完成时使用。退出码 0 表示本次配置要求的所有方法都已完成两套评分，退出码 1 表示仍有缺项。调整 `boxplot_interval_s` 或其他绘图参数后，`run` 和 `report` 均可复用原目录。用于计分的 `time_bin_s` 须与原实验一致。旧版单任务结果可用 `report --output <旧实验目录>` 重绘该目录已有的指标。

## 结果目录

默认输出到 `artifacts/eit-validation/joint-v5/`。v5 在 v4 的失败截断规则下对全部 context 新增位置和姿态分量计分。v4 文件可重绘原有五项 odometry 指标，完整六分量图需使用 v5 结果。v3 的时间箱包含失败后的误差，且失败判定时序与 v4 不同。使用 v4 计分需要重跑仿真。

| 路径 | 内容 |
|---|---|
| `tracking/figures/` | 每种已完成方法的八指标时间曲线、逐 clip MPKPE 热图，以及跨方法的 `tracking-success-by-clip.svg` |
| `odometry/figures/` | 每种已完成方法的五指标时间分箱图，以及独立的六分量图 `*_pose_components_time.svg`，包括所有 clips 汇总及各代表性片段（SVG） |
| `tracking/`、`odometry/` | 各自的 `report.json`、`clip_metrics.csv`、`context_metrics.csv` |
| `tasks/` | 每个分片共享一个 NPZ 和一份 JSON。NPZ 的 `full`、`bins` 是 tracking 指标，`odometry_full`、`odometry_bins` 是 odometry 指标，列名与完成状态保存在同名 JSON |
| `manifest.json`、`domains.json` | 实验协议、checkpoint 和动作文件指纹、源码摘要及域参数样本 |
| `tracking_timing.json` | 每种方法最近一次实际执行评测时的计时。正式计时包含场景初始化、推理、仿真、两套指标计算、传输和分片落盘。编译和预热单列，绘图在计时结束后进行 |

补跑部分 clips 会覆盖该方法先前的计时记录。按当前每个 clip 一个分片的配置，核对 10 分钟目标时，先确认 `clips=40` 且 `resumed_subset=false`，再检查 `evaluation_wall_s` 是否小于或等于 600 秒。

## 指标与统计

**Tracking** 比较仿真真值与当前帧动作参考，报告 root 旋转、位置、速度 RMSE，世界系 MPKPE，末端位置和姿态误差，以及关节位置和速度 RMSE。参考在片头对齐 yaw/xy。世界系误差保留后续的整体漂移。

**Odometry** 比较同一帧的 C 后验与仿真真值。每种方法名表示采集轨迹时使用的控制策略。指标包括 root 旋转、位置、速度 RMSE，以及 RPV9 联合边缘的 NLL 和 NEES。平方根协方差保留旋转、位置与速度间的相关性，采用 float64 三角求解。NLL 包含高斯常数，NEES 的参考值为 9。旋转 RMSE 使用度，位置和速度分别使用 m 和 m/s。

六分量图分别展示世界坐标中根位置 X/Y/Z 的 RMSE（m）和根姿态 roll/pitch/yaw 的 RMSE（度）。每帧把后验和真值姿态分别按 Rz(yaw) Ry(pitch) Rx(roll) 转为欧拉角，再取各轴最短角差。例如 yaw 从 +179° 到 −179° 的误差幅值为 2°。pitch 接近 ±90° 时 roll 和 yaw 会耦合，奇异处约定 roll 为零。统计范围、失败截断和时间分箱与原 odometry 图一致。

每条长度为 L 的轨迹最多计分 L 帧并执行 L−1 次动作。默认控制周期为 0.02 秒，每帧采样。`time_bin_s=2.0` 表示每个完整统计箱包含 100 帧，末箱按实际帧数计算。横轴给出箱起点的秒数，第 50 号箱从 100 秒开始（编号从 0 起）。

两套指标都只计入首次跟踪失败之前的帧。失败发生在箱内时，该 context 的指标仅使用箱内失败前的帧；之后的空箱记为 NaN。每个箱对有计分帧的 context 等权，RMSE 先在各 context 内按帧计算。所有 context 均已失败的箱在图中留空。

Tracking 时间曲线展示每个统计箱。每个箱先在各 clip 内对 contexts 取均值，再用该箱仍有数据的 clips 计算中位数和四分位区间。图底部显示参与统计的 clip 数，全片汇总中每个 clip 等权。

Odometry 箱线图按 `boxplot_interval_s` 选取要画的统计箱，该间隔须为 `time_bin_s` 的正整数倍。当前配置为 4 秒，图上依次展示 `[0, 2)`、`[4, 6)`、`[8, 10)` 秒等区间。汇总图中每个样本是一个 clip 在该箱内的 context 均值。代表性片段图中每个样本是该 clip 的一个 context。

## 执行和回放

默认 CUDA 编译模式使用冻结 C 的完整平方根滤波器、checkpoint 的 Jacobian 后端和配对测量噪声。正式运行前在固定输入上校验 eager/compiled 结果，并按 `warmup_steps` 预热，随后恢复片头状态。短片调试可显式设置 `compile_kernels=false`、`device="cpu"` 和 `max_frames`。截断结果保留在分片中。

每个 clip/context 的 `first_failure_s` 记录首次触发跟踪失败条件时距片头的秒数。沿用训练环境的物理终止项（当前为 `anchor_pos` 和 `anchor_ori`），每次物理步结束后、参考动作推进前检查。触发时记录该步结束的时间，两套评分都排除这一帧及其后续帧；回放可继续到片尾。全程未触发时，NPZ 中该值为 `NaN`。

成功率图每行一个 clip，每列一种已完成方法。`15/16` 表示 16 个 context 中有 15 个成功跟踪到末帧。有失败的格子另列首次失败时间中位数和片尾时刻（秒），中位数只取失败的 context。片尾时刻为 (L−1) × 控制周期。相同统计也写入 `clip_metrics.csv` 和 `report.json`。

数值异常按场景屏蔽。A-truth 的后验异常会使其 odometry 数据不完整，真值控制和 tracking 可继续。默认回放保存 context 0、N//2、N−1 的三层位姿，全部 N 个 contexts 都参与两套评分。场景的 reset、prior 和传感噪声种子按 clip 指纹与 context ID 配对。
