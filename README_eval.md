# SuperMimic 评估与可视化

先完成 [环境安装](README.md#setup)，以下命令均在仓库根目录执行。
`play.py` 加载已有模型，在单个仿真环境中观察策略跟踪效果和世界模型预测，不更新权重。

EIT v7 支持 mean-direct、mean-residual、belief-direct 和 belief-residual。完整片段评测通过 `eval_eit_v7.toml` 选择 checkpoint；每条轨迹同时生成 tracking 和在线 odometry 两套结果。[训练与评测入口](README_eit.md)

## 50k Belief Input 不确定性扫描

50k 配置固定四个 `step-50000` 快照（内部 `collected_steps=50001`）、40 个完整 clip 和 16 个配对噪声 seed。离线报告由两批 nominal 原始轨迹重建。输出根目录为 `artifacts/eit-validation/uncertainty-50001`。[配置与归档身份](eval_eit_sweep_50k.toml)

```bash
.venv/bin/python -m supermimic.scripts.eval_eit_sweep baseline
.venv/bin/python -m supermimic.scripts.eval_eit_sweep run --pilot
.venv/bin/python -m supermimic.scripts.eval_eit_sweep run
.venv/bin/python -m supermimic.scripts.eval_eit_sweep report
```

`baseline` 在 CPU 上重算 nominal 主表、配对区间和时间曲线。`run --pilot` 在专用目录中验证 walk 和 fall/get-up 各 32 个动作，并保存噪声和动作以便核对。`run` 完成四个新增条件后生成全套报告。单独运行一档可用 `run --condition sensor-4`，后续用 `report` 从完整结果离线出图。重复运行会校验已存的 tracking 和同次 Belief 后验，按原批次布局续跑。

| 条件 | 初始先验标准差 | 观测标准差 | 实际新增轨迹 |
|---|---:|---:|---:|
| prior-2 / prior-4 | 2 / 4 倍 | 1 倍 | 每档 1,920；GT-Estimated、Mean、Belief 各 640 条 |
| sensor-2 / sensor-4 | 1 倍 | 2 / 4 倍 | 每档 3,200；五种模式各 640 条 |

先验倍率同时作用于首个观测前的均值扰动与协方差平方根。观测倍率同时作用于实际传感噪声和滤波器使用的测量标准差。标准化样本由 clip 内容、评测 seed 和噪声用途确定。先验扫描中，Prop.+KP 与 GT-Oracle 的结果沿用各自 nominal 轨迹。

- `report/<条件>/tracking-errors-over-time.png` 叠加五种方法的六项误差，包括骨盆中心化、heading 对齐的 MPJPE 以及全局根位置误差。横轴为动作开始后的秒数，每个 1 秒区间取实际执行帧的等权均值，含失败步。下图显示每个方法在该区间有有效帧的轨迹数；虚线给出由 clip 原始长度决定的计划轨迹数。`tracking-time.csv` 保存贡献轨迹数、帧数和各项均值，无有效帧的误差留空。
- `report/uncertainty-sweep.png` 绘制 SR、CR、P95 MPJPE 和尾部 cost 随两种倍率的变化。`uncertainty-sweep.csv` 保存各档估计、相对 nominal 的配对差值及 95% 区间。同一组 2,000 次 clip bootstrap 抽样保留该 clip 全部方法和 seed。
- `<条件>/posterior-report/` 使用该条件同次 rollout 的 Belief 后验和 tracking。校准按实际用于控制的帧计算，速度积分另含末动作后的终点；完成与失败轨迹分组汇总。历史 nominal 后验位于原先的 posterior 复跑目录。

## Belief Input 的 nominal 五模式主表

从仓库根目录运行，默认使用四个 step-5000 残差 checkpoint、40 个完整 clip 和 16 个评测 seed。五种模式共享每个场景的测量噪声，使用滤波的三种模式共享初始扰动。物理参数固定为摩擦系数 0.75、零质心偏移和零编码器偏移。每条轨迹从参考首帧起步。[权重路径和场景配置](eval_eit_nominal.toml)

```bash
.venv/bin/python -m supermimic.scripts.eval_eit_nominal --dry-run
.venv/bin/python -m supermimic.scripts.eval_eit_nominal run
.venv/bin/python -m supermimic.scripts.eval_eit_nominal report
```

单个 GPU 进程按模式分组做 actor 和滤波推理，再统一推进 MuJoCo。GT-Oracle 和 GT-Estimated 共用冻结的 GT actor，GT-Estimated 和 Mean-in-Loop 共用冻结的 Mean 估计器，各轨迹有独立的后验和动作。actor、仿真、评分和 CUDA 滤波重放均使用 `torch.inference_mode()`。EKF 首次捕获 CUDA 计算或使用 CPU 时，在 `no_grad` 下求协方差传播所需的单步状态 Jacobian。后验与动作均不保留跨步梯度图。

`--num-envs` 控制单卡容量。缩小验证规模时可重复传入 `--clip` 和 `--seed`，并用 `--output` 指定独立目录。以 `--layout sequential` 在同一组 clip 和 seed 上测量逐方法运行时间。重复 `run` 会校验并复用已完成的场景。

在 32 GB RTX 5090 上，40 个 clip 各取前 128 个动作做容量测试，1,600 和 3,200 个槽位分别达到约 3.41 万和 4.59 万有效控制步/秒。该卡的整段评测使用 `run --num-envs 3200`。完整片段包含提前失败的轨迹，实际吞吐见本次输出的 `timing.json`。

`tracking.csv` 包含五模式主表，`episodes.csv` 给出每条轨迹的失败原因、计划和执行步数、SR、CR 与代价。`report.json` 保存各项指标及三组方法差值的 95% 配对 clip bootstrap 区间。`episodes/` 中的逐步数据允许离线重建 P95。`timing.json` 记录运行容量、预热时间、有效控制步吞吐和 PyTorch 峰值显存。

误差按实际执行动作后的状态计算，含触发失败的动作。骨架 MPJPE 使用 G1 的骨盆、躯干及双侧髋、膝、踝、肩、肘、腕，共 14 个跟踪刚体。每套骨架分别平移到自身骨盆中心并去除自身 heading。SR、CR、Cost 和最差 10% 的 CVaR 按 episode 等权统计。失败后的剩余步每步计单位代价。

### Nominal 后验重复评测和离线报告

以下命令使用首次运行的四个 checkpoint 和相同噪声 seed，按原顺序将 3,200 个场景放进一个 mixed 批次。五种模式均保存 tracking。Belief-in-Loop 另存 640 个后验文件。重复评测的默认目录为 `artifacts/eit-validation/nominal-v1-posterior`。

```bash
.venv/bin/python -m supermimic.scripts.eval_eit_nominal run --record-belief-posterior --layout mixed --num-envs 3200
.venv/bin/python -m supermimic.scripts.eval_eit_nominal posterior-report
.venv/bin/python -m supermimic.scripts.eval_eit_nominal merge
```

`posterior-report` 和 `merge` 均可离线运行。通过 `--output` 指定重复运行目录，通过 `merge --baseline` 指定首次运行目录。续跑会按原布局重放含缺失或损坏文件的完整批次，同批次的 tracking 和后验一起更新。每个后验文件校验 checkpoint、场景、终止状态、执行帧散列和状态步号。

| 输出 | 口径 |
|---|---|
| `merged-report/pooled-tracking.csv`、`tracking.csv` | 合并主表每模式一行。 首次、本次、合并估计和本次减首次的差值，各附 95% 区间。以 clip 为单位配对重采样 2,000 次，同一 clip 的两次重复及其全部模式和 seed 一起抽取。 |
| `merged-report/paired-episodes.csv`、`repeat-differences.csv` | 按模式、clip、seed 对齐两次运行，列出成功翻转、执行步数和代价差异。 |
| `posteriors/<clip>/<seed>.npz` | τ 个实际动作对应 τ + 1 个状态。首帧为初始测量校正后的后验，末帧为实际末动作和终点测量更新后的后验。世界位置单位为 m，旋转矩阵将机体系映射到世界系，速度为机体系 m/s。 |
| `posterior-report/calibration.csv`、`validation.json` | 对本次 Belief 实际用过的 τ 个后验，误差取真值减均值。roll/pitch 使用 wrap 角差和 ZYX 边际方差。真值或后验的 ∣cos(pitch)∣ < 10⁻⁶ 时，roll 无唯一值。pitch 仍可计算点估计误差；后验在奇异姿态时，roll 和 pitch 的 Euler 方差均不可用。`accuracy_frames` 为误差统计分母，`valid_frames` 为覆盖率分母。联合 NEES 使用重力方向在后验切平面的球面对数、高度及机体系速度，以六维协方差做 Cholesky 求解。 |
| `posterior-report/velocity-integration.csv`、`episodes.csv` | 用真值朝向旋转估计速度，以实际控制周期做梯形积分，仅在起点对齐。按成功和失败分组，对 episode 等权平均；相对漂移要求真实三维路径长度大于 1 m。逐轨迹表含本次同轨 tracking 状态和指标。 |
| `merged-report/tracking.png` | 并列显示两次运行的点估计和合并结果的配对 clip bootstrap 95% 区间。 |
| `posterior-report/velocity-integration.png` | 完成轨迹与失败前缀的误差累积分布，菱形标出每组的 episode 等权均值。 |
| `posterior-report/*.png`、`coverage.csv` | 所有有效控制帧的覆盖率曲线，以及 seed 2718 的指定 walk 和 fallAndGetUp 示例时序、XY 路径和位置误差。 |

复核记录开关时，先用相同的 `--clip walk1_subject1.npz --clip fallAndGetUp1_subject1.npz --seed 2718 --num-envs 10 --audit-controls` 选项分别运行两次，在两个独立目录中关闭和开启 `--record-belief-posterior`。运行 `compare-controls --baseline OFF_DIR --output ON_DIR` 可生成逐场景的首步动作、全部共享动作和噪声前缀、终止状态及 tracking 帧差异。

## EIT 定量评估和播放

评估沿用 mjlab `tasks/tracking/scripts/evaluate.py` 的首个 episode 循环，直接调用其
MPKPE、root-relative MPKPE、关节速度 RMS 和末端位姿误差函数。每个环境从 clip
第 0 帧开始，仅统计首个 episode，直到全部结束。每次命令评估一个 motion 文件。

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m supermimic.scripts.eval_eit \
  --checkpoint logs/eit-v7/seed2027/belief-direct/training.pt \
  --motion-file data/motions/walk1_subject1.npz --device cpu \
  --num-envs 8 --output logs/eit-v7/seed2027/belief-direct/evaluation.json

.venv/bin/python -m supermimic.scripts.play SuperMimic-GenCoord-G1 \
  --eit-checkpoint logs/eit-v7/seed2027/belief-direct/training.pt \
  --motion-file data/motions/walk1_subject1.npz --viewer viser
```

评估和播放在每个真实 episode 开始时按 checkpoint 的带噪真值先验初始化，随后每帧用实际执行的动作和传感测量更新后验。四组均使用在线后验、固定归一化统计和确定性动作。mean 组屏蔽协方差输入，belief 组保留该输入。

| 当前快照 | 用途 |
|---|---|
| `eit_training_v7`，`stage=policy` | `train_eit` 续训、评估、播放和 ONNX actor 导出 |
| `eit_model_training_v5`，`stage=model` | `train_belief` 续训及固定 actor 评估 |

`eval_eit` 允许不同于训练数据的兼容 motion，记录训练和评估两份内容指纹；这不改变
续训的严格 motion 校验。末端名单由 `--ee-body-names` 指定，默认使用 G1 的左右
`ankle_roll_link` 和 `wrist_yaw_link`，不依赖已删除的 `ee_body_pos` 终止项。

结果中每个环境恰有一条 episode 记录。`reason` 区分 `failure`、`clip_complete`、
`timeout` 和 `evaluation_truncated`，物理失败优先于同时发生的 clip 结束。
`timeout` 布尔值表示普通时间上限，`clip_end` 表示抵达末帧，`termination_terms` 列出当步触发的环境终止项。
只有正常 clip 完成计为成功，普通 timeout 不算成功。

默认 `--protocol full_clip` 保留训练环境的初始扰动和物理失败项，将普通时限设在最后一个合法动作之后。结果记录控制周期 `control_period_s`、帧数 `clip_frames` 和配置秒数 `timeout_s`；`timeout_steps` 是环境向上取整后的实际步数。
使用 `--protocol legacy_10s` 可重现原有的 10 秒评估。可加 `--steps 100` 做有预算的 smoke。未结束轨迹仍保存，并标记 `censored=true`；
存在截断时 `success_rate=null`，不能用该 smoke 的完成率代替正式评估。
`mean_episode_return` 和 `mean_episode_length` 包括每条首轨迹已观察到的部分。

跟踪误差按每个 episode 的有效帧求均值，再在 episode 间等权平均。沿用原 evaluator
排除终止步已被自动 reset 替换的状态；回报和 episode 长度仍包含终止动作。
`metric_steps=0` 的轨迹误差为 null，并从误差均值排除；`metric_episode_count`
报告实际纳入的轨迹数。相对身体指标使用评分帧参考与动作后当前根位姿重新对齐，
与 EIT 奖励一致。后验 RMSE、variance 和 innovation 在动作前、actor 使用的
同一后验上计算，只统计首个 episode。

EIT viewer 的蓝紫色半透明模型标为 `online_posterior`，显示当前在线后验均值；
它不是第二条世界模型预测轨迹。暂停、单步、真实 reset 和 clip 切换沿用原 viewer。
`num_steps_rollout` 仅用于下述旧策略预测可视化，不重置 EIT 在线后验或历史。
相同 seed 保证首次 reset/测量可复现，不承诺失败时间不同后的全程噪声配对。

## 共同初始化、快照与批量选模

共享初始化与训练流程见 [EIT 训练教程](README_eit.md)。训练只覆盖本地最新
`training.pt`；需要比较多个历史候选时，先保留各保存点或从 model artifact 取得文件。

冻结场景后先运行验证，再对每组每个 seed 的入选快照运行测试。
清单、重试与逐 clip 等权汇总见 [批量评估](docs/eit_batch.md)。

## 独立诊断和 ONNX actor

```bash
PY=.venv/bin/python
motion=data/motions/walk1_subject1.npz
ckpt=logs/eit-v7/seed2027/belief-direct/training.pt
out=logs/eit-v7/diagnostics
"$PY" -m supermimic.scripts.eval_eit --checkpoint "$ckpt" --motion-file "$motion" \
  --num-envs 1 --seed 4795 --posterior-bin-steps 50 --record "$out/episodes.pt" \
  --export-onnx "$out/actor.onnx" --output "$out/evaluation.json"
"$PY" -m supermimic.scripts.diagnose_eit --checkpoint "$ckpt" --recording "$out/episodes.pt" \
  --window-frames 8 --imagined-horizon 128 --imagined-samples 4 --output "$out/offline.json"
"$PY" -m supermimic.scripts.eval_eit --checkpoint "$ckpt" --motion-file "$motion" \
  --onnx-actor "$out/actor.onnx" --device cpu --num-envs 1 --seed 4795 --output "$out/onnx.json"
```

`--record` 保存每个环境首个 episode 的实测序列、已执行动作、真值、参考目标、上下文、实际初始 prior、奖励和终止原因。用同一份记录跨模型打分时选择 `--mode process` 或 `--mode posterior`。critic 的折扣回报诊断使用本快照控制的独立 episode。A 组另用 `--truth-actor` 采集训练输入条件。可用 `--no-posterior-diagnostics` 关闭在线统计。记录格式、删失处理和固定轨迹漂移验收见[诊断说明](docs/eit_operations.md#固定轨迹诊断)。

EIT 导出图接受 CPU float32 的 `history[1,5,93]`（最旧帧在前）和 `goal[1,700]`。C 组额外接受 `uncertainty[1,450]`。输出 `actions[1,29]` 是限幅前的归一化动作。图内含归一化、motion encoder、decoder 和 C 的 FiLM。图的 `eit_contract` 元数据包含特征排列、关节顺序、动作仿射映射和快照身份。适配器在图外完成滤波、历史、参考构造及限幅。当前 CPU 执行使用 ONNX ReferenceEvaluator，`onnx_timing` 记录本次图执行的平均耗时。播放时将 `--eit-onnx-actor "$out/actor.onnx" --device cpu` 加到对应快照的 `play` 命令。

## 旧 checkpoint 启动播放

编辑 [play_config.toml](play_config.toml)，填写自己的 W&B 用户名或团队名。例如固定使用 `v32`：

```toml
wandb_path = "YOUR_ENTITY/supermimic/91lwm1v7"
checkpoint = "v32" # 也可使用 v25 或 latest；配置文件默认 latest
motion_file = "data/motions/*.npz"
viewer = "viser"
initial_task = "motion"
```

```bash
uv run src/supermimic/scripts/play.py SuperMimic-GenCoord-G1 \
  --config play_config.toml
```

打开终端打印的 Viser 地址。任务名须与训练一致：广义坐标用
`SuperMimic-GenCoord-G1`，最大坐标用 `SuperMimic-MaxCoord-G1`。
`viewer = "native"` 使用本地窗口；`"auto"` 按是否有显示环境选择 native 或 viser。

**命令行显式参数 > TOML 文件 > 默认值**；TOML 用下划线，CLI 用连字符。
路径相对于启动目录，修改配置后需重启。临时换版本、只看一条动作：

```bash
uv run src/supermimic/scripts/play.py SuperMimic-GenCoord-G1 \
  --config play_config.toml --checkpoint v25 \
  --motion-file data/motions/run1_subject5.npz
```

`motion_file` 支持单个 NPZ 或 glob；CLI 中的 glob 要加引号。
省略该项时使用 run 关联的 motion artifacts，因此使用本地动作训练的 run 应显式填写路径。
完整参数见上述启动命令追加 `--help`。

## 旧策略的实体机器人和两个半透明机器人

以下是默认配色，实体指画面中的不透明模型，仍是仿真机器人。

| 外观 | 表示什么 | 状态如何产生 |
| --- | --- | --- |
| 不透明机器人 | 策略在物理仿真中的实际表现 | `actor.pt` 根据仿真观测输出动作，由 MuJoCo 推进，包含接触、碰撞等物理效果。 |
| 绿色半透明机器人 | 参考动作，表示要模仿的目标 | 直接显示当前 motion 帧的根部姿态和关节位置，不参与物理仿真。 |
| 蓝紫色半透明机器人 | 世界模型中的策略预测轨迹 | 从仿真状态初始化；同一个 actor 根据预测状态生成动作，再由 `world_model.pt` 递推下一状态。仅作为可视化叠加。 |

**实体与绿色的差异用于观察动作跟踪效果；实体与蓝紫色的差异用于观察世界模型闭环推演的偏离。**
两个策略分支分别根据自己的状态计算动作，动作可能不同，因此后一种差异包含预测与控制的累积影响，
不能直接当作“相同动作输入下的单步预测误差”。

`num_steps_rollout` 控制世界模型连续推演多少步后重新对齐仿真状态，默认 `1000000`；
仿真回合结束或手动重置时也会重新对齐。它不是播放总步数。
可调小此值观察短时预测；设为 `1` 时每步都从当前仿真状态重新开始预测。

Viser 常用控件：

- **Motion Clip**：选择动作，并从该动作的随机帧重置；每次观察的起点可能不同。
- **Play/Pause、Step、Reset Environment**：播放/暂停、单步、重置。
- **Task**：切换策略任务分支；只评估实际训练过的分支。`motion: 1, velocity: 0` 的 run 应使用 motion。

推演和预测配色见 [viewers.py](src/supermimic/scripts/viewers.py)；绿色参考模型由
[MotionCommand](src/supermimic/motion_command.py) 继承的 mjlab tracking 可视化绘制。
两种坐标表示均使用 `add_ghost_mesh()`；MaxCoord 预测按各跟踪刚体的位姿绘制。

## Run ID 与 checkpoint

`wandb_path` 格式为 `entity/project/run_id`。打开 W&B 运行页面，网址中
`/runs/91lwm1v7` 的最后一段就是 Run ID，与运行显示名称、checkpoint 版本号不同。

- `checkpoint = "latest"`：加载带 latest 别名的 model artifact。
- `checkpoint = "v25"`：固定加载该 run 输出的 v25；不存在或已删除时会报错并列出可用版本。
- 启动日志会打印实际加载的 artifact 名称；同一版本的 `actor.pt`、`world_model.pt` 一起加载，`critic.pt` 不用于播放。

本项目模型集合命名为 `model-<run_id>`。本地三个 `.pt` 是一套模型的组成部分，
每次保存会覆盖同名文件；远端按版本保存，清理规则保留最新 5 个版本、版本号为 25 倍数的版本，
以及带 alias 的版本。规则见 [wandb_artifacts.py](src/supermimic/utils/wandb_artifacts.py)。
旧 checkpoint 播放仍需 W&B run/config/artifact，不能直接把本地 checkpoint 目录填入
`wandb_path`；EIT 使用前述 `--eit-checkpoint`。

## 观察哪些指标

打开训练终端的 `View run at ...` 链接查看曲线：

| 指标 | 用途 |
| --- | --- |
| `Metrics/motion/error_*` | 检查位置、姿态、关节及速度的跟踪误差。 |
| `Train/*`、`Episode_Termination/*` | 结合回报、回合长度和终止原因判断稳定性。 |
| `Loss/*`、`Policy/*`、`Episode_Reward/*` | 查看世界模型学习、策略/价值函数及奖励分项。 |
| `Perf/*` | 查看训练吞吐与耗时。 |

这些是训练统计；`play.py` 用于交互观察，不自动生成跨动作的定量评估报告。
比较 checkpoint 时应固定 motion 和仿真设置，并观察多次重置，避免只比较某一次随机起点。
[benchmark.py](src/supermimic/scripts/benchmark.py) 测量仿真和世界模型的 steps/s，
不衡量跟踪精度；目前只加载 latest，使用首个关联 motion artifact，且不读取 `play_config.toml`。

## 仿真警告

在 `play_config.toml` 中调整对应参数后重启：

| 警告 | 参数 | 当前示例值 |
| --- | --- | --- |
| `broadphase overflow` | `[sim]` 的 `nconmax`：接触/碰撞候选缓冲区容量 | `256` |
| 约束数量超过 `njmax` | `[sim]` 的 `njmax`：约束容量 | 未显式设置；需要时可设 `1024` |
| `iterations limit reached` | `[sim.mujoco]` 的 `iterations`：求解器迭代上限 | `100` |
| `linesearch iterations limit reached` | `[sim.mujoco]` 的 `ls_iterations`：线搜索上限 | `50` |

未显式设置时，以上参数从 run 保存值回退到任务默认值；`nconmax` 额外保证至少为 `256`。
若仍报警，继续增大对应参数并检查机器人状态。更大的容量/上限会增加内存或计算开销，
不能保证所有状态都收敛；关闭打印也不会解决溢出。
