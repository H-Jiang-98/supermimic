# SuperMimic 评估与可视化

先完成 [环境安装](README.md#setup)，以下命令均在仓库根目录执行。
`play.py` 加载已有模型，在单个仿真环境中观察策略跟踪效果和世界模型预测，不更新权重。

旧输入布局继续使用 W&B 模型和原有 viewer。`eit_rpv_v1` 的 A/B/C 现在支持本地
`training.pt` 的定量评估与播放，共用 `EITPolicyAdapter`。benchmark 和 ONNX exporter
尚未接入 EIT。训练及梯度 smoke 仍使用 README 中的 `check_estimator` / `check_eit_training`。

## EIT 定量评估和播放

评估沿用 mjlab `tasks/tracking/scripts/evaluate.py` 的首个 episode 循环，直接调用其
MPKPE、root-relative MPKPE、关节速度 RMS 和末端位姿误差函数。每个环境从 clip
第 0 帧开始，仅统计首个 episode，直到全部结束。每次命令评估一个 motion 文件。

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m supermimic.scripts.eval_eit \
  --checkpoint logs/eit-rpv-v2/B/training.pt \
  --motion-file data/motions/walk1_subject1.npz --device cpu \
  --num-envs 8 --output logs/eit-rpv-v2/B/evaluation.json

.venv/bin/python -m supermimic.scripts.play SuperMimic-GenCoord-G1 \
  --eit-checkpoint logs/eit-rpv-v2/B/training.pt \
  --motion-file data/motions/walk1_subject1.npz --viewer viser
```

本地 EIT 播放不需要 W&B；不要同时传 `--wandb-path`。三组均以 eval 模式使用在线后验
均值，C 额外读取任务 covariance，动作探索关闭，传感噪声和在线滤波保留。
归一化固定，不创建 replay、critic 或 optimizer。每次真实 reset 按 checkpoint 的带噪
真值先验初始化；这是仿真的初始化知识假设，不能解释为已经支持无真值的实机初始化。

评估和播放要求 A/B/C 的联合 checkpoint 内嵌当前 `eit_rpv_belief_critic_v2` 契约。
加载器在创建推理模型前校验契约版本，并将声明的 context 宽度与世界模型的 context 宽度、声明的输入宽度与 critic 首层权重逐一核对。
三组训练和初始化包的用法见 [训练说明](README.md#estimator-第四步完整策略与模型交替训练)。

`eval_eit` 允许不同于训练数据的兼容 motion，记录训练和评估两份内容指纹；这不改变
续训的严格 motion 校验。末端名单由 `--ee-body-names` 指定，默认使用 G1 的左右
`ankle_roll_link` 和 `wrist_yaw_link`，不依赖已删除的 `ee_body_pos` 终止项。

结果中每个环境恰有一条 episode 记录。`reason` 区分 `failure`、`clip_complete`、
`timeout` 和 `evaluation_truncated`，物理失败优先于同时发生的 clip 结束。
`timeout` 布尔值保留 mjlab 原始标志，clip 完成时也可能为 true，应以 `reason` 分类。
只有正常 clip 完成计为成功，普通 timeout 不算成功。

可加 `--steps 100` 做有预算的 smoke。未结束轨迹仍保存，并标记 `censored=true`；
存在截断时 `success_rate=null`，不能用该 smoke 的完成率代替正式评估。
`mean_episode_return` 和 `mean_episode_length` 包括每条首轨迹已观察到的部分。

跟踪误差按每个 episode 的有效帧求均值，再在 episode 间等权平均。沿用原 evaluator
排除终止步已被自动 reset 替换的状态；回报和 episode 长度仍包含终止动作。
`metric_steps=0` 的轨迹误差为 null，并从误差均值排除；`metric_episode_count`
报告实际纳入的轨迹数。相对身体指标使用评分帧参考与动作后当前根位姿重新对齐，
与 EIT 奖励一致。后验 RMSE、variance、jitter 和 innovation 在动作前、actor 使用的
同一后验上计算，只统计首个 episode。

EIT viewer 的蓝紫色半透明模型标为 `online_posterior`，显示当前在线后验均值；
它不是第二条世界模型预测轨迹。暂停、单步、真实 reset 和 clip 切换沿用原 viewer。
`num_steps_rollout` 仅用于下述旧策略预测可视化，不重置 EIT 在线后验或历史。
相同 seed 保证首次 reset/测量可复现，不承诺失败时间不同后的全程噪声配对。

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
