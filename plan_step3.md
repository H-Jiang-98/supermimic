# 第三步　拟合世界模型的预测均值与过程 covariance

`train_belief` 每轮用固定 actor 采集真实轨迹，依次拟合动力学均值与条件过程 covariance。`train_eit` 使用公共 `runner.step()` 和 `Ratio` 逐步采集、调度更新组，复用同一模型 fitter，再训练 actor 和 critic。A/B/C 使用相同的模型拟合目标。[策略训练](plan_step4.md)

## 每轮的数据

`train_belief` 使用模型专用 replay，每帧保存七个字段。均值和 covariance 拟合读取 `state`、`action`、`sim_ctx`，窗口索引读取 `episode_id`、`time_index`、`terminated`、`truncated`。采集 actor 继续使用在线滤波与实测历史，上一动作在物理步前核对。模型 replay 封存、划分 episode 和枚举窗口均不消耗 RNG。

`train_eit` 的完整 replay 还保存实测 `actor_obs`、`prev_action`、参考时间和旋转/平移对齐。独立缓存按全局采集步保存执行动作前的后验。两个入口共用下列模型窗口规则。

1. `train_belief` 使用本轮开始时的模型采集；`train_eit` 使用当前模型逐步采集。按物理 episode ID 分组，默认将 `episode_id % 5 == 0` 的 episode 留作验证。同一 episode 的所有帧始终属于同一组。
2. 在均值更新前，用本轮新采集的训练帧更新状态和动作归一化，每帧一次。动作经过与动力学前向相同的处理。采集超过环形 replay 容量时，在帧被覆写前把该批统计累加到独立副本。封存时检查保留数据有限且有效转移的仿真上下文一致。完整 replay 还可逐帧核对测量中的上一动作和相邻记录的执行动作。校验通过后才将归一化统计写入模型。
3. 封存时按环境和时间扫描一次保留帧的后继关系。每条转移要求 episode 相同、帧号恰好加一、前一帧没有终止或截断。均值窗口和一步对共用这份索引，按 episode 取训练和验证子集。跨物理存储环尾的连续窗口可用。每个片段末帧只有在后继已经记录时才能作为转移起点。
4. 从训练集有放回抽取完整窗口训练均值。随后从有效一步转移中均匀、有放回抽样训练 covariance，抽样次数由 mean 更新预算推导。两到七帧的短片段在默认配置下也贡献一步转移。单帧没有模型拟合目标，若属于训练集且保存了后验，仍可作为策略起点。

## 预测均值

首帧取记录的真实状态。每步应用记录的动作，把预测状态与积分得到的根位姿回喂给下一步。默认窗口有 8 帧，评分 7 个真实后继。实现直接调用原版 `WorldModelLearner` 的 rollout、损失和优化器分组。[确定性 learner](src/supermimic/learners/wm_learner.py)

| 损失分组 | 比较量 | 通道数 d | 标称误差 ε |
|---|---|---:|---:|
| `root_pos` | 积分后的根位置 | 3 | 0.01 |
| `root_ori` | 积分后的根旋转矩阵元素乘以 √½ | 9 | 0.02 |
| `joint_pos` | 关节位置 | 29 | 0.1 |
| `joint_vel` | 关节速度 | 29 | 3.14 |

每组 MSE 对样本、预测步和组内通道取平均。总损失为 Σ(d × MSE / ε²) / Σd。中间线性层权重使用 Muon，首末层权重和偏置使用 Adam。默认学习率均为 `2e-4`，每轮 10 次更新，每批 2048 个窗口。这一阶段只优化均值网络。EIT 在这次 eager 更新内关闭 TorchScript 的自适应优化，使四元数运算在恢复后的首次更新中也使用相同的计算路径。

## 条件过程 covariance

均值更新后冻结均值网络和归一化统计。对每对真实相邻帧，用真实前帧 sₜ、动作 aₜ 和仿真上下文 ξₜ 重新计算预测 μₜ₊₁ = `mean_lie(sₜ, aₜ, ξₜ)`。目标 rₜ = `local_error(μₜ₊₁, sₜ₊₁)` 表示从预测均值到真实后继的 70 维右局部扰动。输入和残差仅对当前抽中批次在无梯度模式下提取，拟合后释放，不保留全量残差缓存。

`noise_head` 读取同一组归一化输入，输出 2485 个数，构造 70×70 的完整下三角因子 L。对角线使用 softplus 加正下限。同一行的非对角系数以该行对角元素为尺度，保留各物理量之间的交叉项。预测过程 covariance 为 LLᵀ。[动力学实现](src/supermimic/models.py)

对每个残差用下三角求解得到 z = L⁻¹r。包含归一化常数的单样本损失为 ℓ = ½‖z‖² + Σ log Lᵢᵢ + 35 log(2π)。独立的 Adam 只更新 `noise_head`，默认学习率 `2e-4`、每批 8192 对。每组抽样次数为 mean updates × batch × (window−1)，默认 143,360，即 17 个整批和一个 4096 尾批，共 18 次更新。损失按实际抽样数加权汇总，允许重复抽样。[两阶段实现](src/supermimic/learners/two_stage_wm.py)

## 策略、验证与恢复

- `train_belief` 始终固定 actor。`train_eit` 每个采集步用新训练帧及新保存 posterior 各更新一次对应归一化，包括 warmup；同一步多组拟合不重复计数。策略起点从整个 replay 中符合条件的训练帧抽取。想象保留动作穿过均值、过程 covariance 和滤波更新的梯度。下一采集步使用新模型和新 actor，延续物理 episode、belief 和观测历史。
- `metrics.jsonl` 分别记录累计 `mean_steps`、`cov_steps`、训练与验证 episode、采集和训练后的模型版本。各集合内的 `Loss/state` 是加权 rollout 误差，四个分组值是原始 MSE，`Loss/process_nll` 按实际转移数平均。`mean_windows` 是实际抽取的窗口数，`cov_pairs` 是可用的一步对数。
- 有验证完整窗口时，默认抽取 4 个打分。验证集只有短片段时，记录 `mean_windows=0`，仍独立、有放回抽取 8192 个验证转移计算 NLL。训练与验证的 RNG 独立且可恢复。模型拟合需要至少一个完整训练窗口和一对验证转移；EIT 还要求可用 posterior，warmup 跳过的机会不补欠账。
- `stage=model` 使用 `eit_model_training_v4`，保存固定 actor 的模型训练状态。`stage=policy` 使用 `eit_training_v5`，另保存 critic、策略配置和优化器、后验保存间隔、缓存协议及模型修订号。跨进程续训恢复对应组件和 RNG，使用空 replay 开始新的物理轨迹。共同初值使用 `eit_initialization_v3`。[训练与恢复](README.md#estimator-第四步完整策略与模型交替训练)
- 纯模型训练的轮次存档保存当前 RNG，可在相同版本及相同输入下复核下一次优化器更新。

## 运行

对固定 actor 运行两轮并使用默认模型拟合配置：

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m supermimic.scripts.train_belief \
  --motion-file data/motions/walk1_subject1.npz --device cuda:0 \
  --output logs/belief-b --group B --num-envs 4 --rounds 2 \
  --collection-steps 12 --buffer-size 24
```

两个训练入口共用 `--mean-window-frames`、`--mean-updates`、`--mean-batch-size`、`--mean-lr`、`--cov-batch-size`、`--cov-pair-budget`、`--cov-validation-pairs` 和 `--cov-lr`。运行 `train_eit` 可额外设置 `--policy-horizon` 等策略参数。恢复时提供同一 motion、对应入口的 `--resume <输出目录>/training.pt` 和 `--output`。`train_belief --rounds` 指新增轮数；`train_eit --max-iterations` 指累计物理采集步目标。当前保存与离线记录配置见 [训练说明](docs/train_eit_pro6000.md)。下文 v3 验收记录保留原始配置，不代表当前预算。

## 专用 replay 验收（v3）

`train_belief`（model replay）和 `train_eit`（full replay）分别使用 5 个并行 MuJoCo 环境，每轮采集 8 步，每个环境保留 7 帧。两个入口均完成两轮训练，再从各自 v3 存档恢复并训练第三轮。第三轮结束时，两个入口的均值累计更新 3 次，过程 covariance 累计更新 9 次。

| 第三轮实测 | `train_belief` | `train_eit` |
|---|---:|---:|
| 每环境每帧的回放字段存储量（字节） | 550 | 1094 |
| 连续后继关系扫描次数 | 1 | 1 |

七字段 replay 每帧节省 544 字节（49.7%），表中只统计按帧存储的字段，不含索引、先验及模拟器显存。该次验收另用固定状态流模拟物理采集，在基线提交和 v3 上分别运行两轮 EIT。两边的全部训练指标及全局、策略 RNG 状态指纹完全一致。

RTX 5090 上的 `train_belief` 使用默认均值网络和拟合批量，5 个环境各采集 12 步、各保留 10 帧。单轮完成 10 次均值更新，每次抽取 2048 个 8 帧窗口。训练集有 36 个有效一步对，完成 1 次 covariance 更新，验证集有 9 个一步对。

两个入口的 v3 存档均通过独立评估及无窗口播放。评估各用 2 个环境、4 步预算，全部 episode 均在预算处截断。播放各推进一步并显示在线后验。

在两个独立 CPU 进程中固定原始数据与拟合 RNG，完整和精简 replay 的训练及验证指标、模型参数、归一化、优化器状态和最终 RNG 均逐位相同。GPU 可用环境的完整回归通过 757 项，pytest 报告 15 条警告。CPU 仿真训练中出现过 MuJoCo 线搜索和求解器迭代达到上限的提示，各轮均完成。[完整验收记录](docs/validation/model_replay_v3.json)

## 两阶段拟合基线验收（v2）

两种入口均在 G1 MuJoCo 环境中运行两轮，使用同一段 walking motion 和 seed 2026。每轮均值网络和噪声头的参数都发生了更新。表中步数为截至该轮的累计值。完整配置与测量见[验证记录](docs/validation/two_stage_world_model.json)。

| 入口与设备 | 轮次 | 均值 / covariance 步数 | 训练 / 验证一步对数 | 验证加权 MSE | 验证过程 NLL |
|---|---:|---:|---:|---:|---:|
| `train_belief`，RTX 5090 | 1 | 10 / 1 | 33 / 11 | 4.5542 | 15521.93 |
| `train_belief`，RTX 5090 | 2 | 20 / 2 | 66 / 22 | 2.0678 | 17592.66 |
| `train_eit`，CPU 小网络 | 1 | 2 / 6 | 36 / 9 | 1.0464 | 27052.75 |
| `train_eit`，CPU 小网络 | 2 | 4 / 17 | 72 / 18 | 0.9452 | 17934.32 |

- RTX 5090 使用默认三层 2048 宽均值网络。20 次完整均值更新每次处理 2048 个 8 帧窗口，同步计时中位数为 26.7 ms。更新期间 PyTorch 已分配显存峰值为 1.24 GiB，包含当时仍存活的模拟器和模型张量。
- 从两个入口生成的轮次存档恢复后，在固定窗口和一步对上，下一次均值与 covariance 更新的损失、模型指纹和优化器状态逐位相同。两个 CLI 也分别从各自存档完成了第三轮真实采集与优化。独立的 A/B/C 两轮检查使用共同初值，比较恢复后模型与 actor/critic 的下一次更新。
- 两个入口还分别加载了修复前按默认配置在 GPU 上训练 10 轮的存档，均完成第 11 轮。`train_belief` 的均值与 covariance 累计更新 110、38 次，`train_eit` 分别更新 110、37 次。
- 闭环评估加载新存档，两个环境各走 4 步，均在评估预算处截断。原生播放器在无窗口测试中完成 reset、一步物理推进及在线后验显示状态更新。该短程测试未产生完整 episode 的成功率。
- 数值与边界回归覆盖逐步均值、四组损失与原版梯度的对照，有交叉项的高斯密度、已知右局部扰动、covariance 阶段的逐位冻结、环形覆盖、缺失后继和验证集隔离。[模型拟合测试](tests/test_estimator_two_stage_wm.py) · [轮次与恢复测试](tests/test_estimator_belief_training.py)
- 新增回归检查非有限均值梯度、Adam 二阶矩溢出、验证指标异常后的存档恢复，以及环形覆写前的分批归一化。另有测试覆盖 replay 动作与上下文校验、确定性存档版本和验收脚本的策略步数别名。[回放测试](tests/test_estimator_replay_restore.py) · [动力学存档测试](tests/test_estimator_stochastic_dynamics.py) · [验收入口测试](tests/test_estimator_eit_rng.py)
- 完整回归在 GPU 可用的环境中通过 729 项，pytest 报告 15 条警告。CPU 小网络训练记录了 1 次 MuJoCo 线搜索达到迭代上限，独立的 A/B/C 检查记录了 5 次求解器达到迭代上限。
