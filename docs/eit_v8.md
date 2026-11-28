# EIT v8 训练

[训练入口](../README_eit.md) · [配置](../train_eit_v8.toml)

v8 覆盖真实采集、replay、模型拟合和策略训练。`policy_version = "v8"` 显式选择新契约；v7 配置和网络保持可用。评估适配器暂不接受 v8 checkpoint，评估接入留到后续阶段。

在 `train_eit_v8.toml` 中设置环境、数据和输出路径，然后运行：

```bash
.venv/bin/python -m supermimic.scripts.launch_eit --config train_eit_v8.toml --dry-run
.venv/bin/python -m supermimic.scripts.launch_eit --config train_eit_v8.toml
```

直接使用 `supermimic.scripts.train_eit` 时，指定 `--policy-version v8 --group gt|mean|belief`。恢复仍使用 `--resume training.pt`；网络契约、噪声、采集参数和优化器设置从 checkpoint 恢复。

## 观测与网络

每帧 actor 输入为 `[proprio64, previous_executed_action29, tracking_error9]`，历史为 oldest-first 的 `5×102`。跟踪误差为：

- `Log(T_state^-1 T_reference)`：右局部 SE(3) 位姿误差，顺序为旋转 3、平移 3。
- `R_state.T @ v_reference_world - v_state_body`：机体系线速度误差 3。

参考保持 episode 初始固定对齐。未来 10 帧的 730 维 motion goal 仍独立进入 motion encoder；其坐标原点与朝向来自当前参考，保留 v7 的整条 rollout 批量编码。残差动作模式继续使用下一参考帧关节角作 PD 前馈。

| 方法 | 当前误差的状态来源 | FiLM | 想象 estimator |
|---|---|---|---|
| `gt` | 每帧独立扰动后的真值 | 旁路 | 直接推进动力学 |
| `mean` | 融合当前有效航向后的 posterior mean | 旁路 | 预测及本体测量更新 |
| `belief` | 同 mean | 跟踪误差协方差的 9 个 log std + 36 个相关系数 | 同 mean |

GT 的右局部旋转、平移和机体线速度标准差默认 `0.03 rad / 0.03 m / 0.20 m/s`，分别通过 `gt_rotation_std`、`gt_position_std`、`gt_velocity_std` 配置；训练期间固定。先扰动状态，再使用同一状态的位姿和机体速度计算误差。

三组 actor 权重形状相同，FiLM 输出层初始化为零。GT/mean 完全旁路 FiLM，并将该分支排除在优化器之外；共享初始化仍包含相同的 FiLM 权重。物理参数 ξ 不进入 actor 或 critic，只进入 world model。

`[training].film_hidden_dims = [128, 64]` 配置 FiLM MLP 的隐藏层：列表长度决定深度，各元素决定对应层宽度。v8 的输入固定为 45 维协方差编码，输出为两组 `actor_hidden_dims[0]` 维的 gamma/beta。直接运行训练入口时使用 `--film-hidden-dims 128 64`。三组采用同一配置以保持共享初始化形状一致，输出层仍为零初始化，学习率沿用 `policy_lr`。checkpoint 保存该结构，恢复时按保存值构建网络；早期缺少此配置字段的 checkpoint 使用原来的 `[128, 64]`。

critic 的公共输入为真值状态特征 68、privileged reference goal 733、本体/动作历史 465，总计 1266 维。mean/belief 追加 posterior 相对真值的 9 维误差及 45 维 covariance 编码，总计 1320 维，两组结构相同。均值误差使用 `Log(T_truth^-1 T_mean)` 和 `R_truth.T @ v_mean_world - v_truth_body`；covariance 编码使用 posterior 自身右局部 RPV9 边缘块。critic 的历史只取每帧前 93 维，GT critic 与 posterior 及 actor 的随机误差输入独立。

belief actor 的 covariance 通过解析 9×9 Jacobian 投影到当前跟踪误差空间，保留位姿、速度之间的交叉项。利用平方根因子的前 9×9 块计算边缘投影，不在 rollout 内重建完整 covariance、执行数值差分或嵌套自动求导。

## 航向与 replay

这一部分连接真实采集和想象训练：真实采集中的稀疏航向会改变在线 estimator 的 posterior；replay 保存选动作时实际使用的观测历史和当时的 posterior；策略训练从这些保存的起点出发，预测未来只有本体测量时的表现。这里的“真实采集”指 MuJoCo 环境产生的转移，与 world model 生成的想象转移相区分。

**航向测量的含义。** 本体观测中的重力方向可以约束倾斜角，但不能单独确定绕世界竖直轴的 yaw。角速度提供旋转变化的信息，仍需要初始航向或外部航向观测来约束世界系朝向。v8 因此在真实采集中提供间歇的绝对 yaw 测量。

设 `R_truth,t` 是当前 root 到世界系的旋转，测量定义为：

```text
yaw_truth,t = atan2(R_truth,t[1, 0], R_truth,t[0, 0])
z_heading,t = wrap(yaw_truth,t + εt)
εt ~ Normal(0, heading_std²)
wrap(a) = atan2(sin(a), cos(a))
```

默认 `heading_std = 0.05 rad`，约为 `2.86°`；滤波使用的测量方差为 `0.0025 rad²`。该测量与经过 episode 初始对齐的参考轨迹共用世界系航向基准。它描述机器人实际朝向；机器人相对参考的 yaw 偏差仍由 posterior 与参考之间的跟踪误差表达。

**一个 episode 内何时有测量。** 每个环境独立维护航向调度，在该环境 reset 时决定本 episode 的缺测状态、频率和相位，此后保持到下一次 reset。

| 配置或变量 | 默认值／分布 | 含义 |
|---|---|---|
| `heading_dropout_probability` | `0.25` | 以此概率将整个 episode 的航向频率设为零 |
| 非缺测 episode 的频率 `f` | `Uniform(0.5, 5.0) Hz` | 每个 episode 采样一次固定频率 |
| 起始相位 `φ` | `Uniform(0, 1)` | 一个周期内的随机相位，使各环境的到包时刻错开 |
| 控制周期 `dt` | `0.02 s` | 在 50 Hz 控制步上判断本步是否收到包 |

因此，`0.25` 表示约四分之一的 episode 全程没有航向包。其余 episode 按各自固定频率接收包，平均间隔为 10–100 个控制步；例如 `f=2 Hz` 时约每 25 步收到一次。部分环境 reset 时，只替换这些环境的频率和相位，其他环境继续原来的调度。

实现用累计相位是否跨过整数周期判断有效性，`t` 是从零开始的 episode 内 `time_index`：

```text
heading_valid,t = (f > 0) and
                  (floor(t × dt × f + φ) > floor((t − 1) × dt × f + φ))
```

这将连续时间频率离散到控制步；非整数步长的周期会落在相邻的整数步间隔上。episode 的第零步也使用同一规则，是否有包取决于相位，没有强制首帧测量。缺测帧的 `heading` 存为零，必须由 `heading_valid` 区分“没有测量”和“测得零航向”。

**同一帧内的处理顺序。** 对持续运行的环境，时刻 `t` 的一个控制步依次执行：

1. 使用上一执行动作 `a[t−1]`、物理参数 ξ 和 world model，将上一 posterior 预测到 `t`。
2. 融合当前 64 维带噪本体观测。
3. 若 `heading_valid[t]` 为真，再融合本帧的航向测量，得到最终的 `b[t|t]`。
4. 构造当前 102 维 actor 帧，追加到历史，计算动作 `a[t]`。
5. 在 `env.step(a[t])` 前复制当前状态、观测、动作和需要保存的 posterior；环境步进完成后，将这些数据连同转移的终止标记提交到 replay。

新 episode 的第一步从初始化 prior 开始，先融合本体观测，再按有效标记融合航向；不会从上一 episode 预测过来。三组真实采集都执行上述在线滤波并记录 posterior。mean/belief 的当前跟踪误差使用步骤 3 之后的 posterior mean；GT 的当前误差独立使用本帧加噪真值。belief 的 FiLM 还使用当前 posterior 投影后的协方差。

航向更新的创新为 `wrap(z_heading − yaw(posterior_mean))`。例如预测为 `179°`、测量为 `−179°` 时，创新是 `+2°`。平方根更新根据该创新和航向 Jacobian 修正完整 belief，再通过 Lie reset 将协方差转换到修正后均值的局部坐标。航向直接约束朝向，也可以通过已有交叉协方差影响其他状态；它本身没有提供绝对位置测量。

`heading` 和 `heading_valid` 不作为额外通道直接拼入 actor history 或 critic。对 mean/belief，它们的作用已经进入当前 posterior，进而影响跟踪误差、belief actor 的 FiLM，以及两组 critic 的 belief 特征。GT actor 使用独立加噪真值，GT critic 不读取 posterior。采集频率、相位和下一包时刻也不进入网络输入。

**Replay 保存的内容和时间含义。** 一条记录以“时刻 `t` 选动作”为基准：状态和 posterior 都属于 `t`，历史里的上一动作是 `a[t−1]`，该记录的 `action` 是接下来执行的 `a[t]`。

| 记录 | 保存内容 | 训练时的用途 |
|---|---|---|
| `actor_obs` | 本帧实际使用的带噪本体观测 64 维与上一执行动作 29 维 | 恢复原始传感／动作历史，不重新加噪 |
| `tracking_error` | 当时算出的 SE(3) 位姿误差 6 维与机体线速度误差 3 维 | 与 `actor_obs` 拼成完整 102 维帧 |
| `heading`、`heading_valid` | 本帧航向值及有效标记 | 保留测量记录；恢复已融合的 posterior 时不再次使用它更新滤波器 |
| `episode_id`、`time_index` | 轨迹身份与 episode 内控制步编号 | 确认历史连续性、reset 边界和 posterior 所属帧 |
| `motion_time`、参考对齐 | motion clip 中的参考帧及 episode 固定对齐 | 恢复与采集一致的参考坐标系，构造当前和未来参考 |
| 真值 `state`、执行 `action`、ξ、终止标记 | 真实转移及其边界 | 拟合 world model、构造想象起点与 privileged critic 信息 |
| 完整 posterior 与模型 revision | 选动作前的均值、协方差平方根及来源模型标识 | 直接恢复策略起点的 belief，保留历史测量造成的相关性 |

传感帧、跟踪误差和航向记录逐帧保存。较大的 posterior 按 `posterior_save_stride` 保存；当前 v8 配置为 `1`，即每帧都保存。若设为 `s>1`，实现按全局 replay 插入序号每隔 `s` 次保存一批环境的 posterior；策略起点从仍保留有效 posterior 的帧中选择。历史帧仍按每个控制步恢复，不会变成间隔 `s` 步的历史；world model 的真实转移采样也不因此降频。

Posterior 使用完整状态：均值以 76 个浮点数保存，其中旋转使用 3×3 矩阵；70 维局部误差的协方差写为 `P = L Lᵀ`，保存下三角平方根 `L` 的 2485 个元素，均采用 float32。belief actor 的不确定性输入仅涉及 9 维跟踪误差空间，后续滤波仍需要完整状态和交叉相关性，所以 replay 保留完整平方根，而不是仅保存 9 维边缘块。

**历史误差保持当时的值。** 设 `f[t] = concat(actor_obs[t], tracking_error[t])`，actor history 为：

```text
history[t] = [f[t−4], f[t−3], f[t−2], f[t−1], f[t]]
```

每个 `tracking_error[k]` 都使用当时可获得的状态计算。即使在 `t` 收到航向、修正了当前 posterior，`f[t−4]` 到 `f[t−1]` 也保留原值；这里没有对历史做平滑回写。GT 历史同样保留各帧自己的扰动结果，采样 replay 时不会重新抽取 GT 噪声。

重新用最新模型或当前 posterior 计算旧误差，会改变当时的估计结果；重新抽 GT 噪声，也会改变当时的观测。因此 replay 直接恢复保存的原始 102 维历史，再交给当前网络的 normalizer。actor 的归一化统计也直接读取这些保存的帧，不重跑滤波或重新投影历史误差。这里保证的是原始观测历史一致，网络权重和归一化统计仍会随训练更新。

episode 开头不足五帧时，重复完整首帧。例如 `t=0` 的历史为 `[f0, f0, f0, f0, f0]`，`t=2` 为 `[f0, f0, f0, f1, f2]`。重复包含首帧的全部 102 维，也包含同一份噪声和误差；填充历史不会产生新测量或额外滤波更新。

环形 replay 覆盖旧数据时，另一个边界缓存保留最近淘汰的 `history_length−1=4` 帧。若主缓冲区最早保留的是某 episode 的第 100 帧，采样它时所需的第 96–99 帧可从边界缓存取回，与第 100 帧组成历史。缓存同时保存 `actor_obs`、`tracking_error` 和帧身份；恢复时核对插入序号、episode 与时间，防止把另一段轨迹或已覆盖的数据接进历史。

**有效测量只融合一次。** 同一帧的航向包和 GT 扰动先被缓存。重复构建 actor 输入时，复用相同测量和完整历史；provider 根据 `episode_id/time_index` 查找已提交 posterior，不再执行本体或航向更新。若输入构造尚未成功，重试可用同一包重新计算候选 posterior，但只有成功提交的候选成为当前状态。环境已经执行、replay 提交尚未完成时，则复用待提交转移。

这避免了把一次航向包当作多次独立观测。重复融合会让滤波器对同一份信息过度确信，并可能继续拉动均值；保持包、噪声、posterior 和 replay 提交的一致性，才能让策略起点对应实际采集时的信息。

**从 replay 进入想象。** mean/belief 直接恢复起点 `t` 保存的完整 `b[t|t]` 和五帧历史，不重放过去的滤波过程，也不再次融合起点的本体或航向测量。GT 恢复自己的五帧历史及真值起点，想象分支直接推进动力学，不需要解包或推进滤波 posterior。之后每一步的顺序为：

```text
当前 history → actor → 执行动作 → world model 推进想象真值状态
    → 生成下一帧带噪本体观测
    → mean/belief：预测并融合本体观测；GT：独立扰动下一帧真值
    → 计算下一帧跟踪误差 → 追加完整 102 维帧
```

例如真实轨迹第 18 帧收到过航向，从第 20 帧开始想象时，保存的 `b[20|20]` 已包含这次测量的影响；历史中的第 16、17 帧保留测量到来前的误差，第 18–20 帧使用各自当时的更新结果。想象的第 21 帧及以后只生成本体测量，历史航向的影响通过 posterior 继续传播。即使真实采集器原本会在稍后收到另一包，想象也不会注入那一包。

因此，return、GAE 和 bootstrap 对应的是“已有起点信息，但未来没有新增航向”的 rollout。mean/belief critic 可以通过起点及传播后的 posterior 感知历史航向带来的状态修正和不确定性变化；它们不会获知未来航向日程。航向记录保留在 replay 中，供核查实际观测过程，而未来 rollout 的观测模型保持为本体测量。

**随机数、恢复和执行开销。** 真实采集的航向调度／测量噪声，与真实采集的 GT 扰动，使用两个独立的可恢复 RNG；想象中的 GT 扰动使用策略 learner 自己的 RNG，并在划分 microbatch 之前预采样。采集器每步为全批量在设备上生成候选调度随机数，但仅通过 reset mask 为新 episode 更新有效频率和相位，持续运行的环境保留原配置。真实采集边界只为有效航向行执行额外修正，想象循环内没有航向调度或航向更新，继续使用批量 motion encoder 和 CUDA Graph replay。

从 replay 采样时恢复的是一条已采集轨迹上的历史和 posterior；从 checkpoint 恢复训练时，当前实现会重新开始物理轨迹和 replay。后者延续模型、优化器及 RNG 状态，但新 episode 会重新采样航向调度，不继续旧 episode 的到包时钟。两种恢复操作的语义不同。

对应实现可查看 [采集调度](../src/supermimic/collection_observations.py)、[采集与转移提交](../src/supermimic/runners/supermimic_runner.py)、[posterior 事务](../src/supermimic/belief_provider.py)、[航向更新](../src/supermimic/estimator.py)、[posterior 与历史缓存](../src/supermimic/posterior_replay.py) 和 [想象时序](../src/supermimic/imagined_rollout.py)。

## 训练、恢复与执行

沿用按 episode 划分的世界模型训练/验证：均值头使用多步预测损失，过程噪声头使用一步局部残差高斯 NLL。策略更新固定 world model 参数，保留动作经过动力学、滤波均值、平方根 covariance 和 FiLM 的梯度。

保留 v7 的批量 motion encoder、编译热点、microbatch 梯度累积及完整 actor–critic CUDA Graph replay。GT rollout 噪声在 microbatch 划分之前一次采样，作为 `PreparedRollout` 的张量输入复制到图的固定存储；图内不抽随机数或进行主机校验。数值状态在设备端累计，更新结束时统一检查。

checkpoint 契约分别为 `eit_training_v8`、`eit_tracking_history_v8`、`eit_tracking_critic_v8`、`eit_initialization_v8`。actor checkpoint 显式保存观测布局和固定噪声；完整训练 checkpoint 另存航向配置、采集 heading/GT RNG 和策略 GT RNG。恢复重新开始物理轨迹与 replay，延续优化器、调度和随机数流；不恢复旧 episode 的航向时钟。

## 验证

```bash
WANDB_MODE=disabled .venv/bin/python -m pytest \
  tests/test_estimator_v8.py tests/test_estimator_heading.py \
  tests/test_estimator_eit_graph.py tests/test_estimator_eit_compile.py \
  tests/test_estimator_v8_cuda.py -q
```

v8 回归包含几何及解析 Jacobian、角度跨界、缺测与部分 reset、事务重试、环形历史逐项一致、三组模型/策略更新及恢复、输入权限、滤波/FiLM 梯度、microbatch 和批量编码一致性。CPU profiler 检查想象循环及 backward 不含标量回读或动态 `nonzero`；AOT 编译检查新增几何的完整图和梯度。CUDA 可用时执行三组 eager/Inductor 与完整 CUDA Graph 更新的对照；无 CUDA 时这些测试显式跳过。

CUDA 回归将两类数值对照分开：固定噪声的 8 步 rollout 比较 eager/Inductor 的状态、返回值和梯度；完整更新比较相同编译设置下的 capture/replay 与图外执行。后者使用非零 FiLM 输出权重，显式检查 covariance 到动作的路径；真实采集、模型/策略更新及保存恢复测试保留默认零 FiLM 初始化，并覆盖三组的 K=1 和 residual actor。GT 扰动通过设备端标量乘法构造，避免在 capture 内从 Python 列表创建 GPU 常量。

2026-11-28 在 RTX 5090、PyTorch 2.14.0+cu130 上验证：12 项 v8 CUDA 回归全部通过；相关 CPU 回归 397 项通过、5 项跳过，launcher 40 项通过。另用小网络、B=2、K=1、H=32 的 belief/residual 配置完成 9 次完整更新，capture=1、replay=6，批量 motion encoder 保持开启。此项为执行 smoke test，不代表配置中 2048 环境的大规模训练性能或收敛结果。

扩展 v7 GPU 回归仍有两项数值断言失败，均在未修改的 git HEAD 副本中复现：`test_baseline_replay_posterior_independence[prop_kp]` 的梯度逐位相等断言，以及 `test_compiled_baseline_graph_matches_eager[True-1-gt]` 的单个 FiLM 权重容差断言。未放宽这两项旧测试的阈值。
