# 第四步 Belief critic 与随机想象策略训练

**实施状态：已实现。** A/B/C 使用同一版本的 critic 输入契约。文末列有输入维度和 A/B/C 轮次验收命令。

本步承接已实现的 [step1](plan_step1.md)、[step2](plan_step2.md) 和 [step3](plan_step3.md)，将固定 actor 的模型学习扩展为“真实采集 → 拟合均值 → 拟合过程 covariance → 冻结模型进行策略与 critic 更新 → 下一轮采集”。完成后，A/B/C 三组都能从共同初值启动完整 EIT 训练，并保存、恢复和执行短程闭环评估。

模型拟合共用第三步的轮次调度。在线滤波和策略起点重建使用矩阵积分与向量化 Jacobian。

**已具备的基础与需要替换的接口。**

| 当前接口 | step4 的工作 |
|---|---|
| `TwoStageWorldModelLearner`、`BeliefModelTraining` | 复用模型学习、episode 划分、归一化及恢复机制；抽取公共轮次操作，新增联合训练调度 |
| `InputStagePolicy`、`SuperMimicRunner.run()` 的 EIT guard | 显式构建新的 belief policy learner；仅在完整训练入口就绪后开放对应路径 |
| `WorldModelEnv._critic_obs()` | 当前只有真值 representation、真值目标和 ξ；补齐分组 critic 输入 |
| `WorldModelEnv.reset(episode_id, time_index)` | 复用当前模型前缀重建，新增可按 K 条轨迹展开的初始条件接口 |
| 旧 `BpttPolicyLearner` / `ActorCriticPolicyLearner` | 复用回报递推思想；新 learner 使用完整 belief reset、显式随机数和 microbatch，不直接调用旧 `reset(start_batch)` |
| `WorldModelEnv.step()` | 复用随机状态、单次测量与滤波更新；补充 clip 结束的有效步和 continuation 契约 |

**三组的信息权限。**

| 组别 | 训练 actor 的目标状态 | actor 显式 covariance 输入 | critic |
|---|---|---|---|
| A | 潜在真值 | 无，FiLM 旁路 | 真值、参考、ξ 和共同控制历史；不含 belief |
| B | 完整后验中的 R/p/v 均值 | 无，FiLM 旁路 | 共同输入 + R/p/v 后验均值 + 9×9 边缘 covariance |
| C | 完整后验中的 R/p/v 均值 | 450 维任务 covariance，驱动 FiLM | 与 B 相同的输入和网络结构 |

三组评估时 actor 都使用后验均值；仅 C 显式读取 covariance。B 的当前 actor 输出不直接依赖 covariance，但多步策略回报可以通过滤波增益、后续均值和 critic 依赖 covariance。测试需要区分这两种依赖，不能要求 B 的整个策略目标对 covariance 的梯度为零。

三组继续使用相同的模型和滤波实现、配置及共同初值，随后独立采集、独立更新。第一次采集后即开始按轮次拟合均值和过程 covariance。

**第一批：实现 RPV belief critic 和输入契约。**

新增 `critic_inputs.py` 中的纯特征 builder，以及显式版本化的 critic 配置。builder 从当前 `BeliefProvider` 取得完整 `BeliefState`，校验帧标识、context、模型版本和完整 belief 的合法性后，仅编码 R/p/v 均值及其 9×9 边缘 covariance；B/C 缺少合法完整 belief 时直接报错。provider 管理完整 belief。critic 投影保留对后验的梯度。

共同输入保留 68 维真值 representation、700 维真值参考目标和 ξ，并补充同一份 5×93 历史及距 clip 结束的时间。历史包含实际上一动作：它既决定 action-rate reward，也影响具有短历史的策略后续行为。该信息对三组一视同仁，不改变 A 排除 belief 的定位。参考目标继续基于固定 episode 对齐与潜在真值构建，不能直接复用 B/C 的 actor goal。

B/C 的后验均值采用 12 维无四元数符号歧义的 R/p/v 编码：

```text
[rot6d(R_trueᵀ R_mean), R_trueᵀ(p_mean − p_true),
 R_trueᵀ(v_mean − v_true)]
```

旋转使用 6 维 rot6d，位置差和线速度差各占 3 维，避免新增 SO(3) Log 的近 π 边界。位置、线速度采用显式固定物理尺度，默认分别为 1 m、1 m/s；rot6d 无量纲。后验角速度、关节角和关节速度的均值不进入 critic；共同特权真值和本体历史仍保留原布局。

covariance 取完整后验的 `Σ[..., :9, :9]` 边缘块，按 [δθ, δρ, δν] 排序，在后验均值的原右局部切空间编码：9 个 log std 加 36 个严格上三角 correlation，共 45 维。保留 R/p/v 之间的交叉项；不使用 Schur complement 或条件协方差。单位尺度按旋转、位置、线速度三块记录；采用既有 epsilon、对称性和合法性检查。均值使用相对真值坐标，covariance 仍在后验切空间。C actor 继续使用自己的 450 维任务 covariance 编码。

由此 A 输入为 `1234 + dim(ξ)`，B/C 为 `1291 + dim(ξ)`。共同的 state、goal 和每帧 history 使用独立的 critic 归一化；新增均值、covariance 和剩余时间使用固定尺度，首版不再增加在线统计。B/C 使用同一 `(256, 128)` MLP 结构和复制得到的初始权重；三组 critic 输出层均零初始化，第一轮 bootstrap 从零值开始。

验收包括输入维度和配置恢复、固定共同输入时 A 无需 belief 且对其合法变化不敏感、B/C 对 R/p/v 均值及 9×9 内非对角 covariance 的响应、对其余合法后验均值和 covariance 变化的直接输入隔离、B/C 初始化一致，以及机器人/参考/后验共同水平旋转平移后的特征不变。分别验证均值与 covariance 的方向导数，当前投影对被排除后验坐标的直接梯度为零；完整滤波仍允许这些坐标通过后续 R/p/v 更新影响回报。critic 的输入响应在输出层离开零初始化后测试。actor 信息隔离测试继续覆盖 B/C 对真值的独立性和 A/B 对直接 covariance 输入的独立性。

三组的 critic 输入版本均为 `eit_rpv_belief_critic_v2`，B/C 的 covariance 编码为 `rpv9_log_std_corr_v1`。联合训练 checkpoint（`eit_training_v3`）和共同初始化包（`eit_initialization_v3`）以内部保存的 critic 契约判定兼容性。三组训练、评估和播放都拒绝旧契约，即使 A 的输入维度相同。续训还需校验策略 optimizer 对应的参数契约，共同初始化须校验全部三个 critic，恢复想象快照时按保存的 critic 配置校验。

R/p/v 投影同时压缩均值与 covariance，是任务相关的价值输入近似；被排除的状态不确定性和交叉项仍可能影响未来滤波与回报。第五步将检验控制收益、长期价值预测及 bootstrap 梯度质量。B/C 继续使用相同 critic 设计，C−B 的结论限定在这一配置下。

**第二批：实现 K 条随机轨迹与明确的终止时序。**

新增按训练 episode 筛选的策略起点采样器。均匀采样保留的训练帧，返回 episode/time 键；不使用验证 episode，也不沿用会采到全部 episode 的旧 `sample_initial_states()`。策略起点可以包含片段尾帧或单帧片段。

对 N 个起点使用当前模型、完整 replay 中已抽取的片段先验和实际测量重建完整 Q₀，前缀放在 `no_grad()` 下。初始潜在状态取 replay 真值。每个起点的 K 条轨迹共享完全相同的潜在状态、Q₀、历史、参考对齐、motion time 和 ξ。恢复起点时直接使用已融合首帧测量的后验。

新增 `BeliefRolloutStart` 一类的完整初始条件对象，提供选择、复制和 `reset_prepared()`。原始记录、历史和后验在边界 detach；对象带模型版本和 replay 帧身份，旧模型版本或被覆盖的帧不能恢复。每个唯一帧只重建一次，再展开 K 份；新接口复用现有 provider 的校验与提交机制，不放宽旧普通 batch reset 的限制。

过程和测量随机数显式保存为 `(H, N, K, 70)` 与 `(H, N, K, 64)` 的标准正态样本，样本间及两类噪声独立。测量标准差只在既有传感器入口乘一次。每个后继时刻仅生成一份测量，同时进入历史和 filter；上一动作使用实际限幅后的归一化动作。采集动作探索与这两类噪声分开配置，首版策略想象不额外叠加动作探索。

明确以下时序，代码使用 `step_done_prob`、`survival` 和 `reference_continuation`，避免复用语义不清的 `c_t`：

```text
动作 aₜ → 潜在 sₜ₊₁ → 奖励 rₜ₊₁ 与单步失败概率 dₜ₊₁
         → 一次测量 yₜ₊₁ → 后验 Qₜ₊₁ → 下一帧 actor/critic 输入

survival₀ = 1
mₜ₊₁ = (1 − dₜ₊₁) × reference_continuationₜ₊₁
survivalₜ₊₁ = survivalₜ × mₜ₊₁
```

若文稿的 cₜ 表示累积 termination probability，则奖励权重对应 `1 − cₜ`。当前 tex 中 `c₀=1` 的递推采用 survival 语义；实现以这里的独立名称和递推为准，不根据文字标签倒置权重。

奖励和 soft failure 继续用动作后真值对比动作前缓存的参考，随后推进参考，与现有真实仿真评分时序一致。沿用现有解析 soft failure，不新增 termination 网络。即使 dₜ₊₁=1，本次转移奖励仍按 survivalₜ 计入；它只切断之后的奖励与 bootstrap。soft failure 不触发离散随机 reset，不对其权重 detach。

motion clip 结束采用有限任务约定：最后一个合法评分动作计入奖励，其后 `reference_continuation=0`，不跨 clip，也不在末帧无限积累奖励。距终点时间作为共同 critic 输入，与此约定同步。用实际仿真边界测试确定末帧动作的索引，不仅根据参考窗口是否重复判断。由策略 horizon 引起的截断仍保留 bootstrap；采集时间上限或轮次 reset 不作为物理失败标签混入想象目标。

microbatch 按轨迹切分，末批和 clip 尾部支持有效 mask，已结束槽位停止推进，不能丢样本或通过额外末帧滚动凑满 H。固定整批起点与随机数后，不同分块应得到相同轨迹和累计梯度。失败重试沿用现有 transition transaction，不重新抽取随机数。

本批先验收 K=1 等价、K>1 起点完全一致但后续噪声独立、展平索引正确、末批无遗漏、没有重复首帧更新，以及终止动作/clip 末帧/horizon 截断三类边界。

**第三批：实现 SVG actor 更新与 critic 回归。**

新增独立 `BeliefActorCriticPolicyLearner`。先使用 eager 普通 Adam，复用现有网络与环境；本步不启用 `torch.compile` 或 CUDA graph。每次 actor 更新固定世界模型、representation、滤波设置和全部归一化，冻结 critic 参数，但保留其对输入的导数。

对每条有效长度为 h 的轨迹优化：

```text
J = mean over N×K trajectories [
      Σ(t=0…h−1) γᵗ survivalₜ rₜ₊₁
      + γʰ survivalₕ Vψ(zₕ)
    ]
L_actor = −J
```

zₜ 为本组完整 critic 输入。clip 结束时 survivalₕ 为零；普通 horizon 截断保留 Vψ。首版只使用 motion task，既有跨 encoder BC 项在该配置下为零。回报按 N×K 平均，不按存活长度或 survival 总和重新归一化。

冻结模型只关闭参数梯度，不能用 `no_grad()` 包住想象过程。应保留以下路径：动作 → 随机动力学的均值与 L → 潜在状态 → 测量；动作/状态 → Jacobian、预测 covariance、修正后的均值与 covariance；C 的任务 covariance → FiLM；最终 critic 输入 → bootstrap；以及状态 → soft survival。窗口内部不 detach，初始条件和抽样的标准正态噪声为常量。

actor 反传完成后，以同一批旧策略生成的 detached 轨迹训练 critic。使用本次 critic 更新前的固定价值预测构造 λ-return：

```text
Gₕ = Vψ(zₕ)
Gₜ = rₜ₊₁ + γ mₜ₊₁ [(1 − λ)Vψ(zₜ₊₁) + λGₜ₊₁]
```

全部 target 停止梯度。critic 使用 `stop_gradient(survivalₜ × valid_maskₜ)` 加权 MSE，按整批权重之和归一化；actor 的 survival 保留梯度。先不引入 EMA target critic，整批 target 在任何 critic optimizer step 之前计算一次。首版沿用 `max_step_reward/(1−γ)` 的价值上界，统一作用于 actor bootstrap、构造 target 的价值预测及最终 target；配置要求 0≤γ<1，并记录 clamp 命中比例。验收方向导数时避开截断边界。

actor 对所有 microbatch 累积按 `1/(N×K)` 缩放的梯度，只执行一次 optimizer step。所有 actor 反传完成前不能修改 critic。critic 再对 detached 数据进行可配置轮数的回归，每个 epoch 按整批权重之和归一化并累积 microbatch，执行一次 optimizer step；小 batch 使用正常切片，不能沿用 `len // 4 * 4` 导致丢弃尾样本或空批次的实现。阶段切换清空旧梯度并恢复参数权限，actor 更新不修改模型/critic，critic 更新不修改 actor/模型。

验收先用标量手算例子覆盖 d=0、d=1、分数 d、λ=0/1 与 clip/horizon 边界，再做固定噪声的两步以上方向导数验证。检查 actor 共享层、B 的滤波反馈、C 的 FiLM、bootstrap 输入梯度及 survival 梯度；C 首次输出层更新后再检验其前层与 covariance 输入梯度。critic 零输出初始化时 bootstrap 输入梯度为零，非零响应在 critic 拟合后检验。另验收整批/microbatch 梯度一致、两阶段参数隔离、连续优化无旧图复用，以及固定数据上 critic loss 下降。

**第四批：接通完整交替训练与归一化生命周期。**

新增 `EITTraining` 调度与 `train_eit` 入口，复用 step3 公共操作，避免通过继承固定 actor 的 `BeliefModelTraining.__init__()` 意外保留冻结状态。轮次顺序固定为：

1. 固定本轮模型、actor 和归一化，在真实仿真采集，保存同一份测量、执行动作及参考对齐。
2. 封存完整 replay，在最早保留的帧和其后每个 episode 的首帧建立策略片段。同一片段首帧保留时复用已抽取的先验。按 episode ID 划分训练和验证，用本轮全部新训练帧更新模型归一化。
3. 用真实窗口的加权 MSE 更新均值，再用冻结均值产生的一步残差拟合 L。在验证集的全部有效一步对上计算过程 NLL，有完整验证窗口时另计算均值误差。
4. 固定更新后的模型，从本轮新采集且仍保留的训练帧重建 actor/critic 特征，集中更新其归一化。A 的 actor goal 使用真值，B/C 使用当前模型重建的后验。history 使用原测量，单帧统计不重复计入 padding。只用真实训练帧更新归一化。本轮没有新训练帧时沿用原统计。
5. 冻结模型、filter 和全部归一化，按配置交替执行 actor 更新和 critic 回归。每次 actor 更新重新生成轨迹；固定模型阶段可复用 detach 后的 Q₀，缓存必须包含模型版本、帧身份及先验身份。
6. 保存完整轮次 checkpoint，丢弃活动想象图，然后显式 reset 物理环境、历史与独立 provider，开始下一轮。继承 step3 对 episode ID 和模型版本的约束。

新入口从共同初始化包启动。包中保存世界模型、actor 共同权重与归一化、B/C critic 共同初值、C 的零输出 FiLM 初值及配置指纹；不能仅依赖 seed 保证一致。A 的 critic 输入维数不同，复制共同输入列与后续层，B/C 额外初始化 belief 输入列。各组加载后检查参数存储互不共享。

最少验收两个完整轮次：模型阶段只有均值/噪声参数更新，策略阶段只有对应 actor/critic 参数更新；下一轮真实动作来自新 actor，在线后验来自新模型；旧 provider/snapshot 被拒绝；训练和验证 episode 不混用。另外验证 `train_belief` 仍固定 actor，旧确定性训练分派保持原行为。

**第五批：完整 checkpoint 与短程闭环评估。**

轮次 checkpoint 保存模型 bundle、actor、critic 和归一化。均值 Muon/Adam、covariance Adam、actor 与 critic 的优化器分别保存，同时记录步数、实际 replay 模式、组别和输入编码、策略配置、滤波与先验配置、motion 指纹和 RNG。续训时检查 replay 模式一致。固定原始输入和噪声可检查下一次更新。

保存采用原子替换，恢复前在副本上验证模型、优化器、配置和 RNG。恢复从新的物理轨迹与空 replay 开始。

新增最小 EIT 评估入口：使用 `EITInputConfig(mode="eval")`，关闭动作探索，保留测量噪声和在线滤波；三组都用后验均值，只有 C 启用 FiLM。复用 actor checkpoint 的 `for_evaluation=True` 校验，在独立进程、环境与 provider 中运行，不更新统计、不写训练 replay、不改变训练 RNG。记录真实回报、episode 长度、结束原因、跟踪误差和后验诊断，区分失败与正常 clip 完成。

本批完成 CPU 的保存恢复与 A/B/C 真实闭环 smoke，再用可用 GPU 验收 float32。完整 ONNX/viewer 适配和多 seed 大规模消融放在后续评估阶段。

**第六批：规模验收和默认配置。**

下表给出训练入口的默认配置。

| 配置 | 起始值 |
|---|---|
| 模型学习 | 均值每轮 10 批，每批 2048 个 8 帧窗口。covariance 每批 256 对，每轮遍历一次 |
| 策略起点 N / 每起点 K | 4 / 2，总计 8 条轨迹 |
| 策略 horizon / microbatch | H=4 / 2 条轨迹；数值检查从 H=2 开始 |
| 每轮 actor 更新 | 1 次，每次累积完整 N×K 后更新 |
| critic 回归 | 同批 detached 轨迹，先用 2 个 epoch，保留尾批 |
| γ / λ | 0.99 / 0.95 |
| actor / critic 学习率 | 各 2e-4，梯度范数上限 1 |
| 网络 | actor 沿用 step1；critic hidden=(256,128)；模型沿用 step3 |
| 真实采集 | 8 环境、每轮 64 步、buffer 128，初版 uniform reset；动作探索沿用显式配置 |
| 执行方式 | eager，固定归一化，不使用 CUDA graph |

基准分别记录前缀重建、随机 rollout 前向、actor 反向、critic 回归和完整轮次耗时与峰值内存。扩大 N/K/H/microbatch 时固定逻辑样本与噪声比较结果；覆盖 H=2/4/8、K=1/2/4、最后不足整批的情况，再决定是否扩大到更长 horizon。测量必须包含默认宽度网络、完整 70 维 covariance、K 条轨迹和 critic bootstrap。

日志至少包含 raw reward、survival 加权回报、bootstrap 占比、最终 survival、有效步数、critic loss/解释方差、各阶段梯度范数、动作限幅率、FiLM 的 γ/β 尺度、后验误差与 covariance/jitter，以及模型/策略版本。小样本 target 方差为零时，解释方差标为不可用。反复更新后检查 provider、history、actor 临时 aux loss 和日志都不持有旧图，显存不随更新次数持续增长。

**修改位置与验证交付。**

| 位置 | 计划修改 |
|---|---|
| 新增 `src/supermimic/critic_inputs.py`，扩展 `models.py` | 分组 critic builder、12 维 R/p/v 均值、45 维边缘 covariance 编码、critic 归一化和共同初始化 |
| `replay_buffer.py`、`envs/world_model_env.py`、`envs/multi_task_world_model_env.py` | 训练起点筛选、完整起点对象、K 展开、microbatch reset、完整 critic 输入和 clip 边界 |
| 新增 `learners/belief_policy_learner.py` | 随机 SVG actor loss、λ-return critic、梯度累积和阶段冻结 |
| `runners/belief_training.py`、`runners/supermimic_runner.py`，新增 `runners/eit_training.py` | 抽取共用轮次操作，接通新 learner、输入归一化、共同初始化和联合 checkpoint |
| 新增 `scripts/train_eit.py`、`scripts/check_eit_training.py` | 训练 CLI、A/B/C 两轮真实采集与更新、恢复及 eval-mode smoke |
| 新增 critic/policy/EIT 测试，扩展现有 replay/filter/integration 测试 | 信息权限、随机时序、梯度、终止与恢复契约；保留旧入口回归 |
| `docs/validation/`、README | 保存 CPU/GPU JSON、实际源码指纹、计时配置和可复现命令 |

按上述六批分别形成可审阅的实现提交，每批先跑相关测试。最终运行受影响模块和既有回归、`check_estimator` 与两阶段拟合测试，再执行新的 EIT smoke 与默认宽度基准。

完成标准是：A/B/C 按既定信息权限完成至少两轮“采集—模型学习—策略/critic 学习”，多步 SVG 梯度经固定噪声数值验证，三阶段参数与归一化权限正确，clip/termination/bootstrap 时序一致，保存恢复和真实 eval-mode 行为通过验证。短训练验收证明流程与计算图可用；控制收益和三组消融结论由后续多 seed 实验给出。


## 实施记录

六批功能已完成。实施基线为 `0e409c5`；critic、起点/clip 边界、策略 learner
分别提交，训练调度与完整 checkpoint 因共享轮次状态合并提交，最后提交验收脚本和记录。

| 批次 | 实际实现及提交 |
|---|---|
| 1 | `CriticInputConfig`、特征 builder 和 `BeliefCritic` 定义各组输入和版本校验，RPV 契约见文末 |
| 2 | `458a8c1`：`BeliefRolloutStart`、`prepare_policy_starts`、`sample_policy_starts`、`reset_prepared` 与图释放；复用现有 replay 原始帧接口，无需改变其存储 schema |
| 3 | `ed1e885`：`BeliefActorCriticPolicyLearner`、SVG actor、survival 加权 λ-return、梯度累积、独立采样 RNG 与 optimizer 恢复 |
| 4–5 | `20b4b74`：`EITTraining`、`train_eit`、`eval_eit`，以及共同初始化、轮次归一化和完整 checkpoint |
| 6 | `check_eit_training`、`check_eit_scale`  |

分组 critic 在环境的 `eit_critic` 路径启用，step1/2 注入接口与 `train_belief`
保持原行为。策略 learner 将起点按有效 horizon 分组，末批可变长，clip 结束后不推进槽位。
prepared reset 和每个 imagined step 在提交前校验完整 critic 输入；已结束 clip 的 snapshot
记录结束状态和 critic 编码配置，恢复后不能再多执行一个动作。

共同初始化位于输出目录的父目录，默认文件名为 `eit_initialization_v3_eit_rpv_belief_critic_v2_seed<seed>.pt`。同一训练 seed 的三组共享这一文件，也可以用 `--initialization` 显式指定当前契约的共同初始化包。
发布采用原子且不覆盖已有文件的方式；加载前校验全部三组组件并计算指纹。
模型使用本轮全部新训练帧更新归一化。actor 与 critic 使用其中仍可重建起点的帧。
均值、covariance 与两个策略 optimizer 都先在副本上校验，再恢复活动状态。

策略日志对 `num_batches` 次更新取平均，因此 `Policy/valid_steps` 表示每次更新的平均
有效步数；默认每轮更新一次时就是该轮策略训练的有效步数。模型的 rollout MSE 对抽取的窗口平均，过程 NLL 对实际的一步转移平均。
完整 ONNX/viewer 导出和多 seed 控制性能消融仍属于后续评估工作。

## RPV critic 输入修订（2026-11-20）

B/C critic 的后验输入已统一缩减到 R/p/v。均值保留相对真值的 rot6d、位置差和线速度差；covariance 使用后验右局部切空间的 9×9 边缘块，保留块内交叉项。完整 `BeliefState` 仍在 provider 边界校验，滤波和 replay 策略起点继续使用完整 70 维误差状态；actor 输入、FiLM 和共同特权输入保持原定义。

| 组别 | 后验均值编码 | 边缘 covariance 编码 | critic 总输入 |
|---|---:|---:|---:|
| A | 0 | 0 | 1234 + context |
| B/C | 12（rot6d 6 + 位置 3 + 线速度 3） | 45（9×9） | 1291 + context |

均值的固定尺度按位置、线速度两块设置，covariance 的固定尺度按旋转、位置、线速度三块设置。

起点、critic 特征与策略梯度的检查位于 `tests/test_estimator_policy_starts.py`、`tests/test_estimator_critic.py` 与 `tests/test_estimator_belief_policy.py`。本轮训练、恢复、评估和播放结果见[实测验收](plan_step3.md#实测验收)。

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m supermimic.scripts.check_eit_training --device cpu
```
