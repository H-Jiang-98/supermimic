# 第三步 均值模型与噪声头联合训练

本步承接 [第一步](plan_step1.md) 和 [第二步](plan_step2.md)，实现从零初始化的均值模型与噪声头通过 posterior NLL 共同学习，完成序列采样、模型更新、后验刷新和保存恢复。沿用审阅后的五批实施安排；第五批先将 GenCoord 动力学改为直接旋转矩阵积分，再接入向量化 Jacobian，最后进行规模、诊断和恢复训练验收。

三组沿用相同的模型与滤波实现、配置和共同初值，随后独立采集与训练。A critic 不含 belief；B/C critic 都包含完整后验均值与 covariance，使用相同输入和结构。5×93 历史、700 维目标、C 的 450 维任务 covariance FiLM，以及完整 70 维滤波误差状态保持既定语义。完整策略优化与 belief critic 接入属于后续阶段。

**第一批：修复遗留问题，建立模型训练接口。**

遗留修正已提交为 `b039e01`：核心 Jacobian 拒绝直接在 inference mode 下运行，评估使用 `no_grad()`；世界模型和 actor checkpoint 在写入前校验全部组件。新增 25 项回归，已跟踪测试合计 458 passed，CPU MuJoCo smoke 通过，详见第二步的审阅修正记录。

新增独立 `BeliefWorldModelLearner`，复用 `predict`、`correct` 和 `posterior_nll`，显式接收测量、片段先验和有效时序。现有确定性 `WorldModelLearner` 的 rollout/MSE 接口继续服务原有训练模式。验收两种 learner 的数据契约与调用分派。

**第二批：实现训练序列采样与窗口起点后验。**

扩展 `ReplayBuffer`，每轮采集结束后确定保留片段与已实现的先验；同一片段在多次 optimizer 更新间复用该先验。T 次预测更新需要 T+1 帧状态、测量与前 T 个动作。窗口从片段内部开始时，用当前模型重放前缀，只在窗口边界 detach 均值与 covariance，然后对窗口内部作可微展开。

首帧测量只融合一次，后续由 aₜ 与 yₜ₊₁ 得到 Qₜ₊₁。支持短片段、有效长度和 reset 边界。当前 replay 缺少终止动作之后、自动 reset 之前的 successor，这种转移不计分；禁止把下一 episode 首帧当作该 successor。验收 buffer 覆写、窗口内部起点、短 episode、动作偏移和有效转移计数，并与独立逐帧参考比较。

**第三批：用完整 posterior NLL 更新两个模型头。**

每个有效转移执行 `Qₜ → predict(aₜ, ξ) → correct(yₜ₊₁) → posterior_nll(sₜ₊₁)`，损失为所有有效后验 NLL 的平均值。均值模型与噪声头随机初始化，共用 optimizer，窗口内部保留经过均值、A、过程噪声、covariance 和修正步骤的梯度。首轮只优化 posterior NLL，一步预测误差与预测分布质量用于诊断。

加入有限损失、有限梯度检查和梯度裁剪。噪声头输出权重零初始化时，首步其前层梯度可以为零；验收应进一步检查输出层更新后前层获得梯度。验证多步损失与独立递推一致、两个头实际更新、选定方向通过有限差分、连续 optimizer step 无旧图复用，以及固定小数据上的 NLL 下降趋势。

**第四批：接通采集、归一化、训练与后验刷新。**

新增模型学习入口，执行“固定模型采集 → 封存训练片段 → 更新归一化 → 联合训练 → 验证与保存 → 刷新采集器 → 下一轮”。归一化在轮次边界更新，训练窗口内部固定；参数或归一化更新后，旧后验与缓存失效。首版在轮次边界显式 reset 环境、历史和 provider，生成新 episode 标识。后续支持连续物理轨迹时，再从完整在线前缀重建。

后验缓存必须带模型版本，每次 optimizer 更新后重新验证；不同版本之间只复用原始测量、动作和已实现的片段先验。验收至少两个完整采集/模型更新轮次，第二轮使用新模型初始化，采集与训练消费 replay 中同一份测量，旧缓存被拒绝。

**第五批：直接旋转矩阵积分、向量化 Jacobian 与规模验收。**

按以下依赖顺序实施；小 batch eager 路径用于前四批正确性验证，第五批以向量化路径作为正式模型学习和想象的实现目标。

1. **建立直接接受 LieState 的矩阵动力学核心。** 在 `representations.py` / `models.py` 中直接从 R、p、v、ω、qⱼ、q̇ⱼ 构造现有 68 维网络特征，保留当前网络输出 Δv_b、Δω_b、Δq̇ⱼ 的含义。滤波预测和随机采样调用该核心；四元数转换集中在 71 维全状态与既有 aux/rep 的输入输出适配边界。沿用旧权重、dt 和 substep_coeff 做等价性对照。

   设 v 为世界系线速度，ω 为机体系角速度，c 为现有 `substep_coeff`。积分定义为：

   ```text
   E       = Exp(Δt × (ω + c Δω_b))
   R_next  = R E
   p_next  = p + Δt × (v + c R Δv_b)
   v_next  = v + R Δv_b
   ω_next  = Eᵀ (ω + Δω_b)
   qⱼ_next = qⱼ + Δt × (q̇ⱼ + c Δq̇ⱼ)
   q̇ⱼ_next = q̇ⱼ + Δq̇ⱼ
   ```

   必须保留 ω 的新机体系表达、半隐式位置/关节积分，以及与预测均值相同右局部坐标下的过程 covariance。SO(3) Exp 复用已有支持零角附近高阶导数的实现。验收一般姿态、近 π 的绝对姿态、零角/小角增量与多步旋转正交性；局部 Log 的有效域继续遵循第二步定义。

2. **在纯 tensor 核心上使用 `vmap(jacrev)`。** 对单个样本求局部 ε 导数，再沿样本维 vmap，输出形状为 B×70×70，避免构造跨 batch 的大 Jacobian。残差仍为 `local_error(f(ŝ,a,ξ), f(ŝ⊕ε,a,ξ))`。名义预测仅对 ε 固定，对模型、输入状态和动作的外层依赖必须保留。

   将形状、有限值、版本和 covariance 合法性检查放在变换之外；变换内部使用固定归一化与互不耦合的样本运算。提供可配置的 batch/Jacobian 分块控制峰值内存，并保留普通 autograd 作为数值参考。模型参数固定时仍保留动作与状态梯度；`no_grad()` 的评估语义和 inference-mode 防护继续有效。

3. **验证高阶导数并接通生产入口。** 使用相同权重、状态、动作、测量和过程噪声，对比旧四元数路径、矩阵 eager 路径与矩阵向量化路径。验证 A 本身、预热后的 ∂A/∂θ、至少两步 posterior NLL 对均值/噪声参数的梯度，以及固定模型后的动作梯度；以有限差分补充参考路径对照。采集、replay 前缀重建、模型学习与随机想象必须使用一致的动力学定义。积分版本和 Jacobian 配置进入 checkpoint/恢复契约，旧版本迁移显式处理，变更后刷新活动后验与缓存。

4. **测量完整训练开销并扩大规模。** 分别记录 Jacobian、滤波前向、完整 posterior NLL 前向/反向的时间与内存，覆盖 batch、horizon、网络宽度及 chunk 配置。先用 microbatch 1–4、窗口 2–4 步和梯度累积验证，再根据矩阵向量化路径的实测结果设定正式配置。CPU float64 用于数值导数；CPU/GPU float32 用于实际运行验收。计时记录设备、dtype、线程数、预热和重复次数，CUDA 计时进行同步。

5. **完成诊断与保存恢复验收。** 验证集按 episode 或 motion clip 划分。记录 posterior NLL、Mahalanobis 项、log determinant、分块误差、covariance 尺度/覆盖率、jitter、两个头梯度范数及一步预测指标。保存完整 bundle、optimizer、步数、归一化和配置；固定下一批输入验证恢复前后更新一致。完成真实采集序列上的短训练与新动力学路径的 CPU/GPU smoke。

本步完成条件：两个随机初始化的模型头能从采集序列中共同学习；矩阵积分和向量化 Jacobian 的值与外层梯度通过参考/数值验证；跨轮次模型版本、验证集诊断和训练恢复正确。

**第五批可行性验证记录（2026-11-20）。**

可复现原型位于 [check_matrix_jacobian.py](src/supermimic/scripts/check_matrix_jacobian.py)。原型直接以 LieState 进行矩阵积分，复用当前模型权重结构、滤波修正、NLL 和李群几何，并分别运行矩阵 eager、矩阵 `vmap(jacrev)` 和现有四元数参考。生产路径的替换按第五批实施。

```bash
OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/supermimic-matrix-mpl \
  .venv/bin/python -m supermimic.scripts.check_matrix_jacobian
```

测试使用固定种子 81、PyTorch 2.14.0+cu130、单线程 CPU，均值网络 hidden=(16,)、噪声网络 hidden=(12,)，主梯度检查为 float64、batch=2、horizon=2。噪声输出权重施加小幅固定种子扰动，以检查噪声前层的非零梯度。该设置验证算法与自动微分兼容性；正式网络宽度、长时序与 GPU 的性能由第五批规模验收确定。

全部断言通过，完整数值、来源 commit 和脚本 SHA-256 保存在 [CPU 验证记录](docs/validation/matrix_jacobian_cpu.json)。

| 检查 | 结果 |
|---|---|
| 矩阵积分与现有四元数积分的状态最大绝对差 | 1.03×10⁻¹¹ |
| 向量化 A 与矩阵普通 autograd 的最大绝对差 | 2.22×10⁻¹⁶ |
| 向量化 A 与 70 个方向中心差分的最大绝对差 | 2.80×10⁻¹⁰ |
| 预热后 ∂A/∂θ 的方向导数相对差 | 1.50×10⁻¹⁰ |
| 两步 NLL 对选定均值、噪声、动作参数的方向导数最大相对差 | 6.46×10⁻⁹ |
| 固定模型后的动作方向导数相对差 | 3.65×10⁻¹⁰ |
| 零角增量处混合方向导数相对差 | 1.65×10⁻¹⁰ |
| float32 A 与矩阵普通 autograd 的最大绝对差 | 1.79×10⁻⁷ |
| float32 混合导数与矩阵普通 autograd 的最大绝对差 | 2.29×10⁻⁵ |

计时比较使用相同矩阵积分与小网络，eager 列为普通非向量化 autograd，向量化列为 `vmap(jacrev)`。单次 Jacobian 在 `no_grad()` 下预热一次后重复三次取平均；两步 NLL 对上述五组参数作前向/反向，预热一次后重复两次取平均。

| CPU 工作量 | eager | 向量化 |
|---|---|---|
| Jacobian，batch=1 | 77.33 ms | 4.78 ms |
| Jacobian，batch=2 | 156.69 ms | 5.07 ms |
| Jacobian，batch=4 | 317.97 ms | 5.19 ms |
| Jacobian，batch=8 | 649.15 ms | 5.79 ms |
| 两步 NLL 前向/反向，batch=2 | 12.85 s | 39.65 ms |

这些结果支持第五批采用“直接矩阵积分 → 向量化 Jacobian”的路线。当前原型已验证小规模的值、混合导数和性能，生产接口迁移、正式网络内存开销及 GPU 运行仍按上述计划验收。

## 实施记录（2026-11-20）

五批实现已完成。完整策略优化与 belief critic 仍按本计划留在后续阶段；本步的入口固定 actor，联合训练均值模型与过程噪声头。

| 批次 | 已落地接口与行为 |
|---|---|
| 1 | `BeliefWorldModelLearner` 与独立配置；启用 `process_noise` 的 runner 分派到 posterior-NLL learner，原确定性 learner 保持原接口 |
| 2 | `BeliefTrainingSequence`、`belief_sequence`、`sample_belief_sequences`；T+1 帧窗口和当前片段前缀，支持短窗口、覆盖后的片段起点，缺失 successor 的末动作不计分 |
| 3 | 当前模型重放前缀并在窗口边界 detach；窗口内部保留均值、A、L、修正和 covariance 梯度；按有效转移加权累积 microbatch，检查有限损失/梯度并裁剪；两头共享 Adam |
| 4 | `BeliefModelTraining` / `train_belief`：固定模型采集、封存、仅训练 episode 新帧更新归一化、训练、验证、保存、reset；每次更新重新重建前缀，不缓存旧后验 |
| 5 | `represent_lie` / `apply_prediction_lie`，以及 `Dynamics.mean_lie` / `distribution_lie` / `sample_lie`；生产滤波默认矩阵 `vmap(jacrev)`，保留 eager 参考及两类分块；CPU/GPU 数值、规模、物理采集和恢复验收通过 |

训练配置默认 `batch_size=4`、`microbatch_size=2`、`horizon=2`、每轮 10 次更新、学习率 `2e-4`、梯度范数上限 1。均值网络默认 `(2048, 2048, 2048)`，噪声网络 `(128, 64)`。均值和噪声头共同初始化；噪声输出层仍按指定初始 covariance 零权重初始化。测试确认输出层第一次更新后，噪声头前层获得非零梯度；固定小数据训练 12 步后的 NLL 低于初值，归一化在这些更新中保持不变。

验证 episode 的默认规则为 `episode_id % 5 == 0`，跨 buffer 覆盖和轮次保持固定。封存后每个片段的长度和边界只校验一次。归一化使用本轮新采集且仍保留的训练帧，检查四元数、转换后的特征和两组候选统计量后统一写入。若本轮只有新验证帧，则沿用已有统计，从保留的训练窗口继续更新。轮次日志累计有效转移数，按转移数汇总 NLL、Mahalanobis、log determinant、分块误差与方差、覆盖率、jitter 和一步预测指标，按优化器更新次数平均两头梯度范数。

`training.pt` 保存模型 bundle、固定 actor、optimizer、步数、归一化、配置及 PyTorch RNG。先在私有副本上校验全部组件，再修改活动模型；同时检查 bundle 中的滤波、先验和测量设置与 runner 配置一致。恢复后清空 replay、reset 物理环境并重建 provider，保证固定下一批原始输入时继续更新一致。它不保证复现恢复前的 MuJoCo/Warp 物理轨迹。CUDA 恢复显式把非 capturable Adam 的标量 step 时钟放回 CPU，并拒绝错误 dtype、非标量、非法布局和非有限 optimizer 状态。

滤波器要求原生矩阵动力学接口 `mean_lie` 和 `distribution_lie`。两种 Jacobian 后端都在矩阵核心上计算。每次旋转矩阵积分后执行一次 Newton–Schulz 正交化。随机动力学权重使用 `stochastic_dynamics_v2` / `right_matrix_v2`。通过 `Dynamics.migrate_checkpoint` 或 `WorldModel.migrate_bundle` 可校验并迁移第二步模型和 `right_matrix_v1` 的权重。第二步 bundle 中缺失的 Jacobian 设置取自目标模型，旧矩阵 bundle 则沿用保存的设置。迁移后重建活动后验与想象快照。

训练 checkpoint `belief_model_training_v2` 记录 motion 内容指纹、CPU RNG 和训练使用的 CUDA 设备 RNG。恢复到已有目录时核对 checkpoint 和最后一轮日志，也可从 checkpoint 写入新目录。旧版训练 checkpoint 缺少 motion 指纹，只支持提取模型权重后显式迁移。

实际训练命令见 [README 第三步](README.md#estimator-第三步均值与噪声头联合学习)。例如：

```bash
OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/supermimic-mpl \
  .venv/bin/python -m supermimic.scripts.train_belief \
  --motion-file /path/to/motion.npz --device cuda:0 --group B \
  --output logs/belief-b --num-envs 8 --rounds 10
```

同一配置加入 `--resume logs/belief-b/training.pt` 继续新增轮次。`metrics.jsonl` 和 `training.pt` 保存在本地输出目录。

**审阅修正后的验收。** 当前工作区完整 `pytest -q` 共 **571 passed、2 skipped**，包含本地测试。覆盖旧接口拒绝、长序列旋转、权重显式迁移、归一化原子更新、片段索引失效、轮次指标、跨设备数恢复、motion 身份校验和输出目录保护。

本轮另修复 provider 覆盖调用方梯度模式的问题：`prepare()` 与 `prepare_restore()` 在退出 inference mode 前记录外层梯度设置，评估时提交无图的普通张量，局部 Jacobian 继续正常计算。新增 6 项测试覆盖 `no_grad()`、`inference_mode()`、部分 reset、恢复后的梯度权限、评估后恢复动作梯度及 8 步想象中的后验图释放；修复前其中 5 项失败，修复后全部通过。provider/integration 两组共 63 项通过。本轮 CPU 的 `check_estimator` 和 `check_belief_training` 均重跑通过；CUDA 不可用，下面的 GPU 数值与规模结果沿用此前验证记录。

| 验收项 | 结果与记录 |
|---|---|
| CPU float64 生产路径 | A 对 eager 最大差 2.22×10⁻¹⁶；70 方向有限差分最大差 3.80×10⁻¹⁰；两步 NLL 参数/动作方向导数相对差不超过 8.56×10⁻⁹；[CPU 记录](docs/validation/belief_scale_cpu.json) |
| CUDA float32 生产路径 | RTX 5090，A 对 eager 最大差 1.19×10⁻⁷，混合导数最大差 3.05×10⁻⁵，两步 NLL 梯度最大相对 L2 差 2.29×10⁻⁶。[CUDA 记录](docs/validation/belief_scale_cuda.json) |
| float32 连续旋转 | 2.3204725 rad/s 匀速旋转，dt = 0.02 秒，执行 1,000 次完整 predict/correct。CPU/GPU 的 RᵀR 对单位矩阵最大绝对误差分别为 1.16×10⁻⁷ 和 1.29×10⁻⁷，末帧旋转对解析解的最大绝对误差分别为 1.86×10⁻⁶ 和 4.18×10⁻⁶。 |
| 规模 | CPU/GPU 共 16 个组合，覆盖网络 32²、256²、2048³、batch 1/2/4、horizon 2/4、两类分块；完整 NLL 前后向均有限 |
| 默认宽度开销 | 2048³、batch=4、horizon=4，CPU 约 0.73 秒，RTX 5090 约 0.195 秒，峰值分配显存约 273 MiB。单线程、预热 1 次、重复 3 次，CUDA 同步计时 |
| 两轮真实 MuJoCo 学习 | CPU/GPU float32，2 环境，每轮 8 步采集、2 次联合更新；两头变化、actor 与其归一化不变、下一轮使用新模型；[CPU](docs/validation/belief_training_cpu.json)、[CUDA](docs/validation/belief_training_cuda.json) |
| 保存恢复 | CPU/GPU 固定下一批输入的 loss、参数及 Adam 状态逐位一致；异常模型/actor/optimizer/配置恢复不修改活动状态或 RNG |
| 随机想象 | 新矩阵路径上 CPU/GPU 均通过原 `check_estimator`：8 步真实采集、2 步随机想象、完整后验前缀对照、actor 梯度有限、快照恢复成功 |
| 命令行入口 | RTX 5090 上使用 `walk1_subject1.npz`、B 组、默认网络宽度和 8 环境启动一轮，再在新进程从磁盘恢复第二轮。累计 1,024 个物理帧、20 次更新，两轮分别计入 79 和 78 个训练转移。 |
| CPU checkpoint 可移植性 | 在可见 RTX 5090 的主机上以 CPU 保存，用 `CUDA_VISIBLE_DEVICES=''` 启动 CPU 进程并完成第三轮。更换为 `dance1_subject1.npz` 时能检测到 motion 内容变化。 |

规模测量隔离 Jacobian、滤波前向、NLL 前向和 NLL 前后向，未包含采集、前缀重建、optimizer、完整 replay 和整轮训练内存。因此默认先采用 microbatch=2、horizon=2。当前测量中，分块减少单次 Jacobian 临时显存，但完整反向的图保留更大、耗时也增加；正式默认不分块，进一步扩大规模时再测量。

另用默认网络在 RTX 5090 上测量梯度累积，覆盖 horizon 4、8、16、32，每个配置预热两次后计时五次。batch 4、microbatch 2 的四种窗口均通过。batch 512 整批反向时，horizon 4 的峰值分配约为 19.6 GiB，horizon 8、16、32 均超出显存。改用 microbatch 64 后四种窗口均通过，horizon 32 每次完整前后向平均耗时 13.07 秒，峰值分配 18.60 GiB。逐次耗时和梯度检查见 [batch 4 / microbatch 2](docs/validation/belief_horizon_batch4_cuda.json)、[batch 512 / microbatch 512](docs/validation/belief_horizon_batch512_cuda.json) 和 [batch 512 / microbatch 64](docs/validation/belief_horizon_batch512_micro64_cuda.json)。

复现物理训练与数值规模验收：

```bash
OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/supermimic-mpl \
  .venv/bin/python -m supermimic.scripts.check_belief_training --device cpu
OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/supermimic-mpl \
  .venv/bin/python -m supermimic.scripts.check_belief_training --device cuda:0
OMP_NUM_THREADS=1 .venv/bin/python -m supermimic.scripts.check_belief_scale --help
```

JSON 保存当前来源文件的 SHA-256；`source_commit` 是实施前的基线，未提交源码由文件指纹标识。真实采集 smoke 只验证短序列学习流程，尚未证明长时训练收敛或闭环控制效果。

## Git 提交拆分与验收

第一、二步主体已在历史提交中。本轮从 `ac611f1` 起，按以下顺序拆分实现。每项提交前，将暂存内容导出到独立目录验证，避免后续未提交代码掩盖依赖；最后对完整候选树运行回归。

| 顺序 | 提交信息 | 内容与边界 |
|---|---|---|
| 1 | `fix: respect caller grad mode in belief providers` | `belief_provider.py`、provider/integration 两份测试的新增回归，以及 README 的 provider 梯度模式说明。 |
| 2 | `fix: refresh kinematics before estimator rewards` | mjlab kinematics patch、`tasks/env_cfgs.py`、runner 的 `forward_before_rl_signals` 配置校验和奖励测试所需的 `record_episode_stats` 开关、`test_estimator_reward_timing.py`，以及 README 的补丁安装和奖励时序说明。 |
| 3 | `feat: vectorize Lie dynamics Jacobians with matrix integration` | `representations.py`、`models.py`、`estimator.py`、`check_matrix_jacobian.py`，及 matrix dynamics、belief filter、stochastic dynamics 测试。包括矩阵积分、正交化、Jacobian 分块和 dynamics 权重迁移。 |
| 4 | `feat: learn stochastic dynamics from replay posterior NLL` | `replay_buffer.py`、`learners/belief_wm_learner.py`、`test_estimator_belief_learner.py`。包含 T+1 窗口、当前模型前缀重放、有效转移加权、两头联合优化和 optimizer 恢复。 |
| 5 | `feat: add resumable belief model training loop` | `runners/belief_training.py`、runner 剩余改动（含 learner 分派和 bundle 迁移）、`train_belief.py`、`train.py`、`check_belief_training.py`，以及 belief training、runner 测试。依赖第 2–4 个提交。 |
| 6 | `test: add belief scale benchmarks and validation records` | `check_belief_scale.py`、`benchmark_belief_horizon.py`、七份新增 validation JSON，以及 README 剩余第三步说明和本计划的实施/验收记录。保留历史 JSON 的原始来源指纹。 |

`supermimic_runner.py` 和 README 按功能块暂存，其他文件按表归组。奖励时序测试依赖 `record_episode_stats`，因此将该开关及其配套改动一并放入第 2 项，保证中间版本可独立验证。

第 1–5 项独立候选树分别通过 63、36、54、47、53 项测试；第 3、4 项各跳过 1 项 CUDA 测试。各项 Python 文件的 Ruff lint/format 均通过。第 2 项的 mjlab 补丁在临时副本中反向撤销后重新应用，文件逐字节一致；补丁保留标准 unified diff 空白上下文行。

提交期间再次运行完整工作区回归，结果为 **571 passed、2 skipped**；`check_estimator --device cpu` 和 `check_belief_training --device cpu` 均通过。CUDA 当前不可用，因此七份历史 validation JSON 保留原始内容与来源指纹，不标记为本轮重测。
