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
