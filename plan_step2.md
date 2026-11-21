**第二步实施计划：随机动力学、可微李群滤波与接口集成**

本计划承接已完成的 [plan_step1.md](plan_step1.md)，将第一步使用的合成或外部注入后验替换为可传播、可更新、可微分的完整 belief。分为三个可独立验收的实现阶段：完整 belief 与几何工具 → 随机动力学与滤波核心 → provider 集成与短序列梯度验证。文末列有几何、滤波和真实采集的验收命令。

**当前实现状态。** 三个阶段的实现已接通：可用真实滤波器运行短序列采集，按保留片段以当前模型重建 replay 后验，并在小规模随机想象序列中验证模型、滤波和 actor 的计算图。第一步主体提交为 `c22a8af`；此后的奖励修正、几何和滤波核心分别独立提交，本记录与 provider 集成同批提交。完整模型训练调度、belief critic 和策略优化循环仍由后续阶段完成。

**第一步审阅结论与实施起点。** 第一阶段的输入布局、任务误差 covariance、FiLM 和采集接口作为本步基础。相关接口检查覆盖 replay 采样、想象入口和动作梯度。

第一步 legacy 外部 provider 的内部更新及调用前的测量噪声采样位于环境回滚边界之外，该接口仍保留此限制。第二步的 `BeliefProvider` 采用纯函数滤波核心及候选状态提交机制，使失败重试具有明确时序。

**已确认的实验与训练约定。**

| 组别 | actor 训练时的目标状态 | actor 测试时的目标状态 | actor 显式 covariance | 后续 critic 输入约定 |
|---|---|---|---|---|
| A | 潜在真值 R、p、v | 后验均值 | 无，旁路 FiLM | 真值、参考和 ξ，不含 belief |
| B | 后验均值 | 后验均值 | 无，旁路 FiLM | 真值、R/p/v 后验均值与 9×9 边缘 covariance、参考和 ξ |
| C | 后验均值 | 后验均值 | Σ_Rpv 经任务误差映射和 FiLM | 与 B 相同的信息和结构，权重独立 |

三组共用实现与配置，从相同的共同初始权重和归一化状态建立独立实例，随后独立采集、更新。B−A 表示估计状态训练与 belief critic 的整体变化；C−B 表示相同 critic 设计下任务 covariance 与 FiLM 路径的增量效果，包括其额外参数和目标相关信息路径。在线数据与模型的后续分化属于各组训练机制的整体效果。

每轮采集后先按原版 rollout 拟合均值，再冻结均值与归一化统计，用全部有效的一步转移拟合过程 covariance。策略更新阶段冻结完整模型，保留预测对动作的导数。[模型训练](plan_step3.md)

actor 使用第一步的 `eit_rpv_v1`、5×93 历史和 700 维目标，每帧采样一次高斯测量，每个 episode 固定参考对齐。A 训练时用真值构造目标，A 评估及 B/C 训练与评估时使用 R/p/v 后验均值。仅 C actor 将当前 9×9 边缘 covariance 投影到十个参考帧，按 `log_std_corr_v1` 编码为 450 维 FiLM 输入。B/C critic 直接将同一边缘块编码为 45 维，配合 12 维 R/p/v 后验均值编码。滤波与策略起点使用完整 belief。详见 [第四步输入契约](plan_step4.md#rpv-critic-输入修订2026-11-20)。

**提交一：完整 belief 状态与右扰动几何。**

建议提交信息：`feat: add full belief state and right-local Lie geometry`。

1. 定义完整状态和版本化布局。

   G1 的关节数 D=29。状态属于 SE₂(3) × ℝ⁶¹，包含 R、p、v、ω_B、q、q̇。与当前 GenCoord 的完整状态交换时，使用已有 71 维布局：

   ```text
   [p_world(3), quat(4), q(29), v_world(3), omega_body(3), qdot(29)]
   ```

   内部几何对象可以使用旋转矩阵；明确区分该存储形式与 70 维切空间，不能直接对 71 维向量相减构造滤波误差。

   ```text
   δ = [δθ(3), δρ(3), δν(3), δω_B(3), δq(29), δqdot(29)]
   ```

   `BeliefState` 保存完整均值与 70×70 covariance，支持批处理、设备和 dtype 转换、clone、显式 detach 及按环境选择。给状态布局和扰动约定分配版本，保存时一并记录。ξ 仍是模型的已知条件输入，本步不将其加入待估计状态。

2. 实现统一的右扰动、局部残差及状态转换。

   建议新增 `src/supermimic/utils/lie_state.py`，提供 SO(3)、SE₂(3) 的 Exp/Log、乘法、逆及完整状态的 `boxplus`、`local_error`。采用右乘约定：

   ```text
   s ⊞ δ = s · Exp(δ)
   local_error(s_hat, s) = Log(inv(s_hat) · s)
   ```

   对 SE₂(3) 部分，R′=R Exp(δθ)，p′=p+R J_l(δθ)δρ，v′=v+R J_l(δθ)δν；欧氏部分逐项相加。有限扰动时保留 J_l 的耦合，不能将旋转、位置、速度完全独立相加。提供与现有 `aux_state`、`rep_state` 和完整状态的显式转换。

   小角度分支应支持有限且正确的一阶、二阶梯度。对旋转对数接近 π 的分支明确有效域和数值行为；滤波局部线性化的主要验收域为零附近与一般非退化姿态。

3. 定义 posterior correction 后的切空间重置工具。

   注入修正 δ 后，covariance 必须转换到新均值的右局部坐标。提供下式所需的 reset Jacobian，并覆盖 SE₂(3) 交叉块：

   ```text
   ψ(e) = Log(Exp(−δ) · Exp(δ + e))
   G = ∂ψ(e)/∂e，取 e=0
   ```

   欧氏块为单位矩阵。当前实现用 18×18 增广矩阵指数计算 SE₂(3) 的右 Jacobian，保留外层导数；验收使用独立的有限差分与矩阵指数参考。

4. 保留第一步的 actor 输入对象作为完整 belief 的投影。

   `PosteriorInput` 继续只提供 RPV 均值和 9×9 covariance。新增 `to_actor_input` 转换，按完整 covariance 的前 9 个切空间维度提取边缘块，保留旋转、位置、速度的交叉项。完整 belief 的所有权属于滤波模块。

| 提交一验收项目 | 通过标准 |
|---|---|
| 状态往返转换 | 完整状态、GenCoord 表示与 belief 均值转换一致；世界线速度和机身角速度单位、坐标一致 |
| 扰动与残差 | 小扰动下 local_error(s, s ⊞ δ)≈δ，覆盖一般姿态及非零位置、速度 |
| 群运算与坐标变换 | 乘法、逆及共同水平旋转和平移后的右局部误差一致 |
| Jacobian | Exp/Log、boxplus、局部残差和 reset Jacobian 通过 float64 数值检查，覆盖旋转与平移/速度交叉项 |
| 高阶梯度 | 零旋转及小角度邻域的一阶、二阶梯度有限；数值参考与自动微分相符 |
| actor 投影 | 完整 belief 投影与第一步直接注入相同 RPV 后验时的目标、covariance 编码一致 |
| 独立性 | 几何与布局测试无需 MuJoCo、完整 runner 或模型训练即可运行 |

**提交二：随机动力学和纯函数滤波核心。**

建议提交信息：`feat: add stochastic dynamics and differentiable belief filtering`。

1. 在 `Dynamics` 中增加可选的过程 covariance 头。

   复用现有均值预测及 GenCoord 积分规则。新增分布预测接口，返回下一步均值和 70×70 下三角因子 L；新增采样接口，显式接收标准正态 ε，并按右扰动生成样本：

   ```text
   ΔΣ(s, a, ξ) = L(s, a, ξ) L(s, a, ξ)ᵀ
   s_next = mean_next ⊞ (L ε)，ε ∼ N(0, I₇₀)
   ```

   完整下三角输出为 70×71/2=2485 维。对角线采用正值参数化及显式数值下界；初始标准差按切空间各物理量配置并保存，初始交叉项设为零或明确的小值。ΔΣ 表示一个控制步的过程 covariance，避免调用方重复乘 dt。

   每行的非对角系数乘以该行对角尺度，得到物理切空间中的 L。均值和噪声头共同保存在模型中。均值的 Muon/Adam 只持有均值网络参数，噪声头由独立的 Adam 更新。

2. 新增 `src/supermimic/estimator.py`，实现无内部状态突变的滤波核心。

   建议核心接口为 `initialize(mean, covariance, context)`、`predict(belief, action, context)`、`correct(predicted_belief, measurement)` 和 `to_actor_input(belief)`。初始化表示先验创建；首帧观测是否已被融合由调用协议明确标记。函数返回新的 belief 及所需诊断，便于单测、批处理、计算图组合与 provider 的候选状态提交。

3. 实现右局部 covariance 传播。

   ```text
   mean_next = f_mean(mean, action, ξ)
   e_next(ε) = local_error(mean_next, f_mean(mean ⊞ ε, action, ξ))
   A = ∂e_next(ε)/∂ε，取 ε=0
   Σ_pred = A Σ Aᵀ + L Lᵀ
   ```

   A 为 70×70，噪声 covariance 位于预测均值的右局部坐标。计算局部 ε 导数时，名义预测、动作和 ξ 对 ε 固定；策略梯度仍需要它们对输入状态和动作的依赖，不能用全局 `detach` 破坏这些路径。先实现 eager 正确性，记录 batch 与 Jacobian 的实际开销，再决定后续性能优化。

   矩阵积分直接使用 LieState。默认通过 `vmap(jacrev)` 逐样本计算 Jacobian，保留 eager 参考实现、分块选项和预热后的混合导数测试。

4. 实现与第一步完全一致的 64 维观测更新。

   ```text
   h(s) = [Rᵀ e₃, omega_body, q − q_default, qdot]
   e₃ = (0, 0, −1)
   ```

   V 直接来自第一步 `GaussianMeasurementConfig`，沿用其已校准 bias 约定。滤波核心接收已经生成的测量，不自行加噪。H 按右扰动求导；重力块为 [Rᵀe₃]×，其余被直接测量的欧氏变量对应单位块。用独立有限差分验证解析 H。

   ```text
   innovation = measurement − h(mean_pred)
   S = H Σ_pred Hᵀ + V
   K = Σ_pred Hᵀ S⁻¹
   δ = K innovation
   mean_post = mean_pred ⊞ δ
   Σ_J = (I − KH)Σ_pred(I − KH)ᵀ + K V Kᵀ
   Σ_post = G Σ_J Gᵀ
   ```

   上式描述满秩情形。实际先按创新方差标准化 S，重力三个分量共用尺度，其余标量观测独立缩放；以 `64 × finfo(dtype).eps` 为相对秩阈值判定可观测方向。满秩采用 Cholesky 分解与线性求解，不显式构造普通逆矩阵；秩亏时在标准化坐标中使用 `torch.linalg.pinv(..., hermitian=True)`，忽略数值不可观测方向。Joseph 更新之后使用提交一的 G 重置 covariance，并消除数值级不对称。

   `FilterConfig` 默认 jitter 初值为 1e-8、倍率 10、上限 1e-4、最多重试 5 次；诊断返回有效秩、实际 jitter 和 Cholesky 尝试次数。显著非半正定或非有限输入直接失败，不通过 jitter 修复。增益求解若添加 jitter，同样加入 Joseph 更新所用的 V。

   第一阶段允许零测量标准差；三维无噪重力观测的局部秩为 2，按上述可观测子空间求解。精确标量测量允许得到半正定后验，不能为了保持正定而无条件注入噪声。

| 提交二验收项目 | 通过标准 |
|---|---|
| 均值兼容性 | 同权重、输入与归一化状态下，新分布接口的均值与原确定性预测一致 |
| 随机采样 | 固定 ε 可复现；ε=0 返回均值；小噪声样本的右局部经验 covariance 与 LLᵀ 在统计容差内相符 |
| 参数化 | L 对角为正、初始尺度有限且可配置，噪声头参与参数保存和优化器参数集合 |
| 传播 Jacobian | A 与 float64 有限差分相符，覆盖非零动作、角速度及姿态 |
| 观测更新 | H、innovation、K 和均值修正与独立参考一致；零 innovation 时均值不变；线性欧氏子问题与标准 Kalman 更新一致 |
| covariance reset | 非零修正时 G 与局部误差映射数值 Jacobian 相符，更新后 covariance 处于新均值的切空间 |
| 数值稳定性 | 默认噪声、较小方差、强相关合法 covariance、部分零测量噪声均有明确且可测试的结果；无静默 NaN/Inf |
| 独立性 | 随机模型和滤波核心可在合成数据上验收，无需完整策略训练或 MuJoCo 闭环成功 |

**提交三：有状态 provider、采集/想象集成与梯度烟雾验证。**

建议提交信息：`feat: integrate belief providers and validate short rollout gradients`。

1. 使用纯函数核心封装真实 provider。

   `BeliefProvider` 保存各环境完整 belief、时间标识、模型版本及必要 context。通过 `prepare(reset_mask=...)/commit` 显式初始化或部分重置，通过 `prepare_restore/restore` 恢复完整后验；provider 没有独立的 `reset()` 方法。runner 的 `reset_collection()` 重置物理环境并准备首帧，下一次 `step()` 融合该帧。默认 `GaussianStatePrior` 在真实状态上施加右局部小扰动，六块标准差为 `(0.01, 0.01, 0.05, 0.05, 0.005, 0.05)`；也可注入 `FixedBeliefPrior` 或自定义先验 factory。正常逐步更新只依赖测量、执行动作及约定的已知 context。

   在新 episode 的首帧创建先验并融合 o₀ 一次，不用 a₋₁ 执行虚构的物理传播。之后用 aₜ 从 Qₜ|ₜ 预测，再融合 oₜ₊₁ 得到 Qₜ₊₁|ₜ₊₁。5 帧历史的 reset 填充不额外调用滤波。支持部分环境 reset，其余环境继续原时序。

   `restore` 接收完整 Qₜ|ₜ、context、时间与版本信息。已融合的 oₜ 不重复更新，下一次接收 aₜ、oₜ₊₁。只给 `PosteriorInput` 的 RPV 视图不足以恢复完整滤波状态，应明确拒绝将其视为完整 restore。注入后验与 provider 混用时也需显式定义或恢复内部状态。

2. 将 provider 与环境推进组成可验证的提交边界。

   新接口按“计算候选 belief → 验证后验与下一帧输入 → 提交 provider 和环境状态”组织。实现可采用 prepare/commit，或对有状态适配器使用完整 snapshot/restore。将同一时间点采样的测量保留到本次推进提交或放弃，重试不能多次推进滤波或悄然更换本次测量。

   对已发生的真实物理步，保留其动作与测量，重试输入构建不再次执行环境 step。对想象环境，候选 latent state、belief、历史和参考时间使用同一提交结果。明确 legacy 外部 provider 与新增真实 provider 的协议差异。

3. 接通现有入口，隔离采集与想象状态。

   在 `supermimic_runner.py` 和 `world_model_env.py` 中通过真实 provider 生成 `PosteriorInput`，继续使用第一步共享目标与 covariance 特征 builder。真实采集和想象必须使用独立 provider 状态；同一世界模型参数可按相应阶段的更新规则使用。

   想象验证需要分别维护潜在真值 s 和 belief Q：潜在状态按随机模型采样，测量从该潜在状态生成一次，滤波在自己的后验均值上预测和更新。最小接口应允许显式传入过程 ε 与测量 η，方便固定随机数进行数值梯度检查。奖励使用潜在真值及既定参考规则，B/C actor 使用后验构造目标。

   使用完整 replay 时，每轮在各环境最早保留的帧和其后真实 reset 处分段。封存时在每个新片段首帧按配置采样完整 70 维右局部扰动，保存实际先验均值和 covariance。重建时，以该帧记录的测量与上一动作填充 actor 历史，融合首帧测量一次，再以当前模型按记录的动作和测量推进到所选帧。保留物理 episode 帧号用于定位，策略片段单独从 0 计时。已在想象中推进的帧也可通过完整环境快照恢复。

   当前入口为 `finish_collection_round()` 与 `training_prefix(episode, t, relative=...)`。想象环境 `reset(episode_id=..., time_index=...)` 或 `reset(episode_id=..., segment_time=...)` 从该前缀重建；同一起点保留已抽样先验，模型变化只重新传播。此重建不保留历史模型学习图，后验作为想象初始条件。模型拟合独立读取真实状态、动作和 context。

4. 明确自动微分与归一化边界。

   当前真实采集入口使用 `torch.inference_mode()`，而滤波传播的 A 仍需局部自动微分。适配器应在局部正确退出 inference mode，将必要输入转换为可参与求导的普通张量，计算滤波 Jacobian 后显式 detach 已提交的采集 belief，避免采集图随时间增长。

   想象路径保留从动作、模型预测、A、L、covariance 更新、目标构造到 actor 输出的计算图。A、G 对模型、动作或状态的外层导数涉及混合二阶导数，应使用支持这些导数的实现，并分别验证模型学习与模型参数固定时的策略梯度。参数固定不能通过整个预测调用的 `no_grad` 实现，否则会阻断动作梯度。

   原 runner 会在采集后更新归一化状态；真实 provider 默认通过 `freeze_estimator_normalization=True` 冻结模型及 actor 的归一化更新，第二步短序列不更新模型参数。`Dynamics.model_version` 对当前权重、buffer、过程噪声与积分配置计算内容指纹，包含归一化变化和 fused optimizer 写入。provider 的查询、restore、连续推进与 commit 检查版本；新 episode 可用当前版本重新初始化。每轮拟合后，策略起点和下一轮采集器按当前模型版本重建。

5. 保存完整配置与完成运行记录。

   checkpoint 包含均值模型、噪声头、模型归一化、噪声参数化与数值配置、完整状态及切空间版本；需要恢复在线轨迹时另行保存完整 belief、context、时序标识及模型版本。普通模型权重恢复与在线 provider 状态恢复应各有明确接口。

   `Dynamics.checkpoint/load_checkpoint` 校验模型内容与配置；runner 的 `WorldModel.bundle/load_bundle` 另外保存测量、滤波、初始先验配置及关节标定。provider 的 `right_local_provider_v3` 快照记录完整后验与实际采样的在线首帧先验。

   `WorldModelEnv.snapshot/restore` 的想象快照使用独立的 `version=2`，额外保存参考数据的 `motion_digest`。SHA-256 覆盖 clip 边界、关节/根状态及 FK 刚体位姿与速度的 shape、dtype 和实际值，不依赖文件路径或设备。保存/恢复时分块重新计算；同长度不同内容、clip 重排及原地修改均拒绝恢复，且不提交 provider 或环境状态。旧 v1 或缺少指纹的快照明确报错，需要从保留 replay 重建，不能通过手改版本号迁移。

   记录短序列设备、dtype、batch、horizon、耗时、covariance 范围、jitter 次数、失败原因、创新量和梯度结果。控制完成率、跟踪误差和校准曲线可作为后续诊断，不作为本步启动后续训练的门槛。

| 提交三验收项目 | 通过标准 |
|---|---|
| 单次测量与时序 | actor 历史、滤波器和 replay 使用同一测量；初始帧、正常推进、部分 reset 均无重复 correction |
| 状态恢复 | 连续滤波结果与 snapshot/restore 后继续滤波一致，当前帧不重复融合 |
| 参考身份 | 同长度不同内容的 motion 不可恢复；原地修改也被检测，失败不推进 provider 或环境；路径迁移但内容相同可接受 |
| 失败重试 | provider 或输入验证失败后，重试不重复物理步、不双重推进 belief，并复用该时间点已生成的测量 |
| 信息权限 | A/B/C 继续满足第一步权限；B critic 的 R/p/v covariance 配置约定保留，critic 在后续接入 |
| 入口一致性 | 真实滤波所得完整后验经直接注入和 provider 两种入口，生成一致的历史、目标和 covariance 特征 |
| 真实采集 | CPU 小 batch MuJoCo 短序列使用真实滤波器，通过正常推进和 episode reset；检查结果有限且无跨 episode 状态污染 |
| 随机想象 | 使用独立潜在状态与 belief，至少 H_rollout=2；固定过程及测量噪声，可复现前向与反向结果 |
| 策略梯度 | 固定模型参数后，想象回报对动作及 actor 参数的梯度通过选定方向检查；有限差分使用相同噪声且避开裁剪拐点 |
| FiLM 梯度 | 零初始化时先验证输出层梯度；输出层更新后再验证 covariance 和上游分支梯度 |
| 图与版本生命周期 | 真实采集的已提交 belief 不保留历史计算图，想象序列保留所需梯度；模型或归一化版本不匹配时明确处理 |
| 保存恢复 | 模型、配置与完整 belief 恢复后，在相同输入和噪声下输出一致；错误版本明确报错 |

**实际模块与阶段归属。** 下列对应当前实现。

| 文件或模块 | 提交 | 职责 |
|---|---|---|
| 新增 `src/supermimic/utils/lie_state.py` | 一 | 完整状态转换、Exp/Log、右扰动、局部误差、reset Jacobian |
| `src/supermimic/belief.py` | 一 | 70 维切空间布局、完整均值与 covariance、批处理、版本、初始先验和 actor 投影 |
| `src/supermimic/utils/validation.py`、`src/supermimic/estimator_inputs.py` | 一 | 共享 SO(3)/covariance 校验与第一步 RPV 边界 |
| `src/supermimic/representations.py` | 复用第一步 | GenCoord 均值积分、64 维合成测量与共享参考窗口，无新增改动 |
| `src/supermimic/models.py` | 二 | 可选噪声头、分布/采样接口、配置与 checkpoint 兼容性 |
| `src/supermimic/estimator.py` | 二 | 纯函数 predict/correct、观测更新、innovation 与 jitter 诊断 |
| `src/supermimic/belief_provider.py` | 三 | 完整后验的 prepare/commit、时序、版本与恢复 |
| `src/supermimic/replay_buffer.py` | 三 | 幂等插入。完整 replay 还保留在线首帧先验、策略片段先验及按键前缀 |
| `src/supermimic/runners/supermimic_runner.py` | 三 | 采集模式、独立 provider、reset/restore、版本及保存恢复 |
| `src/supermimic/envs/world_model_env.py` | 三 | 想象双状态最小路径、随机数注入、原子提交和图边界 |
| `src/supermimic/envs/multi_task_world_model_env.py` | 三 | 实际 runner 入口的必要接线与一致性检查 |
| `src/supermimic/motion_loader.py` | 三 | 想象快照所用参考内容指纹 |
| `src/supermimic/tasks/env_cfgs.py` | 第一步修正 | 真实物理步之后重新构建刚体位姿奖励的对齐参考 |
| `src/supermimic/scripts/check_estimator.py` | 三 | CPU/GPU 可选的真实采集、重建、随机想象及快照 smoke |
| 新增及受影响的测试 | 一至三 | 几何、滤波、数值梯度、生命周期与集成验收 |
| `README.md`、`README_eval.md`、两份计划 | 三 | 运行方式、初始化/恢复契约、可用范围、验收记录与提交安排 |

**执行与验收顺序。**

1. 提交一完成后，独立验收几何、完整状态布局和 actor 投影，同时运行受影响的第一步几何与输入测试。
2. 提交二完成后，在合成数据上验收随机分布、传播、观测更新、covariance reset 和冻结模型后的动作梯度；通过后进入 provider 集成。
3. 提交三完成后，验收时序、失败重试、版本边界、CPU 真实采集与至少两步随机想象的梯度，再运行完整仓库回归测试。GPU 可用时补充实际设备 smoke，并分别记录已验证与未验证的设备。

几何和数值导数检查优先使用 float64；实际运行 dtype 另设合理容差。Monte Carlo 使用固定种子和统计容差，梯度有限差分固定所有随机样本。每个提交均应有可单独执行的验收命令和结果记录；单元测试通过后再进入依赖环境的烟雾验证。

**后续接口。** 策略起点按当前模型从保留片段的先验重建完整后验。两个模型拟合阶段直接使用真实 replay 帧，在线采集和策略想象共用更新后的动力学模型。[训练与验收](plan_step3.md)

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_estimator_lie_state.py tests/test_estimator_matrix_dynamics.py tests/test_estimator_belief_filter.py
OMP_NUM_THREADS=1 .venv/bin/python -m supermimic.scripts.check_estimator --device cpu
```
