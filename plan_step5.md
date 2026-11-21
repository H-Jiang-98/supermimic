# 第五步 正式评估 部署导出与三组消融

**实施状态：计划，启动前修复已完成。** 本步承接 [第四步](plan_step4.md)，在现有训练、评估与播放入口上补齐实验组织、长期诊断、actor 导出及规模验收，随后开展 A/B/C 多 seed 消融。沿用 SuperMimic 与 mjlab 的评估代码；新增代码集中在 EIT 适配和实验需要的扩展。

## 已完成的实施基线

第一至第四步已实现策略输入、完整 belief 滤波、直接旋转矩阵积分与向量化 Jacobian、分阶段拟合均值和过程 covariance，以及 actor/critic 交替训练。第五步启动前又完成以下修正，后续直接复用。

| 已有能力 | 实现与使用约定 |
|---|---|
| 共享 EIT 推理 | [eit_inference.py](src/supermimic/runners/eit_inference.py) 提供 checkpoint 配置解码、模型加载和 `EITPolicyAdapter`；只构建推理所需模型，归一化固定 |
| 定量评估 | [eval_eit.py](src/supermimic/scripts/eval_eit.py) 沿用 mjlab tracking evaluator 的首 episode 循环，直接调用其跟踪指标函数；从 clip 头开始，每次评估一个 motion 文件 |
| 统计边界 | 区分失败、clip 完成、普通 timeout 和评估预算截断；预算截断保留记录，成功率为 null；没有有效评分帧时误差为 null |
| 参考与状态时序 | 跟踪指标使用动作前评分参考和动作后物理状态，相对身体指标按当前根位姿重锚；排除已被自动 reset 替换的终止状态 |
| 播放与在线后验 | 原 [play.py](src/supermimic/scripts/play.py) 和 [viewers.py](src/supermimic/scripts/viewers.py) 支持本地 EIT checkpoint；蓝紫 ghost 标为 `online_posterior`，显示后验均值 |
| 生命周期 | 手动/自动 reset、clip 切换和 Start Here 刷新在线状态；纯参考帧拖动复用当前后验；可视化重启不重置真实后验或历史 |
| 新 motion | 允许兼容的新 motion 用于评估，分别记录训练与评估内容指纹；续训继续严格检查训练 motion 身份 |

真实 walking 参考的 smoke 使用未训练小网络，验证首 episode 记录与失败分类。viewer 已验证真实 MuJoCo 的无窗口执行和绘图回调；实际 GUI 展示与本轮 GPU 验收仍待完成。上述修复已提交为 `b87a0cd`（`fix: share EIT inference across tracking evaluation and playback`），作为第五步编码基线。

现有命令和统计口径见 [README_eval.md](README_eval.md#eit-定量评估和播放)。本步不重复建设 evaluator、在线滤波器或 viewer。

## 实验定义和结论边界

| 组别 | 训练 actor 的 R/p/v 来源 | 评估 actor 的 R/p/v 来源 | actor covariance 路径 | critic |
|---|---|---|---|---|
| A | 潜在真值 | 后验均值 | 无，FiLM 旁路 | 共同特权输入，不含 belief |
| B | 后验均值 | 后验均值 | 无，FiLM 旁路 | 共同输入、12 维 R/p/v 后验均值编码与 45 维 9×9 边缘 covariance 编码 |
| C | 后验均值 | 后验均值 | 450 维任务 covariance 编码与 FiLM | 与 B 相同的输入和结构 |

按 [第四步 RPV critic 修订](plan_step4.md#rpv-critic-输入修订2026-11-20)，B/C 总输入为 `1291 + dim(ξ)`，仅追加 R/p/v 后验均值与 `[δθ, δρ, δν]` 的 9×9 边缘 covariance；保留该块内全部交叉项。完整滤波和 replay 策略起点仍使用 70 维 belief。三组均使用 `eit_rpv_belief_critic_v2` 契约加载共同初始化包和联合训练 checkpoint。批量评估与播放也在加载 checkpoint 时检查其中的 critic 契约。

每个训练 seed 使用一份显式共同初始化包：世界模型和 actor 共同权重一致，B/C critic 初值一致，C 的 FiLM 输出层零初始化。加载后三组独立采集与训练，不共享活动参数或后验。每轮用真实窗口拟合预测均值，再用同一训练集内的全部一步残差拟合过程 covariance。

B−A 衡量训练目标采用估计状态、同时为 critic 增加 belief 信息的整体收益。C−B 衡量相同 critic 设计下任务 covariance 编码与 FiLM 的增量收益；该分支包含额外参数和目标信息路径，不能解释为排除了这些因素的纯 uncertainty 因果效应。主实验保持三组。

pilot 使用独立 rollout 的实测回报检验 RPV critic 的价值预测误差，记录 bootstrap 对 actor 目标的贡献。

当前模拟评估的 episode 起点使用带噪真值 prior，context 使用已知仿真参数。这是三组共同的初始化和参数知识条件。只有本体感知的长期在线结果应报告位置、朝向和速度漂移；不能以片段重新初始化后的短窗口结果代替连续在线表现。

## 分批实施

**第一批：组织可复现的评估清单、checkpoint 和结果汇总。**

复用 `train_eit`、`eval_eit.evaluate()`、`EITTraining` 和现有 JSON 输出，增加薄的批量调度与汇总入口。评估清单记录 group、训练 seed、评估 seed、共同初值指纹、checkpoint 身份、motion 列表、每个 clip 的 episode 数，以及实际环境覆盖项。已有 `runner_config`、模型配置和 motion digest 继续作为配置来源；新增配置只覆盖实验组织所需字段。

先完成训练 motion 上的独立 episode 评估。需要研究泛化时，另行固定 train/validation/test 的 clip 或 subject 划分，写入清单；当前 `episode_id % stride` 留出只属于模型训练内部验证。多 clip 通过逐文件调用现有 evaluator 完成，每个文件均从自身第 0 帧开始，不能把拼接 motion 的全局第 0 帧当成全部 clip 的起点。

`training.pt` 保持最新续训 checkpoint 的职责，保留现有文件锁、原子替换和日志恢复约定。在指定轮次或采集帧数保存额外的不可变快照，记录对应轮次、实际采集帧数和 optimizer 步数；快照复用同一次完整保存的状态，避免额外取样或改变 RNG。验证集按预先固定的规则选择 checkpoint，选定后用于全部测试条件，测试集不参与选模。第一版可采用“验证集 clip 完成率优先，同分比较跟踪误差，仍同分选较早 checkpoint”的规则；无有效误差样本时不能以零误差胜出。

结果按 clip 内 episode、clip、训练 seed 分层汇总，三组使用相同 clip 权重。记录成功率、回报、长度、原有跟踪误差，以及失败、timeout、截断和无有效误差样本的数量。沿用每个 episode 内平均、episode 间等权的指标定义；同时展示失败统计，避免把较短的存活片段误差解释为整体控制优势。批量结果以 checkpoint/clip/seed 的组合标识去重，失败任务保留错误状态，不生成伪造的成功记录。

验收：用不同长度的人工 episode 检查分层汇总；重跑和中断恢复不重复计数；快照与当轮 `training.pt` 的模型/optimizer/RNG 一致；验证选模不读取测试结果；独立评估进程不影响训练状态。已有真实新 motion 集成测试继续保留。

**第二批：增加连续后验诊断，按实验需要扩展噪声条件。**

评估按物理块报告后验误差、边缘 covariance 覆盖率、innovation 和 jitter。训练日志另记录分组多步 MSE 与一步过程 NLL。

诊断消费 actor 实际使用的同一测量和后验，不重新采样、不额外 correct、不更新统计。整段 episode 连续运行，仅真实 reset 时初始化；时间分箱和日志输出不得触发片段 prior 重建。按经过时间报告误差和校准指标，同时记录每个时间区间仍有观测的 episode 数，避免把早期失败轨迹的缺失当作误差下降。

先验证默认 checkpoint 条件。若策略早期失败，使用固定连续状态/动作/测量记录补充滤波器长时诊断，明确区分记录序列的估计实验和真实控制闭环结果。

噪声及 prior 失配属于本批可选实验扩展：仅在开展对应实验时，增加实际传感采样分布与滤波器 V 的独立覆盖项，以及初始均值误差与初始 covariance 的独立配置。默认路径精确复现 checkpoint；覆盖项单独记录，不静默改写保存的训练语义。至少区分默认匹配、实际噪声与 V 同步变化、仅实际噪声变化三种条件。

现有相同 seed 只保证首次 reset/测量条件可复现。需要严格配对噪声实验时，再增加按场景编号和 episode 时间索引的随机流，明确记录实际 reset/prior/测量扰动；不能宣称现有全局 seed 已保证不同失败时间后的全程配对。这项扩展不阻塞默认条件下的主消融。

验收：与独立连续滤波参考比较分块及整段结果；日志开关不改变动作、后验、归一化或 RNG；覆盖率在可手算分布上正确；实现失配覆盖时检查实际采样方差、滤波器 V 和输出标签分别符合配置。

**第三批：复用现有 exporter 导出 EIT actor。**

在 [exporter.py](src/supermimic/runners/exporter.py) 中增加 EIT wrapper，复用现有导出和 metadata 工具。保留 legacy 导出格式。首版 EIT 图只覆盖归一化、motion encoder、decoder 和 C 的 FiLM。

| 导出输入或输出 | 形状与语义 |
|---|---|
| `history` | B×5×93，时间从旧到新；直接使用 EIT 的逐帧布局 |
| `goal` | B×700，已由共享 builder 用后验均值构造 |
| `uncertainty`，仅 C | B×450，既有任务 covariance 编码 |
| `action` | B×29，限幅前的归一化动作 |

滤波、历史维护、参考对齐和几何特征仍由共享 EIT 运行时完成。为 `EITPolicyAdapter` 增加最小的 actor 执行后端接口，使 PyTorch 和 ONNX 消费同一份输入；动作限幅继续在适配层执行一次，随后使用既有物理 scale/offset。metadata 保存组别、输入布局/编码版本、动作顺序及映射、归一化和模型身份。导出文件表示 actor，不宣称包含完整估计闭环或实机初始化。

验收先用 batch=1、CPU float32，比对 A/B/C 的原始动作和最终执行动作。覆盖正常状态、较大 covariance、非零 FiLM、限幅以及 reset 后的输入；C 的响应测试应使用离开零输出初始化的权重。再在同一批已记录输入上比较 ONNX Runtime 与 PyTorch，并用共享运行时做短闭环。以动作逐帧一致性作为数值验收，长期物理轨迹仅作容差内诊断。实际目标设备及 batch 支持在导出元数据中明确。

ONNX 批次可与后续 PyTorch 性能和训练实验并行，不作为主消融的运行依赖。

**第四批：测量完整开销并完成单 seed 真实运动 pilot。**

扩展既有 `benchmark.py` / `check_eit_scale.py`，接入共享适配层和完整轮次计时，使用真实历史与在线滤波。分别报告物理采集、前缀重建、模型更新、策略 rollout/反向、critic 回归、保存及整轮开销。覆盖默认网络宽度、滤波器完整 70 维 covariance、critic 的 12+45 维 RPV belief 输入、实际 episode 前缀长度，以及 batch、K、H 和 microbatch 的变化。critic 降维只减少其特征与网络开销，完整滤波和 Jacobian 的计算仍须计入。

在线路径测量“测量/滤波—特征构造—actor—动作限幅”的完整延迟，记录均值、p95/p99、峰值内存和持续运行后的保留内存。保存设备、dtype、线程、预热、重复次数和 CUDA 同步方式；PyTorch 分配与进程/驱动总占用分开报告。对 50 Hz 控制周期给出相对 20 ms 的余量或超时情况，actor 单独推理时间不代表完整控制周期。

优先使用现有 eager 路径。模型内容哈希、参考摘要或 Python 校验等只有在 profile 显示为瓶颈时再优化；优化后保留版本失效和缓存一致性测试。`no_grad()` 下的在线推理仍需要局部 Jacobian，继续保留现有 inference-mode 适配与防护。

pilot 使用真实运动参考，从每组同一共同初值开始。起始参数沿用第四步的工程配置：

| 参数 | pilot 起始值 |
|---|---|
| 模型学习 | 均值每轮 10 批，每批 2048 个 8 帧窗口。covariance 每批 256 对，打乱遍历一次 |
| 策略学习 | N=4、K=2、H=4、microbatch=2，每轮 1 次 actor 更新和 2 个 critic epoch |
| 真实采集 | 8 环境、每轮 64 步、buffer=128 |
| 网络 | 均值 2048³、噪声 (128,64)、actor (2048,1024,512,256)、critic (256,128) |

先运行 10 轮检查持续更新、保存恢复和实际闭环评估，再依据性能与学习曲线延长训练或统一调整工程配置。该轮数是排错起点，不作为收敛标准。主要监控真实控制指标、均值 MSE、过程 NLL 与校准、动作限幅、FiLM 尺度、bootstrap 占比及各阶段梯度，检查想象回报与真实回报的偏离。

验收：持续更新无旧图积累，实际更新预算与日志一致，恢复后能继续收集和优化；默认宽度 CPU/GPU 分别保存实测结果。GPU 验收记录本次运行的 critic 输入契约与源码指纹。GUI 展示在可用显示环境下检查暂停、单步、clip、Start Here 及 `online_posterior` 标签。

**第五批：锁定配置，运行多 seed 消融并形成报告。**

根据 pilot 确定统一训练预算与工程配置，固定训练/验证/测试清单、评估次数、checkpoint 选择规则、主要指标和报告方式，再运行正式 A/B/C。优先按真实采集帧数和 optimizer 更新数对齐预算，并单列耗时、显存和 actor 参数量；三组分别使用各自数据和模型。

建议先完成 3 个训练 seed 检查实验方差和失败模式，再扩展至至少 5 个训练 seed。每个 seed 的 A/B/C 使用同一初始化包及相同评估清单。完整保存单次 episode、单 clip 和单 seed 结果；以训练 seed 为独立重复单位报告均值和不确定区间，并报告 B−A、C−B 的逐 seed 差值。不能把同一策略的时间步或并行环境数量当作训练重复数。若只完成少量 seed，应相应限定结论强度。

主报告包括训练预算下的学习曲线、最终真实回报/完成率/跟踪误差、后验误差与校准，以及完整计算开销。区分训练 motion 上的控制表现、未见 clip/subject 泛化，以及可选的噪声/prior 失配结果。未训练模型 smoke、正常结束的程序和正向控制收益是不同验收结果；允许消融没有显著增益或存在退化，不能以要求 C 优于 B 作为工程完成条件。

交付可重跑的配置、初始化/模型身份、不可变 checkpoint 索引、原始 JSON/表格、汇总脚本和独立图表。导出的 actor 与报告中对应的评估 checkpoint 一一对应，测试结果和可视化示例不得混用不同版本。

## 修改位置与执行顺序

| 位置 | 剩余修改 |
|---|---|
| `scripts/train_eit.py`、`runners/eit_training.py`、共享保存逻辑 | 周期快照、预算与评估关联；保留既有续训/日志原子性 |
| `scripts/eval_eit.py`，拟新增薄的批量评估/汇总脚本 | 逐 clip 调度、实验清单、分层汇总、后验诊断；需要时增加噪声覆盖 |
| `runners/eit_inference.py` | ONNX actor 执行后端；按实验需要扩展测量/prior 覆盖，复用当前生命周期 |
| `runners/exporter.py`，拟新增 EIT 导出 CLI | EIT actor wrapper、输入/动作 metadata 和导出验证 |
| `scripts/benchmark.py`、`scripts/check_eit_scale.py`、训练计时 | 完整在线延迟、完整轮次耗时、长前缀及内存测量 |
| 既有 evaluator/adapter/viewer 测试及新增导出、汇总测试 | 信息权限、指标口径、后端等价、选模与快照恢复 |
| `README_eval.md`、`README.md`、`docs/validation/` | 新命令、协议、源码指纹、CPU/GPU 验收记录及实验结果索引 |

实现顺序为第一批 → 第二批默认诊断 → 第四批 pilot → 第五批正式实验；第三批 ONNX 可以独立推进。噪声失配和严格随机流配对按对应实验需要加入第二批，不阻塞默认条件主实验。

各批形成独立可审阅提交，先运行受影响测试，再执行相应真实闭环验收。新增或改变训练/推理生产路径后运行完整回归；纯报告生成复核数据、汇总和图表，不重复无关训练。

第五步的工程完成条件是：复用现有评估的批量实验与选模可复现，连续后验诊断正确，EIT actor 导出和完整开销通过目标设备验收。实验完成条件是：固定预算下的三组多 seed 原始结果、统计和结论边界完整交付。实现就绪不等同于已经完成正式训练，也不预设三组性能排序。
