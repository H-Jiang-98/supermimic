# Step 5　整段评估协议与独立诊断

以 `dev_controller@675042e` 为实现基线。Step 1–4 已接通真实采集、模型拟合、随机想象、actor/critic 更新、恢复及独立仿真评估。下一步先解决长 motion 无法完成评分的问题，再判断两项简化是否足以支持闭环控制。原第五步的工作重新分配如下。

| 阶段 | 承接原计划 | 进入下一阶段的条件 |
|---|---|---|
| Step 5　评估协议与诊断 | 第一批的快照和批量评估、第二批的默认条件诊断 | 长 clip 的成功、失败、截断可区分。一次运行可追到快照和原始 episode。过程模型、在线后验和 critic 各有独立评分 |
| [Step 6　单 seed A/B/C 与性能](plan_step6.md) | 第四批的真实 motion pilot 及完整开销 | 在真实 MuJoCo 闭环上跑通三组、恢复并重现固定输入结果，取得当前版本的训练预算与设备开销 |
| [Step 7　多 seed 消融与报告](plan_step7.md) | 第五批的配置锁定、训练和报告 | 固定协议下交付三组逐 seed 配对结果、原始数据和方法说明 |
| 并行交付　ONNX actor | 第三批的部署导出 | Step 5 输入契约确定后可开始。最终导出物与 Step 7 入选 checkpoint 一一对应 |

## Step 4 审阅结论

| 组别 | 构造 actor 目标时的 R/p/v（训练 → 评估） | actor 显式 covariance | critic 输入 |
|---|---|---|---|
| A | 真值 → 后验均值 | 无 | 共同特权输入，1234 + dim(ξ)，无 belief |
| B | 后验均值 → 后验均值 | 无 | 共同输入 + 12 维相对均值 + 45 维 R/p/v 边缘 covariance |
| C | 后验均值 → 后验均值 | 每个参考帧 45 维，共 450 维，经 FiLM | 与 B 相同的 1291 + dim(ξ) 维输入和网络结构 |

共同输入包括真实状态表征、基于真值的参考目标、5 帧实测历史、clip 剩余时间与 ξ。B/C 相对均值用真值机身系下的旋转 rot6d、位置差、线速度差编码。45 维是后验右局部切空间 [δθ, δρ, δν] 的 **9×9 边缘块**，包含 9 个 log std 和 36 个相关系数。C 将同一个块按 10 个参考帧分别投影为任务误差 covariance。滤波、策略起点重建及想象继续传播完整的 70×70 covariance。这个投影与跟踪目标一致，代价是 critic 无法直接区分 R/p/v 边缘相同、其他状态不确定性不同的后验。是否影响 bootstrap，要看独立回报上的误差。[critic 特征](src/supermimic/critic_inputs.py) · [actor 特征](src/supermimic/estimator_inputs.py) · [策略目标](src/supermimic/learners/belief_policy_learner.py)

每个训练 seed 固定相同的初始世界模型和 actor 共同参数。B/C critic 初值相同，C 的 FiLM 输出层从零开始。三组随后独立采集并逐轮做两阶段拟合。先用连续真值窗口和实际动作自由滚动均值模型，默认 8 帧、评分 7 个后继，优化根位置、根朝向、关节位置和关节速度的加权 MSE。再冻结均值与归一化，用训练集中全部有效真实相邻帧计算 70 维右局部一步残差，以过程 NLL 单独训练完整 Cholesky 噪声头。该设计省去了模型拟合时反传穿过整段滤波器的开销，策略梯度仍穿过冻结模型和滤波器。均值存在系统误差时，零均值噪声头会吸收残差偏差的二阶矩。一步 NLL 的下降需要和长时后验校准分别验证。[两阶段实现](src/supermimic/learners/two_stage_wm.py) · [模型拟合与 replay](plan_step3.md)

`train_belief` 默认用七字段 model replay，`train_eit` 要求 full replay，以录下的动作、测量和片段先验重建后验。当前格式为 `eit_training_v3`、`eit_initialization_v3`，策略存档使用 `eit_rpv_belief_critic_v2`。v3 已有两个入口各两轮并恢复第三轮的 CPU 记录、RTX 5090 上默认均值网络的模型拟合记录及 757 项历史回归。v3 记录中两个入口各有一次四步评估，两次均在预算处截断，成功率为空。较早 v2 有 B 组默认 GPU 联合训练从第 10 轮恢复到第 11 轮的记录。当前单 clip evaluator、共享在线推理和无窗口播放可以直接复用。[版本校验](src/supermimic/runners/run_config.py) · [v3 验收](docs/validation/model_replay_v3.json) · [两阶段历史验收](docs/validation/two_stage_world_model.json) · [现有命令](README_eval.md#eit-定量评估和播放)

实验解释沿用既定分组。B−A 包含训练 actor 改用估计状态及 critic 获得 RPV belief 两项变化。C−B 衡量相同 critic 设计下，显式任务 covariance、参考信息路径及额外 FiLM 参数的整体增益。各组独立学习后，采集分布与模型权重也会分化。这三组可回答控制方案比较。判定“两阶段优于联合后验 NLL”或“RPV 优于完整 critic”需要单独的对照实验。

## Step 5　先让评估能回答实验问题

### 5.1 修正整段 motion 的终止协议

现有 `eval_eit.evaluate()` 使用训练任务配置，每步 0.02 s，含普通 10 s timeout。当前本地 40 个 50 Hz 文件长度为 5,109–13,656 帧，`walk1_subject1.npz` 有 13,066 帧。即使零物理失败，从片头开始也会在第 500 步先收到 timeout。现有成功定义只把 `clip_complete` 计为成功，直接以此选模会使全部候选的完成率都为零。[单 clip 评估](src/supermimic/scripts/eval_eit.py) · [环境配置](src/supermimic/tasks/env_cfgs.py) · [末帧终止](src/supermimic/motion_command.py)

为正式评估增加单独版本的**整段协议**。构建评估环境时仅调整普通时间上限，使它严格晚于该文件最后一个合法评分动作。保留物理失败项、原 reset/域随机化、首帧起点、逐文件首 episode 以及末帧奖励。直接切到 play 配置会连带改变初始扰动。结果记录协议版本、控制周期、clip 帧数、实际 timeout 和外部步数预算。原有 10 s 协议与带 `--steps` 的 smoke 各自标记。若另报固定时长存活率，分母和时间窗单列。

验收用超过 500 帧且可控无失败的轨迹，分别验证原协议触发 timeout、整段协议在片尾计一次成功。再覆盖中途失败、物理失败与片尾同帧、外部预算提前耗尽、首步失败无可评分帧。终止动作计入回报和长度。已被 auto-reset 替换的状态不进跟踪误差。预算截断保留原始记录，所在评估的完成率为 null，正式选模要求预定 episode 均有结果。用一个本地真实 motion 核对环境实际时限和终止原因。

### 5.2 固定候选快照、场景清单和选模规则

1. 在输出目录的训练锁内，用当轮已原子写入 `training.pt` 的同一份内容发布按实采帧数索引的不可变快照。索引包含内容摘要、轮次、均值、covariance、actor、critic 的实际更新步数、模型指纹和共同初始化指纹。保留已有原子替换及 `metrics.jsonl` 断点修复。中断后补登记同一快照，拒绝同键异内容。[训练保存](src/supermimic/runners/belief_training.py) · [联合存档](src/supermimic/runners/eit_training.py)
2. 每个训练 seed 以固定的生成组和配置预先生成一份 v3 共同初始化包，三组显式引用同一指纹。批量入口读取存档内容并检查 group、`stage=policy`、`replay_mode=full`、critic 契约、非空初值指纹和实际训练步数。现有 evaluator 也能读取固定 actor 的 `stage=model` 存档，该类型仅进入模型诊断。评估在独立进程调用现有 `evaluate()`，使评估的设 seed 操作与训练 RNG 隔离。[训练入口](src/supermimic/scripts/train_eit.py) · [推理加载](src/supermimic/runners/eit_inference.py)
3. 每个场景固定原始 motion 文件及摘要、加载后的 motion digest、评估 seed、环境数、episode 数、时限、扰动和初始 prior 条件。逐文件从片头运行，用快照摘要、场景与协议摘要作任务键，再按 episode 槽位去重。为训练、选模和最终测试分别划清清单。仅研究已见 motion 时用独立评估 seed。报告跨 clip/subject 泛化时按 clip/subject 预先留出，核对训练实际展开的文件列表。模型内部的 `episode_id % validation_stride` 只服务于拟合验证。
4. 第 0 轮单列为未训练基线。对已训练的候选，在验证集先按 clip 等权的整段完成率选模，同分比较平均有效进度，再比较 MPKPE，仍同分选较早快照。进度直接取首 episode 的 `length / reference_steps`，其中 `length` 含终止动作，同时保留失败原因。早期所有策略都跌倒时，进度仍能区分表现。无有效误差帧时将 MPKPE 记为空，不能以零胜出。选定的每组每 seed 快照用于全部测试条件。

验收覆盖同轮快照和 `training.pt` 的字节摘要、恢复写入和重试去重、缺失/失败任务、model-stage 误入正式排名、同分且早期全失败的候选。用不同长度的人工轨迹手算并核对聚合。每个 episode 内先按有效帧平均，clip 内各 episode 等权，每个训练 seed 内各 clip 等权。记录每个指标的有效样本数。选模程序只读验证结果，任何待补任务都须在报告中可见。

### 5.3 分别验证过程模型、在线后验和 RPV critic

| 对象 | 数据与实施 | 验收重点 |
|---|---|---|
| 过程模型 | 在固定、独立的 MuJoCo 状态/动作/ξ 记录上，对各冻结模型使用相同窗口和一步对。报告四组多步 MSE、R/p/v 一步有符号偏差、过程 NLL、白化残差 L⁻¹r 的均值和二阶矩，注明窗口和转移数。保留各组自身训练与验证日志 | 已知有偏均值和含交叉项的高斯残差能被分辨。短片段只有有效一步对时仍计入过程 NLL。核对 reset 两侧的帧和缺失后继的帧没有被配成转移 |
| 在线后验 | 在 `policy.act` 后读取 actor 当前使用的测量和已提交后验，按经过时间给出旋转、位置、速度误差及 RPV9 的校准。例如用 e = local_error(后验均值, 真值) 的前 9 维和同帧 Σ₉ 检查 95% 椭球经验覆盖率。同时报告 innovation、jitter、在该时间段仍存活的 episode 数 | 同一条连续记录分块或整段打分一致。用可手算的相关 covariance 检查二次型，用合成高斯样本检查经验覆盖率。遇到半正定块记录有效秩与零方差方向的违规。开启和关闭诊断须产生相同的逐帧动作、后验、归一化与 RNG 状态。若真实闭环过早失败，在固定连续记录上补测漂移并单列 |
| RPV critic | 离线加载被评估快照的冻结 critic 和归一化，复用共同特征 builder，从独立 episode 的动作前真值、真值目标、实测历史、片尾剩余时间、ξ 及该帧后验构造输入。用同一 γ 和逐步奖励计算到物理失败/片尾的折扣回报，记录各检查点所用的 V（包括 bootstrap 的取值上限），报告折扣回报预测的偏差、MAE、RMSE 和样本数。A 的真值输入训练策略与后验输入部署策略分别标记。B/C 直接在其后验策略上评分 | 手算含终止奖励的短轨迹。删失样本单列。当前训练解释方差来自同批 λ-target，评估 JSON 的 episode return 是未折扣总和，均不能直接作此对照。物理轨迹用硬终止，想象目标用 soft survival。若出现偏差，再从同一起点按训练的输入和 survival 规则生成独立长想象回报，对照模型与 critic 各自的误差 |

后验诊断沿用 evaluator 的动作前采样点。真实 reset 只初始化一次。统计分箱、暂停或输出均不能重启滤波。当前 online prior 从带噪仿真真值初始化，ξ 来自仿真上下文。模型专用 replay 缺少连续实测历史，共用轨迹应另行保存每个 episode 的初始 prior、实测序列、动作、真值、上下文与真实 episode 边界。记录其来源及划分，隔离诊断数据和拟合数据。[评估已有诊断](src/supermimic/scripts/eval_eit.py) · [共享策略适配器](src/supermimic/runners/eit_inference.py) · [provider](src/supermimic/belief_provider.py)

Step 5 按 5.1 → 5.2 → 5.3 分三次交付。先在 `scripts/eval_eit.py` 接入时限并通过终止分类回归，再在训练保存处发布快照，增加逐文件调度和汇总入口。最后给 evaluator 增加可关闭的后验统计，同时为固定轨迹的模型/critic 评分增加离线入口。届时把实际命令写入 `README_eval.md`，并保存一个含原始记录及期望汇总的可重复样例。

## 并行交付　ONNX actor 与运行时验收

在已有 exporter 增加 EIT actor wrapper，保留归一化、motion encoder、decoder 及 C 的 FiLM。输入是逐帧排列的 history (B,5,93)、已构造的 goal (B,700)，仅 C 再输入 uncertainty (B,450)，输出限幅前的归一化动作 (B,29)。`EITPolicyAdapter` 给 PyTorch 或 ONNX 后端同一份输入，复用测量、滤波、历史和参考构造，输出统一限幅一次后映射至物理动作。元数据记录动作顺序/映射、特征编码、组别、对应 actor 和 world model 身份。[导出入口](src/supermimic/runners/exporter.py) · [在线动作执行](src/supermimic/runners/eit_inference.py)

先固定 batch=1、CPU float32，用记录输入逐帧比对三组原始及最终执行的动作，覆盖非零 FiLM、较大 covariance、限幅与 reset，然后在共享运行时完成短闭环并记录目标设备延迟。GUI 在有显示的环境验证暂停、单步、clip 切换、Start Here 和 `online_posterior`。导出验收交付 actor 图、运行时契约及对应快照。正式控制消融可直接使用 PyTorch。
