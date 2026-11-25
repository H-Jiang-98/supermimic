# EIT v6 模型与策略训练实现

[配置教程](../README_eit.md) · [输入](eit_inputs.md) · [Replay](eit_replay.md) · [CUDA Graph](eit_cuda_graphs.md)

## 按物理步调度

[`EITTraining.run_step()`](../src/supermimic/runners/eit_training.py) 调用公共
`runner.step()` 采集一个物理步，累计帧数增加 `num_envs`，随后更新新帧归一化。
公共 `Ratio` 的训练组频率为：

```text
num_envs × replay_ratio / (mean_updates × mean_batch_size × mean_window_frames)
```

当前 TOML 频率为 0.2，稳态每 5 步一组。
首次调度时若缺少完整均值窗口、验证转移或后验起点，则跳过该机会，不补欠账。
小数余额、零 ratio（只采集）、负 ratio（每步一组）和单步多组沿用公共调度语义。
累计物理步、实际训练组、跳过组与各优化器步数分别记录，不由总步数推算实际更新量。

每组依次执行均值拟合、covariance 拟合、策略更新。
默认验证划分为 `episode_id % validation_stride == 0`，stride 为 5。
训练与验证使用独立且可恢复的抽样 RNG。

## 随机世界模型

[`Dynamics`](../src/supermimic/models.py) 的均值头输出 35 维速度增量：
`[Δv_body(3), Δomega_body(3), Δqdot(29)]`。
矩阵积分得到完整下一状态；每次姿态乘法后以 Newton–Schulz 步控制旋转漂移。
`forward` 返回确定性均值，`distribution_lie` 同时提供过程噪声因子，
`sample` 在预测均值右侧注入 `Lq @ epsilon`。

过程噪声头输出 70×70 下三角因子的 2485 个元素。
对角为 softplus 加正下界，非对角按所在行对角尺度缩放。
六个误差块的默认初始标准差为 `(0.01, 0.001, 0.05, 0.05, 0.005, 0.05)`，
描述一个控制步的过程噪声，与初始先验的配置分开。

拟合入口在 [`two_stage_wm.py`](../src/supermimic/learners/two_stage_wm.py)：

| 阶段 | 样本与目标 | 更新规则 |
|---|---|---|
| mean | 从真实首帧开始自由滚动，以根位置、根朝向、关节位置/速度加权 MSE 拟合多步预测 | 公共 Muon/Adam 分组，只更新均值参数 |
| covariance | 均值更新后提取一步右局部残差，以高斯 NLL 拟合因子 | 固定均值和归一化，独立 Adam |

每组均值抽 `mean_updates × mean_batch_size` 个窗口，有放回抽样。
长度为 W 的窗口含 W−1 个预测转移。
未显式指定 `cov_pair_budget` 时，covariance 抽样数为
`mean_updates × mean_batch_size × (mean_window_frames−1)`。
只计算当前抽中转移的 detached 残差，不缓存整份 replay 的残差。
NLL 按样本数加权，尾批不丢弃；验证另抽 `cov_validation_pairs` 对。

当前 TOML 每组 10 次 mean 更新，143360 个 covariance 对，
17 个 8192 整批加一个 4096 尾批，共 18 次 covariance 更新。
policy horizon、samples 和 buffer 容量不改变这一默认预算。
`train_belief` 仍提供固定 actor 的模型训练，其轮次制接口独立于 `train_eit` 的步数调度。

## 固定时域想象与 actor 梯度

[`imagined_rollout.py`](../src/supermimic/imagined_rollout.py) 将边界准备和张量展开分开：

- `prepare_rollout` 校验 starts、模型/参考身份、后验、历史、context 和固定噪声，
  复制 detached 起点，准备 H+1 个 clip-clamped 参考窗口及常量。
- `observe/imagine_step` 显式传递 `ImaginedState`，执行动作、状态传播、测量及滤波。
  动态失败标志在设备上合并，不更新在线 provider 的 episode、generation 或 pending 状态。

actor 与 critic 的特征编码共用张量函数。
`MultiEncoderAE.forward_with_aux` 显式返回动作和附加损失，不通过 `_aux_loss` 传递训练状态。
`RolloutResult` 提供 returns、critic 数据、metrics、附加损失、最终 belief 和 invalid。
公开 `_rollout` 包装继续用于诊断；生产训练使用张量路径。

每个起点扩展为 K=`samples` 条轨迹，固定展开 H=`policy_horizon` 步。
K=1 时潜在状态使用均值预测，不采过程噪声也不推进该 RNG；K>1 时采过程噪声。
两者都保留测量噪声和滤波器的过程 covariance。
中间 latent、belief、history 保持 autograd 联系；世界模型和 critic 参数在 actor 反传中固定，
但输入导数继续传递到动作、滤波、covariance、FiLM、soft survival 和价值 bootstrap。

每步先用当前观测产生动作，执行并评分，然后构造下一帧观测。
参考超过片尾时重复完整末帧，含其速度。actor 在 H 步后按折扣和累计存活权重 bootstrap。
真实采集/评估的片尾完成不作为此固定时域目标中的硬终止。

完整逻辑 batch 为 `policy_batch_size × samples`。
actor microbatch 仅切分反向计算，按全 batch 加权累积，然后做 norm 1.0 和逐元素 0.1 裁剪，
再更新一次。当前 TOML 每次 actor 更新产生 `2048 × 1 × 32 = 65536` 个想象转移。
actor Adam betas 为 `(0.9, 0.95)`；critic 为 `(0.9, 0.999)`，均复用公共 Muon 权重分组。

## Critic fitting

[`BeliefActorCriticPolicyLearner`](../src/supermimic/learners/belief_policy_learner.py)
用原始 value 构造 GAE（gamma=0.99，lambda=0.95），只裁剪最终 target。
回归 MSE 不按累计 survival 加权。

critic 数据按时间、轨迹顺序汇总，与 actor microbatch 划分无关。
每次 policy update 生成一份 permutation，跨 critic epochs 复用；
当前配置 5 epochs × 4 minibatches，一般为 20 次 critic 更新。
样本不足时减少非空 minibatch 数；`critic_microbatch_size` 只控制内部反传分块。

普通调用在 optimizer 前检查 rollout 和梯度；失败清空未提交梯度并报错。
启用完整 policy CUDA Graph 后，actor 和 critic optimizer 都在图内，
异常改为整图执行后检查，参数可能已经改变；具体边界见 [CUDA Graph](eit_cuda_graphs.md)。

## 保存与恢复

`training.pt` 保存模型、actor、critic、优化器、归一化、各 RNG、Ratio 余额、
累计计数、ReturnSampler 统计和模型 revision；联合训练只接受当前 `eit_training_v6`。
共享初始化格式为 `eit_initialization_v4`，critic 契约为 `eit_rpv_belief_critic_v3`。
各子系统版本独立，名称中的 v4/v3 不代表联合训练落后于 v6。

启动器先为每个 seed 生成 B 组初始化，再供 A/B/C 加载；
初始化指纹、网络契约、噪声、motion 身份在加载边界校验。
改变网络或噪声后应创建新实验，不能复用不匹配的初始化。

恢复保留训练统计与优化状态，但重建物理轨迹、replay、在线 belief 和 ReturnSampler 轨迹窗口。
因此先重新 warmup，不承诺接续同一物理轨迹。
`max_iterations` 是累计目标；组别、seed、环境数量、训练配置从 checkpoint 恢复。
编译/graph 为本次运行选项；MuJoCo 求解器覆盖项由本次命令提供。

固定输入的下一次更新一致性通过 `--resume-reference` 和 `check_eit_resume` 验证，
其证据范围是固定输入优化器更新，不等于物理采样轨迹逐位续接。
四文件发布、日志补齐和 pilot 操作见 [运行说明](eit_operations.md)。
推理仍支持契约匹配的 v5 策略权重；这不表示可将其作为 v6 训练 checkpoint 恢复。

## 代码与回归

- `tests/test_estimator_eit_training.py`、`test_estimator_step_schedule.py`：调度与预算。
- `tests/test_estimator_two_stage_wm.py`：均值/covariance 两阶段训练。
- `tests/test_estimator_belief_policy.py`、`test_estimator_tensor_rollout.py`：目标、梯度与张量展开。
- `tests/test_estimator_critic.py`、`test_estimator_eit_graph.py`：critic 与图执行边界。
- `tests/test_estimator_live_resume_reference.py`、`test_estimator_output_recovery.py`：恢复与发布。

[张量 rollout 微基准](validation/tensor_rollout_float32.json)包含固定输入前向/反向对照，
不包含优化器和物理采集；不能直接当作完整训练速度。
