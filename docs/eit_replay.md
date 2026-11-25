# EIT Replay 与策略起点

[训练概览](../README_eit.md) · [滤波](eit_filter.md) · [训练实现](eit_training.md)

## 真实转移与后验缓存

联合训练使用 full replay，每环境保留 `buffer_size` 帧。
主 replay 保存真实状态、实测 actor 输入、已执行动作、context、参考信息以及
`episode_id/time_index/terminated/truncated`，用于世界模型窗口和后验起点重建。

在线滤波每个物理步都执行；`posterior_save_stride=S` 只控制后验写入频率。
在全局采集步 0、S、2S……保存已融合当前测量、尚未执行当前动作的 `Q[t|t]`。
采集序号跨单环境 reset 连续，S 的合法范围为 1 到 `buffer_size`。

每环境的后验容量是 `ceil(buffer_size/S)`，每条包含：

| 数据 | 存储 |
|---|---|
| 完整均值 | 76 个 float32，朝向使用矩阵 |
| covariance 平方根 | 2485 个 float32 下三角因子元素 |
| posterior revision | 滤波时的整数模型修订号 |

格式为 `incremental_posterior_sqrt_tril_f32_v2`。
读取时直接重建因子，不重新分解协方差，也不重新跑 episode 前缀滤波。
每个后验的均值和因子载荷为 10244 bytes；当前 2048 环境、1000 帧、S=1 时约 19.54 GiB，
尚未包含主 replay、索引、边界历史、网络和 CUDA Graph 工作区。

## 因果历史和采样

`episode_id/time_index` 标识真实帧，`motion_time` 单独表示参考进度。
策略起点包含潜在真值、完整后验、历史、参考对齐和 context，后验作为 detached 边界。
初始测量已经融合过，不会在想象 reset 时再次融合。

控制历史由主 replay 的 `actor_obs` 与最近淘汰的 `history_length−1` 帧边界观测组成。
真实 episode 开头重复首帧补齐；环形覆盖不能被误当成 episode 开头。
`sample_policy_starts` 从仍保留且保存过后验的所有帧均匀、有放回抽样，
单帧片段也可入选；`prepare_policy_starts` 按给定帧标识和顺序读取。

世界模型使用全部有效真实窗口和相邻转移，不受后验 stride 限制。
窗口不跨 episode、时间缺口或 terminated/truncated 边界。
`["wm validation"].validation_stride` 控制模型验证划分，默认 `episode_id % 5 == 0`。
这些 episode 不参与世界模型拟合或其归一化，仅用于世界模型在线验证；
该划分不限制策略起点抽样及 actor/critic 归一化。

## 校验与提交

`ReplayBuffer.insert()` 先转换到实际存储 dtype，再验证新增帧的有限值、旋转、
动作和 context，以及与前帧的有效相邻关系。失败不推进环形计数、缓存或去重 token。
跨 reset、时间缺口、终止和截断边界不比较连续动作/context。

`_sealed_round()` 构建 episode/time 索引并检查 episode ID 结构，不重复扫描全部浮点内容。
显式 `validate_contents()` 可逐帧排查数据损坏，不重建索引或抽取新先验。
直接修改私有 `_storage` 会绕过写入校验。

模型更新不重算已保存后验。采集从已提交后验继续，以新模型处理下一帧；
起点可来自不同 posterior revision，想象更新期间世界模型固定。
只读阶段会校验模型和参考身份，防止中途切换参数或 motion 内容。

## 归一化和预算观测

每个新模型训练帧仅计入一次世界模型归一化；actor/critic 归一化读取新采且仍保留的后验帧，
包含世界模型验证 episode，并将所有选中的帧作为一个 batch 更新输入统计量。
统计先校验再提交，warmup 也更新；同一步多个训练组不会重复计数。
本步没有新保存后验时沿用已有统计。

`Budget/*` 区分有效后验总数、可用策略起点数、归一化样本数、抽取起点数和实际想象步数。
`posterior_revision_gap_*` 描述缓存版本落后程度。
主 replay、后验、边界历史、序号索引和初始先验分别报告字节数。

完整训练 checkpoint 不保存 replay；恢复后重建物理轨迹和缓存，保存序号从新 replay 的 0 开始。
累计训练步与调度余额继续使用 checkpoint 中的值。

## 代码与验证

- [`replay_buffer.py`](../src/supermimic/replay_buffer.py)：转移、环形索引和增量校验。
- [`posterior_replay.py`](../src/supermimic/posterior_replay.py)：因子打包、边界历史和缓存。
- `tests/test_estimator_posterior_replay.py`、`test_estimator_policy_starts.py`：起点与历史。
- `tests/test_estimator_replay_validation.py`、`test_estimator_replay_restore.py`：写入和恢复边界。

[增量校验测量](validation/eit_incremental_validation.json)只测 model replay，
不包含 policy CUDA Graph 的工作区；完整显存排查见 [CUDA 执行](eit_cuda_graphs.md)。
