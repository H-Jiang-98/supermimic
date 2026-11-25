# EIT 输入、状态与网络

[训练概览](../README_eit.md) · [滤波](eit_filter.md) · [训练实现](eit_training.md)

## 状态和测量

EIT 使用 G1 GenCoord，29 个关节、50 Hz；主状态交换格式为：

```text
full_state = [p_world(3), quat_wxyz(4), q(29), v_world(3), omega_body(3), qdot(29)]
local_error = [theta(3), rho(3), nu(3), omega_body(3), q(29), qdot(29)]
```

`LieState` 内部以 3×3 矩阵表示朝向，完整均值共 76 个标量。
右局部误差为 70 维，RPV 对应其前 9 维；`rho/nu` 在名义根朝向的局部坐标系中。
`BeliefState` 保存完整均值和平方根 covariance，输入特征只使用所需的边缘信息。

64 维测量依次为重力方向 3、机体系角速度 3、关节位置 29、关节速度 29。
加上上一执行动作 29 构成每帧 93 维本体输入。
默认观测噪声为未截断高斯分布，四组标准差为
`0.05/3`、`0.2/3`、`0.01/3`、`0.5/3`，只加在传感器上。
`configure_estimator_env_cfg` 统一测量定义并关闭重复 corruption。

## Actor

输入协议为 `eit_rpv_v1`：

| 输入 | 维度 | 内容 |
|---|---:|---|
| history | 5×93 | oldest-first 实测历史；真实 episode 开头重复首帧补齐 |
| goal | 700 | 10 个参考帧，每帧 70 维相对 R/p/v 目标及关节位置、速度 |
| uncertainty（仅 C） | 450 | 每个参考帧 45 维任务 covariance 编码 |
| 输出 | 29 | 归一化动作，执行前按环境限制限幅 |

A 在训练时用真值构造 R/p/v 目标，B/C 用后验均值；三组部署时统一使用后验均值。
`EITInputConfig(mode="eval")` 决定信息来源，`actor.eval()` 只切换网络模式。

每帧任务 covariance 将后验的右局部 9×9 RPV 边缘块映射到目标误差空间，
编码为 9 个 log std 和 36 个上三角相关系数，保留旋转、位置和速度之间的相关。
C 在 decoder 首层 LayerNorm 后用 FiLM 调制；A/B 完全旁路此分支。

actor checkpoint 保存输入契约、噪声、结构和归一化状态。
`build_ablation_actors` 复制共同网络参数并分离存储，避免组间共享可写参数。

## Critic

v6 critic 契约为 `eit_rpv_belief_critic_v3`，不含剩余参考时长。
三组共同读取 `truth68 + truth_goal700 + history465 + context`。
A 输入维度为 `1233 + dim(context)`；B/C 追加 57 维，总计 `1290 + dim(context)`：

- 12 维后验均值：`rot6d(R_true.T @ R_mean)`、
  `R_true.T @ (p_mean - p_true)`、`R_true.T @ (v_mean - v_true)`。
- 45 维 covariance：后验右局部 RPV9 边缘块的 log std 与相关系数。

这里使用边缘块，不做条件化。后验关节/角速度均值与 RPV9 之外的 covariance
不直接进入 critic；滤波和想象起点仍保留完整 belief。
B/C critic 结构和共享初值相同，A 不读取 belief。

## 参考与奖励时序

actor 目标和全局骨盆奖励使用固定 episode 参考对齐。
身体形态奖励按评分时真实根的水平位置和 yaw 重新对齐参考。
动作执行满一个控制步后，以推进后的机器人状态对照选动作时的参考帧计算奖励。
真实环境先刷新刚体位姿/速度，再计算终止与奖励，最后自动 reset 和更新参考。

想象训练在片尾重复完整末帧（包括其速度）；真实采集、评估仍在片尾结束。
参考指纹包含加载后的张量和 clip 边界，路径变化本身不改变指纹。

## 代码与回归

- [`estimator_inputs.py`](../src/supermimic/estimator_inputs.py)：测量、目标、任务 covariance。
- [`critic_inputs.py`](../src/supermimic/critic_inputs.py)：critic 契约、编码和归一化。
- [`models.py`](../src/supermimic/models.py)：actor、FiLM 和动力学网络。
- `tests/test_estimator_inputs.py`、`test_estimator_film.py`、`test_estimator_critic.py`：输入与契约。
- `tests/test_estimator_reward_timing.py`、`test_estimator_motion_alignment.py`：奖励与参考时序。
