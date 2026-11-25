# EIT 状态估计与平方根滤波

[训练入口](../README_eit.md) · [输入设计](eit_inputs.md) · [后验 replay](eit_replay.md)

## increment 模式的 Jacobian 计算

目标是计算下一状态的 **70 维局部误差**对当前状态 **70 维右局部扰动**的 Jacobian。本文统一采用“输出为行、输入为列”的矩阵约定。

### 1. 先对 35 维网络增量求导

固定动作和 context，记当前状态为 `x`，局部扰动为 `ε ∈ R⁷⁰`，扰动后的状态为 `x ⊞ ε`。world model 输出：

```text
g(ε) = [Δv_b(3), Δω_b(3), Δq̇_j(29)] ∈ R³⁵
Δ₀ = g(0)
G = ∂g/∂ε |₀                         # 35×70
```

这一阶段使用 `jacrev(..., has_aux=True)`，同时得到 `G` 和增量值 `Δ₀`。穿过 world model 的反向求导方向由原来的 70 个减为 35 个。

**如果将每个增量输出对 70 维输入的梯度按列堆叠，得到的 70×35 矩阵是 `D = Gᵀ`。后续组装时需要使用 `Dᵀ`，即 35×70 的 `G`。**

### 2. 用积分函数将增量导数映射到完整状态

记现有积分函数为 `I(x, Δ)`，名义下一状态为 `x̄⁺ = I(x, Δ₀)`。定义：

```text
h(ε, Δ) = local_error(x̄⁺, I(x ⊞ ε, Δ)) ∈ R⁷⁰
```

把 `ε` 和 `Δ` 当作两个独立输入，在 `(ε=0, Δ=Δ₀)` 处分别求偏导：

| 矩阵 | 定义 | 维度 | 含义 |
|---|---|---|---|
| `A` | `∂h/∂ε`，固定 `Δ` | 70×70 | 当前状态直接经过积分产生的导数 |
| `B` | `∂h/∂Δ`，固定 `ε` | 70×35 | 35 维增量对下一状态局部误差的影响 |

实际转移误差是 `h(ε, g(ε))`，因此按链式法则：

```text
F = d h(ε, g(ε)) / dε |₀
  = A + B @ G
      (70×35) @ (35×70) → 70×70

若网络梯度保存为 D（70×35），则 F = A + B @ D.T。
```

这里积分导数 `B` 和网络梯度 `D` 虽然都是 **70×35**，但含义不同。不能仅靠转置或拼接一个 70×35 矩阵恢复完整 Jacobian。

`A` 也不能省略。例如关节速度积分为 `q̇⁺ = q̇ + Δq̇`，其导数既包含网络增量的导数，也包含当前关节速度直接传递的单位矩阵项。

### 3. 代码对应与梯度保留

[`LieBeliefFilter._increment_jacobian`](../src/supermimic/estimator.py) 的组装过程为：

```python
increment_jacobian, prediction = jacrev(
  self._local_prediction, argnums=0, has_aux=True, chunk_size=chunk
)(epsilon, state, action, context)  # G: 35×70；prediction: Δ₀

direct, through_increment = jacrev(
  self._local_integration, argnums=(0, 1), chunk_size=chunk
)(epsilon, prediction, state, nominal)  # A: 70×70；B: 70×35

return direct + through_increment @ increment_jacobian  # F: 70×70
```

`A、B` 仍通过现有积分函数的自动微分计算，包括旋转正交化、角速度坐标变换和局部误差映射；这部分不再穿过 world model。名义状态在局部求导时固定，但 `Δ₀、G、A、B` 及名义状态的外层计算图均保留、不做 `detach`，以支持策略训练对动作、状态和模型参数的进一步求导。

批量样本由 `vmap` 独立计算，最终形状为 `(..., 70, 70)`。减半的是穿过网络的局部反向方向数，整体训练耗时还包含积分求导、矩阵乘法和滤波等操作。

## Square-root Kalman filter

`BeliefState.covariance_sqrt` 保存正对角的下三角因子 `L`，满足
`P = L @ L.T`。`.covariance` 是按需计算的完整矩阵，修改它不会修改 belief；
内部构造、索引、拼接、clone、detach、replay 和 provider snapshot 直接传递因子。
外部 `BeliefState(mean, covariance)` 和 `filter.initialize(mean, covariance)`
仍接受正定协方差，只在这个边界校验和分解一次。Gaussian prior 直接使用标准差。

令 `root(A)` 为对 `A.T` 做 QR 后，归一化对角符号得到的 `R.T`：

```text
prediction: L_pred = root([F @ L, Lq])
innovation: U = H @ L_pred; C = root([U, diag(sensor_std)])
gain:       K = (L_pred @ U.T) @ (C @ C.T)^-1  # 实现使用两次三角求解
posterior:  L_post = root([J @ (L_pred - K @ U), (J @ K) * sensor_std])
```

这里 `J` 为修正后的 Lie reset Jacobian；后验是平方根 Joseph 更新。
热路径不构造完整 `P/Q`，不进行秩判定、谱截断、伪逆、Cholesky 或 jitter 重试。
标准传感器噪声严格为正；不支持奇异先验或为极小噪声添加恢复分支。

训练时 QR 前向仅计算 `R`，通过自定义解析 VJP 避免生成和微分 `Q`。
对于 `L=root(A)` 和上游梯度 `G`，令 `M=L.T@G`，
`S=tril(M,-1)+tril(M,-1).T+diag(diag(M))`，则
`grad_A=L^-T @ S @ L^-1 @ A`，同样只用三角求解。
保存返回的 `L` 保留二阶梯度。Lie reset 使用解析 SE(3) 交叉块分别处理
`rho/nu`，避免在 CUDA capture 内执行自适应 `matrix_exp`。

默认 `FilterConfig.validate_numerics=False`：滤波只检查 shape、dtype、device
和版本。显式 `belief.validate()`、初始化、状态加载和开启调试检查时可以同步。
常规 provider 只请求 `return_diagnostics="summary"`，返回 innovation norm；
`True` 仍提供完整诊断。旧 rank/jitter/retry 诊断已删除。
在线 episode/reset 控制和公开 actor/critic 输入校验继续保留；训练专用张量路径见 [训练实现](eit_training.md)。

Replay 使用 `incremental_posterior_sqrt_tril_f32_v2`，保存 2485 个 float32
因子元素，恢复时无需分解。Provider 使用 `right_local_sqrt_provider_v4`。
旧 replay/provider 状态不能恢复；旧模型和策略权重可加载，导入时转换旧 filter
配置，保留网络输入布局。离线完整协方差诊断记录保持原格式。

## 在线状态与边界

[`belief_provider.py`](../src/supermimic/belief_provider.py) 管理完整后验、context、
`episode_id`、`time_index` 和模型 revision。首次测量只融合一次；部分 reset 只初始化
对应环境槽。每次预测使用实际执行的限幅动作，actor 与滤波器共用同一帧测量。

默认初始先验是在真实首帧状态右侧施加高斯扰动，六个误差块的标准差依次为
`(0.01, 0.01, 0.05, 0.05, 0.005, 0.05)`。
`prepare/commit` 将候选计算与提交分开；输入或目标检查失败不发布半完成后验。
真实物理步已经执行、但后续记录提交失败时，runner 重试该帧提交，避免重复动作。

世界模型更新后保留已提交的在线后验，以新模型处理下一帧。
`posterior_revision` 记录后验产生时的版本，与当前 `model_revision` 分开。
provider 的外层梯度只在可微调用且启用 autograd 时保留；在线采集不保留训练图。

## 实现与验证入口

- 状态与右局部几何：[`belief.py`](../src/supermimic/belief.py)、
  [`lie_state.py`](../src/supermimic/utils/lie_state.py)。
- 滤波、Jacobian 和平方根核：[`estimator.py`](../src/supermimic/estimator.py)。
- 回归：`test_estimator_belief_filter.py`、`test_estimator_increment_jacobian.py`、
  `test_estimator_belief_provider.py`，位于 `tests/`。

保留的固定输入测量：[increment Jacobian](validation/eit_increment_jacobian.json)、
[平方根滤波](validation/square_root_filter_float32.json)。这些是局部算子对照，
不代表完整训练吞吐或长期收敛。
