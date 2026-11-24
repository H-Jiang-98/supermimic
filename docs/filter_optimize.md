# increment 模式的 Jacobian 计算

目标是计算下一状态的 **70 维局部误差**对当前状态 **70 维右局部扰动**的 Jacobian。本文统一采用“输出为行、输入为列”的矩阵约定。

## 1. 先对 35 维网络增量求导

固定动作和 context，记当前状态为 `x`，局部扰动为 `ε ∈ R⁷⁰`，扰动后的状态为 `x ⊞ ε`。world model 输出：

```text
g(ε) = [Δv_b(3), Δω_b(3), Δq̇_j(29)] ∈ R³⁵
Δ₀ = g(0)
G = ∂g/∂ε |₀                         # 35×70
```

这一阶段使用 `jacrev(..., has_aux=True)`，同时得到 `G` 和增量值 `Δ₀`。穿过 world model 的反向求导方向由原来的 70 个减为 35 个。

**如果将每个增量输出对 70 维输入的梯度按列堆叠，得到的 70×35 矩阵是 `D = Gᵀ`。后续组装时需要使用 `Dᵀ`，即 35×70 的 `G`。**

## 2. 用积分函数将增量导数映射到完整状态

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

## 3. 代码对应与梯度保留

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
