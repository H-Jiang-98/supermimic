# 滤波实现优化

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
`rho/nu`，替代不能在此 CUDA 路径捕获的自适应 `matrix_exp`；公式见
[Solà 等，式 149](https://arxiv.org/abs/1812.01537)。

默认 `FilterConfig.validate_numerics=False`：滤波只检查 shape、dtype、device
和版本。显式 `belief.validate()`、初始化、状态加载和开启调试检查时可以同步。
常规 provider 只请求 `return_diagnostics="summary"`，返回 innovation norm；
`True` 仍提供完整诊断。旧 rank/jitter/retry 诊断已删除。
episode/reset 控制和独立 actor/critic 输入校验不属于此次同步清理范围。

Replay 使用 `incremental_posterior_sqrt_tril_f32_v2`，保存 2485 个 float32
因子元素，恢复时无需分解。Provider 使用 `right_local_sqrt_provider_v4`。
旧 replay/provider 状态不能恢复；旧模型和策略权重可加载，导入时转换旧 filter
配置，保留网络输入布局。离线完整协方差诊断记录保持原格式。

### float32 验证与性能

验证覆盖 CPU/CUDA 的标准 Kalman/Joseph 参考、一阶和二阶梯度、解析 reset、
100 步正常噪声序列、部分 reset、replay 往返、旧权重加载、CUDA capture，
以及 eager/compiled 策略梯度。共 151 项 float32 测试通过，静态检查通过；未验证 float64。

在远程 PyTorch 2.14.0+cu130 / NVIDIA RPBZZZ6 上，固定输入并预热后测得：

| 测量 | 原实现 | 平方根实现 |
|---|---:|---:|
| 校正前向，batch 1024 | 68.08 ms | 6.92 ms |
| 校正前向加反向，batch 1024 | 76.79 ms | 15.19 ms |
| 校正前向，batch 2048 | 126.21 ms | 8.35 ms |
| 校正前向加反向，batch 2048 | 135.92 ms | 16.26 ms |
| 生产网络、batch 1024、32 步 compiled rollout/backward | 7.20 s | 4.52 s |

校正取 10 次中位数；rollout 取 3 次中位数，使用固定 seed 2027、C 组、
increment Jacobian、512 的 Jacobian chunk 和生产网络宽度。
rollout 耗时减少约 37.3%，测量不包含编译预热、物理采集或 critic 优化。
原始计时与峰值显存见 [验证记录](validation/square_root_filter_float32.json)。

复现脚本为 `docs/validation/benchmark_square_root_filter.py`。在各版本源码根目录运行：

```bash
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src:. python docs/validation/benchmark_square_root_filter.py --output /tmp/filter-kernel.json --repeats 10
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src:. python docs/validation/benchmark_square_root_filter.py --rollout --output /tmp/filter-rollout.json
```

对照旧版时使用同一份脚本，并将 `PYTHONPATH` 指向旧源码及其测试目录。
