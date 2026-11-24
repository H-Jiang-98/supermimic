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
在线 episode/reset 控制和公开 actor/critic 输入校验继续保留；训练专用路径见下节。

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


## 局部张量 imagined rollout

训练不再通过 `WorldModelEnv.step → BeliefProvider.prepare/commit` 展开。
`LieBeliefFilter.predict_correct_kernel` 接受完整 `LieState`、`covariance_sqrt`、
已限幅动作、context、测量及预先准备的传感器标准差，返回下一步 mean、因子和
与 batch 同形状的设备端 boolean `invalid`。在线滤波包装和训练入口共用
`_prediction_kernel` / `_correction_kernel`；平方根传播、Jacobian、三角求解、
Joseph/reset 公式及其一阶、二阶梯度保持不变。

[`imagined_rollout.py`](../src/supermimic/imagined_rollout.py) 分为两个边界：

- `prepare_rollout`：校验 starts、模型/replay/reference 身份、完整初始后验、
  context、历史动作和固定噪声；复制 detached 初始张量，准备 dtype/device 常量和
  H+1 个 clip-clamped reference 窗口。准备数据的内存按 microbatch 分配。
- `observe` / `imagine_step`：只读模型及固定配置，显式接收/返回 `ImaginedState`。
  动态失败标志在 GPU 上合并，不做 reset/live 分流、episode/time 连续性查询、
  generation 更新、pending transition 或 provider 状态写入。

actor/critic 特征编码也共用无主机数值判断的张量函数。初始后验因子已在边界验证，
后续因子由 QR 产生，RPV/task covariance 是其线性映射，因此训练无需逐步重新进行
特征协方差的谱分解。设备端检查仍覆盖有限性、旋转有效性、因子正对角及目标；
外部输入的公开校验函数继续拒绝非法协方差。

`MultiEncoderAE.forward_with_aux` 显式返回 action 和附加损失。训练不读取或替换
actor 的 `_aux_loss`，也不把最终后验写回 provider。`RolloutResult` 返回 returns、
critic 数据、metrics、附加损失、最终状态和 invalid；需要检查最终后验的诊断改读
`result.final_state.belief`。旧 `_rollout` tuple 接口作为带边界检查的诊断包装保留。
恢复验证复制 learner 时，同时重新绑定 actor 的两个 forward 入口。

奖励/termination 仍用推进前的 reference 对推进后的 latent 评分，然后构造下一帧
观测。history 保持 oldest-first，动作只限幅一次使用，尾帧保持、soft survival、
bootstrap、GAE、microbatch 加权和 replay 初始 posterior 不重复融合的语义不变。

### CUDA capture 边界

生产训练已默认使用张量路径。后续已接入显式启用的 actor graph 缓存，使用方式及
固定缓冲区生命周期见 [EIT CUDA Graph Replay](eit_cuda_graphs.md)。optimizer/critic
update 仍在图外。可捕获部分为：

```python
# 图外，模型/reference/config 与 normalization 在本次 update 内只读。
prepared = prepare_rollout(env, start, horizon, process_noise, measurement_noise)

# 固定尺寸的图内入口，噪声为 (H, B, 70) / (H, B, 64)。
result = learner._rollout_kernel(prepared, process_noise, measurement_noise, horizon)
loss = -result.returns.sum() / total_samples
loss = loss + result.auxiliary_loss * microbatch_samples / total_samples
loss.backward()

# 图外：读取 invalid、复核模型/reference 身份与梯度，再允许 optimizer.step。
```

中间后验、latent、history 都保持 autograd 联系。输入噪声和准备数据的 backing
storage 在该次 forward/backward 生命周期内必须保持只读；replay 更换输入时要复制
到已捕获的固定地址，不能依赖 Python 对象替换。准备、版本检查、模式切换、日志、
梯度裁剪、优化器和计数器仍在图外；batch/horizon/配置变化需另行准备/捕获。

原路径每个 microbatch 返回后读取设备失败状态；replay 路径在 GPU 上合并标志，
在整次 actor update 的 optimizer 前检查。任一步失败会拒绝本次
actor update 并清空累计梯度；不会推进 actor/critic optimizer 或更新步数。
在线 `reset/step/snapshot/restore` 继续使用原有事务和相同动作/噪声重试规则。

### 验证与性能

验证覆盖 A/B/C、vmap/increment、逐帧后验/观测/reward/termination、actor 梯度、
初始 mean/因子和模型参数的一阶/二阶梯度、重复 starts、不同 clip、尾帧、尾部
microbatch、失败拒绝、provider 状态隔离及恢复验证。CUDA 测试对 batch 2/1、H=2
执行前向与反向捕获，修改起点和噪声后连续 replay，并检查与 eager 结果一致。
C/increment 额外验证强制 eager 计算及固定梯度缓冲区 `backward()` 的 capture。Profiler 检查整个张量路径不包含
`aten::item`、`aten::_local_scalar_dense`、`aten::nonzero`、
`aten::_linalg_check_errors`。

固定噪声的生产网络性能比较可用以下命令复现：

```bash
CUDA_VISIBLE_DEVICES=5 PYTHONPATH=src:. python docs/validation/benchmark_square_root_filter.py --compare-rollout --batch 1024 --horizon 32 --repeats 3 --output /tmp/tensor-rollout-benchmark.json
```

比较三条路径：保留的事务展开、包含准备阶段的新训练展开、已准备输入的张量展开；
都包含 actor 反向，不包含优化器和 critic 拟合。所有路径使用同一模型、starts 和
噪声，先预热再取三次中位数；不把编译时间算入计时。


2026-11-24 在 PyTorch 2.14.0+cu130 / NVIDIA RPBZZZ6 上的独立复测结果：

| 路径（batch 1024，H=32，生产网络） | 前向及反向中位数 | 峰值 allocated 显存 |
|---|---:|---:|
| 事务展开 | 4637 ms | 51.46 GiB |
| 新路径，包含准备阶段 | 3905 ms | 50.96 GiB |
| 新路径，输入已准备 | 3757 ms | 50.96 GiB |

包含准备阶段的耗时减少约 15.8%；只计已准备张量展开时减少约 19.0%。
这些数字是常规调用张量路径的耗时，不是 CUDA Graph replay 的吞吐率。
原始三次计时、运行配置和显存见
[验证记录](validation/tensor_rollout_float32.json)。

本次分批验证了 614 项相关回归用例及六组 CUDA capture 用例；C/increment
还验证了 eager capture 和使用固定梯度缓冲区的实际 `loss.backward()`，连续
replay 的梯度不会累加。Ruff 与 `git diff --check` 通过。

验证环境的既有问题单独记录：

- `test_first_publication_can_retry_in_its_own_interrupted_output` 与先前启动过 W&B
  的测试同进程运行时失败；未修改的 HEAD 同样复现，独立进程运行通过。本次未修改
  无关的 W&B 生命周期代码。
- Pyright 未作为通过项：远端缺少可用 Node 运行时；本地同依赖配置对未修改 HEAD
  的相关文件也报告大量既有类型错误，因此不能报告完整类型检查通过。
