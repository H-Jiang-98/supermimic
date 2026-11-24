# EIT Actor CUDA Graph Replay

EIT 支持显式启用 imagined rollout 与 actor backward 的 CUDA graph replay。
默认关闭；训练数学、随机流和 checkpoint 配置版本不变。

## 启用

在已有 `train_eit` 命令后添加 `--policy-cuda-graphs`，或修改启动配置：

```toml
[training]
eager = false
policy_cuda_graphs = true
```

`--eager` 优先关闭此功能。`TORCH_COMPILE_DISABLE` 等变量只控制 Inductor，
显式 CUDA graph 也可捕获未编译张量运算。恢复 checkpoint 时仍由本次运行的
开关决定是否启用，不要求重新生成初始化或训练文件。

首阶段支持生产 belief rollout 的 CUDA A/B/C、increment/vmap，以及
`batch_size * num_samples` 可被 `microbatch_size` 整除的配置。
当前 4096 starts、1 sample、1024 microbatch 可直接使用。CPU、自定义环境、
eager Jacobian 或尾批走原有路径；尾批回退整个 actor update，并释放旧图缓存。
不要改动原有 `runner.cuda_graphs`：它控制的是另一套完整 learner capture。

## 执行与所有权

图外准备、校验 starts/噪声及 H+1 帧 reference，随后复制到固定地址。
每个 microbatch 捕获以下计算：actor forward、完整平方根滤波和动力学展开、
reward/soft survival、critic bootstrap、GAE、原比例 loss 和 actor backward。
actor `.grad` 在一次逻辑更新开始时清零，microbatch 之间累积，存储地址不变。

前三个真实 microbatch 在侧流预热；第四个捕获并 replay，此后直接 replay。
预热不会额外消耗 RNG 或执行 optimizer。缓存只保留 detached outputs，
critic 样本在下一次 replay 前复制，指标和失败标志在 GPU 上累积。
graph 使用独立 stream/private pool，不共享原 learner 的全局 pool。
同一 learner 在恢复、重捕获及尾批回退时复用其 actor stream，避免仍存活的
AccumulateGrad 节点与新的 backward 使用不同流。

全部 microbatch 完成后，在图外检查失败标志、模型/replay/reference 身份与
梯度，再执行裁剪和 actor optimizer。critic 拟合、随机排列和步数管理仍在图外。
actor 阶段失败会清空梯度、丢弃缓存，且不会推进 actor/critic optimizer；
critic 阶段沿用原有更新与异常语义。

参数、normalizer 和常量原地更新可被下一次 replay 读取。缓存检查依赖张量
和梯度地址，意外替换存储会在 replay 前报错；配置改变或成功加载 checkpoint
会重新预热。失败的 checkpoint 校验不改变缓存。恢复验证的私有副本显式使用
eager，避免复用在线图。图对象、静态输入和计数器不进入 policy checkpoint 合约。

## 观测与复现

`profiling.jsonl` 的 `runtime.policy_cuda_graph` 包含请求开关、执行/回退原因、
预热次数、捕获次数及 replay 次数。`policy_prepare_seconds` 与
`policy_backward_seconds` 使用 stream events；原有
`policy_rollout_backward_seconds` 仍包含准备、检查及 actor optimizer。
缓存的 `warmup_seconds` 是主机提交/编译耗时；完整启动墙钟见 benchmark 的
`setup_seconds`，不要将冷启动纳入稳定态吞吐。

2026-11-24，PyTorch 2.14.0+cu130 / NVIDIA RPBZZZ6，C 组、increment、生产
网络、batch=4096、microbatch=1024、H=32、固定 stationary 输入，十次中位数：

| 项目 | Inductor、关闭 replay | Inductor、开启 replay |
|---|---:|---:|
| 完整 policy update | 19.497 s | 8.250 s |
| rollout + backward 阶段 | 18.936 s | 7.720 s |
| 峰值 allocated（含预热/捕获） | 50.82 GiB | 50.96 GiB |
| 稳定阶段峰值 reserved | 53.66 GiB | 54.26 GiB |

完整 policy update 加速 2.36 倍，耗时减少 57.7%。原始数据见
[十次计时与显存记录](validation/eit_actor_graph_float32.json)。图只捕获一次，
捕获耗时约 7.03 s，含预热更新共执行 45 次 replay。图私有池会保留中间张量的
地址，即使 steady-state allocated 计数较小，也不能据此估计实际显存需求；
应同时看含捕获的 allocated 峰值与 reserved。

基线先运行，随后 replay 路径复用已生成的编译缓存，因此两条路径的
`setup_seconds` 不用于冷启动优劣比较。这里测量 policy update，未计入真实物理
采集和 world model 拟合，不能将 2.36 倍直接当作完整训练吞吐提升。

在仓库根目录运行固定输入对照（设备按实际空闲情况选择）：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MUJOCO_GL=egl PYTHONPATH=src:. \
  .venv/bin/python docs/validation/benchmark_square_root_filter.py \
  --compare-policy-graph --batch 4096 --microbatch 1024 --horizon 32 \
  --repeats 10 --output /tmp/eit-policy-graph.json
```

两条路径使用相同初始化、starts、噪声和 critic RNG，分别预热后取十次中位数。
输出包含完整 policy update、各阶段时间、启动成本以及 allocated/reserved 显存。
`--small --batch 8 --microbatch 4 --horizon 2` 可作快速脚本检查。

回归覆盖 A/B/C × increment/vmap、多 microbatch 与多 sample、连续 replay、
输入及原地权重/normalizer 变化、MuON/Adam 状态、编译路径、恢复、尾批、
失败拒绝和 CPU 回退。后续可独立推进 actor optimizer、critic 及模型训练的
capture；尾批图池复用不属于本阶段。
