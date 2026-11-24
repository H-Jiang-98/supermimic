# EIT Actor–Critic CUDA Graph Replay

`policy_cuda_graphs` 将一次完整 policy update 捕获为一个 CUDA Graph：
全部 actor microbatch rollout/backward → actor 裁剪与 Adam/Muon 更新 →
全部 critic epochs/minibatches 的 forward/backward、裁剪与 Adam/Muon 更新。
稳定态每个 policy update 只 replay 一次。world model 拟合仍在图外。

## 启用和适用范围

```toml
[training]
eager = false
policy_cuda_graphs = true
```

命令行使用 `--policy-cuda-graphs`，默认关闭；`--eager` 优先关闭。
`TORCH_COMPILE_DISABLE` 只控制 Inductor，显式 CUDA graph 仍可捕获张量计算。
恢复 checkpoint 时由本次运行开关决定，不改变 checkpoint 格式。
原有 `runner.cuda_graphs` 控制另一套 learner capture，不用于 EIT。

支持生产 belief rollout 的 CUDA A/B/C、increment/vmap，要求
`batch_size * num_samples` 可被 actor `microbatch_size` 整除。
CPU、自定义环境、不支持的 Jacobian backend 和 actor 尾批整次回退 eager。
critic 保持可整除前缀、小 batch 缩减 minibatch 数，以及内部 microbatch 尾块语义。

## 单图边界与随机性

图外校验 starts/噪声、准备全部 microbatch 的 H+1 帧 reference，然后复制到
固定地址。critic permutation 在图外生成，每个 policy update 一份，跨 epochs
复用；采样、process、measurement 和 critic 的独立 RNG 流保持原语义。
单样本仍绕过 process sampling，不推进 process RNG。

图内保留原 loss 比例、actor 梯度累积、bootstrap/GAE、time-major critic 样本顺序、
裁剪顺序和更新次数。解释方差使用张量条件表达式；零 target 方差仍返回 NaN。
图中只有 GPU 状态标志和统计输出，没有 host 数值检查或 optimizer 提交门控。

前三次**完整 policy update** 在持久侧流上真实预热；第四次捕获并 replay，之后
每次只 replay 一次。预热没有额外更新或随机抽样。actor/critic 梯度地址固定，
optimizer moments 在真实预热中初始化。图使用私有 pool；恢复、重捕获和 eager
回退都沿用该 learner 的 stream。返回指标复制到独立存储，不被后续 replay 覆写。

## 校验和失败语义

输入、配置及依赖存储校验在 replay 前执行。rollout invalid、loss、原始梯度范数
及 critic 最终预测的异常在**整图执行后**检查。异常时参数和 optimizer 可能已经
改变，不做回滚，不自动重试 eager；清空梯度和缓存并向上报错。
已完成图执行的 actor/critic 计数在结果检查前推进。
关闭 graph 的 eager 路径保留原有更新前检查。

参数、normalizer 和 tensor LR 的原地更新对 replay 可见；Python 超参数改变触发
重捕获，意外替换参数、buffer、梯度或 optimizer 状态存储会在 replay 前报错。
成功加载 checkpoint 清理图，无效 checkpoint 的预检不破坏缓存。
图、静态输入及捕获计数不进入 checkpoint；`clear_actor_graph()` 保留为清除完整
policy 图的兼容入口。

## 观测与性能验证

`profiling.jsonl` 的 `runtime.policy_cuda_graph` 增加 `scope: "actor_critic"`，
保留请求/执行原因以及 warmup、capture、replay 计数。预热与 replay 计数现在以
完整 policy update 为单位，不再以 actor microbatch 为单位。

单图路径提供 `policy_prepare_seconds`、`policy_update_seconds` 和
`policy_check_seconds`，完整 update 计时包含输入复制、图执行及首次捕获成本。
不再把图中一次性的 Python 时间报告为 actor/critic 稳定态分段耗时。
eager 路径继续提供原有 actor/critic 分段时间。

在空闲 GPU 上测量固定输入生产配置：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MUJOCO_GL=egl PYTHONPATH=src:. \
  .venv/bin/python docs/validation/benchmark_square_root_filter.py \
  --compare-policy-graph --batch 4096 --microbatch 1024 --horizon 32 \
  --repeats 10 --output /tmp/eit-policy-graph.json
```

脚本两条路径使用相同初始化和输入，各进行四次完整更新预热，然后取十次中位数。
结果包括启动成本与 allocated/reserved 显存；整图捕获会保留各阶段的中间存储，
应同时检查捕获峰值和稳定态 reserved，不能仅看 steady-state allocated。
`--small --batch 8 --microbatch 4 --horizon 2` 可作快速验证。

旧 actor-only 测量保留在 `validation/eit_actor_graph_float32.json`；它只覆盖
rollout/backward，不能代表当前单图实现。本轮对照以实施前工作区快照为基线。

## 本轮生产配置结果（2026-11-25）

同一 GPU、相同初始化/输入/RNG、各四次完整更新预热，十次计时中位数：

| 指标 | 原 actor-only replay | actor–critic 单图 |
|---|---:|---:|
| 完整 policy update | 8.086 s | 8.044 s |
| 含预热/捕获的 peak allocated | 49.60 GiB | 50.57 GiB |
| 稳定阶段 peak reserved | 53.11 GiB | 54.01 GiB |
| 单次 capture 主机耗时 | 6.11 s | 20.69 s |

本轮增量耗时下降约 0.5%，整体基本持平；完成了单图边界，未观察到显著的
额外吞吐提升。单图启动/捕获成本更高，稳定阶段 reserved 增加约 0.91 GiB。
预热和十次测量期间单图只捕获一次、replay 11 次；旧路径按 microbatch replay 53 次。
另一次关闭 replay 的测量为 15.614 s/update，但采用两次预热，仅作为辅助对照。
完整逐次计时、环境信息和源码摘要见
[本轮验证记录](validation/eit_policy_graph_float32.json)。
