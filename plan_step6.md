# Step 6　单 seed 试验与算力预算

本文记录原始轮次协议的历史实验；其预算、快照及性能结果不适用于当前 `eit_training_v5`。当前 pilot 使用累计物理步和公共保存流程，见 [运行协议](docs/eit_pilot.md) 与 [训练说明](docs/train_eit_pro6000.md)。

使用 Step 5 的整段评估协议、不可变快照和独立诊断，完成 A/B/C 各 10 轮真实 walking 训练及第 5 轮后的新物理轨迹续训。原始试验用于锁定预算，修订后的恢复验收要求第 5 轮训练进程保存独立参考。当前控制效果和在线时延未达部署要求。[评估协议](plan_step5.md) · [总体规划](plan_sketch.md)

## 试验配置与预算

用真实 walking motion 从同一初始化包运行 A/B/C，训练沿用 uniform reset，独立闭环评估从片头开始。排错规模为每组 8 环境 × 64 步/轮，buffer 128，后验保存间隔默认 1。均值 10 批 × 2048 个 8 帧窗口，covariance 每批最多 256 对、遍历一遍。策略每轮 1 次 actor 更新、2 个 critic epoch，N=4、K=2、H=4、microbatch=2。网络沿用默认宽度，均值 (2048,2048,2048)、噪声头 (128,64)、actor (2048,1024,512,256)、critic (256,128)。先跑 10 轮并实际恢复一次，用 Step 5 的独立验证场景评估起点及各快照。10 轮对应每组 5,120 个实采帧、10 次 actor 更新。同一进程跨轮保留物理环境、后验和观测历史。长时行为以独立闭环和连续记录衡量。[轮次调度](src/supermimic/runners/eit_training.py)

对齐**外部预算和采样规则**。固定实采帧数、均值更新数、actor/critic 更新日程，以及 covariance 的 batch size 和 epochs。每轮实际 covariance 步数是 epochs × ⌈有效训练对数 / batch size⌉，不同失败时间会改变可用相邻帧数。逐组记录可用与实际抽取窗口、有效一步对、实际四类更新数、有效想象步数、终止频率和耗时。若无完整训练窗口或无有效验证转移，记录失败原因，再统一修订 pilot 配置并复跑受影响的比较。

## 闭环诊断与性能测量

把真实回报/进度与想象回报、后验校准、动作限幅率、FiLM 尺度、critic 的独立误差并列作图。已有 `Policy/bootstrap_fraction` 是 bootstrap 对总回报的数值比例。若独立 V 误差大或 bootstrap 主导目标，在固定起点和噪声上分别求奖励项与 bootstrap 项的 actor 梯度，报告范数及夹角，再决定是否统一调整策略 H 或数据预算。校准异常时先用 Step 5.3 的固定轨迹定位均值偏差、过程噪声或滤波误差，再决定是否开展单独方法实验。

分别计时物理采集、均值更新、残差提取与 covariance 拟合、后验缓存读取、策略 rollout/反传、critic 回归、保存及整轮。记录后验保存间隔、可用起点数、归一化样本数、来源修订号差距和实际存储字节。在线用 batch=1 测量完整滤波、特征、actor 和限幅的均值与 p95/p99，单列首次 reset，并与 20 ms 控制周期对照。保留 N/K/H、microbatch、dtype、预热、CPU 线程、CUDA 同步和显存峰值/稳定占用。[规模检查](src/supermimic/scripts/check_eit_scale.py) · [训练预算与指标](README.md#estimator-第四步完整策略与模型交替训练)

## 验收与交付

进入 Step 7 前，三组应有可读取的快照、可核对的预算、无预算截断的验证结果和独立校准/价值评分。以固定输入验证恢复后的下一次更新，并以新物理轨迹验证续训。记录 v4 默认宽度的 GPU 联合训练和目标设备在线时延。依据 pilot 的学习曲线、失败类型及实测成本，预先锁定正式的统一预算。[下一步的多 seed 消融](plan_step7.md)

## 锁定预算所用的原始试验（v3）

- 训练 seed 2026，`walk1_subject1.npz`，RTX 5090，float32，单 CPU 线程，使用上文默认宽度和采样配置。每组 5,120 实采帧、100 mean / 10 actor / 20 critic 更新；covariance 实际为 31 / 31 / 32 次。未出现缺少完整训练窗口或验证转移的轮次。
- 三组共享同一 v3 初始化指纹，保留 33 个不可变快照。原始试验从第 5 轮快照在新进程续训至第 10 轮。原验收的两条分支都从同一快照恢复，因此它只测到了重复加载的一致性。
- 验证 seed 17031，每快照 2 个首 episode，共 66 个，无外部预算截断，全部真实失败，完成率均为 0。最后快照的平均长度 A/B/C 为 32 / 29 / 34 步。不能据此宣称方法有效或 C 优于 B。
- 每个快照有独立后验和 critic 评分；最终快照另有奖励/bootstrap 梯度分解、A 真值输入 critic、两份共同 walking 记录及 250 帧连续仰卧记录的跨模型评分。固定片头上 bootstrap/reward 梯度范数比为 2.41% / 2.31% / 1.91%，虽训练回报中 bootstrap 数值占比为 61.8% / 73.4% / 55.0%，这些片头上未见其主导梯度。
- RTX 5090 上 A/B/C 的平均轮次耗时分别为 20.90 / 21.11 / 21.23 s。Ryzen 9950X 单线程 batch=1 的完整连续在线路径约 49.8 / 50.1 / 50.7 ms，超过 20 ms 控制周期。
- 正式预算锁定为 5 个新 seed、各组 100 轮，即每组 51,200 帧、1,000 mean / 100 actor / 200 critic 更新，H=4 保持不变；covariance 仍按有效对数和统一 batch/epoch 规则确定。每 10 轮列为验证候选，保留每轮存档。纯训练预计约 8.8 h，工件盘预算建议 260–300 GiB；完整 clip 成功后的评估成本可能显著增加。Step 7 尚未启动。

## 审查后的恢复复验

当前存储流程由 `runner.cfg.save_interval` 控制，按累计物理步保存最新权重和 `training.pt`，每个保存点形成一个离线 model artifact。EIT 已移除 `snapshot_interval` 及独立历史快照发布。下述历史复验的逐轮存档保留作原始证据；当前配置见 [训练说明](docs/train_eit_pro6000.md)。

在新目录 `artifacts/step6/pilot_verified/` 按同一实采及更新预算复跑 A/B/C。
第 5 轮先从训练进程内的模型及优化器生成固定输入参考，再发布快照。独立验证进程
加载快照，按记录的数值设置预热一次后，三组的下一次模型和策略更新均逐位一致。
从真实 CPU 检查点恢复时人为清零一个非零 Adam 动量，验证器检出了 actor 参数差异。
复跑的 covariance 更新数依次为 33、32、33；三个最终快照的平均验证长度为
32、31.5、33 步。锁定正式预算时使用的数据仍取自上节的原始试验。

[运行与恢复命令](docs/eit_pilot.md) · [原始试验摘要](docs/validation/eit_step6_pilot.json) · [审查复验](docs/validation/eit_step6_recheck.json) · [正式统一预算](docs/validation/eit_step6_budget.json) · [学习曲线](docs/validation/eit_step6_learning_curves.png)
