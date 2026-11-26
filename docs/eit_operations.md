# EIT 运行、日志与诊断

[训练教程](../README_eit.md) · [训练实现](eit_training.md) · [批量评估](eit_batch.md)

## 日志、保存与离线 W&B

`train_eit` 与 SuperMimic 共用 `BaseRunner._init_wandb`、`Logger.log`、`log_model_artifact` 和结束清理。未设置时默认 `WANDB_MODE=offline`；测试可设置 `WANDB_MODE=disabled`，当前 EIT 不开放在线模式。

```bash
WANDB_MODE=offline OMP_NUM_THREADS=1 .venv/bin/python -m supermimic.scripts.train_eit \
  --motion-file data/motions/walk1_subject1.npz --device cuda:0 \
  --group belief --output logs/eit-v7/belief-direct --max-iterations 1000 \
  --wandb-project supermimic --run-name eit-belief-direct-seed2026
```

项目和名称写入公共 runner 配置。`--output` 为运行目录；`--log-interval` 默认 10，`--save-interval` 默认 1000，二者均以累计物理采集步为单位。旧的 `--wandb-mode`、`--wandb-log-models`、`--wandb-group` 和固定 run ID 管理已移除。

每次启动/恢复产生新 run。其 config 包含 `runner_cfg`、`eit_loop`、`eit_policy` 和 `continuation`：来源 checkpoint、起始累计步/帧、父 run 的 ID/名称/项目。终端 Learning iteration 与 W&B step 使用本次 run 的日志序号，从 0 开始；续训的新 run 也从 0 开始。分母包含本次运行的周期日志和结束补记。JSONL 的 step 仍为累计零基物理步，用于恢复和报表。恢复后重新 warmup，但累计计数不归零。

记录包含：

- `Counters/*`：累计物理步、帧、训练组、跳过机会及各优化器逻辑更新次数。
- `Budget/*`：日志区间实际 mean 窗口、covariance 抽样/更新、policy 起点及想象转移数；可用转移与缓存大小为最近一次测量。
- `Model/*`、`ModelValidation/*`、`Policy/*`、`Grad/*`：训练、留出与策略指标。
- `Perf/*`：各阶段耗时、公共 Logger 吞吐和 CUDA peak。子阶段时间不可重复相加。

保存入口在初始、周期和结束调用。周期为 `step % save_interval == 0`，例如 interval=1000 对应已采 1、1001、2001……步。每次保存将 `world_model.pt`、`actor.pt`、带契约的 `critic.pt` 和完整 `training.pt` 作为同一个 `model-<run ID>` artifact。写入新版本前等待上一份后台归档完成。四个文件先写入暂存目录，原子替换 `training.pt` 后发布其余权重。发布中断时，恢复流程在训练目录锁内完成已提交的文件替换。

每个有新采集步的保存点也保留当前日志区间的预算、耗时和累计计数。若进程在保存和写日志之间中断，恢复会补齐 `metrics.jsonl` 与 `profiling.jsonl`。即使日志周期和保存周期不同，恢复也能保留截至该 checkpoint 的最后一段数据。

离线模式只暂存 artifact，不查询或清理在线 registry。disabled 仍写本地四个文件，但不创建 artifact。没有 EIT 专用 W&B 管理器、`wandb_run.json` 或 `snapshots/index.json`。

W&B 的 `wandb/offline-run-*`、`wandb-data` 与 `wandb-cache` 会使用磁盘；后两者可由对应 `WANDB_DATA_DIR`、`WANDB_CACHE_DIR` 环境变量指定。离线版本内容不会因覆盖本地权重而消失。先用实际保存点测量容量，再估算长跑存储。日志不含独立评估结果；评估仍使用 `eval_eit`/`batch_eval_eit`。

需要上传时，由用户在有网络的环境使用已安装 W&B CLI 同步各个 `offline-run-*` 目录。每个启动段是独立 run，保留其父 run 信息，不使用 append 或固定 ID 合并。

### 核心日志对齐

终端默认只展开均值模型 Loss/state、状态分组 Loss/*、
Policy/reward、Policy/grad_norm 和 Policy/value_function。
公共 Logger 的 Mean reward、Mean episode length、环境 extras、步数和性能摘要仍显示。
EIT 专用验证指标、Counters、Budget 和细分诊断继续写入 W&B 和现有 JSONL，
原字段保留以兼容报表。未更新的指标沿用最近一次结果；首次更新前不填零。

| 核心字段 | 来源 |
| --- | --- |
| Loss/state 和状态分组 Loss/* | 相应 Model/Loss/* |
| Policy/reward | Policy/return / (policy_horizon × dt) |
| Policy/grad_norm | Grad/actor_norm |
| Policy/value_function | Policy/critic_loss |

Policy/reward 与普通模式使用同样的 return 归一化；训练目标和奖励函数不变。
Policy/return、Policy/raw_reward 等原始指标仍保留。

物理步触发条件仍是 step % log_interval == 0，结束补记独占一个新日志序号。
例如 interval=2、训练到累计 4 步，JSONL step 为 0、2、3，
终端和 W&B 日志序号为 0、1、2。同目录续训到 6 步时，
JSONL 追加 4、5，新 run 日志序号为 0、1。

性能计时范围保持原样：公共 FPS 以采集、模型和策略阶段计时之和计算，
不代表包含所有保存等开销的端到端吞吐。JSONL 仍不是完整 W&B 副本，
公共 Logger 添加的 episode 指标、Train/* 和 total_fps 不补入 metrics.jsonl。
历史日志与 checkpoint 格式不变；新展示规则在下次启动或续训时生效。

## 短程训练与恢复验证

四方法小批验证使用静止参考片段和 4 个真实环境，各方法训练 5 步并从磁盘 checkpoint 恢复至 10 步。以下命令在选定的空闲 GPU 上运行，输出目录须为新目录。FiLM 检查在加载共享初始化之前扰动输出权重，以便观察各输入通道的响应。两个 baseline 的状态式 replay 恢复和后续推进与函数式 rollout 逐项对照。CUDA graph 对照覆盖正式配置的 residual 单样本和 direct 双样本。

```bash
WANDB_MODE=offline OMP_NUM_THREADS=1 python -m supermimic.scripts.check_eit_four_methods --output ./four-method-smoke
python -m pytest -q tests/test_estimator_eit_graph.py -k baseline
```

`check_eit_training` 用小规模 MuJoCo 环境训练 mean 和 belief 两组，核对固定输入下模型与策略的下一次更新，并在独立进程中评估保存的 checkpoint。

```bash
WANDB_MODE=disabled OMP_NUM_THREADS=1 .venv/bin/python -m supermimic.scripts.check_eit_training --device cpu
```

历史 A/B/C pilot 的归档报告仍可用 `report_eit_pilot` 审阅。

## 历史实验的固定轨迹诊断

在独立评估 seed 的真实仿真中，每个环境槽从片头记录一个 episode。相同的过程模型或滤波器比较应复用同一份 recording 和窗口长度。critic 的真实回报比较则要求由被评估的快照产生动作。训练集内部的验证划分继续用于模型拟合。评分入口用展开的训练文件 SHA-256、加载的 motion 摘要及训练 seed 检查记录来源。与训练共享 seed 和 motion 的记录会被拒绝。历史快照缺少训练 seed 或同 seed 时缺少文件清单，会在 training_overlap_check 中标明无法核验。

诊断支持 `train_belief` 的 v4 模型快照和 `train_eit` 的 v6 策略快照。两类诊断都报告累计采集帧数 `collected_frames`。模型快照另报 `round`，策略快照另报 `collected_steps` 和 `update_groups`。固定记录保存完整 episode 的逐帧测量、动作和初始先验，后验评分以指定快照的模型逐帧滤波。训练时的 `posterior_save_stride` 不影响诊断帧数。

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m supermimic.scripts.eval_eit \
  --checkpoint logs/eit-v6/B/training.pt --motion-file data/motions/walk1_subject1.npz \
  --seed 2718 --num-envs 1 --record artifacts/eit-v6-diagnostics/episodes.pt \
  --output artifacts/eit-v6-diagnostics/evaluation.json

OMP_NUM_THREADS=1 .venv/bin/python -m supermimic.scripts.diagnose_eit \
  --checkpoint logs/eit-v6/B/training.pt --recording artifacts/eit-v6-diagnostics/episodes.pt \
  --window-frames 8 --bin-steps 50 --imagined-horizon 128 --imagined-samples 4 \
  --output artifacts/eit-v6-diagnostics/diagnostics.json
```

`--mode process` 读取片内全部具有连续后继的一步对和全部完整窗口，按动作从第一个真值状态自由滚动。四组多步 MSE 的通道和缩放与均值拟合一致。一步有符号偏差为 local_error(预测, 真值) 的前 9 维。白化残差的二阶矩为完整的 70×70 矩阵，保留维度间的相关。JSON 同时给出窗口数、被窗口评分的后继总数和一步对数。

`--mode posterior` 从实际采样的初始先验开始，消费每个真实传感帧一次。时间分箱使用动作前的右局部误差，位置和速度 RMSE 分别取世界系下后验与真值之差的均方范数开方，另列右局部 ρ 和 ν 的 RMSE。95% 覆盖率使用同一时刻的 RPV9 边缘矩阵、有效秩及其零方差方向；先按各维标准差归一化，再判断有效秩，避免量纲和方差尺度差异造成误判。零方差坐标的任何非零误差均计为支持集违规；支持集外误差的投影 Mahalanobis 距离在标准化空间计算，覆盖判定始终为否。每个时间箱给出帧数、存活 episode 数、innovation。此模式可在相同 recording 上比较不同模型。

`--mode critic` 加载冻结的 critic 及归一化，使用同帧的真值、对齐目标、五帧实测历史、ξ 及重建的后验。每帧的原始 V、用于 bootstrap 的 V 及到终止的真实折扣回报保存在 `critic.episodes`。误差符号为 V − 真实折扣回报。物理失败的回报包含终止步奖励。片尾、普通超时和外部步数截断单列 episode 数、样本数及逐帧 V，回报为空。A 的训练输入策略可在录制时使用 `--truth-actor`，默认部署输入策略使用后验。两种 recording 分别评分。B/C 仅接受各自快照的后验输入策略。启用值截断时，优先读取快照中的 bootstrap 上限；缺少该值时，使用记录的仿真每步奖励上限除以 (1 − γ)。

`--mode all` 默认运行上述评分，对 model-stage 仅计算过程模型和后验。添加 `--imagined-horizon` 后，在每个物理 episode 首帧以同一真值、先验校正后的后验、实测历史、ξ 和参考对齐启动采样。想象采用训练输入和 soft survival，按 γ 累计给定时域内的奖励，片尾重复完整参考末帧并继续计分。JSON 给出每个样本的回报、均值、时域末剩余折扣权重、critic 与模型之差，以及有完整硬终止回报时模型与实测之差。对相同行为策略，程序另报相同时域的物理折扣回报及模型前缀误差。若真实轨迹在该时域内物理终止，之后奖励按零计。提前删失使未观测的前缀为空。想象时域末仍有样本保留正的折扣存活权重时，完整模型回报误差和 critic 对完整模型回报的误差均为空。A 的想象使用真值输入，与后验部署策略回报并列时，跨策略模型误差为空。记录迁移后可用 `--motion-file` 指定原始 motion 的新路径，同时核对文件 SHA-256 和加载内容摘要。

### 固定后验漂移样例

本地样例使用 G1 仰卧静态参考，根位置为 (0, 0, 0.12) m，wxyz 根四元数为 (√0.5, 0, −√0.5, 0)，关节角为 mjlab 的 KNEES_BENT_KEYFRAME，参考速度全零，以 50 Hz 重复 250 帧。真实 MuJoCo 的关节位置伺服追踪参考，沿用评估环境的 reset、域随机化和物理终止条件。这一受地面支撑的姿态提供 5 s 连续记录，单独检验短闭环失败后的滤波漂移。轨迹的行为来源标为 `reference_joint_positions`，可进入固定记录的后验和过程评分。

```bash
.venv/bin/python - <<'PYTHON'
from pathlib import Path
import re
import mujoco
import numpy as np
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML, KNEES_BENT_KEYFRAME
from supermimic.scripts.check_estimator import stationary_motion

path = Path("artifacts/eit-v6-diagnostics/drift_supine.npz")
path.parent.mkdir(parents=True, exist_ok=True)
stationary_motion(path, frames=250)
model = mujoco.MjModel.from_xml_path(str(G1_XML))
with np.load(path) as motion:
    arrays = dict(motion)
arrays["root_pos_w"][:] = (0, 0, 0.12)
arrays["root_quat_w"][:] = (2**-0.5, 0, -(2**-0.5), 0)
hinges = np.flatnonzero(model.jnt_type == int(mujoco.mjtJoint.mjJNT_HINGE))
for column, joint_id in enumerate(hinges):
    for pattern, angle in KNEES_BENT_KEYFRAME.joint_pos.items():
        if re.fullmatch(pattern, model.joint(joint_id).name):
            arrays["joint_pos"][:, column] = angle
np.savez(path, **arrays)
PYTHON

OMP_NUM_THREADS=1 .venv/bin/python -m supermimic.scripts.eval_eit \
  --checkpoint logs/eit-v6/B/training.pt --motion-file artifacts/eit-v6-diagnostics/drift_supine.npz \
  --num-envs 1 --seed 90817 --reference-controller --record artifacts/eit-v6-diagnostics/drift_episodes.pt \
  --output artifacts/eit-v6-diagnostics/drift_recording.json

OMP_NUM_THREADS=1 .venv/bin/python -m supermimic.scripts.diagnose_eit \
  --checkpoint logs/eit-v6/B/training.pt --recording artifacts/eit-v6-diagnostics/drift_episodes.pt \
  --mode posterior --bin-steps 50 --output artifacts/eit-v6-diagnostics/drift_diagnostics.json
```

