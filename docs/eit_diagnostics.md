# EIT 固定轨迹诊断

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

`--mode posterior` 从实际采样的初始先验开始，消费每个真实传感帧一次。时间分箱使用动作前的右局部误差，位置和速度 RMSE 分别取世界系下后验与真值之差的均方范数开方，另列右局部 ρ 和 ν 的 RMSE。95% 覆盖率使用同一时刻的 RPV9 边缘矩阵、有效秩及其零方差方向；先按各维标准差归一化，再判断有效秩，避免量纲和方差尺度差异造成误判。零方差坐标的任何非零误差均计为支持集违规；支持集外误差的投影 Mahalanobis 距离在标准化空间计算，覆盖判定始终为否。每个时间箱给出帧数、存活 episode 数、innovation 和 jitter。此模式可在相同 recording 上比较不同模型。

`--mode critic` 加载冻结的 critic 及归一化，使用同帧的真值、对齐目标、五帧实测历史、ξ 及重建的后验。每帧的原始 V、用于 bootstrap 的 V 及到终止的真实折扣回报保存在 `critic.episodes`。误差符号为 V − 真实折扣回报。物理失败的回报包含终止步奖励。片尾、普通超时和外部步数截断单列 episode 数、样本数及逐帧 V，回报为空。A 的训练输入策略可在录制时使用 `--truth-actor`，默认部署输入策略使用后验。两种 recording 分别评分。B/C 仅接受各自快照的后验输入策略。启用值截断时，优先读取快照中的 bootstrap 上限；缺少该值时，使用记录的仿真每步奖励上限除以 (1 − γ)。

`--mode all` 默认运行上述评分，对 model-stage 仅计算过程模型和后验。添加 `--imagined-horizon` 后，在每个物理 episode 首帧以同一真值、先验校正后的后验、实测历史、ξ 和参考对齐启动采样。想象采用训练输入和 soft survival，按 γ 累计给定时域内的奖励，片尾重复完整参考末帧并继续计分。JSON 给出每个样本的回报、均值、时域末剩余折扣权重、critic 与模型之差，以及有完整硬终止回报时模型与实测之差。对相同行为策略，程序另报相同时域的物理折扣回报及模型前缀误差。若真实轨迹在该时域内物理终止，之后奖励按零计。提前删失使未观测的前缀为空。想象时域末仍有样本保留正的折扣存活权重时，完整模型回报误差和 critic 对完整模型回报的误差均为空。A 的想象使用真值输入，与后验部署策略回报并列时，跨策略模型误差为空。记录迁移后可用 `--motion-file` 指定原始 motion 的新路径，同时核对文件 SHA-256 和加载内容摘要。

## 固定后验漂移样例

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

历史 B 组快照在五个一秒箱中各有 50 个样本，RPV9 椭球分别覆盖 2、0、0、0、0 帧。逐帧原始记录、motion、快照与完整诊断 JSON 位于本地 `artifacts/step5/`。
