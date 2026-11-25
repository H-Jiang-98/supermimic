# SuperMimic

A [SuperTrack](https://montreal.ubisoft.com/en/supertrack-motion-tracking-for-physically-simulated-characters-using-supervised-learning/)
implementation for training world models and policies to mimic motion data, built on [mjlab_for_supermimic](https://github.com/H-Jiang-98/mjlab_for_supermimic).

文档入口：[EIT v6 训练](README_eit.md) · [评估与可视化](README_eval.md) · [环境安装](docs/setup.md) · [普通训练 W&B](docs/wandb.md)

## 快速开始：LAFAN1 G1 数据处理与训练

**先完成 [环境安装](docs/setup.md)，再在 `supermimic/` 仓库根目录运行以下 Bash 命令。**
原始 CSV 应放在 `data/lafan1_g1/`；以下示例使用 GPU 0，并通过
`WANDB_MODE=disabled` 关闭 W&B 上传，无需登录 W&B。

### 1. 批量处理 `lafan1_g1` 数据

将所有 CSV 从 30 Hz 重采样为训练所需的 50 Hz，并保存为
`data/motions/<原文件名>.npz`。已存在且非空的 NPZ 会跳过，中断后重新运行即可
继续转换；如需重新生成某条动作，先移走对应的 NPZ。

```bash
(
  set -euo pipefail
  shopt -s nullglob
  motion_csvs=(data/lafan1_g1/*.csv)
  if ((${#motion_csvs[@]} == 0)); then
    echo "未找到 data/lafan1_g1/*.csv，请先准备数据。" >&2
    exit 1
  fi
  mkdir -p data/motions
  for csv in "${motion_csvs[@]}"; do
    motion_name=$(basename "$csv" .csv)
    motion_npz="data/motions/${motion_name}.npz"
    if [[ -s "$motion_npz" ]]; then
      echo "跳过已转换动作：$motion_npz"
      continue
    fi
    echo "转换：$csv -> $motion_npz"
    CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl WANDB_MODE=disabled \
      uv run python -m mjlab.scripts.csv_to_npz \
      --input-file "$csv" \
      --output-name "$motion_name" \
      --input-fps 30 --output-fps 50 \
      --device cuda:0 --render False
    cp /tmp/motion.npz "${motion_npz}.tmp"
    mv "${motion_npz}.tmp" "$motion_npz"
  done
)
```

当前转换器固定将结果写入 `/tmp/motion.npz`，`--output-name` 只设置 W&B
名称；因此循环中每转换一条就复制到目标目录，转换过程需串行运行。
`WANDB_MODE=disabled` 下转换器仍可能打印上传提示，但不会实际上传。

### 2. 选择单条或全部 motion files 开启训练

训练超参数统一在根目录的 [`train_config.toml`](train_config.toml) 中设置，
每项注释都给出代码默认值。文件采用较小规模的起始配置：1024 个仿真环境、
WM batch size 512、policy batch size 1024、100000 个采集步。
`[runner.wm]` 管理世界模型参数，`[runner.policy]` 管理策略参数。
这些是起始设置，尚未验证完整训练的收敛效果。

**训练单条 motion**：修改配置中的 `motion_file`，或用命令行临时覆盖。

```bash
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled \
  uv run src/supermimic/scripts/train.py SuperMimic-GenCoord-G1 \
  --config train_config.toml
```

**使用全部 motion 联合训练一个模型**：加载 `data/motions/` 下的所有 NPZ。

```bash
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled \
  uv run src/supermimic/scripts/train.py SuperMimic-GenCoord-G1 \
  --config train_config.toml \
  --motion-file 'data/motions/*.npz' \
  --runner.run-name lafan1_all
```

通配符路径必须保留引号，由程序展开。若只选择部分 motion，可传入逗号分隔的
路径，例如 `--motion-file 'data/motions/run1_subject5.npz,data/motions/walk1_subject1.npz'`。
两种训练方式都会训练世界模型和策略，并在本地保存 checkpoint。
如需最大坐标表示，将任务名替换为 `SuperMimic-MaxCoord-G1`；
首次编译耗时及短程验证参数见 [启动排查](docs/setup.md#diagnosing-startup-stalls)。

### 3. 修改超参数与短程验证

参数优先级为 **命令行显式参数 > TOML 配置文件 > 代码默认值**。
TOML 使用下划线键名（如 `batch_size`），命令行使用连字符
（如 `--runner.wm.batch-size`）。配置文件和 motion 文件的相对路径均相对于
运行命令时的工作目录；省略 `--config` 时保留原来的代码默认行为。

在配置文件的 `[runner]` 中设置 `log_root` 可以指定日志和 checkpoint 的根目录：

```toml
[runner]
log_root = "/path/to/training_logs" # 默认："logs"；也支持相对路径和 ~。
run_name = "lafan1_single_baseline"
```

实际保存目录为 `<log_root>/<任务名>/<run_name>/`，未设置 `run_name` 时使用时间戳。
例如上面的配置会写入
`/path/to/training_logs/SuperMimic-GenCoord-G1/lafan1_single_baseline/`，
W&B 启用时的本地文件也保存在该运行目录下。
相对 `log_root` 以启动命令的工作目录为基准；也可通过
`--runner.log-root /path/to/training_logs` 临时覆盖。
同名运行目录已存在时会报错，以免覆盖已有 checkpoint。

先用小规模配置验证采样及世界模型、策略更新：

```bash
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled \
  uv run src/supermimic/scripts/train.py SuperMimic-GenCoord-G1 \
  --config train_config.toml \
  --runner.run-name lafan1_smoke \
  --num-envs 128 \
  --runner.wm.batch-size 64 \
  --runner.policy.batch-size 64 \
  --runner.max-iterations 30 \
  --runner.log-interval 1 \
  --eager True
```

正式训练时去掉上述临时覆盖参数。查看配置文件加载后的参数及帮助：

```bash
uv run src/supermimic/scripts/train.py SuperMimic-GenCoord-G1 \
  --config train_config.toml --help
```

## Setup

依赖安装、motion 格式、坐标约定及启动排查见 [环境安装与基础实现](docs/setup.md)。

## 训练模式

- 普通训练：使用上面的 `train_config.toml`，支持 GenCoord 和 MaxCoord G1。
- EIT v6：在线滤波、后验 replay、随机世界模型与可微策略联合训练，
  入口及 `train_eit_config.toml` 教程见 [README_eit.md](README_eit.md)。
- 评估、播放与 ONNX 导出：[README_eval.md](README_eval.md)。

## Acknowledgements

SuperMimic contains code inspired by these projects:

- [ControlVAE](https://github.com/heyuanYao-pku/Control-VAE) (BSD-3-Clause)
- [robotic_world_model](https://github.com/leggedrobotics/robotic_world_model) (Apache-2.0)
- [DreamerV3](https://github.com/danijar/dreamerv3) (MIT)

## License

Copyright 2026 The SuperMimic Authors.

Released under the [Apache License 2.0](LICENSE).
