# 普通训练模式的 W&B

项目已接入 W&B，可在线记录训练配置、指标和模型 checkpoint。先完成 [环境安装](setup.md)，并按 [数据处理教程](../README.md#1-批量处理-lafan1_g1-数据) 准备
`data/motions/run1_subject5.npz`。以下命令在 `supermimic/` 仓库根目录执行。

### 1. 登录 W&B

在浏览器登录 [W&B 设置页面](https://wandb.ai/settings)，创建并复制 API Key，
然后在终端执行：

```bash
cd ~/Github_Repos/WBC/supermimic_ws/supermimic
WANDB_MODE=online uv run wandb login
```

按终端提示粘贴 API Key 并回车。更换账号时执行
`WANDB_MODE=online uv run wandb login --relogin`。
登录方式见 [W&B 官方说明](https://github.com/wandb/wandb#quickstart)。

### 2. 启动训练并上传指标

若需要指定 W&B 团队，在启动训练前执行以下命令，将占位内容替换为你有写入权限的
团队标识；不设置时使用账号的默认归属：

```bash
export WANDB_ENTITY="你的团队标识"
```

使用当前配置训练动作，并将指标上传到 `supermimic` 项目：
```bash
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=online \
  uv run src/supermimic/scripts/train.py SuperMimic-GenCoord-G1 \
  --config train_config.toml \
  --runner.wandb-project supermimic \
  --runner.run-name "lafan1_all_$(date +%Y%m%d_%H%M%S)"
```

**在线训练应使用 `WANDB_MODE=online`，不要沿用前面示例的 `WANDB_MODE=disabled`。**
运行名附加时间戳，以免与已有日志目录重名；当前代码会拒绝复用同名目录。

### 3. 日志与训练结果

训练曲线、checkpoint 选择及可视化用法见 [评估与可视化](../README_eval.md)。

默认配置下，本地运行目录如下：

```text
logs/SuperMimic-GenCoord-G1/<运行名>/
├── world_model.pt  # 世界模型与仿真上下文参数
├── actor.pt        # 策略网络参数
├── critic.pt       # 价值网络参数（Actor-Critic 模式）
└── wandb/          # W&B 运行记录与调试日志
```

修改 `[runner].log_root` 或传入 `--runner.log-root /path/to/training_logs`
可更换根目录；W&B 本地文件也保存在该运行目录下。
模型按 `[runner].save_interval` 保存（默认每 1000 个采集步），同名 `.pt`
文件会被覆盖，保留最新保存点；不包含优化器状态和 replay buffer。
第一次世界模型和策略更新需要编译，可能几分钟没有新的训练日志；
终端会打印首次更新的开始和完成提示。排查方式见
[Diagnosing startup stalls](setup.md#diagnosing-startup-stalls)。

### 4. 离线训练与后续同步

使用 `WANDB_MODE=offline` 启动训练时，模型仍按上述规则保存。
`wandb/offline-run-<时间>-<ID>/` 内的 `run-<ID>.wandb` 保存训练指标、配置
和运行记录，`files/requirements.txt` 保存依赖版本，`logs/` 保存 W&B 调试日志。
待上传的模型版本还会暂存到 `$WANDB_DATA_DIR/artifacts/staging/`
（未设置 `WANDB_DATA_DIR` 时使用 W&B 默认数据目录），占用随保存次数增长。

训练结束且网络可用后，在训练所在机器登录并同步；将下面的路径替换为实际的
`offline-run-*` 目录。同步前保留该目录及模型暂存文件：

```bash
WANDB_MODE=online uv run wandb login
WANDB_MODE=online uv run wandb beta sync \
  --entity "你的账号或团队标识" \
  --project supermimic \
  --no-skip-synced \
  --verbose \
  "logs/SuperMimic-GenCoord-G1/<运行名>/wandb/offline-run-<时间>-<ID>"
```

该命令上传离线记录的指标、配置和模型 artifacts。`WANDB_MODE=disabled`
不会生成可供后续同步的 W&B 离线记录。

通过本地 `download_training.sh` 完整下载日志和 artifacts 后，脚本会校验模型
文件的大小及 MD5，并将 `.wandb` 中的服务器暂存路径映射为本地绝对路径，打印
可在本机执行的同步命令。原始记录保留为 `*.wandb.before-path-map`；模型文件
和训练指标保持原样，脚本本身不上传数据。`--logs-only` 不处理这些路径。

已经下载的训练结果可直接处理，无需再次连接服务器：

```bash
bash download_training.sh --map-only --dry-run # 校验文件和待映射路径
bash download_training.sh --map-only           # 映射并打印本机同步命令
```

修改脚本顶部的 `LOCAL_DOWNLOAD_ROOT`、`TASK_ID`、`RUN_NAME` 可选择其他运行；
`PYTHON_BIN` 默认使用项目 `.venv/bin/python`，需已安装 W&B。

