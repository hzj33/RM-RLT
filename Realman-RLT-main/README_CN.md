# RealMan USB 插拔任务操作指南

本文档是睿尔曼 RM65 上复现 USB 插拔任务的操作流程，覆盖：

```text
Quest 采集 → HDF5 检查 → LeRobot 转换 → base SmolVLA 训练/部署
→ RL gate 标注/训练 → RLT Stage1 encoder 训练 → Stage2 warmup
```

第一次配置环境请先看：

```text
ENV_SETUP_CN.md
```

任务描述统一使用：

```bash
TASK="把USB头从右侧抓起后插入左侧USB口"
```

RealMan 数据采用 7 维状态/动作：

```text
[joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, gripper]
```

其中前 6 维是关节角，第 7 维是夹爪。Stage2 只训练前 6 维关节残差；夹爪保持由 base SmolVLA 参考动作控制。

## 0. 路径约定

在本文档中，用 `v2` 作为新数据版本示例。实际使用时可以替换成自己的版本名。

```text
raw HDF5:
data/raw_hdf5_insert_v2

LeRobot 数据集:
datasets/realman_usb_insert_v2

base SmolVLA 训练输出:
outputs/train/realman_usb_insert_v2_smolvla_7d_fullft_*

RL gate 输出:
outputs/rlt_gate/rlt_gate_*

RLT Stage1 输出:
outputs/rlt_stage1/rlt_stage1_*

Stage2 输出:
outputs/rlt_stage2_realman
```

本项目内已经放了一份 RealMan/Quest 采集脚本副本：

```text
scripts/collect_data_realman/collect_data.py
```

它来自 `~/software/VLA/lerobot-realman-vla/scripts/collect_data.py`，后续建议直接使用本项目内副本。

## 1. Quest 采集 raw HDF5

USB 插拔任务使用固定初始/复位关节位，和部署命令中的 `--init-qpos` 对齐：

```text
12.817,19.830,82.662,0.205,77.997,42.185,1
```

其中前 6 维用于采集脚本的 `--quest-home-joints`，第 7 维是夹爪打开值。不要在这批任务里使用 `--quest-home-current-on-start`，否则采集 home 会变成脚本启动瞬间的当前关节角。

```bash
cd ~/software/VLA/ur3e_rl
conda activate lerobot

python scripts/collect_data_realman/collect_data.py \
  --arm-ip 192.168.2.19 \
  --save-dir data/raw_hdf5_insert_v2 \
  --task-name realman_usb_insert \
  --fps 20 \
  --cam-top /dev/video6 \
  --cam-wrist 344522071398 \
  --cam-top-width 640 \
  --cam-top-height 480 \
  --cam-wrist-width 1280 \
  --cam-wrist-height 720 \
  --cam-wrist-exposure 500 \
  --quest \
  --quest-home-joints "12.817,19.830,82.662,0.205,77.997,42.185" \
  --quest-home-speed 40 \
  --quest-pos-scale 0.2 \
  --quest-yaw-deg 45 \
  --quest-gripper-mode toggle
```

Quest 操作：

```text
右 Grip       按住后移动机械臂末端
右 Trigger   夹爪开/合切换
左 Grip       开始录制
左 Trigger   停止并保存当前 episode
```

建议：

- 插拔任务默认 `--quest-pos-scale 0.2`。
- 纯 Quest 模式下默认锁定末端姿态，适合当前插拔任务。
- 夹爪建议使用 `toggle`，不要用 `continuous`。
- 每条 episode 只保存语义正确、动作完整的数据。

## 2. 检查 raw HDF5

先用转换脚本的 dry-run 检查字段、shape 和帧数：

```bash
cd ~/software/VLA/ur3e_rl
conda activate lerobot

TASK="把USB头从右侧抓起后插入左侧USB口"

python scripts/collect_data/convert_realman_hdf5_to_lerobot.py \
  --input-dir data/raw_hdf5_insert_v2 \
  --output-dir datasets/realman_usb_insert_v2 \
  --repo-id local/realman_usb_insert_v2 \
  --fps 20 \
  --task "$TASK" \
  --dry-run
```

合格数据应满足：

- `observations/qpos` 是 `(N, 7)`；
- `action` 是 `(N, 7)`；
- `observations/images/cam_high` 和 `observations/images/cam_wrist` 均存在；
- qpos、action、两路图像帧数一致；
- 实际帧率接近 20 Hz；
- task 语义与保存的 episode 对齐。

如果只有少量坏文件，正式转换时可加 `--skip-bad`。

## 3. 转换为 LeRobot 数据集

```bash
cd ~/software/VLA/ur3e_rl
conda activate lerobot

TASK="把USB头从右侧抓起后插入左侧USB口"

python scripts/collect_data/convert_realman_hdf5_to_lerobot.py \
  --input-dir data/raw_hdf5_insert_v2 \
  --output-dir datasets/realman_usb_insert_v2 \
  --repo-id local/realman_usb_insert_v2 \
  --fps 20 \
  --task "$TASK" \
  --overwrite
```

转换后应生成：

```text
datasets/realman_usb_insert_v2/meta/info.json
datasets/realman_usb_insert_v2/meta/stats.json
datasets/realman_usb_insert_v2/data/chunk-000/file-000.parquet
datasets/realman_usb_insert_v2/videos/...
```

转换脚本会把 HDF5 中的 BGR 图像转成训练用 RGB。

## 4. 训练 base SmolVLA

先 dry-run 检查训练命令和数据集：

```bash
cd ~/software/VLA/ur3e_rl
conda activate lerobot

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/train_smolvla_realman/train_realman_smolvla.py \
  --dataset datasets/realman_usb_insert_v2 \
  --repo-id local/realman_usb_insert_v2 \
  --job-name realman_usb_insert_v2_smolvla_7d_fullft \
  --steps 200 \
  --batch-size 2 \
  --num-workers 4 \
  --dry-run
```

正式训练：

```bash
cd ~/software/VLA/ur3e_rl
conda activate lerobot

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/train_smolvla_realman/train_realman_smolvla.py \
  --dataset datasets/realman_usb_insert_v2 \
  --repo-id local/realman_usb_insert_v2 \
  --job-name realman_usb_insert_v2_smolvla_7d_fullft \
  --steps 10000 \
  --batch-size 4 \
  --num-workers 4 \
  --log-freq 100 \
  --save-freq 1000 \
  --no-freeze-vision-encoder \
  --no-train-expert-only
```

输出 checkpoint 示例：

```text
outputs/train/realman_usb_insert_v2_smolvla_7d_fullft_YYYYMMDD_HHMMSS/checkpoints/010000/pretrained_model
```

如果显存足够，可以把 `--batch-size` 调到 8；如果显存紧张，保持 4。

## 5. 部署 base SmolVLA

本项目已内置一份从 `lerobot-realman-vla/scripts/inference.py` 复制并固定 USB home 的 RealMan 推理脚本：
`scripts/inference_realman/inference.py`。它直接使用 RealMan API 和相机参数，不依赖 ROS 图像 topic。

```bash
cd ~/software/VLA/ur3e_rl
conda activate lerobot

TASK="把USB头从右侧抓起后插入左侧USB口"
MODEL="/home/vision123/software/VLA/ur3e_rl/outputs/train/realman_usb_insert_v1_smolvla_7d_fullft_20260715_154106/checkpoints/010000/pretrained_model"

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/inference_realman/inference.py \
  --offline \
  --arm-ip 192.168.2.19 \
  --model "$MODEL" \
  --cam-top /dev/video6 \
  --cam-wrist 344522071398 \
  --cam-top-width 640 \
  --cam-top-height 480 \
  --cam-wrist-width 1280 \
  --cam-wrist-height 720 \
  --cam-wrist-fps 30 \
  --cam-wrist-exposure 500 \
  --freq 20 \
  --task "$TASK" \
  --init-qpos "12.817,19.830,82.662,0.205,77.997,42.185,1" \
  --live-state \
  --joint-command-mode canfd \
  --canfd-trajectory-mode 2 \
  --canfd-radio 80 \
  --binary-gripper
```

部署时注意：

- `--task` 必须和转换数据时的 task 完全一致。
- `--init-qpos` 要和采集 home 一致；当前 USB 插拔默认值已经固定为 `12.817,19.830,82.662,0.205,77.997,42.185,1`。
- 先确认 dry-run 或小范围动作正常，再实际执行。
- 如果重新采集时 home 变了，要同步修改 `--init-qpos`。

## 6. 标注 RL gate

RL gate 标注的是“哪些帧应该交给后续 RL/残差策略处理”。

常用方案：

```text
insert-only:
只标注插入阶段。

grasp+insert:
同一条 episode 中标注抓取阶段和插入阶段两段区间。
```

标注命令：

```bash
cd ~/software/VLA/ur3e_rl
conda activate lerobot

python scripts/rlt_gate/annotate_rlt_gate.py \
  --dataset datasets/realman_usb_insert_v2 \
  --camera both \
  --output-json datasets/realman_usb_insert_v2/meta/rlt_gate_annotations_insert_only.json \
  --output-parquet datasets/realman_usb_insert_v2/meta/rlt_gate_labels_insert_only.parquet
```

按键：

```text
Space  播放/暂停
R      标注区间开始/结束
Z      撤销当前开始点或最后一个区间
A/D    前后移动 1 帧
J/L    前后移动 1 秒
B      回放最后一个标注区间
E      从当前 episode 开头播放
N/P    下一条/上一条 episode
S      保存
Q      退出并保存
```

标注完成后，将选择使用的标签复制成默认文件：

```bash
cp datasets/realman_usb_insert_v2/meta/rlt_gate_labels_insert_only.parquet \
   datasets/realman_usb_insert_v2/meta/rlt_gate_labels.parquet
```

## 7. 训练 RL gate

```bash
cd ~/software/VLA/ur3e_rl
conda activate lerobot
mkdir -p logs

python scripts/rlt_gate/train_rlt_gate.py \
  --dataset datasets/realman_usb_insert_v2 \
  --camera both \
  --model resnet18 \
  --image-size 128 \
  --batch-size 96 \
  --num-workers 4 \
  --device cuda 2>&1 | tee logs/realman_usb_insert_v2_rlt_gate_resnet18_both.log
```

输出：

```text
outputs/rlt_gate/rlt_gate_YYYYMMDD_HHMMSS/best.pt
outputs/rlt_gate/rlt_gate_YYYYMMDD_HHMMSS/last.pt
outputs/rlt_gate/rlt_gate_YYYYMMDD_HHMMSS/metrics.csv
```

## 8. 训练 RLT Stage1 encoder

Stage1 encoder 冻结 base SmolVLA，读取 VLM/context token 和 action expert token，训练得到 `z_rl`。这一步不需要人工救场数据，也不需要 reward。

```bash
cd ~/software/VLA/ur3e_rl
conda activate lerobot
mkdir -p logs

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/rlt_token/train_rlt_stage1.py \
  --policy-path outputs/train/realman_usb_insert_v2_smolvla_7d_fullft_YYYYMMDD_HHMMSS/checkpoints/010000/pretrained_model \
  --batch-size 8 \
  --num-workers 4 \
  --steps 10000 \
  --device cuda 2>&1 | tee logs/realman_usb_insert_v2_rlt_stage1.log
```

如果显存不够：

```bash
--batch-size 4
```

输出：

```text
outputs/rlt_stage1/rlt_stage1_*/best.pt
outputs/rlt_stage1/rlt_stage1_*/last.pt
outputs/rlt_stage1/rlt_stage1_*/summary.json
```

## 9. Stage2 warmup

Stage2 actor-critic 在 RL gate 激活阶段，对 base SmolVLA 的参考动作做小幅 6 维关节残差修正。

Stage2 分两步：先在真机上采集 HIL warmup replay buffer，再离线训练 actor-critic。

### 9.1 真机采集 HIL warmup buffer

采集脚本会同时加载：

- 已训练好的 USB base SmolVLA；
- RLT Stage1 encoder；
- RL gate；
- Quest 手柄输入。

输出目录结构：

```text
outputs/rlt_stage2_realman/hil_warmup_realman_usb_insert_v1/
  episode_000000/
  episode_000001/
  ...
  sessions/
  buffers/
  online_buffer_latest.npz
  intervention_buffer_latest.npz
```

这个目录是固定累计目录。每次重新启动 warmup 脚本时，如果该目录已经存在 `buffers/online_buffer_latest.npz` 和
`buffers/intervention_buffer_latest.npz`，脚本会自动加载并继续累计；新的 episode 编号也会从已有编号后继续递增。

buffer 语义：

```text
online_buffer:
gate 激活阶段的所有 transition

intervention_buffer:
人类 Quest 接管/救场的 transition

reward:
episode 终止时成功为 1，失败为 0
```

第一次建议先不加 `--execute` 做 dry-run，只确认相机、Quest、模型和 gate 都能正常加载。确认后再加 `--execute` 让脚本实际控制机械臂。

```bash
cd ~/software/VLA/ur3e_rl
conda activate lerobot

TASK="把USB头从右侧抓起后插入左侧USB口"

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/rlt_train_realman/collect_hil_warmup_realman.py \
  --policy-path outputs/train/realman_usb_insert_v1_smolvla_7d_fullft_20260715_154106/checkpoints/010000/pretrained_model \
  --stage1-checkpoint outputs/rlt_stage1/rlt_stage1_realman_usb_insert_v1_smolvla_7d_fullft_20260715_154106_010000_20260715_181941/best.pt \
  --gate-checkpoint outputs/rlt_gate/rlt_gate_20260715_180731/best.pt \
  --robot-ip 192.168.2.19 \
  --cam-top /dev/video6 \
  --cam-wrist 344522071398 \
  --cam-top-width 640 \
  --cam-top-height 480 \
  --cam-wrist-width 1280 \
  --cam-wrist-height 720 \
  --camera-fps 30 \
  --cam-wrist-exposure 500 \
  --freq 20 \
  --execution-horizon 50 \
  --task "$TASK" \
  --output-dir outputs/rlt_stage2_realman/hil_warmup_realman_usb_insert_v1 \
  --home-qpos "12.817,19.830,82.662,0.205,77.997,42.185,1" \
  --move-home-on-start \
  --policy-image-channel-order rgb \
  --command-mode canfd \
  --canfd-trajectory-mode 2 \
  --canfd-radio 80 \
  --quest-pos-scale 0.2 \
  --quest-yaw-deg 45 \
  --quest-control-hz 50 \
  --quest-canfd-radio 60 \
  --quest-trigger-action open \
  --record-mode gate \
  --binary-gripper \
  --execute
```

Quest 操作：

- 左 Grip：开始一条 warmup episode；
- 右 Grip 按住：人工接管末端平移；
- 右 Trigger：打开夹爪。这里使用 `--quest-trigger-action open`，适合 insert-only warmup 中“插好后松开”；如果后续抓取阶段也要人工介入，可改成 `--quest-trigger-action toggle`；
- 左 Trigger：结束当前 episode，等待打分；
- Y：按成功保存，terminal reward = 1；
- X：按失败保存，terminal reward = 0；
- A：丢弃当前 episode。

当前默认 `--record-mode gate`，也就是只在 RL gate 激活阶段记录 transition。适合先做 insert-only warmup；如果后面想让抓取阶段也进入 Stage2，再改 gate 标注或使用 `--record-mode all` 做诊断。

Quest 遥操部分和数采脚本保持同一套增量控制逻辑：右 Grip 按住后，以当前 TCP 位姿为目标初值，每一帧累加手柄位移增量。`--quest-yaw-deg` 必须和采集脚本一致；当前插拔任务经调试统一使用 `45`。warmup 中 Quest 遥操是独立 50Hz 线程，`--freq 20` 只控制相机/gate/RLT 记录频率。

### 9.2 离线训练 actor-critic warmup

```bash
cd ~/software/VLA/ur3e_rl
conda activate lerobot

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/rlt_train_realman/train_hil_serl_stage2_offline.py \
  --rlt-buffer-dir outputs/rlt_stage2_realman/hil_warmup_realman_usb_insert_v1/buffers \
  --rlt-startup-updates 300 \
  --rlt-batch-size 128 \
  --rlt-action-chunk-steps 50 \
  --rlt-action-delta-scale-joint-deg 0.5 \
  --rlt-target-noise-joint-deg 0.1 \
  --rlt-target-noise-clip-joint-deg 0.3
```

输出：

```text
outputs/rlt_stage2_realman/hil_serl_stage2_realman_offline_*/checkpoints/stage2_realman_offline.pt
```

参数建议：

- `--rlt-action-delta-scale-joint-deg 0.5` 用于初次训练；
- `--rlt-batch-size 128` 显存不够时改成 64；
- 先做 insert-only intervention buffer，再考虑 grasp+insert。

## 10. 推荐数据规模

建议先小批量走通链路，再扩大数据：

```text
链路测试:
20~30 条 episode

base SmolVLA 可用训练:
80~150 条 episode

RL gate 标注:
先 insert-only，再根据 base 失败模式决定是否加入 grasp

Stage2 warmup:
优先收集插入阶段的人类救场数据
```

判断是否进入 RLT/Stage2 的基本标准：

- base SmolVLA 已有一定成功率；
- 失败主要集中在最后对准/插入阶段；
- gate 标注语义清楚；
- 人类可以稳定接管并完成救场。
