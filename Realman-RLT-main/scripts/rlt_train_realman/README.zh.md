# RealMan RLT Stage2 / HIL-SERL 中文说明

这个目录放的是面向 RealMan RM65 的 RLT Stage2 / HIL-SERL 训练入口，当前目标任务是 USB 插拔/插入类任务。

英文版说明保留在 `README.md`，本文件只作为中文 companion，不覆盖英文内容。

## 动作和状态约定

当前 RealMan 策略仍然使用 7 维关节空间格式：

```text
observation.state = [joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, gripper]
action            = [target_joint_1, ..., target_joint_6, target_gripper]
```

其中：

- 6 个关节角单位是度；
- 第 7 维是夹爪；
- Stage2 只训练前 6 维关节残差；
- 夹爪不交给 actor 高频控制，仍沿用 SmolVLA/reference policy 的夹爪输出。

这样做的原因是 RealMan 夹爪接口会阻塞，不适合作为高频连续残差动作。

## 当前已经准备好的部分

- `scripts/rlt_train/hil_serl_core.py` 已经支持 RealMan 风格的 6 维关节残差训练。
- `config.py` 里给了比较保守的 RealMan 默认配置：
  - `state_dim=7`
  - `action_dim=7`
  - `train_action_dim=6`
  - joint residual scale 默认 `0.5 deg`
  - target policy smoothing noise 默认 `0.1 deg`
- `train_hil_serl_stage2_offline.py` 可以从已有 replay buffer 做离线/startup actor-critic 更新。

离线训练不会连接真机，它只需要 buffer 目录里有：

```text
online_buffer_latest.npz
intervention_buffer_latest.npz
```

buffer 内部使用统一的 `HILTransition` 格式：

```text
z_rl, state, ref_action, action, reward,
next_z_rl, next_state, next_ref_action, done, is_intervention
```

## RealMan HIL warmup 采集

`collect_hil_warmup_realman.py` 用来在 RealMan 真机上产生 Stage2 replay buffer。它会：

1. 冻结已经训练好的 USB SmolVLA；
2. 加载 RLT Stage1 encoder；
3. 加载 RLT gate；
4. 运行 base SmolVLA 控制机械臂；
5. 在 gate 激活阶段记录 transition 到 `online_buffer`；
6. 人类按住 Quest 右 Grip 接管时，额外记录到 `intervention_buffer`；
7. episode 结束时写入成功/失败 terminal reward。

base SmolVLA 的真机控制路径复用 `scripts/inference_realman/inference.py` 中的 RealMan `RobotController`、图像预处理和 policy pre/postprocessor；RLT token 只在需要写入 replay buffer 的帧额外计算。

旧的 UR3e 在线脚本 `scripts/rlt_train/train_hil_serl_stage2.py` 依赖 ROS2 UR3e topics 和笛卡尔阻抗控制，不应该直接跑在 RealMan 上。

第一次建议先去掉 `--execute` 做 dry-run，只检查模型、相机、Quest 和 gate 是否正常；确认后再加 `--execute`。

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

- 左 Grip：开始 episode；
- 右 Grip 按住：人工接管末端平移；
- 右 Trigger：打开夹爪。当前 insert-only warmup 推荐 `--quest-trigger-action open`，如果后续抓取阶段也需要人工介入，可改成 `toggle`；
- 左 Trigger：结束 episode，等待打分；
- Y：成功保存，reward = 1；
- X：失败保存，reward = 0；
- A：丢弃当前 episode。

脚本默认 `--record-mode gate`，也就是只记录 gate 激活阶段，适合 insert-only warmup。输出 buffer 位于固定累计目录：

Quest 遥操部分和数采脚本一样使用增量控制：右 Grip 按住时，以当前 TCP 位姿为目标初值，随后每一帧累加手柄位移增量。`--quest-yaw-deg` 要和数采脚本保持一致；当前插拔任务经调试统一使用 `45`。warmup 里 Quest 遥操是独立 50Hz 线程，`--freq 20` 只控制相机/gate/RLT 记录频率。Quest 的 `rm_movep_canfd` radio 单独使用 `--quest-canfd-radio 60`，与数采脚本一致。

```text
outputs/rlt_stage2_realman/hil_warmup_realman_usb_insert_v1/buffers/
```

再次运行 warmup 脚本时，会自动加载该目录下已有的 `online_buffer_latest.npz` 和
`intervention_buffer_latest.npz` 并继续追加；新的 episode 会从已有编号后继续保存。

## 当前默认 checkpoint

默认路径在 `config.py` 中配置。当前包括：

```text
policy:
outputs/train/realman_usb_insert_v1_smolvla_7d_fullft_20260715_154106/checkpoints/010000/pretrained_model

gate:
outputs/rlt_gate/rlt_gate_20260715_180731/best.pt

stage1:
outputs/rlt_stage1/rlt_stage1_realman_usb_insert_v1_smolvla_7d_fullft_20260715_154106_010000_20260715_181941/best.pt
```

## 离线/startup Stage2 训练命令

有 RealMan HIL buffer 后运行：

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

如果显存紧张，把 batch size 降到：

```bash
--rlt-batch-size 64
```

输出 checkpoint 会保存到：

```text
outputs/rlt_stage2_realman/hil_serl_stage2_realman_offline_*/checkpoints/stage2_realman_offline.pt
```

## 第一次 RealMan RL 试验的安全建议

一开始建议保守：

- `--rlt-action-delta-scale-joint-deg` 先保持 `0.5`；
- actor 接管前至少先收集 `64` 条 warmup transitions；
- 先只做插入阶段 intervention，再决定是否加入抓取/拔出阶段 intervention；
- 夹爪保持二值/参考策略控制，不让 actor 高频控制夹爪。
