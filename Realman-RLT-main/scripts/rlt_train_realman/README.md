# RealMan RLT Stage2 / HIL-SERL

This folder contains the RealMan-specific Stage2 pieces for the USB insertion
task.

The current RealMan policy uses a 7D action/state convention:

- joints 1-6 in degrees
- gripper as the 7th dimension

Stage2 trains only the first 6 dimensions as joint-space residuals on top of
the frozen SmolVLA reference action. The gripper dimension is kept from the
reference policy because the RealMan gripper command is blocking and should not
be driven as a high-frequency residual output.

## What is ready

- `scripts/rlt_train/hil_serl_core.py` now supports arbitrary
  `train_action_dim`, scalar/vector action scales, scalar/vector action bounds,
  and RealMan-style 6D joint residuals.
- `config.py` defines conservative RealMan defaults:
  - `state_dim=7`
  - `action_dim=7`
  - `train_action_dim=6`
  - residual scale `0.5 deg`
  - target policy smoothing noise `0.1 deg`
  - USB SmolVLA policy checkpoint
  - USB insert-only RL gate checkpoint
  - the current USB RLT Stage1 encoder checkpoint path
- `train_hil_serl_stage2_offline.py` runs offline/startup actor-critic updates
  from existing replay buffers.

The offline trainer does not connect to the robot. It expects a buffer directory
with:

```text
online_buffer_latest.npz
intervention_buffer_latest.npz
```

These files use the common `HILTransition` format:

```text
z_rl, state, ref_action, action, reward,
next_z_rl, next_state, next_ref_action, done, is_intervention
```

## RealMan HIL warmup collection

Use `collect_hil_warmup_realman.py` to create the Stage2 replay buffers on the
RealMan robot. It runs the frozen USB SmolVLA policy with the trained RLT
encoder and RL gate, then saves:

- all gate-active transitions into `online_buffer`
- human rescue / Quest override transitions into `intervention_buffer`
- terminal reward `1` for success and `0` for failure

Base SmolVLA execution reuses the RealMan `RobotController`, image conversion,
and policy pre/postprocessing path from `scripts/inference_realman/inference.py`.
RLT token encoding is computed only on frames that are written to the replay
buffer.

The old UR3e online script in `scripts/rlt_train/train_hil_serl_stage2.py`
depends on ROS2 UR3e topics and Cartesian impedance commands, so it is not safe
to run directly on RealMan.

First run without `--execute` as a dry-run. Add `--execute` only after the
model, cameras, Quest input, and gate load correctly.

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

Quest controls:

- left Grip: start episode
- hold right Grip: human Cartesian override
- right Trigger: open gripper. For insert-only warmup, `--quest-trigger-action open` makes one press always release; switch to `toggle` if grasp-stage intervention also needs manual close/open control.
- left Trigger: end episode and wait for reward
- Y: save as success, reward `1`
- X: save as failure, reward `0`
- A: discard current episode

Quest override uses the same incremental Cartesian control convention as the
collection script: while right Grip is held, the current TCP pose is used as the
target seed and controller motion deltas are accumulated frame by frame.
Keep `--quest-yaw-deg` identical to collection; the tuned value for this USB
insertion setup is `45`. Quest override runs in its own 50Hz thread; `--freq 20`
only controls camera/gate/RLT recording. Quest `rm_movep_canfd` uses
`--quest-canfd-radio 60`, matching `collect_data.py`.

Current default checkpoints in `config.py`:

Warmup buffers are accumulated in one fixed directory:

```text
outputs/rlt_stage2_realman/hil_warmup_realman_usb_insert_v1/buffers/
```

When the collector is started again, it auto-loads existing buffers from this
directory and appends new episodes with increasing episode indices.

```text
policy:
outputs/train/realman_usb_insert_v1_smolvla_7d_fullft_20260715_154106/checkpoints/010000/pretrained_model

gate:
outputs/rlt_gate/rlt_gate_20260715_180731/best.pt

stage1:
outputs/rlt_stage1/rlt_stage1_realman_usb_insert_v1_smolvla_7d_fullft_20260715_154106_010000_20260715_181941/best.pt
```

## Offline/startup Stage2 command

After you have a RealMan HIL buffer directory, run:

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

If CUDA memory is tight, reduce `--rlt-batch-size` to `64`.

The output checkpoint is saved under:

```text
outputs/rlt_stage2_realman/hil_serl_stage2_realman_offline_*/checkpoints/stage2_realman_offline.pt
```

## Safety defaults for first RealMan RL trials

Use conservative values initially:

- keep `--rlt-action-delta-scale-joint-deg 0.5`
- require at least `64` warmup transitions before actor control
- collect insert-only intervention first, then decide whether to add grasp
  intervention
- keep the gripper binary / reference-controlled, not actor-controlled
