# RealMan RM65 SmolVLA Rollout

This rollout entry is for the RealMan 7D joint-space policy trained from
`raw_hdf5_auto_v2`.

The base VLA data format is:

- `observation.images.cam_front`
- `observation.images.cam_wrist`
- `observation.state`: `[joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, gripper]`
- `action`: `[target_joint_1, target_joint_2, target_joint_3, target_joint_4, target_joint_5, target_joint_6, target_gripper]`

This intentionally does not use TCP `rotvec` or `rl_mark`:

- TCP orientation is already determined by the 6 joint angles through FK.
- `rl_mark` is not robot state. It is a task/RLT supervision signal and should
  be recorded or labeled separately when collecting RLT data.

## Run cameras

The script subscribes to the same ROS image topics used by the UR3e collector:

```bash
conda activate ur3e_rlt
scripts/collect_data/run_realsense_cameras.sh
```

If the wrist camera is launched at a different resolution from training, the
rollout script resizes the incoming ROS image to the policy input shape by
default.

## Dry run

Dry run still connects to the RealMan arm so it can read the cached joint state,
but it does not send model actions:

```bash
conda activate lerobot
source /opt/ros/humble/setup.bash
python scripts/rollout_smolvla_realman/rollout_realman_smolvla.py \
  --policy-path outputs/train/realman_tape_red_block_auto_v2_smolvla_7d_YYYYMMDD_HHMMSS/checkpoints/last/pretrained_model \
  --robot-ip 192.168.2.19 \
  --no-preview
```

## Execute

Use `--execute` only after checking the dry-run logs and workspace:

```bash
conda activate lerobot
source /opt/ros/humble/setup.bash
python scripts/rollout_smolvla_realman/rollout_realman_smolvla.py \
  --policy-path outputs/train/realman_tape_red_block_auto_v2_smolvla_7d_YYYYMMDD_HHMMSS/checkpoints/last/pretrained_model \
  --robot-ip 192.168.2.19 \
  --execute \
  --move-home-on-start
```

Execution details:

- Joint command: `rm_movej_follow` by default.
- State read: background cache, default 20 Hz.
- Gripper: asynchronous worker with binary hysteresis by default.
- Safety smoothing: joint EMA plus per-step joint delta limit.

## Legacy BGR/RGB diagnostic

Some early RealMan datasets were converted from OpenCV/RealSense BGR frames
without a BGR→RGB conversion.  For those legacy checkpoints, keep the original
task prompt unchanged and run an A/B test by changing only the policy input
channel order:

```bash
# Normal deployment input.
python scripts/rollout_smolvla_realman/rollout_realman_smolvla.py \
  --policy-path outputs/train/realman_tape_red_block_auto_v2_smolvla_7d_imported/checkpoints/last/pretrained_model \
  --robot-ip 192.168.2.19 \
  --policy-image-channel-order rgb

# Diagnostic mode: emulate the old BGR-as-RGB training distribution.
python scripts/rollout_smolvla_realman/rollout_realman_smolvla.py \
  --policy-path outputs/train/realman_tape_red_block_auto_v2_smolvla_7d_imported/checkpoints/last/pretrained_model \
  --robot-ip 192.168.2.19 \
  --policy-image-channel-order bgr
```

The preview window still displays normal colors; only the tensors sent to the
policy are channel-swapped.  If `bgr` works noticeably better, reconvert the
raw HDF5 demos with an explicit BGR→RGB conversion and retrain the policy
instead of keeping this diagnostic flag as the final solution.

## RLT note

The converted `raw_hdf5_auto_v2` dataset is suitable for training the base VLA,
but it does not contain enough supervision for "when should RL intervene".

For the RLT stage, collect or annotate a separate RealMan-RLT dataset with
signals such as:

- `rlt_phase`
- `rl_mark`
- `intervention`
- human takeover start/end
- success/failure/reward labels

Keep those as separate features or sidecar metadata. Do not fake them by adding
a constant dimension to the old 7D qpos demos.
