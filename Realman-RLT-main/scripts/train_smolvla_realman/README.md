# RealMan RM65 SmolVLA 7D Training

This training entry expects a LeRobot dataset converted from the RealMan
Quest/Vive HDF5 demonstrations.

Dataset features:

- `observation.images.cam_front`: top/high RGB camera
- `observation.images.cam_wrist`: wrist RGB camera
- `observation.state`: `[joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, gripper]`
- `action`: `[target_joint_1, target_joint_2, target_joint_3, target_joint_4, target_joint_5, target_joint_6, target_gripper]`

Convert existing Quest HDF5 data:

```bash
conda activate lerobot
python scripts/collect_data/convert_realman_hdf5_to_lerobot.py \
  --input-dir /home/vision123/software/VLA/lerobot-realman-vla/data/raw_hdf5_auto_v2 \
  --output-dir datasets/realman_tape_red_block_auto_v2 \
  --repo-id local/realman_tape_red_block_auto_v2 \
  --fps 20 \
  --task "用夹爪将胶带夹起后放在红色方块上，然后松开夹爪"
```

Smoke-train:

```bash
python scripts/train_smolvla_realman/train_realman_smolvla.py \
  --dataset datasets/realman_tape_red_block_auto_v2 \
  --steps 200 \
  --batch-size 2 \
  --num-workers 4 \
  --dry-run
```

The rollout side must execute the 7D joint action through
`mujoco_env.hardware.realman_api.RealManController`; the gripper command is
asynchronous and should not be treated as a high-frequency servo dimension.

After training, use the RealMan rollout entry:

```bash
python scripts/rollout_smolvla_realman/rollout_realman_smolvla.py \
  --policy-path outputs/train/realman_tape_red_block_auto_v2_smolvla_7d_YYYYMMDD_HHMMSS/checkpoints/last/pretrained_model \
  --robot-ip 192.168.2.19 \
  --no-preview
```

Add `--execute` only after dry-run logs look sane.

For RLT/intervention timing, collect or annotate a separate RealMan-RLT dataset
with real `rlt_phase` / `rl_mark` / `intervention` labels.  Do not add a fake
constant `rl_mark` to the existing 7D qpos demonstrations.
