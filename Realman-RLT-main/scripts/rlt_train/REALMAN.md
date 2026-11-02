# RealMan RLT Migration Notes

The current files in `scripts/rlt_train/` are still the UR3e RLT pipeline.
They assume:

- UR3e ROS topics such as `/ur3e_vr/robot_state`, `/ur3e_vr/ik_target`, and `/ur3e_vr/vr_command`
- no-rotvec TCP state/action dim 4
- fixed UR3e impedance orientation
- VR intervention messages from the UR3e teleop stack

Do not run these Stage2/RLT scripts directly on the RealMan 7D joint-space
policy.  They intentionally fail on non-4D policies.

For RealMan, the base policy should remain:

```text
observation.state = [6 joints + gripper]
action            = [6 target joints + target gripper]
```

To reproduce the RLT intervention workflow, collect or annotate an additional
RealMan-RLT dataset with real labels/signals:

- `rlt_phase` or `rl_mark`
- `intervention`
- human takeover start/end
- per-episode success/failure or reward
- optional gate labels for "RL should take over now"

Keep these labels as separate LeRobot features or sidecar files.  Do not add a
constant fake `rl_mark` dimension to the existing `raw_hdf5_auto_v2` qpos demos.

The converter supports future scalar labels explicitly, for example:

```bash
python scripts/collect_data/convert_realman_hdf5_to_lerobot.py \
  --input-dir data/realman_rlt_hdf5 \
  --output-dir datasets/realman_rlt_labeled \
  --repo-id local/realman_rlt_labeled \
  --fps 20 \
  --task "..." \
  --input-color-order bgr \
  --front-output-key observation.images.cam_front \
  --scalar-key rlt_phase=observation.rlt_phase \
  --scalar-key intervention=observation.intervention
```

This only copies labels that truly exist in the HDF5 files.
