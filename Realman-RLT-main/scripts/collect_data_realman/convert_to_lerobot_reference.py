#!/usr/bin/env python3
# Reference converter for the RealMan USB/RLT workflow.
"""
HDF5 → LeRobot v3 数据集转换

将采集脚本生成的 HDF5 文件转换为 LeRobot 训练所需的标准格式。
转换过程包括：
  1. 读取 HDF5 中的关节角、夹爪状态、双相机图像
  2. 创建 LeRobot Dataset（含视频编码、归一化统计）
  3. 逐帧写入并保存

⚠️ 重要：--fps 必须与采集时的帧率一致！

用法:
    python scripts/convert_to_lerobot.py \\
        --input-dir data/raw_hdf5/pick_cube \\
        --output-dir data/pick_cube_30fps \\
        --repo-id lerobot/pick_cube_30fps \\
        --fps 30 \\
        --task "pick up the cube and place it in the basket"
"""

import h5py
import numpy as np
from pathlib import Path
import argparse
import sys
import shutil
import torch
from PIL import Image

# 确保 LeRobot 可以被导入
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def image_to_rgb(image: np.ndarray, input_color_order: str) -> np.ndarray:
    """Return a contiguous RGB image from the raw HDF5 camera frame."""
    if input_color_order == "bgr":
        return np.ascontiguousarray(image[..., ::-1])
    if input_color_order == "rgb":
        return np.ascontiguousarray(image)
    raise ValueError(f"unsupported input color order: {input_color_order}")


def convert_hdf5_episode(hdf5_path: Path):
    """读取单个 HDF5 文件，返回字典"""
    with h5py.File(hdf5_path, 'r') as f:
        data = {
            'qpos': np.array(f['observations/qpos']),        # (N, 7)
            'action': np.array(f['action']),                  # (N, 7)
            'cam_high': np.array(f['observations/images/cam_high']),    # (N, H, W, 3)
            'cam_wrist': np.array(f['observations/images/cam_wrist']),  # (N, H, W, 3)
        }
    return data


def validate_hdf5_episode(hdf5_path: Path):
    """只检查 HDF5 能否打开以及关键字段是否存在，不读取完整图像数据。"""
    required_keys = [
        'observations/qpos',
        'action',
        'observations/images/cam_high',
        'observations/images/cam_wrist',
    ]
    with h5py.File(hdf5_path, 'r') as f:
        for key in required_keys:
            if key not in f:
                raise KeyError(f"missing key: {key}")
        lengths = [f[key].shape[0] for key in required_keys]
        if len(set(lengths)) != 1:
            raise ValueError(f"frame count mismatch: {dict(zip(required_keys, lengths))}")
        return {
            'qpos': f['observations/qpos'].shape,
            'action': f['action'].shape,
            'cam_high': f['observations/images/cam_high'].shape,
            'cam_wrist': f['observations/images/cam_wrist'].shape,
        }


def main():
    parser = argparse.ArgumentParser(description='HDF5 → LeRobot 数据集转换')
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="HDF5 文件所在目录")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="LeRobot 数据集输出目录")
    parser.add_argument("--repo-id", type=str, required=True,
                        help="数据集标识符 (如 lerobot/pick_cube_30fps)")
    parser.add_argument("--fps", type=int, default=30,
                        help="帧率，应与采集时一致（默认30）")
    parser.add_argument("--task", type=str, default="pick up the cube and place it in the basket",
                        help="任务描述（VLA策略需要，必须与推理时完全一致）")
    parser.add_argument("--skip-bad", action="store_true",
                        help="跳过无法打开或字段不完整的 HDF5 文件")
    parser.add_argument("--vcodec", type=str, default="h264",
                        help="视频编码器，默认 h264；可选 libsvtav1/auto/h264/hevc 等")
    parser.add_argument("--streaming-encoding", action="store_true",
                        help="边加帧边编码视频，跳过临时 PNG，通常更快")
    parser.add_argument("--encoder-queue-maxsize", type=int, default=2048,
                        help="streaming 编码队列大小，调大可避免离线转换时丢帧")
    parser.add_argument("--encoder-threads", type=int, default=None,
                        help="每路视频编码线程数，默认由编码器自动决定")
    parser.add_argument("--input-color-order", choices=["bgr", "rgb"], default="bgr",
                        help="原始 HDF5 图像通道顺序；本项目 OpenCV/RealSense 采集为 bgr（默认）")
    args = parser.parse_args()

    # 检查输出目录
    if args.output_dir.exists():
        print(f"输出目录已存在: {args.output_dir}")
        resp = input("是否删除并重新创建？(y/n): ")
        if resp.lower() == 'y':
            shutil.rmtree(args.output_dir)
        else:
            print("取消转换")
            return

    # 获取所有 HDF5 文件
    hdf5_files = sorted(args.input_dir.glob("*.hdf5"))
    print(f"找到 {len(hdf5_files)} 个 episodes")

    if len(hdf5_files) == 0:
        print("错误：未找到 HDF5 文件")
        return

    valid_hdf5_files = []
    bad_hdf5_files = []
    for hdf5_path in hdf5_files:
        try:
            validate_hdf5_episode(hdf5_path)
            valid_hdf5_files.append(hdf5_path)
        except Exception as e:
            if not args.skip_bad:
                raise
            bad_hdf5_files.append((hdf5_path, e))

    if bad_hdf5_files:
        print(f"跳过 {len(bad_hdf5_files)} 个损坏 episodes:")
        for hdf5_path, err in bad_hdf5_files:
            print(f"  - {hdf5_path.name}: {err}")

    if len(valid_hdf5_files) == 0:
        print("错误：没有可用的 HDF5 文件")
        return

    # 检查第一个有效文件获取维度信息
    sample_data = convert_hdf5_episode(valid_hdf5_files[0])
    state_dim = sample_data['qpos'].shape[1]    # 7
    action_dim = sample_data['action'].shape[1]  # 7
    high_h, high_w = sample_data['cam_high'].shape[1:3]
    wrist_h, wrist_w = sample_data['cam_wrist'].shape[1:3]

    print(f"状态维度: {state_dim}, 动作维度: {action_dim}")
    print(f"顶部相机尺寸: {high_h}x{high_w}")
    print(f"腕部相机尺寸: {wrist_h}x{wrist_w}")
    print(f"目标帧率: {args.fps} Hz")
    print(f"视频编码: {args.vcodec}")
    print(f"流式编码: {args.streaming_encoding}")
    print(f"输入图像通道: {args.input_color_order.upper()} -> RGB")
    print(f"任务描述: \"{args.task}\"")

    # 定义数据集特征
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": ["joint_1", "joint_2", "joint_3", "joint_4",
                       "joint_5", "joint_6", "gripper"],
        },
        "observation.images.cam_high": {
            "dtype": "video",
            "shape": (high_h, high_w, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.cam_wrist": {
            "dtype": "video",
            "shape": (wrist_h, wrist_w, 3),
            "names": ["height", "width", "channels"],
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": ["joint_1", "joint_2", "joint_3", "joint_4",
                       "joint_5", "joint_6", "gripper"],
        },
    }

    # 创建 LeRobot 数据集
    print(f"\n创建数据集: {args.output_dir}")
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=features,
        root=args.output_dir,
        robot_type="realman",
        use_videos=True,
        vcodec=args.vcodec,
        streaming_encoding=args.streaming_encoding,
        encoder_queue_maxsize=args.encoder_queue_maxsize,
        encoder_threads=args.encoder_threads,
    )

    total_frames = 0

    # 逐 episode 转换
    for ep_idx, hdf5_path in enumerate(valid_hdf5_files):
        data = convert_hdf5_episode(hdf5_path)
        num_frames = len(data['qpos'])
        total_frames += num_frames

        for frame_idx in range(num_frames):
            cam_high_rgb = image_to_rgb(
                data['cam_high'][frame_idx], args.input_color_order)
            cam_wrist_rgb = image_to_rgb(
                data['cam_wrist'][frame_idx], args.input_color_order)
            if args.streaming_encoding:
                cam_high = cam_high_rgb
                cam_wrist = cam_wrist_rgb
            else:
                cam_high = Image.fromarray(cam_high_rgb)
                cam_wrist = Image.fromarray(cam_wrist_rgb)

            frame_data = {
                "observation.state": torch.from_numpy(
                    data['qpos'][frame_idx].astype(np.float32)),
                "observation.images.cam_high": cam_high,
                "observation.images.cam_wrist": cam_wrist,
                "action": torch.from_numpy(
                    data['action'][frame_idx].astype(np.float32)),
                "task": args.task,
            }
            dataset.add_frame(frame_data)

        dataset.save_episode()

        if (ep_idx + 1) % 10 == 0 or ep_idx == len(valid_hdf5_files) - 1:
            print(f"  已完成 {ep_idx + 1}/{len(valid_hdf5_files)} episodes ({total_frames} frames)")

    # 结束写入。不同 LeRobot 版本接口不同：
    # 新版使用 finalize() 关闭 parquet/video writer；旧版使用 consolidate()。
    print("\n正在完成数据集写入...")
    if hasattr(dataset, "finalize"):
        dataset.finalize()
    elif hasattr(dataset, "consolidate"):
        dataset.consolidate()

    print(f"\n{'=' * 50}")
    print(f"✓ 数据集转换完成!")
    print(f"{'=' * 50}")
    print(f"  路径: {args.output_dir}")
    print(f"  FPS: {args.fps}")
    print(f"  Episodes: {len(valid_hdf5_files)}")
    print(f"  总帧数: {total_frames}")
    print(f"  Task: \"{args.task}\"")
    print(f"\n下一步: 训练与推理时都使用 --freq {args.fps}")


if __name__ == "__main__":
    main()
