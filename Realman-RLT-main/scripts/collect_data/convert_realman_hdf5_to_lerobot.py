#!/usr/bin/env python3
"""Convert RealMan Quest/Vive HDF5 demonstrations to a LeRobot dataset.

The source format is produced by
``~/software/VLA/lerobot-realman-vla/scripts/collect_data.py``:

    observations/qpos                (N, 7)  [joint_1..joint_6, gripper]
    action                           (N, 7)  next-frame qpos
    observations/images/cam_high     (N, H, W, 3)
    observations/images/cam_wrist    (N, H, W, 3)
    timestamps                       (N,)

This converter keeps the 7D joint action representation.  It maps the high
camera to ``observation.images.cam_front`` by default so the dataset can share
the current ur3e_rl SmolVLA training helpers.  Raw frames collected through the
RealMan reference scripts are OpenCV-style BGR by default, so this converter
normalizes image features to RGB before LeRobot video encoding.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]

STATE_KEY = "observations/qpos"
ACTION_KEY = "action"
HIGH_IMAGE_KEY = "observations/images/cam_high"
WRIST_IMAGE_KEY = "observations/images/cam_wrist"
TIMESTAMP_KEY = "timestamps"


@dataclass(frozen=True)
class EpisodeInfo:
    path: str
    frames: int
    qpos_shape: tuple[int, ...]
    action_shape: tuple[int, ...]
    high_shape: tuple[int, ...]
    wrist_shape: tuple[int, ...]
    fps_attr: float | None
    scalar_shapes: dict[str, tuple[int, ...]]


def _import_deps():
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("Missing dependency: h5py. Run this script in the lerobot/rm_teleop env.") from exc

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Missing dependency: torch. Run this script in the lerobot env.") from exc

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        try:
            from lerobot.datasets import LeRobotDataset
        except ImportError as exc:
            raise RuntimeError("Missing dependency: lerobot. Activate the LeRobot environment first.") from exc

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Missing dependency: pillow.") from exc

    return h5py, torch, LeRobotDataset, Image


def _parse_scalar_mappings(values: list[str] | None) -> list[tuple[str, str]]:
    mappings: list[tuple[str, str]] = []
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"Scalar mapping must be HDF5_KEY=LEROBOT_KEY, got {value!r}")
        source, output = value.split("=", 1)
        source = source.strip()
        output = output.strip()
        if not source or not output:
            raise ValueError(f"Bad scalar mapping: {value!r}")
        mappings.append((source, output))
    return mappings


def _image_to_rgb(image: np.ndarray, input_color_order: str) -> np.ndarray:
    """Return a contiguous RGB image from a raw HDF5 frame."""
    if input_color_order == "rgb":
        return np.ascontiguousarray(image)
    if input_color_order == "bgr":
        return np.ascontiguousarray(image[..., ::-1])
    raise ValueError(f"Unsupported input color order: {input_color_order!r}")


def _load_hdf5(
    path: Path,
    *,
    high_key: str,
    wrist_key: str,
    scalar_mappings: list[tuple[str, str]],
) -> dict[str, np.ndarray]:
    h5py, _torch, _dataset, _image = _import_deps()
    with h5py.File(path, "r") as f:
        data = {
            "qpos": np.asarray(f[STATE_KEY]),
            "action": np.asarray(f[ACTION_KEY]),
            "cam_high": np.asarray(f[high_key]),
            "cam_wrist": np.asarray(f[wrist_key]),
            "timestamps": np.asarray(f[TIMESTAMP_KEY]) if TIMESTAMP_KEY in f else np.arange(f[STATE_KEY].shape[0]),
        }
        data["scalars"] = {output: np.asarray(f[source]) for source, output in scalar_mappings}
        return data


def _validate_hdf5(
    path: Path,
    *,
    high_key: str,
    wrist_key: str,
    scalar_mappings: list[tuple[str, str]],
) -> EpisodeInfo:
    h5py, _torch, _dataset, _image = _import_deps()
    required = [STATE_KEY, ACTION_KEY, high_key, wrist_key, *[source for source, _output in scalar_mappings]]
    with h5py.File(path, "r") as f:
        for key in required:
            if key not in f:
                raise KeyError(f"{path.name}: missing key {key!r}")

        qpos_shape = tuple(int(v) for v in f[STATE_KEY].shape)
        action_shape = tuple(int(v) for v in f[ACTION_KEY].shape)
        high_shape = tuple(int(v) for v in f[high_key].shape)
        wrist_shape = tuple(int(v) for v in f[wrist_key].shape)
        lengths = {
            STATE_KEY: qpos_shape[0],
            ACTION_KEY: action_shape[0],
            high_key: high_shape[0],
            wrist_key: wrist_shape[0],
        }
        scalar_shapes: dict[str, tuple[int, ...]] = {}
        for source, output in scalar_mappings:
            shape = tuple(int(v) for v in f[source].shape)
            if len(shape) == 0:
                raise ValueError(f"{path.name}: scalar key {source!r} must be per-frame, got scalar shape {shape}")
            lengths[source] = shape[0]
            scalar_shapes[output] = shape
        if len(set(lengths.values())) != 1:
            raise ValueError(f"{path.name}: frame count mismatch {lengths}")
        if len(qpos_shape) != 2 or qpos_shape[1] != 7:
            raise ValueError(f"{path.name}: expected qpos shape (N, 7), got {qpos_shape}")
        if len(action_shape) != 2 or action_shape[1] != 7:
            raise ValueError(f"{path.name}: expected action shape (N, 7), got {action_shape}")
        if len(high_shape) != 4 or high_shape[-1] != 3:
            raise ValueError(f"{path.name}: expected high image shape (N,H,W,3), got {high_shape}")
        if len(wrist_shape) != 4 or wrist_shape[-1] != 3:
            raise ValueError(f"{path.name}: expected wrist image shape (N,H,W,3), got {wrist_shape}")

        fps_attr = f.attrs.get("fps", None)
        fps = None if fps_attr is None else float(fps_attr)

    return EpisodeInfo(
        path=str(path),
        frames=qpos_shape[0],
        qpos_shape=qpos_shape,
        action_shape=action_shape,
        high_shape=high_shape,
        wrist_shape=wrist_shape,
        fps_attr=fps,
        scalar_shapes=scalar_shapes,
    )


def _make_features(
    *,
    high_shape: tuple[int, ...],
    wrist_shape: tuple[int, ...],
    front_output_key: str,
    wrist_output_key: str,
    scalar_output_keys: list[str],
    use_videos: bool,
) -> dict[str, Any]:
    joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"]
    high_h, high_w = high_shape[1:3]
    wrist_h, wrist_w = wrist_shape[1:3]
    image_dtype = "video" if use_videos else "image"
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (7,),
            "names": {"motors": joint_names},
        },
        front_output_key: {
            "dtype": image_dtype,
            "shape": (high_h, high_w, 3),
            "names": ["height", "width", "channel"],
        },
        wrist_output_key: {
            "dtype": image_dtype,
            "shape": (wrist_h, wrist_w, 3),
            "names": ["height", "width", "channel"],
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": {"motors": [f"target_{name}" for name in joint_names]},
        },
    }
    for key in scalar_output_keys:
        features[key] = {
            "dtype": "float32",
            "shape": (1,),
            "names": ["value"],
        }
    return features


def _create_dataset(dataset_cls, kwargs: dict[str, Any]):
    optional_keys = [
        "vcodec",
        "streaming_encoding",
        "encoder_queue_maxsize",
        "encoder_threads",
        "image_writer_threads",
    ]
    attempt = dict(kwargs)
    while True:
        try:
            return dataset_cls.create(**attempt)
        except TypeError as exc:
            removed = False
            text = str(exc)
            for key in optional_keys:
                if key in attempt and (key in text or "unexpected keyword" in text):
                    attempt.pop(key)
                    removed = True
                    break
            if not removed:
                raise


def _save_episode(dataset) -> None:
    try:
        dataset.save_episode(parallel_encoding=False)
    except TypeError as exc:
        if "parallel_encoding" not in str(exc):
            raise
        dataset.save_episode()


def _finalize_dataset(dataset) -> None:
    finalize = getattr(dataset, "finalize", None)
    if callable(finalize):
        finalize()
        return
    consolidate = getattr(dataset, "consolidate", None)
    if callable(consolidate):
        consolidate()


def _write_conversion_report(args: argparse.Namespace, episodes: list[EpisodeInfo], output_dir: Path) -> None:
    report = {
        "source": "realman_hdf5",
        "input_dir": str(args.input_dir),
        "output_dir": str(output_dir),
        "repo_id": args.repo_id,
        "fps": args.fps,
        "task": args.task,
        "state_key": STATE_KEY,
        "action_key": ACTION_KEY,
        "high_image_key": args.high_image_key,
        "wrist_image_key": args.wrist_image_key,
        "front_output_key": args.front_output_key,
        "wrist_output_key": args.wrist_output_key,
        "input_color_order": args.input_color_order,
        "output_color_order": "rgb",
        "scalar_mappings": _parse_scalar_mappings(args.scalar_key),
        "episodes": [asdict(ep) for ep in episodes],
    }
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    with (meta_dir / "realman_hdf5_conversion.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert RealMan 7D HDF5 demonstrations to LeRobot format.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing .hdf5 episodes.")
    parser.add_argument("--output-dir", type=Path, required=True, help="LeRobot dataset output directory.")
    parser.add_argument("--repo-id", required=True, help="LeRobot repo_id label, e.g. local/realman_tape_red_block.")
    parser.add_argument("--fps", type=int, default=20, help="Dataset FPS. Must match collection FPS.")
    parser.add_argument("--task", required=True, help="Task string used for VLA training and rollout.")
    parser.add_argument("--robot-type", default="realman_rm65", help="LeRobot robot_type metadata.")
    parser.add_argument("--high-image-key", default=HIGH_IMAGE_KEY)
    parser.add_argument("--wrist-image-key", default=WRIST_IMAGE_KEY)
    parser.add_argument("--front-output-key", default="observation.images.cam_front")
    parser.add_argument("--wrist-output-key", default="observation.images.cam_wrist")
    parser.add_argument(
        "--input-color-order",
        choices=("bgr", "rgb"),
        default="bgr",
        help="Raw HDF5 image channel order. RealMan reference collection stores OpenCV/RealSense frames as BGR by default.",
    )
    parser.add_argument(
        "--scalar-key",
        action="append",
        default=[],
        metavar="HDF5_KEY=LEROBOT_KEY",
        help=(
            "Optional per-frame scalar/label to copy from HDF5 into LeRobot, e.g. "
            "--scalar-key rlt_phase=observation.rlt_phase. Keys must exist in every converted episode. "
            "Use this for newly collected RLT labels; old 7D qpos demos are not given fake labels."
        ),
    )
    parser.add_argument("--no-videos", action="store_true", help="Store image features instead of encoded videos.")
    parser.add_argument("--vcodec", default="h264")
    parser.add_argument("--streaming-encoding", action="store_true")
    parser.add_argument("--encoder-queue-maxsize", type=int, default=2048)
    parser.add_argument("--encoder-threads", type=int, default=None)
    parser.add_argument("--image-writer-threads", type=int, default=4)
    parser.add_argument("--max-episodes", type=int, default=0, help="0 means convert all episodes.")
    parser.add_argument("--skip-bad", action="store_true", help="Skip malformed HDF5 files instead of failing.")
    parser.add_argument("--overwrite", action="store_true", help="Delete output directory if it already exists.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print episode shapes without writing.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser()
    if not output_dir.is_absolute():
        output_dir = (REPO_ROOT / output_dir).resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    hdf5_files = sorted(input_dir.glob("*.hdf5"))
    if args.max_episodes > 0:
        hdf5_files = hdf5_files[: args.max_episodes]
    if not hdf5_files:
        raise FileNotFoundError(f"No .hdf5 files found in {input_dir}")

    valid: list[Path] = []
    infos: list[EpisodeInfo] = []
    bad: list[tuple[Path, Exception]] = []
    scalar_mappings = _parse_scalar_mappings(args.scalar_key)
    for path in hdf5_files:
        try:
            info = _validate_hdf5(
                path,
                high_key=args.high_image_key,
                wrist_key=args.wrist_image_key,
                scalar_mappings=scalar_mappings,
            )
        except Exception as exc:
            if not args.skip_bad:
                raise
            bad.append((path, exc))
            continue
        valid.append(path)
        infos.append(info)

    if bad:
        print(f"[convert] skipped {len(bad)} bad episodes:")
        for path, exc in bad:
            print(f"  - {path.name}: {exc}")
    if not valid:
        raise RuntimeError("No valid HDF5 episodes found.")

    total_frames = sum(info.frames for info in infos)
    print(
        f"[convert] valid_episodes={len(valid)} frames={total_frames} "
        f"fps={args.fps} output={output_dir}"
    )
    print(f"[convert] image_color={args.input_color_order.upper()} -> RGB")
    print(f"[convert] first_episode={infos[0]}")
    if args.dry_run:
        return 0

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output directory exists: {output_dir}. Use --overwrite to replace it.")
        shutil.rmtree(output_dir)

    _h5py, torch, LeRobotDataset, Image = _import_deps()
    use_videos = not args.no_videos
    features = _make_features(
        high_shape=infos[0].high_shape,
        wrist_shape=infos[0].wrist_shape,
        front_output_key=args.front_output_key,
        wrist_output_key=args.wrist_output_key,
        scalar_output_keys=[output for _source, output in scalar_mappings],
        use_videos=use_videos,
    )
    dataset_kwargs = {
        "repo_id": args.repo_id,
        "root": output_dir,
        "fps": int(args.fps),
        "robot_type": args.robot_type,
        "features": features,
        "use_videos": use_videos,
        "vcodec": args.vcodec,
        "streaming_encoding": bool(args.streaming_encoding),
        "encoder_queue_maxsize": int(args.encoder_queue_maxsize),
        "encoder_threads": args.encoder_threads,
        "image_writer_threads": int(args.image_writer_threads),
    }
    dataset = _create_dataset(LeRobotDataset, dataset_kwargs)

    written_frames = 0
    for ep_idx, path in enumerate(valid, start=1):
        data = _load_hdf5(
            path,
            high_key=args.high_image_key,
            wrist_key=args.wrist_image_key,
            scalar_mappings=scalar_mappings,
        )
        frames = int(data["qpos"].shape[0])
        for frame_idx in range(frames):
            high_image = _image_to_rgb(data["cam_high"][frame_idx], args.input_color_order)
            wrist_image = _image_to_rgb(data["cam_wrist"][frame_idx], args.input_color_order)
            if not args.streaming_encoding:
                high_image = Image.fromarray(high_image)
                wrist_image = Image.fromarray(wrist_image)
            frame = {
                "observation.state": torch.from_numpy(data["qpos"][frame_idx].astype(np.float32)),
                args.front_output_key: high_image,
                args.wrist_output_key: wrist_image,
                "action": torch.from_numpy(data["action"][frame_idx].astype(np.float32)),
                "task": args.task,
            }
            for _source, output in scalar_mappings:
                value = np.asarray(data["scalars"][output][frame_idx], dtype=np.float32).reshape(-1)
                if value.size != 1:
                    raise ValueError(f"{path.name}: scalar output {output!r} frame {frame_idx} has shape {value.shape}")
                frame[output] = torch.from_numpy(value.astype(np.float32))
            dataset.add_frame(frame)
        _save_episode(dataset)
        written_frames += frames
        print(f"[convert] episode {ep_idx}/{len(valid)} {path.name}: {frames} frames")

    _finalize_dataset(dataset)
    _write_conversion_report(args, infos, output_dir)
    print(f"[convert] done: episodes={len(valid)} frames={written_frames} root={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
