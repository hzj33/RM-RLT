from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping


DEFAULT_FRONT_VIDEO_KEY = "observation.images.cam_front"
DEFAULT_WRIST_VIDEO_KEY = "observation.images.cam_wrist"


def make_video_keys(
    front_video_key: str | None = None,
    wrist_video_key: str | None = None,
) -> dict[str, str]:
    """Return the logical RLT gate camera names to LeRobot video feature keys."""
    return {
        "front": front_video_key or DEFAULT_FRONT_VIDEO_KEY,
        "wrist": wrist_video_key or DEFAULT_WRIST_VIDEO_KEY,
    }


def required_cameras(camera: str) -> tuple[str, ...]:
    if camera == "both":
        return ("front", "wrist")
    if camera in ("front", "wrist"):
        return (camera,)
    raise ValueError(f"Unsupported camera selection: {camera!r}")


def episode_video_path(
    root: Path,
    camera: str,
    episode: int,
    video_keys: dict[str, str] | None = None,
) -> Path:
    keys = video_keys or make_video_keys()
    return root / "videos" / keys[camera] / "chunk-000" / f"file-{episode:03d}.mp4"


def infer_dataset_fps(root: Path, fallback: float = 30.0) -> float:
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        return float(fallback)
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
        fps = float(info.get("fps", fallback))
        if math.isfinite(fps) and fps > 0:
            return fps
    except Exception:
        pass
    return float(fallback)


def video_metadata_columns(video_keys: dict[str, str]) -> list[str]:
    columns: list[str] = []
    for key in video_keys.values():
        prefix = f"videos/{key}"
        columns.extend(
            [
                f"{prefix}/chunk_index",
                f"{prefix}/file_index",
                f"{prefix}/from_timestamp",
                f"{prefix}/to_timestamp",
            ]
        )
    return columns


def _row_value(row: Mapping[str, Any], key: str, default: Any = None) -> Any:
    try:
        value = row[key]
    except Exception:
        return default
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
    except Exception:
        pass
    return value


def video_source_from_episode_row(
    root: Path,
    camera: str,
    episode: int,
    episode_row: Mapping[str, Any],
    video_keys: dict[str, str] | None = None,
    *,
    fps: float,
) -> tuple[Path, int]:
    """Return (video_path, first_frame_offset) for an episode.

    Newer LeRobot datasets store one long video per video file/chunk and record
    per-episode `from_timestamp` metadata.  Older exports may store one mp4 per
    episode.  This helper supports both layouts.
    """
    keys = video_keys or make_video_keys()
    video_key = keys[camera]
    prefix = f"videos/{video_key}"
    file_index = _row_value(episode_row, f"{prefix}/file_index")
    chunk_index = _row_value(episode_row, f"{prefix}/chunk_index", 0)
    from_timestamp = _row_value(episode_row, f"{prefix}/from_timestamp", 0.0)

    if file_index is not None:
        path = root / "videos" / video_key / f"chunk-{int(chunk_index):03d}" / f"file-{int(file_index):03d}.mp4"
        start_frame = int(round(float(from_timestamp) * float(fps)))
        return path, max(0, start_frame)

    return episode_video_path(root, camera, episode, keys), 0


def add_video_key_args(
    parser: argparse.ArgumentParser,
    *,
    defaults_from_checkpoint: bool = False,
) -> None:
    default_front = None if defaults_from_checkpoint else DEFAULT_FRONT_VIDEO_KEY
    default_wrist = None if defaults_from_checkpoint else DEFAULT_WRIST_VIDEO_KEY
    parser.add_argument(
        "--front-video-key",
        default=default_front,
        help=(
            "LeRobot video feature key used as logical front camera. "
            f"Default: {DEFAULT_FRONT_VIDEO_KEY}"
            if defaults_from_checkpoint
            else "LeRobot video feature key used as logical front camera."
        ),
    )
    parser.add_argument(
        "--wrist-video-key",
        default=default_wrist,
        help=(
            "LeRobot video feature key used as logical wrist camera. "
            f"Default: {DEFAULT_WRIST_VIDEO_KEY}"
            if defaults_from_checkpoint
            else "LeRobot video feature key used as logical wrist camera."
        ),
    )
