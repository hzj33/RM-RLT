from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mujoco_env.hardware.realman_api import DEFAULT_RM_API_PATH
from scripts.collect_data.config import CollectConfig


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RealManSmolVLARolloutConfig:
    policy_path: Path | None = None
    output_root: Path = REPO_ROOT / "outputs" / "train"
    task: str = "用夹爪将胶带夹起后放在红色方块上，然后松开夹爪"

    robot_ip: str = "192.168.2.19"
    robot_port: int = 8080
    rm_api_path: str = DEFAULT_RM_API_PATH

    front_image_topic: str = CollectConfig.front_image_topic
    wrist_image_topic: str = CollectConfig.wrist_image_topic
    front_feature_key: str = "observation.images.cam_front"
    wrist_feature_key: str = "observation.images.cam_wrist"

    fps: float = 20.0
    command_hz: float = 20.0
    state_cache_hz: float = 20.0
    inference_mode: str = "sync"
    replan_every_step: bool = False
    execution_horizon: int = 10
    sync_inference_hz: float = 20.0

    max_dt_front_image: float = 0.10
    max_dt_wrist_image: float = 0.10
    max_state_age_s: float = 0.25
    max_action_age_s: float = 1.00
    buffer_maxlen: int = 2000

    command_mode: str = "follow"
    canfd_follow: bool = False
    canfd_trajectory_mode: int = 2
    canfd_radio: int = 60
    max_joint_step_deg: float = 2.0
    joint_deadband_deg: float = 0.05
    joint_ema_alpha: float = 0.30

    gripper_deadband: float = 0.02
    binary_gripper: bool = True
    gripper_open_threshold: float = 0.6
    gripper_close_threshold: float = 0.4
    gripper_open_value: float = 1.0
    gripper_closed_value: float = 0.0
    initial_gripper_value: float = 1.0

    move_home_on_start: bool = False
    home_qpos: tuple[float, float, float, float, float, float, float] = (0.0, 0.0, 90.0, 0.0, 90.0, 30.0, 1.0)
    home_speed: int = 40
    home_settle_s: float = 1.0

    preview: bool = True
    preview_hz: float = 10.0
    resize_to_policy_shape: bool = True
    policy_image_channel_order: str = "rgb"
    allow_dim_mismatch: bool = False
    stop_on_exit: bool = True
    dry_run_log_hz: float = 2.0
    device: str = "cuda"

    hf_home: Path = REPO_ROOT / ".cache" / "huggingface"


CFG = RealManSmolVLARolloutConfig()
