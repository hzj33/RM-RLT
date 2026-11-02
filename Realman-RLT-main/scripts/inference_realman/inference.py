#!/usr/bin/env python3
"""
通用多策略推理脚本 — RealMan RM65

自动检测策略类型 (ACT / Diffusion / VQ-BeT / SmolVLA / Pi0 / Pi0.5)
加载预处理/后处理管线，实时控制机械臂。

特性:
  - 自动从 config.json 检测策略类型
  - EMA 动作平滑 + 死区过滤（减少抖动）
  - 异步夹爪控制（RealMan 官方夹爪协议，不阻塞主循环）
  - ACT: Temporal Ensemble 支持
  - VLA: Language instruction 支持

用法:
  # ACT
  python scripts/inference_realman/inference.py \\
      --model outputs/act_realman/checkpoints/100000/pretrained_model \\
      --arm-ip <YOUR_ARM_IP> --freq 30

  # SmolVLA (需要 task 描述)
  python scripts/inference_realman/inference.py \\
      --model outputs/smolvla_realman/checkpoints/50000/pretrained_model \\
      --task "pick up the cube" --freq 15

  # Pi0 (离线模式，不下载HuggingFace)
  python scripts/inference_realman/inference.py \\
      --model outputs/pi0_realman/checkpoints/30000/pretrained_model \\
      --task "pick up the cube" --offline
"""

import torch
import numpy as np
import time
import sys
import threading
import cv2
import json
import argparse
import os
from pathlib import Path

# ============ 默认硬件配置（根据你的硬件修改） ============
DEFAULT_ARM_IP = "<YOUR_ARM_IP>"           # 出厂默认 192.168.2.18
DEFAULT_ARM_PORT = 8080
DEFAULT_CAM_TOP_SERIAL = "<YOUR_TOP_CAM_SN>"     # rs-enumerate-devices | grep Serial
DEFAULT_CAM_WRIST_SERIAL = "<YOUR_WRIST_CAM_SN>"
DEFAULT_RM_API_PATH = "/home/vision123/software/remote/Tracker_teleoperate_rm65b/RM_API2/Python"

# RealMan 官方夹爪协议参数
GRIPPER_PORT = 1
GRIPPER_MIN_POSITION = 1
GRIPPER_MAX_POSITION = 1000

# USB 插拔数据的固定起始/复位位姿。最后一维为夹爪值；当前实测 1=打开, 0=闭合。
INIT_POSE = np.array([12.817, 19.830, 82.662, 0.205, 77.997, 42.185, 1.0], dtype=np.float32)


# ============ 图像预处理 ============
def parse_init_qpos(text):
    """Parse a comma/space separated initial qpos string.

    Accepts either 6 joints or 6 joints + gripper.  The gripper defaults to 1.0
    because the RealMan datasets usually start with the gripper open.
    """
    values = [float(item) for item in str(text).replace(",", " ").split()]
    if len(values) not in (6, 7):
        raise argparse.ArgumentTypeError(
            f"--init-qpos expects 6 joints or 6 joints + gripper, got {len(values)} values"
        )
    if len(values) == 6:
        values.append(1.0)
    return np.asarray(values, dtype=np.float32)


def bgr_frame_to_policy_tensor(frame_bgr, channel_order="rgb"):
    """Convert a camera BGR frame to a CHW float tensor for the policy.

    ``channel_order='bgr'`` is a diagnostic mode for legacy RealMan
    checkpoints trained from OpenCV/RealSense BGR frames that were converted
    to LeRobot without BGR->RGB. It keeps the live camera frame in BGR order,
    reproducing the old BGR-as-RGB training distribution while leaving the
    task prompt unchanged.
    """
    channel_order = str(channel_order).lower()
    if channel_order == "rgb":
        image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    elif channel_order == "bgr":
        image = frame_bgr.copy()
    else:
        raise ValueError(f"Unsupported policy image channel order: {channel_order!r}")
    return torch.from_numpy(image).permute(2, 0, 1).float() / 255.0


# ============ 版本兼容性补丁 ============
def _patch_config_compat(config_path):
    """清理 config.json 中当前版本 LeRobot 不支持的字段"""
    import dataclasses
    with open(config_path) as f:
        cfg = json.load(f)

    # 尝试获取对应策略的有效字段
    policy_type = cfg.get("type", "")
    valid_fields = set()

    try:
        if policy_type == "smolvla":
            from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
            valid_fields = {f.name for f in dataclasses.fields(SmolVLAConfig)}
        from lerobot.configs.policies import PreTrainedConfig
        valid_fields |= {f.name for f in dataclasses.fields(PreTrainedConfig)}
    except ImportError:
        return

    removed = []
    for key in list(cfg.keys()):
        if key not in valid_fields and key != "type":
            removed.append(key)
            del cfg[key]

    if removed:
        print(f"  ⚠️  过滤不兼容字段: {removed}")
        with open(config_path, 'w') as f:
            json.dump(cfg, f, indent=4)


# ============ 策略加载器 ============
def load_policy(model_path: str, device: torch.device):
    """自动检测策略类型并加载，返回 (policy, policy_type)"""
    model_path = Path(model_path)
    config_path = model_path / "config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"找不到配置文件: {config_path}")

    with open(config_path) as f:
        config = json.load(f)

    policy_type = config.get("type", "unknown")
    print(f"检测到策略类型: {policy_type}")

    if policy_type == "act":
        from lerobot.policies.act.modeling_act import ACTPolicy
        policy = ACTPolicy.from_pretrained(str(model_path))
        # ACT 特有: Temporal Ensemble 平滑
        if hasattr(policy.config, 'temporal_ensemble_coeff'):
            policy.config.temporal_ensemble_coeff = 0.01
            from lerobot.policies.act.modeling_act import ACTTemporalEnsembler
            policy.temporal_ensembler = ACTTemporalEnsembler(
                temporal_ensemble_coeff=0.01,
                chunk_size=policy.config.chunk_size
            )
            print(f"  Temporal Ensemble: Enabled (coeff=0.01)")

    elif policy_type == "vqbet":
        from lerobot.policies.vqbet.modeling_vqbet import VQBeTPolicy
        policy = VQBeTPolicy.from_pretrained(str(model_path))

    elif policy_type == "diffusion":
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
        policy = DiffusionPolicy.from_pretrained(str(model_path))

    elif policy_type == "smolvla":
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        _patch_config_compat(config_path)
        policy = SmolVLAPolicy.from_pretrained(str(model_path))

    elif policy_type == "pi0":
        from lerobot.policies.pi0.modeling_pi0 import PI0Policy
        policy = PI0Policy.from_pretrained(str(model_path))

    elif policy_type == "pi05":
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        policy = PI05Policy.from_pretrained(str(model_path))

    else:
        raise ValueError(f"不支持的策略类型: {policy_type}")

    policy.to(device)
    policy.eval()
    return policy, policy_type


# ============ 辅助函数 ============
def gripper_value_to_position(value):
    """夹爪值 → RealMan 夹爪位置(1~1000)。当前实测 1=打开, 0=闭合。"""
    value = float(np.clip(value, 0.0, 1.0))
    return int(round(
        GRIPPER_MIN_POSITION +
        value * (GRIPPER_MAX_POSITION - GRIPPER_MIN_POSITION)
    ))


def gripper_position_to_value(position):
    """RealMan 夹爪位置(1~1000) → 夹爪值。当前实测 1=打开, 0=闭合。"""
    position = float(np.clip(position, GRIPPER_MIN_POSITION, GRIPPER_MAX_POSITION))
    return (position - GRIPPER_MIN_POSITION) / (GRIPPER_MAX_POSITION - GRIPPER_MIN_POSITION)


def add_rm_api_path(path):
    """把 RM_API2 Python SDK 加入导入路径。"""
    if path and Path(path).is_dir() and path not in sys.path:
        sys.path.insert(0, path)


# ============ RealSense 相机 ============
import pyrealsense2 as rs


def _parse_video_index(device):
    device = str(device)
    if device.startswith("/dev/video"):
        suffix = device.removeprefix("/dev/video")
        return int(suffix) if suffix.isdigit() else device
    return int(device)


class RealSenseCamera:
    """RealSense 相机异步采集"""

    def __init__(self, serial_number, width=640, height=480, fps=30, exposure=500):
        self.serial_number = str(serial_number)
        self.width, self.height = width, height
        self.exposure = exposure
        self.latest_color = np.zeros((height, width, 3), dtype=np.uint8)
        self.lock = threading.Lock()
        self.stopped = False

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device(self.serial_number)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        try:
            profile = self.pipeline.start(self.config)
            color_sensor = profile.get_device().first_color_sensor()
            if color_sensor.supports(rs.option.enable_auto_exposure):
                color_sensor.set_option(rs.option.enable_auto_exposure, 0)
            if exposure is not None and color_sensor.supports(rs.option.exposure):
                color_sensor.set_option(rs.option.exposure, float(exposure))
            self.thread = threading.Thread(target=self._update_loop, daemon=True)
            self.thread.start()
            print(f"[✓] Camera {self.serial_number} OK ({width}x{height}@{fps}, exp={exposure})")
        except Exception as e:
            print(f"[✗] Camera {self.serial_number}: {e}")
            raise

    def _update_loop(self):
        while not self.stopped:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=2000)
                color_frame = frames.get_color_frame()
                if color_frame:
                    frame_data = np.asanyarray(color_frame.get_data())
                    with self.lock:
                        self.latest_color = frame_data.copy()
            except Exception:
                pass

    def get_frame(self):
        with self.lock:
            return self.latest_color.copy()

    def close(self):
        self.stopped = True
        try:
            self.pipeline.stop()
        except Exception:
            pass


class OpenCVCamera:
    """普通 UVC/OpenCV 相机异步采集"""

    def __init__(self, device, width=640, height=480, fps=30):
        self.device = str(device)
        self.width, self.height = width, height
        self.latest_color = np.zeros((height, width, 3), dtype=np.uint8)
        self.lock = threading.Lock()
        self.stopped = False

        self.cap = cv2.VideoCapture(_parse_video_index(device), cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)

        if not self.cap.isOpened():
            raise RuntimeError(f"Camera {self.device}: OpenCV capture failed")

        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError(f"Camera {self.device}: no frame")

        self.latest_color = cv2.resize(frame, (width, height))
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()
        print(f"[✓] Camera {self.device} OK")

    def _update_loop(self):
        while not self.stopped:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            if frame.shape[:2] != self.latest_color.shape[:2]:
                frame = cv2.resize(frame, (self.width, self.height))
            with self.lock:
                self.latest_color = frame.copy()

    def get_frame(self):
        with self.lock:
            return self.latest_color.copy()

    def close(self):
        self.stopped = True
        try:
            self.cap.release()
        except Exception:
            pass


def create_camera(device, width=640, height=480, fps=30, exposure=500):
    """自动选择相机后端：数字或 /dev/video* 为 OpenCV，其它为 RealSense 序列号。"""
    device_str = str(device)
    if device_str.startswith("/dev/video") or (device_str.isdigit() and len(device_str) <= 2):
        return OpenCVCamera(device_str, width, height, fps)
    return RealSenseCamera(device_str, width, height, fps, exposure=exposure)


# ============ 机械臂控制器 ============
class RobotController:
    """机械臂控制（支持 EMA 平滑 + 死区过滤 + 异步夹爪）

    关键设计:
      - EMA (Exponential Moving Average): 平滑关节角指令，减少抖动
      - 死区过滤: 变化小于阈值时不发送，减少通信压力
      - 异步夹爪: RealMan 官方夹爪指令在独立线程，不阻塞主控制循环
    """

    def __init__(self, arm, joint_command_mode="follow", canfd_follow=False,
                 canfd_trajectory_mode=0, canfd_radio=60, use_live_state=False,
                 sync_gripper=False, binary_gripper=False,
                 gripper_open_threshold=0.6, gripper_close_threshold=0.4,
                 max_joint_step=2.0):
        self.arm = arm
        self.lock = threading.Lock()
        self.gripper_lock = threading.Lock()
        self.joint_command_mode = joint_command_mode
        self.canfd_follow = canfd_follow
        self.canfd_trajectory_mode = canfd_trajectory_mode
        self.canfd_radio = canfd_radio
        self.use_live_state = use_live_state
        self.sync_gripper = sync_gripper
        self.binary_gripper = binary_gripper
        self.gripper_open_threshold = gripper_open_threshold
        self.gripper_close_threshold = gripper_close_threshold
        self.max_joint_step = max_joint_step

        self._last_gripper_cmd = None
        self._cached_gripper_pos = 0.5
        self._pending_gripper_cmd = None
        self._gripper_running = True
        self._gripper_worker = threading.Thread(target=self._gripper_loop, daemon=True)
        self._gripper_worker.start()
        self._estimated_joint = None
        self._smoothed_action = None
        self._last_joint_cmd = None
        self.ema_alpha = 0.3       # EMA 系数 (0.3=平滑, 0.7=响应快)
        self.joint_deadzone = 0.5  # 死区阈值(度)
        self.gripper_deadzone = 0.02

    def get_qpos(self, skip_gripper=True):
        """获取当前 [6关节角 + 夹爪位置]"""
        if not self.use_live_state and self._estimated_joint is not None:
            gripper_pos = 0.5 if self._last_gripper_cmd is None else float(self._last_gripper_cmd)
            return np.array(list(self._estimated_joint) + [gripper_pos], dtype=np.float32)

        with self.lock:
            joint_state = self.arm.rm_get_current_arm_state()
            joint_angles = joint_state[1]['joint'][:6]

            if skip_gripper:
                gripper_pos = 0.5 if self._last_gripper_cmd is None else float(self._last_gripper_cmd)
            else:
                gripper_pos = self._cached_gripper_pos
                try:
                    ret, state = self.arm.rm_get_gripper_state()
                    actpos = state.get('actpos', -1) if ret == 0 else -1
                    if GRIPPER_MIN_POSITION <= actpos <= GRIPPER_MAX_POSITION:
                        gripper_pos = gripper_position_to_value(actpos)
                        self._cached_gripper_pos = gripper_pos
                except Exception:
                    pass

            return np.array(joint_angles + [gripper_pos], dtype=np.float32)

    def set_qpos(self, qpos, use_smoothing=True):
        """设置目标关节角 + 夹爪"""
        with self.lock:
            raw_joint_target = qpos[:6].copy()
            joint_target = qpos[:6].copy()

            # EMA 平滑
            if use_smoothing:
                if self._smoothed_action is None:
                    self._smoothed_action = joint_target.copy()
                else:
                    self._smoothed_action = (
                        self.ema_alpha * joint_target +
                        (1 - self.ema_alpha) * self._smoothed_action
                    )
                joint_target = self._smoothed_action.copy()

            # 硬限幅。VLA 偶尔会输出跨 IK 分支的大跳变，follow 模式不能直接发送。
            if self._last_joint_cmd is not None and self.max_joint_step > 0:
                delta = joint_target - self._last_joint_cmd
                delta = np.clip(delta, -self.max_joint_step, self.max_joint_step)
                joint_target = self._last_joint_cmd + delta
                self._smoothed_action = joint_target.copy()

            # 死区过滤
            should_send = True
            if self._last_joint_cmd is not None:
                max_delta = np.max(np.abs(joint_target - self._last_joint_cmd))
                if max_delta < self.joint_deadzone:
                    should_send = False

            if should_send:
                ret = self._send_joint_target(joint_target)
                if ret != 0:
                    print(f"\n[!] 关节指令失败: mode={self.joint_command_mode}, ret={ret}")
                self._last_joint_cmd = joint_target.copy()
                self._estimated_joint = joint_target.copy()

            # 异步连续夹爪位置控制
            gripper_value = float(np.clip(qpos[6], 0.0, 1.0))
            gripper_value = self._process_gripper_value(gripper_value)
            if (self._last_gripper_cmd is None or
                    abs(gripper_value - self._last_gripper_cmd) >= self.gripper_deadzone):
                self._last_gripper_cmd = gripper_value
                self._cached_gripper_pos = gripper_value
                if self.sync_gripper:
                    ret = self._send_gripper_value(gripper_value)
                    if ret != 0:
                        print(f"\n[!] 夹爪指令失败: value={gripper_value:.2f}, ret={ret}")
                else:
                    with self.gripper_lock:
                        self._pending_gripper_cmd = gripper_value

            return {
                "raw_joint_target": raw_joint_target.copy(),
                "joint_target": joint_target.copy(),
                "sent": should_send,
            }

    def _gripper_loop(self):
        """夹爪后台发送。和关节控制解耦，避免夹爪 SDK 阻塞拖慢主循环。"""
        while self._gripper_running:
            gripper_value = None
            with self.gripper_lock:
                if self._pending_gripper_cmd is not None:
                    gripper_value = self._pending_gripper_cmd
                    self._pending_gripper_cmd = None
            if gripper_value is None:
                time.sleep(0.005)
                continue

            try:
                ret = self._send_gripper_value(gripper_value)
                if ret != 0:
                    print(f"\n[!] 夹爪指令失败: value={gripper_value:.2f}, ret={ret}")
            except Exception:
                pass

    def _send_gripper_value(self, gripper_value):
        position = gripper_value_to_position(gripper_value)
        return self.arm.rm_set_gripper_position(position, False, 0)

    def _process_gripper_value(self, gripper_value):
        if not self.binary_gripper:
            return gripper_value

        if self._last_gripper_cmd is None:
            return 1.0 if gripper_value >= 0.5 else 0.0

        # Hysteresis: avoid rapid open/close toggling when the model output
        # hovers near the decision boundary.
        if self._last_gripper_cmd >= 0.5:
            return 0.0 if gripper_value <= self.gripper_close_threshold else 1.0
        return 1.0 if gripper_value >= self.gripper_open_threshold else 0.0

    def _send_joint_target(self, joint_target):
        joint_list = [float(x) for x in joint_target[:6]]
        if self.joint_command_mode == "follow":
            return self.arm.rm_movej_follow(joint_list)
        if self.joint_command_mode == "canfd":
            return self.arm.rm_movej_canfd(
                joint_list + [0.0],
                self.canfd_follow,
                0,
                self.canfd_trajectory_mode,
                self.canfd_radio,
            )
        return self.arm.rm_movej(joint_list, 50, 0, 0, 0)

    def last_joint_cmd(self):
        with self.lock:
            if self._last_joint_cmd is None:
                return None
            return self._last_joint_cmd.copy()

    def move_to_init(self, init_pose):
        """移动到初始位姿（阻塞）"""
        print(f"移动到初始位姿...")
        self.arm.rm_movej(init_pose[:6].tolist(), 80, 0, 0, 1)
        self._estimated_joint = init_pose[:6].copy()
        self._smoothed_action = init_pose[:6].copy()
        self._last_joint_cmd = init_pose[:6].copy()

        gripper_value = float(np.clip(init_pose[6], 0.0, 1.0))
        position = gripper_value_to_position(gripper_value)
        self.arm.rm_set_gripper_position(position, True, 5)
        self._last_gripper_cmd = gripper_value
        self._cached_gripper_pos = gripper_value
        time.sleep(1.0)

    def seed_current_as_init(self, gripper_value=1.0):
        """Use the current robot joint state as controller's initial state."""
        qpos = self.get_qpos(skip_gripper=True)
        self._estimated_joint = qpos[:6].copy()
        self._smoothed_action = qpos[:6].copy()
        self._last_joint_cmd = qpos[:6].copy()
        gripper_value = float(np.clip(gripper_value, 0.0, 1.0))
        self._last_gripper_cmd = gripper_value
        self._cached_gripper_pos = gripper_value
        print(f"使用当前关节角作为初始位姿: {np.round(qpos[:6], 3).tolist()}, gripper={gripper_value:.2f}")
        return qpos

    def stop(self):
        try:
            self.arm.rm_set_arm_stop()
        except Exception:
            pass

    def close(self):
        self._gripper_running = False
        if self._gripper_worker.is_alive():
            self._gripper_worker.join(timeout=1.0)
        self.arm.rm_delete_robot_arm()


# ============ 主推理循环 ============
def main():
    parser = argparse.ArgumentParser(description='RealMan 通用策略推理')
    parser.add_argument('--model', type=str, required=True,
                        help='模型路径 (如 outputs/act/checkpoints/100000/pretrained_model)')
    parser.add_argument('--arm-ip', type=str, default=DEFAULT_ARM_IP, help='机械臂IP')
    parser.add_argument('--arm-port', type=int, default=DEFAULT_ARM_PORT, help='机械臂端口')
    parser.add_argument('--cam-top', type=str, default=DEFAULT_CAM_TOP_SERIAL, help='顶部相机序列号')
    parser.add_argument('--cam-wrist', type=str, default=DEFAULT_CAM_WRIST_SERIAL, help='腕部相机序列号')
    parser.add_argument('--cam-top-width', type=int, default=640, help='顶部相机宽度')
    parser.add_argument('--cam-top-height', type=int, default=480, help='顶部相机高度')
    parser.add_argument('--cam-top-fps', type=int, default=None, help='顶部相机采集帧率，默认等于 --freq')
    parser.add_argument('--cam-wrist-width', type=int, default=1280, help='腕部 RealSense 宽度')
    parser.add_argument('--cam-wrist-height', type=int, default=720, help='腕部 RealSense 高度')
    parser.add_argument('--cam-wrist-fps', type=int, default=30, help='腕部 RealSense 采集帧率')
    parser.add_argument('--cam-wrist-exposure', type=float, default=500, help='腕部 RealSense 曝光')
    parser.add_argument('--freq', type=float, default=15.0,
                        help='控制频率(Hz)，应与训练数据fps一致')
    parser.add_argument('--task', type=str, default='pick up the cube',
                        help='VLA 任务描述 (Pi0/SmolVLA 需要)')
    parser.add_argument('--headless', action='store_true', help='无GUI模式')
    parser.add_argument('--offline', action='store_true', help='离线模式 (不从HuggingFace下载)')
    parser.add_argument('--dry-run', action='store_true',
                        help='只读相机/状态并运行模型，不移动到初始位姿，也不发送机械臂或夹爪命令')
    parser.add_argument('--init-qpos', type=parse_init_qpos, default=INIT_POSE,
                        help='部署前移动到的初始姿态：6个关节角，或6个关节角+夹爪值；例如 "12.817,19.830,82.662,0.205,77.997,42.185,1"')
    parser.add_argument('--no-move-init', action='store_true',
                        help='启动时不执行回初始位姿运动；只读取当前关节角作为控制器初始状态')
    parser.add_argument('--policy-image-channel-order', choices=['rgb', 'bgr'], default='rgb',
                        help='送入策略的图像通道顺序；bgr 用于诊断旧数据 BGR/RGB 转换错误')
    parser.add_argument('--live-state', action='store_true',
                        help='每帧实时读取机械臂状态；更真实但可能被 SDK 状态查询阻塞')
    parser.add_argument('--joint-command-mode', type=str, default='follow',
                        choices=['follow', 'canfd', 'movej'],
                        help='在线关节控制方式；默认 follow，比 movej 更适合连续推理')
    parser.add_argument('--canfd-follow', action='store_true',
                        help='joint-command-mode=canfd 时使用高跟随；20Hz 下默认关闭更稳')
    parser.add_argument('--canfd-trajectory-mode', type=int, default=0,
                        help='joint-command-mode=canfd 时的轨迹模式：0透传、1曲线拟合、2滤波')
    parser.add_argument('--canfd-radio', type=int, default=60,
                        help='joint-command-mode=canfd 时的平滑系数')
    parser.add_argument('--sync-gripper', action='store_true',
                        help='同步发送夹爪命令；抓取时序更接近策略输出，但夹爪 SDK 阻塞时会拉低循环频率')
    parser.add_argument('--binary-gripper', action='store_true',
                        help='将策略输出的夹爪值离散为 0/1，只在开闭状态变化时发送命令')
    parser.add_argument('--gripper-open-threshold', type=float, default=0.6,
                        help='binary-gripper 下，从闭合切到打开的阈值（默认0.6）')
    parser.add_argument('--gripper-close-threshold', type=float, default=0.4,
                        help='binary-gripper 下，从打开切到闭合的阈值（默认0.4）')
    parser.add_argument('--max-joint-step', type=float, default=2.0,
                        help='每个控制周期允许发送的最大关节变化量(度)，<=0 关闭限幅；默认2.0')
    parser.add_argument('--rm-api-path', type=str, default=os.environ.get('RM_API2_PYTHON_PATH', DEFAULT_RM_API_PATH),
                        help='RM_API2/Python 路径')
    args = parser.parse_args()

    if args.offline:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'

    add_rm_api_path(args.rm_api_path)

    # 导入机械臂SDK
    from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e
    from lerobot.processor.pipeline import DataProcessorPipeline

    print("=" * 50)
    print("  RealMan RM65 通用策略推理")
    print("=" * 50)
    if args.dry_run:
        print("  模式: DRY-RUN（不发送机械臂/夹爪命令）")
    print(f"  策略图像通道: {args.policy_image_channel_order}")

    # 1. 加载模型
    print("\n[1/5] Loading policy...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, policy_type = load_policy(args.model, device)
    print(f"      Device: {device}")

    preprocessor = DataProcessorPipeline.from_pretrained(
        args.model, config_filename='policy_preprocessor.json')
    postprocessor = DataProcessorPipeline.from_pretrained(
        args.model, config_filename='policy_postprocessor.json')

    # 检测输入特征
    input_features = policy.config.input_features
    front_image_key = None
    if 'observation.images.cam_front' in input_features:
        front_image_key = 'observation.images.cam_front'
    elif 'observation.images.cam_high' in input_features:
        front_image_key = 'observation.images.cam_high'
    use_cam_front = front_image_key is not None
    use_cam_wrist = 'observation.images.cam_wrist' in input_features
    print(f"      Input features: {list(input_features.keys())}")
    if front_image_key is not None:
        print(f"      Front/top camera feature: {front_image_key}")

    # 2. 初始化硬件
    print("\n[2/5] Initializing hardware...")
    arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    handle = arm.rm_create_robot_arm(args.arm_ip, args.arm_port)
    if handle.id == -1:
        raise RuntimeError(f"机械臂连接失败: {args.arm_ip}:{args.arm_port}")

    arm.rm_stop_drag_teach()
    arm.rm_set_tool_voltage(3)
    arm.rm_close_modbus_mode(GRIPPER_PORT)
    time.sleep(0.3)

    robot = RobotController(
        arm,
        joint_command_mode=args.joint_command_mode,
        canfd_follow=args.canfd_follow,
        canfd_trajectory_mode=args.canfd_trajectory_mode,
        canfd_radio=args.canfd_radio,
        use_live_state=args.live_state,
        sync_gripper=args.sync_gripper,
        binary_gripper=args.binary_gripper,
        gripper_open_threshold=args.gripper_open_threshold,
        gripper_close_threshold=args.gripper_close_threshold,
        max_joint_step=args.max_joint_step,
    )

    cam_top_fps = args.cam_top_fps if args.cam_top_fps is not None else int(round(args.freq))
    cam_top = create_camera(
        args.cam_top,
        width=args.cam_top_width,
        height=args.cam_top_height,
        fps=cam_top_fps,
    ) if use_cam_front else None
    cam_wrist = create_camera(
        args.cam_wrist,
        width=args.cam_wrist_width,
        height=args.cam_wrist_height,
        fps=args.cam_wrist_fps,
        exposure=args.cam_wrist_exposure,
    ) if use_cam_wrist else None
    time.sleep(1)

    # 3. 移动到初始位姿
    if args.dry_run:
        print("\n[3/5] Dry run: skip moving to initial pose")
    elif args.no_move_init:
        print("\n[3/5] Using current pose as initial pose (no move)")
        robot.seed_current_as_init(gripper_value=float(args.init_qpos[6]))
    else:
        print("\n[3/5] Moving to initial pose...")
        robot.move_to_init(args.init_qpos)

    # 4. 等待确认
    print("\n[4/5] Ready to execute")
    if not args.headless:
        print("      按 Enter 开始推理...")
        input()
    else:
        print("      3秒后开始...")
        time.sleep(3)

    # 5. 推理循环
    print(f"\n[5/5] Executing {policy_type} at {args.freq}Hz (Ctrl+C to stop)...")
    step_count = 0
    control_period = 1.0 / args.freq
    policy.reset()

    try:
        while True:
            start_time = time.time()

            # 获取观测
            state_start = time.time()
            qpos = robot.get_qpos(skip_gripper=True)
            state_dt = time.time() - state_start
            observation = {
                'observation.state': torch.from_numpy(qpos).float(),
            }

            camera_start = time.time()
            if cam_top:
                observation[front_image_key] = bgr_frame_to_policy_tensor(
                    cam_top.get_frame(),
                    channel_order=args.policy_image_channel_order,
                )

            if cam_wrist:
                observation['observation.images.cam_wrist'] = bgr_frame_to_policy_tensor(
                    cam_wrist.get_frame(),
                    channel_order=args.policy_image_channel_order,
                )
            camera_dt = time.time() - camera_start

            # VLA 需要 language instruction
            if policy_type in ("pi0", "pi05", "smolvla"):
                observation['task'] = args.task

            # 预处理 → 推理 → 后处理
            infer_start = time.time()
            observation = preprocessor(observation)

            with torch.no_grad():
                action_tensor = policy.select_action(observation)

            action_dict = postprocessor({'action': action_tensor})
            action = action_dict['action'][0].cpu().numpy()
            infer_dt = time.time() - infer_start

            # 执行
            command_start = time.time()
            if args.dry_run:
                command_info = {
                    "raw_joint_target": action[:6].copy(),
                    "joint_target": qpos[:6].copy(),
                }
            else:
                command_info = robot.set_qpos(action)
            command_dt = time.time() - command_start

            # 日志
            work_elapsed = time.time() - start_time
            sleep_dt = max(0.0, control_period - work_elapsed)
            if step_count % 10 == 0:
                loop_freq = 1.0 / max(control_period, work_elapsed)
                gripper_state = f"{np.clip(action[6], 0.0, 1.0) * 100:.0f}% open"
                sent_joint = command_info["joint_target"]
                raw_joint = command_info["raw_joint_target"]
                mode_text = "DRY" if args.dry_run else "RUN"
                print(f"[{mode_text} {policy_type}] Step {step_count:4d} | "
                      f"J1:{qpos[0]:6.1f}→raw{raw_joint[0]:6.1f}/cmd{sent_joint[0]:6.1f} | "
                      f"夹爪:{gripper_state} | loop {loop_freq:.1f}Hz | "
                      f"state {state_dt*1000:.0f}ms cam {camera_dt*1000:.0f}ms "
                      f"infer {infer_dt*1000:.0f}ms cmd {command_dt*1000:.0f}ms "
                      f"sleep {sleep_dt*1000:.0f}ms")

            # 可视化
            if not args.headless:
                try:
                    frames = []
                    if cam_top:
                        frames.append(cv2.resize(cam_top.get_frame(), (320, 240)))
                    if cam_wrist:
                        frames.append(cv2.resize(cam_wrist.get_frame(), (320, 240)))
                    if frames:
                        display = np.hstack(frames) if len(frames) > 1 else frames[0]
                        cv2.putText(display, f"{policy_type} Step: {step_count}",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.imshow("Inference", display)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                except cv2.error:
                    args.headless = True

            step_count += 1

            # 频率控制
            remaining = control_period - (time.time() - start_time)
            if remaining > 0:
                time.sleep(remaining)

    except KeyboardInterrupt:
        print("\n\n推理终止")
    finally:
        robot.stop()
        try:
            if not args.headless:
                cv2.destroyAllWindows()
        except Exception:
            pass
        robot.close()
        if cam_top:
            cam_top.close()
        if cam_wrist:
            cam_wrist.close()
        print("硬件已关闭")


if __name__ == "__main__":
    main()
