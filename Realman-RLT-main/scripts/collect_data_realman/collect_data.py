#!/usr/bin/env python3
# RealMan USB/RLT data-collection workflow.
"""
数据采集脚本 — RealMan RM65 + Vive/Quest 遥操作 + 双相机

支持五种模式：
  1. Vive 遥操作模式：通过 Vive Tracker 远程控制机械臂
  2. 示教模式 (--teaching)：手动拖动机械臂，无需 Vive
  3. Quest 模式 (--quest)：Quest 右手柄控制末端，右扳机控制夹爪
  4. 键盘模式 (--keyboard)：锁定末端姿态，仅用键盘控制 XYZ 平移
  5. 混合模式 (--vive-quest)：Vive 控制末端，Quest 控制夹爪和录制

采集数据格式：HDF5
  - observations/qpos: (N, 7)   [6关节角 + 夹爪]
  - observations/images/cam_high: (N, H, W, 3)
  - observations/images/cam_wrist: (N, H, W, 3)
  - action: (N, 7)              [下一帧的 qpos]
  - timestamps: (N,)            [相对时间戳(秒)]

用法:
  # Vive 遥操作
  python scripts/collect_data.py --arm-ip <YOUR_ARM_IP> --save-dir data/raw_hdf5 --fps 30

  # 示教模式
  python scripts/collect_data.py --arm-ip <YOUR_ARM_IP> --save-dir data/raw_hdf5 --fps 30 --teaching

  # Quest 手柄遥操作
  python scripts/collect_data.py --arm-ip <YOUR_ARM_IP> --save-dir data/raw_hdf5 --fps 30 --quest

  # 键盘平移控制
  python scripts/collect_data.py --arm-ip <YOUR_ARM_IP> --save-dir data/raw_hdf5 --fps 30 --keyboard

  # Vive + Quest 混合控制
  python scripts/collect_data.py --arm-ip <YOUR_ARM_IP> --save-dir data/raw_hdf5 --fps 30 --vive-quest
"""

import select
import tty
import termios
import time
import threading
import sys
import os
import h5py
import numpy as np
import cv2
import argparse

# ============ 配置区域（根据您的硬件修改） ============
DEFAULT_ARM_IP = "<YOUR_ARM_IP>"           # 出厂默认 192.168.2.18
DEFAULT_ARM_PORT = 8080
DEFAULT_CAM_TOP_SERIAL = "<YOUR_TOP_CAM_SN>"     # rs-enumerate-devices | grep Serial
DEFAULT_CAM_WRIST_SERIAL = "<YOUR_WRIST_CAM_SN>"
DEFAULT_TRACKER_SERIAL = "<YOUR_TRACKER_SN>"   # SteamVR 中 Tracker 序列号
DEFAULT_RM_API_PATH = ""

# RealMan 官方夹爪协议参数
GRIPPER_PORT = 1
GRIPPER_MIN_POSITION = 1
GRIPPER_MAX_POSITION = 1000
GRIPPER_SPEED = 500
GRIPPER_OPEN_VALUE = 1.0
GRIPPER_CLOSED_VALUE = 0.0
QUEST_STOP_LIFT_DISTANCE = 0.10
QUEST_STOP_HOME_JOINTS = [12.817, 19.830, 82.662, 0.205, 77.997, 42.185]

# 机械臂初始位姿（Vive遥操作零点对应的机械臂笛卡尔位姿）
ROBOT_INIT_POS = np.array([-0.218, 0.06, 0.357])
ROBOT_INIT_ORI = np.array([-3.126, 0.001, -0.015])

# Quest3-Teleoperation 中使用的坐标变换：XR headset frame -> world frame
QUEST_R_HEADSET_TO_WORLD = np.array(
    [
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
    ],
    dtype=float,
)
# ============ 配置区域结束 ============


def get_next_filename(save_dir, task_name):
    """获取下一个可用的文件名 (自增编号)"""
    os.makedirs(save_dir, exist_ok=True)
    idx = 0
    while os.path.exists(os.path.join(save_dir, f"{task_name}_{idx}.hdf5")):
        idx += 1
    return os.path.join(save_dir, f"{task_name}_{idx}.hdf5")


def add_rm_api_path(path):
    """把 RM_API2 Python SDK 加入导入路径。"""
    if path and os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)


def parse_joint_list(text):
    values = [float(part.strip()) for part in str(text).split(',') if part.strip()]
    if len(values) != 6:
        raise argparse.ArgumentTypeError(
            f'需要 6 个逗号分隔的关节角，收到 {len(values)} 个: {text!r}'
        )
    return values


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


def gripper_value_is_closed(value):
    return abs(value - GRIPPER_CLOSED_VALUE) <= abs(value - GRIPPER_OPEN_VALUE)


class RealManGripper:
    """RealMan 官方夹爪协议封装。

    夹爪开闭统一走 rm_set_gripper_position；当前实测 1=打开、0=闭合。
    """

    def __init__(self, arm, arm_lock):
        self.arm = arm
        self.arm_lock = arm_lock
        self._value = 0.0
        self._command_lock = threading.Lock()
        self._pending_command = None
        self._worker_running = True
        self._worker = threading.Thread(target=self._command_loop, daemon=True)
        self._worker.start()

    def initialize(self):
        with self.arm_lock:
            self.arm.rm_close_modbus_mode(GRIPPER_PORT)
            self.arm.rm_set_tool_voltage(3)
        time.sleep(0.3)
        self.refresh_from_state()

    def refresh_from_state(self):
        try:
            with self.arm_lock:
                ret, state = self.arm.rm_get_gripper_state()
            actpos = state.get('actpos', -1) if ret == 0 else -1
            if GRIPPER_MIN_POSITION <= actpos <= GRIPPER_MAX_POSITION:
                self._value = gripper_position_to_value(actpos)
        except Exception:
            pass
        return self._value

    def get_value(self):
        return self._value

    def _send_value(self, value, block=False, timeout=0):
        position = gripper_value_to_position(value)
        with self.arm_lock:
            ret = self.arm.rm_set_gripper_position(position, block, timeout)
        if ret == 0:
            self._value = value
        return ret

    def set_value(self, value, block=False, timeout=0):
        value = float(np.clip(value, 0.0, 1.0))
        if not block:
            self._value = value
            with self._command_lock:
                self._pending_command = (value, block, timeout)
            return 0
        return self._send_value(value, block=block, timeout=timeout)

    def open(self, block=False, timeout=0):
        return self.set_value(GRIPPER_OPEN_VALUE, block=block, timeout=timeout)

    def close(self, block=False, timeout=0):
        return self.set_value(GRIPPER_CLOSED_VALUE, block=block, timeout=timeout)

    def _command_loop(self):
        while self._worker_running:
            command = None
            with self._command_lock:
                if self._pending_command is not None:
                    command = self._pending_command
                    self._pending_command = None
            if command is None:
                time.sleep(0.005)
                continue

            value, block, timeout = command
            ret = self._send_value(value, block=block, timeout=timeout)
            if ret != 0:
                print(f"\n[!] 夹爪命令发送失败: value={value:.2f}, ret={ret}")

    def shutdown(self):
        self._worker_running = False
        if self._worker.is_alive():
            self._worker.join(timeout=1.0)


# ============ RealSense 相机模块 ============
import pyrealsense2 as rs


def _parse_video_index(device):
    device = str(device)
    if device.startswith("/dev/video"):
        suffix = device.removeprefix("/dev/video")
        return int(suffix) if suffix.isdigit() else device
    return int(device)


def _is_opencv_device(device):
    device_str = str(device)
    return device_str.startswith("/dev/video") or (device_str.isdigit() and len(device_str) <= 2)


def reset_realsense_device(serial_number, wait_seconds=6.0):
    """Reset a RealSense camera and wait for USB re-enumeration."""
    serial_number = str(serial_number)
    wait_seconds = max(0.0, float(wait_seconds))

    try:
        ctx = rs.context()
        target = None
        for dev in ctx.query_devices():
            try:
                if dev.get_info(rs.camera_info.serial_number) == serial_number:
                    target = dev
                    break
            except Exception:
                continue

        if target is None:
            print(f"[!] D435 reset: 未找到序列号 {serial_number}，跳过复位")
            return False

        name = target.get_info(rs.camera_info.name)
        print(f"D435 reset: {name} ({serial_number}) hardware_reset()")
        target.hardware_reset()

        deadline = time.time() + wait_seconds
        while time.time() < deadline:
            time.sleep(0.5)
            try:
                ctx = rs.context()
                for dev in ctx.query_devices():
                    if dev.get_info(rs.camera_info.serial_number) == serial_number:
                        print(f"D435 reset: 设备已重新枚举，等待稳定中...")
                        time.sleep(1.0)
                        return True
            except Exception:
                pass

        print(f"[!] D435 reset: {wait_seconds:.1f}s 内未确认重新枚举，继续尝试启动相机")
        return False
    except Exception as e:
        print(f"[!] D435 reset failed: {e}")
        return False


class RealSenseCamera:
    """RealSense 相机异步采集

    使用独立线程持续读取帧，get_frame() 返回最新帧（零拷贝代价）。
    关闭自动曝光以确保采集一致性。
    """

    def __init__(self, serial_number, width=640, height=480, fps=30, exposure=150):
        self.serial_number = str(serial_number)
        self.width, self.height = width, height
        self.fps = fps
        self.exposure = exposure
        self.target_shape = (height, width, 3)
        self.latest_color = np.zeros(self.target_shape, dtype=np.uint8)
        self.latest_frame_number = None
        self.last_receive_time = 0.0
        self.last_unique_frame_time = 0.0
        self.repeated_frame_count = 0
        self.lock = threading.Lock()
        self.stopped = False
        self.is_active = False

        self.pipeline = None
        self.config = None
        self.thread = None
        self._start_pipeline()

    def _reset_cached_frame(self):
        with self.lock:
            self.latest_color = np.zeros(self.target_shape, dtype=np.uint8)
            self.latest_frame_number = None
            self.last_receive_time = 0.0
            self.last_unique_frame_time = 0.0
            self.repeated_frame_count = 0

    def _build_pipeline(self):
        pipeline = rs.pipeline()
        config = rs.config()
        if self.serial_number:
            config.enable_device(self.serial_number)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        return pipeline, config

    def _start_pipeline(self):
        try:
            self.pipeline, self.config = self._build_pipeline()
            profile = self.pipeline.start(self.config)
            self.is_active = True

            # 固定曝光（避免亮度波动影响训练）
            color_sensor = profile.get_device().first_color_sensor()
            if color_sensor.supports(rs.option.enable_auto_exposure):
                color_sensor.set_option(rs.option.enable_auto_exposure, 0)
            if self.exposure is not None and color_sensor.supports(rs.option.exposure):
                try:
                    color_sensor.set_option(rs.option.exposure, float(self.exposure))
                except Exception as e:
                    print(f"[!] Camera {self.serial_number}: exposure set failed ({e})")

            self.thread = threading.Thread(target=self._update_loop, daemon=True)
            self.thread.start()
        except Exception as e:
            print(f"[!] Camera {self.serial_number}: {e}")
            self.is_active = False

    def restart_pipeline(self, wait_seconds=3.0):
        """Restart only the RealSense pipeline, without USB hardware reset."""
        print(f"D435 pipeline restart: {self.serial_number}")
        self.is_active = False
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        try:
            if self.pipeline is not None:
                self.pipeline.stop()
        except Exception:
            pass

        self._reset_cached_frame()
        self._start_pipeline()
        return self.wait_for_valid_frame(timeout=wait_seconds)

    def _update_loop(self):
        while not self.stopped and self.is_active:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=2000)
                color_frame = frames.get_color_frame()
                if color_frame:
                    now = time.time()
                    frame_number = int(color_frame.get_frame_number())
                    frame_data = np.asanyarray(color_frame.get_data())
                    if frame_data.shape == self.target_shape:
                        with self.lock:
                            self.last_receive_time = now
                            if frame_number != self.latest_frame_number:
                                self.latest_frame_number = frame_number
                                self.last_unique_frame_time = now
                                self.repeated_frame_count = 0
                            else:
                                self.repeated_frame_count += 1
                            self.latest_color = frame_data.copy()
            except Exception:
                pass

    def get_frame(self):
        with self.lock:
            return self.latest_color.copy()

    def wait_for_valid_frame(self, timeout=3.0, min_mean=1.0):
        """Wait until the camera has a fresh non-black frame."""
        deadline = time.time() + max(0.0, float(timeout))
        while time.time() < deadline:
            health = self.get_health(max_stale_seconds=max(0.5, float(timeout)))
            frame = self.get_frame()
            if health.get("ok") and frame.mean() > min_mean and frame.std() > 1.0:
                return True
            time.sleep(0.05)
        return False

    def get_health(self, max_stale_seconds=2.0):
        """Return stream health based on RealSense frame numbers, not image motion."""
        now = time.time()
        with self.lock:
            unique_age = now - self.last_unique_frame_time if self.last_unique_frame_time else None
            receive_age = now - self.last_receive_time if self.last_receive_time else None
            ok = (
                self.is_active
                and unique_age is not None
                and unique_age <= max_stale_seconds
            )
            return {
                "ok": ok,
                "active": self.is_active,
                "frame_number": self.latest_frame_number,
                "unique_age": unique_age,
                "receive_age": receive_age,
                "repeated_frame_count": self.repeated_frame_count,
            }

    def close(self):
        self.stopped = True
        if self.is_active:
            try:
                self.pipeline.stop()
            except Exception:
                pass


class OpenCVCamera:
    """普通 UVC/OpenCV 相机异步采集"""

    def __init__(self, device, width=640, height=480, fps=30, exposure=None):
        self.device = str(device)
        self.width, self.height = width, height
        self.target_shape = (height, width, 3)
        self.latest_color = np.zeros(self.target_shape, dtype=np.uint8)
        self.latest_frame_number = 0
        self.last_receive_time = 0.0
        self.last_unique_frame_time = 0.0
        self.lock = threading.Lock()
        self.stopped = False
        self.is_active = False

        self.cap = cv2.VideoCapture(_parse_video_index(device), cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        if exposure is not None:
            self.cap.set(cv2.CAP_PROP_EXPOSURE, float(exposure))

        if self.cap.isOpened():
            ok, frame = self.cap.read()
            if ok:
                self.latest_color = cv2.resize(frame, (width, height))
                now = time.time()
                self.latest_frame_number = 1
                self.last_receive_time = now
                self.last_unique_frame_time = now
                self.is_active = True
                self.thread = threading.Thread(target=self._update_loop, daemon=True)
                self.thread.start()

        if not self.is_active:
            print(f"[!] Camera {self.device}: OpenCV capture failed")

    def _update_loop(self):
        while not self.stopped and self.is_active:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            if frame.shape[:2] != self.target_shape[:2]:
                frame = cv2.resize(frame, (self.width, self.height))
            with self.lock:
                now = time.time()
                self.latest_frame_number += 1
                self.last_receive_time = now
                self.last_unique_frame_time = now
                self.latest_color = frame.copy()

    def get_frame(self):
        with self.lock:
            return self.latest_color.copy()

    def get_health(self, max_stale_seconds=2.0):
        now = time.time()
        with self.lock:
            unique_age = now - self.last_unique_frame_time if self.last_unique_frame_time else None
            receive_age = now - self.last_receive_time if self.last_receive_time else None
            ok = (
                self.is_active
                and unique_age is not None
                and unique_age <= max_stale_seconds
            )
            return {
                "ok": ok,
                "active": self.is_active,
                "frame_number": self.latest_frame_number,
                "unique_age": unique_age,
                "receive_age": receive_age,
                "repeated_frame_count": 0,
            }

    def close(self):
        self.stopped = True
        try:
            self.cap.release()
        except Exception:
            pass


def create_camera(device, width=640, height=480, fps=30, exposure=None):
    """自动选择相机后端：数字或 /dev/video* 为 OpenCV，其它为 RealSense 序列号。"""
    if _is_opencv_device(device):
        return OpenCVCamera(str(device), width, height, fps, exposure=exposure)
    return RealSenseCamera(str(device), width, height, fps, exposure=150 if exposure is None else exposure)


# ============ Vive 遥控模块 ============
class ViveController:
    """Vive Tracker 遥操作控制器

    原理：读取 Tracker 的笛卡尔位姿增量，映射到机械臂的笛卡尔空间。
    坐标映射关系（Vive → Robot）：
      Robot_X = -Vive_Z
      Robot_Y = -Vive_X
      Robot_Z = +Vive_Y
    """

    def __init__(
        self,
        arm,
        arm_lock,
        tracker_serial=None,
        enable_vive=True,
        lock_orientation=False,
        yaw_correction_deg=0.0,
    ):
        self.arm = arm
        self.arm_lock = arm_lock
        self.tracker_serial = tracker_serial or DEFAULT_TRACKER_SERIAL
        self.enable_vive = enable_vive
        self.lock_orientation = lock_orientation
        self.yaw_correction_rad = np.deg2rad(float(yaw_correction_deg))

        self.robot_init_pos = ROBOT_INIT_POS.copy()
        self.robot_init_ori = ROBOT_INIT_ORI.copy()

        self.vive_init_pos = None
        self.vive_init_ori = None
        self.control_enabled = False
        self.running = True
        self.vive = None
        self.tracker = None

        if self.enable_vive:
            self._init_vive()
            self.thread = threading.Thread(target=self._control_loop, daemon=True)
            self.thread.start()
        else:
            print("示教模式: 不使用 Vive 遥控，手动移动机械臂")

    def _init_vive(self):
        try:
            # 需要 hardware/vive_tracker.py
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'hardware'))
            import vive_tracker as triad_lib
            self.vive = triad_lib.triad_openvr()
            for name, device in self.vive.devices.items():
                if "tracker" in name.lower():
                    serial = device.get_serial()
                    if serial == self.tracker_serial:
                        self.tracker = device
                        break
            if self.tracker is None:
                for name, device in self.vive.devices.items():
                    if "tracker" in name.lower():
                        self.tracker = device
                        break
        except Exception as e:
            print(f"[!] Vive: {e}")

    def _get_current_robot_pose(self):
        try:
            with self.arm_lock:
                code, state = self.arm.rm_get_current_arm_state()
            if code != 0 or not isinstance(state, dict):
                return None

            for key in ("pose", "tool_pose", "waypoint"):
                pose = state.get(key)
                if pose is None:
                    continue
                pose = np.asarray(pose, dtype=float).reshape(-1)
                if pose.size >= 6 and np.all(np.isfinite(pose[:6])):
                    return pose[:3].copy(), pose[3:6].copy()
        except Exception:
            pass
        return None

    def _apply_yaw_correction(self, delta_xyz):
        if abs(self.yaw_correction_rad) < 1e-9:
            return delta_xyz
        c = np.cos(self.yaw_correction_rad)
        s = np.sin(self.yaw_correction_rad)
        corrected = delta_xyz.copy()
        corrected[:2] = np.array([
            c * delta_xyz[0] - s * delta_xyz[1],
            s * delta_xyz[0] + c * delta_xyz[1],
        ])
        return corrected

    def calibrate(self):
        """校准：记录当前 Vive Tracker 位姿作为零点"""
        if self.tracker is None:
            print("[!] Vive 未连接")
            return False

        print("校准中... 保持 Tracker 静止")
        poses = []
        for _ in range(30):
            euler = self.tracker.get_pose_euler()
            if euler:
                poses.append(euler)
            time.sleep(0.033)

        if not poses:
            print("[!] 校准失败")
            return False

        avg_pose = np.mean(poses, axis=0)
        self.vive_init_pos = np.array([avg_pose[0], avg_pose[1], avg_pose[2]])
        self.vive_init_ori = np.array([avg_pose[3], avg_pose[4], avg_pose[5]])
        print("校准完成")
        return True

    def enable(self):
        if self.vive_init_pos is None:
            print("请先校准(c)")
            return
        current_pose = self._get_current_robot_pose()
        if current_pose is not None:
            self.robot_init_pos, current_ori = current_pose
            if self.lock_orientation:
                self.robot_init_ori = current_ori
        else:
            print("[!] 读取当前机械臂末端位姿失败，继续使用上一次遥控基准")
        self.control_enabled = True
        if self.lock_orientation:
            print("遥控已启用（当前位置为基准，姿态锁定）")
        else:
            print("遥控已启用（当前位置为基准）")

    def disable(self):
        self.control_enabled = False
        print("遥控已暂停")

    def _control_loop(self):
        """20Hz 遥控循环"""
        interval = 0.05

        while self.running:
            time.sleep(interval)
            if not self.control_enabled or self.tracker is None or self.vive_init_pos is None:
                continue

            try:
                euler = self.tracker.get_pose_euler()
                if euler is None:
                    continue

                cur_pos = np.array([euler[0], euler[1], euler[2]])
                cur_ori = np.array([euler[3], euler[4], euler[5]])

                delta_pos = cur_pos - self.vive_init_pos
                delta_ori = cur_ori - self.vive_init_ori

                scale_pos = 0.5
                scale_ori = 0.3

                # 坐标映射: Vive → Robot，再按现场坐标系绕 Z 轴做 yaw 校正。
                delta_robot = np.array([
                    -delta_pos[2] * scale_pos,
                    -delta_pos[0] * scale_pos,
                    delta_pos[1] * scale_pos
                ])
                delta_robot = self._apply_yaw_correction(delta_robot)
                target_pos = self.robot_init_pos + delta_robot

                if self.lock_orientation:
                    target_ori = self.robot_init_ori.copy()
                else:
                    target_ori = self.robot_init_ori + np.array([
                        -delta_ori[2] * scale_ori * np.pi / 180,
                        -delta_ori[1] * scale_ori * np.pi / 180,
                        delta_ori[0] * scale_ori * np.pi / 180
                    ])

                # 安全限位
                target_pos[0] = np.clip(target_pos[0], -0.5, 0.1)
                target_pos[1] = np.clip(target_pos[1], -0.3, 0.3)
                target_pos[2] = np.clip(target_pos[2], 0.1, 0.6)

                target_6d = [
                    float(target_pos[0]), float(target_pos[1]), float(target_pos[2]),
                    float(target_ori[0]), float(target_ori[1]), float(target_ori[2])
                ]

                with self.arm_lock:
                    self.arm.rm_movep_canfd(target_6d, False, 0, 60)

            except Exception:
                pass

    def shutdown(self):
        self.running = False
        if hasattr(self, "thread"):
            self.thread.join(timeout=1.0)


# ============ Quest 遥控模块 ============
class QuestController:
    """Quest 手柄遥操作控制器。

    控制约定来自 Quest3-Teleoperation：
      - 右 Grip：按住后用右手柄增量控制机械臂末端
      - 右 Trigger：控制夹爪；toggle 模式下按一下切换开/闭
      - 左 Grip：开始录制
      - 左 Trigger：停止并保存
    """

    def __init__(
        self,
        arm,
        arm_lock,
        gripper,
        start_recording=None,
        stop_recording=None,
        pos_scale=1.2,
        rot_scale=0.3,
        yaw_correction_deg=45.0,
        control_input="right_grip",
        grip_threshold=0.8,
        button_threshold=0.8,
        button_debounce=0.3,
        gripper_deadband=0.02,
        lock_orientation=False,
        enable_gripper=True,
        invert_trigger=False,
        gripper_mode="toggle",
        gripper_init="keep",
        enable_motion=True,
        calibrate_callback=None,
        enable_callback=None,
        disable_callback=None,
        post_stop_callback=None,
        debug=False,
    ):
        self.arm = arm
        self.arm_lock = arm_lock
        self.gripper = gripper
        self.start_recording = start_recording
        self.stop_recording = stop_recording
        self.pos_scale = pos_scale
        self.rot_scale = rot_scale
        self.yaw_correction_rad = np.deg2rad(float(yaw_correction_deg))
        self.control_input = control_input
        self.grip_threshold = grip_threshold
        self.button_threshold = button_threshold
        self.button_debounce = button_debounce
        self.gripper_deadband = gripper_deadband
        self.lock_orientation = lock_orientation
        self.enable_gripper = enable_gripper
        self.invert_trigger = invert_trigger
        self.gripper_mode = gripper_mode
        self.gripper_init = gripper_init
        self.enable_motion = enable_motion
        self.calibrate_callback = calibrate_callback
        self.enable_callback = enable_callback
        self.disable_callback = disable_callback
        self.post_stop_callback = post_stop_callback
        self.debug = debug

        self.running = True
        self.available = False
        self.active = False
        self.motion_blocked_until_control_release = False
        self.prev_controller_xyz = None
        self.prev_controller_rot = None
        self.target_pos = ROBOT_INIT_POS.copy()
        self.target_ori = ROBOT_INIT_ORI.copy()
        self.gripper_baseline = None
        self.gripper_armed = False
        self.gripper_toggle_closed = False
        self.trigger_was_pressed = False
        self.last_gripper_toggle = 0.0
        self.last_gripper_value = None
        self.last_gripper_send = 0.0
        self.last_start_press = 0.0
        self.last_stop_press = 0.0
        self.last_calibrate_press = 0.0
        self.last_enable_press = 0.0
        self.last_disable_press = 0.0
        self.left_grip_was_pressed = False
        self.left_trigger_was_pressed = False
        self.right_grip_was_pressed = False
        self.a_button_was_pressed = False
        self.b_button_was_pressed = False
        self.last_debug_print = 0.0

        try:
            import xrobotoolkit_sdk as xrt

            xrt.init()
            self.xrt = xrt
            if self.enable_motion:
                from scipy.spatial.transform import Rotation
                self.Rotation = Rotation
                self.headset_rot = Rotation.from_matrix(QUEST_R_HEADSET_TO_WORLD)
            else:
                self.Rotation = None
                self.headset_rot = None
            self.available = True
            self._initialize_gripper()
            self.thread = threading.Thread(target=self._control_loop, daemon=True)
            self.thread.start()
            print("Quest: OK")
        except Exception as e:
            self.xrt = None
            self.Rotation = None
            print(f"[!] Quest SDK 未就绪: {e}")
            print("    请先在 lerobot 环境安装/启动 xrobotoolkit_sdk 与 Quest 端服务")

    def _get_key(self, name):
        if name == "left_trigger":
            return float(self.xrt.get_left_trigger())
        if name == "right_trigger":
            return float(self.xrt.get_right_trigger())
        if name == "left_grip":
            return float(self.xrt.get_left_grip())
        if name == "right_grip":
            return float(self.xrt.get_right_grip())
        if name == "a_button":
            return float(self.xrt.get_A_button())
        if name == "b_button":
            return float(self.xrt.get_B_button())
        if name == "x_button":
            return float(self.xrt.get_X_button())
        if name == "y_button":
            return float(self.xrt.get_Y_button())
        if name == "right_axis_click":
            return float(self.xrt.get_right_axis_click())
        if name == "left_axis_click":
            return float(self.xrt.get_left_axis_click())
        return 0.0

    def _get_trigger_gripper_value(self):
        raw_value = float(np.clip(self._get_key("right_trigger"), 0.0, 1.0))
        return 1.0 - raw_value if self.invert_trigger else raw_value

    def _initialize_gripper(self):
        if not self.enable_gripper:
            return

        trigger_value = self._get_trigger_gripper_value()
        self.gripper_baseline = trigger_value

        if self.gripper_init == "open":
            ret = self.gripper.open()
            if ret == 0:
                self.last_gripper_value = GRIPPER_OPEN_VALUE
                self.gripper_toggle_closed = False
            print(f"Quest 夹爪初始化: 打开 ({'OK' if ret == 0 else 'FAIL'})")
        elif self.gripper_init == "trigger":
            ret = self.gripper.set_value(trigger_value)
            if ret == 0:
                self.last_gripper_value = trigger_value
                self.gripper_toggle_closed = gripper_value_is_closed(trigger_value)
            print(f"Quest 夹爪初始化: 跟随右扳机 {trigger_value:.2f} ({'OK' if ret == 0 else 'FAIL'})")
        else:
            self.last_gripper_value = self.gripper.get_value()
            self.gripper_toggle_closed = gripper_value_is_closed(self.last_gripper_value)

    def _get_right_controller_pose(self):
        pose = np.asarray(self.xrt.get_right_controller_pose(), dtype=float).reshape(-1)
        if pose.size < 7 or not np.all(np.isfinite(pose[:7])) or np.allclose(pose[:7], 0.0):
            return None, None

        xyz = QUEST_R_HEADSET_TO_WORLD @ pose[:3]
        raw_rot = self.Rotation.from_quat(pose[3:7])  # SDK: x,y,z,w
        rot = self.headset_rot * raw_rot * self.headset_rot.inv()
        return xyz, rot

    def _get_current_robot_pose(self):
        try:
            with self.arm_lock:
                code, state = self.arm.rm_get_current_arm_state()
            if code != 0 or not isinstance(state, dict):
                return None

            for key in ("pose", "tool_pose", "waypoint"):
                pose = state.get(key)
                if pose is None:
                    continue
                pose = np.asarray(pose, dtype=float).reshape(-1)
                if pose.size >= 6 and np.all(np.isfinite(pose[:6])):
                    return pose[:3].copy(), pose[3:6].copy()
        except Exception:
            pass
        return None

    def _apply_yaw_correction(self, delta_xyz):
        if abs(self.yaw_correction_rad) < 1e-9:
            return delta_xyz
        c = np.cos(self.yaw_correction_rad)
        s = np.sin(self.yaw_correction_rad)
        corrected = delta_xyz.copy()
        corrected[:2] = np.array([
            c * delta_xyz[0] - s * delta_xyz[1],
            s * delta_xyz[0] + c * delta_xyz[1],
        ])
        return corrected

    def _activate(self):
        current_pose = self._get_current_robot_pose()
        if current_pose is not None:
            self.target_pos, self.target_ori = current_pose
        self.prev_controller_xyz = None
        self.prev_controller_rot = None
        self.active = True
        print("\nQuest 遥控: 已启用")

    def _deactivate(self):
        self.active = False
        self.prev_controller_xyz = None
        self.prev_controller_rot = None
        print("\nQuest 遥控: 已暂停")

    def pause_motion(self, print_result=False):
        self.active = False
        self.prev_controller_xyz = None
        self.prev_controller_rot = None
        if print_result:
            print("\nQuest 遥控: 已暂停")

    def sync_to_current_robot_pose(self, lock_orientation=None):
        current_pose = self._get_current_robot_pose()
        if current_pose is None:
            return False

        self.target_pos, self.target_ori = current_pose
        self.prev_controller_xyz = None
        self.prev_controller_rot = None
        if lock_orientation is not None:
            self.lock_orientation = bool(lock_orientation)
        return True

    def _handle_buttons(self, now):
        left_grip = self._get_key("left_grip")
        left_trigger = self._get_key("left_trigger")
        right_grip = self._get_key("right_grip")
        a_button = self._get_key("a_button")
        b_button = self._get_key("b_button")
        left_grip_pressed = left_grip > self.button_threshold
        left_trigger_pressed = left_trigger > self.button_threshold
        right_grip_pressed = right_grip > self.button_threshold
        a_button_pressed = a_button > self.button_threshold
        b_button_pressed = b_button > self.button_threshold

        if left_grip_pressed and not self.left_grip_was_pressed and now - self.last_start_press > self.button_debounce:
            self.last_start_press = now
            if self.start_recording is not None:
                self.start_recording()

        if left_trigger_pressed and not self.left_trigger_was_pressed and now - self.last_stop_press > self.button_debounce:
            self.last_stop_press = now
            did_stop = True
            if self.stop_recording is not None:
                did_stop = self.stop_recording()
            if did_stop and self.post_stop_callback is not None:
                self.pause_motion()
                self.motion_blocked_until_control_release = True
                self.post_stop_callback()

        if right_grip_pressed and not self.right_grip_was_pressed and now - self.last_calibrate_press > self.button_debounce:
            self.last_calibrate_press = now
            if self.calibrate_callback is not None:
                self.calibrate_callback()

        if b_button_pressed and not self.b_button_was_pressed and now - self.last_enable_press > self.button_debounce:
            self.last_enable_press = now
            if self.enable_callback is not None:
                self.enable_callback()

        if a_button_pressed and not self.a_button_was_pressed and now - self.last_disable_press > self.button_debounce:
            self.last_disable_press = now
            if self.disable_callback is not None:
                self.disable_callback()

        self.left_grip_was_pressed = left_grip_pressed
        self.left_trigger_was_pressed = left_trigger_pressed
        self.right_grip_was_pressed = right_grip_pressed
        self.a_button_was_pressed = a_button_pressed
        self.b_button_was_pressed = b_button_pressed

    def _handle_gripper(self, now):
        if not self.enable_gripper:
            return

        if self.gripper_mode == "toggle":
            self._handle_gripper_toggle(now)
            return

        self._handle_gripper_continuous(now)

    def _handle_gripper_toggle(self, now):
        pressed = self._get_key("right_trigger") > 0.5
        rising_edge = pressed and not self.trigger_was_pressed
        self.trigger_was_pressed = pressed

        if not rising_edge or now - self.last_gripper_toggle < 0.25:
            return

        if self.gripper_toggle_closed:
            ret = self.gripper.open()
            target_value = GRIPPER_OPEN_VALUE
            action = "打开"
        else:
            ret = self.gripper.close()
            target_value = GRIPPER_CLOSED_VALUE
            action = "闭合"

        self.last_gripper_toggle = now
        if ret == 0:
            self.gripper_toggle_closed = not self.gripper_toggle_closed
            self.last_gripper_value = target_value
        print(f"\nQuest 夹爪: {action} ({'OK' if ret == 0 else 'FAIL'})")

    def _handle_gripper_continuous(self, now):
        value = self._get_trigger_gripper_value()

        if self.gripper_baseline is None:
            self.gripper_baseline = value
            self.last_gripper_value = self.gripper.get_value()
            return

        if not self.gripper_armed:
            if abs(value - self.gripper_baseline) < self.gripper_deadband:
                return
            self.gripper_armed = True
            print("\nQuest 夹爪: 已接管")

        changed = self.last_gripper_value is None or abs(value - self.last_gripper_value) >= self.gripper_deadband
        periodic = now - self.last_gripper_send >= 0.2
        if changed and periodic:
            ret = self.gripper.set_value(value)
            if ret == 0:
                self.last_gripper_value = value
                self.last_gripper_send = now

    def _handle_motion(self):
        control_value = self._get_key(self.control_input)
        if self.motion_blocked_until_control_release:
            if control_value <= self.grip_threshold:
                self.motion_blocked_until_control_release = False
            else:
                return

        if control_value <= self.grip_threshold:
            if self.active:
                self._deactivate()
            return

        if not self.active:
            self._activate()

        xyz, rot = self._get_right_controller_pose()
        if xyz is None:
            return

        if self.prev_controller_xyz is None:
            self.prev_controller_xyz = xyz
            self.prev_controller_rot = rot
            return

        delta_xyz = (xyz - self.prev_controller_xyz) * self.pos_scale
        delta_xyz = self._apply_yaw_correction(delta_xyz)
        delta_rotvec = (rot * self.prev_controller_rot.inv()).as_rotvec() * self.rot_scale
        self.prev_controller_xyz = xyz
        self.prev_controller_rot = rot

        self.target_pos = self.target_pos + delta_xyz
        if not self.lock_orientation:
            self.target_ori = self.target_ori + delta_rotvec

        # 安全限位，与 Vive 模式保持一致。
        self.target_pos[0] = np.clip(self.target_pos[0], -0.5, 0.1)
        self.target_pos[1] = np.clip(self.target_pos[1], -0.3, 0.3)
        self.target_pos[2] = np.clip(self.target_pos[2], 0.1, 0.6)

        target_6d = [
            float(self.target_pos[0]), float(self.target_pos[1]), float(self.target_pos[2]),
            float(self.target_ori[0]), float(self.target_ori[1]), float(self.target_ori[2])
        ]

        with self.arm_lock:
            ret = self.arm.rm_movep_canfd(target_6d, False, 0, 60)

        if self.debug and ret != 0:
            print(f"\n[!] Quest movep_canfd ret={ret}, target={target_6d}")

    def _debug_print(self, now):
        if not self.debug or now - self.last_debug_print < 1.0:
            return
        self.last_debug_print = now
        try:
            pose = np.asarray(self.xrt.get_right_controller_pose(), dtype=float).reshape(-1)
            pose_head = np.round(pose[:3], 3).tolist() if pose.size >= 3 else []
            print(
                f"\n[Quest debug] rt={self._get_key('right_trigger'):.3f} "
                f"rg={self._get_key('right_grip'):.3f} "
                f"lt={self._get_key('left_trigger'):.3f} "
                f"lg={self._get_key('left_grip'):.3f} "
                f"A={self._get_key('a_button'):.3f} "
                f"B={self._get_key('b_button'):.3f} "
                f"X={self._get_key('x_button'):.3f} "
                f"Y={self._get_key('y_button'):.3f} "
                f"control={self.control_input}:{self._get_key(self.control_input):.3f} "
                f"pose_xyz={pose_head} active={self.active} "
                f"gripper_mode={self.gripper_mode} "
                f"gripper_closed={self.gripper_toggle_closed}"
            )
        except Exception as e:
            print(f"\n[Quest debug] failed: {e}")

    def _control_loop(self):
        interval = 0.02
        while self.running:
            time.sleep(interval)
            try:
                now = time.time()
                self._handle_buttons(now)
                self._handle_gripper(now)
                if self.enable_motion:
                    self._handle_motion()
                self._debug_print(now)
            except Exception as e:
                print(f"\n[!] Quest 控制异常: {e}")
                time.sleep(0.2)

    def shutdown(self):
        self.running = False
        if hasattr(self, "thread"):
            self.thread.join(timeout=1.0)
        if self.xrt is not None:
            try:
                self.xrt.close()
            except Exception:
                pass


# ============ 键盘控制模块 ============
class KeyboardController:
    """键盘平移控制器：锁定末端姿态，只控制 XYZ 平移。"""

    MOVE_KEYS = {
        'w': np.array([1.0, 0.0, 0.0]),   # X+
        's': np.array([-1.0, 0.0, 0.0]),  # X-
        'a': np.array([0.0, 1.0, 0.0]),   # Y+
        'd': np.array([0.0, -1.0, 0.0]),  # Y-
        'q': np.array([0.0, 0.0, 1.0]),   # Z+
        'e': np.array([0.0, 0.0, -1.0]),  # Z-
    }

    def __init__(self, arm, arm_lock, step=0.01):
        self.arm = arm
        self.arm_lock = arm_lock
        self.step = step
        self.target_pos = ROBOT_INIT_POS.copy()
        self.target_ori = ROBOT_INIT_ORI.copy()
        self.reset_to_current(print_result=False)
        print(f"键盘控制: 姿态已锁定, step={self.step:.3f} m")

    def _get_current_robot_pose(self):
        try:
            with self.arm_lock:
                code, state = self.arm.rm_get_current_arm_state()
            if code != 0 or not isinstance(state, dict):
                return None

            for key in ("pose", "tool_pose", "waypoint"):
                pose = state.get(key)
                if pose is None:
                    continue
                pose = np.asarray(pose, dtype=float).reshape(-1)
                if pose.size >= 6 and np.all(np.isfinite(pose[:6])):
                    return pose[:3].copy(), pose[3:6].copy()
        except Exception:
            pass
        return None

    def reset_to_current(self, print_result=True):
        current_pose = self._get_current_robot_pose()
        if current_pose is not None:
            self.target_pos, self.target_ori = current_pose
            if print_result:
                print(f"\n键盘控制: 已重置到当前末端位姿 pos={np.round(self.target_pos, 3).tolist()}")
            return True
        if print_result:
            print("\n[!] 键盘控制: 读取当前末端位姿失败，继续使用默认位姿")
        return False

    def adjust_step(self, scale):
        self.step = float(np.clip(self.step * scale, 0.001, 0.1))
        print(f"\n键盘控制: step={self.step:.3f} m")

    def move(self, key):
        key = key.lower()
        if key not in self.MOVE_KEYS:
            return

        self.target_pos = self.target_pos + self.MOVE_KEYS[key] * self.step
        self.target_pos[0] = np.clip(self.target_pos[0], -0.5, 0.1)
        self.target_pos[1] = np.clip(self.target_pos[1], -0.3, 0.3)
        self.target_pos[2] = np.clip(self.target_pos[2], 0.1, 0.6)

        target_6d = [
            float(self.target_pos[0]), float(self.target_pos[1]), float(self.target_pos[2]),
            float(self.target_ori[0]), float(self.target_ori[1]), float(self.target_ori[2])
        ]

        with self.arm_lock:
            ret = self.arm.rm_movep_canfd(target_6d, False, 0, 60)

        print(f"\r键盘移动 {key}: pos={np.round(self.target_pos, 3).tolist()} ({'OK' if ret == 0 else 'FAIL'})", end='', flush=True)

    def shutdown(self):
        pass


# ============ 机械臂状态缓存 ============
class ArmStateCache:
    """后台读取机械臂状态，避免录制线程被 SDK 调用阻塞。"""

    def __init__(self, arm, arm_lock, poll_hz=50):
        self.arm = arm
        self.arm_lock = arm_lock
        self.poll_interval = 1.0 / max(1, float(poll_hz))
        self.lock = threading.Lock()
        self.running = True
        self.latest_joint = [0] * 6
        self.latest_state = None
        self.latest_code = -1
        self.latest_time = 0.0
        self.last_read_dt = 0.0
        self.max_read_dt = 0.0
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            start = time.time()
            try:
                with self.arm_lock:
                    code, state = self.arm.rm_get_current_arm_state()
                read_dt = time.time() - start
                joint = state.get('joint', [0] * 6) if code == 0 and isinstance(state, dict) else [0] * 6
                with self.lock:
                    self.latest_code = code
                    self.latest_state = state
                    self.latest_joint = list(joint)
                    self.latest_time = time.time()
                    self.last_read_dt = read_dt
                    self.max_read_dt = max(self.max_read_dt, read_dt)
            except Exception:
                pass

            elapsed = time.time() - start
            time.sleep(max(0.0, self.poll_interval - elapsed))

    def get_joint(self):
        with self.lock:
            age = time.time() - self.latest_time if self.latest_time else None
            return list(self.latest_joint), self.latest_code, age, self.last_read_dt

    def shutdown(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)


# ============ 数据录制模块 ============
class DataRecorder:
    """以固定频率录制机械臂状态、相机图像到 HDF5

    数据格式:
      observations/qpos:             (N, 7) float32  [6关节角 + 夹爪位置]
      observations/images/cam_high:  (N, H, W, 3) uint8
      observations/images/cam_wrist: (N, H, W, 3) uint8
      action:                        (N, 7) float32  [下一帧的qpos，即行为克隆标签]
      timestamps:                    (N,) float64    [相对时间戳(秒)]
    """

    def __init__(
        self,
        arm,
        arm_lock,
        gripper,
        cam_top,
        cam_wrist,
        arm_state_cache=None,
        target_fps=30,
        camera_freeze_timeout=3.0,
        camera_reject_timeout=8.0,
        reject_stale_wrist=True,
    ):
        self.arm = arm
        self.arm_lock = arm_lock
        self.gripper = gripper
        self.cam_top = cam_top
        self.cam_wrist = cam_wrist
        self.arm_state_cache = arm_state_cache
        self.is_recording = False
        self.filename = None
        self.target_fps = target_fps
        self.camera_freeze_timeout = camera_freeze_timeout
        self.camera_reject_timeout = camera_reject_timeout
        self.reject_stale_wrist = reject_stale_wrist
        self.recording_had_wrist_warning = False
        self.recording_has_stale_wrist = False
        self.stale_wrist_reason = None
        self.max_wrist_stale_age = 0.0
        self.last_wrist_warning_time = 0.0
        self.data_buffer = {
            'qpos': [],
            'images_top': [],
            'images_wrist': [],
            'timestamps': [],
            'record_arm_dt': [],
            'record_state_age': [],
            'record_camera_dt': [],
        }
        self.save_threads = []

    def start(self, filename):
        if self.is_recording:
            return
        self.filename = filename
        self.is_recording = True
        self.recording_had_wrist_warning = False
        self.recording_has_stale_wrist = False
        self.stale_wrist_reason = None
        self.max_wrist_stale_age = 0.0
        self.last_wrist_warning_time = 0.0
        self.data_buffer = {
            'qpos': [],
            'images_top': [],
            'images_wrist': [],
            'timestamps': [],
            'record_arm_dt': [],
            'record_state_age': [],
            'record_camera_dt': [],
        }
        self.record_start_time = time.time()
        self.thread = threading.Thread(target=self._record_loop, daemon=True)
        self.thread.start()
        print(f"录制开始: {os.path.basename(filename)} (目标 {self.target_fps}Hz)")

    def stop(self, async_save=False):
        if not self.is_recording:
            return False
        self.is_recording = False
        self.thread.join()
        filename = self.filename
        data_buffer = {
            'qpos': list(self.data_buffer['qpos']),
            'images_top': list(self.data_buffer['images_top']),
            'images_wrist': list(self.data_buffer['images_wrist']),
            'timestamps': list(self.data_buffer['timestamps']),
            'record_arm_dt': list(self.data_buffer['record_arm_dt']),
            'record_state_age': list(self.data_buffer['record_state_age']),
            'record_camera_dt': list(self.data_buffer['record_camera_dt']),
            'recording_had_wrist_warning': self.recording_had_wrist_warning,
            'recording_has_stale_wrist': self.recording_has_stale_wrist,
            'stale_wrist_reason': self.stale_wrist_reason,
            'max_wrist_stale_age': self.max_wrist_stale_age,
        }
        if self.reject_stale_wrist and self.recording_has_stale_wrist:
            print(f"[!] 拒绝保存: 腕部相机卡流/无新帧 ({self.stale_wrist_reason})")
            print("    请检查 D435 USB 连接/分辨率/帧率后重采本条数据。")
            return False
        if self.recording_had_wrist_warning:
            print(f"[!] 腕部相机曾短时疑似卡流，最长 {self.max_wrist_stale_age:.2f}s；未达到拒绝阈值，继续保存。")
        if async_save:
            thread = threading.Thread(target=self._save, args=(filename, data_buffer), daemon=False)
            thread.start()
            self.save_threads.append(thread)
        else:
            self._save(filename, data_buffer)
        return True

    def wait_for_saves(self):
        for thread in self.save_threads:
            if thread.is_alive():
                thread.join()
        self.save_threads = []

    def _record_loop(self):
        interval = 1.0 / self.target_fps
        next_time = time.time()

        while self.is_recording:
            now = time.time()
            if now < next_time:
                time.sleep(0.001)
                continue
            next_time += interval

            try:
                timestamp = now - self.record_start_time

                arm_start = time.time()
                if self.arm_state_cache is not None:
                    joint_angles, code, state_age, _ = self.arm_state_cache.get_joint()
                else:
                    with self.arm_lock:
                        code, state = self.arm.rm_get_current_arm_state()
                    joint_angles = state['joint'] if code == 0 else [0] * 6
                    state_age = 0.0
                arm_dt = time.time() - arm_start
                gripper_val = self.gripper.get_value()

                camera_start = time.time()
                img_top = self.cam_top.get_frame()
                img_wrist = self.cam_wrist.get_frame()
                camera_dt = time.time() - camera_start
                self._check_wrist_health(now)

                self.data_buffer['qpos'].append(joint_angles + [gripper_val])
                self.data_buffer['images_top'].append(img_top)
                self.data_buffer['images_wrist'].append(img_wrist)
                self.data_buffer['timestamps'].append(timestamp)
                self.data_buffer['record_arm_dt'].append(arm_dt)
                self.data_buffer['record_state_age'].append(-1.0 if state_age is None else state_age)
                self.data_buffer['record_camera_dt'].append(camera_dt)
            except Exception as e:
                print(f" >> Record error: {e}")

    def _check_wrist_health(self, now):
        health_fn = getattr(self.cam_wrist, "get_health", None)
        if health_fn is None:
            return
        health = health_fn(self.camera_freeze_timeout)
        if health.get("ok"):
            return

        unique_age = health.get("unique_age")
        receive_age = health.get("receive_age")
        frame_number = health.get("frame_number")
        if unique_age is None and now - self.record_start_time < self.camera_freeze_timeout:
            return

        if unique_age is None:
            reason = "尚未收到腕部相机有效帧"
        else:
            receive_age_text = "None" if receive_age is None else f"{receive_age:.2f}s"
            reason = (
                f"{unique_age:.2f}s 未收到新的 frame_number"
                f" (last={frame_number}, receive_age={receive_age_text})"
            )
            self.max_wrist_stale_age = max(self.max_wrist_stale_age, unique_age)

        self.recording_had_wrist_warning = True
        self.stale_wrist_reason = reason
        if unique_age is None:
            reject = now - self.record_start_time >= self.camera_reject_timeout
        else:
            reject = unique_age >= self.camera_reject_timeout
        if reject:
            self.recording_has_stale_wrist = True

        if now - self.last_wrist_warning_time >= 1.0:
            self.last_wrist_warning_time = now
            level = "严重卡流" if reject else "短时疑似卡流"
            print(f"\n[!] 腕部相机{level}: {reason}")

    def _save(self, filename, data_buffer):
        if not data_buffer['qpos']:
            print(" >> 无数据")
            return

        qpos = data_buffer['qpos']
        # 行为克隆标签: action[t] = qpos[t+1]
        actions = qpos[1:] + [qpos[-1]]
        timestamps = data_buffer['timestamps']

        if len(timestamps) > 1:
            intervals = np.diff(timestamps)
            actual_fps = 1.0 / np.mean(intervals)
            print(f"  实际帧率: {actual_fps:.1f} Hz (目标: {self.target_fps} Hz)")
            print(
                "  帧间隔: "
                f"max={np.max(intervals):.3f}s, "
                f">0.1s={int(np.sum(intervals > 0.1))}, "
                f">0.5s={int(np.sum(intervals > 0.5))}"
            )
        if data_buffer.get('record_arm_dt'):
            arm_dt = np.array(data_buffer['record_arm_dt'])
            state_age = np.array(data_buffer.get('record_state_age', []))
            valid_state_age = state_age[state_age >= 0] if state_age.size else np.array([])
            camera_dt = np.array(data_buffer['record_camera_dt'])
            state_age_text = ""
            if valid_state_age.size:
                state_age_text = f", state_age max={np.max(valid_state_age):.3f}s"
            print(
                "  录制耗时: "
                f"arm max={np.max(arm_dt):.3f}s, "
                f"camera max={np.max(camera_dt):.3f}s"
                f"{state_age_text}"
            )

        try:
            tmp_filename = f"{filename}.tmp"
            with h5py.File(tmp_filename, 'w') as f:
                f.attrs['sim'] = False
                f.attrs['fps'] = self.target_fps
                f.create_dataset('observations/qpos', data=np.array(qpos))
                f.create_dataset('action', data=np.array(actions))
                f.create_dataset('timestamps', data=np.array(timestamps))
                f.create_dataset('observations/images/cam_high',
                                 data=np.array(data_buffer['images_top']),
                                 compression="gzip")
                f.create_dataset('observations/images/cam_wrist',
                                 data=np.array(data_buffer['images_wrist']),
                                 compression="gzip")
            os.replace(tmp_filename, filename)
            print(f"保存: {os.path.basename(filename)} ({len(qpos)} frames)")
        except Exception as e:
            print(f"[!] 保存失败: {e}")


# ============ 主程序 ============
def main():
    parser = argparse.ArgumentParser(description='RealMan RM65 数据采集')
    parser.add_argument('--arm-ip', type=str, default=DEFAULT_ARM_IP, help='机械臂IP地址')
    parser.add_argument('--arm-port', type=int, default=DEFAULT_ARM_PORT, help='机械臂端口')
    parser.add_argument('--cam-top', type=str, default=DEFAULT_CAM_TOP_SERIAL, help='顶部相机序列号')
    parser.add_argument('--cam-wrist', type=str, default=DEFAULT_CAM_WRIST_SERIAL, help='腕部相机序列号')
    parser.add_argument('--cam-top-width', type=int, default=640, help='顶部相机图像宽度（默认640）')
    parser.add_argument('--cam-top-height', type=int, default=480, help='顶部相机图像高度（默认480）')
    parser.add_argument('--cam-wrist-width', type=int, default=1280, help='腕部相机图像宽度（默认1280，D435 16:9完整视野）')
    parser.add_argument('--cam-wrist-height', type=int, default=720, help='腕部相机图像高度（默认720，D435 16:9完整视野）')
    parser.add_argument('--cam-top-fps', type=int, default=None,
                        help='顶部相机出流帧率；默认跟随 --fps')
    parser.add_argument('--cam-wrist-fps', type=int, default=30,
                        help='腕部 RealSense 出流帧率（默认30；D435 1280x720 不支持20fps）')
    parser.add_argument('--cam-top-exposure', type=float, default=None,
                        help='顶部相机曝光值；UVC/OpenCV相机不一定支持')
    parser.add_argument('--cam-wrist-exposure', type=float, default=500,
                        help='腕部 RealSense 相机曝光值（默认500）')
    parser.add_argument('--no-reset-wrist-camera', action='store_true',
                        help='启动采集时不对腕部 RealSense 执行 hardware_reset')
    parser.add_argument('--wrist-reset-wait', type=float, default=6.0,
                        help='腕部 RealSense 复位后等待重新枚举的最长秒数（默认6.0）')
    parser.add_argument('--wrist-preflight-wait', type=float, default=3.0,
                        help='每条开始录制前等待腕部 RealSense 有效非黑帧的秒数（默认3.0）')
    parser.add_argument('--no-restart-wrist-on-stale', action='store_true',
                        help='每条开始录制前发现腕部 RealSense 异常时，不自动重启 pipeline')
    parser.add_argument('--reset-wrist-on-stale', action='store_true',
                        help='每条开始录制前重启 pipeline 仍失败时，再执行 D435 hardware_reset')
    parser.add_argument('--camera-freeze-timeout', type=float, default=3.0,
                        help='腕部相机超过该秒数没有新的 frame_number 时打印警告（默认3.0秒）')
    parser.add_argument('--camera-reject-timeout', type=float, default=8.0,
                        help='腕部相机连续超过该秒数没有新的 frame_number 时拒绝保存（默认8.0秒）')
    parser.add_argument('--allow-stale-wrist', action='store_true',
                        help='即使检测到腕部相机卡流也允许保存；仅建议排障或单相机临时测试使用')
    parser.add_argument('--allow-camera-fail', action='store_true',
                        help='即使相机初始化失败也继续运行；会保存黑帧，仅建议排障使用')
    parser.add_argument('--save-dir', type=str, default='data/raw_hdf5', help='数据保存目录')
    parser.add_argument('--task-name', type=str, default='task_pick_cube', help='任务名称')
    parser.add_argument('--fps', type=int, default=30, help='采集帧率')
    parser.add_argument('--teaching', action='store_true', help='示教模式（不用Vive）')
    parser.add_argument('--keyboard', action='store_true', help='键盘平移控制模式（锁定末端姿态）')
    parser.add_argument('--keyboard-step', type=float, default=0.002, help='键盘每次平移步长，单位米（默认0.002）')
    parser.add_argument('--vive-yaw-deg', type=float, default=0.0,
                        help='Vive 平面位移绕机械臂 Z 轴的 yaw 校正角，单位度；Z轴不受影响')
    parser.add_argument('--quest', action='store_true', help='Quest 手柄遥操作模式（不用Vive）')
    parser.add_argument('--vive-quest', action='store_true',
                        help='混合模式：Vive Tracker 控制机械臂末端，Quest 手柄控制夹爪和录制')
    parser.add_argument('--quest-pos-scale', type=float, default=0.2,
                        help='Quest 手柄位移缩放系数；插拔/精细操作默认0.2，粗移动可显式调大')
    parser.add_argument('--quest-rot-scale', type=float, default=0.3, help='Quest 手柄姿态缩放系数')
    parser.add_argument('--quest-yaw-deg', type=float, default=45.0,
                        help='Quest 位移增量绕机械臂 Z 轴的 yaw 校正角，单位度；插拔任务调试值默认45，Z轴不受影响')
    parser.add_argument('--quest-control-input', type=str, default='right_grip',
                        choices=[
                            'right_grip', 'right_trigger', 'a_button', 'b_button',
                            'left_grip', 'left_trigger', 'x_button', 'y_button',
                            'right_axis_click', 'left_axis_click',
                        ],
                        help='Quest 控制机械臂离合的输入')
    parser.add_argument('--quest-grip-threshold', type=float, default=0.8, help='右 Grip 超过该值才控制机械臂')
    parser.add_argument('--quest-button-threshold', type=float, default=0.8, help='左手录制按钮触发阈值')
    parser.add_argument('--quest-lock-orientation', action='store_true', help='Quest 模式只控位置，锁定末端姿态')
    parser.add_argument('--quest-no-gripper', action='store_true', help='Quest 模式下暂不控制夹爪')
    parser.add_argument('--quest-invert-trigger', action='store_true', help='反转右扳机夹爪方向')
    parser.add_argument('--quest-gripper-mode', type=str, default='toggle',
                        choices=['toggle', 'continuous'],
                        help='Quest 夹爪控制模式：toggle=扳机上升沿开闭切换，continuous=扳机模拟量连续控制')
    parser.add_argument('--quest-gripper-init', type=str, default='keep',
                        choices=['keep', 'open', 'trigger'],
                        help='Quest 夹爪启动初始化方式：keep保持当前，open打开，trigger跟随当前右扳机')
    parser.add_argument('--quest-home-current-on-start', action='store_true',
                        help='纯 Quest 模式启动时读取当前6维关节角作为本次采集 home；每条停止录制后回到该姿态')
    parser.add_argument('--quest-home-joints', type=parse_joint_list, default=None,
                        help='纯 Quest 模式固定 home 关节角，6个逗号分隔的角度值；不设置时使用当前 USB 插拔固定 home')
    parser.add_argument('--quest-home-speed', type=int, default=40,
                        help='纯 Quest 模式启动/停止后回 home 的 movej 速度（默认40）')
    parser.add_argument('--quest-no-stop-home', action='store_true',
                        help='纯 Quest 模式下左 Trigger 停止录制后不执行打开夹爪/上抬/回关节位')
    parser.add_argument('--quest-debug', action='store_true', help='打印 Quest 手柄输入调试信息')
    parser.add_argument('--rm-api-path', type=str,
                        default=os.environ.get('RM_API2_PYTHON_PATH') or os.environ.get('RM_API_PATH') or DEFAULT_RM_API_PATH,
                        help='RM_API2/Python 路径')
    args = parser.parse_args()

    if sum([args.teaching, args.quest, args.keyboard, args.vive_quest]) > 1:
        parser.error('--teaching、--quest、--keyboard、--vive-quest 不能同时使用')

    if not args.rm_api_path:
        parser.error('未配置 RM_API2：请设置 RM_API2_PYTHON_PATH 或传入 --rm-api-path')
    add_rm_api_path(args.rm_api_path)

    # 导入机械臂SDK
    from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e

    print("=" * 50)
    if args.teaching:
        mode_name = "示教"
    elif args.quest:
        mode_name = "Quest遥操作"
    elif args.keyboard:
        mode_name = "键盘平移"
    elif args.vive_quest:
        mode_name = "Vive+Quest混合"
    else:
        mode_name = "Vive遥操作"
    print(f"  RealMan RM65 {mode_name} 数据采集")
    print("=" * 50)

    if not args.no_reset_wrist_camera and not _is_opencv_device(args.cam_wrist):
        reset_realsense_device(args.cam_wrist, wait_seconds=args.wrist_reset_wait)

    # 初始化双相机。cam 参数为 /dev/video* 或数字时使用 OpenCV，否则使用 RealSense 序列号。
    cam_top = create_camera(
        args.cam_top,
        width=args.cam_top_width,
        height=args.cam_top_height,
        fps=args.cam_top_fps or args.fps,
        exposure=args.cam_top_exposure,
    )
    cam_wrist = create_camera(
        args.cam_wrist,
        width=args.cam_wrist_width,
        height=args.cam_wrist_height,
        fps=args.cam_wrist_fps,
        exposure=args.cam_wrist_exposure,
    )
    print(f"相机: top={'OK' if cam_top.is_active else 'FAIL'}, "
          f"wrist={'OK' if cam_wrist.is_active else 'FAIL'}")
    if (not cam_top.is_active or not cam_wrist.is_active) and not args.allow_camera_fail:
        print("[!] 相机初始化失败，已停止采集，避免保存黑帧数据。")
        print("    如果只是排障，可加 --allow-camera-fail 强制继续。")
        sys.exit(1)

    # 初始化机械臂
    arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    handle = arm.rm_create_robot_arm(args.arm_ip, args.arm_port)
    if handle.id == -1:
        print("[!] 机械臂连接失败")
        sys.exit(1)
    print(f"机械臂: OK ({args.arm_ip}:{args.arm_port})")

    arm.rm_set_arm_run_mode(1)
    arm_lock = threading.Lock()
    arm_state_cache = ArmStateCache(arm, arm_lock, poll_hz=max(30, args.fps * 2))
    gripper = RealManGripper(arm, arm_lock)
    gripper.initialize()

    # 初始化录制器
    recorder = DataRecorder(
        arm,
        arm_lock,
        gripper,
        cam_top,
        cam_wrist,
        arm_state_cache,
        args.fps,
        camera_freeze_timeout=args.camera_freeze_timeout,
        camera_reject_timeout=args.camera_reject_timeout,
        reject_stale_wrist=not args.allow_stale_wrist,
    )

    def ensure_wrist_ready():
        if not isinstance(cam_wrist, RealSenseCamera):
            return True

        if cam_wrist.wait_for_valid_frame(timeout=args.wrist_preflight_wait):
            return True

        print("[!] 腕部 D435 预检失败：没有拿到新鲜非黑帧")
        if not args.no_restart_wrist_on_stale:
            if cam_wrist.restart_pipeline(wait_seconds=args.wrist_preflight_wait):
                print("腕部 D435 pipeline 重启后恢复")
                return True

        if args.reset_wrist_on_stale:
            print("[!] 腕部 D435 pipeline 重启无效，执行 hardware_reset")
            cam_wrist.is_active = False
            try:
                if cam_wrist.pipeline is not None:
                    cam_wrist.pipeline.stop()
            except Exception:
                pass
            reset_realsense_device(args.cam_wrist, wait_seconds=args.wrist_reset_wait)
            if cam_wrist.restart_pipeline(wait_seconds=args.wrist_preflight_wait):
                print("腕部 D435 hardware_reset 后恢复")
                return True

        print("[!] 腕部 D435 仍未恢复，取消本次开始录制，避免保存坏数据")
        return False

    def get_current_joint():
        try:
            with arm_lock:
                code, state = arm.rm_get_current_arm_state()
            if code != 0 or not isinstance(state, dict):
                return None
            joint = np.asarray(state.get("joint", []), dtype=float).reshape(-1)
            if joint.size >= 6 and np.all(np.isfinite(joint[:6])):
                return joint[:6].copy()
        except Exception:
            pass
        return None

    def wait_until_joint(target, timeout=15.0, tolerance_deg=1.0, label="movej"):
        target = np.asarray(target, dtype=float)[:6]
        deadline = time.time() + timeout
        last_err = None
        while time.time() < deadline:
            current = get_current_joint()
            if current is not None:
                last_err = float(np.max(np.abs(current - target)))
                if last_err <= tolerance_deg:
                    return True
            time.sleep(0.03)
        print(f"[!] {label}: 等待关节到位超时，max_err={last_err}")
        return False

    quest_home_joints = np.asarray(
        args.quest_home_joints if args.quest_home_joints is not None else QUEST_STOP_HOME_JOINTS,
        dtype=float,
    )
    if args.quest and args.quest_home_current_on_start:
        current = get_current_joint()
        if current is None:
            print("[!] Quest home 捕获失败：无法读取当前关节角")
            gripper.shutdown()
            arm_state_cache.shutdown()
            cam_top.close()
            cam_wrist.close()
            arm.rm_delete_robot_arm()
            sys.exit(1)
        quest_home_joints = current.astype(float)
        print(f"Quest home: 使用启动时当前关节角 {np.round(quest_home_joints, 3).tolist()}")

    def prepare_quest_episode_start():
        if not args.quest:
            return True

        if quest_ctrl is not None:
            quest_ctrl.pause_motion(print_result=True)
            quest_ctrl.motion_blocked_until_control_release = True
        return True

    def initialize_quest_start_pose():
        if not args.quest:
            return True

        home_joints = quest_home_joints.tolist()
        if args.quest_home_current_on_start:
            print("Quest 启动初始化: 当前姿态已作为 home，不执行启动回 home 运动")
            if not wait_until_joint(home_joints, timeout=2.0, tolerance_deg=0.8, label="Quest init current home"):
                return False
        else:
            print(f"Quest 启动初始化: 回关节位 {np.round(quest_home_joints, 3).tolist()}")
            with arm_lock:
                ret = arm.rm_movej(home_joints, int(args.quest_home_speed), 0, 0, 1)
            if ret != 0:
                print(f"[!] Quest 启动初始化: 回关节位失败 ret={ret}")
                return False
            if not wait_until_joint(home_joints, timeout=20.0, tolerance_deg=0.8, label="Quest init home"):
                return False

        pose = get_current_tcp_pose()
        if pose is None:
            print("[!] Quest 启动初始化: 读取末端姿态失败")
            return False
        print(f"Quest 启动初始化: 锁定末端姿态 rpy={np.round(pose[3:6], 4).tolist()}")
        return True

    def start_episode():
        if not prepare_quest_episode_start():
            return
        if not ensure_wrist_ready():
            return
        filename = get_next_filename(args.save_dir, args.task_name)
        recorder.start(filename)

    def stop_episode(async_save=False):
        was_recording = recorder.is_recording
        recorder.stop(async_save=async_save)
        return was_recording

    def get_current_tcp_pose():
        try:
            with arm_lock:
                code, state = arm.rm_get_current_arm_state()
            if code != 0 or not isinstance(state, dict):
                return None

            for key in ("pose", "tool_pose", "waypoint"):
                pose = state.get(key)
                if pose is None:
                    continue
                pose = np.asarray(pose, dtype=float).reshape(-1)
                if pose.size >= 6 and np.all(np.isfinite(pose[:6])):
                    return pose[:6].copy()
        except Exception:
            pass
        return None

    def quest_stop_home_sequence():
        if not args.quest:
            return

        print("\nQuest 停止后收尾: 暂停遥控")
        if quest_ctrl is not None:
            quest_ctrl.pause_motion()

        ret = gripper.open(block=True, timeout=5)
        print(f"Quest 停止后收尾: 打开夹爪 ({'OK' if ret == 0 else 'FAIL'})")
        time.sleep(0.2)

        current_pose = get_current_tcp_pose()
        if current_pose is not None:
            lift_pose = current_pose.copy()
            lift_pose[2] = min(0.6, lift_pose[2] + QUEST_STOP_LIFT_DISTANCE)
            with arm_lock:
                ret = arm.rm_movel(lift_pose.tolist(), 30, 0, 0, 1)
            print(f"Quest 停止后收尾: 末端上移 {QUEST_STOP_LIFT_DISTANCE:.2f}m ({'OK' if ret == 0 else 'FAIL'})")
        else:
            print("[!] Quest 停止后收尾: 读取当前末端位姿失败，跳过上移")

        home_joints = quest_home_joints.tolist()
        with arm_lock:
            ret = arm.rm_movej(home_joints, int(args.quest_home_speed), 0, 0, 1)
        print(f"Quest 停止后收尾: 回关节位 {np.round(quest_home_joints, 3).tolist()} ({'OK' if ret == 0 else 'FAIL'})")

    # 初始化遥控
    vive_ctrl = None
    quest_ctrl = None
    keyboard_ctrl = None
    if args.quest and not initialize_quest_start_pose():
        print("[!] Quest 启动初始化失败，已停止采集")
        gripper.shutdown()
        arm_state_cache.shutdown()
        cam_top.close()
        cam_wrist.close()
        arm.rm_delete_robot_arm()
        sys.exit(1)

    if args.quest or args.vive_quest:
        if args.vive_quest:
            vive_ctrl = ViveController(
                arm,
                arm_lock,
                enable_vive=True,
                lock_orientation=True,
                yaw_correction_deg=args.vive_yaw_deg,
            )
        quest_ctrl = QuestController(
            arm,
            arm_lock,
            gripper,
            start_recording=start_episode,
            stop_recording=(lambda: stop_episode(async_save=args.quest and not args.quest_no_stop_home)),
            pos_scale=args.quest_pos_scale,
            rot_scale=args.quest_rot_scale,
            yaw_correction_deg=args.quest_yaw_deg,
            control_input=args.quest_control_input,
            grip_threshold=args.quest_grip_threshold,
            button_threshold=args.quest_button_threshold,
            lock_orientation=args.quest_lock_orientation or args.quest,
            enable_gripper=not args.quest_no_gripper,
            invert_trigger=args.quest_invert_trigger,
            gripper_mode=args.quest_gripper_mode,
            gripper_init=args.quest_gripper_init,
            enable_motion=not args.vive_quest,
            calibrate_callback=vive_ctrl.calibrate if args.vive_quest and vive_ctrl is not None else None,
            enable_callback=vive_ctrl.enable if args.vive_quest and vive_ctrl is not None else None,
            disable_callback=vive_ctrl.disable if args.vive_quest and vive_ctrl is not None else None,
            post_stop_callback=quest_stop_home_sequence if args.quest and not args.quest_no_stop_home else None,
            debug=args.quest_debug,
        )
    elif args.keyboard:
        keyboard_ctrl = KeyboardController(arm, arm_lock, step=args.keyboard_step)
    else:
        vive_ctrl = ViveController(
            arm,
            arm_lock,
            enable_vive=not args.teaching,
            yaw_correction_deg=args.vive_yaw_deg,
        )

    if vive_ctrl is not None:
        print(f"Vive: {'OK' if vive_ctrl.tracker else 'FAIL'}")

    print("\n" + "-" * 50)
    if args.teaching:
        print("命令: s=录制  d=保存  g <0-100>=夹爪  c=闭合  o=打开  q=退出")
    elif args.quest:
        print(f"Quest: {args.quest_control_input}=末端离合  右Trigger=夹爪  左Grip=录制  左Trigger=保存")
        if args.quest_no_gripper:
            print("      当前已禁用 Quest 夹爪控制")
        print("键盘: s=录制  d=保存  g <0-100>=夹爪  c=闭合  o=打开  q=退出")
    elif args.vive_quest:
        print("Vive: 右Grip=校准  B=启用末端平移遥控  A=暂停末端遥控  姿态=锁定")
        print("Quest: 右Trigger=夹爪  左Grip=录制  左Trigger=保存")
        print("键盘备用: v=校准  w=启用  e=暂停")
        if args.quest_no_gripper:
            print("      当前已禁用 Quest 夹爪控制")
        print("键盘: s=录制  d=保存  g <0-100>=夹爪  c=闭合  o=打开  q=退出")
    elif args.keyboard:
        print("键盘平移: W/S=X+/X-  A/D=Y+/Y-  Q/E=Z+/Z-  R=重置当前位姿")
        print(f"          当前步长={args.keyboard_step:.3f}m  +/-=调整步长  Z=录制  X=保存  C=闭合  O=打开  Esc=退出")
    else:
        print("命令: v=校准  w=遥控  e=停止  s=录制  d=保存")
        print("      g <0-100>=夹爪  c=闭合  o=打开  q=退出")
    print("-" * 50)

    old_settings = termios.tcgetattr(sys.stdin)
    cmd_buffer = ""

    try:
        tty.setcbreak(sys.stdin.fileno())
        print("> ", end='', flush=True)

        should_exit = False
        while not should_exit:
            rlist, _, _ = select.select([sys.stdin], [], [], 0.01)
            if rlist:
                char = sys.stdin.read(1)
                if args.keyboard and char != '\n':
                    key = char.lower()
                    if char == '\x1b':
                        print()
                        should_exit = True
                    elif key in KeyboardController.MOVE_KEYS:
                        keyboard_ctrl.move(key)
                    elif char == '+':
                        keyboard_ctrl.adjust_step(1.5)
                    elif char == '-':
                        keyboard_ctrl.adjust_step(1 / 1.5)
                    elif key == 'r':
                        keyboard_ctrl.reset_to_current()
                    elif key == 'z':
                        print()
                        start_episode()
                    elif key == 'x':
                        print()
                        stop_episode()
                    elif key == 'c':
                        print()
                        ret = gripper.close()
                        print(f"夹爪: 闭合 ({'OK' if ret == 0 else 'FAIL'})")
                    elif key == 'o':
                        print()
                        ret = gripper.open()
                        print(f"夹爪: 打开 ({'OK' if ret == 0 else 'FAIL'})")
                    continue

                if char == '\n':
                    cmd = cmd_buffer.strip()
                    if cmd == 'q':
                        print()
                        break
                    elif cmd == 'v' and vive_ctrl is not None:
                        print(); vive_ctrl.calibrate()
                    elif cmd == 'w' and vive_ctrl is not None:
                        print(); vive_ctrl.enable()
                    elif cmd == 'e' and vive_ctrl is not None:
                        print(); vive_ctrl.disable()
                    elif cmd == 's':
                        print()
                        start_episode()
                    elif cmd == 'd':
                        print(); stop_episode()
                    elif cmd.startswith('g '):
                        print()
                        try:
                            val = max(0, min(100, int(cmd.split()[1])))
                            ret = gripper.set_value(val / 100.0)
                            print(f"夹爪: {val}% ({'OK' if ret == 0 else 'FAIL'})")
                        except Exception:
                            print("格式: g <0-100>")
                    elif cmd == 'c':
                        print()
                        ret = gripper.close()
                        print(f"夹爪: 闭合 ({'OK' if ret == 0 else 'FAIL'})")
                    elif cmd == 'o':
                        print()
                        ret = gripper.open()
                        print(f"夹爪: 打开 ({'OK' if ret == 0 else 'FAIL'})")
                    elif cmd:
                        print("\n未知命令")
                    cmd_buffer = ""
                    print("> ", end='', flush=True)
                elif char == '\x7f':
                    if cmd_buffer:
                        cmd_buffer = cmd_buffer[:-1]
                        sys.stdout.write('\b \b')
                        sys.stdout.flush()
                else:
                    cmd_buffer += char
                    sys.stdout.write(char)
                    sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n")
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        if recorder.is_recording:
            recorder.stop()
        recorder.wait_for_saves()
        if vive_ctrl is not None:
            vive_ctrl.shutdown()
        if quest_ctrl is not None:
            quest_ctrl.shutdown()
        if keyboard_ctrl is not None:
            keyboard_ctrl.shutdown()
        gripper.shutdown()
        arm_state_cache.shutdown()
        cam_top.close()
        cam_wrist.close()
        arm.rm_delete_robot_arm()
        print("已退出")


if __name__ == '__main__':
    main()
