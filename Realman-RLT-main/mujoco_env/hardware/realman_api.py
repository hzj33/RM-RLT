"""RealMan RM65 hardware backend.

This module intentionally keeps the RealMan SDK details behind a small API
that matches the needs of the RM65 VLA/RLT port:

- joint actions are 7D: six joint angles in degrees plus a normalized gripper;
- the main control loop sends joint targets without synchronous state reads;
- robot state and gripper commands run in background paths so slow SDK calls do
  not stall policy execution;
- Cartesian residuals are composed through SDK FK/IK with the reference joint
  command as the IK seed.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np

DEFAULT_RM_API_PATH = os.environ.get("RM_API2_PYTHON_PATH") or os.environ.get("RM_API_PATH") or ""

CommandMode = Literal["follow", "canfd"]
StateReadMode = Literal["arm_state", "joint_degree"]


@dataclass(frozen=True)
class RealManConfig:
    robot_ip: str = "192.168.2.19"
    robot_port: int = 8080
    rm_api_path: str = DEFAULT_RM_API_PATH

    state_cache_hz: float = 20.0
    state_read_mode: StateReadMode = "arm_state"
    start_state_cache: bool = True
    start_gripper_worker: bool = True

    command_mode: CommandMode = "follow"
    canfd_follow: bool = False
    canfd_trajectory_mode: int = 2
    canfd_radio: int = 60
    max_joint_step_deg: float = 2.0
    joint_deadband_deg: float = 0.05
    joint_ema_alpha: float = 0.0

    gripper_min_position: int = 1
    gripper_max_position: int = 1000
    gripper_deadband: float = 0.02
    binary_gripper: bool = True
    gripper_open_threshold: float = 0.6
    gripper_close_threshold: float = 0.4
    gripper_open_value: float = 1.0
    gripper_closed_value: float = 0.0
    initial_gripper_value: float = 1.0


@dataclass(frozen=True)
class RealManState:
    stamp: float
    joint_deg: np.ndarray
    tcp_pos: np.ndarray
    tcp_euler: np.ndarray
    tcp_quat_wxyz: np.ndarray
    gripper: float

    @property
    def tcp_quat_xyzw(self) -> np.ndarray:
        q = self.tcp_quat_wxyz
        return np.asarray([q[1], q[2], q[3], q[0]], dtype=float)

    def as_qpos(self) -> np.ndarray:
        return np.r_[self.joint_deg, self.gripper].astype(np.float32)

    def as_robot_state_dict(self) -> dict[str, Any]:
        return {
            "stamp": self.stamp,
            "joint_deg": self.joint_deg.copy(),
            "tcp_pos": self.tcp_pos.copy(),
            "tcp_euler": self.tcp_euler.copy(),
            "tcp_quat_wxyz": self.tcp_quat_wxyz.copy(),
            "tcp_quat": self.tcp_quat_xyzw,
            "gripper": float(self.gripper),
        }


def add_rm_api_path(path: str | None = None) -> None:
    sdk_path = path or os.environ.get("RM_API2_PYTHON_PATH") or os.environ.get("RM_API_PATH") or DEFAULT_RM_API_PATH
    if sdk_path and sdk_path not in sys.path:
        sys.path.insert(0, sdk_path)


def _load_rm_sdk(path: str | None = None) -> tuple[Any, Any, Any]:
    add_rm_api_path(path)
    from Robotic_Arm.rm_robot_interface import (  # type: ignore
        RoboticArm,
        rm_inverse_kinematics_params_t,
        rm_thread_mode_e,
    )

    return RoboticArm, rm_thread_mode_e, rm_inverse_kinematics_params_t


def _as_vec(values: Any, size: int, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or inf: {arr}.")
    return arr


class RealManController:
    """RM65 controller with cached state and asynchronous gripper commands."""

    def __init__(self, config: RealManConfig | None = None, *, auto_connect: bool = False) -> None:
        self.config = config or RealManConfig()
        self.arm = None
        self.handle = None
        self.robot_info: dict[str, Any] = {}
        self._ik_params_type = None

        self._state_lock = threading.Lock()
        self._latest_state: Optional[RealManState] = None
        self._last_state_error: Optional[BaseException] = None
        self._state_running = False
        self._state_thread: Optional[threading.Thread] = None

        self._joint_command_lock = threading.Lock()
        self._last_joint_cmd: Optional[np.ndarray] = None
        self._smoothed_joint_cmd: Optional[np.ndarray] = None

        self._gripper_lock = threading.Lock()
        self._pending_gripper_value: Optional[float] = None
        self._last_gripper_cmd: Optional[float] = None
        self._cached_gripper_value = float(np.clip(self.config.initial_gripper_value, 0.0, 1.0))
        self._gripper_running = False
        self._gripper_thread: Optional[threading.Thread] = None

        if auto_connect:
            self.connect()

    @property
    def is_connected(self) -> bool:
        return self.arm is not None

    def connect(self) -> None:
        if self.is_connected:
            return
        RoboticArm, thread_mode, ik_params = _load_rm_sdk(self.config.rm_api_path)
        self._ik_params_type = ik_params
        self.arm = RoboticArm(thread_mode.RM_TRIPLE_MODE_E)
        self.handle = self.arm.rm_create_robot_arm(self.config.robot_ip, self.config.robot_port)
        if getattr(self.handle, "id", 0) == -1:
            self.close()
            raise RuntimeError(f"Failed to connect RealMan arm at {self.config.robot_ip}:{self.config.robot_port}")
        ret, info = self.arm.rm_get_robot_info()
        if ret != 0:
            self.close()
            raise RuntimeError(f"rm_get_robot_info failed: ret={ret}")
        self.robot_info = dict(info)
        if int(self.robot_info.get("arm_dof", 0)) != 6:
            self.close()
            raise RuntimeError(f"Expected 6-DOF RealMan arm, got {self.robot_info}.")

        try:
            self.refresh_gripper_from_state()
        except Exception:
            pass

        if self.config.start_state_cache:
            self.start_state_cache()
        if self.config.start_gripper_worker:
            self.start_gripper_worker()

    def close(self) -> None:
        self.stop_gripper_worker()
        self.stop_state_cache()
        if self.arm is not None:
            try:
                self.arm.rm_delete_robot_arm()
            finally:
                self.arm = None
                self.handle = None

    def __enter__(self) -> "RealManController":
        self.connect()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def start_state_cache(self) -> None:
        if self._state_thread is not None and self._state_thread.is_alive():
            return
        self._state_running = True
        self._state_thread = threading.Thread(target=self._state_loop, name="realman-state-cache", daemon=True)
        self._state_thread.start()

    def stop_state_cache(self) -> None:
        self._state_running = False
        thread = self._state_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        self._state_thread = None

    def start_gripper_worker(self) -> None:
        if self._gripper_thread is not None and self._gripper_thread.is_alive():
            return
        self._gripper_running = True
        self._gripper_thread = threading.Thread(target=self._gripper_loop, name="realman-gripper", daemon=True)
        self._gripper_thread.start()

    def stop_gripper_worker(self) -> None:
        self._gripper_running = False
        thread = self._gripper_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._gripper_thread = None

    def _state_loop(self) -> None:
        hz = max(float(self.config.state_cache_hz), 1e-6)
        period = 1.0 / hz
        while self._state_running:
            t0 = time.monotonic()
            try:
                state = self.read_state_once()
            except BaseException as exc:
                with self._state_lock:
                    self._last_state_error = exc
            else:
                with self._state_lock:
                    self._latest_state = state
                    self._last_state_error = None
            elapsed = time.monotonic() - t0
            time.sleep(max(0.0, period - elapsed))

    def read_state_once(self) -> RealManState:
        self._require_connected()
        if self.config.state_read_mode == "arm_state":
            ret, payload = self.arm.rm_get_current_arm_state()
            if ret != 0:
                raise RuntimeError(f"rm_get_current_arm_state failed: ret={ret}")
            joint_deg = _as_vec(payload["joint"][:6], 6, "joint")
            pose_eul = _as_vec(payload["pose"][:6], 6, "pose")
        elif self.config.state_read_mode == "joint_degree":
            ret, joint = self.arm.rm_get_joint_degree()
            if ret != 0:
                raise RuntimeError(f"rm_get_joint_degree failed: ret={ret}")
            joint_deg = _as_vec(joint[:6], 6, "joint")
            pose_eul = self.fk_euler(joint_deg)
        else:
            raise ValueError(f"Unsupported state_read_mode: {self.config.state_read_mode}")

        quat_wxyz = _as_vec(self.arm.rm_algo_euler2quaternion(pose_eul[3:].tolist()), 4, "quat")
        return RealManState(
            stamp=time.time(),
            joint_deg=joint_deg,
            tcp_pos=pose_eul[:3].copy(),
            tcp_euler=pose_eul[3:].copy(),
            tcp_quat_wxyz=quat_wxyz,
            gripper=self.get_cached_gripper(),
        )

    def get_state(self, *, max_age_s: float | None = None, read_if_stale: bool = False) -> RealManState:
        with self._state_lock:
            state = self._latest_state
            err = self._last_state_error
        if state is None:
            if read_if_stale:
                return self.read_state_once()
            if err is not None:
                raise RuntimeError(f"No cached RealMan state; last read error: {err!r}")
            raise RuntimeError("No cached RealMan state is available yet.")
        if max_age_s is not None and time.time() - state.stamp > max_age_s:
            if read_if_stale:
                return self.read_state_once()
            raise RuntimeError(f"Cached RealMan state is stale: age={time.time() - state.stamp:.3f}s")
        return state

    def get_qpos(self, *, read_if_stale: bool = False) -> np.ndarray:
        return self.get_state(read_if_stale=read_if_stale).as_qpos()

    def send_joint_target(
        self,
        joint_deg: Any,
        *,
        mode: CommandMode | None = None,
        limit_step: bool = True,
    ) -> dict[str, Any]:
        self._require_connected()
        target = _as_vec(joint_deg, 6, "joint_deg").astype(float)
        raw_target = target.copy()
        with self._joint_command_lock:
            target = self._ema_joint_target_unlocked(target)
            if limit_step:
                target = self._limited_joint_target(target)
                if float(self.config.joint_ema_alpha) > 0.0:
                    self._smoothed_joint_cmd = target.copy()
            sent = True
            if self._last_joint_cmd is not None:
                max_delta = float(np.max(np.abs(target - self._last_joint_cmd)))
                if max_delta < max(float(self.config.joint_deadband_deg), 0.0):
                    sent = False
            ret = 0
            if sent:
                ret = self._send_joint_target_unlocked(target, mode or self.config.command_mode)
                if ret == 0:
                    self._last_joint_cmd = target.copy()
        return {
            "ret": ret,
            "sent": sent,
            "raw_joint_target": raw_target,
            "joint_target": target.copy(),
        }

    def _ema_joint_target_unlocked(self, target: np.ndarray) -> np.ndarray:
        alpha = float(self.config.joint_ema_alpha)
        if alpha <= 0.0:
            return target
        alpha = float(np.clip(alpha, 0.0, 1.0))
        if self._smoothed_joint_cmd is None:
            self._smoothed_joint_cmd = target.copy()
        else:
            self._smoothed_joint_cmd = alpha * target + (1.0 - alpha) * self._smoothed_joint_cmd
        return self._smoothed_joint_cmd.copy()

    def set_qpos(
        self,
        qpos: Any,
        *,
        mode: CommandMode | None = None,
        limit_step: bool = True,
        queue_gripper: bool = True,
    ) -> dict[str, Any]:
        arr = np.asarray(qpos, dtype=float)
        if arr.shape not in ((6,), (7,)):
            raise ValueError(f"qpos must have shape (6,) or (7,), got {arr.shape}.")
        result = self.send_joint_target(arr[:6], mode=mode, limit_step=limit_step)
        if arr.shape == (7,) and queue_gripper:
            result["gripper_queued"] = self.queue_gripper(float(arr[6]))
        return result

    def _limited_joint_target(self, target: np.ndarray) -> np.ndarray:
        max_step = float(self.config.max_joint_step_deg)
        if max_step <= 0.0:
            return target
        reference = self._last_joint_cmd
        if reference is None:
            try:
                reference = self.get_state(max_age_s=0.5).joint_deg
            except Exception:
                return target
        delta = np.clip(target - reference, -max_step, max_step)
        return reference + delta

    def _send_joint_target_unlocked(self, target: np.ndarray, mode: CommandMode) -> int:
        values = target.astype(float).tolist()
        if mode == "follow":
            return int(self.arm.rm_movej_follow(values))
        if mode == "canfd":
            return int(
                self.arm.rm_movej_canfd(
                    values,
                    bool(self.config.canfd_follow),
                    trajectory_mode=int(self.config.canfd_trajectory_mode),
                    radio=int(self.config.canfd_radio),
                )
            )
        raise ValueError(f"Unsupported RealMan command mode: {mode}")

    def movej_blocking(
        self,
        joint_deg: Any,
        *,
        speed: int = 40,
        trajectory_connect: int = 0,
        block: bool = True,
    ) -> int:
        self._require_connected()
        q = _as_vec(joint_deg, 6, "joint_deg")
        ret = int(self.arm.rm_movej(q.astype(float).tolist(), int(speed), 0, int(trajectory_connect), int(block)))
        if ret == 0:
            with self._joint_command_lock:
                self._last_joint_cmd = q.copy()
                self._smoothed_joint_cmd = q.copy()
        return ret

    def set_gripper_sync(self, value: float, *, block: bool = True, timeout_s: float = 5.0) -> int:
        self._require_connected()
        processed = self._process_gripper_value(float(np.clip(value, 0.0, 1.0)))
        position = self.gripper_value_to_position(processed)
        ret = int(self.arm.rm_set_gripper_position(position, bool(block), int(timeout_s)))
        if ret == 0:
            with self._gripper_lock:
                self._cached_gripper_value = processed
                self._last_gripper_cmd = processed
                self._pending_gripper_value = None
        return ret

    def stop_arm(self) -> None:
        if self.arm is None:
            return
        try:
            self.arm.rm_set_arm_stop()
        except Exception:
            pass

    def fk_euler(self, joint_deg: Any) -> np.ndarray:
        self._require_connected()
        q = _as_vec(joint_deg, 6, "joint_deg")
        return _as_vec(self.arm.rm_algo_forward_kinematics(q.tolist(), 1), 6, "fk_euler")

    def fk_quat_wxyz(self, joint_deg: Any) -> np.ndarray:
        self._require_connected()
        q = _as_vec(joint_deg, 6, "joint_deg")
        return _as_vec(self.arm.rm_algo_forward_kinematics(q.tolist(), 0), 7, "fk_quat_wxyz")

    def ik_euler(self, pose_euler: Any, seed_joint_deg: Any) -> tuple[int, np.ndarray]:
        self._require_connected()
        pose = _as_vec(pose_euler, 6, "pose_euler")
        seed = _as_vec(seed_joint_deg, 6, "seed_joint_deg")
        params = self._ik_params_type(q_in=seed.tolist() + [0.0], q_pose=pose.tolist(), flag=1)
        ret, q_out = self.arm.rm_algo_inverse_kinematics(params)
        return int(ret), _as_vec(q_out[:6], 6, "ik_out")

    def ik_quat_wxyz(self, pose_quat_wxyz: Any, seed_joint_deg: Any) -> tuple[int, np.ndarray]:
        self._require_connected()
        pose = _as_vec(pose_quat_wxyz, 7, "pose_quat_wxyz")
        seed = _as_vec(seed_joint_deg, 6, "seed_joint_deg")
        params = self._ik_params_type(q_in=seed.tolist() + [0.0], q_pose=pose.tolist(), flag=0)
        ret, q_out = self.arm.rm_algo_inverse_kinematics(params)
        return int(ret), _as_vec(q_out[:6], 6, "ik_out")

    def compose_xyz_residual(
        self,
        reference_joint_deg: Any,
        delta_xyz_m: Any,
        *,
        seed_joint_deg: Any | None = None,
    ) -> np.ndarray:
        ref_q = _as_vec(reference_joint_deg, 6, "reference_joint_deg")
        delta = _as_vec(delta_xyz_m, 3, "delta_xyz_m")
        target_pose = self.fk_euler(ref_q)
        target_pose[:3] += delta
        seed = ref_q if seed_joint_deg is None else _as_vec(seed_joint_deg, 6, "seed_joint_deg")
        ret, q_cmd = self.ik_euler(target_pose, seed)
        if ret != 0:
            raise RuntimeError(f"RealMan IK failed for xyz residual: ret={ret}")
        return q_cmd

    def refresh_gripper_from_state(self) -> float:
        self._require_connected()
        ret, state = self.arm.rm_get_gripper_state()
        if ret != 0:
            raise RuntimeError(f"rm_get_gripper_state failed: ret={ret}")
        actpos = state.get("actpos", None) if isinstance(state, dict) else None
        if actpos is not None:
            self._cached_gripper_value = self.gripper_position_to_value(float(actpos))
        return self._cached_gripper_value

    def get_cached_gripper(self) -> float:
        with self._gripper_lock:
            return float(self._cached_gripper_value)

    def queue_gripper(self, value: float, *, force: bool = False) -> bool:
        processed = self._process_gripper_value(float(np.clip(value, 0.0, 1.0)))
        with self._gripper_lock:
            if (
                not force
                and self._last_gripper_cmd is not None
                and abs(processed - self._last_gripper_cmd) < max(float(self.config.gripper_deadband), 0.0)
            ):
                return False
            self._last_gripper_cmd = processed
            self._cached_gripper_value = processed
            self._pending_gripper_value = processed
        return True

    def _process_gripper_value(self, value: float) -> float:
        if not self.config.binary_gripper:
            return value
        open_value = float(self.config.gripper_open_value)
        closed_value = float(self.config.gripper_closed_value)
        mid = (open_value + closed_value) * 0.5
        previous = self._last_gripper_cmd
        if previous is None:
            return open_value if value >= mid else closed_value
        previous_is_open = abs(previous - open_value) <= abs(previous - closed_value)
        if previous_is_open:
            return closed_value if value <= float(self.config.gripper_close_threshold) else open_value
        return open_value if value >= float(self.config.gripper_open_threshold) else closed_value

    def _gripper_loop(self) -> None:
        while self._gripper_running:
            value = None
            with self._gripper_lock:
                if self._pending_gripper_value is not None:
                    value = self._pending_gripper_value
                    self._pending_gripper_value = None
            if value is None:
                time.sleep(0.005)
                continue
            try:
                ret = self._send_gripper_value(value)
                if ret != 0:
                    with self._gripper_lock:
                        self._pending_gripper_value = value
            except Exception:
                with self._gripper_lock:
                    self._pending_gripper_value = value
                time.sleep(0.05)

    def _send_gripper_value(self, value: float) -> int:
        self._require_connected()
        position = self.gripper_value_to_position(value)
        ret = int(self.arm.rm_set_gripper_position(position, False, 0))
        if ret == 0:
            with self._gripper_lock:
                self._cached_gripper_value = float(np.clip(value, 0.0, 1.0))
        return ret

    def gripper_value_to_position(self, value: float) -> int:
        clipped = float(np.clip(value, 0.0, 1.0))
        lo = int(self.config.gripper_min_position)
        hi = int(self.config.gripper_max_position)
        return int(round(lo + clipped * (hi - lo)))

    def gripper_position_to_value(self, position: float) -> float:
        lo = float(self.config.gripper_min_position)
        hi = float(self.config.gripper_max_position)
        clipped = float(np.clip(position, lo, hi))
        return (clipped - lo) / max(hi - lo, 1e-9)

    def _require_connected(self) -> None:
        if self.arm is None:
            raise RuntimeError("RealManController is not connected.")
