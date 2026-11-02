#!/usr/bin/env python3
"""Interactive Quest 3 -> RealMan RM65 hardware smoke test.

This tool intentionally does not open cameras or create HDF5 files. It is an
explicit, operator-supervised motion test for the Quest/PC-Service/Python/
RealMan/gripper control chain.
"""

from __future__ import annotations

import argparse
import os
import select
import sys
import termios
import time
import tty
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


HOME_JOINTS = np.asarray([0, 0, 90, 0, 90, 30], dtype=float)
HOME_TOLERANCE_DEG = 0.8
HOME_SPEED = 20
DEFAULT_POSITION_SCALE = 0.2
GRIPPER_OPEN_VALUE = 1.0
GRIPPER_CLOSED_VALUE = 0.0
GRIPPER_MIN_POSITION = 1
GRIPPER_MAX_POSITION = 1000
QUEST_R_HEADSET_TO_WORLD = np.asarray([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
WORKSPACE_MIN = np.asarray([-0.5, -0.3, 0.1], dtype=float)
WORKSPACE_MAX = np.asarray([0.1, 0.3, 0.6], dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Operator-supervised Quest 3 -> RealMan RM65 motion smoke test (no cameras or HDF5)."
    )
    parser.add_argument("--arm-ip", required=True, help="RealMan controller IP address")
    parser.add_argument("--arm-port", type=int, default=8080, help="RealMan controller port")
    parser.add_argument(
        "--rm-api-path",
        default=os.environ.get("RM_API2_PYTHON_PATH") or os.environ.get("RM_API_PATH"),
        help="RM_API2/Python directory; defaults to RM_API2_PYTHON_PATH",
    )
    parser.add_argument("--home-speed", type=int, default=HOME_SPEED, help="Blocking movej speed for return-to-home")
    parser.add_argument("--home-timeout", type=float, default=20.0, help="Seconds allowed for home-joint convergence")
    parser.add_argument(
        "--pos-scale",
        type=float,
        default=DEFAULT_POSITION_SCALE,
        help="Quest controller position scale (default: 0.2; first-test range: 0.1-0.3)",
    )
    parser.add_argument("--grip-threshold", type=float, default=0.8, help="Right Grip threshold for end-effector control")
    parser.add_argument("--yaw-deg", type=float, default=180.0, help="Quest-to-robot XY yaw correction")
    parser.add_argument("--status-hz", type=float, default=2.0, help="Terminal status refresh rate")
    return parser.parse_args()


def add_rm_api_path(path: str | None) -> None:
    if not path:
        raise RuntimeError("RM_API2 is not configured. Set RM_API2_PYTHON_PATH or pass --rm-api-path.")
    sdk_path = Path(path).expanduser()
    if not sdk_path.is_dir():
        raise RuntimeError(f"RM_API2 Python directory does not exist: {sdk_path}")
    if str(sdk_path) not in sys.path:
        sys.path.insert(0, str(sdk_path))


def valid_quest_pose(pose: Any) -> np.ndarray | None:
    values = np.asarray(pose, dtype=float).reshape(-1)
    if values.size < 7 or not np.all(np.isfinite(values[:7])) or np.allclose(values[:7], 0.0):
        return None
    return values[:7]


def clamp_workspace(position: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(position, dtype=float), WORKSPACE_MIN, WORKSPACE_MAX)


def yaw_rotate(delta_xyz: np.ndarray, yaw_deg: float) -> np.ndarray:
    radians = np.deg2rad(yaw_deg)
    c, s = np.cos(radians), np.sin(radians)
    result = np.asarray(delta_xyz, dtype=float).copy()
    result[:2] = [c * result[0] - s * result[1], s * result[0] + c * result[1]]
    return result


def gripper_value_to_position(value: float) -> int:
    value = float(np.clip(value, 0.0, 1.0))
    return int(round(GRIPPER_MIN_POSITION + value * (GRIPPER_MAX_POSITION - GRIPPER_MIN_POSITION)))


def gripper_position_to_value(position: float) -> float:
    position = float(np.clip(position, GRIPPER_MIN_POSITION, GRIPPER_MAX_POSITION))
    return (position - GRIPPER_MIN_POSITION) / (GRIPPER_MAX_POSITION - GRIPPER_MIN_POSITION)


@dataclass
class SmokeResult:
    quest_input_changed: bool = False
    motion_command_ok: bool = False
    motion_observed: bool = False
    gripper_command_ok: bool = False
    gripper_observed: bool = False

    @property
    def passed(self) -> bool:
        return all(
            [
                self.quest_input_changed,
                self.motion_command_ok,
                self.motion_observed,
                self.gripper_command_ok,
                self.gripper_observed,
            ]
        )

    def summary(self) -> str:
        return (
            "Quest输入=" + ("OK" if self.quest_input_changed else "WAIT") + " | "
            "末端命令=" + ("OK" if self.motion_command_ok else "WAIT") + " | "
            "末端实际位移=" + ("OK" if self.motion_observed else "WAIT") + " | "
            "夹爪命令=" + ("OK" if self.gripper_command_ok else "WAIT") + " | "
            "夹爪状态=" + ("OK" if self.gripper_observed else "WAIT")
        )


class QuestRealManSmokeTest:
    def __init__(self, args: argparse.Namespace, arm: Any, xrt: Any):
        self.args = args
        self.arm = arm
        self.xrt = xrt
        self.result = SmokeResult()
        self.target_position: np.ndarray | None = None
        self.locked_orientation: np.ndarray | None = None
        self.previous_controller_xyz: np.ndarray | None = None
        self.previous_input_pose: np.ndarray | None = None
        self.trigger_was_pressed = False
        self.gripper_value: float | None = None
        self.last_status = 0.0

    def current_joint(self) -> np.ndarray | None:
        code, state = self.arm.rm_get_current_arm_state()
        if code != 0 or not isinstance(state, dict):
            return None
        joint = np.asarray(state.get("joint", []), dtype=float).reshape(-1)
        return joint[:6].copy() if joint.size >= 6 and np.all(np.isfinite(joint[:6])) else None

    def current_tcp_pose(self) -> np.ndarray | None:
        code, state = self.arm.rm_get_current_arm_state()
        if code != 0 or not isinstance(state, dict):
            return None
        for key in ("pose", "tool_pose", "waypoint"):
            pose = np.asarray(state.get(key, []), dtype=float).reshape(-1)
            if pose.size >= 6 and np.all(np.isfinite(pose[:6])):
                return pose[:6].copy()
        return None

    def current_gripper_value(self) -> float | None:
        ret, state = self.arm.rm_get_gripper_state()
        if ret != 0 or not isinstance(state, dict):
            return None
        for key in ("actpos", "position", "current_position"):
            if key in state:
                return gripper_position_to_value(float(state[key]))
        return None

    def wait_for_home(self) -> bool:
        deadline = time.monotonic() + self.args.home_timeout
        max_error: float | None = None
        while time.monotonic() < deadline:
            joint = self.current_joint()
            if joint is not None:
                max_error = float(np.max(np.abs(joint - HOME_JOINTS)))
                if max_error <= HOME_TOLERANCE_DEG:
                    print(f"[OK] 采集零位到位，最大误差 {max_error:.3f}°")
                    return True
            time.sleep(0.05)
        print(f"[FAIL] 采集零位到位超时，最大误差={max_error}")
        return False

    def return_home(self) -> bool:
        print("\n[home] 正在低速回到采集零位……")
        ret = self.arm.rm_movej(HOME_JOINTS.tolist(), int(self.args.home_speed), 0, 0, 1)
        if ret != 0:
            print(f"[FAIL] 回零命令失败: ret={ret}")
            return False
        return self.wait_for_home()

    def ensure_safe_start(self) -> None:
        print("\n安全检查：确认急停可用、机械臂周围无障碍，且采集零位在当前工位安全。")
        if input("输入 HOME 并回车以低速回到采集零位（其他输入取消）：").strip() != "HOME":
            raise RuntimeError("操作者取消回零；未发送机械臂运动命令。")
        if not self.return_home():
            raise RuntimeError("回零失败；已停止测试。请人工检查机械臂。")
        if input("输入 READY 并回车以启用 Quest 遥操作（其他输入取消）：").strip() != "READY":
            raise RuntimeError("操作者取消遥操作；将回到采集零位。")

    def initialize_control_reference(self) -> None:
        pose = self.current_tcp_pose()
        if pose is None:
            raise RuntimeError("无法读取机械臂末端位姿；拒绝启用遥操作。")
        self.target_position = pose[:3].copy()
        self.locked_orientation = pose[3:6].copy()
        self.gripper_value = self.current_gripper_value()
        print(f"[OK] 末端姿态已锁定: {np.round(self.locked_orientation, 4).tolist()}")
        print("操作：右 Grip 按住后小幅移动手柄控制末端；右 Trigger 按一下切换夹爪；q 退出并回零。")

    def update_quest_input(self) -> np.ndarray | None:
        pose = valid_quest_pose(self.xrt.get_right_controller_pose())
        if pose is None:
            return None
        if self.previous_input_pose is not None and np.linalg.norm(pose[:3] - self.previous_input_pose[:3]) >= 0.005:
            self.result.quest_input_changed = True
        self.previous_input_pose = pose.copy()
        return pose

    def handle_gripper(self) -> None:
        pressed = float(self.xrt.get_right_trigger()) > 0.5
        rising = pressed and not self.trigger_was_pressed
        self.trigger_was_pressed = pressed
        if not rising:
            return

        current = self.current_gripper_value()
        if current is None:
            raise RuntimeError("无法读取夹爪状态；停止发送夹爪命令。")
        target = GRIPPER_OPEN_VALUE if current < 0.5 else GRIPPER_CLOSED_VALUE
        ret = self.arm.rm_set_gripper_position(gripper_value_to_position(target), True, 5)
        self.result.gripper_command_ok = self.result.gripper_command_ok or ret == 0
        after = self.current_gripper_value()
        if after is not None and abs(after - current) >= 0.1:
            self.result.gripper_observed = True
        print(f"[gripper] {'打开' if target == GRIPPER_OPEN_VALUE else '闭合'} ret={ret}, state={after}")

    def handle_motion(self, pose: np.ndarray | None) -> None:
        if pose is None or float(self.xrt.get_right_grip()) <= self.args.grip_threshold:
            self.previous_controller_xyz = None
            return
        if self.target_position is None or self.locked_orientation is None:
            raise RuntimeError("遥操作基准未初始化。")

        controller_xyz = QUEST_R_HEADSET_TO_WORLD @ pose[:3]
        if self.previous_controller_xyz is None:
            self.previous_controller_xyz = controller_xyz
            return
        delta = yaw_rotate((controller_xyz - self.previous_controller_xyz) * self.args.pos_scale, self.args.yaw_deg)
        self.previous_controller_xyz = controller_xyz
        if np.linalg.norm(delta) < 1e-6:
            return

        before = self.current_tcp_pose()
        if before is None:
            raise RuntimeError("无法读取机械臂末端位姿；停止发送末端命令。")
        self.target_position = clamp_workspace(self.target_position + delta)
        target = [*self.target_position.tolist(), *self.locked_orientation.tolist()]
        ret = self.arm.rm_movep_canfd(target, False, 0, 60)
        self.result.motion_command_ok = self.result.motion_command_ok or ret == 0
        time.sleep(0.05)
        after = self.current_tcp_pose()
        if after is not None and np.linalg.norm(after[:3] - before[:3]) >= 0.002:
            self.result.motion_observed = True
        print(f"[motion] ret={ret}, target_xyz={np.round(self.target_position, 4).tolist()}")

    def print_status(self) -> None:
        interval = 1.0 / max(self.args.status_hz, 0.1)
        if time.monotonic() - self.last_status < interval:
            return
        self.last_status = time.monotonic()
        pose = valid_quest_pose(self.xrt.get_right_controller_pose())
        xyz = np.round(pose[:3], 3).tolist() if pose is not None else [0.0, 0.0, 0.0]
        grip = float(self.xrt.get_right_grip())
        trigger = float(self.xrt.get_right_trigger())
        print(
            f"\r[test] rg={grip:.2f} rt={trigger:.2f} pose={xyz} | {self.result.summary()}",
            end="",
            flush=True,
        )

    def run_interactive(self) -> bool:
        self.initialize_control_reference()
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            while True:
                pose = self.update_quest_input()
                self.handle_gripper()
                self.handle_motion(pose)
                self.print_status()
                if self.result.passed:
                    print("\n[PASS] Quest→PC Service→RM65→夹爪完整控制链路已验证。")
                    return True
                readable, _, _ = select.select([sys.stdin], [], [], 0.02)
                if readable and sys.stdin.read(1).lower() == "q":
                    print("\n[INFO] 操作者结束测试。")
                    return self.result.passed
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


def main() -> int:
    args = parse_args()
    arm = None
    xrt_started = False
    return_home_on_exit = False
    tester: QuestRealManSmokeTest | None = None
    try:
        add_rm_api_path(args.rm_api_path)
        from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e
        import xrobotoolkit_sdk as xrt

        xrt.init()
        xrt_started = True
        arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        handle = arm.rm_create_robot_arm(args.arm_ip, args.arm_port)
        if getattr(handle, "id", -1) == -1:
            raise RuntimeError(f"无法连接 RealMan: {args.arm_ip}:{args.arm_port}")
        print(f"[OK] RealMan 已连接: handle={getattr(handle, 'id', 'unknown')}")
        print("[OK] Quest SDK 已连接到 PC Service")

        tester = QuestRealManSmokeTest(args, arm, xrt)
        tester.ensure_safe_start()
        return_home_on_exit = True
        passed = tester.run_interactive()
        home_ok = tester.return_home()
        return_home_on_exit = False
        return 0 if passed and home_ok else 1
    except (KeyboardInterrupt, EOFError):
        return_home_on_exit = False
        print("\n[INFO] 操作者中断测试。")
        return 1
    except Exception as exc:
        return_home_on_exit = False
        print(f"\n[FAIL] 测试启动或运行异常: {exc}", file=sys.stderr)
        print("[SAFETY] 已停止发送运动命令。请人工确认机械臂状态，必要时使用急停。", file=sys.stderr)
        return 1
    finally:
        if tester is not None and arm is not None and return_home_on_exit:
            tester.return_home()
        if xrt_started:
            try:
                xrt.close()
            except Exception:
                pass
        if arm is not None:
            try:
                arm.rm_delete_robot_arm()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
