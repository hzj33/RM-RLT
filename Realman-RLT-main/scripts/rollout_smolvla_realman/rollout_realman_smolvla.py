#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mujoco_env.hardware.realman_api import RealManConfig, RealManController  # noqa: E402
from scripts.rollout_smolvla.rollout_ur3e_smolvla import (  # noqa: E402
    ActionPacket,
    AsyncSmolVLAWorker,
    ObservationPacket,
    SmolVLAInferencer,
    SyncBuffer,
    decode_image_msg,
    fmt_vec,
    normalize_policy_path,
    now_sec,
    resize_to_height,
)
from scripts.rollout_smolvla_realman.config import CFG  # noqa: E402


def resolve_realman_policy_path(path: Path | None, output_root: Path) -> Path:
    if path is not None:
        return normalize_policy_path(path.expanduser())

    output_root = output_root.expanduser().resolve()
    patterns = [
        "*realman*/checkpoints/*/pretrained_model",
        "realman*/checkpoints/*/pretrained_model",
        "*/checkpoints/*/pretrained_model",
    ]
    candidates: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for pretrained in output_root.glob(pattern):
            if (pretrained / "model.safetensors").exists() and pretrained not in seen:
                seen.add(pretrained)
                candidates.append(pretrained)
        if candidates and "realman" in pattern:
            break
    if not candidates:
        raise FileNotFoundError(
            "No RealMan policy checkpoint found. Pass --policy-path pointing to a pretrained_model "
            "directory, or train first with scripts/train_smolvla_realman/train_realman_smolvla.py."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime).resolve()


def feature_shape(feature: Any) -> tuple[int, ...] | None:
    shape = getattr(feature, "shape", None)
    if shape is None and isinstance(feature, dict):
        shape = feature.get("shape")
    if shape is None:
        return None
    try:
        return tuple(int(v) for v in shape)
    except Exception:
        return None


def vector_feature_dim(feature: Any) -> int | None:
    shape = feature_shape(feature)
    if not shape:
        return None
    if len(shape) == 1:
        return int(shape[0])
    return int(shape[-1])


def image_feature_hw(feature: Any) -> tuple[int, int] | None:
    shape = feature_shape(feature)
    if not shape or len(shape) != 3:
        return None
    # LeRobot configs are usually CHW, dataset metadata is often HWC.
    if shape[0] in (1, 3, 4):
        return int(shape[1]), int(shape[2])
    return int(shape[0]), int(shape[1])


def policy_feature_dict(policy_config: Any, attr: str) -> dict[str, Any]:
    value = getattr(policy_config, attr, None)
    return value if isinstance(value, dict) else {}


def parse_qpos(text: str) -> np.ndarray:
    values = [float(part.strip()) for part in str(text).split(",") if part.strip()]
    arr = np.asarray(values, dtype=float)
    if arr.shape != (7,):
        raise argparse.ArgumentTypeError(f"qpos must be 7 comma-separated numbers, got {arr.shape}: {text!r}")
    return arr


def image_msg_to_tensor(
    msg: Any,
    *,
    expected_hw: tuple[int, int] | None = None,
    channel_order: str = "rgb",
) -> torch.Tensor:
    image = decode_image_msg(msg)
    if expected_hw is not None and image.shape[:2] != expected_hw:
        try:
            import cv2

            height, width = expected_hw
            image = cv2.resize(image, (int(width), int(height)), interpolation=cv2.INTER_AREA)
        except Exception as exc:
            raise RuntimeError(f"Failed to resize ROS image from {image.shape[:2]} to {expected_hw}: {exc}") from exc
    channel_order = str(channel_order).lower()
    if channel_order == "bgr":
        # decode_image_msg returns RGB.  Some legacy RealMan datasets were
        # converted from OpenCV/RealSense BGR frames without BGR->RGB, so the
        # trained policy effectively saw BGR data interpreted as RGB.  This
        # diagnostic mode reproduces that training distribution at rollout time.
        image = image[:, :, [2, 1, 0]].copy()
    elif channel_order != "rgb":
        raise ValueError(f"Unsupported policy image channel order: {channel_order!r}")
    return torch.from_numpy(image).permute(2, 0, 1).contiguous().float() / 255.0


class RealManSmolVLARolloutNode:
    def __init__(self, node: Any, args: argparse.Namespace) -> None:
        from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import Image

        self.node = node
        self.args = args
        self.execute = bool(args.execute)
        self.last_log_time = 0.0
        self.last_sync_infer_start = 0.0
        self.sync_infer_lock = threading.Lock()
        self.latest_action_lock = threading.Lock()
        self.sync_action_queue: deque[ActionPacket] = deque()
        self.last_dt_map: dict[str, float | None] = {}
        self.topic_counts = {"front": 0, "wrist": 0}
        self.topic_last = {"front": 0.0, "wrist": 0.0}
        self.preview_failed = False
        self.preview_stop = threading.Event()
        self.preview_thread: threading.Thread | None = None

        self.buffers = {
            "front": SyncBuffer(args.buffer_maxlen),
            "wrist": SyncBuffer(args.buffer_maxlen),
        }

        rm_cfg = RealManConfig(
            robot_ip=args.robot_ip,
            robot_port=args.robot_port,
            rm_api_path=args.rm_api_path,
            state_cache_hz=args.state_cache_hz,
            command_mode=args.command_mode,
            canfd_follow=args.canfd_follow,
            canfd_trajectory_mode=args.canfd_trajectory_mode,
            canfd_radio=args.canfd_radio,
            max_joint_step_deg=args.max_joint_step_deg,
            joint_deadband_deg=args.joint_deadband_deg,
            joint_ema_alpha=args.joint_ema_alpha,
            gripper_deadband=args.gripper_deadband,
            binary_gripper=args.binary_gripper,
            gripper_open_threshold=args.gripper_open_threshold,
            gripper_close_threshold=args.gripper_close_threshold,
            gripper_open_value=args.gripper_open_value,
            gripper_closed_value=args.gripper_closed_value,
            initial_gripper_value=args.initial_gripper_value,
        )
        self.robot = RealManController(rm_cfg, auto_connect=True)
        self.node.get_logger().info(f"Connected RealMan arm: {self.robot.robot_info}")

        if self.execute and args.move_home_on_start:
            self._move_home_blocking(args.home_qpos)

        self.inference_mode = str(args.inference_mode)
        self.worker: AsyncSmolVLAWorker | None = None
        self.inferencer: SmolVLAInferencer | None = None
        if self.inference_mode == "async":
            self.worker = AsyncSmolVLAWorker(
                args.policy_path,
                args.device,
                replan_every_step=args.replan_every_step,
                execution_horizon=args.execution_horizon,
            )
            self.worker.start()
            policy_config = self.worker.inferencer.policy.config
            ckpt_action_horizon = self.worker.inferencer.ckpt_action_horizon
            execution_horizon = self.worker.inferencer.execution_horizon
        else:
            self.inferencer = SmolVLAInferencer(
                args.policy_path,
                args.device,
                replan_every_step=args.replan_every_step,
                execution_horizon=args.execution_horizon,
            )
            policy_config = self.inferencer.policy.config
            ckpt_action_horizon = self.inferencer.ckpt_action_horizon
            execution_horizon = self.inferencer.execution_horizon
        self.execution_horizon = int(execution_horizon)
        self.input_features = policy_feature_dict(policy_config, "input_features")
        self.output_features = policy_feature_dict(policy_config, "output_features")
        self.front_feature_key, self.wrist_feature_key = self._resolve_image_feature_keys()
        self.expected_hw = {
            self.front_feature_key: image_feature_hw(self.input_features.get(self.front_feature_key)),
            self.wrist_feature_key: image_feature_hw(self.input_features.get(self.wrist_feature_key)),
        }
        self._validate_policy_dims()

        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.sub_group = ReentrantCallbackGroup()
        self.sample_group = MutuallyExclusiveCallbackGroup()
        self.command_group = MutuallyExclusiveCallbackGroup()
        self.subs = [
            node.create_subscription(
                Image,
                args.front_image_topic,
                self._on_front,
                image_qos,
                callback_group=self.sub_group,
            ),
            node.create_subscription(
                Image,
                args.wrist_image_topic,
                self._on_wrist,
                image_qos,
                callback_group=self.sub_group,
            ),
        ]
        self.sample_timer = node.create_timer(
            1.0 / max(args.fps, 1e-6),
            self._sample_tick,
            callback_group=self.sample_group,
        )
        self.command_timer = node.create_timer(
            1.0 / max(args.command_hz, 1e-6),
            self._command_tick,
            callback_group=self.command_group,
        )
        if args.preview:
            self.preview_thread = threading.Thread(target=self._preview_loop, name="realman-rollout-preview", daemon=True)
            self.preview_thread.start()

        mode = "EXECUTE" if self.execute else "DRY-RUN"
        self.node.get_logger().info(
            f"RealMan SmolVLA rollout ready ({mode}). policy={args.policy_path}, "
            f"fps={args.fps:.1f}, command_hz={args.command_hz:.1f}, inference_mode={self.inference_mode}, "
            f"chunk_size={policy_config.chunk_size}, execution_horizon={execution_horizon}/{ckpt_action_horizon}, "
            f"front_key={self.front_feature_key}, wrist_key={self.wrist_feature_key}, "
            f"state_dim={vector_feature_dim(self.input_features.get('observation.state'))}, "
            f"action_dim={vector_feature_dim(self.output_features.get('action'))}, "
            f"command_mode={args.command_mode}, max_joint_step={args.max_joint_step_deg:.2f}deg, "
            f"joint_ema_alpha={args.joint_ema_alpha:.2f}, binary_gripper={args.binary_gripper}, "
            f"policy_image_channel_order={args.policy_image_channel_order}"
        )

    def close(self) -> None:
        self.preview_stop.set()
        if self.preview_thread is not None:
            self.preview_thread.join(timeout=1.0)
        if self.args.preview and not self.preview_failed:
            try:
                import cv2

                cv2.destroyWindow("RealMan SmolVLA Rollout")
            except Exception:
                pass
        if self.worker is not None:
            self.worker.close()
        if self.execute and self.args.stop_on_exit:
            self.robot.stop_arm()
        self.robot.close()

    def _move_home_blocking(self, qpos: np.ndarray) -> None:
        self.node.get_logger().info(f"Moving to RealMan home qpos: {fmt_vec(qpos)}")
        ret = self.robot.movej_blocking(qpos[:6], speed=int(self.args.home_speed), block=True)
        if ret != 0:
            raise RuntimeError(f"RealMan home move failed: ret={ret}")
        ret = self.robot.set_gripper_sync(float(qpos[6]), block=True, timeout_s=5.0)
        if ret != 0:
            raise RuntimeError(f"RealMan home gripper failed: ret={ret}")
        time.sleep(max(0.0, float(self.args.home_settle_s)))

    def _resolve_image_feature_keys(self) -> tuple[str, str]:
        front = str(self.args.front_feature_key)
        wrist = str(self.args.wrist_feature_key)
        if front not in self.input_features and "observation.images.cam_high" in self.input_features:
            front = "observation.images.cam_high"
        if wrist not in self.input_features and "observation.images.cam_wrist" in self.input_features:
            wrist = "observation.images.cam_wrist"
        return front, wrist

    def _validate_policy_dims(self) -> None:
        state_dim = vector_feature_dim(self.input_features.get("observation.state"))
        action_dim = vector_feature_dim(self.output_features.get("action"))
        problems = []
        if state_dim is not None and state_dim != 7:
            problems.append(f"observation.state dim is {state_dim}, expected 7")
        if action_dim is not None and action_dim != 7:
            problems.append(f"action dim is {action_dim}, expected 7")
        for key in (self.front_feature_key, self.wrist_feature_key):
            if key not in self.input_features:
                problems.append(f"policy missing image feature {key!r}")
        if problems and not self.args.allow_dim_mismatch:
            raise ValueError("; ".join(problems) + ". Pass --allow-dim-mismatch only for debugging.")
        if problems:
            self.node.get_logger().warn("Policy feature mismatch: " + "; ".join(problems))

    def _push(self, key: str, stamp: float, value: Any) -> None:
        self.buffers[key].push(stamp, value)
        self.topic_counts[key] += 1
        self.topic_last[key] = now_sec()

    def _on_front(self, msg: Any) -> None:
        self._push("front", now_sec(), msg)

    def _on_wrist(self, msg: Any) -> None:
        self._push("wrist", now_sec(), msg)

    def _sample_tick(self) -> None:
        if self.worker is not None and self.worker.error is not None:
            self.node.get_logger().error(f"SmolVLA inference failed: {self.worker.error}")
            return

        stamp = now_sec()
        front, dt_front = self.buffers["front"].nearest(stamp, self.args.max_dt_front_image)
        wrist, dt_wrist = self.buffers["wrist"].nearest(stamp, self.args.max_dt_wrist_image)
        try:
            state = self.robot.get_state(max_age_s=self.args.max_state_age_s, read_if_stale=False)
            state_age = time.time() - state.stamp
        except Exception as exc:
            if stamp - self.last_log_time > 2.0:
                self.last_log_time = stamp
                self.node.get_logger().warn(f"Waiting for RealMan cached state: {exc}")
            return

        self.last_dt_map = {"front": dt_front, "wrist": dt_wrist, "state": state_age}
        if front is None or wrist is None:
            if stamp - self.last_log_time > 2.0:
                self.last_log_time = stamp
                self.node.get_logger().warn(f"Waiting for aligned camera images: {self._format_status()}")
            return

        try:
            batch = {
                self.front_feature_key: image_msg_to_tensor(
                    front,
                    expected_hw=self.expected_hw.get(self.front_feature_key)
                    if self.args.resize_to_policy_shape
                    else None,
                    channel_order=self.args.policy_image_channel_order,
                ),
                self.wrist_feature_key: image_msg_to_tensor(
                    wrist,
                    expected_hw=self.expected_hw.get(self.wrist_feature_key)
                    if self.args.resize_to_policy_shape
                    else None,
                    channel_order=self.args.policy_image_channel_order,
                ),
                "observation.state": torch.from_numpy(state.as_qpos()).float(),
                "task": self.args.task,
            }
        except Exception as exc:
            self.node.get_logger().warn(f"Failed to build RealMan observation: {exc}")
            return

        packet = ObservationPacket(stamp=stamp, batch=batch, dt_map=self.last_dt_map.copy())
        if self.worker is not None:
            self.worker.submit(packet)
            return
        if self.inferencer is None:
            return
        with self.latest_action_lock:
            if self.sync_action_queue:
                return
        if self.args.replan_every_step and stamp - self.last_sync_infer_start < 1.0 / max(
            self.args.sync_inference_hz, 1e-6
        ):
            return
        if not self.sync_infer_lock.acquire(blocking=False):
            return
        self.last_sync_infer_start = stamp
        try:
            sequence_len = 1 if self.args.replan_every_step else self.execution_horizon
            action_packets = self.inferencer.infer_sequence(packet, sequence_len)
            with self.latest_action_lock:
                self.sync_action_queue.extend(action_packets)
            if stamp - self.last_log_time > 1.0 / max(self.args.dry_run_log_hz, 1e-6):
                self.last_log_time = stamp
                self.node.get_logger().info(
                    f"sync inference done actions={len(action_packets)} "
                    f"infer={(action_packets[0].inference_s if action_packets else 0.0) * 1000.0:.0f}ms "
                    f"{self._format_status()}"
                )
        except Exception as exc:
            self.node.get_logger().error(f"SmolVLA synchronous inference failed: {exc}")
        finally:
            self.sync_infer_lock.release()

    def _get_latest_action(self) -> ActionPacket | None:
        if self.worker is not None:
            return self.worker.get_latest()
        with self.latest_action_lock:
            if self.sync_action_queue:
                return self.sync_action_queue.popleft()
        return None

    def _command_tick(self) -> None:
        packet = self._get_latest_action()
        if packet is None:
            return
        stamp = now_sec()
        action_age = stamp - packet.stamp
        if action_age > self.args.max_action_age_s:
            if stamp - self.last_log_time > 2.0:
                self.last_log_time = stamp
                self.node.get_logger().warn(f"Holding: stale RealMan action age={action_age:.3f}s")
            return

        action = np.asarray(packet.action, dtype=float).reshape(-1)
        if action.size < 7:
            self.node.get_logger().warn(f"Bad RealMan model action shape: {action.shape}; expected at least 7")
            return
        qpos_cmd = action[:7].copy()
        qpos_cmd[6] = float(np.clip(qpos_cmd[6], 0.0, 1.0))

        current = None
        try:
            current = self.robot.get_qpos()
        except Exception:
            pass

        command_info: dict[str, Any] = {
            "sent": False,
            "ret": 0,
            "raw_joint_target": qpos_cmd[:6],
            "joint_target": qpos_cmd[:6],
            "gripper_queued": False,
        }
        command_dt = 0.0
        if self.execute:
            t0 = now_sec()
            try:
                command_info = self.robot.set_qpos(qpos_cmd, limit_step=True, queue_gripper=True)
            except Exception as exc:
                self.node.get_logger().error(f"RealMan command failed: {exc}")
                return
            command_dt = now_sec() - t0

        if stamp - self.last_log_time > 1.0 / max(self.args.dry_run_log_hz, 1e-6):
            self.last_log_time = stamp
            prefix = "execute" if self.execute else "dry-run"
            current_text = "none" if current is None else fmt_vec(current)
            self.node.get_logger().info(
                f"{prefix} q={current_text} raw={fmt_vec(qpos_cmd)} "
                f"cmd={fmt_vec(np.r_[command_info['joint_target'], qpos_cmd[6]])} "
                f"sent={command_info.get('sent')} ret={command_info.get('ret')} "
                f"gripper={qpos_cmd[6]:.3f} queued={command_info.get('gripper_queued')} "
                f"action_age={action_age * 1000.0:.0f}ms infer={packet.inference_s * 1000.0:.0f}ms "
                f"cmd_dt={command_dt * 1000.0:.1f}ms {self._format_status()}"
            )

    def _preview_loop(self) -> None:
        try:
            import cv2

            cv2.namedWindow("RealMan SmolVLA Rollout", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("RealMan SmolVLA Rollout", 1280, 480)
            self.node.get_logger().info("OpenCV RealMan rollout preview window started.")
        except Exception as exc:
            self.preview_failed = True
            self.node.get_logger().warn(f"OpenCV RealMan rollout preview disabled: {exc}")
            return

        period = 1.0 / max(self.args.preview_hz, 1.0)
        next_t = now_sec()
        while not self.preview_stop.is_set() and not self.preview_failed:
            self._preview_tick()
            next_t += period
            time.sleep(max(0.001, next_t - now_sec()))

    def _preview_tick(self) -> None:
        if self.preview_failed:
            return
        front, front_age = self.buffers["front"].latest()
        wrist, wrist_age = self.buffers["wrist"].latest()
        try:
            import cv2

            front_img = preview_image(front, "front", front_age)
            wrist_img = preview_image(wrist, "wrist", wrist_age)
            height = min(front_img.shape[0], wrist_img.shape[0])
            frame = np.hstack([resize_to_height(front_img, height), resize_to_height(wrist_img, height)])
            cv2.imshow("RealMan SmolVLA Rollout", frame)
            cv2.waitKey(1)
        except Exception as exc:
            self.preview_failed = True
            self.node.get_logger().warn(f"OpenCV RealMan rollout preview disabled: {exc}")

    def _format_status(self) -> str:
        dt = " ".join(
            f"{key}={'none' if value is None else f'{value * 1000.0:.0f}ms'}"
            for key, value in self.last_dt_map.items()
        )
        now = now_sec()
        topics = " ".join(
            f"{key}#{self.topic_counts[key]}@"
            f"{'none' if self.topic_counts[key] <= 0 else f'{(now - self.topic_last[key]) * 1000.0:.0f}ms'}"
            for key in ("front", "wrist")
        )
        return f"sync[{dt}] topics[{topics}]"


def preview_image(msg: Any, label: str, age: float | None) -> np.ndarray:
    import cv2

    if msg is None:
        image = np.zeros((360, 480, 3), dtype=np.uint8)
        status = "waiting"
    else:
        image = cv2.cvtColor(decode_image_msg(msg), cv2.COLOR_RGB2BGR)
        status = f"age={age * 1000.0:.0f}ms" if age is not None else "age=none"
    cv2.rectangle(image, (0, 0), (image.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(
        image,
        f"{label}  {status}",
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SmolVLA 7D joint-space rollout for RealMan RM65.")
    parser.add_argument("--policy-path", type=Path, default=CFG.policy_path)
    parser.add_argument("--output-root", type=Path, default=CFG.output_root)
    parser.add_argument("--task", default=CFG.task)
    parser.add_argument("--robot-ip", default=CFG.robot_ip)
    parser.add_argument("--robot-port", type=int, default=CFG.robot_port)
    parser.add_argument("--rm-api-path", default=CFG.rm_api_path)
    parser.add_argument("--front-image-topic", default=CFG.front_image_topic)
    parser.add_argument("--wrist-image-topic", default=CFG.wrist_image_topic)
    parser.add_argument("--front-feature-key", default=CFG.front_feature_key)
    parser.add_argument("--wrist-feature-key", default=CFG.wrist_feature_key)
    parser.add_argument("--fps", type=float, default=CFG.fps)
    parser.add_argument("--command-hz", type=float, default=CFG.command_hz)
    parser.add_argument("--state-cache-hz", type=float, default=CFG.state_cache_hz)
    parser.add_argument("--inference-mode", choices=["async", "sync"], default=CFG.inference_mode)
    parser.add_argument("--replan-every-step", action=argparse.BooleanOptionalAction, default=CFG.replan_every_step)
    parser.add_argument("--execution-horizon", type=int, default=CFG.execution_horizon)
    parser.add_argument("--sync-inference-hz", type=float, default=CFG.sync_inference_hz)
    parser.add_argument("--max-dt-front-image", type=float, default=CFG.max_dt_front_image)
    parser.add_argument("--max-dt-wrist-image", type=float, default=CFG.max_dt_wrist_image)
    parser.add_argument("--max-state-age-s", type=float, default=CFG.max_state_age_s)
    parser.add_argument("--max-action-age-s", type=float, default=CFG.max_action_age_s)
    parser.add_argument("--buffer-maxlen", type=int, default=CFG.buffer_maxlen)
    parser.add_argument("--command-mode", choices=["follow", "canfd"], default=CFG.command_mode)
    parser.add_argument("--canfd-follow", action=argparse.BooleanOptionalAction, default=CFG.canfd_follow)
    parser.add_argument("--canfd-trajectory-mode", type=int, default=CFG.canfd_trajectory_mode)
    parser.add_argument("--canfd-radio", type=int, default=CFG.canfd_radio)
    parser.add_argument("--max-joint-step-deg", type=float, default=CFG.max_joint_step_deg)
    parser.add_argument("--joint-deadband-deg", type=float, default=CFG.joint_deadband_deg)
    parser.add_argument("--joint-ema-alpha", type=float, default=CFG.joint_ema_alpha)
    parser.add_argument("--gripper-deadband", type=float, default=CFG.gripper_deadband)
    parser.add_argument("--binary-gripper", action=argparse.BooleanOptionalAction, default=CFG.binary_gripper)
    parser.add_argument("--gripper-open-threshold", type=float, default=CFG.gripper_open_threshold)
    parser.add_argument("--gripper-close-threshold", type=float, default=CFG.gripper_close_threshold)
    parser.add_argument("--gripper-open-value", type=float, default=CFG.gripper_open_value)
    parser.add_argument("--gripper-closed-value", type=float, default=CFG.gripper_closed_value)
    parser.add_argument("--initial-gripper-value", type=float, default=CFG.initial_gripper_value)
    parser.add_argument("--move-home-on-start", action=argparse.BooleanOptionalAction, default=CFG.move_home_on_start)
    parser.add_argument("--home-qpos", type=parse_qpos, default=np.asarray(CFG.home_qpos, dtype=float))
    parser.add_argument("--home-speed", type=int, default=CFG.home_speed)
    parser.add_argument("--home-settle-s", type=float, default=CFG.home_settle_s)
    parser.add_argument("--preview", action=argparse.BooleanOptionalAction, default=CFG.preview)
    parser.add_argument("--preview-hz", type=float, default=CFG.preview_hz)
    parser.add_argument("--resize-to-policy-shape", action=argparse.BooleanOptionalAction, default=CFG.resize_to_policy_shape)
    parser.add_argument(
        "--policy-image-channel-order",
        choices=["rgb", "bgr"],
        default=CFG.policy_image_channel_order,
        help=(
            "Channel order used only for tensors sent to the policy. Use 'bgr' "
            "to diagnose legacy RealMan checkpoints trained from BGR frames "
            "converted without BGR->RGB. Preview remains normal-color."
        ),
    )
    parser.add_argument("--allow-dim-mismatch", action=argparse.BooleanOptionalAction, default=CFG.allow_dim_mismatch)
    parser.add_argument("--stop-on-exit", action=argparse.BooleanOptionalAction, default=CFG.stop_on_exit)
    parser.add_argument("--dry-run-log-hz", type=float, default=CFG.dry_run_log_hz)
    parser.add_argument("--device", default=CFG.device)
    parser.add_argument("--execute", action="store_true", help="Actually send 7D model actions to the RealMan arm.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("HF_HOME", str(CFG.hf_home))
    os.environ.setdefault("HF_DATASETS_CACHE", str(CFG.hf_home / "datasets"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    args.output_root = args.output_root.expanduser()
    if not args.output_root.is_absolute():
        args.output_root = (REPO_ROOT / args.output_root).resolve()
    args.policy_path = resolve_realman_policy_path(args.policy_path, args.output_root)

    import rclpy
    from rclpy.executors import MultiThreadedExecutor

    rclpy.init()
    node = rclpy.create_node("realman_smolvla_rollout")
    rollout = None
    executor = MultiThreadedExecutor(num_threads=3)
    try:
        rollout = RealManSmolVLARolloutNode(node, args)
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if rollout is not None:
            rollout.close()
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
