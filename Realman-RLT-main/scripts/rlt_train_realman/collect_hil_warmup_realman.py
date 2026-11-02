#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.collect_data_realman.collect_data import QUEST_R_HEADSET_TO_WORLD  # noqa: E402
from scripts.inference_realman.inference import (  # noqa: E402
    GRIPPER_MAX_POSITION,
    GRIPPER_MIN_POSITION,
    GRIPPER_PORT,
    RobotController,
    _patch_config_compat,
    add_rm_api_path,
    bgr_frame_to_policy_tensor as inference_bgr_frame_to_policy_tensor,
    create_camera as create_inference_camera,
    gripper_position_to_value,
    gripper_value_to_position,
    load_policy,
)
from scripts.rlt_gate.eval_rlt_gate import load_checkpoint as load_gate_checkpoint  # noqa: E402
from scripts.rlt_gate.live_rlt_gate_monitor import compose_input  # noqa: E402
from scripts.rlt_token.train_rlt_stage1 import make_stage1_model  # noqa: E402
from scripts.rlt_train.hil_serl_core import HILSERLConfig, HILSERLTrainer, HILTransition, ReplayBuffer  # noqa: E402
from scripts.rlt_train_realman.config import CFG as STAGE2_CFG  # noqa: E402
from scripts.rollout_smolvla.rollout_ur3e_smolvla import (  # noqa: E402
    ActionPacket,
    ObservationPacket,
    fmt_vec,
    now_sec,
)
from scripts.rollout_smolvla_realman.rollout_realman_smolvla import (  # noqa: E402
    normalize_policy_path,
    resolve_realman_policy_path,
    vector_feature_dim,
)
from scripts.rollout_smolvla_realman.config import CFG as ROLLOUT_CFG  # noqa: E402


DEFAULT_USB_TASK = "把USB头从右侧抓起后插入左侧USB口"
USB_INSERT_HOME_QPOS = np.asarray([12.817, 19.830, 82.662, 0.205, 77.997, 42.185, 1.0], dtype=float)


@dataclass(slots=True)
class WarmActionPacket:
    stamp: float
    action: np.ndarray
    ref_action_chunk: np.ndarray
    step_index: int
    z_rl: np.ndarray
    gate_prob: float
    inference_s: float


@dataclass(slots=True)
class QuestMotionSample:
    stamp: float
    state_qpos: np.ndarray
    target_pose: np.ndarray
    ret: int


class ButtonEdges:
    def __init__(self) -> None:
        self.prev: dict[str, bool] = {}

    def rising(self, name: str, value: bool) -> bool:
        was = bool(self.prev.get(name, False))
        self.prev[name] = bool(value)
        return bool(value) and not was


@dataclass(slots=True)
class DirectRobotState:
    joint_deg: np.ndarray
    tcp_pos: np.ndarray
    tcp_euler: np.ndarray
    gripper: float

    def as_qpos(self) -> np.ndarray:
        return np.r_[self.joint_deg, self.gripper].astype(np.float32)


class DirectPolicyRunner:
    """Policy runner matching scripts/inference_realman/inference.py."""

    def __init__(self, policy_path: Path, device: str) -> None:
        from lerobot.processor.pipeline import DataProcessorPipeline

        self.policy_path = Path(policy_path)
        self.device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        self.policy, self.policy_type = self._load_policy()
        self.preprocessor = DataProcessorPipeline.from_pretrained(
            str(self.policy_path),
            config_filename="policy_preprocessor.json",
        )
        self.postprocessor = DataProcessorPipeline.from_pretrained(
            str(self.policy_path),
            config_filename="policy_postprocessor.json",
        )
        self.policy.to(self.device)
        self.policy.eval()
        self.ckpt_action_horizon = int(
            getattr(self.policy.config, "n_action_steps", getattr(self.policy.config, "chunk_size", 1))
        )

    def _load_policy(self) -> tuple[Any, str]:
        config_path = self.policy_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"找不到配置文件: {config_path}")
        with config_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        policy_type = str(cfg.get("type", "unknown"))
        print(f"检测到策略类型: {policy_type}")
        if policy_type != "smolvla":
            return load_policy(str(self.policy_path), self.device)

        # SmolVLA is large enough that loading on CPU and then moving to CUDA
        # creates a high transient host-memory peak.  Match the rollout path and
        # pass the target device into from_pretrained directly.
        import lerobot.policies.smolvla.processor_smolvla  # noqa: F401
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        _patch_config_compat(config_path)
        device_override = str(self.device)
        config = PreTrainedConfig.from_pretrained(
            str(self.policy_path),
            local_files_only=True,
            cli_overrides=[f"--device={device_override}", "--load_vlm_weights=false"],
        )
        print(
            f"Loading SmolVLA checkpoint directly on {device_override} "
            "with load_vlm_weights=False",
            flush=True,
        )
        policy = SmolVLAPolicy.from_pretrained(
            str(self.policy_path),
            config=config,
            local_files_only=True,
        )
        policy.to(self.device)
        policy.eval()
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return policy, policy_type

    @torch.inference_mode()
    def infer(self, packet: ObservationPacket) -> ActionPacket:
        start = now_sec()
        observation = self.preprocessor(dict(packet.batch))
        action_tensor = self.policy.select_action(observation)
        action_dict = self.postprocessor({"action": action_tensor})
        action = np.asarray(action_dict["action"][0].detach().cpu(), dtype=np.float32).reshape(-1)
        return ActionPacket(stamp=packet.stamp, action=action, inference_s=now_sec() - start)

    def reset(self) -> None:
        try:
            self.policy.reset()
        except Exception:
            pass


class SharedSmolVLATokenRunner:
    """RLT token helper that reuses the already-loaded SmolVLA policy.

    The warmup collector also needs SmolVLA internals to build the Stage1 RLT
    token.  Loading a second SmolVLA checkpoint for that path can push the
    32GB host over the edge during startup, so this runner shares the policy
    owned by DirectPolicyRunner and only loads the lightweight processors.
    """

    def __init__(
        self,
        *,
        policy: Any,
        policy_path: Path,
        device: str,
        execution_horizon: int = 50,
    ) -> None:
        # Importing this module registers SmolVLA processor steps used in saved processor configs.
        import lerobot.policies.smolvla.processor_smolvla  # noqa: F401
        from lerobot.processor import PolicyProcessorPipeline
        from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
        from lerobot.utils.constants import (
            OBS_LANGUAGE_ATTENTION_MASK,
            OBS_LANGUAGE_TOKENS,
            POLICY_POSTPROCESSOR_DEFAULT_NAME,
            POLICY_PREPROCESSOR_DEFAULT_NAME,
        )

        self.policy = policy
        self.policy_path = Path(policy_path)
        self.device = device
        self.ckpt_action_horizon = int(getattr(self.policy.config, "n_action_steps", 1))
        chunk_size = int(getattr(self.policy.config, "chunk_size", self.ckpt_action_horizon))
        self.execution_horizon = max(1, min(int(execution_horizon), chunk_size))
        self.policy.config.n_action_steps = self.execution_horizon
        try:
            self.policy.reset()
        except Exception:
            pass
        self.preprocessor = PolicyProcessorPipeline.from_pretrained(
            self.policy_path,
            config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
            local_files_only=True,
            overrides={"device_processor": {"device": device}},
        )
        self.postprocessor = PolicyProcessorPipeline.from_pretrained(
            self.policy_path,
            config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
            local_files_only=True,
            overrides={"device_processor": {"device": "cpu"}},
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        )
        self.input_keys = {
            *self.policy.config.input_features.keys(),
            OBS_LANGUAGE_TOKENS,
            OBS_LANGUAGE_ATTENTION_MASK,
        }

    @torch.inference_mode()
    def infer_sequence(self, packet: ObservationPacket, count: int) -> list[ActionPacket]:
        start = now_sec()
        batch = self.preprocessor(packet.batch)
        batch = {key: value for key, value in batch.items() if key in self.input_keys}
        action_chunk = self.policy.predict_action_chunk(batch)
        action_chunk = self.postprocessor(action_chunk.detach().cpu())
        arr = np.asarray(action_chunk.detach().cpu(), dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[0]
        elif arr.ndim == 1:
            arr = arr.reshape(1, -1)
        arr = arr[: max(1, int(count))]
        inference_s = now_sec() - start
        return [ActionPacket(stamp=packet.stamp, action=action.reshape(-1), inference_s=inference_s) for action in arr]


class DirectQuest:
    """Small Quest input/teleop wrapper for RealMan warmup collection.

    It intentionally mirrors the already-tested RealMan collection script:
    right Grip is the Cartesian clutch, and right Trigger toggles the gripper.
    """

    def __init__(
        self,
        *,
        pos_scale: float,
        rot_scale: float,
        yaw_deg: float,
        grip_threshold: float,
        button_threshold: float,
        button_debounce_s: float,
        lock_orientation: bool,
        debug: bool,
    ) -> None:
        import xrobotoolkit_sdk as xrt
        from scipy.spatial.transform import Rotation

        xrt.init()
        self.xrt = xrt
        self.Rotation = Rotation
        self.headset_rot = Rotation.from_matrix(QUEST_R_HEADSET_TO_WORLD)
        self.pos_scale = float(pos_scale)
        self.rot_scale = float(rot_scale)
        self.yaw_rad = np.deg2rad(float(yaw_deg))
        self.grip_threshold = float(grip_threshold)
        self.button_threshold = float(button_threshold)
        self.button_debounce_s = max(0.0, float(button_debounce_s))
        self.lock_orientation = bool(lock_orientation)
        self.debug = bool(debug)

        self.active = False
        self.blocked_until_release = False
        self.prev_xyz: np.ndarray | None = None
        self.prev_rot: Any | None = None
        self.target_pos: np.ndarray | None = None
        self.target_euler: np.ndarray | None = None
        self.last_debug_t = 0.0
        self._event_queue: queue.Queue[str] = queue.Queue()
        self._button_prev: dict[str, bool] = {}
        self._button_last_event_t: dict[str, float] = {}
        self._state_reader: Any | None = None
        self._command_sender: Any | None = None
        self._motion_period_s = 0.02
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._latest_sample: QuestMotionSample | None = None
        self._intervention_active = False
        self._last_debug_keys: dict[str, float] = {}

    def close(self) -> None:
        self.stop_motion_loop()
        try:
            self.xrt.close()
        except Exception:
            pass

    def start_motion_loop(
        self,
        *,
        state_reader: Any,
        command_sender: Any,
        period_s: float = 0.02,
    ) -> None:
        """Start a collection-script-style 50Hz Quest motion loop."""
        self._state_reader = state_reader
        self._command_sender = command_sender
        self._motion_period_s = max(0.005, float(period_s))
        if self._thread is not None and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(target=self._motion_loop, name="realman_warmup_quest", daemon=True)
        self._thread.start()

    def stop_motion_loop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def key(self, name: str) -> float:
        if name == "left_trigger":
            return float(self.xrt.get_left_trigger())
        if name == "right_trigger":
            return float(self.xrt.get_right_trigger())
        if name == "left_grip":
            return float(self.xrt.get_left_grip())
        if name == "right_grip":
            return float(self.xrt.get_right_grip())
        if name == "a":
            return float(self.xrt.get_A_button())
        if name == "b":
            return float(self.xrt.get_B_button())
        if name == "x":
            return float(self.xrt.get_X_button())
        if name == "y":
            return float(self.xrt.get_Y_button())
        return 0.0

    def pressed(self, name: str) -> bool:
        return self.key(name) > self.button_threshold

    def intervention_pressed(self) -> bool:
        return self.key("right_grip") > self.grip_threshold

    def right_controller_pose(self) -> tuple[np.ndarray | None, Any | None]:
        pose = np.asarray(self.xrt.get_right_controller_pose(), dtype=float).reshape(-1)
        if pose.size < 7 or not np.all(np.isfinite(pose[:7])) or np.allclose(pose[:7], 0.0):
            return None, None
        xyz = QUEST_R_HEADSET_TO_WORLD @ pose[:3]
        raw_rot = self.Rotation.from_quat(pose[3:7])
        rot = self.headset_rot * raw_rot * self.headset_rot.inv()
        return xyz.astype(float), rot

    def _apply_yaw(self, delta_xyz: np.ndarray) -> np.ndarray:
        if abs(self.yaw_rad) < 1e-9:
            return delta_xyz
        c = float(np.cos(self.yaw_rad))
        s = float(np.sin(self.yaw_rad))
        corrected = delta_xyz.copy()
        corrected[:2] = np.asarray(
            [
                c * delta_xyz[0] - s * delta_xyz[1],
                s * delta_xyz[0] + c * delta_xyz[1],
            ],
            dtype=float,
        )
        return corrected

    def _activate_from_robot(self, state: Any) -> None:
        self.target_pos = np.asarray(state.tcp_pos, dtype=float).reshape(3).copy()
        self.target_euler = np.asarray(state.tcp_euler, dtype=float).reshape(3).copy()
        self.prev_xyz = None
        self.prev_rot = None
        self.active = True
        print("\n[quest] 右 Grip 接管：已启用末端平移控制", flush=True)

    def deactivate(self) -> None:
        if self.active:
            print("\n[quest] 右 Grip 松开：退出人工接管", flush=True)
        self.active = False
        self.prev_xyz = None
        self.prev_rot = None
        with self._lock:
            self._intervention_active = False
            self._latest_sample = None

    def pause_motion(self, *, block_until_release: bool = False, print_result: bool = False) -> None:
        was_active = self.active
        self.active = False
        self.prev_xyz = None
        self.prev_rot = None
        if block_until_release:
            self.blocked_until_release = True
        with self._lock:
            self._intervention_active = False
            self._latest_sample = None
        if print_result or was_active:
            suffix = "，等待右 Grip 松开后再允许接管" if block_until_release else ""
            print(f"\n[quest] 遥控暂停{suffix}", flush=True)

    def target_pose(self, robot_state: Any) -> np.ndarray | None:
        pressed = self.intervention_pressed()
        if self.blocked_until_release:
            if not pressed:
                self.blocked_until_release = False
            return None
        if not pressed:
            self.deactivate()
            return None
        if not self.active:
            self._activate_from_robot(robot_state)

        xyz, rot = self.right_controller_pose()
        if xyz is None:
            return None
        if self.prev_xyz is None:
            self.prev_xyz = xyz
            self.prev_rot = rot
            return None

        assert self.prev_rot is not None
        assert self.target_pos is not None
        assert self.target_euler is not None
        delta_xyz = (xyz - self.prev_xyz) * self.pos_scale
        delta_xyz = self._apply_yaw(delta_xyz)
        delta_rot = (rot * self.prev_rot.inv()).as_rotvec() * self.rot_scale
        self.prev_xyz = xyz
        self.prev_rot = rot

        self.target_pos = self.target_pos + delta_xyz
        if not self.lock_orientation:
            self.target_euler = self.target_euler + delta_rot

        # Same conservative box as the RealMan Quest collection script.
        self.target_pos[0] = np.clip(self.target_pos[0], -0.5, 0.1)
        self.target_pos[1] = np.clip(self.target_pos[1], -0.3, 0.3)
        self.target_pos[2] = np.clip(self.target_pos[2], 0.1, 0.6)
        return np.r_[self.target_pos, self.target_euler].astype(np.float32)

    def drain_events(self) -> list[str]:
        events: list[str] = []
        while True:
            try:
                events.append(self._event_queue.get_nowait())
            except queue.Empty:
                return events

    def latest_intervention(self) -> tuple[bool, QuestMotionSample | None]:
        with self._lock:
            return bool(self._intervention_active), self._latest_sample

    def _push_button_event(self, name: str, pressed: bool) -> None:
        was = bool(self._button_prev.get(name, False))
        self._button_prev[name] = bool(pressed)
        now = time.monotonic()
        last = float(self._button_last_event_t.get(name, -1e9))
        if pressed and not was and now - last >= self.button_debounce_s:
            self._button_last_event_t[name] = now
            self._event_queue.put(name)

    def _sample_buttons(self) -> None:
        for name in ("left_grip", "left_trigger", "y", "x", "a", "right_trigger"):
            threshold = 0.5 if name == "right_trigger" else self.button_threshold
            self._push_button_event(name, self.key(name) > threshold)

    def _motion_loop(self) -> None:
        assert self._state_reader is not None
        assert self._command_sender is not None
        while self._running:
            t0 = time.monotonic()
            try:
                self._sample_buttons()
                self._handle_motion_once()
                self.debug_print()
            except Exception as exc:
                print(f"\n[quest] motion loop failed: {exc}", flush=True)
                time.sleep(0.2)
            sleep_s = self._motion_period_s - (time.monotonic() - t0)
            if sleep_s > 0:
                time.sleep(sleep_s)

    def _handle_motion_once(self) -> None:
        assert self._state_reader is not None
        assert self._command_sender is not None
        pressed = self.intervention_pressed()
        if self.blocked_until_release:
            if not pressed:
                self.blocked_until_release = False
            with self._lock:
                self._intervention_active = False
                self._latest_sample = None
            return
        if not pressed:
            self.deactivate()
            return

        with self._lock:
            self._intervention_active = True
        state = self._state_reader()
        target_pose = self.target_pose(state)
        if target_pose is None:
            return
        ret = int(self._command_sender(target_pose))
        sample = QuestMotionSample(
            stamp=now_sec(),
            state_qpos=state.as_qpos().astype(np.float32),
            target_pose=np.asarray(target_pose, dtype=np.float32),
            ret=ret,
        )
        with self._lock:
            self._latest_sample = sample

    def debug_print(self) -> None:
        if not self.debug:
            return
        now = time.monotonic()
        if now - self.last_debug_t < 1.0:
            return
        self.last_debug_t = now
        try:
            pose = np.asarray(self.xrt.get_right_controller_pose(), dtype=float).reshape(-1)
            pose_xyz = np.round(pose[:3], 3).tolist() if pose.size >= 3 else []
            print(
                "\n[quest debug] "
                f"rt={self.key('right_trigger'):.3f} rg={self.key('right_grip'):.3f} "
                f"lt={self.key('left_trigger'):.3f} lg={self.key('left_grip'):.3f} "
                f"A={self.key('a'):.3f} B={self.key('b'):.3f} "
                f"X={self.key('x'):.3f} Y={self.key('y'):.3f} "
                f"pose_xyz={pose_xyz} active={self.active}",
                flush=True,
            )
        except Exception as exc:
            print(f"\n[quest debug] failed: {exc}", flush=True)


class OnlineRLTokenEncoder:
    """Online Stage1 RLT encoder for a frozen SmolVLA policy/action chunk."""

    def __init__(self, runner: Any, stage1_path: Path, device: str, *, amp: bool) -> None:
        from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
        from lerobot.utils.constants import (
            ACTION,
            OBS_LANGUAGE_ATTENTION_MASK,
            OBS_LANGUAGE_TOKENS,
        )

        self.runner = runner
        self.device = device
        self.amp = bool(amp and device == "cuda")
        self.action_key = ACTION
        self.obs_language_tokens = OBS_LANGUAGE_TOKENS
        self.obs_language_attention_mask = OBS_LANGUAGE_ATTENTION_MASK
        self.make_att_2d_masks = make_att_2d_masks

        ckpt = torch.load(stage1_path, map_location=device)
        model_cfg = ckpt["model_config"]
        self.model_cfg = model_cfg
        self.architecture = str(model_cfg.get("architecture", "pooled"))
        self.chunk_size = int(model_cfg.get("chunk_size", runner.ckpt_action_horizon))
        self.model = make_stage1_model(model_cfg).to(device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    @staticmethod
    def masked_mean(embs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.to(dtype=embs.dtype).unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (embs * weights).sum(dim=1) / denom

    def action_chunk(self, actions: list[ActionPacket]) -> np.ndarray:
        arr = np.stack([np.asarray(item.action, dtype=np.float32).reshape(-1) for item in actions], axis=0)
        if arr.shape[0] < self.chunk_size:
            arr = np.concatenate([arr, np.repeat(arr[-1:], self.chunk_size - arr.shape[0], axis=0)], axis=0)
        return arr[: self.chunk_size].astype(np.float32, copy=True)

    @torch.inference_mode()
    def encode(self, packet: ObservationPacket, actions: list[ActionPacket]) -> np.ndarray:
        if not actions:
            raise ValueError("Cannot encode RLT token without a SmolVLA action chunk.")
        autocast = torch.autocast(device_type="cuda") if self.amp else torch.no_grad()
        with autocast:
            batch = dict(packet.batch)
            batch[self.action_key] = torch.from_numpy(self.action_chunk(actions)).unsqueeze(0)
            processed = self.runner.preprocessor(batch)
            input_keys = {
                *self.runner.policy.config.input_features.keys(),
                self.obs_language_tokens,
                self.obs_language_attention_mask,
                self.action_key,
            }
            processed = {key: value for key, value in processed.items() if key in input_keys}
            processed = self.runner.policy._prepare_batch(processed)

            images, img_masks = self.runner.policy.prepare_images(processed)
            state = self.runner.policy.prepare_state(processed)
            action_chunk = self.runner.policy.prepare_action(processed)
            lang_tokens = processed[self.obs_language_tokens]
            lang_masks = processed[self.obs_language_attention_mask]

            prefix_embs, prefix_pad_masks, prefix_att_masks = self.runner.policy.model.embed_prefix(
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                state=state,
            )
            timestep = torch.zeros(action_chunk.shape[0], dtype=torch.float32, device=action_chunk.device)
            suffix_embs, suffix_pad_masks, suffix_att_masks = self.runner.policy.model.embed_suffix(action_chunk, timestep)
            pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
            att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
            att_2d_masks = self.make_att_2d_masks(pad_masks, att_masks)
            position_ids = torch.cumsum(pad_masks, dim=1) - 1
            (prefix_out, suffix_out), _ = self.runner.policy.model.vlm_with_expert.forward(
                attention_mask=att_2d_masks,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                fill_kv_cache=False,
            )
            if self.architecture == "sequence":
                z_rl = self.model.encode(
                    prefix_out.float(),
                    suffix_out.float(),
                    prefix_pad_masks.to(dtype=torch.bool),
                    suffix_pad_masks.to(dtype=torch.bool),
                )
            else:
                vlm = self.masked_mean(prefix_out.float(), prefix_pad_masks)
                expert = self.masked_mean(suffix_out.float(), suffix_pad_masks)
                z_rl = self.model.encode(vlm, expert)
        return z_rl.detach().cpu().numpy().astype(np.float32).reshape(-1)


def resolve_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    path = path.expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def parse_qpos(text: str) -> np.ndarray:
    values = [float(part.strip()) for part in str(text).split(",") if part.strip()]
    arr = np.asarray(values, dtype=float)
    if arr.shape != (7,):
        raise argparse.ArgumentTypeError(f"qpos must be 7 comma-separated numbers, got {arr.shape}: {text!r}")
    return arr


def default_output_dir(root: Path) -> Path:
    root = resolve_path(root)
    assert root is not None
    return root / "hil_warmup_realman_usb_insert_v1"


def next_episode_index(output_dir: Path) -> int:
    max_idx = -1
    for path in output_dir.glob("episode_*"):
        if not path.is_dir():
            continue
        try:
            idx = int(path.name.rsplit("_", 1)[-1])
        except ValueError:
            continue
        max_idx = max(max_idx, idx)
    return max_idx + 1


def bgr_frame_to_policy_tensor(
    frame_bgr: np.ndarray,
    *,
    expected_hw: tuple[int, int] | None,
    channel_order: str,
) -> torch.Tensor:
    import cv2

    channel_order = str(channel_order).lower()
    if channel_order == "rgb":
        image = frame_bgr[:, :, ::-1].copy()
    elif channel_order == "bgr":
        image = frame_bgr.copy()
    else:
        raise ValueError(f"Unsupported policy image channel order: {channel_order!r}")
    if expected_hw is not None and image.shape[:2] != expected_hw:
        height, width = expected_hw
        image = cv2.resize(image, (int(width), int(height)), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255.0


def camera_is_ok(camera: Any) -> bool:
    return bool(getattr(camera, "is_active", True))


def parse_fps_fallbacks(text: str) -> list[int]:
    values: list[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    return values


class RealManWarmupCollector:
    def _startup_profile(self, label: str) -> None:
        if not bool(getattr(self.args, "startup_profile", False)):
            return
        try:
            import resource

            rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
            cuda_mb = torch.cuda.memory_allocated() / 1024.0 / 1024.0 if torch.cuda.is_available() else 0.0
            print(f"[startup] {label}: rss={rss_mb:.1f}MB cuda={cuda_mb:.1f}MB", flush=True)
        except Exception as exc:
            print(f"[startup] {label}: profile failed: {exc}", flush=True)

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.execute = bool(args.execute)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = resolve_path(args.output_dir) if args.output_dir is not None else default_output_dir(args.output_root)
        assert self.output_dir is not None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "buffers").mkdir(parents=True, exist_ok=True)

        self.policy_path = normalize_policy_path(args.policy_path) if args.policy_path is not None else resolve_realman_policy_path(
            None, ROLLOUT_CFG.output_root
        )
        self.stage1_checkpoint = resolve_path(args.stage1_checkpoint)
        self.gate_checkpoint = resolve_path(args.gate_checkpoint)
        self.stage2_checkpoint = resolve_path(args.stage2_checkpoint)
        assert self.stage1_checkpoint is not None
        assert self.gate_checkpoint is not None
        self._startup_profile("paths")

        add_rm_api_path(args.rm_api_path)
        from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e

        arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        handle = arm.rm_create_robot_arm(args.robot_ip, args.robot_port)
        if getattr(handle, "id", -1) == -1:
            raise RuntimeError(f"机械臂连接失败: {args.robot_ip}:{args.robot_port}")
        self._startup_profile("robot connected")
        try:
            arm.rm_stop_drag_teach()
            arm.rm_set_tool_voltage(3)
            arm.rm_close_modbus_mode(GRIPPER_PORT)
            time.sleep(0.3)
        except Exception as exc:
            print(f"[warn] RealMan init command failed: {exc}", flush=True)
        self._startup_profile("robot init")
        self.robot = RobotController(
            arm,
            joint_command_mode=args.command_mode,
            canfd_follow=args.canfd_follow,
            canfd_trajectory_mode=args.canfd_trajectory_mode,
            canfd_radio=args.canfd_radio,
            use_live_state=True,
            sync_gripper=False,
            binary_gripper=args.binary_gripper,
            gripper_open_threshold=args.gripper_open_threshold,
            gripper_close_threshold=args.gripper_close_threshold,
            max_joint_step=args.max_joint_step_deg,
        )
        self.robot.joint_deadzone = float(args.joint_deadband_deg)
        self.robot.ema_alpha = float(args.joint_ema_alpha)
        print("[realman] connected through copied inference RobotController", flush=True)
        self._startup_profile("robot controller")
        if self.execute and args.move_home_on_start:
            print(f"[realman] moving home: {fmt_vec(args.home_qpos)}", flush=True)
            self.robot.move_to_init(np.asarray(args.home_qpos, dtype=np.float32))
            time.sleep(max(0.0, float(args.home_settle_s)))

        self.quest = DirectQuest(
            pos_scale=args.quest_pos_scale,
            rot_scale=args.quest_rot_scale,
            yaw_deg=args.quest_yaw_deg,
            grip_threshold=args.quest_grip_threshold,
            button_threshold=args.quest_button_threshold,
            button_debounce_s=args.quest_button_debounce_s,
            lock_orientation=args.quest_lock_orientation,
            debug=args.quest_debug,
        )
        self.buttons = ButtonEdges()
        self._startup_profile("quest")

        self.control_runner = DirectPolicyRunner(self.policy_path, args.device)
        self._startup_profile("control runner")
        policy_config = self.control_runner.policy.config
        self.input_features = getattr(policy_config, "input_features", {})
        self.output_features = getattr(policy_config, "output_features", {})
        self.front_feature_key, self.wrist_feature_key = self._resolve_image_feature_keys()
        self._validate_policy_dims()
        self.token_runner = SharedSmolVLATokenRunner(
            policy=self.control_runner.policy,
            policy_path=self.policy_path,
            device=args.device,
            execution_horizon=int(args.execution_horizon),
        )
        self._startup_profile("token runner")
        self.token_encoder = OnlineRLTokenEncoder(
            self.token_runner,
            self.stage1_checkpoint,
            args.device,
            amp=bool(args.amp),
        )
        self._startup_profile("token encoder")

        self.gate_device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
        self.gate_model, gate_cfg = load_gate_checkpoint(self.gate_checkpoint, self.gate_device)
        self._startup_profile("gate")
        self.gate_camera = args.gate_camera if args.gate_camera != "auto" else str(gate_cfg.get("camera", "both"))
        self.gate_image_size = int(args.gate_image_size or gate_cfg.get("image_size", 224))
        default_t = float(gate_cfg.get("threshold", 0.5))
        self.gate_pos_t = float(args.gate_positive_threshold if args.gate_positive_threshold is not None else default_t)
        self.gate_neg_t = float(args.gate_negative_threshold if args.gate_negative_threshold is not None else default_t)
        self.gate_hold_frames = max(1, int(args.gate_hold_frames))
        self.gate_pos_count = 0
        self.gate_neg_count = 0
        self.gate_phase = 0
        self.gate_prob: float | None = None

        self.stage2_trainer: HILSERLTrainer | None = None
        if self.stage2_checkpoint is not None:
            self.stage2_trainer = self._load_stage2_trainer(self.stage2_checkpoint, args.device)
            self._startup_profile("stage2 actor")

        # Open cameras only after all heavyweight models are loaded.  Some
        # camera backends buffer frames aggressively while the process is busy
        # loading SmolVLA/RLT/Gate; opening cameras first can push host memory
        # into OOM before the collector reaches [ready].
        fallback_fps = parse_fps_fallbacks(args.camera_fps_fallbacks)
        self.front_cam, self.front_camera_fps = self._create_camera_with_fallback(
            "top/front",
            args.cam_top,
            width=int(args.cam_top_width),
            height=int(args.cam_top_height),
            fps=int(args.camera_fps),
            exposure=args.cam_top_exposure,
            fallback_fps=fallback_fps,
        )
        self.wrist_cam, self.wrist_camera_fps = self._create_camera_with_fallback(
            "wrist",
            args.cam_wrist,
            width=int(args.cam_wrist_width),
            height=int(args.cam_wrist_height),
            fps=int(args.camera_fps),
            exposure=args.cam_wrist_exposure,
            fallback_fps=fallback_fps,
        )
        for name, camera in (("front", self.front_cam), ("wrist", self.wrist_cam)):
            wait = getattr(camera, "wait_for_valid_frame", None)
            if callable(wait) and not wait(timeout=float(args.camera_warmup_s)):
                print(f"[warn] {name} camera did not report a valid fresh frame during warmup", flush=True)
        self._startup_profile("cameras")

        self.online_buffer = ReplayBuffer(int(args.replay_capacity))
        self.intervention_buffer = ReplayBuffer(int(args.replay_capacity))
        self._startup_profile("buffers empty")
        explicit_buffer_dir = resolve_path(args.load_buffer_dir) if args.load_buffer_dir is not None else None
        auto_buffer_dir = self.output_dir / "buffers"
        buffer_dir = explicit_buffer_dir
        if buffer_dir is None and args.auto_load_existing_buffer:
            online_path = auto_buffer_dir / "online_buffer_latest.npz"
            intervention_path = auto_buffer_dir / "intervention_buffer_latest.npz"
            if online_path.exists() and intervention_path.exists():
                buffer_dir = auto_buffer_dir
        if buffer_dir is not None:
            online_n = self.online_buffer.load_npz(buffer_dir / "online_buffer_latest.npz")
            intervention_n = self.intervention_buffer.load_npz(buffer_dir / "intervention_buffer_latest.npz")
            load_kind = "explicit" if explicit_buffer_dir is not None else "auto"
            print(f"[buffers] {load_kind}-loaded online={online_n} intervention={intervention_n} from {buffer_dir}", flush=True)
        self._startup_profile("buffers loaded")

        self.action_queue: list[WarmActionPacket] = []
        self.episode_active = False
        self.awaiting_reward = False
        self.saved_episodes = next_episode_index(self.output_dir)
        self.pending_seed: dict[str, Any] | None = None
        self.episode_transitions: list[HILTransition] = []
        self.current_samples: list[dict[str, Any]] = []
        self.inserted_transitions = len(self.online_buffer)
        self.intervention_transitions = len(self.intervention_buffer)
        self.gripper_toggle_value = self._sync_gripper_toggle_value()
        self.gripper_toggle_closed = self._gripper_value_is_closed(self.gripper_toggle_value)
        self.last_gripper_toggle_t = 0.0
        self.manual_gripper_override_until = 0.0
        self.last_log_t = 0.0
        self.last_buffer_save_t = 0.0

        self._write_metadata()
        mode = "EXECUTE" if self.execute else "DRY-RUN"
        print(
            f"[ready] RealMan Stage2 warmup collector ({mode})\n"
            f"  session={self.session_id}\n"
            f"  output={self.output_dir}\n"
            f"  policy={self.policy_path}\n"
            f"  stage1={self.stage1_checkpoint}\n"
            f"  gate={self.gate_checkpoint}\n"
            f"  stage2={self.stage2_checkpoint or 'disabled'} scale={args.stage2_residual_scale:.2f}\n"
            f"  task={args.task}\n"
            f"  fixed_home={fmt_vec(args.home_qpos)} reset_after_episode={args.reset_after_episode}\n"
            f"  execution_horizon={args.execution_horizon}\n"
            f"  freq={args.freq:.1f}Hz command_mode={args.command_mode} "
            f"quest={args.quest_control_hz:.1f}Hz yaw={args.quest_yaw_deg:.1f} radio={args.quest_canfd_radio} "
            f"camera_fps={self.front_camera_fps}/{self.wrist_camera_fps} "
            f"gate={self.gate_camera}/{self.gate_image_size} pos={self.gate_pos_t:.2f} neg={self.gate_neg_t:.2f}",
            flush=True,
        )
        print(
            "[buttons] 左Grip=开始episode | 右Grip按住=人工接管 | 右Trigger=夹爪开闭 | "
            "左Trigger=结束等待打分 | Y=成功保存 | X=失败保存 | A=丢弃",
            flush=True,
        )
        self.quest.start_motion_loop(
            state_reader=self._read_robot_state,
            command_sender=self._send_quest_pose,
            period_s=1.0 / max(float(args.quest_control_hz), 1e-6),
        )
        print(f"[quest] motion loop started at {args.quest_control_hz:.1f}Hz", flush=True)

    def _create_camera_with_fallback(
        self,
        name: str,
        device: str,
        *,
        width: int,
        height: int,
        fps: int,
        exposure: float | None,
        fallback_fps: list[int],
    ) -> tuple[Any, int]:
        tried: list[int] = []
        for candidate in [int(fps), *fallback_fps]:
            candidate = int(candidate)
            if candidate <= 0 or candidate in tried:
                continue
            tried.append(candidate)
            try:
                camera = create_inference_camera(device, width=width, height=height, fps=candidate, exposure=exposure)
                if candidate != int(fps):
                    print(
                        f"[camera] {name} {device}: requested {width}x{height}@{fps} failed, "
                        f"using {width}x{height}@{candidate}",
                        flush=True,
                    )
                return camera, candidate
            except Exception as exc:
                print(f"[camera] {name} {device}: {width}x{height}@{candidate} failed: {exc}", flush=True)
        raise RuntimeError(
            f"{name} camera is not active: {device}. Tried {width}x{height}@{tried}. "
            "For RealSense, check supported modes with `rs-enumerate-devices -c` "
            "and make sure no other process is using the camera."
        )

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
            print("[warn] policy feature mismatch: " + "; ".join(problems), flush=True)

    def _write_metadata(self) -> None:
        meta = {
            "kind": "realman_stage2_hil_warmup",
            "session_id": self.session_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "output_dir": str(self.output_dir),
            "policy_path": str(self.policy_path),
            "stage1_checkpoint": str(self.stage1_checkpoint),
            "gate_checkpoint": str(self.gate_checkpoint),
            "stage2_checkpoint": None if self.stage2_checkpoint is None else str(self.stage2_checkpoint),
            "task": str(self.args.task),
            "buffer_format": (
                "online_buffer stores all selected transitions; intervention_buffer stores transitions whose "
                "executed action came from right-Grip Quest override. ref_action/action are action chunks."
            ),
            "args": json_safe(vars(self.args)),
        }
        (self.output_dir / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        sessions_dir = self.output_dir / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)
        (sessions_dir / f"session_{self.session_id}.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _load_stage2_trainer(self, checkpoint: Path, device: str) -> HILSERLTrainer:
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing Stage2 checkpoint: {checkpoint}")
        raw = torch.load(checkpoint, map_location="cpu")
        cfg_raw = dict(raw.get("config", {}))
        if not cfg_raw:
            raise RuntimeError(f"Stage2 checkpoint has no config: {checkpoint}")
        cfg = HILSERLConfig(**cfg_raw)
        trainer = HILSERLTrainer(cfg, device)
        trainer.load(checkpoint)
        trainer.actor.eval()
        print(
            f"[stage2] loaded actor checkpoint: {checkpoint} "
            f"chunk={trainer.action_chunk_steps} train_dim={trainer.train_action_dim}",
            flush=True,
        )
        return trainer

    def close(self) -> None:
        self._save_buffers()
        try:
            self.quest.close()
        except Exception:
            pass
        for camera in (getattr(self, "front_cam", None), getattr(self, "wrist_cam", None)):
            if camera is not None:
                try:
                    camera.close()
                except Exception:
                    pass
        if getattr(self, "robot", None) is not None:
            if self.execute and self.args.stop_on_exit:
                self.robot.stop()
            self.robot.close()

    def _build_observation(self, state: Any, front_bgr: np.ndarray, wrist_bgr: np.ndarray) -> ObservationPacket:
        batch = {
            self.front_feature_key: inference_bgr_frame_to_policy_tensor(
                front_bgr,
                channel_order=self.args.policy_image_channel_order,
            ),
            self.wrist_feature_key: inference_bgr_frame_to_policy_tensor(
                wrist_bgr,
                channel_order=self.args.policy_image_channel_order,
            ),
            "observation.state": torch.from_numpy(state.as_qpos()).float(),
            "task": self.args.task,
        }
        return ObservationPacket(stamp=now_sec(), batch=batch, dt_map={})

    def _update_gate(self, front_bgr: np.ndarray, wrist_bgr: np.ndarray) -> None:
        x = compose_input(front_bgr, wrist_bgr, self.gate_camera, self.gate_image_size).to(self.gate_device)
        with torch.no_grad():
            prob = float(torch.sigmoid(self.gate_model(x).view(-1))[0].detach().cpu())
        self.gate_prob = prob
        prev = self.gate_phase
        if prob >= self.gate_pos_t:
            self.gate_pos_count += 1
            self.gate_neg_count = 0
        elif prob <= self.gate_neg_t:
            self.gate_neg_count += 1
            self.gate_pos_count = 0
        else:
            self.gate_pos_count = 0
            self.gate_neg_count = 0
        if self.gate_phase == 0 and self.gate_pos_count >= self.gate_hold_frames:
            self.gate_phase = 1
        elif self.gate_phase == 1 and self.gate_neg_count >= self.gate_hold_frames:
            self.gate_phase = 0
        if self.gate_phase != prev:
            print(f"\n[gate] phase={self.gate_phase} prob={prob:.3f}", flush=True)

    def _make_record_packet(self, packet: ObservationPacket, control_action: np.ndarray) -> WarmActionPacket:
        """Build RLT token/chunk for replay without driving the control action."""
        t0 = now_sec()
        token_actions = self.token_runner.infer_sequence(packet, max(1, int(self.args.execution_horizon)))
        z_rl = self.token_encoder.encode(packet, token_actions)
        chunk = self.token_encoder.action_chunk(token_actions)
        control = np.asarray(control_action, dtype=np.float32).reshape(-1)
        if chunk.shape[1] >= 7 and control.size >= 7:
            chunk[0, :7] = control[:7]
        gate_prob = float(self.gate_prob if self.gate_prob is not None else 0.0)
        return WarmActionPacket(
            stamp=packet.stamp,
            action=control.astype(np.float32, copy=True),
            ref_action_chunk=chunk,
            step_index=0,
            z_rl=z_rl,
            gate_prob=gate_prob,
            inference_s=now_sec() - t0,
        )

    def _apply_stage2_actor(
        self,
        *,
        obs: ObservationPacket,
        state_vec: np.ndarray,
        base_action: np.ndarray,
    ) -> tuple[np.ndarray, WarmActionPacket]:
        if self.stage2_trainer is None:
            raise RuntimeError("Stage2 actor is not loaded.")
        packet = self._make_record_packet(obs, base_action)
        ref_chunk = np.asarray(packet.ref_action_chunk, dtype=np.float32)
        stage2_chunk = self.stage2_trainer.act_chunk(packet.z_rl, state_vec, ref_chunk)
        scale = float(self.args.stage2_residual_scale)
        if scale != 1.0:
            stage2_chunk = ref_chunk + (np.asarray(stage2_chunk, dtype=np.float32) - ref_chunk) * scale
        action = np.asarray(stage2_chunk[0], dtype=np.float32).reshape(-1)
        if action.size >= 7:
            # Keep gripper semantics on the base/reference policy path.  Stage2
            # only trains joint residuals; gripper is not actor-controlled.
            action[6] = float(np.asarray(base_action, dtype=np.float32).reshape(-1)[6])
        return action, packet

    def _stage2_should_act(self) -> bool:
        return self.stage2_trainer is not None and self.gate_phase == 1

    def _gripper_state_name(self, value: float) -> str:
        return "闭合" if self._gripper_value_is_closed(value) else "打开"

    def _gripper_value_is_closed(self, value: float) -> bool:
        open_value = float(self.args.gripper_open_value)
        closed_value = float(self.args.gripper_closed_value)
        value = float(value)
        return abs(value - closed_value) <= abs(value - open_value)

    def _sync_gripper_toggle_value(self) -> float:
        fallback = float(np.clip(self.args.initial_gripper_value, 0.0, 1.0))
        value = fallback
        source = "initial"
        if self.execute:
            try:
                with self.robot.lock:
                    ret, state = self.robot.arm.rm_get_gripper_state()
                actpos = state.get("actpos", -1) if ret == 0 and isinstance(state, dict) else -1
                if GRIPPER_MIN_POSITION <= int(actpos) <= GRIPPER_MAX_POSITION:
                    value = float(np.clip(gripper_position_to_value(actpos), 0.0, 1.0))
                    source = "robot"
            except Exception as exc:
                print(f"[quest] 夹爪状态读取失败，使用 initial_gripper_value={fallback:.2f}: {exc}", flush=True)
        self.robot._last_gripper_cmd = value
        self.robot._cached_gripper_pos = value
        print(f"[quest] 夹爪toggle初始状态: {self._gripper_state_name(value)} value={value:.2f} source={source}", flush=True)
        return value

    def _queue_gripper_value(self, value: float, *, reason: str) -> bool:
        value = float(np.clip(value, 0.0, 1.0))
        if self.execute:
            self.robot._last_gripper_cmd = value
            self.robot._cached_gripper_pos = value
            with self.robot.gripper_lock:
                self.robot._pending_gripper_cmd = value
        self.gripper_toggle_value = value
        self.gripper_toggle_closed = self._gripper_value_is_closed(value)
        self.manual_gripper_override_until = now_sec() + max(0.0, float(self.args.manual_gripper_hold_s))
        print(f"\n[quest] 右Trigger夹爪{reason} -> {self._gripper_state_name(value)} value={value:.2f}", flush=True)
        return True

    def _handle_gripper_trigger(self) -> None:
        action = str(self.args.quest_trigger_action)
        open_value = float(self.args.gripper_open_value)
        closed_value = float(self.args.gripper_closed_value)
        if action == "open":
            target_value = open_value
            reason = "打开"
        elif action == "close":
            target_value = closed_value
            reason = "闭合"
        else:
            if self.gripper_toggle_closed:
                target_value = open_value
                reason = "切换打开"
            else:
                target_value = closed_value
                reason = "切换闭合"
        self._queue_gripper_value(target_value, reason=reason)

    def _handle_buttons(self) -> None:
        for event in self.quest.drain_events():
            if event == "left_grip":
                self._start_episode()
            elif event == "left_trigger":
                self._end_episode()
            elif event == "y":
                self._finalize_episode(reward=1.0)
            elif event == "x":
                self._finalize_episode(reward=0.0)
            elif event == "a":
                self._discard_episode()
            elif event == "right_trigger" and not self.args.quest_no_gripper:
                self._handle_gripper_trigger()

    def _start_episode(self) -> None:
        self.episode_active = True
        self.awaiting_reward = False
        self.pending_seed = None
        self.episode_transitions = []
        self.current_samples = []
        self.action_queue = []
        self.quest.pause_motion(block_until_release=True, print_result=True)
        try:
            self.control_runner.reset()
            self.token_runner.policy.reset()
        except Exception:
            pass
        print("\n[episode] start", flush=True)

    def _end_episode(self) -> None:
        if not self.episode_active and not self.current_samples and self.pending_seed is None:
            print("\n[episode] left trigger ignored: no active/staged episode", flush=True)
            return
        self.episode_active = False
        self.awaiting_reward = True
        self.action_queue = []
        self.quest.pause_motion(block_until_release=True, print_result=False)
        print(
            f"\n[episode] end: staged={len(self.episode_transitions)} pending_seed={self.pending_seed is not None}. "
            "按 Y=成功保存，X=失败保存，A=丢弃。",
            flush=True,
        )

    def _discard_episode(self) -> None:
        if not self.episode_active and not self.awaiting_reward and not self.current_samples and self.pending_seed is None:
            return
        removed = len(self.episode_transitions)
        self.episode_active = False
        self.awaiting_reward = False
        self.pending_seed = None
        self.episode_transitions = []
        self.current_samples = []
        self.action_queue = []
        print(f"\n[episode] discarded staged_transitions={removed}", flush=True)
        self._reset_after_episode(reason="discard")

    def _should_record(self, is_intervention: bool) -> bool:
        if not self.episode_active:
            return False
        mode = str(self.args.record_mode)
        if mode == "all":
            return True
        if mode == "intervention":
            return bool(is_intervention)
        return self.gate_phase == 1

    def _make_transition(self, current: dict[str, Any], nxt: dict[str, Any], *, reward: float, done: float) -> HILTransition:
        return HILTransition(
            z_rl=np.asarray(current["z_rl"], dtype=np.float32),
            state=np.asarray(current["state"], dtype=np.float32),
            ref_action=np.asarray(current["ref_action"], dtype=np.float32),
            action=np.asarray(current["action"], dtype=np.float32),
            reward=float(reward),
            next_z_rl=np.asarray(nxt["z_rl"], dtype=np.float32),
            next_state=np.asarray(nxt["state"], dtype=np.float32),
            next_ref_action=np.asarray(nxt["ref_action"], dtype=np.float32),
            done=float(done),
            is_intervention=float(current["is_intervention"]),
        )

    def _append_transition(self, transition: HILTransition, *, terminal: bool) -> None:
        self.episode_transitions.append(transition)
        sample = transition.to_npz_dict()
        sample["terminal"] = float(terminal)
        self.current_samples.append(sample)

    def _record_seed(
        self,
        *,
        stamp: float,
        z_rl: np.ndarray,
        state_vec: np.ndarray,
        ref_action_chunk: np.ndarray,
        executed_action_chunk: np.ndarray,
        is_intervention: bool,
        source: str,
    ) -> None:
        seed = {
            "timestamp": float(stamp),
            "z_rl": np.asarray(z_rl, dtype=np.float32),
            "state": np.asarray(state_vec, dtype=np.float32),
            "ref_action": np.asarray(ref_action_chunk, dtype=np.float32),
            "action": np.asarray(executed_action_chunk, dtype=np.float32),
            "is_intervention": float(is_intervention),
            "source": source,
            "gate_prob": float(self.gate_prob if self.gate_prob is not None else 0.0),
        }
        if self.pending_seed is not None:
            transition = self._make_transition(self.pending_seed, seed, reward=0.0, done=0.0)
            self._append_transition(transition, terminal=False)
        self.pending_seed = seed

    def _finalize_episode(self, *, reward: float) -> None:
        if not self.awaiting_reward and self.episode_active:
            print("\n[episode] 请先按左Trigger结束 episode，再按 Y/X 打分保存。", flush=True)
            return
        if self.pending_seed is not None:
            terminal = self._make_transition(self.pending_seed, self.pending_seed, reward=float(reward), done=1.0)
            self._append_transition(terminal, terminal=True)
            self.pending_seed = None
        if not self.episode_transitions:
            print("\n[episode] no transitions to save; discarded empty episode", flush=True)
            self._discard_episode()
            return

        for transition in self.episode_transitions:
            self.online_buffer.insert(transition)
            self.inserted_transitions += 1
            if transition.is_intervention > 0.5:
                self.intervention_buffer.insert(transition)
                self.intervention_transitions += 1

        ep = self.saved_episodes
        ep_dir = self.output_dir / f"episode_{ep:06d}"
        while ep_dir.exists():
            ep += 1
            ep_dir = self.output_dir / f"episode_{ep:06d}"
        self.saved_episodes = ep
        ep_dir.mkdir(parents=True, exist_ok=True)
        keys = [
            "z_rl",
            "state",
            "ref_action",
            "action",
            "reward",
            "next_z_rl",
            "next_state",
            "next_ref_action",
            "done",
            "is_intervention",
            "terminal",
        ]
        arrays = {key: np.asarray([sample[key] for sample in self.current_samples], dtype=np.float32) for key in keys}
        np.savez_compressed(ep_dir / "episode_transitions.npz", **arrays)
        meta = {
            "session_id": self.session_id,
            "episode": ep,
            "transitions": len(self.episode_transitions),
            "reward": float(reward),
            "online_buffer_size": len(self.online_buffer),
            "intervention_buffer_size": len(self.intervention_buffer),
            "inserted_transitions": int(self.inserted_transitions),
            "intervention_transitions": int(self.intervention_transitions),
        }
        (ep_dir / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        with (self.output_dir / "episodes.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(meta, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.saved_episodes = ep + 1
        self._save_buffers()

        print(
            f"\n[episode] saved episode_{ep:06d}: reward={reward:.0f} transitions={len(self.episode_transitions)} "
            f"online={len(self.online_buffer)} intervention={len(self.intervention_buffer)}",
            flush=True,
        )
        self.episode_active = False
        self.awaiting_reward = False
        self.episode_transitions = []
        self.current_samples = []
        self.action_queue = []
        self._reset_after_episode(reason=f"saved reward={reward:.0f}")

    def _reset_policy_and_quest_tracking(self) -> None:
        self.action_queue = []
        self.pending_seed = None
        self.quest.deactivate()
        try:
            self.control_runner.reset()
            self.token_runner.policy.reset()
        except Exception:
            pass

    def _reset_after_episode(self, *, reason: str) -> None:
        self._reset_policy_and_quest_tracking()
        if not self.args.reset_after_episode:
            return
        home = np.asarray(self.args.home_qpos, dtype=float).reshape(7)
        if not self.execute:
            print(f"[reset] dry-run skip fixed home reset ({reason}): {fmt_vec(home)}", flush=True)
            return

        print(f"[reset] {reason}: 打开夹爪 -> 抬高末端 -> 回固定 home", flush=True)
        try:
            ret = self.robot.arm.rm_set_gripper_position(gripper_value_to_position(float(home[6])), True, 5)
            self.robot._last_gripper_cmd = float(home[6])
            self.robot._cached_gripper_pos = float(home[6])
            print(f"[reset] gripper {'OK' if ret == 0 else f'FAIL ret={ret}'}", flush=True)
        except Exception as exc:
            print(f"[reset] gripper failed: {exc}", flush=True)

        time.sleep(0.2)
        try:
            state = self._read_robot_state()
            lift_pose = np.r_[state.tcp_pos, state.tcp_euler].astype(float)
            lift_pose[2] = min(float(self.args.reset_lift_z_max), float(lift_pose[2]) + float(self.args.reset_lift_distance))
            with self.robot.lock:
                ret = int(self.robot.arm.rm_movel(lift_pose.tolist(), int(self.args.reset_lift_speed), 0, 0, 1))
            print(f"[reset] lift {'OK' if ret == 0 else f'FAIL ret={ret}'}", flush=True)
        except Exception as exc:
            print(f"[reset] lift skipped/failed: {exc}", flush=True)

        try:
            self.robot.move_to_init(home.astype(np.float32))
            print("[reset] movej home OK", flush=True)
        except Exception as exc:
            print(f"[reset] movej home failed: {exc}", flush=True)

        time.sleep(max(0.0, float(self.args.home_settle_s)))

    def _save_buffers(self) -> None:
        if not hasattr(self, "online_buffer"):
            return
        self.online_buffer.save_npz(self.output_dir / "buffers" / "online_buffer_latest.npz")
        self.intervention_buffer.save_npz(self.output_dir / "buffers" / "intervention_buffer_latest.npz")
        self.last_buffer_save_t = now_sec()

    def _maybe_periodic_save(self) -> None:
        interval = float(self.args.snapshot_every_s)
        if interval <= 0:
            return
        if now_sec() - self.last_buffer_save_t >= interval:
            self._save_buffers()

    def _read_robot_state(self) -> DirectRobotState:
        with self.robot.lock:
            ret, payload = self.robot.arm.rm_get_current_arm_state()
        if ret != 0 or not isinstance(payload, dict):
            raise RuntimeError(f"rm_get_current_arm_state failed: ret={ret}")
        joint = np.asarray(payload["joint"][:6], dtype=np.float32)
        pose = np.asarray(payload["pose"][:6], dtype=np.float32)
        gripper = 0.5 if self.robot._last_gripper_cmd is None else float(self.robot._last_gripper_cmd)
        return DirectRobotState(
            joint_deg=joint,
            tcp_pos=pose[:3].copy(),
            tcp_euler=pose[3:6].copy(),
            gripper=float(gripper),
        )

    def _execute_base_action(self, action: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        action = np.asarray(action, dtype=float).reshape(-1)
        if action.size < 7:
            raise RuntimeError(f"Bad RealMan model action shape: {action.shape}; expected at least 7")
        qpos_cmd = action[:7].copy()
        qpos_cmd[6] = float(np.clip(qpos_cmd[6], 0.0, 1.0))
        manual_gripper_hold = now_sec() < float(self.manual_gripper_override_until)
        if manual_gripper_hold:
            qpos_cmd[6] = float(np.clip(self.gripper_toggle_value, 0.0, 1.0))
        command_info: dict[str, Any] = {
            "sent": False,
            "ret": 0,
            "raw_joint_target": qpos_cmd[:6],
            "joint_target": qpos_cmd[:6],
            "gripper_queued": False,
            "manual_gripper_hold": manual_gripper_hold,
        }
        if self.execute:
            command_info = self.robot.set_qpos(qpos_cmd, use_smoothing=True)
            command_info["manual_gripper_hold"] = manual_gripper_hold
        executed_step = np.r_[np.asarray(command_info.get("joint_target", qpos_cmd[:6]), dtype=float), qpos_cmd[6]]
        return executed_step.astype(np.float32), command_info

    def _send_quest_pose(self, target_pose: np.ndarray) -> int:
        target_pose = np.asarray(target_pose, dtype=float).reshape(6)
        if not self.execute:
            return 0
        with self.robot.lock:
            return int(
                self.robot.arm.rm_movep_canfd(
                    target_pose.astype(float).tolist(),
                    False,
                    0,
                    int(self.args.quest_canfd_radio),
                )
            )

    def _execute_quest_action(self, state: Any, target_pose: np.ndarray) -> tuple[np.ndarray, int]:
        target_pose = np.asarray(target_pose, dtype=float).reshape(6)
        ret = self._send_quest_pose(target_pose)
        q_target = np.asarray(state.joint_deg, dtype=float).reshape(6)
        gripper = 0.5 if self.robot._last_gripper_cmd is None else float(self.robot._last_gripper_cmd)
        return np.r_[q_target, gripper].astype(np.float32), ret

    def _log_step(
        self,
        *,
        source: str,
        state_vec: np.ndarray,
        executed_step: np.ndarray | None,
        command_info: dict[str, Any] | None = None,
        ret: int | None = None,
    ) -> None:
        if float(self.args.log_hz) <= 0.0:
            return
        now = now_sec()
        if now - self.last_log_t < 1.0 / max(float(self.args.log_hz), 1e-6):
            return
        self.last_log_t = now
        gate_prob = -1.0 if self.gate_prob is None else float(self.gate_prob)
        if not self.args.verbose_step_log:
            print(
                f"[step] src={source} ep={int(self.episode_active)} wait_reward={int(self.awaiting_reward)} "
                f"gate={self.gate_phase}:{gate_prob:.3f} staged={len(self.episode_transitions)} "
                f"buf={len(self.online_buffer)}/{len(self.intervention_buffer)}",
                flush=True,
            )
            return
        executed_text = "none" if executed_step is None else fmt_vec(executed_step)
        ret_text = "" if ret is None else f" ret={ret}"
        sent_text = ""
        if command_info is not None:
            sent_text = f" sent={command_info.get('sent')} ret={command_info.get('ret')}"
        print(
            f"[step] src={source} ep={int(self.episode_active)} wait_reward={int(self.awaiting_reward)} "
            f"gate={self.gate_phase}:{gate_prob:.3f} staged={len(self.episode_transitions)} "
            f"buf={len(self.online_buffer)}/{len(self.intervention_buffer)} "
            f"q={fmt_vec(state_vec)} exec={executed_text}{sent_text}{ret_text}",
            flush=True,
        )

    def run(self) -> int:
        period = 1.0 / max(float(self.args.freq), 1e-6)
        next_t = now_sec()
        while True:
            loop_start = now_sec()
            self._handle_buttons()
            front_bgr = self.front_cam.get_frame()
            wrist_bgr = self.wrist_cam.get_frame()
            state = self._read_robot_state()
            state_vec = state.as_qpos().astype(np.float32)
            self._update_gate(front_bgr, wrist_bgr)

            is_intervention, quest_sample = self.quest.latest_intervention()

            executed_step: np.ndarray | None = None
            source = "idle"
            command_info: dict[str, Any] | None = None
            quest_ret: int | None = None
            obs: ObservationPacket | None = None
            control_action: np.ndarray | None = None
            record_packet: WarmActionPacket | None = None

            if self.episode_active:
                obs = self._build_observation(state, front_bgr, wrist_bgr)
                if is_intervention:
                    source = "quest_override"
                    if quest_sample is not None:
                        executed_step = np.asarray(quest_sample.state_qpos, dtype=np.float32)
                        quest_ret = int(quest_sample.ret)
                    if self._should_record(is_intervention):
                        base_packet = self.control_runner.infer(obs)
                        control_action = np.asarray(base_packet.action, dtype=np.float32).reshape(-1)
                else:
                    base_packet = self.control_runner.infer(obs)
                    base_action = np.asarray(base_packet.action, dtype=np.float32).reshape(-1)
                    control_action = base_action
                    source = "base_vla"
                    if self._stage2_should_act():
                        control_action, record_packet = self._apply_stage2_actor(
                            obs=obs,
                            state_vec=state_vec,
                            base_action=base_action,
                        )
                        source = "stage2_rlt"
                    executed_step, command_info = self._execute_base_action(control_action)
            elif is_intervention:
                source = "quest_manual"
                if quest_sample is not None:
                    executed_step = np.asarray(quest_sample.state_qpos, dtype=np.float32)
                    quest_ret = int(quest_sample.ret)

            if (
                obs is not None
                and control_action is not None
                and executed_step is not None
                and self._should_record(is_intervention)
            ):
                packet = record_packet if record_packet is not None else self._make_record_packet(obs, control_action)
                ref_chunk = np.asarray(packet.ref_action_chunk, dtype=np.float32).copy()
                executed_chunk = ref_chunk.copy()
                executed_chunk[0, :7] = executed_step[:7]
                self._record_seed(
                    stamp=loop_start,
                    z_rl=packet.z_rl,
                    state_vec=state_vec,
                    ref_action_chunk=ref_chunk,
                    executed_action_chunk=executed_chunk,
                    is_intervention=is_intervention,
                    source=source,
                )

            self._log_step(
                source=source,
                state_vec=state_vec,
                executed_step=executed_step,
                command_info=command_info,
                ret=quest_ret,
            )
            self._maybe_periodic_save()

            next_t += period
            sleep_s = next_t - now_sec()
            if sleep_s < -period:
                next_t = now_sec()
            elif sleep_s > 0:
                time.sleep(sleep_s)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect RealMan Stage2 HIL warmup replay buffers with frozen SmolVLA, "
            "RLT Stage1 encoder, RL gate, and Quest human override."
        )
    )
    parser.add_argument("--policy-path", type=Path, default=STAGE2_CFG.policy_path)
    parser.add_argument("--stage1-checkpoint", type=Path, default=STAGE2_CFG.stage1_checkpoint)
    parser.add_argument("--gate-checkpoint", type=Path, default=STAGE2_CFG.gate_checkpoint)
    parser.add_argument(
        "--stage2-checkpoint",
        type=Path,
        default=None,
        help="Optional trained Stage2 actor-critic checkpoint. When set, gate-active non-intervention steps use Stage2 residual actions.",
    )
    parser.add_argument(
        "--stage2-residual-scale",
        type=float,
        default=1.0,
        help="Multiply Stage2 residual relative to the reference action. Use <1.0 for conservative first deployment.",
    )
    parser.add_argument("--task", default=DEFAULT_USB_TASK)
    parser.add_argument("--output-root", type=Path, default=STAGE2_CFG.output_dir)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--load-buffer-dir", type=Path, default=None)
    parser.add_argument(
        "--auto-load-existing-buffer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When true, auto-load output_dir/buffers before appending new warmup episodes.",
    )

    parser.add_argument("--robot-ip", default=ROLLOUT_CFG.robot_ip)
    parser.add_argument("--robot-port", type=int, default=ROLLOUT_CFG.robot_port)
    parser.add_argument("--rm-api-path", default=ROLLOUT_CFG.rm_api_path)
    parser.add_argument("--cam-top", default="/dev/video6")
    parser.add_argument("--cam-wrist", default="344522071398")
    parser.add_argument("--cam-top-width", type=int, default=640)
    parser.add_argument("--cam-top-height", type=int, default=480)
    parser.add_argument("--cam-wrist-width", type=int, default=1280)
    parser.add_argument("--cam-wrist-height", type=int, default=720)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument(
        "--camera-fps-fallbacks",
        default="30,15,6",
        help="Comma-separated fallback FPS values if a RealSense mode cannot be opened.",
    )
    parser.add_argument("--cam-top-exposure", type=float, default=None)
    parser.add_argument("--cam-wrist-exposure", type=float, default=500)
    parser.add_argument("--camera-warmup-s", type=float, default=3.0)

    parser.add_argument("--front-feature-key", default=ROLLOUT_CFG.front_feature_key)
    parser.add_argument("--wrist-feature-key", default=ROLLOUT_CFG.wrist_feature_key)
    parser.add_argument("--policy-image-channel-order", choices=("rgb", "bgr"), default="rgb")
    parser.add_argument("--resize-to-policy-shape", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-dim-mismatch", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--freq", type=float, default=20.0)
    parser.add_argument("--execution-horizon", type=int, default=STAGE2_CFG.action_chunk_steps)
    parser.add_argument("--replan-every-step", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-state-age-s", type=float, default=0.25)
    parser.add_argument("--device", default=STAGE2_CFG.device)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--offline", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--gate-camera", choices=("auto", "front", "wrist", "both"), default="auto")
    parser.add_argument("--gate-image-size", type=int, default=None)
    parser.add_argument("--gate-positive-threshold", type=float, default=None)
    parser.add_argument("--gate-negative-threshold", type=float, default=None)
    parser.add_argument("--gate-hold-frames", type=int, default=3)
    parser.add_argument("--record-mode", choices=("gate", "intervention", "all"), default="gate")

    parser.add_argument("--command-mode", choices=("follow", "canfd"), default="canfd")
    parser.add_argument("--canfd-follow", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--canfd-trajectory-mode", type=int, default=2)
    parser.add_argument("--canfd-radio", type=int, default=80)
    parser.add_argument("--max-joint-step-deg", type=float, default=2.0)
    parser.add_argument("--joint-deadband-deg", type=float, default=0.05)
    parser.add_argument("--joint-ema-alpha", type=float, default=0.30)
    parser.add_argument("--gripper-deadband", type=float, default=0.02)
    parser.add_argument("--binary-gripper", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gripper-open-threshold", type=float, default=0.6)
    parser.add_argument("--gripper-close-threshold", type=float, default=0.4)
    parser.add_argument("--gripper-open-value", type=float, default=1.0)
    parser.add_argument("--gripper-closed-value", type=float, default=0.0)
    parser.add_argument("--initial-gripper-value", type=float, default=1.0)
    parser.add_argument(
        "--manual-gripper-hold-s",
        type=float,
        default=1.0,
        help="After a Quest gripper command, override base-policy gripper output for this many seconds.",
    )

    parser.add_argument("--quest-pos-scale", type=float, default=0.2)
    parser.add_argument("--quest-rot-scale", type=float, default=0.3)
    parser.add_argument("--quest-yaw-deg", type=float, default=45.0)
    parser.add_argument("--quest-control-hz", type=float, default=50.0)
    parser.add_argument(
        "--quest-canfd-radio",
        type=int,
        default=60,
        help="CANFD smoothing radio for Quest Cartesian override; collect_data.py uses 60.",
    )
    parser.add_argument("--quest-grip-threshold", type=float, default=0.8)
    parser.add_argument("--quest-button-threshold", type=float, default=0.8)
    parser.add_argument(
        "--quest-button-debounce-s",
        type=float,
        default=0.25,
        help="Quest button rising-edge debounce, matching collect_data.py default semantics.",
    )
    parser.add_argument("--quest-lock-orientation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quest-no-gripper", action="store_true")
    parser.add_argument(
        "--quest-trigger-action",
        choices=("toggle", "open", "close"),
        default="toggle",
        help="Right Trigger gripper semantics. For insert-only warmup, use open so one press always releases.",
    )
    parser.add_argument("--quest-debug", action="store_true")

    parser.add_argument("--move-home-on-start", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--home-qpos", type=parse_qpos, default=USB_INSERT_HOME_QPOS.copy())
    parser.add_argument("--home-speed", type=int, default=40)
    parser.add_argument("--home-settle-s", type=float, default=1.0)
    parser.add_argument("--reset-after-episode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reset-lift-distance", type=float, default=0.10)
    parser.add_argument("--reset-lift-z-max", type=float, default=0.60)
    parser.add_argument("--reset-lift-speed", type=int, default=30)
    parser.add_argument("--stop-on-exit", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--replay-capacity", type=int, default=200_000)
    parser.add_argument("--snapshot-every-s", type=float, default=30.0)
    parser.add_argument("--log-hz", type=float, default=0.0)
    parser.add_argument("--verbose-step-log", action="store_true")
    parser.add_argument("--startup-profile", action="store_true", help="Print RSS/GPU memory after each startup stage.")
    parser.add_argument("--execute", action="store_true", help="Actually command the RealMan arm.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HOME", str((REPO_ROOT / ".cache" / "huggingface").resolve()))
    os.environ.setdefault("HF_DATASETS_CACHE", str((REPO_ROOT / ".cache" / "huggingface" / "datasets").resolve()))
    if args.offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    collector: RealManWarmupCollector | None = None
    try:
        collector = RealManWarmupCollector(args)
        return collector.run()
    except KeyboardInterrupt:
        print("\n[exit] KeyboardInterrupt", flush=True)
        return 130
    finally:
        if collector is not None:
            collector.close()


if __name__ == "__main__":
    raise SystemExit(main())
