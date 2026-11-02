from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RealManStage2Config:
    policy_path: Path = (
        REPO_ROOT
        / "outputs/train/realman_usb_insert_v1_smolvla_7d_fullft_20260715_154106/checkpoints/010000/pretrained_model"
    )
    stage1_checkpoint: Path = (
        REPO_ROOT
        / "outputs/rlt_stage1/rlt_stage1_realman_usb_insert_v1_smolvla_7d_fullft_20260715_154106_010000_20260715_181941/best.pt"
    )
    gate_checkpoint: Path = REPO_ROOT / "outputs/rlt_gate/rlt_gate_20260715_180731/best.pt"
    output_dir: Path = REPO_ROOT / "outputs/rlt_stage2_realman"
    rlt_checkpoint: Path | None = None
    rlt_buffer_dir: Path | None = REPO_ROOT / "outputs/rlt_stage2_realman/hil_warmup_realman_usb_insert_v1/buffers"

    device: str = "cuda"
    z_dim: int = 256
    state_dim: int = 7
    action_dim: int = 7
    train_action_dim: int = 6
    action_chunk_steps: int = 50

    # Joint-space residuals in degrees.  Keep this small for the first
    # RealMan run; the final command is still clipped by the RealMan controller.
    action_delta_scale_joint_deg: float = 0.5
    action_low: tuple[float, ...] = (-360.0, -360.0, -360.0, -360.0, -360.0, -360.0)
    action_high: tuple[float, ...] = (360.0, 360.0, 360.0, 360.0, 360.0, 360.0)

    warmup_steps: int = 64
    startup_updates: int = 300
    batch_size: int = 128
    replay_demo_ratio: float = 0.5
    policy_delay: int = 2
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 0.98
    tau: float = 0.005
    bc_weight: float = 0.2
    target_noise_joint_deg: float = 0.1
    target_noise_clip_joint_deg: float = 0.3
    actor_hidden_dim: int = 256
    critic_hidden_dim: int = 256
    fusion_mode: str = "direct"
    fusion_dim: int = 128
    log_interval: int = 50


CFG = RealManStage2Config()
