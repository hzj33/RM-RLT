#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.rlt_train.hil_serl_core import HILSERLConfig, HILSERLTrainer, ReplayBuffer  # noqa: E402
from scripts.rlt_train_realman.config import CFG  # noqa: E402


def resolve_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    path = path.expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def default_output_dir(root: Path) -> Path:
    root = resolve_path(root)
    assert root is not None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return root / f"hil_serl_stage2_realman_offline_{stamp}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline/startup updates for RealMan Stage2 HIL-SERL buffers. "
            "This does not connect to the robot; it only trains actor/critic "
            "from existing online/intervention replay buffers."
        )
    )
    parser.add_argument("--rlt-buffer-dir", type=Path, default=CFG.rlt_buffer_dir)
    parser.add_argument("--rlt-checkpoint", type=Path, default=CFG.rlt_checkpoint)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=CFG.output_dir)
    parser.add_argument("--device", default=CFG.device)

    parser.add_argument("--z-dim", type=int, default=CFG.z_dim)
    parser.add_argument("--state-dim", type=int, default=CFG.state_dim)
    parser.add_argument("--action-dim", type=int, default=CFG.action_dim)
    parser.add_argument("--train-action-dim", type=int, default=CFG.train_action_dim)
    parser.add_argument("--rlt-action-chunk-steps", type=int, default=CFG.action_chunk_steps)
    parser.add_argument("--rlt-action-delta-scale-joint-deg", type=float, default=CFG.action_delta_scale_joint_deg)
    parser.add_argument("--rlt-warmup-steps", type=int, default=CFG.warmup_steps)
    parser.add_argument("--rlt-startup-updates", type=int, default=CFG.startup_updates)
    parser.add_argument("--rlt-batch-size", type=int, default=CFG.batch_size)
    parser.add_argument("--rlt-replay-demo-ratio", type=float, default=CFG.replay_demo_ratio)
    parser.add_argument("--rlt-policy-delay", type=int, default=CFG.policy_delay)
    parser.add_argument("--rlt-actor-lr", type=float, default=CFG.actor_lr)
    parser.add_argument("--rlt-critic-lr", type=float, default=CFG.critic_lr)
    parser.add_argument("--rlt-gamma", type=float, default=CFG.gamma)
    parser.add_argument("--rlt-tau", type=float, default=CFG.tau)
    parser.add_argument("--rlt-bc-weight", type=float, default=CFG.bc_weight)
    parser.add_argument("--rlt-target-noise-joint-deg", type=float, default=CFG.target_noise_joint_deg)
    parser.add_argument("--rlt-target-noise-clip-joint-deg", type=float, default=CFG.target_noise_clip_joint_deg)
    parser.add_argument("--rlt-actor-hidden-dim", type=int, default=CFG.actor_hidden_dim)
    parser.add_argument("--rlt-critic-hidden-dim", type=int, default=CFG.critic_hidden_dim)
    parser.add_argument("--rlt-fusion-mode", choices=("direct", "projected"), default=CFG.fusion_mode)
    parser.add_argument("--rlt-fusion-dim", type=int, default=CFG.fusion_dim)
    parser.add_argument("--log-interval", type=int, default=CFG.log_interval)
    return parser.parse_args()


def make_trainer(args: argparse.Namespace) -> HILSERLTrainer:
    scale = tuple([float(args.rlt_action_delta_scale_joint_deg)] * int(args.train_action_dim))
    cfg = HILSERLConfig(
        z_dim=int(args.z_dim),
        state_dim=int(args.state_dim),
        action_dim=int(args.action_dim),
        train_action_dim=int(args.train_action_dim),
        action_chunk_steps=int(args.rlt_action_chunk_steps),
        actor_hidden_dim=int(args.rlt_actor_hidden_dim),
        critic_hidden_dim=int(args.rlt_critic_hidden_dim),
        fusion_mode=str(args.rlt_fusion_mode),
        fusion_dim=int(args.rlt_fusion_dim),
        action_delta_scale=scale,
        actor_lr=float(args.rlt_actor_lr),
        critic_lr=float(args.rlt_critic_lr),
        gamma=float(args.rlt_gamma),
        tau=float(args.rlt_tau),
        batch_size=int(args.rlt_batch_size),
        replay_demo_ratio=float(args.rlt_replay_demo_ratio),
        bc_weight=float(args.rlt_bc_weight),
        target_noise=float(args.rlt_target_noise_joint_deg),
        target_noise_clip=float(args.rlt_target_noise_clip_joint_deg),
        train_after=int(args.rlt_warmup_steps),
        policy_delay=int(args.rlt_policy_delay),
        action_low=CFG.action_low[: int(args.train_action_dim)],
        action_high=CFG.action_high[: int(args.train_action_dim)],
    )
    return HILSERLTrainer(cfg, args.device)


def main() -> int:
    args = parse_args()
    buffer_dir = resolve_path(args.rlt_buffer_dir)
    if buffer_dir is None:
        raise SystemExit("--rlt-buffer-dir is required for offline Stage2 training.")
    online_path = buffer_dir / "online_buffer_latest.npz"
    intervention_path = buffer_dir / "intervention_buffer_latest.npz"
    if not online_path.exists():
        raise FileNotFoundError(f"Missing online replay buffer: {online_path}")
    if not intervention_path.exists():
        raise FileNotFoundError(f"Missing intervention replay buffer: {intervention_path}")

    output_dir = resolve_path(args.output_dir) if args.output_dir is not None else default_output_dir(args.output_root)
    assert output_dir is not None
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer = make_trainer(args)
    ckpt = resolve_path(args.rlt_checkpoint)
    if ckpt is not None:
        trainer.load(ckpt)
        print(f"[stage2-realman] loaded checkpoint: {ckpt}")

    online = ReplayBuffer(capacity=1_000_000)
    intervention = ReplayBuffer(capacity=1_000_000)
    online_n = online.load_npz(online_path)
    intervention_n = intervention.load_npz(intervention_path)
    print(
        f"[stage2-realman] buffers online={online_n} intervention={intervention_n} "
        f"warmup={args.rlt_warmup_steps} output={output_dir}"
    )

    metadata = {
        "kind": "realman_stage2_offline_startup",
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "hil_serl_config": asdict(trainer.cfg),
        "buffer_dir": str(buffer_dir),
        "online_buffer_size": online_n,
        "intervention_buffer_size": intervention_n,
    }
    (output_dir / "stage2_realman_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    updates = max(0, int(args.rlt_startup_updates))
    if online_n < int(args.rlt_warmup_steps):
        print(
            f"[stage2-realman] skip updates: online_buffer={online_n} < warmup={args.rlt_warmup_steps}"
        )
        updates = 0

    start_t = time.perf_counter()
    last_info = None
    last_actor_info = None
    for idx in range(1, updates + 1):
        info = trainer.update(online, intervention)
        if info is not None:
            last_info = info
            if info.get("actor_updated", 0.0) > 0.5:
                last_actor_info = info
        if idx == 1 or idx == updates or (args.log_interval > 0 and idx % int(args.log_interval) == 0):
            actor_info = last_actor_info or last_info or {}
            train_info = last_info or {}
            elapsed = max(time.perf_counter() - start_t, 1e-9)
            print(
                f"[stage2-realman] update {idx}/{updates} "
                f"{idx / elapsed:.1f} upd/s "
                f"critic={train_info.get('critic_loss', 0.0):.4f} "
                f"actor={actor_info.get('actor_loss', 0.0):.4f} "
                f"bc={actor_info.get('bc_loss', 0.0):.4f} "
                f"q={actor_info.get('actor_q_loss', 0.0):.4f}"
            )

    ckpt_dir = output_dir / "checkpoints"
    trainer.save(
        ckpt_dir / "stage2_realman_offline.pt",
        metadata={
            "online_buffer_size": online_n,
            "intervention_buffer_size": intervention_n,
            "startup_updates": updates,
        },
    )
    trainer.save(ckpt_dir / "last.pt")
    print(f"[done] checkpoint={ckpt_dir / 'stage2_realman_offline.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
