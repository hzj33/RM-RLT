#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.rlt_train.hil_serl_core import HILTransition, ReplayBuffer  # noqa: E402


BUFFER_NAMES = ("online_buffer_latest.npz", "intervention_buffer_latest.npz")


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def load_transitions(path: Path) -> list[HILTransition]:
    if not path.exists():
        return []
    with np.load(path) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
    if "reward" not in arrays:
        return []
    size = int(arrays["size"][0]) if "size" in arrays else int(len(arrays["reward"]))
    out: list[HILTransition] = []
    for i in range(size):
        out.append(
            HILTransition(
                z_rl=np.asarray(arrays["z_rl"][i], dtype=np.float32),
                state=np.asarray(arrays["state"][i], dtype=np.float32),
                ref_action=np.asarray(arrays["ref_action"][i], dtype=np.float32),
                action=np.asarray(arrays["action"][i], dtype=np.float32),
                reward=float(arrays["reward"][i]),
                next_z_rl=np.asarray(arrays["next_z_rl"][i], dtype=np.float32),
                next_state=np.asarray(arrays["next_state"][i], dtype=np.float32),
                next_ref_action=np.asarray(arrays["next_ref_action"][i], dtype=np.float32),
                done=float(arrays["done"][i]),
                is_intervention=float(arrays["is_intervention"][i]),
            )
        )
    return out


def summarize(items: list[HILTransition]) -> str:
    rewards = sum(float(x.reward) for x in items)
    terminals = sum(1 for x in items if float(x.done) > 0.5)
    interventions = sum(1 for x in items if float(x.is_intervention) > 0.5)
    return f"n={len(items)} reward_sum={rewards:.1f} terminals={terminals} interventions={interventions}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge RealMan Stage2 replay buffer directories.")
    parser.add_argument(
        "--input-buffer-dir",
        type=Path,
        action="append",
        required=True,
        help="Input directory containing online_buffer_latest.npz and intervention_buffer_latest.npz. Repeatable.",
    )
    parser.add_argument("--output-buffer-dir", type=Path, required=True)
    parser.add_argument("--capacity", type=int, default=1_000_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dirs = [resolve_path(path) for path in args.input_buffer_dir]
    output_dir = resolve_path(args.output_buffer_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    merged: dict[str, ReplayBuffer] = {
        name: ReplayBuffer(int(args.capacity))
        for name in BUFFER_NAMES
    }
    for input_dir in input_dirs:
        if not input_dir.exists():
            raise FileNotFoundError(f"Missing input buffer dir: {input_dir}")
        print(f"[input] {input_dir}")
        for name in BUFFER_NAMES:
            path = input_dir / name
            items = load_transitions(path)
            print(f"  {name}: {summarize(items)}")
            for item in items:
                merged[name].insert(item)

    for name, buffer in merged.items():
        out_path = output_dir / name
        buffer.save_npz(out_path)
        print(f"[output] {out_path}: n={len(buffer)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
