#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
import threading
import time
from contextlib import suppress
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mujoco_env.hardware.realman_api import RealManConfig, RealManController


def _ms() -> float:
    return time.perf_counter() * 1000.0


def _summary(name: str, values: list[float]) -> None:
    if not values:
        print(f"{name}: no samples")
        return
    p95 = sorted(values)[int(0.95 * (len(values) - 1))]
    print(
        f"{name}: n={len(values)} min={min(values):.2f}ms "
        f"mean={statistics.mean(values):.2f}ms p95={p95:.2f}ms max={max(values):.2f}ms"
    )


def _timed(label: str, fn, n: int, sleep_s: float = 0.0):
    values = []
    rets = []
    last = None
    for _ in range(n):
        t0 = _ms()
        last = fn()
        values.append(_ms() - t0)
        if isinstance(last, tuple):
            rets.append(last[0])
        elif isinstance(last, dict) and "ret" in last:
            rets.append(last["ret"])
        else:
            rets.append("ok")
        if sleep_s:
            time.sleep(sleep_s)
    _summary(label, values)
    print(f"{label}_rets: {sorted(set(map(str, rets)))}")
    return last


def run(args: argparse.Namespace) -> int:
    cfg = RealManConfig(
        robot_ip=args.robot_ip,
        robot_port=args.robot_port,
        rm_api_path=args.rm_api_path,
        state_cache_hz=args.state_cache_hz,
        command_mode=args.command_mode,
        start_state_cache=False,
        start_gripper_worker=not args.skip_gripper,
    )
    robot = RealManController(cfg, auto_connect=True)
    try:
        print(f"robot_info: {robot.robot_info}")
        state = _timed("read_state_once", robot.read_state_once, args.samples)
        print("state_qpos:", [round(float(x), 6) for x in state.as_qpos()])
        print("state_tcp_pos:", [round(float(x), 6) for x in state.tcp_pos])
        print("state_tcp_euler:", [round(float(x), 6) for x in state.tcp_euler])

        q = state.joint_deg.copy()
        _timed("fk_euler", lambda: robot.fk_euler(q), args.samples)
        pose = robot.fk_euler(q)
        _timed("ik_current_pose", lambda: robot.ik_euler(pose, q), args.samples)

        if args.noop_commands:
            _timed(
                f"{args.command_mode}_noop_joint_command",
                lambda: robot.send_joint_target(q, mode=args.command_mode, limit_step=False),
                args.samples,
                sleep_s=max(0.0, 1.0 / max(args.command_hz, 1e-6)),
            )

        if not args.skip_gripper:
            print("Testing one queued gripper command. This may block the gripper worker for about 1s.")
            before = _ms()
            queued = robot.queue_gripper(robot.get_cached_gripper(), force=True)
            time.sleep(args.gripper_wait_s)
            print(f"gripper_queued: {queued}, waited_ms: {_ms() - before:.2f}")

        if args.cache_loop_s > 0:
            robot.start_state_cache()
            command_dts: list[float] = []
            periods: list[float] = []
            ages: list[float] = []
            deadline = time.monotonic() + args.cache_loop_s
            last_start = None
            while time.monotonic() < deadline:
                loop_start = _ms()
                if last_start is not None:
                    periods.append(loop_start - last_start)
                last_start = loop_start
                t0 = _ms()
                robot.send_joint_target(q, mode=args.command_mode, limit_step=False)
                command_dts.append(_ms() - t0)
                with suppress(Exception):
                    ages.append(time.time() * 1000.0 - robot.get_state().stamp * 1000.0)
                elapsed = (_ms() - loop_start) / 1000.0
                time.sleep(max(0.0, 1.0 / max(args.command_hz, 1e-6) - elapsed))
            _summary("cached_loop_command_call", command_dts)
            _summary("cached_loop_actual_period", periods)
            _summary("cached_state_age", ages)

        if args.concurrent_gripper and not args.skip_gripper:
            state_dts: list[float] = []
            command_dts: list[float] = []
            stop = threading.Event()

            def read_loop() -> None:
                while not stop.is_set():
                    t0 = _ms()
                    robot.read_state_once()
                    state_dts.append(_ms() - t0)
                    time.sleep(0.02)

            thread = threading.Thread(target=read_loop, daemon=True)
            thread.start()
            robot.queue_gripper(robot.get_cached_gripper(), force=True)
            deadline = time.monotonic() + args.gripper_wait_s
            while time.monotonic() < deadline:
                t0 = _ms()
                robot.send_joint_target(q, mode=args.command_mode, limit_step=False)
                command_dts.append(_ms() - t0)
                time.sleep(1.0 / max(args.command_hz, 1e-6))
            stop.set()
            thread.join(timeout=1.0)
            _summary("concurrent_state_read", state_dts)
            _summary("concurrent_joint_command", command_dts)
    finally:
        robot.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="RealMan RM65 backend diagnostics.")
    parser.add_argument("--robot-ip", default=RealManConfig().robot_ip)
    parser.add_argument("--robot-port", type=int, default=RealManConfig().robot_port)
    parser.add_argument("--rm-api-path", default=RealManConfig().rm_api_path)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--state-cache-hz", type=float, default=20.0)
    parser.add_argument("--command-hz", type=float, default=20.0)
    parser.add_argument("--command-mode", choices=["follow", "canfd"], default="follow")
    parser.add_argument("--noop-commands", action="store_true", help="Send current joints as no-op commands.")
    parser.add_argument("--skip-gripper", action="store_true", help="Do not queue gripper commands.")
    parser.add_argument("--gripper-wait-s", type=float, default=1.3)
    parser.add_argument("--cache-loop-s", type=float, default=0.0)
    parser.add_argument("--concurrent-gripper", action="store_true")
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
