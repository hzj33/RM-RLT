#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check the Realman-RLT Python/runtime environment.")
    parser.add_argument("--rm-api-path", default=os.environ.get("RM_API2_PYTHON_PATH") or os.environ.get("RM_API_PATH"))
    parser.add_argument("--skip-hardware", action="store_true", help="Skip RealMan/Quest/RealSense/ROS hardware checks.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when any required check fails.")
    return parser.parse_args()


def version_of(module_name: str) -> str:
    try:
        import importlib.metadata as md

        package_name = {
            "cv2": "opencv-python",
            "PIL": "Pillow",
            "pyarrow": "pyarrow",
        }.get(module_name, module_name)
        return md.version(package_name)
    except Exception:
        module = sys.modules.get(module_name)
        return str(getattr(module, "__version__", "unknown"))


def check_import(module_name: str, *, required: bool = True, import_name: str | None = None) -> bool:
    name = import_name or module_name
    try:
        module = importlib.import_module(name)
        sys.modules[name] = module
        print(f"[OK] {module_name}: {version_of(name)}")
        return True
    except Exception as exc:
        level = "FAIL" if required else "WARN"
        print(f"[{level}] {module_name}: {exc}")
        return not required


def check_python() -> bool:
    version = sys.version_info
    ok = version.major == 3 and version.minor == 10
    print(f"[{'OK' if ok else 'WARN'}] python: {sys.version.split()[0]} (recommended: 3.10)")
    return True


def check_torch() -> bool:
    try:
        import torch

        print(f"[OK] torch: {torch.__version__}")
        print(f"[{'OK' if torch.cuda.is_available() else 'WARN'}] cuda available: {torch.cuda.is_available()}")
        print(f"[OK] torch cuda runtime: {torch.version.cuda}")
        if torch.cuda.is_available():
            print(f"[OK] gpu: {torch.cuda.get_device_name(0)}")
        return True
    except Exception as exc:
        print(f"[FAIL] torch: {exc}")
        return False


def check_opencv_gui() -> bool:
    try:
        import cv2

        info = cv2.getBuildInformation()
        gui_lines = [line.strip() for line in info.splitlines() if line.strip().startswith("GUI:")]
        gui_text = gui_lines[0] if gui_lines else "GUI: unknown"
        ok = "NONE" not in gui_text.upper()
        print(f"[{'OK' if ok else 'WARN'}] opencv gui: {gui_text}")
        return True
    except Exception as exc:
        print(f"[FAIL] opencv gui: {exc}")
        return False


def check_realman_sdk(path: str | None) -> bool:
    if path:
        sdk_path = Path(path).expanduser()
        if str(sdk_path) not in sys.path:
            sys.path.insert(0, str(sdk_path))
        print(f"[INFO] RealMan SDK path: {sdk_path}")
    try:
        from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e  # noqa: F401

        print("[OK] RealMan SDK: Robotic_Arm.rm_robot_interface import OK")
        return True
    except Exception as exc:
        print(f"[WARN] RealMan SDK: {exc}")
        return False


def check_command(name: str, args: list[str], *, required: bool = False) -> bool:
    try:
        result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=8, check=False)
        if result.returncode == 0:
            first = (result.stdout or result.stderr).splitlines()
            suffix = f" ({first[0]})" if first else ""
            print(f"[OK] {name}:{suffix}")
            return True
        print(f"[{'FAIL' if required else 'WARN'}] {name}: returncode={result.returncode}")
        return not required
    except Exception as exc:
        print(f"[{'FAIL' if required else 'WARN'}] {name}: {exc}")
        return not required


def check_gitignore() -> bool:
    paths = ["data/example.hdf5", "datasets/example.parquet", "outputs/model.pt", "logs/run.log", "__pycache__/x.pyc"]
    ok = True
    for rel in paths:
        result = subprocess.run(
            ["git", "check-ignore", "-q", rel],
            cwd=REPO_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        ignored = result.returncode == 0
        ok = ok and ignored
        print(f"[{'OK' if ignored else 'WARN'}] gitignore {rel}: {'ignored' if ignored else 'not ignored'}")
    return ok


def main() -> int:
    args = parse_args()
    failures = 0

    checks = [
        check_python(),
        check_torch(),
        check_import("torchvision"),
        check_import("lerobot"),
        check_import("transformers"),
        check_import("datasets"),
        check_import("numpy"),
        check_import("pandas"),
        check_import("pyarrow"),
        check_import("h5py"),
        check_import("Pillow", import_name="PIL"),
        check_import("cv2"),
        check_opencv_gui(),
        check_import("av"),
        check_import("tqdm"),
        check_import("scipy", required=False),
        check_import("einops", required=False),
        check_gitignore(),
    ]

    if not args.skip_hardware:
        checks.extend(
            [
                check_import("pyrealsense2", required=False),
                check_realman_sdk(args.rm_api_path),
                check_import("xrobotoolkit_sdk", required=False),
                check_import("rclpy", required=False),
                check_command("rs-enumerate-devices", ["rs-enumerate-devices", "--version"], required=False),
            ]
        )

    failures = sum(1 for ok in checks if not ok)
    if failures:
        print(f"[SUMMARY] completed with {failures} failed required checks.")
    else:
        print("[SUMMARY] environment checks passed.")
    return 1 if failures and args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())

