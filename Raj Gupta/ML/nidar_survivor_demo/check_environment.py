"""Phase 0 checks only: no camera access, model download, or inference.

Run from the repository root with .venv/Scripts/python.exe.
Each check emits a JSON record. A failed check produces exit code 1.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Callable


def check_python() -> dict:
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("Use the project .venv interpreter, not system Python.")
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("This phase is validated with Python 3.11.")
    return {"version": platform.python_version(), "virtual_environment": True}


def check_packages() -> dict:
    packages = {"numpy": "numpy", "cv2": "opencv-python", "torch": "torch",
                "torchvision": "torchvision", "ultralytics": "ultralytics"}
    versions = {}
    for module, distribution in packages.items():
        importlib.import_module(module)
        versions[distribution] = importlib.metadata.version(distribution)
    for forbidden in ("opencv-python-headless", "opencv-contrib-python", "opencv-contrib-python-headless"):
        try:
            importlib.metadata.version(forbidden)
        except importlib.metadata.PackageNotFoundError:
            continue
        raise RuntimeError(f"Conflicting OpenCV distribution installed: {forbidden}")
    return versions


def check_pip() -> dict:
    result = subprocess.run([sys.executable, "-m", "pip", "check"],
                            capture_output=True, text=True, timeout=60)
    if result.returncode:
        raise RuntimeError(result.stdout.strip() or result.stderr.strip())
    return {"dependency_check": result.stdout.strip()}


def check_cuda() -> dict:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA unavailable; torch={torch.__version__}, runtime={torch.version.cuda}. "
                           "Install the pinned CUDA build into .venv. CPU fallback is not accepted.")
    name = torch.cuda.get_device_name(0)
    if "RTX 3050" not in name:
        raise RuntimeError(f"Expected RTX 3050, found {name}.")
    # A real operation catches failures that device enumeration alone misses.
    with torch.inference_mode():
        matrix = torch.ones((128, 128), device="cuda:0")
        result = matrix @ matrix
        torch.cuda.synchronize()
        if not torch.allclose(result, torch.full_like(result, 128)):
            raise RuntimeError("CUDA matrix result was incorrect.")
    return {"available": True, "gpu": name, "runtime": torch.version.cuda,
            "vram_mib": round(torch.cuda.get_device_properties(0).total_memory / 2**20),
            "matrix_multiplication": "PASS"}


def check_video() -> dict:
    import cv2
    import numpy as np
    build = cv2.getBuildInformation()
    gui = next((line.strip() for line in build.splitlines() if line.strip().startswith("GUI:")), "GUI: unknown")
    if "NONE" in gui or "unknown" in gui:
        raise RuntimeError("A GUI-enabled OpenCV build is required.")
    # Synthetic footage tests local video I/O without filming the user.
    with tempfile.TemporaryDirectory(prefix="nidar_phase0_") as folder:
        video = str(Path(folder) / "synthetic.avi")
        writer = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"MJPG"), 10, (160, 120))
        try:
            if not writer.isOpened():
                raise RuntimeError("OpenCV could not create the synthetic test video.")
            for index in range(10):
                writer.write(np.full((120, 160, 3), index * 20, dtype=np.uint8))
        finally:
            writer.release()
        capture = cv2.VideoCapture(video)
        count = 0
        try:
            if not capture.isOpened():
                raise RuntimeError("OpenCV could not open the local test video.")
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame.shape != (120, 160, 3):
                    raise RuntimeError("Decoded frame dimensions were incorrect.")
                if abs(float(frame.mean()) - count * 20) > 5:
                    raise RuntimeError("Decoded frame content/order was incorrect.")
                count += 1
        finally:
            capture.release()
        if count != 10:
            raise RuntimeError(f"Expected 10 decoded frames, received {count}.")
    return {"gui_build": gui, "synthetic_frames_decoded": count, "temporary_video_removed": True}


def check_gui() -> dict:
    import cv2
    import numpy as np
    try:
        cv2.imshow("NIDAR Phase 0 - Environment check", np.zeros((160, 480, 3), dtype=np.uint8))
        cv2.waitKey(500)
    finally:
        cv2.destroyAllWindows()
    return {"window_create_and_destroy": "PASS"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gui", action="store_true", help="Also open and close a brief blank test window.")
    args = parser.parse_args()
    checks: list[tuple[str, Callable[[], dict]]] = [
        ("python", check_python), ("packages", check_packages), ("pip", check_pip),
        ("cuda", check_cuda), ("video", check_video),
    ]
    if args.gui:
        checks.append(("gui", check_gui))
    failed = 0
    for name, operation in checks:
        try:
            details = operation()
            record = {"check": name, "status": "PASS", "details": details}
        except Exception as exc:
            failed += 1
            record = {"check": name, "status": "FAIL", "error": str(exc)}
        print(json.dumps(record), flush=True)
    print(json.dumps({"phase": 0, "status": "FAIL" if failed else "PASS", "failed_checks": failed}))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
