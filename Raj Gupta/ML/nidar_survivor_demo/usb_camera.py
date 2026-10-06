"""Discover and validate Windows webcam indices without starting the ML pipeline."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from typing import Callable

import cv2
import numpy as np

from .camera.phone_stream import open_capture
from .config import CameraConfig


@dataclass(frozen=True)
class CameraProbe:
    index: int
    available: bool
    width: int | None
    height: int | None
    fps: float | None
    backend: str
    reason: str


def probe_camera(index: int, *, width: int = 1280, height: int = 720, fps: int = 30,
                 backend: str = "dshow", factory: Callable = open_capture) -> CameraProbe:
    """Open one camera, require a decoded frame, and always release the device."""
    if type(index) is not int or not 0 <= index <= 20:
        raise ValueError("Camera index must be an integer from 0 to 20.")
    config = CameraConfig(source=index, width=width, height=height, fps=fps, backend=backend)
    capture = None
    try:
        capture = factory(config)
        if not capture.isOpened():
            return CameraProbe(index, False, None, None, None, backend, "open_failed")
        frame = None
        for _ in range(5):
            ok, candidate = capture.read()
            if (ok and isinstance(candidate, np.ndarray) and candidate.dtype == np.uint8
                    and candidate.ndim == 3 and candidate.shape[2] == 3 and candidate.size):
                frame = candidate
                break
        if frame is None:
            return CameraProbe(index, False, None, None, None, backend, "no_valid_frame")
        actual_fps = float(capture.get(cv2.CAP_PROP_FPS))
        return CameraProbe(index, True, int(frame.shape[1]), int(frame.shape[0]),
                           round(actual_fps, 2) if actual_fps > 0 else None,
                           backend, "ready")
    except Exception:
        return CameraProbe(index, False, None, None, None, backend, "backend_error")
    finally:
        if capture is not None:
            capture.release()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--index", type=int, help="Validate one OpenCV camera index.")
    target.add_argument("--list", action="store_true", help="Probe indices from zero through --max-index.")
    parser.add_argument("--max-index", type=int, default=5)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--backend", choices=("auto", "dshow", "msmf"), default="dshow")
    return parser


def run(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0 <= args.max_index <= 20:
        raise SystemExit("--max-index must be between 0 and 20")
    indices = range(args.max_index + 1) if args.list else (args.index,)
    results = [probe_camera(index, width=args.width, height=args.height, fps=args.fps,
                            backend=args.backend) for index in indices]
    available = [result for result in results if result.available]
    print(json.dumps({
        "event": "usb_camera_probe",
        "available_indices": [item.index for item in available],
        "cameras": [asdict(item) for item in results],
        "next_step": ("Run Start-USBCameraDemo.ps1 -CameraIndex <index>"
                      if available else "Connect/enable a webcam, check Windows camera permission, then probe again"),
    }, indent=2), flush=True)
    if args.list:
        return 0 if available else 2
    return 0 if results[0].available else 2


if __name__ == "__main__":
    raise SystemExit(run())
