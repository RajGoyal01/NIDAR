"""Validated camera configuration: CLI overrides environment, then defaults."""
from dataclasses import dataclass
from pathlib import Path
import argparse
import math
import os
from urllib.parse import urlsplit


def parse_source(value: str) -> int | str:
    value = value.strip()
    if not value:
        raise ValueError("Camera source cannot be empty.")
    if value.lstrip("-").isdigit():
        index = int(value)
        if index < 0:
            raise ValueError("Camera index must be non-negative.")
        return index
    if "://" in value:
        try:
            parts = urlsplit(value)
            valid = parts.scheme in {"http", "https", "rtsp"} and parts.hostname and parts.port != 0
        except ValueError:
            valid = False
        if not valid:
            raise ValueError("Use a valid HTTP(S)/RTSP video URL, webcam index, or local video file.")
        return value
    if Path(value).is_file():
        return str(Path(value).resolve())
    raise ValueError("Source is neither a supported stream URL nor an existing video file.")


@dataclass(frozen=True)
class CameraConfig:
    source: int | str = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    reconnect_delay: float = 2.0
    stale_after: float = 1.0
    open_timeout_ms: int = 3000
    read_timeout_ms: int = 2000
    backend: str = "auto"
    transport: str = "opencv"
    decoder_threads: int = 0
    rotation: int = 0

    def __post_init__(self) -> None:
        if type(self.rotation) is not int or self.rotation not in (0, 90, 180, 270):
            raise ValueError("Camera rotation must be 0, 90, 180 or 270 degrees clockwise.")
        for value in (self.width, self.height, self.fps, self.open_timeout_ms, self.read_timeout_ms,
                      self.reconnect_delay, self.stale_after):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Dimensions, FPS, delays and timeouts must be finite and positive.")
        if self.backend not in {"auto", "dshow", "msmf"}:
            raise ValueError("Unsupported webcam backend.")
        if self.transport not in {"opencv", "snapshot", "mjpeg"}:
            raise ValueError("Unsupported camera transport.")
        if not 0 <= self.decoder_threads <= 32:
            raise ValueError("Decoder threads must be between 0 (automatic) and 32.")
        if self.transport in {"snapshot", "mjpeg"} and (not isinstance(self.source, str) or
                urlsplit(self.source).scheme not in {"http", "https"}):
            raise ValueError("HTTP image transport requires an HTTP(S) endpoint.")

    @property
    def network(self) -> bool:
        return isinstance(self.source, str) and "://" in self.source

    @property
    def file(self) -> bool:
        return isinstance(self.source, str) and not self.network


def arguments(argv: list[str] | None = None) -> tuple[CameraConfig, argparse.Namespace]:
    parser = argparse.ArgumentParser(description="NIDAR camera, person detection, tracking and temporal verification.")
    from .detector import DEFAULT_MODEL, DetectorConfig
    from .hybrid_detector import DEFAULT_FUSION_CONFIG, DEFAULT_SPECIALIST_MODEL, FusionPolicy
    from .tracking import DEFAULT_TRACKER_CONFIG
    from .verification import DEFAULT_VERIFICATION_CONFIG
    from .reid import DEFAULT_REID_CONFIG
    from .setup_reid import MODEL as REID_MODEL
    from .survivors import DEFAULT_MANAGER_CONFIG
    parser.add_argument("--dashboard", action="store_true", help="Phase 7 local read-only web dashboard; implies --manage.")
    parser.add_argument("--dashboard-port", type=int, default=8765)
    parser.add_argument("--dashboard-fps", type=int, default=30, help="JPEG preview ceiling, not guaranteed inference FPS (1-60).")
    parser.add_argument("--cpu-threads", type=int, default=2, help="Bound PyTorch/OpenCV CPU workers (1-16).")
    parser.add_argument("--manage", action="store_true", help="Phase 6 persistent mission records and estimated unique count; implies --reid.")
    parser.add_argument("--mission-db", type=Path, help="New local SQLite mission path (default: unique runs/missions folder).")
    parser.add_argument("--resume-mission", action="store_true", help="Explicitly resume the supplied compatible mission database.")
    parser.add_argument("--manager-config", type=Path, default=DEFAULT_MANAGER_CONFIG)
    parser.add_argument("--reid", action="store_true", help="Phase 5 selective appearance references; implies --verify.")
    parser.add_argument("--reid-config", type=Path, default=DEFAULT_REID_CONFIG)
    parser.add_argument("--reid-model", type=Path, default=REID_MODEL)
    parser.add_argument("--verify", action="store_true", help="Phase 4 temporal evidence; implies --track and --detect.")
    parser.add_argument("--verification-config", type=Path, default=DEFAULT_VERIFICATION_CONFIG)
    parser.add_argument("--track", action="store_true", help="Phase 3 BoT-SORT temporary IDs; implies --detect.")
    parser.add_argument("--tracker-config", type=Path, default=DEFAULT_TRACKER_CONFIG)
    parser.add_argument("--track-log", type=Path, help="Optional new JSONL file of per-frame boxes/IDs, no images.")
    parser.add_argument("--detect", action="store_true", help="Enable person detection; no survivor counting.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--hybrid-detector", action="store_true",
                        help="Fuse the general-person model with the validated posture/small-person specialist.")
    parser.add_argument("--specialist-model", type=Path, default=DEFAULT_SPECIALIST_MODEL)
    parser.add_argument("--fusion-config", type=Path, default=DEFAULT_FUSION_CONFIG)
    parser.add_argument("--imgsz", type=int, choices=(512, 640), default=640)
    parser.add_argument("--confidence", type=float, default=None)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", choices=("auto", "cpu", "0"), default="auto")
    parser.add_argument("--fp32", action="store_true", help="Disable CUDA FP16 for comparison/debugging.")
    parser.add_argument("--source", default=os.getenv("NIDAR_CAMERA_SOURCE", "0"))
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--rotation", type=int, choices=(0, 90, 180, 270), default=0,
                        help="Clockwise camera correction before detection, tracking and display.")
    parser.add_argument("--display-fps", type=int, default=60,
                        help="Preview refresh ceiling, not phone capture FPS (1-240).")
    parser.add_argument("--reconnect-delay", type=float, default=2)
    parser.add_argument("--stale-after", type=float, default=1)
    parser.add_argument("--open-timeout-ms", type=int, default=3000)
    parser.add_argument("--read-timeout-ms", type=int, default=2000)
    parser.add_argument("--backend", choices=("auto", "dshow", "msmf"), default="auto")
    parser.add_argument("--transport", choices=("opencv", "snapshot", "mjpeg"), default="opencv",
                        help="Snapshot fetches one latest JPEG at a time; use the phone's /shot.jpg URL.")
    parser.add_argument("--decoder-threads", type=int, default=0,
                        help="FFmpeg decoder threads; 1 for live latency, 0 for automatic.")
    parser.add_argument("--headless", action="store_true", help="Console metrics without a window.")
    parser.add_argument("--seconds", type=float, default=0, help="Auto-stop after N seconds; 0 runs until quit.")
    args = parser.parse_args(argv)
    args.manage = args.manage or args.dashboard
    args.reid = args.reid or args.manage
    args.verify = args.verify or args.reid
    args.track = args.track or args.verify
    args.detect = args.detect or args.track
    if args.confidence is None:
        args.confidence = .1 if args.track else .35
    try:
        if not 1 <= args.dashboard_fps <= 60 or not 1 <= args.cpu_threads <= 16:
            raise ValueError("Dashboard FPS must be 1-60 and CPU threads 1-16.")
        if not 1024 <= args.dashboard_port <= 65535:
            raise ValueError("--dashboard-port must be between 1024 and 65535.")
        if (args.mission_db or args.resume_mission) and not args.manage:
            raise ValueError("Mission options require --manage.")
        if args.resume_mission and not args.mission_db:
            raise ValueError("--resume-mission requires an explicit --mission-db.")
        if args.track_log and not args.track:
            raise ValueError("--track-log requires --track.")
        DetectorConfig(args.model, args.imgsz, args.confidence, args.iou, args.device, args.fp32)
        if args.hybrid_detector:
            if not args.specialist_model.is_file():
                raise ValueError(f"Hybrid specialist model is missing: {args.specialist_model}")
            FusionPolicy.load(args.fusion_config)
        if not math.isfinite(args.seconds) or args.seconds < 0:
            raise ValueError("--seconds must be finite and non-negative.")
        if not 1 <= args.display_fps <= 240:
            raise ValueError("--display-fps must be between 1 and 240.")
        config = CameraConfig(source=parse_source(args.source), width=args.width, height=args.height,
                              fps=args.fps, reconnect_delay=args.reconnect_delay,
                              stale_after=args.stale_after, open_timeout_ms=args.open_timeout_ms,
                              read_timeout_ms=args.read_timeout_ms, backend=args.backend,
                              transport=args.transport, decoder_threads=args.decoder_threads, rotation=args.rotation)
    except ValueError as exc:
        parser.error(str(exc))
    return config, args
