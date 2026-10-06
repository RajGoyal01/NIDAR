"""Single-owner capture thread and one latest-frame slot; no frame queue."""
from dataclasses import dataclass
import logging
import os
import threading
import time
from typing import Callable

# Native backend errors can include a URL with credentials. Use our sanitized
# status messages, not FFmpeg/OpenCV raw URL diagnostics in the normal app.
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
import cv2
import numpy as np

from ..config import CameraConfig
from ..utils.fps import FPSCounter

LOG = logging.getLogger(__name__)


def orient_frame(frame, rotation):
    """One shared coordinate system for perception and display; no crop/stretch."""
    if rotation == 0:
        return frame
    code = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE}[rotation]
    return cv2.rotate(frame, code)


@dataclass(frozen=True)
class FramePacket:
    frame: np.ndarray
    sequence: int
    received_at: float  # Monotonic timestamp AFTER decode; not sensor capture time.


@dataclass(frozen=True)
class Snapshot:
    state: str
    packet: FramePacket | None
    capture_fps: float
    age_ms: float | None
    frames: int
    connections: int
    attempts: int


def open_capture(config: CameraConfig):
    if config.transport == "mjpeg":
        from .mjpeg_capture import MjpegCapture
        return MjpegCapture(config)
    if config.transport == "snapshot":
        from .snapshot_capture import SnapshotCapture
        return SnapshotCapture(config)
    capture = cv2.VideoCapture()
    try:
        if isinstance(config.source, int):
            backend = {"auto": cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY,
                       "dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF}[config.backend]
            capture.open(config.source, backend)
            if capture.isOpened():
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
                capture.set(cv2.CAP_PROP_FPS, config.fps)
        else:
            capture.open(config.source, cv2.CAP_FFMPEG, [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, config.open_timeout_ms,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC, config.read_timeout_ms,
                cv2.CAP_PROP_N_THREADS, config.decoder_threads,
            ])
        if capture.isOpened():
            # Advisory: not every backend supports this. The application slot
            # remains bounded even when backend buffering cannot be controlled.
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture
    except Exception:
        capture.release()
        raise


class PhoneStream:
    def __init__(self, config: CameraConfig, factory: Callable = open_capture) -> None:
        self.config = config
        self._factory = factory
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._packet: FramePacket | None = None
        self._state = "STOPPED"
        self._fps = FPSCounter()
        self._frames = self._connections = self._attempts = 0

    def _state_to(self, state: str, clear: bool = False) -> None:
        with self._lock:
            changed = state != self._state
            self._state = state
            if clear:
                self._packet = None
                self._fps.reset()
        if changed:
            LOG.info("camera_state=%s", state)

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("Create a new PhoneStream for a new session.")
        self._state_to("CONNECTING")
        self._thread = threading.Thread(target=self._run, name="phone-capture", daemon=True)
        self._thread.start()

    def snapshot(self) -> Snapshot:
        now = time.monotonic()
        with self._lock:
            packet = self._packet
            age = (now - packet.received_at) if packet else None
            stale = age is not None and age > self.config.stale_after
            state = "STALE" if stale and self._state == "ONLINE" else self._state
            # Do not hand an expired picture to a future detector.
            return Snapshot(state, None if stale else packet,
                            self._fps.value(now) if state == "ONLINE" else 0.0,
                            age * 1000 if age is not None else None,
                            self._frames, self._connections, self._attempts)

    def stop(self, timeout: float | None = None) -> bool:
        self._stop.set()
        if self._thread:
            budget = timeout if timeout is not None else (
                max(self.config.open_timeout_ms, self.config.read_timeout_ms) / 1000 + 2)
            self._thread.join(budget)
            if self._thread.is_alive():
                self._state_to("SHUTDOWN_TIMEOUT", clear=True)
                LOG.error("Capture backend did not stop within budget; clean shutdown is unverified.")
                return False
        self._state_to("STOPPED", clear=True)
        return True

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                capture = None
                connected = False
                with self._lock:
                    self._attempts += 1
                try:
                    capture = self._factory(self.config)
                    if not capture.isOpened():
                        raise RuntimeError("open_failed")
                    file_rate = capture.get(cv2.CAP_PROP_FPS) if self.config.file else 0
                    interval = 1 / file_rate if 0 < file_rate <= 240 else 1 / self.config.fps
                    while not self._stop.is_set():
                        ok, frame = capture.read()
                        if not ok or frame is None or frame.size == 0 or frame.ndim != 3 or frame.shape[2] != 3:
                            raise RuntimeError("read_failed_or_invalid_frame")
                        if self._stop.is_set():
                            break
                        now = time.monotonic()
                        frame = orient_frame(frame, self.config.rotation)
                        with self._lock:
                            self._frames += 1
                            self._packet = FramePacket(frame, self._frames, now)
                            self._fps.tick(now)
                            if not connected:
                                self._connections += 1
                        if not connected:
                            self._state_to("ONLINE")
                            connected = True
                        # Local regression clips are paced. Never sleep here for
                        # live sources: drain them continuously to avoid backlog.
                        if self.config.file:
                            self._stop.wait(interval)
                except Exception as exc:
                    # Exception messages from native readers can contain secrets.
                    if not self._stop.is_set():
                        LOG.warning("camera_failure=%s; check source/network/permissions", type(exc).__name__)
                finally:
                    if capture is not None:
                        capture.release()  # Only capture thread touches native handle.
                if not self._stop.is_set():
                    self._state_to("RECONNECTING", clear=True)
                    self._stop.wait(self.config.reconnect_delay)
        finally:
            self._state_to("STOPPED", clear=True)
