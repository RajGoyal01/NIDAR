"""Direct multipart JPEG: drain network separately and decode only latest JPEG."""
import re
import socket
import threading
import logging
import time

import cv2
import numpy as np

from .snapshot_capture import SnapshotCapture


class MultipartJpegs:
    """Incremental Content-Length parser; no unbounded byte/frame queue."""

    MAX_BYTES = 8 * 1024 * 1024
    MAX_HEADER = 16384

    def __init__(self):
        self.buffer = bytearray()
        self.expected = None

    def feed(self, chunk: bytes) -> bytes | None:
        self.buffer.extend(chunk)
        latest = None
        while True:
            if self.expected is None:
                end = self.buffer.find(b"\r\n\r\n")
                if end < 0:
                    if len(self.buffer) > self.MAX_HEADER:
                        raise ValueError("multipart_header_too_large")
                    break
                if end > self.MAX_HEADER:
                    raise ValueError("multipart_header_too_large")
                header = bytes(self.buffer[:end])
                match = re.search(br"(?im)^Content-Length:[ \t]*(\d+)[ \t]*\r?$", header)
                if not match or b"image/jpeg" not in header.lower():
                    raise ValueError("multipart_requires_jpeg_and_length")
                self.expected = int(match[1])
                if not 0 < self.expected <= self.MAX_BYTES:
                    raise ValueError("multipart_image_too_large")
                del self.buffer[:end + 4]
            if len(self.buffer) < self.expected:
                break
            latest = bytes(self.buffer[:self.expected])
            del self.buffer[:self.expected]
            self.expected = None
        return latest


class MjpegCapture(SnapshotCapture):
    def __init__(self, config):
        super().__init__(config)
        self._timeout = config.read_timeout_ms / 1000
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._latest = None
        self._sequence = self._consumed = 0
        self._failed = False
        self.invalid_frames = 0
        self._reader = threading.Thread(target=self._receive, name="mjpeg-receiver", daemon=True)
        self._reader.start()

    def _receive(self):
        try:
            self._connection.request("GET", self._path, headers={"Cache-Control": "no-cache"})
            response = self._connection.getresponse()
            if response.status != 200 or "multipart/x-mixed-replace" not in response.getheader("Content-Type", "").lower():
                raise ValueError("not_multipart_video")
            parser = MultipartJpegs()
            while not self._stop.is_set():
                chunk = response.read1(65536)
                if not chunk:
                    raise EOFError("camera_closed")
                jpeg = parser.feed(chunk)
                if jpeg is not None:
                    with self._condition:
                        self._latest = jpeg
                        self._sequence += 1
                        self._condition.notify()
        except Exception as exc:
            # Do not propagate URLs/credentials from network exceptions.
            if not self._stop.is_set():
                logging.getLogger(__name__).warning("mjpeg_receiver_failure=%s", type(exc).__name__)
        finally:
            self._connection.close()
            with self._condition:
                self._failed = True
                self._condition.notify_all()

    def read(self):
        # A malformed image is not necessarily a dead connection. Skip it, but
        # keep a fixed deadline so endless bad images cannot hide a real outage.
        deadline = time.perf_counter() + self._timeout
        while time.perf_counter() < deadline:
            with self._condition:
                ready = self._condition.wait_for(
                    lambda: self._sequence != self._consumed or self._failed or self._stop.is_set(),
                    timeout=max(0, deadline - time.perf_counter()))
                if not ready or self._failed or self._stop.is_set():
                    if not self._stop.is_set():
                        logging.getLogger(__name__).warning("mjpeg_read_failure=%s", "receiver_closed" if self._failed else "frame_wait_timeout")
                    return False, None
                jpeg = self._latest
                self._consumed = self._sequence
            try:
                frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            except cv2.error:
                frame = None
            if frame is not None and frame.size:
                return True, frame
            self.invalid_frames += 1
            if self.invalid_frames == 1 or self.invalid_frames % 100 == 0:
                logging.getLogger(__name__).warning("mjpeg_bad_frame_skipped; total=%s", self.invalid_frames)
        logging.getLogger(__name__).warning("mjpeg_read_failure=no_valid_frame_within_deadline")
        return False, None

    def release(self):
        self._closed = True
        self._stop.set()
        # Wake a pending socket read without closing/reusing its handle in this
        # thread. The receiver owns close(); shutdown only cancels its I/O.
        sock = self._connection.sock
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        with self._condition:
            self._condition.notify_all()
        self._reader.join(self._timeout + 1)
        if self._reader.is_alive():
            raise RuntimeError("mjpeg_reader_shutdown_timeout")
