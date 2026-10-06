"""Pull the latest JPEG on demand instead of consuming a continuous video queue."""
import http.client
from urllib.parse import urlsplit

import cv2
import numpy as np

from ..config import CameraConfig


class SnapshotCapture:
    """Capture-compatible adapter. Only the capture worker owns this connection."""

    MAX_BYTES = 8 * 1024 * 1024

    def __init__(self, config: CameraConfig) -> None:
        parts = urlsplit(str(config.source))
        if parts.username or parts.password:
            raise ValueError("Snapshot URL credentials are not supported; use the existing video transport.")
        connection_type = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
        self._connection = connection_type(parts.hostname, parts.port,
                                           timeout=config.read_timeout_ms / 1000)
        self._path = (parts.path or "/") + ("?" + parts.query if parts.query else "")
        self._closed = False

    def isOpened(self) -> bool:
        return not self._closed

    def read(self):
        self._connection.request("GET", self._path,
                                 headers={"Cache-Control": "no-cache", "Accept": "image/jpeg"})
        response = self._connection.getresponse()
        try:
            if response.status != 200 or response.getheader("Content-Type", "").split(";")[0].lower() != "image/jpeg":
                raise RuntimeError("snapshot_not_jpeg")
            body = response.read(self.MAX_BYTES + 1)
            if len(body) > self.MAX_BYTES:
                raise RuntimeError("snapshot_too_large")
        finally:
            response.close()
        frame = cv2.imdecode(np.frombuffer(body, dtype=np.uint8), cv2.IMREAD_COLOR) if body else None
        return frame is not None, frame

    def release(self) -> None:
        self._closed = True
        self._connection.close()
