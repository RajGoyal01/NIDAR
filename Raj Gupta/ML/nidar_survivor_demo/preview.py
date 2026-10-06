"""Phase 1 camera health overlay; no perception or survivor counting."""
import cv2
import numpy as np
from .camera.phone_stream import Snapshot

TITLE = "NIDAR AIRMOUSE CAMERA TEST"

# Build the fixed background once. Broadcasting an RGB tuple over 921,600
# pixels on every refresh was avoidable work; a contiguous copy is cheaper.
_BACKGROUND = np.full((720, 1280, 3), (23, 19, 16), dtype=np.uint8)
_BACKGROUND.setflags(write=False)


def render(snapshot: Snapshot, displayed_fps: float, detection_status: str | None = None,
           *, source_label: str = "CAMERA", connection_label: str = "Reconnects") -> np.ndarray:
    canvas = _BACKGROUND.copy()
    if snapshot.packet is not None:
        frame = snapshot.packet.frame
        h, w = frame.shape[:2]
        scale = min(1280 / w, 580 / h)
        resized = cv2.resize(frame, (max(1, round(w * scale)), max(1, round(h * scale))))
        rh, rw = resized.shape[:2]
        top, left = 100 + (580 - rh) // 2, (1280 - rw) // 2
        canvas[top:top + rh, left:left + rw] = resized
    online = snapshot.state == "ONLINE"
    color = (135, 230, 65) if online else (80, 165, 255)
    health = "ONLINE" if online else f"OFFLINE / {snapshot.state}"
    lines = [(TITLE, (20, 32), (240, 240, 240), 0.8),
             (f"{source_label}: {health}   Capture: {snapshot.capture_fps:.1f} FPS   "
              f"Displayed: {displayed_fps:.1f} FPS", (20, 65), color, 0.65)]
    age = "--" if snapshot.age_ms is None else f"{snapshot.age_ms:.0f} ms"
    size = "--" if snapshot.packet is None else f"{snapshot.packet.frame.shape[1]}x{snapshot.packet.frame.shape[0]}"
    lines.append((f"Local frame age (NOT total delay): {age}   Size: {size}   "
                  f"{connection_label}: {max(0, snapshot.connections - 1)}", (20, 92), (185, 185, 185), 0.55))
    if snapshot.packet is None:
        lines.append(("Waiting for fresh video - check phone stream and local network",
                      (120, 370), color, 0.7))
    lines.append((detection_status or "Q / Esc: quit  |  Camera-to-screen delay: NOT MEASURED  |  Phase 1: camera only",
                  (20, 704), (185, 185, 185), 0.5))
    for message, position, shade, scale in lines:
        cv2.putText(canvas, message, position, cv2.FONT_HERSHEY_SIMPLEX, scale, shade, 1, cv2.LINE_AA)
    return canvas
