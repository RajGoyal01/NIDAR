"""Short-term BoT-SORT association, deliberately without appearance Re-ID."""
from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path
from types import SimpleNamespace
import time
import cv2
import numpy as np
from .detector import DetectionResult

DEFAULT_TRACKER_CONFIG = Path(__file__).resolve().parent / "settings/botsort.json"


@dataclass(frozen=True)
class TrackerConfig:
    track_high_thresh: float = .35
    track_low_thresh: float = .1
    new_track_thresh: float = .4
    track_buffer: int = 30
    match_thresh: float = .8
    fuse_score: bool = True
    gmc_method: str = "sparseOptFlow"
    gmc_downscale: int = 2
    max_gap_seconds: float = 1.0

    def __post_init__(self) -> None:
        for value in (self.track_high_thresh, self.track_low_thresh, self.new_track_thresh, self.match_thresh):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError("Tracking thresholds must be finite numbers in (0, 1].")
        if not self.track_low_thresh < self.track_high_thresh <= self.new_track_thresh:
            raise ValueError("Require low threshold < high threshold <= new-track threshold.")
        if type(self.track_buffer) is not int or not 1 <= self.track_buffer <= 120:
            raise ValueError("Track buffer must be an integer from 1 to 120 processed frames.")
        if type(self.fuse_score) is not bool or self.gmc_method not in ("sparseOptFlow", "none"):
            raise ValueError("Use boolean fuse_score and sparseOptFlow or none camera compensation.")
        if type(self.gmc_downscale) is not int or self.gmc_downscale not in (2, 4, 8):
            raise ValueError("GMC downscale must be 2, 4 or 8.")
        if not isinstance(self.max_gap_seconds, (int, float)) or isinstance(self.max_gap_seconds, bool) or not math.isfinite(self.max_gap_seconds) or not 0 < self.max_gap_seconds <= 5:
            raise ValueError("Max tracking gap must be finite and in (0, 5] seconds.")

    @classmethod
    def load(cls, path: Path) -> "TrackerConfig":
        try:
            values = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(values, dict):
                raise ValueError
            return cls(**values)
        except (OSError, TypeError, ValueError) as exc:
            raise ValueError("Invalid tracker JSON: check keys, thresholds and file existence.") from exc


@dataclass(frozen=True)
class Track:
    track_id: str  # E<epoch>:T<temporary ID>, never a persistent survivor ID.
    xyxy: tuple[float, float, float, float]
    confidence: float


@dataclass(frozen=True)
class TrackingResult:
    tracks: tuple[Track, ...]
    tracking_ms: float
    events: tuple[dict, ...]


class PersonTracker:
    """One tracker per process; consumes only chronological, fresh detections."""
    def __init__(self, config: TrackerConfig):
        # Check explicitly BEFORE importing Ultralytics tracking, which otherwise
        # tries to install this optional dependency during application startup.
        import importlib.util
        if importlib.util.find_spec("lap") is None:
            raise ValueError("Tracking needs pinned lap==0.5.12; install project requirements.")
        from ultralytics.trackers.bot_sort import BOTSORT
        from ultralytics.engine.results import Boxes
        self._boxes_type = Boxes
        self.config = config
        self.engine = BOTSORT(SimpleNamespace(**asdict(config), with_reid=False,
                                             proximity_thresh=.5, appearance_thresh=.8, model="auto"))
        self.engine.gmc.downscale = config.gmc_downscale
        self.epoch = 1
        self.resets = 0
        self.started_tracks = 0  # Track episodes, NOT unique people.
        self._last_time = None
        self._last_sequence = None
        self._connection = None
        self._shape = None
        self._seen: set[int] = set()
        self._visible: set[int] = set()

    def clear(self, reason: str = "camera_unavailable") -> dict | None:
        if self._last_time is None:
            return None
        self.engine.reset()
        self.epoch += 1
        self.resets += 1
        self._last_time = self._last_sequence = self._connection = self._shape = None
        self._seen.clear()
        self._visible.clear()
        return {"event": "tracker_reset", "reason": reason, "epoch": self.epoch}

    def update(self, detection: DetectionResult, frame: np.ndarray, *, sequence: int,
               received_at: float, connection: int) -> TrackingResult:
        if not math.isfinite(received_at):
            raise ValueError("Tracking timestamp must be finite.")
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3 or not frame.size:
            raise ValueError("Tracker requires non-empty uint8 BGR input.")
        started = time.perf_counter()
        events = []
        reason = None
        if self._last_time is not None:
            if connection != self._connection:
                reason = "connection_changed"
            elif frame.shape != self._shape:
                reason = "resolution_changed"
            elif received_at - self._last_time > self.config.max_gap_seconds:
                reason = "long_frame_gap"
            elif sequence <= self._last_sequence or received_at < self._last_time:
                raise ValueError("Tracker input must advance chronologically; do not repeat a frame.")
        if reason:
            events.append(self.clear(reason))
        data = np.asarray([(*p.xyxy, p.confidence, 0) for p in detection.people], dtype=np.float32).reshape(-1, 6)
        if not np.isfinite(data).all():
            raise ValueError("Non-finite detection values cannot enter tracker.")
        boxes = self._boxes_type(data, frame.shape[:2])
        output = self.engine.update(boxes, frame)  # Also update on an empty detection set.
        tracks = []
        visible = set()
        for row in output:
            local_id = int(row[4])
            visible.add(local_id)
            label = f"E{self.epoch}:T{local_id}"
            tracks.append(Track(label, tuple(float(v) for v in row[:4]), float(row[5])))
            if local_id not in self._seen:
                self.started_tracks += 1
                events.append({"event": "track_started", "track_id": label})
            elif local_id not in self._visible:
                events.append({"event": "track_refound", "track_id": label})
        for local_id in sorted(self._visible - visible):
            events.append({"event": "track_lost", "track_id": f"E{self.epoch}:T{local_id}"})
        # Bound our own bookkeeping to current/lost tracks, not whole-session history.
        retained = {t.track_id for t in self.engine.tracked_stracks + self.engine.lost_stracks}
        self._seen = (self._seen | visible) & retained
        self._visible = visible
        self._last_time, self._last_sequence = received_at, sequence
        self._connection, self._shape = connection, frame.shape
        return TrackingResult(tuple(tracks), (time.perf_counter() - started) * 1000, tuple(events))


def annotate_tracks(frame: np.ndarray, result: TrackingResult) -> np.ndarray:
    image = frame.copy()
    h, w = image.shape[:2]
    for track in result.tracks:
        x1, y1, x2, y2 = track.xyxy
        x1, x2 = [max(0, min(w - 1, round(x))) for x in (x1, x2)]
        y1, y2 = [max(0, min(h - 1, round(y))) for y in (y1, y2)]
        number = int(track.track_id.split(":T")[1])
        color = (80 + number * 47 % 175, 100 + number * 67 % 155, 90 + number * 29 % 165)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f"{track.track_id} person {track.confidence:.2f}", (x1, max(18, y1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2, cv2.LINE_AA)
    return image
