"""Bounded, rolling evidence for temporary tracks; never persistent identity."""
from collections import deque
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
import cv2
import numpy as np
from .tracking import Track, TrackingResult

DEFAULT_VERIFICATION_CONFIG = Path(__file__).resolve().parent / "settings/verification.json"


@dataclass(frozen=True)
class VerificationConfig:
    window: int = 5
    required: int = 3
    min_confidence: float = .35
    min_visible_area_ratio: float = .0001
    max_gap_seconds: float = 1.0

    def __post_init__(self) -> None:
        if type(self.window) is not int or not 1 <= self.window <= 60:
            raise ValueError("Verification window must be an integer from 1 to 60.")
        if type(self.required) is not int or not 1 <= self.required <= self.window:
            raise ValueError("Required evidence must be between 1 and window size.")
        for value in (self.min_confidence, self.min_visible_area_ratio, self.max_gap_seconds):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("Verification thresholds must be finite positive numbers.")
        if self.min_confidence > 1 or self.min_visible_area_ratio > 1 or self.max_gap_seconds > 5:
            raise ValueError("Confidence/area must be <= 1 and gap <= 5 seconds.")

    @classmethod
    def load(cls, path: Path) -> "VerificationConfig":
        try:
            return cls(**json.loads(path.read_text(encoding="utf-8")))
        except (OSError, TypeError, ValueError) as exc:
            raise ValueError("Invalid verification JSON: check keys, values and file existence.") from exc


@dataclass(frozen=True)
class VerifiedTrack:
    track: Track
    state: str
    hits: int
    eligible: bool


@dataclass(frozen=True)
class VerificationResult:
    tracks: tuple[VerifiedTrack, ...]
    events: tuple[dict, ...]
    verification_ms: float

    @property
    def confirmed_count(self) -> int:
        return sum(t.state == "CONFIRMED" for t in self.tracks)


@dataclass
class _Evidence:
    history: deque
    first_seen: float
    missing: int = 0
    state: str = "VERIFYING"


class TemporalVerifier:
    """Each update is one processed fresh frame, not one preview redraw."""
    def __init__(self, config: VerificationConfig):
        self.config = config
        self._evidence: dict[str, _Evidence] = {}
        self._last_sequence = self._last_time = self._epoch = None

    def clear(self, reason: str = "camera_unavailable") -> tuple[dict, ...]:
        active = self._last_sequence is not None
        self._evidence.clear()
        self._last_sequence = self._last_time = self._epoch = None
        return ({"event": "verification_reset", "reason": reason},) if active else ()

    def update(self, result: TrackingResult, *, sequence: int, received_at: float,
               epoch: int, frame_shape: tuple[int, ...]) -> VerificationResult:
        started = time.perf_counter()
        if type(sequence) is not int or sequence < 1 or not math.isfinite(received_at):
            raise ValueError("Verification needs a positive sequence and finite timestamp.")
        h, w = frame_shape[:2]
        if h <= 0 or w <= 0:
            raise ValueError("Verification needs positive frame dimensions.")
        visible = {t.track_id: t for t in result.tracks}
        if len(visible) != len(result.tracks):
            raise ValueError("Duplicate track IDs in a single frame.")
        for track in result.tracks:
            if not all(math.isfinite(v) for v in (*track.xyxy, track.confidence)) or not 0 <= track.confidence <= 1:
                raise ValueError("Invalid track coordinates/confidence.")
        events = []
        if self._last_sequence is not None:
            if sequence <= self._last_sequence or received_at < self._last_time:
                raise ValueError("Verification cannot reuse or reverse a frame.")
            if epoch != self._epoch or received_at - self._last_time > self.config.max_gap_seconds:
                events.extend(self.clear("tracker_epoch_or_time_gap"))
        for label in visible:
            if label not in self._evidence:
                self._evidence[label] = _Evidence(deque(maxlen=self.config.window), received_at)
        output = []
        for label, evidence in list(self._evidence.items()):
            track = visible.get(label)
            eligible = False
            if track:
                x1, y1, x2, y2 = track.xyxy
                area = max(0, min(w, x2) - max(0, x1)) * max(0, min(h, y2) - max(0, y1))
                eligible = track.confidence >= self.config.min_confidence and area / (w * h) >= self.config.min_visible_area_ratio
            evidence.history.append(bool(eligible))
            evidence.missing = 0 if track else evidence.missing + 1
            hits = sum(evidence.history)
            # Current good evidence is required: never show a weak/lost box as confirmed.
            state = "CONFIRMED" if eligible and hits >= self.config.required else "VERIFYING"
            if state != evidence.state:
                events.append({"event": "verification_confirmed" if state == "CONFIRMED" else "verification_revoked",
                               "track_id": label, "hits": hits, "sequence": sequence,
                               "evidence_elapsed_ms": round((received_at - evidence.first_seen) * 1000, 2)})
            evidence.state = state
            if track:
                output.append(VerifiedTrack(track, state, hits, bool(eligible)))
            if evidence.missing >= self.config.window:
                del self._evidence[label]
                events.append({"event": "verification_expired", "track_id": label})
        self._last_sequence, self._last_time, self._epoch = sequence, received_at, epoch
        return VerificationResult(tuple(output), tuple(events), (time.perf_counter() - started) * 1000)


def annotate_verification(frame: np.ndarray, result: VerificationResult, config: VerificationConfig) -> np.ndarray:
    image = frame.copy()
    h, w = image.shape[:2]
    for item in result.tracks:
        x1, y1, x2, y2 = item.track.xyxy
        x1, x2 = [max(0, min(w - 1, round(x))) for x in (x1, x2)]
        y1, y2 = [max(0, min(h - 1, round(y))) for y in (y1, y2)]
        color = (70, 220, 70) if item.state == "CONFIRMED" else (0, 200, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"{item.track.track_id} {item.state} {item.hits}/{config.window} {item.track.confidence:.2f}"
        cv2.putText(image, label, (x1, max(18, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2, cv2.LINE_AA)
    return image
