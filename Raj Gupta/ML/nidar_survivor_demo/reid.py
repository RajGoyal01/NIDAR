"""Selective session-local appearance matching, not a survivor database/counter."""
from collections import deque
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import time
from typing import Callable
import cv2
import numpy as np
from .reid_encoder import unit_vector
from .verification import VerificationResult

DEFAULT_REID_CONFIG = Path(__file__).resolve().parent / "settings/reid.json"


@dataclass(frozen=True)
class ReIDConfig:
    match_threshold: float = .815
    novelty_threshold: float = .55
    ambiguity_margin: float = .08
    samples: int = 3
    sample_interval: float = .15
    retry_interval: float = .5
    min_confidence: float = .5
    min_width: int = 32
    min_height: int = 96
    min_visible_fraction: float = .9
    min_blur_variance: float = 35.0
    max_overlap: float = .35
    max_batch: int = 2
    max_gallery: int = 64
    gallery_ttl: float = 600.0
    edge_margin_fraction: float = .01
    novelty_seconds: float = 1.5
    novelty_batches: int = 3
    sample_memory_seconds: float = .75
    partial_match_threshold: float = .9
    max_reference_samples: int = 12
    allow_initial_partial: bool = True
    allow_partial_novelty: bool = False

    def __post_init__(self):
        if type(self.allow_initial_partial) is not bool or type(self.allow_partial_novelty) is not bool:
            raise ValueError("Initial partial enrollment must be boolean.")
        if (type(self.edge_margin_fraction) not in (int, float) or
                not math.isfinite(self.edge_margin_fraction) or not 0 <= self.edge_margin_fraction <= .1):
            raise ValueError("Edge margin must be between zero and .1.")
        for name in ("match_threshold", "novelty_threshold", "ambiguity_margin", "min_confidence", "min_visible_fraction", "max_overlap"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"Invalid Re-ID fraction: {name}.")
        if self.novelty_threshold >= self.match_threshold:
            raise ValueError("Novelty threshold must be below match threshold.")
        for name, maximum in (("samples", 8), ("min_width", 2048), ("min_height", 2048), ("max_batch", 8), ("max_gallery", 256), ("novelty_batches", 10)):
            if type(getattr(self, name)) is not int or not 1 <= getattr(self, name) <= maximum:
                raise ValueError(f"Invalid Re-ID integer: {name}.")
        for name in ("sample_interval", "retry_interval", "min_blur_variance", "gallery_ttl", "novelty_seconds", "sample_memory_seconds"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid Re-ID positive value: {name}.")
        if self.sample_memory_seconds > 1:
            raise ValueError("Identity sample memory cannot exceed one second.")
        if (type(self.partial_match_threshold) not in (int, float) or
                not math.isfinite(self.partial_match_threshold) or
                not self.match_threshold <= self.partial_match_threshold <= 1):
            raise ValueError("Partial matches require a threshold at least as strict as full matches.")
        if type(self.max_reference_samples) is not int or not self.samples <= self.max_reference_samples <= 24:
            raise ValueError("Reference sample capacity must cover the initial batch, at most 24.")

    @classmethod
    def load(cls, path: Path):
        try:
            return cls(**json.loads(path.read_text(encoding="utf-8")))
        except (OSError, TypeError, ValueError) as exc:
            raise ValueError("Invalid Re-ID JSON: check keys, thresholds and file.") from exc


def quality_crop(frame, track, other_tracks, config, *, allow_edge=False):
    """Return a crop or an explicit rejection reason; never save pixels."""
    x1, y1, x2, y2 = track.xyxy
    if not all(math.isfinite(x) for x in (*track.xyxy, track.confidence)):
        return None, "invalid_box"
    if track.confidence < config.min_confidence:
        return None, "low_confidence"
    area = max(0, x2 - x1) * max(0, y2 - y1)
    if not area:
        return None, "invalid_box"
    h, w = frame.shape[:2]
    a, b, c, d = max(0, math.floor(x1)), max(0, math.floor(y1)), min(w, math.ceil(x2)), min(h, math.ceil(y2))
    crop_width, crop_height = c - a, d - b
    upright = crop_width >= config.min_width and crop_height >= config.min_height
    lying_horizontal = crop_width >= config.min_height and crop_height >= config.min_width
    if not upright and not lying_horizontal:
        return None, "small_crop"
    if (max(0, min(w, x2) - max(0, x1)) * max(0, min(h, y2) - max(0, y1))) / area < config.min_visible_fraction:
        return None, "clipped_crop"
    # Detector boxes are often already clipped: visible fraction alone is then 1.
    if not allow_edge and config.edge_margin_fraction and (x1 <= w * config.edge_margin_fraction or
            y1 <= h * config.edge_margin_fraction or x2 >= w * (1 - config.edge_margin_fraction) or
            y2 >= h * (1 - config.edge_margin_fraction)):
        return None, "frame_edge_crop"
    for other in other_tracks:
        if other.track_id == track.track_id:
            continue
        u, v, s, t = other.xyxy
        overlap = max(0, min(x2, s) - max(x1, u)) * max(0, min(y2, t) - max(y1, v))
        if overlap / area > config.max_overlap:
            return None, "overlapping_person"
    crop = frame[b:d, a:c]
    # OSNet was trained on upright pedestrian crops. Normalize a wide,
    # horizontal body crop before resizing; keep the original frame/box intact.
    if not upright and lying_horizontal:
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
    gray = cv2.cvtColor(cv2.resize(crop, (128, 256)), cv2.COLOR_BGR2GRAY)
    if cv2.Laplacian(gray, cv2.CV_64F).var() < config.min_blur_variance:
        return None, "blurred_crop"
    return crop, "good"


def compare_gallery(queries, gallery):
    """Conservative multi-view cosine: weakest query's mean of its best two references."""
    query = np.stack([unit_vector(x) for x in queries])
    ranked = []
    for reference, vectors in gallery.items():
        scores = query @ np.stack([unit_vector(x) for x in vectors]).T
        best = np.sort(scores, axis=1)[:, -min(2, scores.shape[1]):].mean(axis=1)
        ranked.append((reference, float(best.min()), float(scores.max())))
    return sorted(ranked, key=lambda row: (-row[1], row[0]))


@dataclass
class _Reference:
    vectors: tuple[np.ndarray, ...]
    last_seen: float


@dataclass
class _Pending:
    vectors: deque = field(default_factory=deque)
    sample_times: deque = field(default_factory=deque)
    partial_views: deque = field(default_factory=deque)
    last_sample: float = -float("inf")
    last_attempt: float = -float("inf")
    state: str = "COLLECTING"
    reason: str = "need_good_views"
    score: float | None = None
    margin: float | None = None
    novelty_started: float | None = None
    novelty_hits: int = 0
    novelty_samples: int = 0
    novelty_anchor: np.ndarray | None = None

    def reset_novelty(self):
        self.novelty_started = None
        self.novelty_hits = self.novelty_samples = 0
        self.novelty_anchor = None


@dataclass(frozen=True)
class ReIDObservation:
    track_id: str
    reference_id: str | None
    state: str
    reason: str
    score: float | None = None
    margin: float | None = None


@dataclass(frozen=True)
class ReIDResult:
    observations: tuple[ReIDObservation, ...]
    events: tuple[dict, ...]
    reid_ms: float
    crops_encoded: int


class AppearanceMatcher:
    def __init__(self, config: ReIDConfig, encoder, *, retain_gallery: bool = False):
        self.config, self.encoder = config, encoder
        self.retain_gallery = retain_gallery
        self.gallery: dict[str, _Reference] = {}
        self._bindings: dict[str, str] = {}
        self._pending: dict[str, _Pending] = {}
        self._counter = 0
        self._last_sequence = None
        self._last_time = None
        self._epoch = None

    def clear_tracks(self):
        """Keep only session gallery through a camera outage; require fresh matching."""
        self._bindings.clear()
        self._pending.clear()
        self._epoch = None

    def update(self, frame, verified: VerificationResult, *, sequence: int, received_at: float,
               epoch: int, is_fresh: Callable[[], bool] = lambda: True) -> ReIDResult:
        started = time.perf_counter()
        if type(sequence) is not int or sequence < 1 or not math.isfinite(received_at):
            raise ValueError("Re-ID requires a valid frame sequence/time.")
        if self._last_sequence is not None and (sequence <= self._last_sequence or received_at < self._last_time):
            raise ValueError("Re-ID cannot repeat or reverse frames.")
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3 or not frame.size:
            raise ValueError("Re-ID requires a uint8 BGR frame.")
        tracks = [item.track for item in verified.tracks]
        if len({t.track_id for t in tracks}) != len(tracks):
            raise ValueError("Duplicate Re-ID track IDs.")
        if epoch != self._epoch or (self._last_time is not None and received_at - self._last_time > 1):
            self.clear_tracks()
        self._epoch = epoch
        self._last_sequence, self._last_time = sequence, received_at
        visible = {t.track_id for t in tracks}
        self._bindings = {k: v for k, v in self._bindings.items() if k in visible}
        self._pending = {k: v for k, v in self._pending.items() if k in visible}
        occupied = set(self._bindings.values())
        events = []
        for ref in list(self.gallery):
            if not self.retain_gallery and ref not in occupied and received_at - self.gallery[ref].last_seen > self.config.gallery_ttl:
                del self.gallery[ref]
                events.append({"event": "reid_reference_expired", "reference_id": ref})
        confirmed = [v.track for v in verified.tracks if v.state == "CONFIRMED"]
        for item in verified.tracks:
            if item.state != "CONFIRMED" and item.track.track_id not in self._bindings:
                pending = self._pending.get(item.track.track_id)
                if pending is not None:
                    pending.reset_novelty()
                    if received_at - pending.last_sample > self.config.sample_memory_seconds:
                        self._pending.pop(item.track.track_id, None)
        crops, labels = [], []
        # max_batch protects GPU latency/VRAM; it must not become an identity
        # count limit.  Build every usable candidate first, then select the
        # least-recently-sampled tracks so a crowded frame is served fairly.
        sample_candidates = []
        for track in confirmed:
            label = track.track_id
            if label in self._bindings:
                continue  # No every-frame embedding for already associated tracks.
            pending = self._pending.setdefault(label, _Pending(deque(maxlen=self.config.samples)))
            # Expire individual good views, not just time since the most recent
            # sample: intermittent good frames must not preserve old evidence.
            while pending.sample_times and received_at - pending.sample_times[0] > self.config.sample_memory_seconds:
                pending.sample_times.popleft()
                pending.vectors.popleft()
                pending.partial_views.popleft()
                pending.reset_novelty()
            # Sampling and comparison have independent clocks. A failed match
            # must not prevent collecting fresh, spaced evidence.
            if received_at - pending.last_sample < self.config.sample_interval:
                continue
            crop, reason = quality_crop(frame, track, tracks, self.config)
            partial = reason == "frame_edge_crop"
            if partial and (self.gallery or self.config.allow_initial_partial):
                # Empty-gallery bootstrap is a separate policy: no previous
                # identity exists to split. Afterwards partial views are
                # recognition-only and cannot claim novelty.
                crop, reason = quality_crop(frame, track, tracks, self.config, allow_edge=True)
            if crop is None:
                pending.reset_novelty()
                # Brief blur/confidence dips can reuse recent GOOD samples.
                # Clipping/overlap may show a different body region/person, so
                # they still invalidate the entire batch immediately.
                if reason not in {"blurred_crop", "low_confidence"}:
                    pending.vectors.clear()
                    pending.sample_times.clear()
                    pending.partial_views.clear()
                pending.state, pending.reason = "LOW_QUALITY", reason
                continue
            sample_candidates.append((pending.last_sample, label, crop, partial))
        for _, label, crop, partial in sorted(
                sample_candidates, key=lambda item: (item[0], item[1]))[:self.config.max_batch]:
            crops.append(crop)
            labels.append(label)
            pending = self._pending[label]
            pending.partial_views.append(partial)
            while len(pending.partial_views) > self.config.samples:
                pending.partial_views.popleft()
        vectors = self.encoder.encode(crops) if crops else []
        if len(vectors) != len(crops):
            raise ValueError("Re-ID encoder returned wrong batch length.")
        vectors = [unit_vector(v) for v in vectors]
        if vectors and len({len(v) for v in vectors} | {len(v) for r in self.gallery.values() for v in r.vectors}
                           | {len(v) for p in self._pending.values() for v in p.vectors}) > 1:
            raise ValueError("Re-ID embedding dimensions changed.")
        if not is_fresh():
            self.clear_tracks()
            return ReIDResult((), ({"event": "reid_stale_discard"},), (time.perf_counter() - started) * 1000, len(crops))
        for label, vector in zip(labels, vectors):
            pending = self._pending[label]
            pending.vectors.append(vector)
            pending.sample_times.append(received_at)
            while len(pending.sample_times) > self.config.samples:
                pending.sample_times.popleft()
            pending.novelty_samples += 1
            pending.last_sample = received_at
            pending.state, pending.reason = "COLLECTING", "need_good_views"
        proposals = {}
        # Evaluate against one unchanged gallery before any assignment this frame.
        for label in labels:
            pending = self._pending[label]
            if len(pending.vectors) < self.config.samples:
                continue
            if received_at - pending.last_attempt < self.config.retry_interval:
                continue
            pending.last_attempt = received_at
            ranked = compare_gallery(pending.vectors, {k: r.vectors for k, r in self.gallery.items()})
            top = ranked[0] if ranked else None
            partial = any(pending.partial_views)
            match_floor = self.config.partial_match_threshold if partial else self.config.match_threshold
            margin = top[1] - ranked[1][1] if len(ranked) > 1 else 1.0
            pending.score, pending.margin = (top[1] if top else None), margin
            # Queries themselves must agree; avoid enrolling an ID-switched mixed batch.
            self_similarity = np.stack(pending.vectors) @ np.stack(pending.vectors).T
            if float(self_similarity.min()) < self.config.match_threshold:
                pending.reset_novelty()
                pending.state, pending.reason = "UNCERTAIN", "inconsistent_views"
            elif top and top[1] >= match_floor and margin >= self.config.ambiguity_margin:
                pending.reset_novelty()
                if top[0] in occupied:
                    pending.state, pending.reason = "UNCERTAIN", "reference_visible_on_other_track"
                else:
                    proposals[label] = top[0]
            elif partial and ((top and not self.config.allow_partial_novelty) or
                              (not top and not self.config.allow_initial_partial)):
                pending.reset_novelty()
                pending.state, pending.reason = "UNCERTAIN", "partial_view_needs_full_evidence"
            elif not top or max(row[2] for row in ranked) < self.config.novelty_threshold:
                if top and (self.retain_gallery or partial):
                    # Require fresh, non-overlapping batches over time, not all
                    # earlier people simultaneously in view. Keep a fixed anchor
                    # so a gradually changing/mixed track cannot accumulate votes.
                    if (pending.novelty_anchor is not None and
                            min(float(v @ pending.novelty_anchor) for v in pending.vectors) < self.config.match_threshold):
                        pending.reset_novelty()
                    if pending.novelty_started is None:
                        pending.novelty_started = received_at
                        pending.novelty_anchor = pending.vectors[0].copy()
                        pending.novelty_hits = 1
                        pending.novelty_samples = 0
                    elif pending.novelty_samples >= self.config.samples:
                        pending.novelty_hits += 1
                        pending.novelty_samples = 0
                    if (pending.novelty_hits < self.config.novelty_batches or
                            received_at - pending.novelty_started < self.config.novelty_seconds):
                        pending.state, pending.reason = "COLLECTING", "confirming_new_person"
                        continue
                proposals[label] = None
            else:
                pending.reset_novelty()
                pending.state, pending.reason = "UNCERTAIN", "similarity_or_margin_ambiguous"
        for label, reference in proposals.items():
            pending = self._pending[label]
            if reference and list(proposals.values()).count(reference) > 1:
                pending.state, pending.reason = "UNCERTAIN", "simultaneous_match_conflict"
                continue
            if reference is None:
                if any(pending.partial_views) and self.gallery and not self.config.allow_partial_novelty:
                    pending.state, pending.reason = "UNCERTAIN", "partial_view_needs_full_evidence"
                    continue
                # Earlier proposals may have enrolled a similar reference this frame.
                current = compare_gallery(pending.vectors, {k: r.vectors for k, r in self.gallery.items()})
                if current and max(row[2] for row in current) >= self.config.novelty_threshold:
                    pending.state, pending.reason = "UNCERTAIN", "simultaneous_novelty_conflict"
                    continue
                if len(self.gallery) >= self.config.max_gallery:
                    pending.state, pending.reason = "UNCERTAIN", "gallery_capacity"
                    continue
                self._counter += 1
                reference = f"R{self._counter}"
                self.gallery[reference] = _Reference(tuple(v.copy() for v in pending.vectors), received_at)
                state = "NEW_REFERENCE"
            else:
                state = "MATCHED"
                # Never replace the original anchor batch. Learn only clear
                # full-view matches that ALSO pass against those fixed anchors,
                # preventing a chain of weak updates from drifting identities.
                stored = self.gallery[reference].vectors
                anchors = stored[:self.config.samples]
                anchored = compare_gallery(pending.vectors, {reference: anchors})[0][1]
                if self.retain_gallery and not any(pending.partial_views) and anchored >= self.config.match_threshold:
                    additions = list(stored[self.config.samples:])
                    for vector in pending.vectors:
                        if max(float(vector @ v) for v in (*anchors, *additions)) < .98:
                            additions.append(vector.copy())
                    capacity = self.config.max_reference_samples - self.config.samples
                    self.gallery[reference].vectors = tuple(anchors) + tuple(additions[-capacity:] if capacity else [])
            self._bindings[label] = reference
            events.append({"event": "reid_new_reference" if state == "NEW_REFERENCE" else "reid_match",
                           "track_id": label, "reference_id": reference, "sequence": sequence,
                           "score": pending.score, "margin": pending.margin})
            # Initial anchors remain immutable; bounded additions are guarded above.
            pending.state, pending.reason = state, "session_appearance_only"
        observations = []
        for item in verified.tracks:
            label = item.track.track_id
            if item.state != "CONFIRMED":
                observations.append(ReIDObservation(label, None, "VERIFYING", "temporal_gate"))
                continue
            pending = self._pending.get(label)
            ref = self._bindings.get(label)
            if ref:
                self.gallery[ref].last_seen = received_at
            observations.append(ReIDObservation(label, ref, pending.state if pending else "COLLECTING",
                                                pending.reason if pending else "need_good_views",
                                                pending.score if pending else None, pending.margin if pending else None))
        return ReIDResult(tuple(observations), tuple(events), (time.perf_counter() - started) * 1000, len(crops))

    def export_gallery(self) -> dict[str, list[list[float]]]:
        return {key: [vector.tolist() for vector in value.vectors] for key, value in self.gallery.items()}

    def restore_gallery(self, values: dict[str, list[list[float]]]) -> None:
        """Restore validated vectors, never old tracker bindings or monotonic timestamps."""
        if self.gallery or self._last_sequence is not None or len(values) > self.config.max_gallery:
            raise ValueError("Restore needs an unused matcher and a bounded gallery.")
        restored = {}
        dimensions = set()
        for key, vectors in values.items():
            if not key.startswith("R") or not key[1:].isdigit() or int(key[1:]) < 1 or key != f"R{int(key[1:])}":
                raise ValueError("Invalid appearance reference ID.")
            if not self.config.samples <= len(vectors) <= self.config.max_reference_samples:
                raise ValueError("Wrong reference sample count.")
            normalized = tuple(unit_vector(np.asarray(v, dtype=np.float32)) for v in vectors)
            dimensions.update(len(v) for v in normalized)
            restored[key] = _Reference(normalized, 0.0)
        if len(dimensions) > 1:
            raise ValueError("Mixed restored embedding dimensions.")
        self.gallery = restored
        self._counter = max((int(k[1:]) for k in values), default=0)


def annotate_reid(image, verified, result):
    output = image.copy()
    boxes = {v.track.track_id: v.track.xyxy for v in verified.tracks}
    for item in result.observations:
        x1, y1, _, _ = boxes[item.track_id]
        text = f"{item.reference_id or '?'} {item.state}"
        if item.score is not None:
            text += f" cos={item.score:.2f}"
        cv2.putText(output, text, (max(0, min(output.shape[1] - 180, round(x1))), max(36, min(output.shape[0] - 8, round(y1) + 24))),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 210, 80), 2, cv2.LINE_AA)
    return output
