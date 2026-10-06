"""Validated fusion of a general-person detector and a posture specialist.

The general COCO model remains the primary detector.  The specialist may only add
non-overlapping evidence that passes the validation-selected policy.  Both models
still detect one semantic class: a visible person/body candidate.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import time

import numpy as np

from .detector import Detection, DetectionResult, DetectorConfig, PersonDetector


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_SPECIALIST_MODEL = PROJECT / "models" / "nidar-person-posture-specialist-yolo11n.pt"
DEFAULT_FUSION_CONFIG = Path(__file__).resolve().parent / "settings" / "hybrid_detector.json"


@dataclass(frozen=True)
class FusionPolicy:
    primary_confidence: float = 0.20
    general_confidence: float = 0.70
    posture_confidence: float = 0.25
    horizontal_aspect: float = 1.0
    small_confidence: float = 0.20
    small_area_ratio: float = 0.006
    dedupe_iou: float = 0.50

    def __post_init__(self) -> None:
        probabilities = (
            self.primary_confidence,
            self.general_confidence,
            self.posture_confidence,
            self.small_confidence,
            self.dedupe_iou,
        )
        if any(not math.isfinite(value) or not 0 < value <= 1 for value in probabilities):
            raise ValueError("Hybrid confidence and IoU values must be finite and in (0, 1].")
        if not math.isfinite(self.horizontal_aspect) or self.horizontal_aspect <= 0:
            raise ValueError("Hybrid horizontal aspect must be finite and positive.")
        if not math.isfinite(self.small_area_ratio) or not 0 < self.small_area_ratio <= 1:
            raise ValueError("Hybrid small-area ratio must be finite and in (0, 1].")

    @classmethod
    def load(cls, path: Path) -> "FusionPolicy":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read hybrid detector configuration: {path}") from exc
        if set(payload) != set(cls.__dataclass_fields__):
            raise ValueError("Hybrid detector configuration has missing or unknown fields.")
        try:
            return cls(**payload)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid hybrid detector configuration: {path}") from exc


def intersection_over_union(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_first = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_second = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    return intersection / max(area_first + area_second - intersection, 1e-9)


def fuse(
    primary: tuple[Detection, ...],
    specialist: tuple[Detection, ...],
    frame_shape: tuple[int, int],
    policy: FusionPolicy,
) -> tuple[Detection, ...]:
    """Preserve primary detections and add only qualified specialist evidence."""
    height, width = frame_shape
    frame_area = max(1, height * width)
    accepted = [item for item in primary if item.confidence >= policy.primary_confidence]
    for item in sorted(specialist, key=lambda value: value.confidence, reverse=True):
        if any(intersection_over_union(item.xyxy, existing.xyxy) >= policy.dedupe_iou
               for existing in accepted):
            continue
        x1, y1, x2, y2 = item.xyxy
        box_width, box_height = max(0.0, x2 - x1), max(0.0, y2 - y1)
        horizontal = box_width / max(box_height, 1e-9) >= policy.horizontal_aspect
        small = box_width * box_height / frame_area <= policy.small_area_ratio
        if (item.confidence >= policy.general_confidence
                or horizontal and item.confidence >= policy.posture_confidence
                or small and item.confidence >= policy.small_confidence):
            accepted.append(item)
    return tuple(accepted)


class HybridPersonDetector:
    """Run primary and specialist YOLO models, then fuse their person boxes."""

    def __init__(
        self,
        config: DetectorConfig,
        specialist_model: Path = DEFAULT_SPECIALIST_MODEL,
        fusion_config: Path = DEFAULT_FUSION_CONFIG,
    ) -> None:
        configured_policy = FusionPolicy.load(fusion_config)
        # Honour the caller's primary threshold exactly.  Tracking intentionally
        # requests 0.10 so BoT-SORT can recover weak boxes; a presentation replay
        # may request 0.25 to suppress borderline duplicates.
        self.policy = replace(
            configured_policy,
            primary_confidence=config.confidence,
        )
        self.primary = PersonDetector(config)
        specialist_floor = min(
            self.policy.general_confidence,
            self.policy.posture_confidence,
            self.policy.small_confidence,
        )
        self.specialist = PersonDetector(replace(
            config,
            model=Path(specialist_model),
            confidence=min(config.confidence, specialist_floor),
        ))
        self.config = config
        self.device = self.primary.device
        self.half = self.primary.half and self.specialist.half
        self.class_name = "person_candidate"
        self.specialist_model = Path(specialist_model)
        self.fusion_config = Path(fusion_config)

    def predict(self, frame: np.ndarray) -> DetectionResult:
        started = time.perf_counter()
        primary = self.primary.predict(frame)
        specialist = self.specialist.predict(frame)
        people = fuse(primary.people, specialist.people, frame.shape[:2], self.policy)
        return DetectionResult(people, (time.perf_counter() - started) * 1000)
