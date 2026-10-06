"""Evaluate one-class body detection and report recall for every source posture."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import cv2
import numpy as np

from .detector import DEFAULT_MODEL, DetectorConfig, PersonDetector
from .prepare_fallen_person import EXPECTED_CLASSES, SOURCE, parse_label


def iou(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def match(predictions: list[tuple[float, ...]], truths: list[tuple[int, tuple[float, ...]]],
          threshold: float = 0.5) -> tuple[set[int], set[int]]:
    candidates = sorted(((iou(prediction, box), p, t)
                         for p, prediction in enumerate(predictions)
                         for t, (_, box) in enumerate(truths)), reverse=True)
    used_predictions: set[int] = set()
    used_truths: set[int] = set()
    for score, prediction, truth in candidates:
        if score < threshold:
            break
        if prediction not in used_predictions and truth not in used_truths:
            used_predictions.add(prediction)
            used_truths.add(truth)
    return used_predictions, used_truths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--split", choices=("valid", "test"), default="test")
    parser.add_argument("--confidence", type=float, default=0.2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    detector = PersonDetector(DetectorConfig(model=args.model, imgsz=640, confidence=args.confidence))
    totals: Counter = Counter()
    found: Counter = Counter()
    false_positives = images = 0
    times = []
    image_dir = args.source / args.split / "images"
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise ValueError(f"Unreadable image: {image_path}")
        height, width = frame.shape[:2]
        rows = parse_label(args.source / args.split / "labels" / f"{image_path.stem}.txt")
        truths = [(class_id, ((x - w / 2) * width, (y - h / 2) * height,
                              (x + w / 2) * width, (y + h / 2) * height))
                  for class_id, (x, y, w, h) in rows]
        result = detector.predict(frame)
        used_predictions, used_truths = match([item.xyxy for item in result.people], truths)
        totals.update(class_id for class_id, _ in truths)
        found.update(truths[index][0] for index in used_truths)
        false_positives += len(result.people) - len(used_predictions)
        times.append(result.inference_ms)
        images += 1
    true_positives = sum(found.values())
    total = sum(totals.values())
    report = {
        "model": str(args.model.resolve()), "model_class": detector.class_name,
        "split": args.split, "images": images, "confidence": args.confidence,
        "match_iou": 0.5, "true_positives": true_positives,
        "false_positives": false_positives, "false_negatives": total - true_positives,
        "precision": true_positives / max(1, true_positives + false_positives),
        "recall": true_positives / max(1, total),
        "pose_recall": {EXPECTED_CLASSES[index]: found[index] / max(1, totals[index])
                        for index in range(len(EXPECTED_CLASSES))},
        "pose_counts": {EXPECTED_CLASSES[index]: {"found": found[index], "total": totals[index]}
                        for index in range(len(EXPECTED_CLASSES))},
        "inference_median_ms": float(np.median(times)),
        "inference_p95_ms": float(np.percentile(times, 95)),
        "warning": "Supplied split shares source groups; diagnostic, not independent field accuracy.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
