"""Evaluate a NIDAR one-class detector by source and target size on held-out data."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

import cv2
import numpy as np

from .detector import DetectorConfig, PersonDetector
from .evaluate_pose_detector import iou, match
from .prepare_fallen_person import EXPECTED_CLASSES, parse_label


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT / "datasets" / "nidar_person_v1"
RAW_FALLEN = PROJECT / "datasets" / "fallen_person" / "roboflow_v2"


def source_name(filename: str) -> str:
    return filename.split("_", 1)[0]


def size_bin(box: tuple[float, ...]) -> str:
    width, height = max(0.0, box[2] - box[0]), max(0.0, box[3] - box[1])
    area = width * height
    return "tiny" if area < 16**2 else "small" if area < 32**2 else "medium" if area < 96**2 else "large"


def original_fallen_labels() -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for split in ("train", "valid", "test"):
        for label in (RAW_FALLEN / split / "labels").glob("*.txt"):
            if label.stem in result:
                raise ValueError(f"Duplicate fallen source stem: {label.stem}")
            result[label.stem] = [class_id for class_id, _ in parse_label(label)]
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--split", choices=("valid", "test"), default="test")
    parser.add_argument("--confidence", type=float, default=0.2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    detector = PersonDetector(DetectorConfig(model=args.model, imgsz=640, confidence=args.confidence))
    images = args.dataset / "images" / args.split
    labels = args.dataset / "labels" / args.split
    fallen_classes = original_fallen_labels()
    totals: Counter[str] = Counter()
    found: Counter[str] = Counter()
    source_totals: dict[str, Counter] = defaultdict(Counter)
    size_totals: dict[str, Counter] = defaultdict(Counter)
    pose_totals: Counter[str] = Counter()
    pose_found: Counter[str] = Counter()
    latencies: list[float] = []
    image_count = 0
    for image_path in sorted(images.iterdir()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise ValueError(f"Unreadable image: {image_path}")
        height, width = frame.shape[:2]
        rows = parse_label(labels / f"{image_path.stem}.txt")
        truths = [(class_id, ((x - w / 2) * width, (y - h / 2) * height,
                              (x + w / 2) * width, (y + h / 2) * height))
                  for class_id, (x, y, w, h) in rows]
        inference = detector.predict(frame)
        _, used_truths = match([item.xyxy for item in inference.people], truths)
        group = source_name(image_path.name)
        totals["truths"] += len(truths)
        totals["tp"] += len(used_truths)
        totals["fp"] += len(inference.people) - len(used_truths)
        source_totals[group]["truths"] += len(truths)
        source_totals[group]["tp"] += len(used_truths)
        source_totals[group]["fp"] += len(inference.people) - len(used_truths)
        source_totals[group]["images"] += 1
        if not truths and inference.people:
            source_totals[group]["negative_images_with_fp"] += 1
        for index, (_, box) in enumerate(truths):
            name = size_bin(box)
            size_totals[name]["truths"] += 1
            size_totals[name]["tp"] += int(index in used_truths)
        if group == "fallen":
            source_stem = image_path.stem[len("fallen_"):]
            classes = fallen_classes.get(source_stem)
            if classes is None or len(classes) != len(truths):
                raise ValueError(f"Cannot map fallen pose labels for {image_path.name}")
            for index, class_id in enumerate(classes):
                pose = EXPECTED_CLASSES[class_id]
                pose_totals[pose] += 1
                pose_found[pose] += int(index in used_truths)
        latencies.append(inference.inference_ms)
        image_count += 1

    def metrics(values: Counter) -> dict[str, float | int]:
        tp, truth, fp = values["tp"], values["truths"], values["fp"]
        return {**dict(values), "precision": tp / max(1, tp + fp), "recall": tp / max(1, truth)}

    def recall_metrics(values: Counter) -> dict[str, float | int]:
        """Return target-size recall without inventing a size-specific precision.

        A missed ground-truth box has a well-defined size, so recall can be grouped
        by target size.  A false-positive detection has no matching ground truth,
        therefore assigning it to a target-size bucket would be arbitrary.  Overall
        and source-level precision remain available above.
        """
        tp, truth = values["tp"], values["truths"]
        return {**dict(values), "recall": tp / max(1, truth)}

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": str(args.model.resolve()),
        "model_class": detector.class_name,
        "dataset": str(args.dataset.resolve()),
        "split": args.split,
        "images": image_count,
        "confidence": args.confidence,
        "match_iou": 0.5,
        "overall": metrics(totals),
        "by_source": {name: metrics(value) for name, value in sorted(source_totals.items())},
        "by_size": {name: recall_metrics(value) for name, value in sorted(size_totals.items())},
        "fallen_pose_recall": {name: pose_found[name] / max(1, pose_totals[name]) for name in EXPECTED_CLASSES},
        "fallen_pose_counts": {name: {"found": pose_found[name], "total": pose_totals[name]} for name in EXPECTED_CLASSES},
        "inference_median_ms": float(np.median(latencies)),
        "inference_p95_ms": float(np.percentile(latencies, 95)),
        "interpretation": "person_candidate means visible human/body evidence, not survivor health, injury, liveness or death.",
        "known_gap": "No project-owned final OAK-D indoor field test is included.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
