"""Calibrate and evaluate a general-person + posture-specialist detector fusion.

Calibration uses only the validation split.  The chosen policy can then be applied
once to the unchanged test split.  This prevents test-set threshold tuning.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from itertools import product
import json
from pathlib import Path

import cv2
import numpy as np

from .detector import DetectorConfig, PersonDetector
from .evaluate_nidar_detector import original_fallen_labels, size_bin, source_name
from .evaluate_pose_detector import match
from .hybrid_detector import FusionPolicy, fuse
from .prepare_fallen_person import EXPECTED_CLASSES, parse_label


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT / "datasets" / "nidar_person_v4_full_visdrone"
DEFAULT_PRIMARY = PROJECT / "models" / "yolo11n.pt"
DEFAULT_SPECIALIST = (
    PROJECT / "runs" / "training" / "nidar-person-yolo11n-v7-curriculum"
    / "weights" / "best-person-candidate.pt"
)


def collect_predictions(
    dataset: Path,
    split: str,
    primary_model: Path,
    specialist_model: Path,
) -> list[dict]:
    primary = PersonDetector(DetectorConfig(primary_model, 640, 0.10, 0.45, "0", False))
    specialist = PersonDetector(DetectorConfig(specialist_model, 640, 0.10, 0.45, "0", False))
    records: list[dict] = []
    images, labels = dataset / "images" / split, dataset / "labels" / split
    for number, image_path in enumerate(sorted(images.iterdir()), start=1):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise ValueError(f"Unreadable image: {image_path}")
        height, width = frame.shape[:2]
        truths = [
            (class_id, ((x - w / 2) * width, (y - h / 2) * height,
                        (x + w / 2) * width, (y + h / 2) * height))
            for class_id, (x, y, w, h) in parse_label(labels / f"{image_path.stem}.txt")
        ]
        base = primary.predict(frame)
        domain = specialist.predict(frame)
        records.append({
            "name": image_path.name,
            "shape": (height, width),
            "truths": truths,
            "primary": base.people,
            "specialist": domain.people,
            "latency_ms": base.inference_ms + domain.inference_ms,
        })
        if number % 250 == 0:
            print(json.dumps({"event": "hybrid_inference_progress", "split": split, "images": number}), flush=True)
    return records


def report(
    records: list[dict],
    policy: FusionPolicy,
    dataset: Path,
    split: str,
    fallen_classes: dict[str, list[int]] | None = None,
) -> dict:
    totals: Counter = Counter()
    sources: dict[str, Counter] = defaultdict(Counter)
    sizes: dict[str, Counter] = defaultdict(Counter)
    pose_totals: Counter = Counter()
    pose_found: Counter = Counter()
    # Loading the source labels is relatively expensive.  Calibration evaluates
    # many policies, so callers can load this mapping once and reuse it.
    fallen_classes = fallen_classes or original_fallen_labels()
    latencies: list[float] = []
    for record in records:
        detections = fuse(record["primary"], record["specialist"], record["shape"], policy)
        truths = record["truths"]
        _, used_truths = match([item.xyxy for item in detections], truths)
        source = source_name(record["name"])
        totals["truths"] += len(truths)
        totals["tp"] += len(used_truths)
        totals["fp"] += len(detections) - len(used_truths)
        sources[source]["truths"] += len(truths)
        sources[source]["tp"] += len(used_truths)
        sources[source]["fp"] += len(detections) - len(used_truths)
        sources[source]["images"] += 1
        if not truths and detections:
            sources[source]["negative_images_with_fp"] += 1
        for index, (_, box) in enumerate(truths):
            group = size_bin(box)
            sizes[group]["truths"] += 1
            sizes[group]["tp"] += int(index in used_truths)
        if source == "fallen":
            stem = Path(record["name"]).stem[len("fallen_"):]
            classes = fallen_classes.get(stem)
            if classes is None or len(classes) != len(truths):
                raise ValueError(f"Cannot map fallen poses for {record['name']}")
            for index, class_id in enumerate(classes):
                pose = EXPECTED_CLASSES[class_id]
                pose_totals[pose] += 1
                pose_found[pose] += int(index in used_truths)
        latencies.append(record["latency_ms"])

    def metrics(values: Counter) -> dict:
        tp, truth, fp = values["tp"], values["truths"], values["fp"]
        return {**dict(values), "precision": tp / max(1, tp + fp), "recall": tp / max(1, truth)}

    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_class": "person_candidate",
        "mode": "hybrid_general_plus_posture_specialist",
        "dataset": str(dataset.resolve()), "split": split,
        "policy": asdict(policy), "images": len(records),
        "overall": metrics(totals),
        "by_source": {name: metrics(value) for name, value in sorted(sources.items())},
        "by_size": {name: {**dict(value), "recall": value["tp"] / max(1, value["truths"])}
                    for name, value in sorted(sizes.items())},
        "fallen_pose_recall": {name: pose_found[name] / max(1, pose_totals[name]) for name in EXPECTED_CLASSES},
        "inference_median_ms": float(np.median(latencies)),
        "inference_p95_ms": float(np.percentile(latencies, 95)),
        "interpretation": "Visible person/body candidate only; not medical status or unique identity.",
    }


def proxy_checks(candidate: dict, baseline: dict) -> dict[str, bool]:
    sources, poses, sizes = candidate["by_source"], candidate["fallen_pose_recall"], candidate["by_size"]
    base_sources = baseline["by_source"]
    return {
        "coco_recall": sources["coco"]["recall"] >= 0.65,
        "coco_precision": sources["coco"]["precision"] >= 0.75,
        "coco_negative_fp_images": sources["coco"].get("negative_images_with_fp", 0)
        <= base_sources["coco"].get("negative_images_with_fp", 0) + 20,
        "fallen_recall": sources["fallen"]["recall"] >= 0.85,
        "lying_recall": poses["lying"] >= 0.65,
        "large_recall": sizes["large"]["recall"] >= 0.75,
        "small_recall_improvement": sizes["small"]["recall"] >= baseline["by_size"]["small"]["recall"] * 2,
        "aerial_recall_improvement": sources["visdrone"]["recall"] >= base_sources["visdrone"]["recall"] * 2,
        "p95_latency_ms": candidate["inference_p95_ms"] <= 70,
    }


def calibrate(records: list[dict], dataset: Path) -> tuple[FusionPolicy, dict, dict]:
    fallen_classes = original_fallen_labels()
    # Keep the normal COCO model at the same operating threshold while disabling
    # every specialist-only path.  This is the fair primary-only comparison.
    baseline_policy = FusionPolicy(0.20, 1.0, 1.0, 2.0, 1.0, 0.001, 0.5)
    baseline = report(records, baseline_policy, dataset, "valid", fallen_classes)
    best: tuple | None = None
    for general, posture, aspect, small_conf, small_area, dedupe in product(
        (0.40, 0.55, 0.70),
        (0.15, 0.20, 0.25),
        (1.0, 1.30),
        (0.20, 0.30),
        (0.006, 0.015),
        (0.50,),
    ):
        policy = FusionPolicy(0.20, general, posture, aspect, small_conf, small_area, dedupe)
        candidate = report(records, policy, dataset, "valid", fallen_classes)
        checks = proxy_checks(candidate, baseline)
        sources = candidate["by_source"]
        utility = (
            sum(checks.values()),
            sources["coco"]["precision"] + sources["coco"]["recall"]
            + sources["fallen"]["recall"] + candidate["fallen_pose_recall"]["lying"]
            + sources["visdrone"]["recall"] + candidate["by_size"]["small"]["recall"]
            - sources["coco"].get("negative_images_with_fp", 0) / 1000,
        )
        if best is None or utility > best[0]:
            best = (utility, policy, candidate, checks)
    assert best is not None
    selected = {**best[2], "calibration_checks": best[3], "calibration_score": list(best[0])}
    return best[1], selected, baseline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--primary", type=Path, default=DEFAULT_PRIMARY)
    parser.add_argument("--specialist", type=Path, default=DEFAULT_SPECIALIST)
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--policy", type=Path, help="Calibration JSON required for test evaluation.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.split == "test" and not args.policy:
        parser.error("Test evaluation requires a validation-selected --policy.")
    records = collect_predictions(args.dataset.resolve(), args.split, args.primary, args.specialist)
    if args.split == "valid":
        policy, result, baseline = calibrate(records, args.dataset.resolve())
        payload = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "selection_split": "valid", "policy": asdict(policy),
            "selected_validation_report": result, "primary_validation_report": baseline,
            "primary_model": str(args.primary.resolve()), "specialist_model": str(args.specialist.resolve()),
        }
    else:
        policy_data = json.loads(args.policy.read_text(encoding="utf-8"))
        policy = FusionPolicy(**policy_data["policy"])
        payload = report(records, policy, args.dataset.resolve(), "test")
        payload["primary_model"] = str(args.primary.resolve())
        payload["specialist_model"] = str(args.specialist.resolve())
        payload["policy_source"] = str(args.policy.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
