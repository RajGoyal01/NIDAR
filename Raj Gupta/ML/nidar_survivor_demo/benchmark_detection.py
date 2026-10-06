"""Reproducible balanced COCO smoke evaluation, not official COCO mAP."""
import argparse
import json
from pathlib import Path
import random
import numpy as np
import cv2
import torch
from .detector import DEFAULT_MODEL, PersonDetector, DetectorConfig


def overlap(a, b):
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def match_counts(people, truth):
    unused = set(range(len(truth)))
    tp = 0
    for person in sorted(people, key=lambda x: x.confidence, reverse=True):
        best = max(unused, key=lambda i: overlap(person.xyxy, truth[i]), default=None)
        if best is not None and overlap(person.xyxy, truth[best]) >= 0.5:
            unused.remove(best)
            tp += 1
    return tp, len(people) - tp, len(unused)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(__file__).resolve().parents[1] / "datasets/coco2017")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--sizes", type=int, nargs="+", choices=(512, 640), default=(512, 640))
    parser.add_argument("--per-group", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("logs/phase2_benchmark.json"))
    args = parser.parse_args()
    if not 1 <= args.per_group <= 2307:
        parser.error("--per-group must be between 1 and 2307")
    groups = {True: [], False: []}
    for line in (args.dataset / "person_val.txt").read_text().splitlines():
        image = args.dataset / line
        label = args.dataset / "labels/val2017" / (image.stem + ".txt")
        rows = [list(map(float, row.split())) for row in label.read_text().splitlines() if row.strip()]
        groups[bool(rows)].append((image, rows))
    rng = random.Random(42)
    selected = sum((rng.sample(sorted(groups[key]), args.per_group) for key in (True, False)), [])
    rng.shuffle(selected)
    summary = {"seed": 42, "images": [p.name for p, _ in selected],
               "positive_images": args.per_group, "negative_images": args.per_group,
               "model": str(args.model.resolve()), "match_iou": 0.5,
               "confidence": args.confidence, "nms_iou": 0.45,
               "scope": "Balanced sample, crowd-person excluded; not mAP or survivor validation", "results": []}
    for size in args.sizes:
        detector = PersonDetector(DetectorConfig(model=args.model, imgsz=size, confidence=args.confidence))
        if detector.device != "cpu":
            torch.cuda.reset_peak_memory_stats()
        tp = fp = fn = negative_fp_images = 0
        times = []
        for path, rows in selected:
            frame = cv2.imread(str(path))
            if frame is None:
                raise ValueError("Unreadable evaluation image")
            h, w = frame.shape[:2]
            truth = [((x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h)
                     for _, x, y, bw, bh in rows]
            result = detector.predict(frame)
            counts = match_counts(result.people, truth)
            tp += counts[0]
            fp += counts[1]
            fn += counts[2]
            negative_fp_images += int(not truth and bool(result.people))
            times.append(result.inference_ms)
        record = {"imgsz": size, "device": detector.device, "fp16": detector.half,
                  "tp": tp, "fp": fp, "fn": fn, "precision": tp / max(1, tp + fp),
                  "recall": tp / max(1, tp + fn), "negative_images_with_false_positive": negative_fp_images,
                  "inference_median_ms": float(np.median(times)), "inference_p95_ms": float(np.percentile(times, 95)),
                  "peak_allocated_vram_mib": torch.cuda.max_memory_allocated() / 2**20 if detector.device != "cpu" else None}
        summary["results"].append(record)
        print(json.dumps(record), flush=True)
        del detector
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
