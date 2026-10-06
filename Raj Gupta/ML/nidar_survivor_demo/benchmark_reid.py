"""COCO within-image different-person hard-negative proxy; NOT cross-camera accuracy."""
import argparse
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import time
import cv2
import numpy as np
from .reid_encoder import AppearanceEncoder
from .reid import ReIDConfig, quality_crop
from .tracking import Track


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(__file__).resolve().parents[1] / "datasets/coco2017")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads((args.dataset / "annotations/instances_val2017.json").read_text())
    groups = defaultdict(list)
    for ann in data["annotations"]:
        if ann["category_id"] == 1 and not ann["iscrowd"]:
            groups[ann["image_id"]].append(ann)
    config = ReIDConfig()
    encoder = AppearanceEncoder()
    records, latencies = [], []
    for image_id in sorted(groups):
        anns = groups[image_id]
        if not 2 <= len(anns) <= 8:
            continue
        frame = cv2.imread(str(args.dataset / "images/val2017" / f"{image_id:012d}.jpg"))
        if frame is None:
            continue
        tracks = [Track(str(a["id"]), (a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]), 1) for a in anns]
        good = []
        for track in tracks:
            crop, _ = quality_crop(frame, track, tracks, config)
            if crop is not None:
                hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist([hsv], [0, 1], None, [12, 8], [0, 180, 0, 256]).ravel()
                hist /= max(float(np.linalg.norm(hist)), 1e-8)
                good.append((track, crop, hist))
        if len(good) < 2:
            continue
        # Hard-negative proxy: choose the two most similar colour histograms in this image.
        pairs = [(float(a[2] @ b[2]), a, b) for i, a in enumerate(good) for b in good[i + 1:]]
        color_similarity, a, b = max(pairs, key=lambda row: row[0])
        augmented = cv2.convertScaleAbs(cv2.flip(a[1], 1), alpha=.9, beta=8)
        start = time.perf_counter()
        va, vb, vp = encoder.encode([a[1], b[1], augmented])
        latencies.append((time.perf_counter() - start) * 1000)
        records.append({"image_id": image_id, "annotation_ids": [a[0].track_id, b[0].track_id],
                        "color_similarity": color_similarity, "negative_cosine": float(va @ vb),
                        "augmented_positive_cosine": float(va @ vp)})
        if len(records) == 40:
            break
    if len(records) < 40:
        raise RuntimeError("Need 40 qualifying independent images for fixed proxy split.")
    calibration, heldout = records[:20], records[20:]
    threshold = min(.99, max(.70, max(r["negative_cosine"] for r in calibration) + .03))
    report = {"scope": "COCO hardest-colour within-image negative and augmented-positive proxy, NOT re-entry ground truth",
              "split": "first 20 qualifying image IDs calibration, next 20 held out; no image shared",
              "configuration": asdict(config), "calibrated_match_threshold": threshold,
              "calibration_false_matches": sum(r["negative_cosine"] >= threshold for r in calibration),
              "heldout_false_matches": sum(r["negative_cosine"] >= threshold for r in heldout),
              "heldout_augmented_positive_accepts": sum(r["augmented_positive_cosine"] >= threshold for r in heldout),
              "heldout_pairs": len(heldout), "batch3_latency_median_ms": float(np.median(latencies)),
              "batch3_latency_p95_ms": float(np.percentile(latencies, 95)), "device": encoder.device,
              "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(report, out, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    run()
