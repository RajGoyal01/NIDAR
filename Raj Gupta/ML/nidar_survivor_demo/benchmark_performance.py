"""Repeatable camera-free Phase 8 timings; not an accuracy benchmark."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from .detector import DetectorConfig, PersonDetector
from .tracking import TrackerConfig, PersonTracker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    frame = cv2.imread(str(root / "datasets/coco2017/images/val2017/000000018150.jpg"))
    if frame is None:
        raise ValueError("Installed COCO test image missing")
    frame = cv2.resize(frame, (1280, 720))
    results = []
    original = {"torch_threads": torch.get_num_threads(), "opencv_threads": cv2.getNumThreads()}
    for threads in (original["torch_threads"], 1, 2, 4):
        torch.set_num_threads(threads)
        cv2.setNumThreads(threads)
        for size in (640, 512):
            detector = PersonDetector(DetectorConfig(imgsz=size, confidence=.1))
            tracker = PersonTracker(TrackerConfig())
            infer, track = [], []
            for i in range(35):
                found = detector.predict(frame)
                tracked = tracker.update(found, frame, sequence=i + 1, received_at=i * .1, connection=1)
                if i >= 5:
                    infer.append(found.inference_ms)
                    track.append(tracked.tracking_ms)
            row = {"threads": threads, "imgsz": size, "inference_median_ms": float(np.median(infer)),
                   "inference_p95_ms": float(np.percentile(infer, 95)), "tracking_median_ms": float(np.median(track)),
                   "tracking_p95_ms": float(np.percentile(track, 95)), "visible_tracks": len(tracked.tracks)}
            results.append(row)
            print(json.dumps(row), flush=True)
    report = {"scope": "Static COCO image timing only; not detection/re-ID accuracy or live latency", "original": original, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(report, out, indent=2)


if __name__ == "__main__":
    main()
