"""Synthetic labelled motion regression for BoT-SORT; not real-world ID accuracy."""
import json
from pathlib import Path
import cv2
import numpy as np
from .detector import Detection, DetectionResult
from .tracking import PersonTracker, TrackerConfig
from .benchmark_detection import overlap


def run_scenario(name: str) -> dict:
    tracker = PersonTracker(TrackerConfig())
    rng = np.random.default_rng(42)
    background = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    last_ids = {}
    switches = matched = expected = 0
    times = []
    for step in range(80):
        pan = step * 2 if name == "camera_pan" else 0
        frame = cv2.warpAffine(background, np.float32([[1, 0, pan], [0, 1, 0]]), (640, 480))
        truth = {}
        for person in range(4):
            if name == "short_occlusion" and person == 0 and 25 <= step <= 28:
                continue
            if name == "crossing":
                x = 70 + step * 4 if person % 2 == 0 else 440 - step * 4
            else:
                x = 40 + person * 110 + step + pan
            y = 30 + person * 80
            truth[person] = (x, y, x + 45, y + 100)
        # Known detector inputs isolate association correctness from YOLO misses.
        detections = DetectionResult(tuple(Detection(box, .9) for box in truth.values()), 0)
        result = tracker.update(detections, frame, sequence=1 + step * 2,
                                received_at=step * .05, connection=1)
        times.append(result.tracking_ms)
        expected += len(truth)
        unused = set(range(len(result.tracks)))
        for person, box in truth.items():
            best = max(unused, key=lambda i: overlap(box, result.tracks[i].xyxy), default=None)
            if best is None or overlap(box, result.tracks[best].xyxy) < .5:
                continue
            unused.remove(best)
            label = result.tracks[best].track_id
            switches += int(person in last_ids and last_ids[person] != label)
            last_ids[person] = label
            matched += 1
    return {"scenario": name, "frames": 80, "expected_observations": expected,
            "matched_observations": matched, "coverage": matched / expected,
            "id_switches": switches, "tracking_median_ms": float(np.median(times)),
            "tracking_p95_ms": float(np.percentile(times, 95))}


def main():
    results = [run_scenario(name) for name in ("steady_motion", "crossing", "short_occlusion", "camera_pan")]
    report = {"scope": "Synthetic oracle detections, no YOLO, no real-world accuracy claim", "results": results}
    path = Path("logs/phase3_tracking_regression.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if any(row["id_switches"] or row["coverage"] < .95 for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
