"""Apply explicit engineering promotion gates to a NIDAR detector report."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def build_gate_report(baseline: dict, candidate: dict) -> dict:
    """Compare one candidate with the fixed COCO baseline.

    These thresholds are project engineering decisions, not official NIDAR rules.
    Passing them still does not replace the missing OAK-D field evaluation.
    """
    base_sources = baseline["by_source"]
    sources = candidate["by_source"]
    poses = candidate["fallen_pose_recall"]
    sizes = candidate["by_size"]
    checks = {
        "class_schema": candidate["model_class"] == "person_candidate",
        "coco_recall": sources["coco"]["recall"] >= 0.65,
        "coco_precision": sources["coco"]["precision"] >= 0.75,
        "coco_negative_fp_images": sources["coco"].get("negative_images_with_fp", 0)
        <= base_sources["coco"].get("negative_images_with_fp", 0) + 20,
        "fallen_recall": sources["fallen"]["recall"] >= 0.85,
        "lying_recall": poses["lying"] >= 0.65,
        "large_recall": sizes["large"]["recall"] >= 0.75,
        "small_recall_improvement": sizes["small"]["recall"]
        >= baseline["by_size"]["small"]["recall"] * 2.0,
        "aerial_recall_improvement": sources["visdrone"]["recall"]
        >= base_sources["visdrone"]["recall"] * 2.0,
        "p95_latency_ms": candidate["inference_p95_ms"] <= 70.0,
    }
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ENGINEERING-GATE-PASS" if all(checks.values()) else "REJECTED-NOT-PROMOTED",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "scope": "Public-data laptop engineering gate only; OAK-D field acceptance remains OPEN.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    result = build_gate_report(baseline, candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "ENGINEERING-GATE-PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
