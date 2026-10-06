"""Run the COCO recovery curriculum, held-out evaluation and unchanged gate."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import yaml

from .full_visdrone_pipeline import load_json, run_module, sha256
from .prepare_detector_recovery import DEFAULT_OUTPUT, DEFAULT_SOURCE, prepare


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_MODEL = (
    PROJECT / "runs" / "training" / "nidar-person-yolo11n-v8-full-visdrone"
    / "weights" / "best-person-candidate.pt"
)
DEFAULT_RUNS = PROJECT / "runs" / "training"
DEFAULT_LOGS = PROJECT / "logs"
DEFAULT_NAME = "nidar-person-yolo11n-v9-coco-recovery"


def validate_recovery_manifest(manifest: dict, coco_repeats: int = 4) -> None:
    counts = manifest.get("counts", {})
    if manifest.get("class") != "person_candidate":
        raise ValueError("Recovery dataset must remain one-class person_candidate.")
    if manifest.get("sampling_policy", {}).get("coco_repeats") != coco_repeats:
        raise ValueError("Recovery COCO sampling weight is not the reviewed value.")
    if counts.get("train", {}).get("coco_images") != 2859 * coco_repeats:
        raise ValueError("Recovery dataset does not contain every reviewed COCO replay.")
    if counts.get("train", {}).get("fallen_images") != 2021:
        raise ValueError("Recovery dataset must retain every fallen training image.")
    expected_test = {"fallen_images": 447, "coco_images": 899, "visdrone_images": 1610}
    for key, value in expected_test.items():
        if counts.get("test", {}).get(key) != value:
            raise ValueError(f"Recovery test must keep {value} {key}.")


def _rewrite_path(dataset: Path) -> None:
    config_path = dataset / "data.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["path"] = str(dataset.resolve())
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def prepare_recovery_dataset(output: Path) -> dict:
    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        manifest = load_json(manifest_path)
        validate_recovery_manifest(manifest)
        return manifest
    if output.exists():
        raise FileExistsError(f"Incomplete recovery dataset needs inspection: {output}")
    staging = output.with_name(output.name + ".building")
    if staging.exists():
        raise FileExistsError(f"Previous recovery staging directory needs inspection: {staging}")
    manifest = prepare(staging, DEFAULT_SOURCE, coco_repeats=4, aerial_box_budget=6000)
    validate_recovery_manifest(manifest)
    staging.replace(output)
    _rewrite_path(output)
    manifest = load_json(output / "manifest.json")
    manifest["integration"] = {
        "status": "COMPLETE",
        "strategy": "COCO replay + all fallen + deterministic aerial reminder",
        "full_visdrone_predecessor": str(DEFAULT_BASE_MODEL.resolve()),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    result.add_argument("--project", type=Path, default=DEFAULT_RUNS)
    result.add_argument("--logs", type=Path, default=DEFAULT_LOGS)
    result.add_argument("--name", default=DEFAULT_NAME)
    result.add_argument("--epochs", type=int, default=8)
    result.add_argument("--batch", type=int, default=16)
    result.add_argument("--workers", type=int, default=2)
    result.add_argument("--device", choices=("0", "cpu"), default="0")
    result.add_argument("--prepare-only", action="store_true")
    result.add_argument("--resume-training", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if not args.base_model.is_file():
        raise SystemExit(f"Reviewed v8 predecessor is missing: {args.base_model}")
    dataset = args.dataset.resolve()
    manifest = prepare_recovery_dataset(dataset)
    print(json.dumps({"event": "recovery_dataset_ready", "counts": manifest["counts"]}), flush=True)
    if args.prepare_only:
        return 0

    run_dir = args.project.resolve() / args.name
    report = args.logs.resolve() / f"{args.name}-test.json"
    gate = args.logs.resolve() / f"{args.name}-gate.json"
    status_path = args.logs.resolve() / f"{args.name}-pipeline.json"
    training_args = [
        "--data", str(dataset / "data.yaml"), "--base-model", str(args.base_model.resolve()),
        "--project", str(args.project.resolve()), "--name", args.name,
        "--epochs", str(args.epochs), "--batch", str(args.batch),
        "--workers", str(args.workers), "--device", args.device, "--profile", "recovery",
    ]
    if args.resume_training:
        training_args.append("--resume")
    run_module("nidar_survivor_demo.train_nidar_detector", training_args)
    candidate = run_dir / "weights" / "best-person-candidate.pt"
    run_module("nidar_survivor_demo.evaluate_nidar_detector", [
        "--model", str(candidate), "--dataset", str(DEFAULT_SOURCE), "--output", str(report),
    ])
    gate_code = run_module("nidar_survivor_demo.gate_nidar_detector", [
        "--baseline", str(args.logs.resolve() / "nidar-person-baseline-test.json"),
        "--candidate", str(report), "--output", str(gate),
    ], allowed=frozenset({0, 1}))
    gate_report = load_json(gate)
    status = {
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "status": gate_report["status"], "passed": gate_report["passed"],
        "candidate": str(candidate.resolve()), "candidate_sha256": sha256(candidate),
        "predecessor": str(args.base_model.resolve()), "dataset": str(dataset),
        "evaluation": str(report), "gate": str(gate), "gate_exit_code": gate_code,
        "runtime_default_changed": False,
        "promotion_blockers": [
            "Project-owned OAK-D indoor locked-field evaluation is required.",
            "Edge export and numerical/runtime comparison are required.",
            "VisDrone2019 intended-use terms remain review-required.",
        ],
    }
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps({"event": "recovery_pipeline_complete", **status}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
