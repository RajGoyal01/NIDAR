"""Prepare, train and evaluate the full VisDrone NIDAR research candidate.

The pipeline deliberately does not replace the live demo model.  It produces a
candidate, evaluates the unchanged held-out split and applies the public-data
engineering gate.  Final promotion still requires licence review and project-
owned OAK-D field evidence.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import yaml

from .detector import COCO_MODEL
from .prepare_nidar_person import prepare as prepare_mixture


PROJECT = Path(__file__).resolve().parents[1]
VISDRONE_MANIFEST = PROJECT / "datasets" / "visdrone_person" / "manifest.json"
DEFAULT_DATASET = PROJECT / "datasets" / "nidar_person_v4_full_visdrone"
DEFAULT_RUNS = PROJECT / "runs" / "training"
DEFAULT_LOGS = PROJECT / "logs"
DEFAULT_NAME = "nidar-person-yolo11n-v8-full-visdrone"
EXPECTED_VISDRONE_IMAGES = {"train": 6471, "valid": 548, "test": 1610}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Missing or invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def validate_source_manifest(manifest: dict) -> None:
    """Require the complete official DET train/val/test-dev conversion."""
    if manifest.get("class_mapping") != {
        "1 pedestrian": "person_candidate",
        "2 people": "person_candidate",
    }:
        raise ValueError("VisDrone person class mapping is missing or changed.")
    counts = manifest.get("counts", {})
    for split, expected in EXPECTED_VISDRONE_IMAGES.items():
        if counts.get(split, {}).get("images") != expected:
            raise ValueError(f"VisDrone {split} must contain exactly {expected} images.")
    if manifest.get("split_policy") != "Official train/val/test-dev boundaries preserved.":
        raise ValueError("VisDrone official split policy is not recorded.")


def validate_mixture_manifest(manifest: dict) -> None:
    """Prove the unified dataset includes every converted VisDrone DET image."""
    if manifest.get("class") != "person_candidate":
        raise ValueError("Unified dataset must have one person_candidate class.")
    if manifest.get("visdrone_train_limit") != 0 or manifest.get("visdrone_valid_limit") != 0:
        raise ValueError("Full integration requires zero limits (all VisDrone images).")
    counts = manifest.get("counts", {})
    for split, expected in EXPECTED_VISDRONE_IMAGES.items():
        if counts.get(split, {}).get("visdrone_images") != expected:
            raise ValueError(f"Unified {split} split does not include all {expected} VisDrone images.")


def _rewrite_dataset_path(dataset: Path) -> None:
    config_path = dataset / "data.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["path"] = str(dataset.resolve())
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def prepare_full_dataset(output: Path = DEFAULT_DATASET) -> dict:
    """Create the full mixture through a staging directory, then validate it."""
    source_manifest = load_json(VISDRONE_MANIFEST)
    validate_source_manifest(source_manifest)
    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        manifest = load_json(manifest_path)
        validate_mixture_manifest(manifest)
        return manifest
    if output.exists():
        raise FileExistsError(f"Incomplete final dataset exists; inspect it manually: {output}")
    staging = output.with_name(output.name + ".building")
    if staging.exists():
        raise FileExistsError(f"Previous staging dataset needs inspection: {staging}")
    manifest = prepare_mixture(staging, visdrone_train_limit=0, visdrone_valid_limit=0)
    validate_mixture_manifest(manifest)
    staging.replace(output)
    _rewrite_dataset_path(output)
    manifest = load_json(output / "manifest.json")
    manifest["full_visdrone_integration"] = {
        "status": "COMPLETE",
        "source_manifest": str(VISDRONE_MANIFEST.resolve()),
        "source_manifest_sha256": sha256(VISDRONE_MANIFEST),
        "all_official_det_splits": True,
        "deployment_policy": "Research-only pending exact VisDrone2019 terms and OAK-D field gate.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def run_module(
    module: str,
    arguments: list[str],
    *,
    allowed: frozenset[int] = frozenset({0}),
) -> int:
    result = subprocess.run([sys.executable, "-m", module, *arguments], cwd=PROJECT)
    if result.returncode not in allowed:
        raise RuntimeError(f"{module} failed with exit code {result.returncode}")
    return result.returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--project", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--logs", type=Path, default=DEFAULT_LOGS)
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--base-model", type=Path, default=COCO_MODEL)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", choices=("0", "cpu"), default="0")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--resume-training",
        action="store_true",
        help="Resume this named run from weights/last.pt after an interruption.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 1 <= args.epochs <= 300 or not 1 <= args.batch <= 32 or not 0 <= args.workers <= 8:
        raise SystemExit("Use epochs 1-300, batch 1-32 and workers 0-8.")
    dataset = args.dataset.resolve()
    manifest = prepare_full_dataset(dataset)
    print(json.dumps({
        "event": "full_visdrone_dataset_ready",
        "dataset": str(dataset),
        "visdrone_images": EXPECTED_VISDRONE_IMAGES,
        "train_total_images": sum(
            manifest["counts"]["train"][key]
            for key in ("fallen_images", "coco_images", "visdrone_images")
        ),
    }), flush=True)
    if args.prepare_only:
        return 0

    run_dir = args.project.resolve() / args.name
    report = args.logs.resolve() / f"{args.name}-test.json"
    gate = args.logs.resolve() / f"{args.name}-gate.json"
    status_path = args.logs.resolve() / f"{args.name}-pipeline.json"
    training_arguments = [
        "--data", str(dataset / "data.yaml"), "--base-model", str(args.base_model.resolve()),
        "--project", str(args.project.resolve()), "--name", args.name,
        "--epochs", str(args.epochs), "--batch", str(args.batch),
        "--workers", str(args.workers), "--device", args.device,
    ]
    if args.resume_training:
        training_arguments.append("--resume")
    run_module("nidar_survivor_demo.train_nidar_detector", training_arguments)
    candidate = run_dir / "weights" / "best-person-candidate.pt"
    run_module("nidar_survivor_demo.evaluate_nidar_detector", [
        "--model", str(candidate), "--dataset", str(dataset),
        "--output", str(report),
    ])
    gate_code = run_module("nidar_survivor_demo.gate_nidar_detector", [
        "--baseline", str(args.logs.resolve() / "nidar-person-baseline-test.json"),
        "--candidate", str(report), "--output", str(gate),
    ], allowed={0, 1})
    gate_report = load_json(gate)
    status = {
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "status": gate_report["status"],
        "candidate": str(candidate.resolve()),
        "candidate_sha256": sha256(candidate),
        "dataset": str(dataset),
        "evaluation": str(report),
        "gate": str(gate),
        "gate_exit_code": gate_code,
        "runtime_default_changed": False,
        "promotion_blockers": [
            "Exact VisDrone2019 reuse terms must be approved for intended use.",
            "Project-owned OAK-D indoor locked-field evaluation is still required.",
            "Edge export and numerical/runtime comparison are still required.",
        ],
    }
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps({"event": "full_visdrone_pipeline_complete", **status}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
