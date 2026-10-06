"""Train a promotion candidate for NIDAR visible person/body detection."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import torch
from ultralytics import YOLO

from .detector import COCO_MODEL
from .train_fallen_person import stamp_person_candidate


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT / "datasets" / "nidar_person_v1" / "data.yaml"
DEFAULT_PROJECT = PROJECT / "runs" / "training"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--base-model", type=Path, default=COCO_MODEL)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--name", default="nidar-person-yolo11n-v4")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", choices=("0", "cpu"), default="0")
    parser.add_argument(
        "--profile",
        choices=("standard", "recovery"),
        default="standard",
        help="Recovery uses a frozen backbone and low learning rate after domain training.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the same named run from weights/last.pt.",
    )
    args = parser.parse_args(argv)
    if not args.data.is_file() or not args.base_model.is_file():
        parser.error("Prepared NIDAR data.yaml and local base weights are required.")
    if not 1 <= args.epochs <= 300 or not 1 <= args.batch <= 32 or not 0 <= args.workers <= 8:
        parser.error("Use epochs 1-300, batch 1-32 and workers 0-8.")
    if args.device == "0" and not torch.cuda.is_available():
        parser.error("CUDA device 0 requested but unavailable.")
    run_dir = args.project / args.name
    request_path = run_dir / "nidar_training_request.json"
    result_path = run_dir / "nidar_training_result.json"
    last = run_dir / "weights" / "last.pt"
    if args.resume:
        if not run_dir.is_dir() or not request_path.is_file() or not last.is_file():
            parser.error("Resume requires an interrupted run with metadata and weights/last.pt.")
        if result_path.exists():
            parser.error("This run already completed; it must not be resumed.")
        try:
            request = json.loads(request_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            parser.error("The existing training request metadata is unreadable.")
        settings = request.get("settings", {})
        if Path(request.get("dataset", "")).resolve() != args.data.resolve():
            parser.error("Resume data.yaml differs from the original training request.")
        expected = {
            "epochs": args.epochs,
            "batch": args.batch,
            "workers": args.workers,
            "device": args.device,
            "project": str(args.project.resolve()),
            "name": args.name,
        }
        mismatches = [key for key, value in expected.items() if settings.get(key) != value]
        if mismatches:
            parser.error("Resume settings differ for: " + ", ".join(mismatches))
    elif run_dir.exists():
        parser.error("Training run already exists; choose a new --name or use --resume.")
    return args


def main(argv: list[str] | None = None) -> int:
    args = arguments(argv)
    run_dir = args.project / args.name
    request_path = run_dir / "nidar_training_request.json"
    result_path = run_dir / "nidar_training_result.json"
    settings = {
        "data": str(args.data.resolve()), "epochs": args.epochs, "imgsz": 640,
        "batch": args.batch, "device": args.device, "workers": args.workers,
        # The reviewed metadata directory is created immediately below.  True
        # prevents Ultralytics from silently changing the run name to ``-2``;
        # the explicit existence check above still protects prior runs.
        "project": str(args.project.resolve()), "name": args.name, "exist_ok": True,
        "seed": 42, "deterministic": True, "single_cls": False,
        "patience": 12, "close_mosaic": 8, "cos_lr": True,
        "degrees": 25.0, "translate": 0.12, "scale": 0.55,
        "fliplr": 0.5, "flipud": 0.15, "mosaic": 1.0,
        "amp": True, "cache": False, "plots": True, "verbose": True,
    }
    if args.profile == "recovery":
        settings.update({
            "optimizer": "AdamW", "lr0": 0.0005, "lrf": 0.10,
            "warmup_epochs": 1.0, "freeze": 10, "patience": 6,
            "close_mosaic": 2, "degrees": 15.0, "translate": 0.08,
            "scale": 0.35, "flipud": 0.05, "mosaic": 0.5,
        })
    if args.resume:
        metadata = json.loads(request_path.read_text(encoding="utf-8"))
        metadata.setdefault("resume_events_utc", []).append(datetime.now(timezone.utc).isoformat())
        request_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        YOLO(str(run_dir / "weights" / "last.pt"), task="detect").train(resume=True)
    else:
        metadata = {
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": "visible person/body candidate; possible-fallen evidence only, no death/liveness inference",
            "training_profile": args.profile,
            "base_model": str(args.base_model.resolve()),
            "base_model_sha256": file_sha256(args.base_model),
            "dataset": str(args.data.resolve()),
            "settings": settings,
            "promotion_status": "candidate_only",
        }
        run_dir.mkdir(parents=True)
        request_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        YOLO(str(args.base_model), task="detect").train(**settings)
    best = run_dir / "weights" / "best.pt"
    if not best.is_file():
        raise RuntimeError("Training finished without best.pt")
    candidate = run_dir / "weights" / "best-person-candidate.pt"
    stamp_person_candidate(best, candidate)
    metadata.update({
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "best_model": str(candidate.resolve()),
        "best_model_bytes": candidate.stat().st_size,
        "best_model_sha256": file_sha256(candidate),
    })
    result_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
