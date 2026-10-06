"""Reproducibly fine-tune YOLO11n for visible people in varied body poses.

This creates candidate weights; promotion into the live default is a separate,
measured decision. It never interprets posture as alive/dead/injured status.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import torch
from ultralytics import YOLO

from .detector import COCO_MODEL

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT / "datasets" / "fallen_person" / "person_candidate" / "data.yaml"
DEFAULT_PROJECT = PROJECT / "runs" / "training"


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--base-model", type=Path, default=COCO_MODEL)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--name", default="fallen-person-yolo11n")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", choices=("0", "cpu"), default="0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if not args.data.is_file() or not args.base_model.is_file():
        parser.error("Prepared data.yaml and local base-model weights are required.")
    if not 1 <= args.epochs <= 300 or not 1 <= args.batch <= 32 or not 0 <= args.workers <= 8:
        parser.error("Use epochs 1-300, batch 1-32 and workers 0-8.")
    if args.device == "0" and not torch.cuda.is_available():
        parser.error("CUDA device 0 requested but unavailable.")
    return args


def main(argv: list[str] | None = None) -> int:
    args = arguments(argv)
    run_dir = args.project / args.name
    if run_dir.exists() and not args.resume:
        raise FileExistsError(f"Training run already exists: {run_dir}. Use a new --name or --resume.")
    model_path = run_dir / "weights" / "last.pt" if args.resume else args.base_model
    if args.resume and not model_path.is_file():
        raise FileNotFoundError(f"Cannot resume because last.pt is missing: {model_path}")
    model = YOLO(str(model_path), task="detect")
    settings = {
        "data": str(args.data.resolve()), "epochs": args.epochs, "imgsz": 640,
        "batch": args.batch, "device": args.device, "workers": args.workers,
        "project": str(args.project.resolve()), "name": args.name,
        # We already reject accidental reuse above. True prevents Ultralytics from
        # silently changing our reviewed output name after metadata is written.
        "exist_ok": True, "resume": args.resume,
        # The prepared dataset already has exactly one class. Keeping single_cls
        # disabled preserves its reviewed `person_candidate` name in the weights.
        "seed": 42, "deterministic": True, "single_cls": False,
        "patience": 10, "close_mosaic": 5, "cos_lr": True,
        "degrees": 20.0, "translate": 0.1, "scale": 0.5,
        "fliplr": 0.5, "flipud": 0.1, "mosaic": 1.0,
        "amp": True, "cache": False, "plots": True, "verbose": True,
    }
    metadata = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "visible person/body candidate detection; no medical-state inference",
        "base_model": str(args.base_model.resolve()), "settings": settings,
    }
    run_dir.mkdir(parents=True, exist_ok=args.resume)
    (run_dir / "nidar_training_request.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    model.train(**settings)
    best = run_dir / "weights" / "best.pt"
    if not best.is_file():
        raise RuntimeError("Training finished without best.pt")
    deployed_candidate = run_dir / "weights" / "best-person-candidate.pt"
    stamp_person_candidate(best, deployed_candidate)
    metadata.update({"finished_utc": datetime.now(timezone.utc).isoformat(),
                     "best_model": str(deployed_candidate.resolve()),
                     "best_model_bytes": deployed_candidate.stat().st_size})
    (run_dir / "nidar_training_result.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)
    return 0


def stamp_person_candidate(source: Path, destination: Path) -> None:
    """Give a verified one-class checkpoint its explicit runtime semantic name."""
    model = YOLO(str(source), task="detect")
    if model.task != "detect" or len(model.names) != 1 or 0 not in model.names:
        raise ValueError("Only a one-class detection checkpoint can be finalized.")
    model.model.names = {0: "person_candidate"}
    destination.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(destination))
    check = YOLO(str(destination), task="detect")
    if check.names != {0: "person_candidate"}:
        destination.unlink(missing_ok=True)
        raise RuntimeError("Final checkpoint did not preserve person_candidate schema.")


if __name__ == "__main__":
    raise SystemExit(main())
