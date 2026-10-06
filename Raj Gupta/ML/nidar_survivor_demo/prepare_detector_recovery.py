"""Build a COCO-heavy recovery curriculum from the accepted full mixture.

The full VisDrone stage taught the candidate aerial/small-person features but its
dense aerial annotations dominated ordinary COCO people and negatives.  This
curriculum intentionally replays COCO samples more often while retaining all
fallen-person training images and a deterministic aerial reminder subset.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import yaml

from .prepare_nidar_person import hardlink_or_copy


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT / "datasets" / "nidar_person_v4_full_visdrone"
DEFAULT_OUTPUT = PROJECT / "datasets" / "nidar_person_v5_coco_recovery"
SPLITS = ("train", "valid", "test")
EXPECTED_SOURCE_IMAGES = {
    "train": {"fallen": 2021, "coco": 2859, "visdrone": 6471},
    "valid": {"fallen": 408, "coco": 1015, "visdrone": 548},
    "test": {"fallen": 447, "coco": 899, "visdrone": 1610},
}


def label_box_count(path: Path) -> int:
    return sum(bool(line.strip()) for line in path.read_text(encoding="utf-8").splitlines())


def image_files(directory: Path, prefix: str) -> list[Path]:
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.name.startswith(f"{prefix}_")
        and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def deterministic_order(paths: list[Path]) -> list[Path]:
    return sorted(paths, key=lambda path: hashlib.sha256(path.name.encode()).hexdigest())


def select_aerial_reminders(paths: list[Path], labels: Path, box_budget: int) -> list[Path]:
    """Select reproducibly until the requested number of annotated boxes is covered."""
    if box_budget < 1:
        raise ValueError("Aerial box budget must be positive.")
    selected: list[Path] = []
    boxes = 0
    for image in deterministic_order(paths):
        selected.append(image)
        boxes += label_box_count(labels / f"{image.stem}.txt")
        if boxes >= box_budget:
            break
    if boxes < box_budget:
        raise ValueError(f"Only {boxes} aerial boxes are available; requested {box_budget}.")
    return selected


def _add(
    image: Path,
    source_labels: Path,
    output: Path,
    split: str,
    output_stem: str,
    source: str,
    counts: dict[str, Counter],
) -> None:
    label = source_labels / f"{image.stem}.txt"
    if not label.is_file():
        raise FileNotFoundError(f"Missing label for {image}")
    destination_image = output / "images" / split / f"{output_stem}{image.suffix.lower()}"
    destination_label = output / "labels" / split / f"{output_stem}.txt"
    hardlink_or_copy(image, destination_image)
    hardlink_or_copy(label, destination_label)
    boxes = label_box_count(label)
    counts[split][f"{source}_images"] += 1
    counts[split][f"{source}_boxes"] += boxes
    counts[split]["positive_images" if boxes else "negative_images"] += 1


def _validate_source(source: Path) -> None:
    if not (source / "manifest.json").is_file():
        raise FileNotFoundError(f"Full-mixture manifest is required: {source}")
    for split, expected in EXPECTED_SOURCE_IMAGES.items():
        for name, count in expected.items():
            actual = len(image_files(source / "images" / split, name))
            if actual != count:
                raise ValueError(f"Source {split}/{name} must contain {count} images, found {actual}.")


def prepare(
    output: Path = DEFAULT_OUTPUT,
    source: Path = DEFAULT_SOURCE,
    *,
    coco_repeats: int = 4,
    aerial_box_budget: int = 6000,
    validation_aerial_images: int = 100,
) -> dict:
    """Prepare recovery train/valid/test sets without changing held-out test data."""
    if not 1 <= coco_repeats <= 8:
        raise ValueError("COCO repeats must be between 1 and 8.")
    if not 1 <= validation_aerial_images <= EXPECTED_SOURCE_IMAGES["valid"]["visdrone"]:
        raise ValueError("Validation aerial image count is outside the available range.")
    source, output = source.resolve(), output.resolve()
    _validate_source(source)
    if output.exists() and any((output / "images" / split).exists() for split in SPLITS):
        raise FileExistsError(f"Output already contains data: {output}")
    counts: dict[str, Counter] = {split: Counter() for split in SPLITS}

    train_images = source / "images" / "train"
    train_labels = source / "labels" / "train"
    for image in image_files(train_images, "fallen"):
        _add(image, train_labels, output, "train", image.stem, "fallen", counts)
    coco_train = image_files(train_images, "coco")
    for repeat in range(coco_repeats):
        for image in coco_train:
            _add(image, train_labels, output, "train", f"coco_r{repeat}_{image.stem}", "coco", counts)
    aerial_train = select_aerial_reminders(
        image_files(train_images, "visdrone"), train_labels, aerial_box_budget
    )
    for image in aerial_train:
        _add(image, train_labels, output, "train", image.stem, "visdrone", counts)

    for split in ("valid", "test"):
        source_images = source / "images" / split
        source_labels = source / "labels" / split
        for name in ("fallen", "coco"):
            for image in image_files(source_images, name):
                _add(image, source_labels, output, split, image.stem, name, counts)
        aerial = image_files(source_images, "visdrone")
        if split == "valid":
            aerial = deterministic_order(aerial)[:validation_aerial_images]
        for image in aerial:
            _add(image, source_labels, output, split, image.stem, "visdrone", counts)

    output.mkdir(parents=True, exist_ok=True)
    config = {
        "path": str(output),
        "train": "images/train",
        "val": "images/valid",
        "test": "images/test",
        "names": {0: "person_candidate"},
    }
    (output / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "COCO precision/recall recovery after complete VisDrone training.",
        "class": "person_candidate",
        "source_dataset": str(source),
        "sampling_policy": {
            "coco_repeats": coco_repeats,
            "coco_repeat_interpretation": "Sampling weight only; repeated links are not new evidence.",
            "fallen_train": "all",
            "aerial_train_box_budget": aerial_box_budget,
            "aerial_train_selection": "SHA-256 filename order until box budget is reached",
            "validation_aerial_images": validation_aerial_images,
            "test": "unchanged complete source test split",
        },
        "leakage_controls": [
            "No source sample crosses its original train/valid/test boundary.",
            "COCO repetition occurs only inside the training split.",
            "The full held-out test split is unchanged.",
        ],
        "counts": {split: dict(values) for split, values in counts.items()},
        "known_gap": "Public replay cannot replace project-owned locked OAK-D field data.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest

