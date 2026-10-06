"""Validate the downloaded fallen-person dataset and derive one-class YOLO labels.

The source export has four pose labels.  The NIDAR perception pipeline detects a
visible human body first, so every source pose maps to one `person_candidate`
class.  Source images and labels are preserved unchanged.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

from PIL import Image
import yaml

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "datasets" / "fallen_person"
SOURCE = ROOT / "roboflow_v2"
DERIVED = ROOT / "person_candidate"
ARCHIVE = ROOT / "archives" / "fallen-person-roboflow-v2-yolov11.zip"
EXPECTED_SPLITS = {"train": 2000, "valid": 576, "test": 300}
EXPECTED_CLASSES = ["fallen", "lying", "sitting", "standing"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_group(name: str) -> str:
    """Recover the coarse source sequence name for leakage reporting."""
    match = re.match(r"^(.+?)_[0-9]+_(?:png|jpg)\.rf\.", Path(name).stem)
    if match:
        return match.group(1)
    match = re.match(r"^(.+?)\.rf\.", Path(name).stem)
    return match.group(1) if match else Path(name).stem


def parse_label(path: Path) -> list[tuple[int, tuple[float, float, float, float]]]:
    parsed = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{path}:{number}: expected class plus four box values")
        try:
            class_id = int(fields[0])
            box = tuple(float(value) for value in fields[1:])
        except ValueError as exc:
            raise ValueError(f"{path}:{number}: non-numeric YOLO label") from exc
        if class_id not in range(len(EXPECTED_CLASSES)):
            raise ValueError(f"{path}:{number}: unexpected class {class_id}")
        if not all(0 <= value <= 1 for value in box) or box[2] <= 0 or box[3] <= 0:
            raise ValueError(f"{path}:{number}: invalid normalized box")
        parsed.append((class_id, box))
    return parsed


def link_or_verify(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size != source.stat().st_size:
            raise ValueError(f"Existing derived image differs: {destination}")
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def write_or_verify(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") != content:
        raise ValueError(f"Refusing to overwrite different derived content: {path}")
    path.write_text(content, encoding="utf-8")


def prepare() -> dict:
    if not ARCHIVE.is_file() or not SOURCE.is_dir():
        raise FileNotFoundError("Download and extract the Roboflow YOLO11 archive first.")
    metadata = yaml.safe_load((SOURCE / "data.yaml").read_text(encoding="utf-8"))
    if metadata.get("names") != EXPECTED_CLASSES or metadata.get("nc") != 4:
        raise ValueError("Source class schema differs from the reviewed four-pose export.")

    class_counts: Counter = Counter()
    split_class_counts: dict[str, Counter] = defaultdict(Counter)
    split_boxes: Counter = Counter()
    groups_by_split: dict[str, set[str]] = defaultdict(set)
    decoded = 0
    for split, expected in EXPECTED_SPLITS.items():
        image_dir, label_dir = SOURCE / split / "images", SOURCE / split / "labels"
        images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
        labels = {path.stem: path for path in label_dir.glob("*.txt")}
        if len(images) != expected or len(labels) != expected:
            raise ValueError(f"{split}: expected {expected} image-label pairs")
        if {path.stem for path in images} != set(labels):
            raise ValueError(f"{split}: image/label filenames do not match")
        for image_path in images:
            with Image.open(image_path) as image:
                image.load()
                if image.size != (640, 640):
                    raise ValueError(f"Unexpected image size: {image_path} -> {image.size}")
            decoded += 1
            groups_by_split[split].add(source_group(image_path.name))
            boxes = parse_label(labels[image_path.stem])
            class_counts.update(class_id for class_id, _ in boxes)
            split_class_counts[split].update(class_id for class_id, _ in boxes)
            split_boxes[split] += len(boxes)
            output = "\n".join("0 " + " ".join(f"{value:.8f}" for value in box) for _, box in boxes)
            if output:
                output += "\n"
            link_or_verify(image_path, DERIVED / "images" / split / image_path.name)
            write_or_verify(DERIVED / "labels" / split / f"{image_path.stem}.txt", output)

    overlaps = {}
    for left in EXPECTED_SPLITS:
        for right in EXPECTED_SPLITS:
            if left < right:
                shared = sorted(groups_by_split[left] & groups_by_split[right])
                if shared:
                    overlaps[f"{left}:{right}"] = shared

    config = {
        "path": str(DERIVED.resolve()),
        "train": "images/train",
        "val": "images/valid",
        "test": "images/test",
        "names": {0: "person_candidate"},
    }
    write_or_verify(DERIVED / "data.yaml", yaml.safe_dump(config, sort_keys=False))
    manifest = {
        "dataset": "Fallen Person Roboflow export v2, derived one-class view",
        "prepared_utc": datetime.now(timezone.utc).isoformat(),
        "source_url": metadata["roboflow"]["url"],
        "source_license_claim": metadata["roboflow"]["license"],
        "source_provider_note": "Roboflow user export; provenance should be audited before external redistribution",
        "archive": {"file": ARCHIVE.name, "bytes": ARCHIVE.stat().st_size, "sha256": sha256(ARCHIVE)},
        "images_decoded": decoded,
        "split_images": EXPECTED_SPLITS,
        "split_boxes": dict(split_boxes),
        "source_class_boxes": {EXPECTED_CLASSES[key]: class_counts[key] for key in range(4)},
        "source_class_boxes_by_split": {
            split: {EXPECTED_CLASSES[key]: counts[key] for key in range(4)}
            for split, counts in split_class_counts.items()
        },
        "derived_mapping": {name: "person_candidate" for name in EXPECTED_CLASSES},
        "source_groups_by_split": {key: sorted(value) for key, value in groups_by_split.items()},
        "coarse_group_overlap": overlaps,
        "evaluation_warning": "Roboflow split shares coarse source groups; do not report it as independent field accuracy",
        "medical_status": "No dead/alive/injured inference. Label means visible human-body candidate.",
    }
    (DERIVED / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


if __name__ == "__main__":
    prepare()
