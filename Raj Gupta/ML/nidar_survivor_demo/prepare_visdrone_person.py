"""Download and convert VisDrone DET into one-class NIDAR person candidates.

Only VisDrone categories ``pedestrian`` and ``people`` become class 0.  Ignored
regions and all vehicle classes are intentionally not converted into positives.
The official train/val/test-dev boundaries are preserved.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile

from PIL import Image
import yaml


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT / "datasets" / "visdrone_person"
ASSET_BASE = "https://github.com/ultralytics/assets/releases/download/v0.0.0"
ARCHIVES = {
    "train": "VisDrone2019-DET-train.zip",
    "valid": "VisDrone2019-DET-val.zip",
    "test": "VisDrone2019-DET-test-dev.zip",
}
SOURCE_FOLDERS = {
    "train": "VisDrone2019-DET-train",
    "valid": "VisDrone2019-DET-val",
    "test": "VisDrone2019-DET-test-dev",
}
PERSON_CATEGORY_IDS = frozenset({1, 2})  # 1 pedestrian, 2 people/other posture.


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size:
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "NIDAR-dataset-preparer/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def convert_split(source: Path, output: Path, split: str) -> dict[str, int]:
    images_source = source / "images"
    labels_source = source / "annotations"
    images_output = output / "images" / split
    labels_output = output / "labels" / split
    images_output.mkdir(parents=True, exist_ok=True)
    labels_output.mkdir(parents=True, exist_ok=True)
    counts = {"images": 0, "positive_images": 0, "negative_images": 0, "person_boxes": 0}
    for image_path in sorted(images_source.glob("*.jpg")):
        label_path = labels_source / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing VisDrone annotation: {label_path}")
        width, height = Image.open(image_path).size
        rows: list[str] = []
        for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
            fields = line.split(",")
            if len(fields) < 6:
                raise ValueError(f"Malformed annotation {label_path}:{line_number}")
            x, y, box_width, box_height = map(int, fields[:4])
            score, category = int(fields[4]), int(fields[5])
            if score == 0 or category not in PERSON_CATEGORY_IDS:
                continue
            if box_width <= 0 or box_height <= 0:
                continue
            center_x = (x + box_width / 2) / width
            center_y = (y + box_height / 2) / height
            rows.append(
                f"0 {center_x:.6f} {center_y:.6f} "
                f"{box_width / width:.6f} {box_height / height:.6f}\n"
            )
        shutil.copy2(image_path, images_output / image_path.name)
        (labels_output / f"{image_path.stem}.txt").write_text("".join(rows), encoding="utf-8")
        counts["images"] += 1
        counts["person_boxes"] += len(rows)
        counts["positive_images" if rows else "negative_images"] += 1
    return counts


def prepare(output: Path = DEFAULT_OUTPUT) -> dict:
    archives = output / "archives"
    extracted = output / "source"
    archive_metadata: dict[str, dict[str, str | int]] = {}
    split_counts: dict[str, dict[str, int]] = {}
    for split, filename in ARCHIVES.items():
        archive = archives / filename
        download(f"{ASSET_BASE}/{filename}", archive)
        source = extracted / SOURCE_FOLDERS[split]
        if not source.is_dir():
            extracted.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive) as bundle:
                bundle.extractall(extracted)
        split_counts[split] = convert_split(source, output, split)
        archive_metadata[split] = {
            "url": f"{ASSET_BASE}/{filename}",
            "filename": filename,
            "bytes": archive.stat().st_size,
            "sha256": sha256(archive),
        }
    config = {
        "path": str(output.resolve()),
        "train": "images/train",
        "val": "images/valid",
        "test": "images/test",
        "names": {0: "person_candidate"},
    }
    (output / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": "VisDrone2019-DET by the AISKYEYE team, Tianjin University",
        "source_page": "https://github.com/VisDrone/VisDrone-Dataset",
        "download_documentation": "https://docs.ultralytics.com/datasets/detect/visdrone/",
        "class_mapping": {"1 pedestrian": "person_candidate", "2 people": "person_candidate"},
        "medical_status": "No alive/dead/injured inference is present.",
        "split_policy": "Official train/val/test-dev boundaries preserved.",
        "license_status": "REVIEW-REQUIRED: upstream repository provides citation but no clear reusable-data licence file was verified.",
        "use_policy": "Research evaluation/training only until the team verifies organiser and upstream data terms.",
        "archives": archive_metadata,
        "counts": split_counts,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    prepare(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
