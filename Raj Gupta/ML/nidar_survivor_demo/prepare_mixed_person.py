"""Build a leakage-separated mixed body/person dataset using hard links."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil

import yaml

from .prepare_fallen_person import DERIVED as FALLEN, PROJECT

COCO = PROJECT / "datasets" / "coco2017"
OUTPUT = PROJECT / "datasets" / "person_mixed"


def bucket(name: str) -> int:
    return int(hashlib.sha256(name.encode("utf-8")).hexdigest()[:8], 16) % 100


def link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size != source.stat().st_size:
            raise ValueError(f"Existing mixed file differs: {destination}")
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def prepare() -> dict:
    if not FALLEN.is_dir() or not (COCO / "person_val.txt").is_file():
        raise FileNotFoundError("Prepared fallen-person and COCO datasets are required.")
    counts: dict[str, Counter] = {name: Counter() for name in ("train", "valid", "test")}
    for split in counts:
        for image in sorted((FALLEN / "images" / split).iterdir()):
            label = FALLEN / "labels" / split / f"{image.stem}.txt"
            link(image, OUTPUT / "images" / split / f"fallen_{image.name}")
            link(label, OUTPUT / "labels" / split / f"fallen_{image.stem}.txt")
            counts[split]["fallen_images"] += 1
            counts[split]["fallen_boxes"] += len(label.read_text(encoding="utf-8").splitlines())

    for relative in (COCO / "person_val.txt").read_text(encoding="utf-8").splitlines():
        image = COCO / relative
        label = COCO / "labels" / "val2017" / f"{image.stem}.txt"
        value = bucket(image.name)
        split = "train" if value < 60 else "valid" if value < 80 else "test"
        link(image, OUTPUT / "images" / split / f"coco_{image.name}")
        link(label, OUTPUT / "labels" / split / f"coco_{image.stem}.txt")
        boxes = len(label.read_text(encoding="utf-8").splitlines())
        counts[split]["coco_images"] += 1
        counts[split]["coco_positive_images" if boxes else "coco_negative_images"] += 1
        counts[split]["coco_boxes"] += boxes

    config = {"path": str(OUTPUT.resolve()), "train": "images/train",
              "val": "images/valid", "test": "images/test",
              "names": {0: "person_candidate"}}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    manifest = {
        "purpose": "pose-specialized detection without forgetting normal people/hard negatives",
        "split_policy": "fallen supplied splits plus COCO filename SHA256 buckets: train 0-59, val 60-79, test 80-99",
        "counts": {split: dict(values) for split, values in counts.items()},
        "medical_status": "person_candidate is not an alive/dead/injured classification",
        "warning": "Fallen source groups overlap its supplied splits; field evaluation still required."
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


if __name__ == "__main__":
    prepare()
