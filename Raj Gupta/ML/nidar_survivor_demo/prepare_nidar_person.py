"""Build a leakage-aware one-class NIDAR detector dataset from approved sources."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

import yaml


PROJECT = Path(__file__).resolve().parents[1]
FALLEN = PROJECT / "datasets" / "fallen_person" / "person_candidate"
COCO = PROJECT / "datasets" / "coco2017"
VISDRONE = PROJECT / "datasets" / "visdrone_person"
DEFAULT_OUTPUT = PROJECT / "datasets" / "nidar_person_v1"
SPLITS = ("train", "valid", "test")


def hardlink_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size != source.stat().st_size:
            raise ValueError(f"Existing output differs: {destination}")
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def fallen_group(stem: str) -> str:
    clean = stem.split(".rf.", 1)[0]
    match = re.match(r"(split\d+|img)(?:_|-)", clean)
    return match.group(1) if match else clean


def assign_groups(group_sizes: dict[str, int]) -> dict[str, str]:
    """Greedily allocate entire source groups near a 70/15/15 image target."""
    total = sum(group_sizes.values())
    targets = {"train": total * 0.70, "valid": total * 0.15, "test": total * 0.15}
    assigned = Counter()
    result: dict[str, str] = {}
    ordered = sorted(group_sizes, key=lambda item: (-group_sizes[item], hashlib.sha256(item.encode()).hexdigest()))
    for group in ordered:
        split = max(SPLITS, key=lambda name: (targets[name] - assigned[name], name == "train"))
        result[group] = split
        assigned[split] += group_sizes[group]
    return result


def collect_fallen() -> tuple[list[tuple[Path, Path, str, str]], dict[str, str]]:
    samples: list[tuple[Path, Path, str, str]] = []
    sizes = Counter()
    for supplied_split in SPLITS:
        for image in sorted((FALLEN / "images" / supplied_split).iterdir()):
            label = FALLEN / "labels" / supplied_split / f"{image.stem}.txt"
            group = fallen_group(image.stem)
            samples.append((image, label, group, supplied_split))
            sizes[group] += 1
    allocation = assign_groups(dict(sizes))
    return samples, allocation


def add_sample(image: Path, label: Path, output: Path, split: str, prefix: str,
               counts: dict[str, Counter]) -> None:
    image_name = f"{prefix}_{image.name}"
    label_name = f"{Path(image_name).stem}.txt"
    hardlink_or_copy(image, output / "images" / split / image_name)
    hardlink_or_copy(label, output / "labels" / split / label_name)
    boxes = len(label.read_text(encoding="utf-8").splitlines())
    counts[split][f"{prefix}_images"] += 1
    counts[split][f"{prefix}_boxes"] += boxes
    counts[split]["positive_images" if boxes else "negative_images"] += 1


def prepare(
    output: Path = DEFAULT_OUTPUT,
    visdrone_train_limit: int = 4000,
    visdrone_valid_limit: int = 0,
) -> dict:
    required = [FALLEN / "data.yaml", COCO / "person_val.txt", VISDRONE / "data.yaml"]
    if not all(item.is_file() for item in required):
        raise FileNotFoundError("Prepared fallen, COCO and VisDrone datasets are required.")
    if output.exists() and any((output / "images" / split).exists() for split in SPLITS):
        raise FileExistsError(f"Output already contains data: {output}")
    counts: dict[str, Counter] = {split: Counter() for split in SPLITS}

    fallen, allocation = collect_fallen()
    for image, label, group, _ in fallen:
        add_sample(image, label, output, allocation[group], "fallen", counts)

    coco_rows = (COCO / "person_val.txt").read_text(encoding="utf-8").splitlines()
    for relative in coco_rows:
        image = COCO / relative
        label = COCO / "labels" / "val2017" / f"{image.stem}.txt"
        value = int(hashlib.sha256(image.name.encode()).hexdigest()[:8], 16) % 100
        split = "train" if value < 60 else "valid" if value < 80 else "test"
        add_sample(image, label, output, split, "coco", counts)

    for split in SPLITS:
        images = sorted((VISDRONE / "images" / split).glob("*.jpg"))
        limit = visdrone_train_limit if split == "train" else visdrone_valid_limit if split == "valid" else 0
        if limit:
            images = sorted(images, key=lambda p: hashlib.sha256(p.name.encode()).hexdigest())[:limit]
        for image in images:
            label = VISDRONE / "labels" / split / f"{image.stem}.txt"
            add_sample(image, label, output, split, "visdrone", counts)

    config = {
        "path": str(output.resolve()),
        "train": "images/train",
        "val": "images/valid",
        "test": "images/test",
        "names": {0: "person_candidate"},
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "High-recall visible body detection for NIDAR; not medical status classification.",
        "class": "person_candidate",
        "sources": {
            "fallen_person_v2": "CC BY 4.0 per included Roboflow dataset card",
            "coco2017_val": "COCO annotations plus image-specific upstream licences",
            "visdrone2019_det": "REVIEW-REQUIRED before non-research deployment",
        },
        "fallen_group_allocation": allocation,
        "leakage_controls": [
            "Fallen-person coarse source groups never cross train/valid/test.",
            "COCO image IDs are disjoint deterministic buckets.",
            "VisDrone official train/val/test-dev splits are preserved.",
        ],
        "visdrone_train_limit": visdrone_train_limit,
        "visdrone_valid_limit": visdrone_valid_limit,
        "counts": {split: dict(value) for split, value in counts.items()},
        "known_gap": "Project-owned indoor OAK-D/drone footage and competition dummy data are not yet available.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--visdrone-train-limit", type=int, default=4000)
    parser.add_argument(
        "--visdrone-valid-limit", type=int, default=0,
        help="Deterministic validation subset; 0 keeps the full official validation split.",
    )
    args = parser.parse_args(argv)
    if args.visdrone_train_limit < 0 or args.visdrone_valid_limit < 0:
        parser.error("VisDrone limits must be >= 0 (0 means all).")
    prepare(args.output.resolve(), args.visdrone_train_limit, args.visdrone_valid_limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
