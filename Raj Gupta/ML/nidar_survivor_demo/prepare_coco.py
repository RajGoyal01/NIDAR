"""Download official COCO 2017 validation data and derive person-only labels.

No model is loaded or trained. Safe to repeat for this dedicated dataset folder.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time
import zipfile

from PIL import Image
import requests
import yaml

ROOT = Path(__file__).resolve().parents[1] / "datasets" / "coco2017"
SOURCES = {
    "val2017.zip": "https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip",
    "annotations_trainval2017.zip": "https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip",
}


def download(item: tuple[str, str]) -> dict:
    name, url = item
    target = ROOT / "archives" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        partial = target.with_suffix(".zip.part")
        for attempt in range(3):
            try:
                with requests.head(url, timeout=(20, 30)) as metadata:
                    metadata.raise_for_status()
                    expected = int(metadata.headers["Content-Length"])
                offset = partial.stat().st_size if partial.exists() else 0
                if offset == expected:
                    partial.replace(target)
                    break
                if offset > expected:
                    raise IOError("Partial archive is larger than the source; inspect before retrying")
                headers = {"Range": f"bytes={offset}-"} if offset else {}
                with requests.get(url, headers=headers, stream=True, timeout=(20, 90)) as response:
                    response.raise_for_status()
                    if offset and response.status_code == 206:
                        if not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                            raise IOError("Server returned an unexpected byte range")
                        mode = "ab"
                        print(f"Resuming {name} at {offset / 2**20:.0f} MiB", flush=True)
                    else:
                        offset, mode = 0, "wb"
                    received = offset
                    reported = time.monotonic()
                    with partial.open(mode) as output:
                        for chunk in response.iter_content(1024 * 1024):
                            output.write(chunk)
                            received += len(chunk)
                            if time.monotonic() - reported > 15:
                                print(f"{name}: {received / expected:.0%} ({received / 2**20:.0f} MiB)", flush=True)
                                reported = time.monotonic()
                    if received != expected:
                        raise IOError("Download length mismatch")
                partial.replace(target)
                break
            except (requests.RequestException, OSError):
                if attempt == 2:
                    raise
                print(f"Retrying {name}, attempt {attempt + 2}", flush=True)
                time.sleep(10)
    with target.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    with zipfile.ZipFile(target) as archive:
        bad = archive.testzip()
        if bad:
            raise IOError(f"ZIP CRC failure: {bad}; inspect archive before retrying")
    print(f"Verified archive: {name}", flush=True)
    return {"file": name, "url": url, "bytes": target.stat().st_size, "sha256": digest,
            "verification": "ZIP CRC and local SHA256; not an independently published hash"}


def prepare() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as executor:
        sources = list(executor.map(download, SOURCES.items()))
    annotation_path = ROOT / "annotations" / "instances_val2017.json"
    annotation_path.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(ROOT / "archives" / "annotations_trainval2017.zip") as archive:
        with archive.open("annotations/instances_val2017.json") as source, annotation_path.open("wb") as target:
            shutil.copyfileobj(source, target)
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    assert len(data["images"]) == 5000, "Unexpected validation image count"
    assert next(c for c in data["categories"] if c["id"] == 1)["name"] == "person"
    image_dir = ROOT / "images" / "val2017"
    image_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ROOT / "archives" / "val2017.zip") as archive:
        for info in data["images"]:
            name = info["file_name"]
            if Path(name).name != name:
                raise ValueError("Unexpected image filename")
            destination = image_dir / name
            with archive.open(f"val2017/{name}") as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
            with Image.open(destination) as image:
                if image.size != (info["width"], info["height"]):
                    raise ValueError(f"Image dimensions differ: {name}")
                image.load()  # Decode every image, not just its file header.
    people = [a for a in data["annotations"] if a["category_id"] == 1]
    crowd_images = {a["image_id"] for a in people if a.get("iscrowd", 0)}
    # YOLO txt has no COCO crowd-ignore representation. Exclude such images from
    # this derived evaluation set, retaining them in the official source files.
    grouped: dict[int, list] = {}
    for ann in people:
        grouped.setdefault(ann["image_id"], []).append(ann)
    label_dir = ROOT / "labels" / "val2017"
    label_dir.mkdir(parents=True, exist_ok=True)
    evaluation = []
    boxes = positive = negative = 0
    for info in data["images"]:
        if info["id"] in crowd_images:
            continue
        width, height = info["width"], info["height"]
        labels = []
        for ann in grouped.get(info["id"], []):
            x, y, w, h = ann["bbox"]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(width, x + w), min(height, y + h)
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f"Invalid person box: {ann['id']}")
            values = ((x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
                      (x2 - x1) / width, (y2 - y1) / height)
            if not all(0 <= value <= 1 for value in values):
                raise ValueError("Invalid normalized label")
            labels.append("0 " + " ".join(f"{value:.8f}" for value in values))
        (label_dir / Path(info["file_name"]).with_suffix(".txt")).write_text(
            "\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
        evaluation.append(f"./images/val2017/{info['file_name']}")
        boxes += len(labels)
        positive += bool(labels)
        negative += not bool(labels)
    (ROOT / "person_val.txt").write_text("\n".join(evaluation) + "\n", encoding="utf-8")
    config = {"path": ROOT.as_posix(), "train": None, "val": "person_val.txt", "names": {0: "person"}}
    (ROOT / "person_val.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    manifest = {"dataset": "COCO 2017 validation", "prepared_utc": datetime.now(timezone.utc).isoformat(),
                "sources": sources, "images_verified": len(data["images"]),
                "official_person_annotations": len(people), "derived_evaluation_images": len(evaluation),
                "person_positive_images": positive, "negative_images": negative,
                "person_boxes": boxes, "excluded_person_crowd_images": len(crowd_images),
                "category_mapping": {"COCO person category 1": "YOLO person class 0"},
                "terms": "https://cocodataset.org/#termsofuse",
                "license_metadata": data.get("licenses", []),
                "purpose": "Evaluation only; not a train/test split or official COCO metric reproduction",
                "disk_bytes": sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())}
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("sources", "license_metadata")}, indent=2), flush=True)


if __name__ == "__main__":
    prepare()
