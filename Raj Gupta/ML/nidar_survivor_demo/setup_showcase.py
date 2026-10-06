"""Download the six small, checksum-pinned COCO showcase images.

Raw training datasets are intentionally not stored in Git.  This module prepares
only the presentation assets and records each image's upstream licence in a local
manifest committed with the source code.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = Path(__file__).resolve().parent / "settings" / "showcase_assets.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path = DEFAULT_MANIFEST) -> tuple[dict, ...]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload["assets"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"Invalid showcase asset manifest: {path}") from exc
    if not isinstance(rows, list) or not rows:
        raise ValueError("Showcase asset manifest must contain assets.")
    required = {"path", "url", "sha256", "bytes", "license_name", "license_url"}
    checked = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != required:
            raise ValueError("Every showcase asset must use the reviewed manifest schema.")
        relative = Path(row["path"])
        parsed = urlsplit(row["url"])
        # COCO's annotation metadata publishes an HTTP image endpoint. Integrity
        # is enforced independently with exact byte size and pinned SHA-256 before
        # an atomic rename, so an altered response is never accepted.
        if (relative.is_absolute() or ".." in relative.parts or parsed.scheme not in {"http", "https"}
                or not parsed.hostname or len(row["sha256"]) != 64
                or type(row["bytes"]) is not int or row["bytes"] <= 0):
            raise ValueError("Unsafe or invalid showcase asset entry.")
        checked.append(row)
    return tuple(checked)


def prepare(manifest: Path = DEFAULT_MANIFEST) -> list[Path]:
    prepared = []
    for row in load_manifest(manifest):
        destination = (ROOT / row["path"]).resolve()
        if ROOT.resolve() not in destination.parents:
            raise ValueError("Showcase destination escaped the project root.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if (destination.is_file() and destination.stat().st_size == row["bytes"]
                and sha256(destination) == row["sha256"]):
            prepared.append(destination)
            continue
        request = Request(row["url"], headers={"User-Agent": "NIDAR-AirMouse/1.0"})
        temporary = None
        try:
            with tempfile.NamedTemporaryFile("wb", delete=False, dir=destination.parent,
                                             prefix=destination.name + ".", suffix=".part") as output:
                temporary = Path(output.name)
                with urlopen(request, timeout=30) as response:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
            if temporary.stat().st_size != row["bytes"] or sha256(temporary) != row["sha256"]:
                raise ValueError(f"Showcase checksum/size mismatch: {destination.name}")
            os.replace(temporary, destination)
            temporary = None
            prepared.append(destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return prepared


def main() -> int:
    paths = prepare()
    print(json.dumps({"event": "showcase_assets_ready", "count": len(paths),
                      "paths": [str(path.relative_to(ROOT)) for path in paths]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
