"""Explicit download of official weights; normal camera startup is offline."""
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen
from .detector import COCO_MODEL

URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt"


def main():
    COCO_MODEL.parent.mkdir(parents=True, exist_ok=True)
    if not COCO_MODEL.exists():
        temporary = COCO_MODEL.with_suffix(".download")
        try:
            with urlopen(URL, timeout=60) as response, temporary.open("wb") as target:
                total = 0
                while block := response.read(1024 * 1024):
                    total += len(block)
                    if total > 30 * 1024 * 1024:
                        raise ValueError("Unexpected model size; download aborted.")
                    target.write(block)
            if total < 1024 * 1024:
                raise ValueError("Model download unexpectedly small.")
            temporary.replace(COCO_MODEL)
        finally:
            temporary.unlink(missing_ok=True)
    print(json.dumps({"path": str(COCO_MODEL), "source": URL,
                      "sha256": hashlib.sha256(COCO_MODEL.read_bytes()).hexdigest(),
                      "bytes": COCO_MODEL.stat().st_size,
                      "note": "Hash records local bytes; not an independently published checksum."}))


if __name__ == "__main__":
    main()
