"""Explicit, checksum-verified download of author-hosted OSNet Re-ID weights."""
import hashlib
from pathlib import Path
import tempfile
from urllib.request import urlopen

MODEL = Path(__file__).resolve().parents[1] / "models/osnet_x0_25_msmt17.pth"
SHA256 = "cf55163d78fc44c62c82f85ab62d39f10438679b5abe8c698ae08cfa84aa6e18"
REVISION = "a5c5cc037c24235cda3b21085b93ad77c9616224"
FILENAME = "osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth"
URL = f"https://huggingface.co/kaiyangzhou/osnet/resolve/{REVISION}/{FILENAME}"


def verify_model(path: Path) -> None:
    if not path.is_file() or path.stat().st_size != 9336983:
        raise ValueError("OSNet weights missing/wrong size. Run python -m nidar_survivor_demo.setup_reid.")
    if hashlib.sha256(path.read_bytes()).hexdigest() != SHA256:
        raise ValueError("OSNet checksum mismatch; refusing to load weights.")


def main() -> None:
    if MODEL.exists():
        verify_model(MODEL)
        print("Existing OSNet checkpoint matches publisher SHA-256.")
        return
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(dir=MODEL.parent, suffix=".download", delete=False) as output:
            temp = Path(output.name)
            with urlopen(URL, timeout=30) as response:
                size = 0
                while chunk := response.read(1024 * 1024):
                    size += len(chunk)
                    if size > 10_000_000:
                        raise ValueError("Unexpected checkpoint size.")
                    output.write(chunk)
        verify_model(temp)
        if MODEL.exists():
            raise FileExistsError("Checkpoint appeared during download; not overwriting.")
        temp.rename(MODEL)
        print(f"Verified OSNet: {MODEL}; SHA256={SHA256}")
    finally:
        if temp and temp.exists():
            temp.unlink()  # Only our exact temporary download, never user data.


if __name__ == "__main__":
    main()
