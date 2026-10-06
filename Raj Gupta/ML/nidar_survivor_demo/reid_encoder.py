"""OSNet x0.25 feature extraction with explicit local, trusted weights."""
from pathlib import Path
import cv2
import numpy as np
from .setup_reid import MODEL, verify_model


def unit_vector(values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
        raise ValueError("Embedding must be a finite non-empty vector.")
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError("Embedding norm is invalid.")
    return vector / norm


class AppearanceEncoder:
    def __init__(self, model: Path = MODEL, device: str = "auto"):
        import torch
        from .vendor.osnet import OSNet, OSBlock
        verify_model(model)
        if device not in ("auto", "cpu", "0"):
            raise ValueError("Use auto, cpu or 0 for Re-ID device.")
        self.device = "cuda:0" if device == "0" or (device == "auto" and torch.cuda.is_available()) else "cpu"
        # No arbitrary pickle execution; checksum plus weights-only loading.
        weights = torch.load(model, map_location="cpu", weights_only=True)
        if "state_dict" in weights:
            weights = weights["state_dict"]
        weights = {k.removeprefix("module."): v for k, v in weights.items()}
        self.model = OSNet(1, [OSBlock] * 3, [2, 2, 2], [16, 64, 96, 128])
        weights = {k: v for k, v in weights.items() if not k.startswith("classifier.")}
        missing, unexpected = self.model.load_state_dict(weights, strict=False)
        if set(missing) != {"classifier.weight", "classifier.bias"} or unexpected:
            raise ValueError("OSNet architecture/checkpoint mismatch; no partial backbone allowed.")
        self.model.eval().to(self.device)
        self._mean = torch.tensor([.485, .456, .406], device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor([.229, .224, .225], device=self.device).view(1, 3, 1, 1)
        for _ in range(3):
            self.encode([np.full((256, 128, 3), 128, np.uint8)])

    def encode(self, crops: list[np.ndarray]) -> list[np.ndarray]:
        import torch
        if not crops:
            return []
        if len(crops) > 8:
            raise ValueError("Re-ID batch must contain at most 8 crops.")
        prepared = []
        for crop in crops:
            if crop.dtype != np.uint8 or crop.ndim != 3 or crop.shape[2] != 3 or not crop.size:
                raise ValueError("Re-ID requires non-empty uint8 BGR crops.")
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            # Match upstream test preprocessing: PIL bilinear resize then ImageNet normalization.
            from PIL import Image
            resized = np.asarray(Image.fromarray(rgb).resize((128, 256), Image.Resampling.BILINEAR))
            prepared.append(resized.transpose(2, 0, 1))
        tensor = torch.from_numpy(np.stack(prepared)).to(self.device, dtype=torch.float32) / 255
        with torch.inference_mode():
            features = self.model((tensor - self._mean) / self._std).cpu().numpy()
        return [unit_vector(row) for row in features]
