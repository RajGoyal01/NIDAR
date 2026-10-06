"""Person-candidate YOLO adapter. No tracking, identity or medical classification."""
from dataclasses import dataclass
from pathlib import Path
import math
import time
import numpy as np
import cv2

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
COCO_MODEL = MODEL_DIR / "yolo11n.pt"
SPECIALIZED_MODEL = MODEL_DIR / "nidar-person-candidate-yolo11n.pt"
# A clean checkout can still bootstrap with COCO. Once the reviewed specialized
# weights are deployed, every new demo process selects them automatically.
DEFAULT_MODEL = SPECIALIZED_MODEL if SPECIALIZED_MODEL.is_file() else COCO_MODEL
SUPPORTED_CLASS_NAMES = frozenset({"person", "person_candidate"})


def validate_person_model(task: str, names: dict[int, str] | list[str]) -> str:
    """Accept only reviewed detectors whose class zero means a visible person/body."""
    class_name = names.get(0) if isinstance(names, dict) else names[0] if names else None
    if task != "detect" or class_name not in SUPPORTED_CLASS_NAMES:
        raise ValueError("Expected a detection model with class 0 named person or person_candidate.")
    return class_name


@dataclass(frozen=True)
class DetectorConfig:
    model: Path = DEFAULT_MODEL
    imgsz: int = 640
    confidence: float = 0.35
    iou: float = 0.45
    device: str = "auto"
    fp32: bool = False

    def __post_init__(self):
        if self.imgsz not in (512, 640):
            raise ValueError("Detection size must be 512 or 640.")
        for value in (self.confidence, self.iou):
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError("Confidence and IoU must be finite and in (0, 1].")
        if self.device not in ("auto", "cpu", "0"):
            raise ValueError("Device must be auto, cpu or 0.")


@dataclass(frozen=True)
class Detection:
    xyxy: tuple[float, float, float, float]
    confidence: float


@dataclass(frozen=True)
class DetectionResult:
    people: tuple[Detection, ...]
    inference_ms: float  # Preprocess + GPU execution + postprocess + CPU result transfer.


class PersonDetector:
    def __init__(self, config: DetectorConfig):
        import torch
        from ultralytics import YOLO
        if not Path(config.model).is_file():
            raise ValueError("Model missing: run python -m nidar_survivor_demo.setup_model first.")
        self.config = config
        self.device = ("0" if torch.cuda.is_available() else "cpu") if config.device == "auto" else config.device
        if self.device == "0" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable; check environment or select --device cpu.")
        self.half = self.device != "cpu" and not config.fp32
        self.model = YOLO(str(config.model), task="detect")
        self.class_name = validate_person_model(self.model.task, self.model.names)
        # Pay initial CUDA/model setup cost before starting the live camera timer.
        for _ in range(3):
            self.predict(np.zeros((720, 1280, 3), dtype=np.uint8))

    def predict(self, frame: np.ndarray) -> DetectionResult:
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3 or not frame.size:
            raise ValueError("Detector requires a non-empty uint8 BGR image.")
        started = time.perf_counter()
        output = self.model.predict(frame, imgsz=self.config.imgsz, conf=self.config.confidence,
                                    iou=self.config.iou, classes=[0], device=self.device,
                                    quantize=16 if self.half else 32, rect=False,
                                    verbose=False, save=False, max_det=100)[0]
        rows = output.boxes.data.cpu().numpy()  # CPU copy waits for CUDA to finish.
        people = tuple(Detection(tuple(float(v) for v in row[:4]), float(row[4]))
                       for row in rows if int(row[5]) == 0)
        return DetectionResult(people, (time.perf_counter() - started) * 1000)


def annotate(frame: np.ndarray, result: DetectionResult) -> np.ndarray:
    """Draw on a copy; the camera-owned image remains untouched."""
    image = frame.copy()
    h, w = image.shape[:2]
    for detection in result.people:
        x1, y1, x2, y2 = detection.xyxy
        x1, x2 = [max(0, min(w - 1, round(x))) for x in (x1, x2)]
        y1, y2 = [max(0, min(h - 1, round(y))) for y in (y1, y2)]
        cv2.rectangle(image, (x1, y1), (x2, y2), (80, 230, 90), 2)
        cv2.putText(image, f"person {detection.confidence:.2f}", (x1, max(18, y1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 230, 90), 2, cv2.LINE_AA)
    return image
