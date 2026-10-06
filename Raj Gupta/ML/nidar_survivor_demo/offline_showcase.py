"""Phone-free, dashboard-free multi-scene person-detection showcase.

Every green box comes from the configured YOLO model.  The P labels are local
box numbers for the current picture; they are deliberately not persistent S IDs.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time

import cv2
import numpy as np

from .camera.phone_stream import FramePacket, Snapshot
from .detector import DEFAULT_MODEL, DetectionResult, DetectorConfig, PersonDetector
from .hybrid_detector import (
    DEFAULT_FUSION_CONFIG,
    DEFAULT_SPECIALIST_MODEL,
    HybridPersonDetector,
)
from .preview import TITLE, render
from .utils.fps import FPSCounter


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SHOWCASE = Path(__file__).resolve().parent / "settings" / "offline_showcase.json"
CANVAS_SIZE = (1280, 720)


@dataclass(frozen=True)
class ShowcaseScene:
    name: str
    image_path: Path
    expected_detections: int


@dataclass
class PreparedScene:
    definition: ShowcaseScene
    image: np.ndarray
    result: DetectionResult


def load_scenes(path: Path = DEFAULT_SHOWCASE) -> tuple[ShowcaseScene, ...]:
    """Read and validate the small, reviewable scene playlist."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        rows = payload["scenes"]
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise ValueError("Offline showcase configuration is missing or invalid.") from exc
    if not isinstance(rows, list) or not rows:
        raise ValueError("Offline showcase needs at least one scene.")
    scenes: list[ShowcaseScene] = []
    for row in rows:
        try:
            name = row["name"].strip()
            relative = Path(row["image"])
            expected = row["expected_detections"]
        except (AttributeError, KeyError, TypeError) as exc:
            raise ValueError("Every showcase scene needs a name, image and expected count.") from exc
        if not name or relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Showcase image paths must be safe repository-relative paths.")
        if type(expected) is not int or not 1 <= expected <= 100:
            raise ValueError("Expected detections must be an integer between 1 and 100.")
        image_path = (ROOT / relative).resolve()
        if not image_path.is_file() or ROOT.resolve() not in image_path.parents:
            raise ValueError(f"Showcase image is missing: {relative}")
        scenes.append(ShowcaseScene(name, image_path, expected))
    return tuple(scenes)


def prepare_scenes(detector: PersonDetector, scenes: tuple[ShowcaseScene, ...]) -> list[PreparedScene]:
    """Run real inference once per image and cache only the display result."""
    prepared = []
    for index, scene in enumerate(scenes, start=1):
        image = cv2.imread(str(scene.image_path), cv2.IMREAD_COLOR)
        if image is None or image.dtype != np.uint8 or image.ndim != 3:
            raise ValueError(f"Could not decode showcase scene {index}: {scene.image_path.name}")
        result = detector.predict(image)
        prepared.append(PreparedScene(scene, image, result))
        print(json.dumps({
            "event": "offline_scene_ready",
            "scene": index,
            "name": scene.name,
            "expected": scene.expected_detections,
            "detected": len(result.people),
            "inference_ms": round(result.inference_ms, 2),
        }), flush=True)
    return prepared


def _draw_detections(image: np.ndarray, result: DetectionResult) -> np.ndarray:
    output = image.copy()
    height, width = output.shape[:2]
    people = sorted(result.people, key=lambda item: (item.xyxy[0], item.xyxy[1]))
    for number, person in enumerate(people, start=1):
        x1, y1, x2, y2 = person.xyxy
        left, right = sorted((max(0, min(width - 1, round(x1))), max(0, min(width - 1, round(x2)))))
        top, bottom = sorted((max(0, min(height - 1, round(y1))), max(0, min(height - 1, round(y2)))))
        cv2.rectangle(output, (left, top), (right, bottom), (70, 235, 95), 3)
        label = f"P{number} | PERSON | {person.confidence:.2f}"
        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .55, 2)
        label_top = max(0, top - text_size[1] - baseline - 6)
        cv2.rectangle(output, (left, label_top),
                      (min(width - 1, left + text_size[0] + 8), top), (18, 45, 24), -1)
        cv2.putText(output, label, (left + 4, max(text_size[1] + 1, top - baseline - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (90, 255, 115), 2, cv2.LINE_AA)
    return output


def animate_source(image: np.ndarray, elapsed: float) -> np.ndarray:
    """Apply tiny replay-camera motion so consecutive inference frames are real and distinct."""
    height, width = image.shape[:2]
    dx = 3.0 * np.sin(elapsed * 1.15)
    dy = 2.0 * np.cos(elapsed * .9)
    brightness = 2.0 * np.sin(elapsed * .7)
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    moved = cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT_101)
    return cv2.convertScaleAbs(moved, alpha=1.0, beta=brightness)


def render_scene(scene: PreparedScene, index: int, total: int, *, sequence: int = 1,
                 replay_fps: float = 0.0, displayed_fps: float = 0.0,
                 scene_switches: int = 0, paused: bool = False,
                 frame: np.ndarray | None = None) -> np.ndarray:
    """Reuse the original camera-test layout with honest local-replay telemetry."""
    boxed = _draw_detections(scene.image if frame is None else frame, scene.result)
    actual = len(scene.result.people)
    label = "PAUSED" if paused else "ACTIVE"
    cv2.rectangle(boxed, (8, 8), (min(boxed.shape[1] - 8, 390), 72), (18, 25, 20), -1)
    cv2.putText(boxed, f"LIVE DETECTION: {label}", (20, 34), cv2.FONT_HERSHEY_SIMPLEX,
                .62, (80, 245, 105), 2, cv2.LINE_AA)
    cv2.putText(boxed, f"CURRENT PEOPLE: {actual}   SCENE {index + 1}/{total}",
                (20, 62), cv2.FONT_HERSHEY_SIMPLEX, .52, (235, 235, 235), 1, cv2.LINE_AA)
    now = time.monotonic()
    snapshot = Snapshot("ONLINE", FramePacket(boxed, sequence, now), replay_fps, 0.0,
                        sequence, scene_switches + 1, scene_switches + 1)
    status = (f"LIVE DETECTION | People in THIS frame: {actual} | YOLO: {scene.result.inference_ms:.1f} ms | "
              f"Scene {index + 1}/{total} | N/P: change | R: re-detect | Space: pause | Q: quit")
    return render(snapshot, displayed_fps, status,
                  source_label="REPLAY SOURCE", connection_label="Scene switches")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_SHOWCASE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--hybrid-detector", action="store_true")
    parser.add_argument("--specialist-model", type=Path, default=DEFAULT_SPECIALIST_MODEL)
    parser.add_argument("--fusion-config", type=Path, default=DEFAULT_FUSION_CONFIG)
    parser.add_argument("--device", choices=("auto", "cpu", "0"), default="auto")
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--imgsz", type=int, choices=(512, 640), default=640)
    parser.add_argument("--confidence", type=float, default=.25)
    parser.add_argument("--iou", type=float, default=.45)
    parser.add_argument("--scene-seconds", type=float, default=5.0)
    parser.add_argument("--inference-fps", type=int, choices=range(1, 31), default=12,
                        help="Continuous real-YOLO replay rate; not camera capture FPS.")
    parser.add_argument("--cycles", type=int, default=0, help="0 repeats until Q/Esc; positive values auto-stop.")
    parser.add_argument("--headless", action="store_true", help="Validate every scene without opening a window.")
    return parser


def run(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not .5 <= args.scene_seconds <= 120:
        raise SystemExit("--scene-seconds must be between 0.5 and 120")
    if not 0 <= args.cycles <= 1000:
        raise SystemExit("--cycles must be between 0 and 1000")
    scenes = load_scenes(args.config)
    detector_config = DetectorConfig(args.model, args.imgsz, args.confidence,
                                     args.iou, args.device, args.fp32)
    detector = (HybridPersonDetector(detector_config, args.specialist_model, args.fusion_config)
                if args.hybrid_detector else PersonDetector(detector_config))
    prepared = prepare_scenes(detector, scenes)
    mismatches = [item for item in prepared
                  if len(item.result.people) != item.definition.expected_detections]
    motion_counts = None
    motion_stable = None
    if args.headless:
        phases = (0.0, .75, 1.5, 2.25, 3.0, 3.75, 4.5)
        motion_counts = [[len(detector.predict(animate_source(item.image, phase)).people)
                          for phase in phases] for item in prepared]
        motion_stable = all(all(count == item.definition.expected_detections for count in counts)
                            for item, counts in zip(prepared, motion_counts))
    summary = {
        "event": "offline_showcase_ready",
        "scenes": len(prepared),
        "counts": [len(item.result.people) for item in prepared],
        "expected": [item.definition.expected_detections for item in prepared],
        "all_counts_match": not mismatches,
        "motion_counts": motion_counts,
        "motion_stable": motion_stable,
        "meaning": "current-frame YOLO detections, not persistent survivor identities",
        "detector_mode": "validated_hybrid" if args.hybrid_detector else "single_model",
    }
    print(json.dumps(summary), flush=True)
    if args.headless:
        return 0 if not mismatches and motion_stable else 2

    cv2.namedWindow(TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(TITLE, *CANVAS_SIZE)
    index = completed_cycles = sequence = scene_switches = 0
    paused = False
    scene_started = time.monotonic()
    next_inference = 0.0
    replay_rate = FPSCounter()
    display_rate = FPSCounter()
    display_frame = prepared[index].image
    try:
        while True:
            now = time.monotonic()
            if not paused and now >= next_inference:
                display_frame = animate_source(prepared[index].image, now - scene_started)
                prepared[index].result = detector.predict(display_frame)
                sequence += 1
                replay_rate.tick(now)
                next_inference = now + 1 / args.inference_fps
            display_rate.tick(now)
            cv2.imshow(TITLE, render_scene(prepared[index], index, len(prepared), sequence=max(1, sequence),
                                          replay_fps=replay_rate.value(now), displayed_fps=display_rate.value(now),
                                          scene_switches=scene_switches, paused=paused, frame=display_frame))
            key = cv2.waitKey(25) & 0xFF
            direction = 0
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("n"), ord("N"), 83):
                direction = 1
            elif key in (ord("p"), ord("P"), 81):
                direction = -1
            elif key == ord(" "):
                paused = not paused
                scene_started = time.monotonic()
            elif key in (ord("r"), ord("R")):
                prepared[index].result = detector.predict(animate_source(prepared[index].image, 0))
                scene_started = time.monotonic()
            elif not paused and time.monotonic() - scene_started >= args.scene_seconds:
                direction = 1
            if direction:
                previous = index
                index = (index + direction) % len(prepared)
                scene_switches += 1
                display_frame = prepared[index].image
                if direction > 0 and previous == len(prepared) - 1:
                    completed_cycles += 1
                    if args.cycles and completed_cycles >= args.cycles:
                        break
                scene_started = time.monotonic()
                next_inference = 0.0
            if cv2.getWindowProperty(TITLE, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cv2.destroyAllWindows()
    return 0 if not mismatches else 2


if __name__ == "__main__":
    raise SystemExit(run())
