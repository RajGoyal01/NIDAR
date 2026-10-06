from pathlib import Path
import json
from unittest.mock import patch

from nidar_survivor_demo.prepare_nidar_person import assign_groups, fallen_group
from nidar_survivor_demo.evaluate_nidar_detector import size_bin, source_name
from nidar_survivor_demo.gate_nidar_detector import build_gate_report
from nidar_survivor_demo.full_visdrone_pipeline import (
    EXPECTED_VISDRONE_IMAGES,
    validate_mixture_manifest,
    validate_source_manifest,
)
from nidar_survivor_demo.prepare_detector_recovery import select_aerial_reminders
from nidar_survivor_demo.recover_nidar_detector import validate_recovery_manifest
from nidar_survivor_demo.hybrid_detector import (
    FusionPolicy,
    HybridPersonDetector,
    fuse,
    intersection_over_union,
)
from nidar_survivor_demo.detector import Detection, DetectorConfig


def test_fallen_group_removes_roboflow_hash() -> None:
    assert fallen_group("split5_123_png.rf.abcdef") == "split5"
    assert fallen_group("img_001_png.rf.abcdef") == "img"


def test_group_split_never_leaks() -> None:
    sizes = {"scene_a": 100, "scene_b": 90, "scene_c": 80, "scene_d": 70,
             "scene_e": 60, "scene_f": 50}
    allocation = assign_groups(sizes)
    assert set(allocation) == set(sizes)
    assert set(allocation.values()) == {"train", "valid", "test"}
    assert allocation == assign_groups(sizes)


def test_training_entrypoint_defaults_are_local() -> None:
    source = Path(__file__).parents[1] / "train_nidar_detector.py"
    text = source.read_text(encoding="utf-8")
    assert "nidar_person_v1" in text
    assert "COCO_MODEL" in text
    assert "candidate_only" in text
    assert '"exist_ok": True' in text


def test_evaluation_groups_sources_and_small_targets() -> None:
    assert source_name("visdrone_00001.jpg") == "visdrone"
    assert source_name("fallen_img_001.jpg") == "fallen"
    assert size_bin((0.0, 0.0, 10.0, 10.0)) == "tiny"
    assert size_bin((0.0, 0.0, 20.0, 20.0)) == "small"
    assert size_bin((0.0, 0.0, 50.0, 50.0)) == "medium"
    assert size_bin((0.0, 0.0, 100.0, 100.0)) == "large"


def test_promotion_gate_rejects_general_person_regression() -> None:
    baseline = {
        "by_source": {"coco": {"negative_images_with_fp": 10}, "visdrone": {"recall": 0.04}},
        "by_size": {"small": {"recall": 0.06}},
    }
    candidate = {
        "model_class": "person_candidate",
        "by_source": {
            "coco": {"recall": 0.50, "precision": 0.90, "negative_images_with_fp": 10},
            "fallen": {"recall": 0.95},
            "visdrone": {"recall": 0.20},
        },
        "fallen_pose_recall": {"lying": 0.90},
        "by_size": {"large": {"recall": 0.90}, "small": {"recall": 0.30}},
        "inference_p95_ms": 40.0,
    }
    result = build_gate_report(baseline, candidate)
    assert result["status"] == "REJECTED-NOT-PROMOTED"
    assert result["checks"]["coco_recall"] is False


def test_full_visdrone_source_requires_every_official_split() -> None:
    manifest = {
        "class_mapping": {
            "1 pedestrian": "person_candidate",
            "2 people": "person_candidate",
        },
        "split_policy": "Official train/val/test-dev boundaries preserved.",
        "counts": {split: {"images": count} for split, count in EXPECTED_VISDRONE_IMAGES.items()},
    }
    validate_source_manifest(manifest)
    manifest["counts"]["train"]["images"] -= 1
    try:
        validate_source_manifest(manifest)
    except ValueError as exc:
        assert "exactly 6471" in str(exc)
    else:
        raise AssertionError("Partial VisDrone training split was accepted")


def test_full_mixture_rejects_a_visdrone_subset() -> None:
    manifest = {
        "class": "person_candidate",
        "visdrone_train_limit": 0,
        "visdrone_valid_limit": 0,
        "counts": {split: {"visdrone_images": count}
                   for split, count in EXPECTED_VISDRONE_IMAGES.items()},
    }
    validate_mixture_manifest(manifest)
    manifest["visdrone_train_limit"] = 1000
    try:
        validate_mixture_manifest(manifest)
    except ValueError as exc:
        assert "all VisDrone images" in str(exc)
    else:
        raise AssertionError("Limited VisDrone mixture was accepted as full")


def test_recovery_manifest_keeps_unchanged_test_and_coco_replay() -> None:
    manifest = {
        "class": "person_candidate",
        "sampling_policy": {"coco_repeats": 4},
        "counts": {
            "train": {"coco_images": 2859 * 4, "fallen_images": 2021},
            "test": {"fallen_images": 447, "coco_images": 899, "visdrone_images": 1610},
        },
    }
    validate_recovery_manifest(manifest)
    manifest["counts"]["test"]["coco_images"] -= 1
    try:
        validate_recovery_manifest(manifest)
    except ValueError as exc:
        assert "unchanged" not in str(exc) or "899" in str(exc)
    else:
        raise AssertionError("Changed recovery test split was accepted")


def test_aerial_reminder_selection_is_deterministic(tmp_path: Path) -> None:
    images = []
    labels = tmp_path / "labels"
    labels.mkdir()
    for index, boxes in enumerate((1, 3, 2, 4)):
        image = tmp_path / f"visdrone_{index}.jpg"
        image.write_bytes(b"image")
        (labels / f"{image.stem}.txt").write_text("0 .5 .5 .1 .1\n" * boxes, encoding="utf-8")
        images.append(image)
    first = select_aerial_reminders(images, labels, 5)
    second = select_aerial_reminders(list(reversed(images)), labels, 5)
    assert [item.name for item in first] == [item.name for item in second]
    assert sum(len((labels / f"{item.stem}.txt").read_text().splitlines()) for item in first) >= 5


def test_hybrid_fusion_preserves_primary_and_qualifies_specialist() -> None:
    primary = (Detection((10, 10, 50, 90), .8),)
    specialist = (
        Detection((11, 11, 49, 89), .9),      # duplicate
        Detection((60, 60, 140, 100), .22),   # horizontal posture
        Detection((180, 180, 190, 195), .26), # small body
        Detection((200, 10, 240, 90), .19),   # weak ordinary box
    )
    policy = FusionPolicy(posture_confidence=.20, small_confidence=.25)
    result = fuse(primary, specialist, (300, 300), policy)
    assert primary[0] in result
    assert specialist[0] not in result
    assert specialist[1] in result
    assert specialist[2] in result
    assert specialist[3] not in result


def test_box_iou_handles_overlap_and_disjoint() -> None:
    assert intersection_over_union((0, 0, 10, 10), (0, 0, 10, 10)) == 1
    assert intersection_over_union((0, 0, 10, 10), (20, 20, 30, 30)) == 0


def test_hybrid_detector_preserves_callers_primary_threshold(tmp_path: Path) -> None:
    fusion_path = tmp_path / "fusion.json"
    fusion_path.write_text(json.dumps({
        "primary_confidence": .2,
        "general_confidence": .7,
        "posture_confidence": .25,
        "horizontal_aspect": 1.0,
        "small_confidence": .2,
        "small_area_ratio": .006,
        "dedupe_iou": .5,
    }), encoding="utf-8")
    specialist = tmp_path / "specialist.pt"
    specialist.write_bytes(b"test")
    received = []

    class StubDetector:
        def __init__(self, config):
            received.append(config)
            self.device = "cpu"
            self.half = False

    with patch("nidar_survivor_demo.hybrid_detector.PersonDetector", StubDetector):
        detector = HybridPersonDetector(
            DetectorConfig(model=tmp_path / "primary.pt", confidence=.25, device="cpu"),
            specialist,
            fusion_path,
        )
    assert detector.policy.primary_confidence == .25
    assert received[0].confidence == .25
    assert received[1].confidence == .2


def test_hybrid_policy_rejects_unknown_configuration_fields(tmp_path: Path) -> None:
    path = tmp_path / "invalid.json"
    path.write_text('{"primary_confidence": 0.2, "surprise": true}', encoding="utf-8")
    try:
        FusionPolicy.load(path)
    except ValueError as exc:
        assert "missing or unknown" in str(exc)
    else:
        raise AssertionError("Invalid hybrid configuration was accepted")
