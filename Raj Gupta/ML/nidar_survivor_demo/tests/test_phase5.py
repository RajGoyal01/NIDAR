import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from nidar_survivor_demo import main
from nidar_survivor_demo.config import arguments, CameraConfig
from nidar_survivor_demo.camera.phone_stream import PhoneStream
from nidar_survivor_demo.tracking import Track
from nidar_survivor_demo.verification import VerificationResult, VerifiedTrack
from nidar_survivor_demo.reid import ReIDConfig, AppearanceMatcher, compare_gallery, quality_crop
from nidar_survivor_demo.reid_encoder import unit_vector
from nidar_survivor_demo.setup_reid import verify_model
from .test_phase1 import FakeCapture
from .test_phase2 import FakeDetector


class FakeEncoder:
    device = "cpu"
    def __init__(self):
        self.vector = np.array([1., 0., 0.], np.float32)
        self.calls = 0
    def encode(self, crops):
        self.calls += len(crops)
        return [self.vector.copy() for _ in crops]


class Phase5Tests(unittest.TestCase):
    def setUp(self):
        self.frame = np.random.default_rng(42).integers(0, 255, (320, 500, 3), dtype=np.uint8)
        self.encoder = FakeEncoder()
        self.config = ReIDConfig(sample_interval=.01, retry_interval=.01)
        self.matcher = AppearanceMatcher(self.config, self.encoder)
        self.n = 0

    def update(self, ids=("E1:T1",), state="CONFIRMED", epoch=1, **kwargs):
        self.n += 1
        observations = tuple(VerifiedTrack(Track(label, (10 + i * 150, 10, 110 + i * 150, 300), .9), state, 3, True)
                             for i, label in enumerate(ids))
        return self.matcher.update(self.frame, VerificationResult(observations, (), 0),
                                   sequence=self.n, received_at=self.n * .1, epoch=epoch, **kwargs)

    def enroll(self, label="E1:T1"):
        for _ in range(3):
            result = self.update((label,))
        return result

    def test_three_samples_and_selective_embedding(self):
        result = self.enroll()
        self.assertEqual(result.observations[0].reference_id, "R1")
        for _ in range(20):
            self.update()
        self.assertEqual(self.encoder.calls, 3)
        self.assertEqual(len(self.matcher.gallery["R1"].vectors), 3)

    def test_reentry_new_temporary_id_recovers_reference(self):
        self.enroll()
        self.update(())
        result = self.enroll("E1:T9")
        self.assertEqual(result.observations[0].reference_id, "R1")
        self.assertEqual(result.events[0]["event"], "reid_match")
        self.assertEqual(len(self.matcher.gallery), 1)

    def test_same_temporary_id_reappearance_rechecks(self):
        self.enroll()
        self.update(())
        result = self.enroll()
        self.assertEqual(self.encoder.calls, 6)
        self.assertEqual(result.observations[0].state, "MATCHED")

    def test_distinct_person_gets_distinct_reference(self):
        self.enroll()
        self.update(())
        self.encoder.vector = np.array([0., 1., 0.])
        self.assertEqual(self.enroll("E1:T9").observations[0].reference_id, "R2")

    def test_middle_similarity_remains_uncertain(self):
        self.enroll()
        self.update(())
        self.encoder.vector = unit_vector(np.array([.65, .76, 0]))
        result = self.enroll("E1:T9")
        self.assertIsNone(result.observations[0].reference_id)
        self.assertEqual(result.observations[0].state, "UNCERTAIN")
        self.assertEqual(len(self.matcher.gallery), 1)

    def test_near_tie_is_uncertain(self):
        self.enroll()
        self.matcher.gallery["R2"] = self.matcher.gallery["R1"]
        self.update(())
        result = self.enroll("E1:T9")
        self.assertEqual(result.observations[0].state, "UNCERTAIN")
        self.assertEqual(result.observations[0].margin, 0)

    def test_simultaneously_visible_reference_cannot_be_reused(self):
        self.enroll()
        for _ in range(3):
            result = self.update(("E1:T1", "E1:T2"))
        self.assertIsNone(result.observations[1].reference_id)
        self.assertEqual(result.observations[1].reason, "reference_visible_on_other_track")

    def test_two_queries_cannot_claim_same_reference(self):
        self.enroll()
        self.update(())
        for _ in range(3):
            result = self.update(("E1:T7", "E1:T8"))
        self.assertTrue(all(o.reason == "simultaneous_match_conflict" for o in result.observations))

    def test_temporal_gate_prevents_encoding(self):
        for _ in range(4):
            self.update(state="VERIFYING")
        self.assertEqual(self.encoder.calls, 0)
        self.assertFalse(self.matcher.gallery)

    def test_long_temporal_revocation_discards_unbound_samples(self):
        self.update()
        self.update()
        for _ in range(9):
            self.update(state="VERIFYING")
        result = self.update()
        self.assertIsNone(result.observations[0].reference_id)
        self.assertEqual(len(self.matcher._pending["E1:T1"].vectors), 1)

    def test_model_output_validation_and_repeated_frame(self):
        self.update()
        with self.assertRaises(ValueError):
            self.matcher.update(self.frame, VerificationResult((), (), 0), sequence=1, received_at=.1, epoch=1)
        self.encoder.vector = np.ones(4)
        with self.assertRaises(ValueError):
            self.update()
        self.assertFalse(self.matcher.gallery)

    def test_stale_match_keeps_existing_gallery_unchanged(self):
        self.enroll()
        old = tuple(v.copy() for v in self.matcher.gallery["R1"].vectors)
        self.update(())
        self.update(("E1:T2",))
        self.update(("E1:T2",))
        self.update(("E1:T2",), is_fresh=lambda: False)
        self.assertFalse(self.matcher._bindings)
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(old, self.matcher.gallery["R1"].vectors)))

    def test_stale_encoding_cannot_enroll(self):
        self.update()
        self.update()
        result = self.update(is_fresh=lambda: False)
        self.assertFalse(self.matcher.gallery)
        self.assertEqual(result.events[0]["event"], "reid_stale_discard")

    def test_reset_keeps_reference_but_requires_recheck(self):
        self.enroll()
        self.matcher.clear_tracks()
        result = self.update(("E2:T1",), epoch=2)
        self.assertIsNone(result.observations[0].reference_id)
        self.assertEqual(len(self.matcher.gallery), 1)

    def test_gallery_capacity_and_ttl(self):
        self.matcher.config = replace(self.config, max_gallery=1, gallery_ttl=.8)
        self.enroll()
        self.update(())
        self.encoder.vector = np.array([0., 1., 0.])
        result = self.enroll("E1:T2")
        self.assertEqual(result.observations[0].reason, "gallery_capacity")
        for _ in range(10):
            self.update(())
        self.assertFalse(self.matcher.gallery)
        self.assertEqual(self.enroll("E1:T3").observations[0].reference_id, "R2")

    def test_no_mixed_identity_query_enrollment(self):
        self.update()
        self.encoder.vector = np.array([0., 1., 0.])
        self.update()
        result = self.update()
        self.assertEqual(result.observations[0].reason, "inconsistent_views")
        self.assertFalse(self.matcher.gallery)

    def test_quality_filters(self):
        track = Track("E1:T1", (10, 10, 110, 300), .9)
        self.assertEqual(quality_crop(self.frame, track, [track], self.config)[1], "good")
        for changed, expected in ((replace(track, confidence=.1), "low_confidence"),
                                  (replace(track, xyxy=(0, 0, 10, 20)), "small_crop"),
                                  (replace(track, xyxy=(-60, 10, 100, 300)), "clipped_crop")):
            self.assertEqual(quality_crop(self.frame, changed, [changed], self.config)[1], expected)
        self.assertEqual(quality_crop(np.zeros_like(self.frame), track, [track], self.config)[1], "blurred_crop")
        self.assertEqual(quality_crop(self.frame, track, [track, replace(track, track_id="other")], self.config)[1], "overlapping_person")

    def test_unit_vectors_and_similarity(self):
        self.assertAlmostEqual(float(np.linalg.norm(unit_vector(np.array([2., 3.])))), 1, places=6)
        for value in (np.zeros(3), np.array([float("nan")]), np.ones((2, 3))):
            with self.assertRaises(ValueError):
                unit_vector(value)
        ranked = compare_gallery([np.array([1., 0.])], {"a": [np.array([1., 0.])], "b": [np.array([0., 1.])]})
        self.assertEqual(ranked[0][0], "a")

    def test_config_cli_and_weight_integrity(self):
        args = arguments(["--reid"])[1]
        self.assertTrue(args.verify and args.track and args.detect)
        for kwargs in ({"samples": True}, {"max_batch": 0}, {"match_threshold": float("nan")},
                       {"novelty_threshold": .99}, {"sample_interval": 0}, {"max_gallery": 1000}):
            with self.assertRaises(ValueError):
                ReIDConfig(**kwargs)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.pth"
            path.write_bytes(b"not a model")
            with self.assertRaises(ValueError):
                verify_model(path)

    def test_encoder_failure_releases_camera(self):
        class BrokenEncoder(FakeEncoder):
            def encode(self, crops):
                raise RuntimeError("test failure")
        cap = FakeCapture()
        config = replace(self.config, min_width=1, min_height=1, min_blur_variance=.0001)
        with patch.object(main, "AppearanceEncoder", return_value=BrokenEncoder()), \
             patch.object(main.ReIDConfig, "load", return_value=config), \
             patch.object(main, "PersonDetector", return_value=FakeDetector()), \
             patch.object(main, "PhoneStream", return_value=PhoneStream(CameraConfig(), lambda _: cap)), \
             patch("nidar_survivor_demo.reid.quality_crop", return_value=(self.frame, "good")), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main.run(["--reid", "--headless", "--seconds", ".5"]), 1)
        self.assertTrue(cap.released)

    def test_main_integration_and_no_embeddings_in_logs(self):
        cap = FakeCapture()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "reid.jsonl"
            with patch.object(main, "AppearanceEncoder", return_value=FakeEncoder()), \
                 patch.object(main.ReIDConfig, "load", return_value=self.config), \
                 patch.object(main, "PersonDetector", return_value=FakeDetector()), \
                 patch.object(main, "PhoneStream", return_value=PhoneStream(CameraConfig(), lambda _: cap)), \
                 patch("nidar_survivor_demo.reid.quality_crop", return_value=(self.frame, "good")), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main.run(["--reid", "--headless", "--seconds", ".5", "--track-log", str(path)]), 0)
            records = [json.loads(line) for line in path.read_text().splitlines()]
            events = [e for r in records if r["event"] == "reid_frame" for e in r["reid"]["events"]]
            self.assertTrue(any(e["event"] == "reid_new_reference" for e in events))
            self.assertNotIn('"vectors"', path.read_text())
            self.assertTrue(cap.released)


if __name__ == "__main__":
    unittest.main()
