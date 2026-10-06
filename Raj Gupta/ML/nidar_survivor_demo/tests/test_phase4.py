import contextlib
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
from nidar_survivor_demo import main
from nidar_survivor_demo.config import arguments, CameraConfig
from nidar_survivor_demo.camera.phone_stream import PhoneStream
from nidar_survivor_demo.tracking import Track, TrackingResult
from nidar_survivor_demo.verification import VerificationConfig, TemporalVerifier, annotate_verification
from .test_phase1 import FakeCapture
from .test_phase2 import FakeDetector


class Phase4Tests(unittest.TestCase):
    def setUp(self):
        self.verifier = TemporalVerifier(VerificationConfig())

    def update(self, n, ids=("E1:T1",), score=.9, box=(20, 20, 100, 200), **kwargs):
        params = dict(sequence=n, received_at=n * .05, epoch=1, frame_shape=(720, 1280, 3))
        params.update(kwargs)
        return self.verifier.update(TrackingResult(tuple(Track(label, box, score) for label in ids), 0, ()), **params)

    def test_three_hits_confirm_on_third_observation(self):
        self.assertEqual(self.update(1).confirmed_count, 0)
        self.assertEqual(self.update(2).confirmed_count, 0)
        result = self.update(3)
        self.assertEqual(result.confirmed_count, 1)
        self.assertEqual(result.events[0]["evidence_elapsed_ms"], 100)
        self.assertFalse(self.update(4).events)

    def test_flash_never_confirms_and_expires(self):
        events = list(self.update(1).events)
        for n in range(2, 7):
            result = self.update(n, ())
            self.assertEqual(result.confirmed_count, 0)
            events.extend(result.events)
        self.assertNotIn("verification_confirmed", [e["event"] for e in events])
        self.assertEqual(events[-1]["event"], "verification_expired")
        self.assertFalse(self.verifier._evidence)

    def test_three_of_five_with_gaps(self):
        for n in range(1, 6):
            result = self.update(n, () if n in (2, 4) else ("E1:T1",))
        self.assertEqual(result.confirmed_count, 1)

    def test_two_of_five_not_enough(self):
        for n in range(1, 6):
            result = self.update(n, ("E1:T1",) if n in (1, 5) else ())
        self.assertEqual(result.confirmed_count, 0)

    def test_current_low_confidence_revokes(self):
        for n in range(1, 4):
            self.update(n)
        result = self.update(4, score=.2)
        self.assertEqual(result.confirmed_count, 0)
        self.assertEqual(result.events[0]["event"], "verification_revoked")

    def test_quality_rejects_tiny_offscreen_and_inverted_boxes(self):
        for box in ((1, 1, 2, 2), (-100, 0, -1, 200), (100, 100, 20, 20)):
            self.verifier.clear()
            for n in range(1, 6):
                result = self.update(n, box=box)
            self.assertEqual(result.confirmed_count, 0)

    def test_missing_is_hidden_and_old_evidence_ages_out(self):
        for n in range(1, 4):
            self.update(n)
        for n in range(4, 8):
            self.assertFalse(self.update(n, ()).tracks)
        self.assertEqual(self.update(8).confirmed_count, 0)

    def test_ids_are_independent(self):
        self.update(1, ("E1:T1", "E1:T2"))
        self.update(2)
        result = self.update(3, ("E1:T1", "E1:T2"))
        self.assertEqual([t.state for t in result.tracks], ["CONFIRMED", "VERIFYING"])

    def test_reset_epoch_and_time_gap_require_new_evidence(self):
        for reset in ("clear", "epoch", "gap"):
            self.verifier.clear()
            for n in range(1, 4):
                self.update(n)
            kwargs = {}
            if reset == "clear":
                self.assertTrue(self.verifier.clear())
                self.assertFalse(self.verifier.clear())
            elif reset == "epoch":
                kwargs["epoch"] = 2
            else:
                kwargs["received_at"] = 2
            self.assertEqual(self.update(4, **kwargs).confirmed_count, 0)

    def test_repeated_and_backwards_input_rejected_without_evidence(self):
        self.update(2)
        for n in (2, 1):
            with self.assertRaises(ValueError):
                self.update(n)
        self.assertEqual(self.update(3).confirmed_count, 0)

    def test_invalid_tracks_rejected(self):
        for kwargs in ({"score": float("nan")}, {"score": 2}, {"ids": ("x", "x")},
                       {"box": (0, 0, float("inf"), 20)}, {"received_at": float("nan")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.update(1, **kwargs)

    def test_history_and_identity_storage_bounded(self):
        for n in range(1, 501):
            self.update(n, (f"E1:T{n}",))
        self.assertLessEqual(len(self.verifier._evidence), 5)
        self.assertTrue(all(len(e.history) <= 5 for e in self.verifier._evidence.values()))

    def test_config_and_cli(self):
        args = arguments(["--verify"])[1]
        self.assertTrue(args.track and args.detect)
        for kwargs in ({"window": True}, {"window": 0}, {"required": 6}, {"required": 1.5},
                       {"min_confidence": float("nan")}, {"min_visible_area_ratio": 2}, {"max_gap_seconds": 6}):
            with self.assertRaises(ValueError):
                VerificationConfig(**kwargs)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.json"
            path.write_text('{"unexpected": true}')
            with self.assertRaises(ValueError):
                VerificationConfig.load(path)
        with patch.object(main, "PhoneStream") as stream:
            self.assertEqual(main.run(["--verify", "--verification-config", "nonexistent.json"]), 1)
            stream.assert_not_called()

    def test_annotation_does_not_modify_source(self):
        frame = np.zeros((720, 1280, 3), np.uint8)
        image = annotate_verification(frame, self.update(1), self.verifier.config)
        self.assertFalse(frame.any())
        self.assertTrue(image.any())

    def test_main_end_to_end_metadata(self):
        cap = FakeCapture()
        stream = PhoneStream(CameraConfig(), lambda _: cap)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "evidence.jsonl"
            with patch.object(main, "PersonDetector", return_value=FakeDetector()), \
                 patch.object(main, "PhoneStream", return_value=stream), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main.run(["--verify", "--headless", "--seconds", ".5", "--track-log", str(path)]), 0)
            records = [json.loads(line) for line in path.read_text().splitlines()]
            frames = [r for r in records if r["event"] == "verified_frame"]
            self.assertGreaterEqual(len(frames), 3)
            self.assertEqual(frames[0]["verification"]["tracks"][0]["state"], "VERIFYING")
            self.assertTrue(any(t["state"] == "CONFIRMED" for f in frames for t in f["verification"]["tracks"]))
            self.assertTrue(cap.released)

    def test_stale_inference_cannot_create_confirmation(self):
        cap = FakeCapture()
        stream = PhoneStream(CameraConfig(stale_after=.001), lambda _: cap)
        class SlowDetector(FakeDetector):
            def predict(self, frame):
                time.sleep(.06)
                return super().predict(frame)
        verifier = TemporalVerifier(VerificationConfig(required=1))
        with patch.object(main, "PersonDetector", return_value=SlowDetector()), \
             patch.object(main, "PhoneStream", return_value=stream), \
             patch.object(main, "TemporalVerifier", return_value=verifier), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main.run(["--verify", "--headless", "--seconds", ".3", "--stale-after", ".001"]), 0)
        self.assertNotIn('"event": "verification_confirmed"', output.getvalue())
        self.assertFalse(verifier._evidence)


if __name__ == "__main__":
    unittest.main()
