import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from nidar_survivor_demo.config import arguments, CameraConfig
from nidar_survivor_demo.detector import Detection, DetectionResult
from nidar_survivor_demo.tracking import PersonTracker, TrackerConfig, annotate_tracks
from nidar_survivor_demo.camera.phone_stream import PhoneStream
from nidar_survivor_demo import main
from .test_phase1 import FakeCapture
from .test_phase2 import FakeDetector


def detection(x=20, score=.9):
    return DetectionResult((Detection((x, 20, x + 50, 180), score),), 1)


class Phase3Tests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((240, 320, 3), np.uint8)

    def tracker(self, **kwargs):
        return PersonTracker(TrackerConfig(gmc_method="none", **kwargs))

    def update(self, tracker, result, sequence, **kwargs):
        params = dict(sequence=sequence, received_at=sequence * .05, connection=1)
        params.update(kwargs)
        return tracker.update(result, self.frame, **params)

    def test_cli_implies_detection_and_preserves_phase2_default(self):
        _, args = arguments(["--track"])
        self.assertTrue(args.detect)
        self.assertEqual(args.confidence, .1)
        self.assertEqual(arguments(["--detect"])[1].confidence, .35)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            arguments(["--track-log", "unused.jsonl"])

    def test_config_validation_and_no_reid_keys(self):
        for kwargs in ({"track_low_thresh": .5}, {"new_track_thresh": .2}, {"track_buffer": True},
                       {"track_buffer": 0}, {"match_thresh": float("nan")}, {"fuse_score": "yes"},
                       {"gmc_method": "bad"}, {"max_gap_seconds": float("inf")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                TrackerConfig(**kwargs)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            path.write_text('{"with_reid": true}')
            with self.assertRaises(ValueError):
                TrackerConfig.load(path)

    def test_continuity_with_skipped_camera_sequences(self):
        tracker = self.tracker()
        ids = []
        for n in range(10):
            result = tracker.update(detection(20 + n * 2), self.frame,
                                    sequence=1 + n * 3, received_at=n * .05, connection=1)
            ids.append(result.tracks[0].track_id)
        self.assertEqual(len(set(ids)), 1)
        self.assertEqual(tracker.engine.frame_id, 10)
        self.assertEqual(tracker.started_tracks, 1)

    def test_low_score_recovers_but_does_not_start_track(self):
        tracker = self.tracker()
        first = self.update(tracker, detection(), 1)
        recovered = self.update(tracker, detection(21, .2), 2)
        self.assertEqual(first.tracks[0].track_id, recovered.tracks[0].track_id)
        tracker.clear()
        self.assertFalse(self.update(tracker, detection(score=.2), 3).tracks)

    def test_empty_frame_hides_track_then_refinds(self):
        tracker = self.tracker()
        first = self.update(tracker, detection(), 1)
        missing = self.update(tracker, DetectionResult((), 1), 2)
        self.assertFalse(missing.tracks)
        self.assertEqual(missing.events[0]["event"], "track_lost")
        found = self.update(tracker, detection(), 3)
        self.assertEqual(first.tracks[0].track_id, found.tracks[0].track_id)
        self.assertEqual(found.events[0]["event"], "track_refound")

    def test_lost_track_eventually_expires(self):
        tracker = self.tracker(track_buffer=2)
        old = self.update(tracker, detection(), 1).tracks[0].track_id
        for n in range(2, 8):
            self.update(tracker, DetectionResult((), 1), n)
        self.update(tracker, detection(), 8)
        new = self.update(tracker, detection(), 9).tracks[0].track_id
        self.assertNotEqual(old, new)

    def test_reset_is_idempotent_and_epoch_prevents_id_alias(self):
        tracker = self.tracker()
        old = self.update(tracker, detection(), 1).tracks[0].track_id
        self.assertIsNotNone(tracker.clear())
        self.assertIsNone(tracker.clear())
        new = self.update(tracker, detection(), 2).tracks[0].track_id
        self.assertNotEqual(old, new)
        self.assertEqual(tracker.resets, 1)

    def test_connection_shape_and_time_gaps_reset(self):
        for change in ("connection", "gap", "shape"):
            tracker = self.tracker()
            self.update(tracker, detection(), 1)
            frame = self.frame if change != "shape" else np.zeros((300, 320, 3), np.uint8)
            result = tracker.update(detection(), frame, sequence=2,
                                    received_at=2 if change == "gap" else .1,
                                    connection=2 if change == "connection" else 1)
            self.assertTrue(result.tracks[0].track_id.startswith("E2:"))
            self.assertEqual(result.events[0]["event"], "tracker_reset")

    def test_duplicate_or_backwards_frames_rejected(self):
        tracker = self.tracker()
        self.update(tracker, detection(), 2)
        for n in (2, 1):
            with self.assertRaises(ValueError):
                self.update(tracker, detection(), n)
        self.assertEqual(tracker.engine.frame_id, 1)

    def test_invalid_detections_rejected(self):
        tracker = self.tracker()
        with self.assertRaises(ValueError):
            self.update(tracker, detection(float("nan")), 1)

    def test_drawing_preserves_camera_frame(self):
        tracker = self.tracker()
        result = self.update(tracker, detection(), 1)
        annotated = annotate_tracks(self.frame, result)
        self.assertFalse(self.frame.any())
        self.assertTrue(annotated.any())

    def test_gmc_enabled_without_reid_or_encoder(self):
        tracker = PersonTracker(TrackerConfig())
        self.assertIsNone(tracker.engine.encoder)
        self.assertEqual(tracker.engine.gmc.method, "sparseOptFlow")
        for n in range(1, 4):
            self.assertTrue(self.update(tracker, detection(), n).tracks)

    def test_main_tracking_and_metadata_log(self):
        cap = FakeCapture()
        stream = PhoneStream(CameraConfig(), lambda _: cap)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tracks.jsonl"
            with patch.object(main, "PersonDetector", return_value=FakeDetector()), \
                 patch.object(main, "PhoneStream", return_value=stream), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main.run(["--track", "--headless", "--seconds", ".25", "--track-log", str(path)]), 0)
            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(records[0]["id_scope"], "temporary")
            self.assertTrue(any(record["event"] == "tracked_frame" for record in records))
            self.assertTrue(cap.released)

    def test_high_detector_threshold_rejected_before_camera(self):
        with patch.object(main, "PhoneStream") as stream:
            self.assertEqual(main.run(["--track", "--confidence", ".5", "--headless"]), 1)
            stream.assert_not_called()
