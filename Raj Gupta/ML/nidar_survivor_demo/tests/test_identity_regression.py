"""Regression coverage for live false splits and clipped enrollment."""
import unittest
import numpy as np
from . import test_phase5
from nidar_survivor_demo.reid import quality_crop
from nidar_survivor_demo.tracking import Track


class IdentityRegressionTests(unittest.TestCase):
    setUp = test_phase5.Phase5Tests.setUp
    update = test_phase5.Phase5Tests.update
    enroll = test_phase5.Phase5Tests.enroll
    def test_managed_dissimilar_return_does_not_inflate_gallery(self):
        self.matcher.retain_gallery = True
        self.enroll()
        self.update(())
        self.encoder.vector = np.array([0., 1., 0.])
        result = self.enroll("E1:T9")
        self.assertIsNone(result.observations[0].reference_id)
        self.assertEqual(result.observations[0].reason, "confirming_new_person")
        self.assertEqual(len(self.matcher.gallery), 1)
        # A later usable view still recovers the original ID; never force a merge.
        self.encoder.vector = np.array([1., 0., 0.])
        result = self.enroll("E1:T9")
        self.assertEqual(result.observations[0].reference_id, "R1")

    def test_managed_distinct_covisible_person_can_enroll(self):
        self.matcher.retain_gallery = True
        self.enroll()
        self.encoder.vector = np.array([0., 1., 0.])
        for _ in range(25):
            result = self.update(("E1:T1", "E1:T2"))
        self.assertEqual(result.observations[1].reference_id, "R2")

    def test_sequential_new_person_enrolls_without_original_present(self):
        self.matcher.retain_gallery = True
        self.enroll()
        self.update(())
        self.encoder.vector = np.array([0., 1., 0.])
        for _ in range(25):
            result = self.update(("E1:T2",))
        self.assertEqual(result.observations[0].reference_id, "R2")
        self.update(())
        self.encoder.vector = np.array([1., 0., 0.])
        self.assertEqual(self.enroll("E1:T3").observations[0].reference_id, "R1")
        self.assertEqual(len(self.matcher.gallery), 2)

    def test_bad_quality_resets_novelty_evidence(self):
        from unittest.mock import patch
        self.matcher.retain_gallery = True
        self.enroll()
        self.update(())
        self.encoder.vector = np.array([0., 1., 0.])
        self.enroll("E1:T2")
        with patch("nidar_survivor_demo.reid.quality_crop", return_value=(None, "blurred_crop")):
            self.update(("E1:T2",))
        self.assertIsNone(self.matcher._pending["E1:T2"].novelty_started)
        self.assertEqual(len(self.matcher._pending["E1:T2"].vectors), 3)

    def test_sequential_database_commit_and_restart_return(self):
        import tempfile
        import sqlite3
        from contextlib import closing
        from pathlib import Path
        from nidar_survivor_demo.reid import AppearanceMatcher
        from nidar_survivor_demo.survivors import SurvivorManager
        from nidar_survivor_demo.verification import VerificationResult, VerifiedTrack
        self.encoder.encode = lambda crops: [np.pad(self.encoder.vector, (0, 509)) for _ in crops]
        self.matcher.retain_gallery = True
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "mission.sqlite3"
            manager = SurvivorManager(path, self.matcher)
            def step(label):
                self.n += 1
                verified = VerificationResult((VerifiedTrack(Track(label, (10, 10, 110, 300), .9), "CONFIRMED", 3, True),) if label else (), (), 0)
                found = self.matcher.update(self.frame, verified, sequence=self.n, received_at=self.n * .1, epoch=1)
                return manager.update(verified, found, sequence=self.n, received_at=self.n * .1)
            try:
                for _ in range(6):
                    step("E1:T1")
                step(None)
                self.encoder.vector = np.array([0., 1., 0.])
                for _ in range(30):
                    result = step("E1:T2")
                self.assertEqual(result.unique_survivors, 2)
            finally:
                manager.close()
            with closing(sqlite3.connect(path)) as db:
                self.assertEqual(db.execute("PRAGMA integrity_check").fetchone()[0], "ok")
                self.assertEqual(db.execute("SELECT COUNT(*) FROM survivors").fetchone()[0], 2)
            self.matcher = AppearanceMatcher(self.config, self.encoder, retain_gallery=True)
            manager = SurvivorManager(path, self.matcher, resume=True)
            try:
                self.encoder.vector = np.array([1., 0., 0.])
                for _ in range(6):
                    result = step("E2:T9")
                self.assertEqual(result.track_survivors["E2:T9"], "S1")
                self.assertEqual(result.unique_survivors, 2)
                self.assertEqual(result.duplicates_prevented, 1)
            finally:
                manager.close()

    def test_same_frame_similar_new_candidates_not_two_records(self):
        for _ in range(3):
            result = self.update(("E1:T1", "E1:T2"))
        self.assertEqual(len(self.matcher.gallery), 1)
        self.assertIsNone(result.observations[1].reference_id)

    def test_already_clipped_detector_box_is_rejected(self):
        for box in ((0, 10, 110, 300), (10, 0, 110, 300), (390, 10, 500, 300), (10, 10, 110, 320)):
            track = Track("E1:T1", box, .9)
            self.assertEqual(quality_crop(self.frame, track, [track], self.config)[1], "frame_edge_crop")
