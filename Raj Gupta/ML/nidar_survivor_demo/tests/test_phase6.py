import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from nidar_survivor_demo import main
from nidar_survivor_demo.config import arguments, CameraConfig
from nidar_survivor_demo.camera.phone_stream import PhoneStream
from nidar_survivor_demo.reid import AppearanceMatcher, ReIDConfig, ReIDObservation, ReIDResult
from nidar_survivor_demo.survivors import SurvivorManager, ManagerConfig
from nidar_survivor_demo.mission_report import read_report
from nidar_survivor_demo.tracking import Track
from nidar_survivor_demo.verification import VerificationResult, VerifiedTrack
from .test_phase1 import FakeCapture
from .test_phase2 import FakeDetector


class Encoder:
    device = "cpu"
    def encode(self, crops):
        vector = np.zeros(512, np.float32)
        vector[0] = 1
        return [vector.copy() for _ in crops]


class Phase6Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "mission.sqlite3"
        self.matcher = AppearanceMatcher(ReIDConfig(), Encoder(), retain_gallery=True)
        vectors = {}
        for i in range(2):
            v = np.zeros(512, np.float32)
            v[i] = 1
            vectors[f"R{i + 1}"] = [v.tolist()] * 3
        self.matcher.restore_gallery(vectors)
        self.manager = SurvivorManager(self.path, self.matcher)
        self.n = 0

    def tearDown(self):
        self.manager.close()
        self.temp.cleanup()

    def update(self, pairs=(("E1:T1", "R1"),), verified=True, state="NEW_REFERENCE", fresh=True):
        self.n += 1
        temporal = VerificationResult(tuple(VerifiedTrack(Track(label, (1, 2, 70, 200), .9),
                                         "CONFIRMED" if verified else "VERIFYING", 3, True) for label, ref in pairs), (), 0)
        reid = ReIDResult(tuple(ReIDObservation(label, ref, state, "test", .95, .2) for label, ref in pairs), (), 0, 0)
        return self.manager.update(temporal, reid, sequence=self.n, received_at=self.n * .1, fresh=fresh)

    def enroll(self, pairs=(("E1:T1", "R1"),)):
        for _ in range(3):
            result = self.update(pairs)
        return result

    def test_three_hits_required_and_one_frame_never_counts(self):
        self.assertEqual(self.update().unique_survivors, 0)
        self.assertEqual(self.update().unique_survivors, 0)
        result = self.update()
        self.assertEqual(result.unique_survivors, 1)
        self.assertEqual(result.track_survivors, {"E1:T1": "S1"})

    def test_unknown_and_unverified_do_not_count(self):
        for _ in range(5):
            result = self.update((("E1:T1", None),), state="UNCERTAIN")
            self.assertEqual(result.unique_survivors, 0)
            self.assertEqual(result.pending_persons, 1)
        for _ in range(5):
            self.assertEqual(self.update(verified=False).unique_survivors, 0)

    def test_away_return_keeps_count_and_duplicate_event_exactly_once(self):
        self.enroll()
        self.update(())
        result = self.update((("E1:T9", "R1"),), state="MATCHED")
        self.assertEqual((result.unique_survivors, result.duplicates_prevented), (1, 1))
        self.assertEqual(result.track_survivors["E1:T9"], "S1")
        for _ in range(15):
            self.assertEqual(self.update((("E1:T9", "R1"),), state="MATCHED").duplicates_prevented, 1)

    def test_same_tracker_id_reappearance_counts_one_return_episode(self):
        self.enroll()
        self.update(())
        self.assertEqual(self.update(state="MATCHED").duplicates_prevented, 1)

    def test_second_person_new_then_first_returns(self):
        self.enroll()
        result = self.enroll((("E1:T1", "R1"), ("E1:T2", "R2")))
        self.assertEqual((result.unique_survivors, result.current_persons), (2, 2))
        self.update(())
        result = self.update((("E1:T3", "R1"),), state="MATCHED")
        self.assertEqual((result.unique_survivors, result.duplicates_prevented), (2, 1))

    def test_conflicting_references_are_not_merged(self):
        result = self.enroll((("E1:T1", "R1"), ("E1:T2", "R1")))
        self.assertEqual(result.unique_survivors, 0)
        self.assertEqual(result.pending_persons, 2)

    def test_missing_reference_cannot_create_survivor(self):
        self.assertEqual(self.enroll((("E1:T1", "R99"),)).unique_survivors, 0)

    def test_loss_resets_candidate_hits(self):
        self.update()
        self.update()
        self.update(())
        self.assertEqual(self.update().unique_survivors, 0)

    def test_temporal_flicker_does_not_increment_duplicates(self):
        self.enroll()
        self.update(verified=False)
        result = self.update()
        self.assertEqual(result.duplicates_prevented, 0)

    def test_stale_cannot_create_or_increment_and_current_is_unknown(self):
        self.update()
        self.update()
        result = self.update(fresh=False)
        self.assertIsNone(result.current_persons)
        self.assertEqual(result.unique_survivors, 0)
        self.assertEqual(self.update().unique_survivors, 0)

    def test_outage_retains_records_and_null_current(self):
        self.enroll()
        result = self.manager.unavailable()
        self.assertEqual(result.unique_survivors, 1)
        self.assertIsNone(result.current_persons)
        self.assertFalse(self.manager.unavailable().events)
        self.assertEqual(self.update().duplicates_prevented, 1)

    def test_resume_preserves_ids_and_does_not_reuse_track_namespace(self):
        self.enroll()
        self.manager.close()
        matcher = AppearanceMatcher(ReIDConfig(), Encoder(), retain_gallery=True)
        self.manager = SurvivorManager(self.path, matcher, resume=True)
        self.matcher = matcher
        self.n = 0
        result = self.update(state="MATCHED")
        self.assertEqual(result.track_survivors["E1:T1"], "S1")
        self.assertEqual(result.duplicates_prevented, 1)
        record = self.manager.snapshot()["records"][0]
        self.assertEqual(len(record["track_history"]), 2)
        self.assertNotEqual(*record["track_history"])

    def test_repeated_sequence_rejected_without_duplicate_write(self):
        self.enroll()
        before = self.manager.snapshot()
        self.n -= 1
        with self.assertRaises(ValueError):
            self.update()
        self.assertEqual(self.manager.snapshot(), before)

    def test_sqlite_integrity_and_audit(self):
        self.enroll()
        self.update(())
        self.update((("E1:T9", "R1"),), state="MATCHED")
        with contextlib.closing(sqlite3.connect(self.path)) as db:
            self.assertEqual(db.execute("PRAGMA integrity_check").fetchone()[0], "ok")
            events = [json.loads(row[0]) for row in db.execute("SELECT payload FROM events ORDER BY id")]
            self.assertEqual(sum(e["event"] == "survivor_created" for e in events), 1)
            self.assertEqual(sum(e["event"] == "duplicate_prevented" for e in events), 1)
            self.assertTrue(all("mission_id" in e and "recorded_at" in e for e in events))

    def test_history_is_bounded_and_snapshot_is_detached(self):
        self.manager.config = replace(self.manager.config, max_observations_per_survivor=2, max_track_history=2, observation_interval=.01)
        self.enroll()
        for i in range(10):
            self.update(())
            self.update(((f"E1:T{i + 2}", "R1"),), state="MATCHED")
        records = self.manager.snapshot()["records"]
        self.assertEqual(len(records[0]["observations"]), 2)
        self.assertEqual(len(records[0]["track_history"]), 2)
        records[0]["survivor_id"] = "corrupted"
        self.assertEqual(self.manager.snapshot()["records"][0]["survivor_id"], "S1")

    def test_existing_database_and_second_writer_rejected(self):
        with self.assertRaises(FileExistsError):
            SurvivorManager(self.path, self.matcher)
        self.manager.close()
        with self.assertRaises(FileExistsError):
            SurvivorManager(self.path, self.matcher)
        self.assertTrue(self.path.exists())

    def test_incompatible_resume_rejected(self):
        self.manager.close()
        matcher = AppearanceMatcher(replace(ReIDConfig(), match_threshold=.9), Encoder(), retain_gallery=True)
        with self.assertRaises(ValueError):
            SurvivorManager(self.path, matcher, resume=True)

    def test_database_failure_cannot_advance_memory_count(self):
        self.update()
        self.update()
        self.manager._db.execute("PRAGMA query_only=ON")
        with self.assertRaises(sqlite3.OperationalError):
            self.update()
        self.assertEqual(self.manager.snapshot()["unique_survivors"], 0)

    def test_retained_gallery_no_ttl_for_managed_mission(self):
        self.matcher.update(np.zeros((20, 20, 3), np.uint8), VerificationResult((), (), 0),
                            sequence=1, received_at=10000, epoch=2)
        self.assertEqual(len(self.matcher.gallery), 2)

    def test_read_only_report_omits_embeddings_and_unknown_current(self):
        self.enroll()
        report = read_report(self.path, 2)
        self.assertEqual(report["unique_survivors"], 1)
        self.assertIsNone(report["current_persons"])
        self.assertLessEqual(len(report["recent_events"]), 2)
        self.assertNotIn("gallery", report)
        self.assertNotIn("vectors", json.dumps(report))
        self.assertFalse(read_report(self.path, 0)["recent_events"])

    def test_event_capacity_stops_without_erasing_audit(self):
        self.manager.config = replace(self.manager.config, max_events=self.manager._events_count)
        with self.assertRaises(RuntimeError):
            self.update()
        self.assertEqual(self.manager.snapshot()["unique_survivors"], 0)

    def test_invalid_resume_gallery_fails_without_count_reset(self):
        self.enroll()
        self.manager.close()
        with contextlib.closing(sqlite3.connect(self.path)) as db:
            db.execute("UPDATE metadata SET value='{}' WHERE key='gallery'")
            db.commit()
        matcher = AppearanceMatcher(ReIDConfig(), Encoder(), retain_gallery=True)
        with self.assertRaises(ValueError):
            SurvivorManager(self.path, matcher, resume=True)
        self.assertEqual(read_report(self.path)["unique_survivors"], 1)

    def test_resume_missing_db_does_not_create_empty_mission(self):
        missing = Path(self.temp.name) / "missing.sqlite3"
        with self.assertRaises(ValueError):
            SurvivorManager(missing, AppearanceMatcher(ReIDConfig(), Encoder(), retain_gallery=True), resume=True)
        self.assertFalse(missing.exists())

    def test_config_and_cli_validation(self):
        args = arguments(["--manage"])[1]
        self.assertTrue(args.reid and args.verify and args.track and args.detect)
        for argv in (["--resume-mission"], ["--manage", "--resume-mission"], ["--mission-db", "a.sqlite3"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                arguments(argv)
        for kwargs in ({"new_identity_hits": True}, {"max_events": 0}, {"observation_interval": float("nan")}):
            with self.assertRaises(ValueError):
                ManagerConfig(**kwargs)

    def test_main_end_to_end_persists_without_images(self):
        cap = FakeCapture()
        path = Path(self.temp.name) / "main.sqlite3"
        config = replace(ReIDConfig(), sample_interval=.001, retry_interval=.001)
        with patch.object(main, "AppearanceEncoder", return_value=Encoder()), \
             patch.object(main.ReIDConfig, "load", return_value=config), \
             patch.object(main, "PersonDetector", return_value=FakeDetector()), \
             patch.object(main, "PhoneStream", return_value=PhoneStream(CameraConfig(), lambda _: cap)), \
             patch("nidar_survivor_demo.reid.quality_crop", return_value=(np.ones((100, 50, 3), np.uint8), "good")), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main.run(["--manage", "--mission-db", str(path), "--headless", "--seconds", ".6"]), 0)
        with contextlib.closing(sqlite3.connect(path)) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM survivors").fetchone()[0], 1)
        self.assertTrue(cap.released)
        self.assertFalse(path.with_suffix(".sqlite3.writer-lock").exists())


if __name__ == "__main__":
    unittest.main()
