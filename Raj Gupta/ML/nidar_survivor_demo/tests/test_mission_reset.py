"""Explicit reset, CSRF/mission guards, archive preservation and fresh identity state."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient
from nidar_survivor_demo.dashboard import DashboardState, create_app
from nidar_survivor_demo.mission_control import rotate_mission
from nidar_survivor_demo.survivors import SurvivorManager
from nidar_survivor_demo.reid import AppearanceMatcher, ReIDConfig
from nidar_survivor_demo.tracking import PersonTracker, TrackerConfig
from nidar_survivor_demo.verification import TemporalVerifier, VerificationConfig
from nidar_survivor_demo.mission_report import read_report
from .test_phase6 import Encoder


class ResetTests(unittest.TestCase):
    def setUp(self):
        self.state = DashboardState(allow_reset=True)
        self.state.publish(status="ONLINE", mission={"mission_id": "old", "records": []}, counts={"unique_survivors": 2})
        self.client = TestClient(create_app(self.state, 8766), base_url="http://127.0.0.1:8766")
        self.headers = {"origin": "http://127.0.0.1:8766", "x-reset-token": self.state.reset_token, "x-mission-id": "old"}

    def tearDown(self):
        self.client.close()

    def test_refresh_never_resets(self):
        for _ in range(3):
            self.assertEqual(self.client.get("/").status_code, 200)
            self.assertEqual(self.client.get("/api/state").json()["counts"]["unique_survivors"], 2)
        self.assertFalse(self.state.take_reset())

    def test_explicit_reset_guard_and_single_consumption(self):
        self.assertEqual(self.client.post("/api/mission/reset").status_code, 403)
        self.assertEqual(self.client.post("/api/mission/reset", headers={**self.headers, "origin": "https://evil.test"}).status_code, 403)
        self.assertEqual(self.client.post("/api/mission/reset", headers={**self.headers, "x-mission-id": "stale"}).status_code, 409)
        self.assertEqual(self.client.post("/api/mission/reset", headers=self.headers).status_code, 202)
        self.assertEqual(self.client.post("/api/mission/reset", headers=self.headers).status_code, 409)
        self.assertTrue(self.state.take_reset())
        self.assertFalse(self.state.take_reset())

    def test_archive_and_new_mission_guard(self):
        self.state.request_reset("old")
        self.state.take_reset()
        self.state.publish(status="STARTING", mission={"mission_id": "new", "records": []}, counts={"unique_survivors": 0})
        self.state.complete_reset({"mission_id": "old", "unique_survivors": 2, "duplicates_prevented": 1, "meaning": "estimate", "records": []})
        self.assertEqual(self.client.get("/api/archive/old").json()["unique_survivors"], 2)
        self.assertEqual(self.client.get("/api/archive/unknown").status_code, 404)
        self.assertEqual(self.client.post("/api/mission/reset", headers=self.headers).status_code, 409)
        self.assertNotIn("reset", self.client.get("/api/report").json())

    def test_archive_cannot_reset(self):
        state = DashboardState("ARCHIVE", allow_reset=True)
        self.assertFalse(state.snapshot()["reset"]["available"])
        self.assertEqual(state.request_reset("anything"), 409)

    def test_rotation_preserves_database_and_replaces_all_state(self):
        with tempfile.TemporaryDirectory() as folder:
            old_path, new_path = Path(folder)/"old.sqlite3", Path(folder)/"new.sqlite3"
            matcher = AppearanceMatcher(ReIDConfig(), Encoder(), retain_gallery=True)
            manager = SurvivorManager(old_path, matcher)
            tracker, verifier = PersonTracker(TrackerConfig()), TemporalVerifier(VerificationConfig())
            old_id = manager.mission_id
            with patch("nidar_survivor_demo.mission_control.new_mission_path", return_value=new_path):
                new, fresh, track, verify, report = rotate_mission(manager, tracker, verifier)
            try:
                self.assertNotEqual(new.mission_id, old_id)
                self.assertEqual(new.snapshot()["unique_survivors"], 0)
                self.assertFalse(fresh.gallery)
                self.assertIsNot(track, tracker)
                self.assertIsNot(verify, verifier)
                self.assertIs(fresh.encoder, matcher.encoder)
                self.assertEqual(read_report(old_path)["mission_id"], old_id)
                self.assertEqual(read_report(old_path)["recent_events"][-1]["event"], "mission_closed")
            finally:
                new.close()

    def test_new_database_failure_keeps_old_open(self):
        with tempfile.TemporaryDirectory() as folder:
            manager = SurvivorManager(Path(folder)/"old.sqlite3", AppearanceMatcher(ReIDConfig(), Encoder(), retain_gallery=True))
            try:
                with patch("nidar_survivor_demo.mission_control.SurvivorManager", side_effect=OSError("disk")):
                    with self.assertRaises(OSError):
                        rotate_mission(manager, PersonTracker(TrackerConfig()), TemporalVerifier(VerificationConfig()))
                self.assertFalse(manager._closed)
            finally:
                manager.close()
