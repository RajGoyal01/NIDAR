"""Dashboard contract, privacy, lifecycle and pipeline integration checks."""
import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import urllib.request
import urllib.error
import numpy as np
from fastapi.testclient import TestClient
from nidar_survivor_demo.dashboard import DashboardState, DashboardServer, create_app, annotate_dashboard
from nidar_survivor_demo.config import arguments, CameraConfig
from nidar_survivor_demo import main
from nidar_survivor_demo.camera.phone_stream import PhoneStream
from nidar_survivor_demo.reid import ReIDConfig
from .test_phase1 import FakeCapture
from .test_phase2 import FakeDetector
from .test_phase6 import Encoder


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.state = DashboardState()
        self.client = TestClient(create_app(self.state, 8765), base_url="http://127.0.0.1:8765")
        self.counts = dict(current_persons=1, resolved_visible=1, pending_persons=0, unique_survivors=2, duplicates_prevented=1)
        self.mission = {"mission_id": "test", "gallery": {"private": [1]}, "records": [
            {"survivor_id": "S1", "reference_id": "R1", "status": "VISIBLE", "embeddings": [1], "observations": ["private"]}]}

    def tearDown(self):
        self.client.close()

    def publish(self, **kwargs):
        self.state.publish(status="ONLINE", counts=self.counts, mission=self.mission, **kwargs)

    def test_startup_unknown_not_zero(self):
        result = self.client.get("/api/state").json()
        self.assertIsNone(result["counts"]["current_persons"])
        self.assertFalse(result["frame_available"])

    def test_counts_preserved_not_recomputed(self):
        self.publish()
        self.assertEqual(self.client.get("/api/state").json()["counts"], self.counts)

    def test_private_fields_not_exported(self):
        self.publish(events=[{"event": "survivor_created", "credential": "secret"}])
        response = self.client.get("/api/report")
        for forbidden in ("embeddings", "gallery", "observations", "credential", "secret"):
            self.assertNotIn(forbidden, response.text)
        self.assertIn("attachment", response.headers["content-disposition"])

    def test_stale_clears_current_but_preserves_totals(self):
        self.publish(modules={"YOLO": "ACTIVE"}, metrics={"capture_fps": 30})
        self.state._updated = time.monotonic() - 10
        data = self.state.snapshot()
        self.assertEqual(data["status"], "STALE")
        self.assertEqual(data["counts"]["unique_survivors"], 2)
        self.assertIsNone(data["counts"]["current_persons"])
        self.assertEqual(data["records"][0]["status"], "UNKNOWN")
        self.assertIsNone(data["metrics"]["capture_fps"])

    def test_latest_frame_and_no_stale_frame(self):
        self.publish(frame=np.zeros((60, 80, 3), np.uint8))
        self.assertEqual(self.client.get("/api/frame").status_code, 200)
        self.state._frame_at -= 10
        self.assertEqual(self.client.get("/api/frame").status_code, 204)
        self.assertFalse(self.state.snapshot()["frame_available"])

    def test_unavailable_and_recovery(self):
        self.publish(frame=np.zeros((60, 80, 3), np.uint8))
        self.state.stop("RECONNECTING")
        self.assertIsNone(self.state.frame())
        self.assertIsNone(self.state.snapshot()["counts"]["pending_persons"])
        self.state._last_encode = 0
        self.publish(frame=np.zeros((60, 80, 3), np.uint8))
        self.assertTrue(self.state.snapshot()["frame_available"])

    def test_event_ring_bounded_and_detached(self):
        self.publish(events=[{"event": "identity_pending", "sequence": i} for i in range(150)])
        value = self.state.snapshot()
        self.assertEqual(len(value["events"]), 100)
        self.assertEqual(value["events"][0]["sequence"], 50)
        value["records"].clear()
        self.assertEqual(len(self.state.snapshot()["records"]), 1)

    def test_archive_never_live(self):
        state = DashboardState("ARCHIVE")
        state.publish(status="ARCHIVE", counts=self.counts, mission=self.mission)
        self.assertIsNone(state.snapshot()["counts"]["current_persons"])
        self.assertFalse(state.snapshot()["frame_available"])

    def test_switching_missions_drops_previous_mission_events(self):
        self.publish(events=[{"event": "survivor_created"}])
        self.state.publish(status="ONLINE", mission={"mission_id": "different", "records": []})
        self.assertEqual(self.state.snapshot()["events"], [])

    def test_host_origin_and_cross_site_denied(self):
        for headers in ({"host": "evil.example:8765"}, {"origin": "https://evil.example"}, {"sec-fetch-site": "cross-site"}):
            self.assertEqual(self.client.get("/api/state", headers=headers).status_code, 403)

    def test_no_write_routes_no_filesystem_access(self):
        self.assertEqual(self.client.post("/api/state", json={"unique_survivors": 999}).status_code, 405)
        for url in ("/settings/reid.json", "/api/mission?path=C:/Windows/win.ini", "/models", "/docs", "/openapi.json"):
            self.assertEqual(self.client.get(url).status_code, 404)

    def test_assets_security_headers(self):
        for path in ("/", "/style.css", "/app.js"):
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["cache-control"], "no-store")
            self.assertIn("frame-ancestors 'none'", response.headers["content-security-policy"])
        self.assertIn("Identity register", self.client.get("/").text)

    def test_server_shutdown_releases_port(self):
        server = DashboardServer(self.state, 0)
        port = server.port
        try:
            with urllib.request.urlopen(server.url + "/api/state", timeout=3) as response:
                self.assertEqual(response.status, 200)
            with self.assertRaises(OSError):
                DashboardServer(self.state, port)
        finally:
            server.close()
        self.assertFalse(server.thread.is_alive())
        again = DashboardServer(self.state, port)
        again.close()

    def test_cli_implies_manager_and_validates_port(self):
        parsed = arguments(["--dashboard"])[1]
        self.assertTrue(parsed.manage and parsed.reid and parsed.verify and parsed.track)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            arguments(["--dashboard-port", "80"])

    def test_browser_uses_text_not_injected_html(self):
        source = self.client.get("/app.js").text
        self.assertNotIn("innerHTML", source)
        self.assertIn("textContent", source)
        self.assertIn("AbortSignal.timeout", source)

    def test_file_mode_is_explicitly_not_a_live_phone(self):
        state = DashboardState("LOCAL VIDEO")
        self.assertEqual(state.snapshot()["mode"], "LOCAL VIDEO")
        self.assertIn("Full perception pipeline on a file", self.client.get("/app.js").text)

    def test_dashboard_annotation_does_not_modify_source(self):
        from types import SimpleNamespace
        from nidar_survivor_demo.tracking import Track
        image = np.zeros((100, 160, 3), np.uint8)
        verified = SimpleNamespace(tracks=[SimpleNamespace(track=Track("E1:T1", (0, 0, 155, 90), .9), state="CONFIRMED", hits=3)])
        managed = SimpleNamespace(track_survivors={"E1:T1": "S1"})
        result = annotate_dashboard(image, verified, managed)
        self.assertTrue(result.any())
        self.assertFalse(image.any())

    def test_main_publishes_actual_manager_and_stops_server(self):
        cap = FakeCapture()
        state = DashboardState()
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(main, "AppearanceEncoder", return_value=Encoder()), \
             patch.object(main.ReIDConfig, "load", return_value=replace(ReIDConfig(), sample_interval=.001, retry_interval=.001)), \
             patch.object(main, "PersonDetector", return_value=FakeDetector()), \
             patch.object(main, "PhoneStream", return_value=PhoneStream(CameraConfig(), lambda _: cap)), \
             patch("nidar_survivor_demo.reid.quality_crop", return_value=(np.ones((100, 50, 3), np.uint8), "good")), \
             patch("nidar_survivor_demo.dashboard.DashboardState", return_value=state), \
             patch("nidar_survivor_demo.dashboard.DashboardServer") as server, \
             contextlib.redirect_stdout(io.StringIO()):
            server.return_value.url = "http://127.0.0.1:8765"
            self.assertEqual(main.run(["--dashboard", "--headless", "--seconds", ".8", "--mission-db", str(Path(temp)/"test.sqlite3")]), 0)
            server.return_value.close.assert_called_once()
        result = state.snapshot()
        self.assertEqual(result["counts"]["unique_survivors"], 1)
        self.assertEqual(result["status"], "STOPPED")
        self.assertIsNone(result["counts"]["current_persons"])


if __name__ == "__main__":
    unittest.main()
