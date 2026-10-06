"""Performance changes must preserve freshness and avoid duplicate encoding."""
import unittest
from unittest.mock import patch
import numpy as np
from fastapi.testclient import TestClient
from nidar_survivor_demo.dashboard import DashboardState, create_app
from nidar_survivor_demo.config import arguments


class PerformanceTests(unittest.TestCase):
    def test_limits(self):
        for value in (0, 61, True, 1.5):
            with self.assertRaises(ValueError):
                DashboardState(preview_fps=value)
        _, args = arguments(["--source", "0", "--dashboard-fps", "24", "--cpu-threads", "2"])
        self.assertEqual(args.dashboard_fps, 24)
        for flag in ("--dashboard-fps", "--cpu-threads"):
            with self.assertRaises(SystemExit):
                arguments([flag, "0"])

    def test_new_frames_not_duplicate_frames(self):
        state = DashboardState()
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        with patch("nidar_survivor_demo.dashboard.time.monotonic", return_value=10):
            state.publish(status="ONLINE", frame=frame, frame_at=10)
        with patch("nidar_survivor_demo.dashboard.time.monotonic", return_value=10.04):
            state.publish(status="ONLINE", frame=frame, frame_at=10)
            self.assertEqual(state.snapshot()["frame_sequence"], 1)
            state.publish(status="ONLINE", frame=frame, frame_at=10.04)
            self.assertEqual(state.snapshot()["frame_sequence"], 2)
            with TestClient(create_app(state, 8766), base_url="http://127.0.0.1:8766") as client:
                response = client.get("/api/frame")
                self.assertEqual(response.headers["x-frame-sequence"], "2")
        with patch("nidar_survivor_demo.dashboard.time.monotonic", return_value=12):
            self.assertIsNone(state.frame(with_sequence=True))
            self.assertFalse(state.snapshot()["frame_available"])

    def test_encoding_ceiling(self):
        state = DashboardState(preview_fps=30)
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        for i in range(100):
            now = 10 + i / 100
            with patch("nidar_survivor_demo.dashboard.time.monotonic", return_value=now):
                state.publish(status="ONLINE", frame=frame, frame_at=now)
        self.assertLessEqual(state._data["frame_sequence"], 30)
        self.assertGreater(state._data["frame_sequence"], 10)


if __name__ == "__main__":
    unittest.main()
