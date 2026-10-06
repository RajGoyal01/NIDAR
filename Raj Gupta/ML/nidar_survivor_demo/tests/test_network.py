"""Real HTTP MJPEG -> OpenCV FFmpeg tests, all on localhost, no phone required."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import unittest

import cv2
import numpy as np
from nidar_survivor_demo.camera.phone_stream import PhoneStream
from nidar_survivor_demo.config import CameraConfig
from .test_phase1 import until


class NetworkTests(unittest.TestCase):
    def setUp(self):
        self.stall = threading.Event()
        self.done = threading.Event()
        stall, done = self.stall, self.done
        ok, encoded = cv2.imencode(".jpg", np.full((120, 160, 3), 90, np.uint8))
        assert ok
        jpeg = encoded.tobytes()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                if self.path == "/shot.jpg":
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(jpeg)))
                    self.end_headers()
                    self.wfile.write(jpeg)
                    return
                if self.path.startswith("/missing"):
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while not done.is_set():
                        if not stall.is_set():
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " +
                                             str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                            self.wfile.flush()
                        done.wait(1 / 30)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/video"

    def tearDown(self):
        self.done.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)

    def test_real_stream_stall_reconnect_and_stop(self):
        stream = PhoneStream(CameraConfig(source=self.url, read_timeout_ms=600,
                             open_timeout_ms=1500, stale_after=0.2, reconnect_delay=0.1))
        stream.start()
        try:
            until(lambda: stream.snapshot().frames >= 20, 8)
            self.assertEqual(stream.snapshot().packet.frame.shape, (120, 160, 3))
            self.stall.set()
            until(lambda: stream.snapshot().state == "STALE", 4)
            self.assertIsNone(stream.snapshot().packet)
            until(lambda: stream.snapshot().state == "RECONNECTING", 4)
            self.stall.clear()
            until(lambda: stream.snapshot().connections >= 2, 8)
            # Allow startup probing/buffer bursts to settle before reporting FPS.
            time.sleep(2.2)
            snap = stream.snapshot()
            print(json.dumps({"integration": "HTTP MJPEG recovery", "capture_fps": snap.capture_fps,
                              "decoded_age_ms": snap.age_ms, "connections": snap.connections}))
        finally:
            started = time.monotonic()
            self.assertTrue(stream.stop())
            print(json.dumps({"stop_seconds": time.monotonic() - started}))

    def test_snapshot_transport(self):
        stream = PhoneStream(CameraConfig(source=self.url.replace("/video", "/shot.jpg"), transport="snapshot"))
        stream.start()
        try:
            until(lambda: stream.snapshot().frames >= 5, 5)
            self.assertEqual(stream.snapshot().packet.frame.shape, (120, 160, 3))
        finally:
            self.assertTrue(stream.stop())

    def test_direct_mjpeg_stale_reconnect_and_stop(self):
        stream = PhoneStream(CameraConfig(source=self.url, transport="mjpeg", stale_after=0.1,
                                         read_timeout_ms=500, reconnect_delay=0.05))
        stream.start()
        try:
            until(lambda: stream.snapshot().frames >= 5, 4)
            self.assertEqual(stream.snapshot().packet.frame.shape, (120, 160, 3))
            self.stall.set()
            until(lambda: stream.snapshot().state == "STALE", 2)
            self.assertIsNone(stream.snapshot().packet)
            until(lambda: stream.snapshot().state == "RECONNECTING", 3)
            self.stall.clear()
            until(lambda: stream.snapshot().connections >= 2, 4)
        finally:
            self.assertTrue(stream.stop())
        until(lambda: not any(t.name == "mjpeg-receiver" for t in threading.enumerate()), 2)

    def test_snapshot_rejects_non_jpeg_response(self):
        from nidar_survivor_demo.camera.snapshot_capture import SnapshotCapture
        capture = SnapshotCapture(CameraConfig(source=self.url.replace("/video", "/missing"), transport="snapshot"))
        try:
            with self.assertRaises(RuntimeError):
                capture.read()
        finally:
            capture.release()
        self.assertFalse(capture.isOpened())

    def test_real_gui_app_timed_exit(self):
        # Real OpenCV window, auto-closes; synthetic frames contain no personal data.
        result = subprocess.run([sys.executable, "-m", "nidar_survivor_demo.main", "--source", self.url,
                                 "--seconds", "3"], cwd=Path(__file__).resolve().parents[2],
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        records = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
        self.assertTrue(records[-1]["clean"])
        self.assertGreater(records[-1]["unique_frames_consumed"], 0)
        print(json.dumps({"integration": "GUI timed run", "result": records[-1]}))

    def test_unavailable_stream_returns_failure_without_credentials(self):
        result = subprocess.run([sys.executable, "-m", "nidar_survivor_demo.main", "--source",
                                 self.url.replace("/video", "/missing?token=hidden-test-token"),
                                 "--headless", "--seconds", "1", "--reconnect-delay", "0.1"],
                                cwd=Path(__file__).resolve().parents[2], capture_output=True,
                                text=True, timeout=10)
        self.assertEqual(result.returncode, 1)
        self.assertIn('"clean": true', result.stdout)
        self.assertNotIn("hidden-test-token", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
