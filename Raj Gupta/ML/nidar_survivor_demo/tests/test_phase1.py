from dataclasses import replace
import contextlib
import io
import os
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
from nidar_survivor_demo.config import CameraConfig, arguments, parse_source
from nidar_survivor_demo.camera.phone_stream import PhoneStream
from nidar_survivor_demo.utils.fps import FPSCounter
from nidar_survivor_demo.preview import render
from nidar_survivor_demo import main


def until(predicate, seconds=2):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("Condition did not become true within test budget")


class FakeCapture:
    def __init__(self, fail_after=None, block=None, opened=True):
        self.count = 0
        self.fail_after = fail_after
        self.block = block
        self.opened = opened
        self.released = False

    def isOpened(self):
        return self.opened

    def get(self, prop):
        return 30

    def read(self):
        time.sleep(0.005)
        self.count += 1
        if self.block and self.count > 1:
            self.block.wait(2)
        if self.fail_after is not None and self.count > self.fail_after:
            return False, None
        return True, np.full((24, 32, 3), self.count % 255, np.uint8)

    def release(self):
        self.released = True


class Phase1Tests(unittest.TestCase):
    def test_sources(self):
        self.assertEqual(parse_source(" 2 "), 2)
        self.assertEqual(parse_source("http://localhost:1234/video"), "http://localhost:1234/video")
        for invalid in ("", "-1", "rtsp://", "ftp://localhost", "http://host:bad", "missing.avi"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                parse_source(invalid)

    def test_cli_overrides_environment(self):
        with patch.dict(os.environ, {"NIDAR_CAMERA_SOURCE": "3"}):
            self.assertEqual(arguments([])[0].source, 3)
            self.assertEqual(arguments(["--source", "1"])[0].source, 1)

    def test_display_rate_is_independent_and_validated(self):
        config, args = arguments(["--fps", "30", "--display-fps", "60"])
        self.assertEqual(config.fps, 30)
        self.assertEqual(args.display_fps, 60)
        for value in ("0", "-1", "241"):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                arguments(["--display-fps", value])

    def test_render_outputs_are_independent(self):
        from nidar_survivor_demo.camera.phone_stream import Snapshot
        snapshot = Snapshot("CONNECTING", None, 0, None, 0, 0, 0)
        first = render(snapshot, 0)
        expected = first.copy()
        first[:] = 0
        self.assertTrue(np.array_equal(render(snapshot, 0), expected))

    def test_invalid_configuration(self):
        for field, value in (("width", 0), ("fps", -1), ("stale_after", float("nan")),
                             ("reconnect_delay", float("inf")), ("decoder_threads", -1),
                             ("decoder_threads", 33)):
            with self.assertRaises(ValueError):
                replace(CameraConfig(), **{field: value})

    def test_fps_and_expiry(self):
        fps = FPSCounter()
        for n in range(31):
            fps.tick(n / 30)
        self.assertAlmostEqual(fps.value(1), 30)
        self.assertEqual(fps.value(4), 0)

    def test_latest_frame_skips_backlog(self):
        cap = FakeCapture()
        stream = PhoneStream(CameraConfig(), lambda _: cap)
        stream.start()
        try:
            until(lambda: stream.snapshot().frames >= 2)
            old = stream.snapshot().packet
            time.sleep(0.15)  # Consumer deliberately slower than capture.
            new = stream.snapshot().packet
            self.assertGreater(new.sequence, old.sequence + 5)
            self.assertEqual(new.sequence, stream.snapshot().frames)
            self.assertLess(stream.snapshot().age_ms, 100)
            render(stream.snapshot(), 30)
            self.assertTrue(np.all(new.frame == new.sequence % 255))
        finally:
            self.assertTrue(stream.stop())
        self.assertTrue(cap.released)
        self.assertTrue(stream.stop())
        self.assertIsNone(stream.snapshot().packet)

    def test_reconnect_releases_previous_capture(self):
        first, second = FakeCapture(fail_after=2), FakeCapture()
        captures = iter([first, second])
        stream = PhoneStream(CameraConfig(reconnect_delay=0.02), lambda _: next(captures))
        stream.start()
        try:
            until(lambda: stream.snapshot().connections == 2)
            self.assertTrue(first.released)
            self.assertEqual(stream.snapshot().state, "ONLINE")
        finally:
            self.assertTrue(stream.stop())
        self.assertTrue(second.released)

    def test_stale_frame_not_returned_and_blocked_stop_reported(self):
        unblock = threading.Event()
        cap = FakeCapture(block=unblock)
        stream = PhoneStream(CameraConfig(stale_after=0.03), lambda _: cap)
        stream.start()
        try:
            until(lambda: stream.snapshot().state == "STALE")
            self.assertIsNone(stream.snapshot().packet)
            self.assertEqual(stream.snapshot().capture_fps, 0)
            self.assertFalse(stream.stop(timeout=0.01))
        finally:
            unblock.set()
            self.assertTrue(stream.stop())
        self.assertTrue(cap.released)

    def test_stop_interrupts_reconnect_delay(self):
        cap = FakeCapture(opened=False)
        stream = PhoneStream(CameraConfig(reconnect_delay=30), lambda _: cap)
        stream.start()
        until(lambda: stream.snapshot().state == "RECONNECTING")
        started = time.monotonic()
        self.assertTrue(stream.stop(timeout=0.2))
        self.assertLess(time.monotonic() - started, 0.2)
        self.assertTrue(cap.released)

    def test_invalid_empty_frame_reconnects(self):
        first, second = FakeCapture(), FakeCapture()
        first.read = lambda: (True, np.empty((0, 0, 3), np.uint8))
        captures = iter([first, second])
        stream = PhoneStream(CameraConfig(reconnect_delay=0.01), lambda _: next(captures))
        stream.start()
        try:
            until(lambda: stream.snapshot().state == "ONLINE")
            self.assertTrue(first.released)
            self.assertEqual(stream.snapshot().attempts, 2)
        finally:
            stream.stop()

    def test_ctrl_c_releases_capture(self):
        cap = FakeCapture()
        stream = PhoneStream(CameraConfig(), lambda _: cap)
        with patch.object(main, "PhoneStream", return_value=stream), \
             patch.object(main.cv2, "namedWindow"), patch.object(main.cv2, "resizeWindow"), \
             patch.object(main.cv2, "imshow"), patch.object(main.cv2, "waitKey", side_effect=KeyboardInterrupt), \
             patch.object(main.cv2, "destroyAllWindows") as destroy, contextlib.redirect_stdout(io.StringIO()):
            main.run([])
        self.assertEqual(stream.snapshot().state, "STOPPED")
        self.assertTrue(cap.released)
        destroy.assert_called_once()

    def test_exception_does_not_log_credentials(self):
        def broken(_):
            raise RuntimeError("rtsp://secret:password@host/video?token=private")
        with self.assertLogs("nidar_survivor_demo.camera.phone_stream", "INFO") as output:
            stream = PhoneStream(CameraConfig(), broken)
            stream.start()
            until(lambda: stream.snapshot().state == "RECONNECTING")
            stream.stop()
        self.assertNotIn("password", str(output.output))
        self.assertNotIn("private", str(output.output))

    def test_window_exit_routes_cleanup(self):
        for key, visible in ((ord("q"), 1), (ord("Q"), 1), (27, 1), (-1, 0)):
            with self.subTest(key=key, visible=visible):
                cap = FakeCapture()
                stream = PhoneStream(CameraConfig(), lambda _: cap)
                with patch.object(main, "PhoneStream", return_value=stream), \
                     patch.object(main.cv2, "namedWindow"), patch.object(main.cv2, "resizeWindow"), \
                     patch.object(main.cv2, "imshow"), patch.object(main.cv2, "waitKey", return_value=key), \
                     patch.object(main.cv2, "getWindowProperty", return_value=visible), \
                     patch.object(main.cv2, "destroyAllWindows") as destroy, contextlib.redirect_stdout(io.StringIO()):
                    main.run(["--seconds", "1"])
                destroy.assert_called_once()
                self.assertEqual(stream.snapshot().state, "STOPPED")


if __name__ == "__main__":
    unittest.main()
