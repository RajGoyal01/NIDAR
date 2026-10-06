import contextlib
from dataclasses import replace
import io
from pathlib import Path
import time
import threading
import cv2
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from nidar_survivor_demo.detector import (DetectorConfig, Detection, DetectionResult,
                                          PersonDetector, annotate, validate_person_model)
from nidar_survivor_demo.config import CameraConfig, arguments
from nidar_survivor_demo.camera.phone_stream import PhoneStream
from nidar_survivor_demo import main
from nidar_survivor_demo.evaluate_pose_detector import iou, match
from .test_phase1 import FakeCapture


class FakeDetector:
    device = "cpu"
    half = False

    def predict(self, frame):
        time.sleep(0.01)
        return DetectionResult((Detection((1, 1, 20, 20), 0.8),), 10)


class Phase2Tests(unittest.TestCase):
    def test_reviewed_general_and_specialized_model_schemas(self):
        self.assertEqual(validate_person_model("detect", {0: "person", 1: "bicycle"}), "person")
        self.assertEqual(validate_person_model("detect", {0: "person_candidate"}), "person_candidate")
        for task, names in (("segment", {0: "person"}), ("detect", {0: "fallen"}),
                            ("detect", {1: "person"}), ("detect", [])):
            with self.subTest(task=task, names=names), self.assertRaises(ValueError):
                validate_person_model(task, names)

    def test_pose_evaluator_uses_one_to_one_iou_matching(self):
        truths = [(0, (0, 0, 10, 10)), (1, (20, 20, 30, 30))]
        predictions = [(0, 0, 10, 10), (1, 1, 9, 9), (20, 20, 30, 30)]
        used_predictions, used_truths = match(predictions, truths)
        self.assertEqual(used_truths, {0, 1})
        self.assertEqual(len(used_predictions), 2)
        self.assertEqual(iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
    def test_bad_jpeg_skipped_without_reconnection(self):
        from nidar_survivor_demo.camera.mjpeg_capture import MjpegCapture
        capture = MjpegCapture.__new__(MjpegCapture)
        capture._timeout = .2
        capture._condition = threading.Condition()
        capture._stop = threading.Event()
        capture._sequence, capture._consumed = 1, 0
        capture._failed = False
        capture._latest = b"bad jpeg"
        capture.invalid_frames = 0
        valid = cv2.imencode(".jpg", np.zeros((20, 20, 3), np.uint8))[1].tobytes()
        def send():
            time.sleep(.03)
            with capture._condition:
                capture._latest = valid
                capture._sequence += 1
                capture._condition.notify()
        sender = threading.Thread(target=send)
        sender.start()
        try:
            ok, image = capture.read()
            self.assertTrue(ok)
            self.assertEqual(image.shape, (20, 20, 3))
            self.assertEqual(capture.invalid_frames, 1)
        finally:
            sender.join()
        # No fresh valid image must still time out, rather than spin/reuse it.
        started = time.perf_counter()
        self.assertEqual(capture.read(), (False, None))
        self.assertLess(time.perf_counter() - started, .5)

    def test_prediction_filters_person_and_uses_bounded_configuration(self):
        detector = PersonDetector.__new__(PersonDetector)
        detector.config = DetectorConfig()
        detector.device, detector.half = "0", True
        detector.model = MagicMock()
        detector.model.predict.return_value[0].boxes.data.cpu.return_value.numpy.return_value = np.array([
            [1, 2, 10, 20, .9, 0], [1, 2, 10, 20, .9, 2]])
        result = detector.predict(np.zeros((24, 32, 3), np.uint8))
        self.assertEqual(len(result.people), 1)
        kwargs = detector.model.predict.call_args.kwargs
        self.assertEqual(kwargs["classes"], [0])
        self.assertEqual(kwargs["quantize"], 16)
        self.assertFalse(kwargs["rect"])
        self.assertFalse(kwargs["save"])
        with self.assertRaises(ValueError):
            detector.predict(np.zeros((1, 2), np.uint8))

    def test_slow_stale_result_is_not_displayed(self):
        cap = FakeCapture()
        stream = PhoneStream(CameraConfig(stale_after=.001), lambda _: cap)
        frames = []
        class SlowDetector(FakeDetector):
            def predict(self, frame):
                time.sleep(.06)  # Exceed Windows' coarse monotonic clock tick.
                return super().predict(frame)
        def check_render(snapshot, fps, status):
            frames.append(snapshot)
            return np.zeros((20, 20, 3), np.uint8)
        with patch.object(main, "PersonDetector", return_value=SlowDetector()), patch.object(main, "PhoneStream", return_value=stream), \
             patch.object(main, "render", side_effect=check_render), patch.object(main.cv2, "namedWindow"), \
             patch.object(main.cv2, "resizeWindow"), patch.object(main.cv2, "imshow"), \
             patch.object(main.cv2, "waitKey", return_value=-1), patch.object(main.cv2, "getWindowProperty", return_value=1), \
             patch.object(main.cv2, "destroyAllWindows"), contextlib.redirect_stdout(io.StringIO()):
            main.run(["--detect", "--seconds", ".15", "--stale-after", ".001"])
        self.assertTrue(frames)
        self.assertTrue(all(frame.packet is None for frame in frames))

    def test_validated_config(self):
        for field, value in (("imgsz", 320), ("confidence", float("nan")), ("iou", 0),
                             ("confidence", 1.1), ("device", "bad")):
            with self.assertRaises(ValueError):
                replace(DetectorConfig(), **{field: value})

    def test_cli(self):
        _, args = arguments(["--detect", "--imgsz", "512", "--confidence", ".5", "--fp32"])
        self.assertTrue(args.detect)
        self.assertTrue(args.fp32)
        self.assertEqual(args.imgsz, 512)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            arguments(["--confidence", "nan"])

    def test_annotations_do_not_change_camera_image(self):
        original = np.zeros((100, 100, 3), np.uint8)
        result = DetectionResult((Detection((-1, -5, 150, 100), .9),), 5)
        rendered = annotate(original, result)
        self.assertFalse(original.any())
        self.assertTrue(rendered.any())
        self.assertFalse(np.shares_memory(original, rendered))

    def test_no_people_valid_result(self):
        original = np.zeros((20, 20, 3), np.uint8)
        self.assertTrue(np.array_equal(annotate(original, DetectionResult((), 1)), original))

    def test_startup_failure_does_not_open_camera(self):
        with patch.object(main, "PersonDetector", side_effect=ValueError), patch.object(main, "PhoneStream") as stream:
            self.assertEqual(main.run(["--detect", "--headless"]), 1)
            stream.assert_not_called()

    def test_detector_error_releases_camera(self):
        cap = FakeCapture()
        stream = PhoneStream(CameraConfig(), lambda _: cap)
        detector = FakeDetector()
        detector.predict = lambda _: (_ for _ in ()).throw(RuntimeError("inference failed"))
        with patch.object(main, "PersonDetector", return_value=detector), patch.object(main, "PhoneStream", return_value=stream), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main.run(["--detect", "--headless", "--seconds", ".2"]), 1)
        self.assertTrue(cap.released)

    def test_live_loop_with_detector(self):
        cap = FakeCapture()
        stream = PhoneStream(CameraConfig(), lambda _: cap)
        with patch.object(main, "PersonDetector", return_value=FakeDetector()), patch.object(main, "PhoneStream", return_value=stream), contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main.run(["--detect", "--headless", "--seconds", ".2"]), 0)
        self.assertIn('"unique_counting": false', output.getvalue())
        self.assertTrue(cap.released)

    def test_camera_only_does_not_load_model(self):
        with patch.object(main, "PersonDetector") as detector, contextlib.redirect_stdout(io.StringIO()):
            cap = FakeCapture()
            with patch.object(main, "PhoneStream", return_value=PhoneStream(CameraConfig(), lambda _: cap)):
                self.assertEqual(main.run(["--headless", "--seconds", ".05"]), 0)
            detector.assert_not_called()
