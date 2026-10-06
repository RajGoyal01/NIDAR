"""Camera orientation and independent evidence/matching clocks."""
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import numpy as np
from . import test_phase5
from .test_phase1 import FakeCapture, until
from nidar_survivor_demo.config import CameraConfig, arguments
from nidar_survivor_demo.camera.phone_stream import PhoneStream, orient_frame
from nidar_survivor_demo.dashboard import annotate_dashboard
from nidar_survivor_demo.tracking import Track
from nidar_survivor_demo.verification import VerifiedTrack, VerificationResult


class CountingInputTests(unittest.TestCase):
    setUp = test_phase5.Phase5Tests.setUp
    update = test_phase5.Phase5Tests.update
    enroll = test_phase5.Phase5Tests.enroll

    def test_all_quarter_turns_preserve_pixels(self):
        frame = np.arange(4 * 7 * 3, dtype=np.uint8).reshape(4, 7, 3)
        for degrees in (0, 90, 180, 270):
            rotated = orient_frame(frame, degrees)
            np.testing.assert_array_equal(rotated, np.rot90(frame, -(degrees // 90)))
        self.assertIs(orient_frame(frame, 0), frame)

    def test_rotation_cli_validation(self):
        self.assertEqual(arguments(['--rotation', '90'])[0].rotation, 90)
        for invalid in (-90, 45, True):
            with self.assertRaises(ValueError):
                CameraConfig(rotation=invalid)

    def test_capture_rotates_before_publishing(self):
        stream = PhoneStream(CameraConfig(rotation=90), lambda _: FakeCapture())
        stream.start()
        try:
            until(lambda: stream.snapshot().packet is not None)
            self.assertEqual(stream.snapshot().packet.frame.shape, (32, 24, 3))
        finally:
            stream.stop()

    def test_retry_does_not_block_sampling(self):
        self.matcher.retain_gallery = True
        self.enroll()
        self.update(())
        self.encoder.vector = np.array([0., 1., 0.])
        self.matcher.config = replace(self.config, sample_interval=.15, retry_interval=.5)
        self.enroll('E1:T2')
        self.update(('E1:T2',))
        self.update(('E1:T2',))  # third sample / first comparison
        calls = self.encoder.calls
        self.update(('E1:T2',))
        self.update(('E1:T2',))  # new sample within retry cooldown
        self.assertGreater(self.encoder.calls, calls)
        self.assertEqual(len(self.matcher.gallery), 1)
        for _ in range(30):
            result = self.update(('E1:T2',))
        self.assertEqual(result.observations[0].reference_id, 'R2')

    def test_pending_overlay_explains_edge_rejection(self):
        verified = VerificationResult((VerifiedTrack(Track('E1:T1', (10,10,110,300), .9), 'CONFIRMED', 3, True),), (), 0)
        matched = SimpleNamespace(observations=[SimpleNamespace(track_id='E1:T1', reason='frame_edge_crop')])
        with patch('nidar_survivor_demo.dashboard.cv2.putText') as draw:
            annotate_dashboard(self.frame, verified, SimpleNamespace(track_survivors={}), matched)
        self.assertIn('keep body inside frame', draw.call_args.args[1])
