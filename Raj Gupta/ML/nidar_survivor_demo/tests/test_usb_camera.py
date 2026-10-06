from pathlib import Path
import shutil
import subprocess
import unittest

import numpy as np

from nidar_survivor_demo.usb_camera import probe_camera


class FakeCamera:
    def __init__(self, *, opened=True, valid=True):
        self.opened = opened
        self.valid = valid
        self.released = False

    def isOpened(self):
        return self.opened

    def read(self):
        if self.valid:
            return True, np.full((480, 640, 3), 90, np.uint8)
        return False, None

    def get(self, _):
        return 30

    def release(self):
        self.released = True


class UsbCameraProbeTests(unittest.TestCase):
    def test_success_reports_actual_frame_and_releases_camera(self):
        camera = FakeCamera()
        result = probe_camera(1, factory=lambda _: camera)
        self.assertTrue(result.available)
        self.assertEqual((result.width, result.height, result.fps), (640, 480, 30.0))
        self.assertTrue(camera.released)

    def test_failure_is_recoverable_and_releases_camera(self):
        camera = FakeCamera(valid=False)
        result = probe_camera(2, factory=lambda _: camera)
        self.assertFalse(result.available)
        self.assertEqual(result.reason, "no_valid_frame")
        self.assertTrue(camera.released)

    def test_invalid_index_is_rejected_before_open(self):
        with self.assertRaises(ValueError):
            probe_camera(-1, factory=lambda _: self.fail("factory should not run"))


SHELL = shutil.which("pwsh") or shutil.which("powershell")
SCRIPT = Path(__file__).resolve().parents[2] / "Start-USBCameraDemo.ps1"


@unittest.skipUnless(SHELL, "PowerShell is required for the Windows launcher")
class UsbLauncherTests(unittest.TestCase):
    def test_invalid_camera_index_is_rejected(self):
        result = subprocess.run([SHELL, "-NoProfile", "-File", str(SCRIPT),
                                 "-CameraIndex", "-1"], capture_output=True, text=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)

    def test_missing_model_is_rejected_before_camera_probe(self):
        result = subprocess.run([SHELL, "-NoProfile", "-File", str(SCRIPT),
                                 "-Model", "missing.pt"], capture_output=True, text=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
