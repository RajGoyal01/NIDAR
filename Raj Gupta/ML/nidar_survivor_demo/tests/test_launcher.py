"""Validate Windows entrypoint without touching a phone or changing settings."""
from pathlib import Path
import shutil
import subprocess
import unittest


SHELL = shutil.which("pwsh") or shutil.which("powershell")
SCRIPT = Path(__file__).resolve().parents[2] / "Start-PhoneDemo.ps1"


@unittest.skipUnless(SHELL, "PowerShell is required for the Windows launcher")
class LauncherTests(unittest.TestCase):
    def test_rejects_unsafe_or_non_base_address(self):
        for address in ("not-a-url", "ftp://localhost:1234", "http://localhost:1234/video",
                        "http://secret:password@localhost:1234", "http://localhost:1234/?token=secret"):
            with self.subTest(address=address):
                result = subprocess.run([SHELL, "-NoProfile", "-File", str(SCRIPT),
                                         "-PhoneUrl", address, "-Seconds", "1"],
                                        capture_output=True, text=True, timeout=10)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Use the base HTTP(S) camera address", result.stdout + result.stderr)

    def test_rejects_unknown_profile(self):
        result = subprocess.run([SHELL, "-NoProfile", "-File", str(SCRIPT),
                                 "-PhoneUrl", "http://localhost:1234", "-Profile", "4k"],
                                capture_output=True, text=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ValidateSet", result.stdout + result.stderr)

    def test_rejects_missing_custom_model_before_camera_access(self):
        result = subprocess.run([SHELL, "-NoProfile", "-File", str(SCRIPT),
                                 "-PhoneUrl", "http://localhost:1234", "-Model", "missing.pt",
                                 "-Seconds", "1"], capture_output=True, text=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
