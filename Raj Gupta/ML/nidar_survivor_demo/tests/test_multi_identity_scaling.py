"""Crowded-scene identity allocation must scale beyond the Re-ID batch size."""
from pathlib import Path
from contextlib import closing
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from nidar_survivor_demo.reid import AppearanceMatcher, ReIDConfig
from nidar_survivor_demo.survivors import ManagerConfig, SurvivorManager
from nidar_survivor_demo.tracking import Track
from nidar_survivor_demo.verification import VerificationResult, VerifiedTrack


class MarkerEncoder:
    """Turn each synthetic crop marker into a distinct 512-D appearance vector."""
    device = "cpu"

    def encode(self, crops):
        vectors = []
        for crop in crops:
            vector = np.zeros(512, np.float32)
            vector[int(crop[0, 0, 0])] = 1.0
            vectors.append(vector)
        return vectors


def marked_crop(_frame, track, _tracks, _config, **_kwargs):
    # The marker follows the physical box position, not the temporary track ID,
    # so a returning person can receive a new tracker ID but the same appearance.
    marker = int(track.xyxy[0] // 120)
    return np.full((16, 16, 3), marker, dtype=np.uint8), "good"


class MultiIdentityScalingTests(unittest.TestCase):
    def test_eight_people_receive_sequential_ids_and_return_without_inflation(self):
        config = ReIDConfig(
            samples=1,
            sample_interval=.01,
            retry_interval=.01,
            max_batch=2,
            novelty_seconds=.01,
            novelty_batches=1,
            sample_memory_seconds=.75,
        )
        matcher = AppearanceMatcher(config, MarkerEncoder(), retain_gallery=True)
        frame = np.ones((240, 960, 3), np.uint8)
        first_tracks = tuple(
            VerifiedTrack(Track(f"E1:T{index}", (index * 120, 20, index * 120 + 90, 220), .95),
                          "CONFIRMED", 3, True)
            for index in range(1, 9)
        )

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "six.sqlite3"
            manager = SurvivorManager(path, matcher, ManagerConfig(new_identity_hits=1))
            sequence = 0

            def step(tracks):
                nonlocal sequence
                sequence += 1
                verified = VerificationResult(tracks, (), 0)
                reid = matcher.update(
                    frame,
                    verified,
                    sequence=sequence,
                    received_at=sequence * .1,
                    epoch=1,
                )
                return manager.update(
                    verified,
                    reid,
                    sequence=sequence,
                    received_at=sequence * .1,
                )

            try:
                with patch("nidar_survivor_demo.reid.quality_crop", side_effect=marked_crop):
                    for _ in range(12):
                        result = step(first_tracks)
                    self.assertEqual(result.unique_survivors, 8)
                    self.assertEqual(set(result.track_survivors.values()),
                                     {f"S{index}" for index in range(1, 9)})
                    self.assertEqual(len(matcher.gallery), 8)

                    step(())
                    returning = tuple(
                        VerifiedTrack(Track(f"E1:T{index + 10}", track.track.xyxy, .95),
                                      "CONFIRMED", 3, True)
                        for index, track in enumerate(first_tracks, start=1)
                    )
                    for _ in range(8):
                        result = step(returning)
                    self.assertEqual(result.unique_survivors, 8)
                    self.assertEqual(result.duplicates_prevented, 8)
                    self.assertEqual(set(result.track_survivors.values()),
                                     {f"S{index}" for index in range(1, 9)})
            finally:
                manager.close()

            with closing(sqlite3.connect(path)) as database:
                self.assertEqual(database.execute("PRAGMA integrity_check").fetchone()[0], "ok")
                self.assertEqual(database.execute("SELECT COUNT(*) FROM survivors").fetchone()[0], 8)
