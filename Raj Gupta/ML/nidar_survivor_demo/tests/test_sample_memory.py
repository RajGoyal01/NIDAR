"""Short quality dips may preserve clear samples, never count bad evidence."""
from dataclasses import replace
from unittest.mock import patch
import unittest
from . import test_phase5
from nidar_survivor_demo.reid import ReIDConfig


class SampleMemoryTests(unittest.TestCase):
    setUp = test_phase5.Phase5Tests.setUp
    update = test_phase5.Phase5Tests.update

    def test_brief_temporal_dip_retains_only_recent_good_samples(self):
        self.update()
        self.update()
        self.assertIsNone(self.update(state='VERIFYING').observations[0].reference_id)
        self.assertEqual(self.update().observations[0].reference_id, 'R1')

    def test_short_blur_does_not_erase_good_views(self):
        self.update()
        self.update()
        with patch('nidar_survivor_demo.reid.quality_crop', return_value=(None, 'blurred_crop')):
            blurred = self.update()
        self.assertIsNone(blurred.observations[0].reference_id)
        self.assertEqual(self.encoder.calls, 2)
        self.assertEqual(self.update().observations[0].reference_id, 'R1')

    def test_long_blur_expires_every_sample(self):
        self.update()
        self.update()
        with patch('nidar_survivor_demo.reid.quality_crop', return_value=(None, 'blurred_crop')):
            for _ in range(9):
                self.update()
        self.assertFalse(self.matcher._pending['E1:T1'].vectors)
        self.assertFalse(self.matcher.gallery)
        self.assertIsNone(self.update().observations[0].reference_id)

    def test_edge_or_overlap_still_discards_samples(self):
        self.matcher.config = replace(self.config, allow_initial_partial=False)
        for reason in ('frame_edge_crop', 'overlapping_person'):
            self.update()
            with patch('nidar_survivor_demo.reid.quality_crop', return_value=(None, reason)):
                self.update()
            self.assertFalse(self.matcher._pending['E1:T1'].vectors)
            self.assertFalse(self.matcher._pending['E1:T1'].sample_times)

    def test_expiration_is_per_sample(self):
        self.matcher.config = replace(self.config, sample_memory_seconds=.15)
        for _ in range(9):
            result = self.update()
        self.assertIsNone(result.observations[0].reference_id)
        self.assertEqual(len(self.matcher._pending['E1:T1'].vectors), 2)

    def test_memory_validation(self):
        for value in (0, -1, 2, True, float('nan')):
            with self.assertRaises(ValueError):
                ReIDConfig(sample_memory_seconds=value)
