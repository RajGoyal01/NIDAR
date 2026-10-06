"""Partial recognition is separate from new-person enrollment."""
from dataclasses import replace
import unittest
import numpy as np
from . import test_phase5
from nidar_survivor_demo.reid import AppearanceMatcher, ReIDConfig, quality_crop
from nidar_survivor_demo.tracking import Track
from nidar_survivor_demo.verification import VerificationResult, VerifiedTrack


class IdentityCoreTests(unittest.TestCase):
    setUp = test_phase5.Phase5Tests.setUp
    update = test_phase5.Phase5Tests.update
    enroll = test_phase5.Phase5Tests.enroll

    def edge(self):
        self.n += 1
        track = Track('E1:T9', (0, 10, 110, 300), .95)
        return self.matcher.update(self.frame,
            VerificationResult((VerifiedTrack(track, 'CONFIRMED', 3, True),), (), 0),
            sequence=self.n, received_at=self.n*.1, epoch=1)

    def test_strong_partial_return_recovers_without_new_record(self):
        self.enroll()
        original = self.matcher.export_gallery()
        self.update(())
        for _ in range(3):
            result = self.edge()
        self.assertEqual(result.observations[0].reference_id, 'R1')
        self.assertEqual(result.events[0]['event'], 'reid_match')
        self.assertEqual(self.matcher.export_gallery(), original)

    def test_partial_novel_person_cannot_enroll(self):
        self.enroll()
        self.update(())
        self.encoder.vector = np.array([0., 1., 0.])
        for _ in range(25):
            result = self.edge()
        self.assertIsNone(result.observations[0].reference_id)
        self.assertEqual(result.observations[0].reason, 'partial_view_needs_full_evidence')
        self.assertEqual(len(self.matcher.gallery), 1)

    def test_partial_uses_stricter_match_floor(self):
        self.enroll()
        self.update(())
        self.encoder.vector = np.array([.85, np.sqrt(1-.85**2), 0.])
        for _ in range(3):
            result = self.edge()
        self.assertIsNone(result.observations[0].reference_id)

    def test_sustained_distinct_partial_person_can_enroll_when_enabled(self):
        self.matcher.config = replace(self.config, allow_partial_novelty=True)
        self.enroll()
        self.update(())
        self.encoder.vector = np.array([0., 1., 0.])
        for _ in range(3):
            result = self.edge()
        self.assertIsNone(result.observations[0].reference_id)
        for _ in range(25):
            result = self.edge()
        self.assertEqual(result.observations[0].reference_id, 'R2')
        self.update(())
        self.encoder.vector = np.array([1., 0., 0.])
        for _ in range(3):
            result = self.edge()
        self.assertEqual(result.observations[0].reference_id, 'R1')
        self.assertEqual(len(self.matcher.gallery), 2)

    def test_initial_partial_enrolls_after_consistent_samples(self):
        self.assertIsNone(self.edge().observations[0].reference_id)
        self.assertIsNone(self.edge().observations[0].reference_id)
        self.assertEqual(self.edge().observations[0].reference_id, 'R1')

    def test_initial_partial_can_be_disabled(self):
        self.matcher.config = replace(self.config, allow_initial_partial=False)
        for _ in range(10):
            result = self.edge()
        self.assertFalse(self.matcher.gallery)
        self.assertEqual(result.observations[0].reason, 'frame_edge_crop')

    def test_full_return_adds_view_without_moving_anchors(self):
        self.matcher.retain_gallery = True
        self.enroll()
        anchors = self.matcher.export_gallery()['R1']
        self.update(())
        self.encoder.vector = np.array([.9, np.sqrt(1-.9**2), 0.])
        result = self.enroll('E1:T2')
        self.assertEqual(result.observations[0].reference_id, 'R1')
        stored = self.matcher.export_gallery()['R1']
        self.assertEqual(stored[:3], anchors)
        self.assertGreater(len(stored), 3)
        resumed = AppearanceMatcher(self.config, self.encoder, retain_gallery=True)
        resumed.restore_gallery(self.matcher.export_gallery())
        self.assertEqual(len(resumed.gallery['R1'].vectors), len(stored))

    def test_partial_quality_still_rejects_blur_overlap_and_confidence(self):
        track = Track('E1:T1', (0, 10, 110, 300), .95)
        self.assertEqual(quality_crop(self.frame*0, track, [track], self.config, allow_edge=True)[1], 'blurred_crop')
        self.assertEqual(quality_crop(self.frame, replace(track, confidence=.2), [track], self.config, allow_edge=True)[1], 'low_confidence')
        other = Track('E1:T2', (0,10,110,300), .95)
        self.assertEqual(quality_crop(self.frame, track, [track,other], self.config, allow_edge=True)[1], 'overlapping_person')

    def test_config_rejects_unsafe_partial_floor_and_capacity(self):
        for kwargs in ({'partial_match_threshold': .5}, {'partial_match_threshold': float('nan')},
                       {'max_reference_samples': 2}, {'max_reference_samples': 25}):
            with self.assertRaises(ValueError):
                ReIDConfig(**kwargs)

    def test_partial_return_and_second_person_commit_to_database(self):
        import tempfile
        from pathlib import Path
        from nidar_survivor_demo.survivors import SurvivorManager
        self.encoder.encode = lambda crops: [np.pad(self.encoder.vector, (0, 509)) for _ in crops]
        self.config = replace(self.config, allow_partial_novelty=True)
        self.matcher.config = self.config
        self.matcher.retain_gallery = True
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'mission.sqlite3'
            manager = SurvivorManager(path, self.matcher)
            def step(label=None, edge=False):
                self.n += 1
                box = (0 if edge else 10, 10, 110, 300)
                tracks = (VerifiedTrack(Track(label, box, .95), 'CONFIRMED', 3, True),) if label else ()
                verified = VerificationResult(tracks, (), 0)
                found = self.matcher.update(self.frame, verified, sequence=self.n, received_at=self.n*.1, epoch=1)
                return manager.update(verified, found, sequence=self.n, received_at=self.n*.1)
            try:
                for _ in range(6): result = step('E1:T1', edge=True)
                self.assertEqual(result.unique_survivors, 1)
                step()
                for _ in range(10): result = step('E1:T9', edge=True)
                self.assertEqual((result.unique_survivors, result.duplicates_prevented), (1, 1))
                self.assertEqual(result.track_survivors['E1:T9'], 'S1')
                step()
                self.encoder.vector = np.array([0., 1., 0.])
                for _ in range(30): result = step('E1:T10', edge=True)
                self.assertEqual((result.unique_survivors, result.duplicates_prevented), (2, 1))
            finally:
                manager.close()
            restored = AppearanceMatcher(self.config, self.encoder, retain_gallery=True)
            manager = SurvivorManager(path, restored, resume=True)
            try:
                self.assertEqual(manager.snapshot()['unique_survivors'], 2)
                self.assertEqual(manager.snapshot()['duplicates_prevented'], 1)
            finally:
                manager.close()

    def test_gallery_capacity_and_anchors_remain_bounded(self):
        self.matcher.retain_gallery = True
        self.matcher.config = replace(self.config, max_reference_samples=4)
        self.enroll()
        anchors = self.matcher.export_gallery()['R1'][:3]
        for i, angle in enumerate((.4, -.4, .5, -.5)):
            self.update(())
            self.encoder.vector = np.array([np.cos(angle), np.sin(angle), 0.])
            self.enroll(f'E1:T{i+2}')
            self.assertLessEqual(len(self.matcher.gallery['R1'].vectors), 4)
            self.assertEqual(self.matcher.export_gallery()['R1'][:3], anchors)
