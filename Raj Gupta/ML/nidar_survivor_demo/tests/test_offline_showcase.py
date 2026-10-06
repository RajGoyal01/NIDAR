import unittest

import numpy as np

from nidar_survivor_demo.detector import Detection, DetectionResult
from nidar_survivor_demo.offline_showcase import (
    PreparedScene,
    ShowcaseScene,
    animate_source,
    load_scenes,
    render_scene,
)


class OfflineShowcaseTests(unittest.TestCase):
    def test_playlist_demonstrates_multiple_people_counts(self):
        scenes = load_scenes()
        counts = [scene.expected_detections for scene in scenes]
        self.assertIn(1, counts)
        self.assertIn(2, counts)
        self.assertGreaterEqual(counts.count(4), 2)
        self.assertGreaterEqual(counts.count(6), 2)
        self.assertEqual(len({scene.image_path for scene in scenes}), len(scenes))

    def test_renderer_does_not_modify_source_image(self):
        image = np.full((240, 320, 3), 90, np.uint8)
        original = image.copy()
        result = DetectionResult((Detection((20, 20, 120, 220), .9),), 12.5)
        scene = PreparedScene(ShowcaseScene("test", load_scenes()[0].image_path, 1), image, result)
        rendered = render_scene(scene, 0, 1)
        self.assertEqual(rendered.shape, (720, 1280, 3))
        self.assertTrue(np.array_equal(image, original))

    def test_replay_motion_changes_frame_without_changing_shape(self):
        image = np.full((120, 160, 3), 80, np.uint8)
        image[30:90, 50:110] = (20, 180, 240)
        moved = animate_source(image, 1.25)
        self.assertEqual(moved.shape, image.shape)
        self.assertEqual(moved.dtype, image.dtype)
        self.assertFalse(np.array_equal(moved, image))


if __name__ == "__main__":
    unittest.main()
