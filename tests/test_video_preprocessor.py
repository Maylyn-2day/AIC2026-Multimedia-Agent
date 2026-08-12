import csv
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from backend.offline_indexing.video_preprocessor import l1_distance, preprocess_video
from backend.offline_indexing.keyframe_validator import validate_keyframes


class VideoPreprocessorTest(unittest.TestCase):
    def test_extracts_distinct_scenes_and_mapping(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            video_path = root / "demo.mp4"
            writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
            for value in (0, 255, 255):
                frame = np.full((48, 64, 3), value, dtype=np.uint8)
                for _ in range(20):
                    writer.write(frame)
            writer.release()

            report = preprocess_video(
                video_path,
                root / "output",
                scene_threshold=10,
                dedup_threshold=0.04,
                minimum_scene_frames=5,
            )
            with (root / "output/map-keyframes/demo.csv").open(newline="") as file:
                rows = list(csv.DictReader(file))

            self.assertEqual(report["keyframes"], 2)
            self.assertEqual([int(row["frame_idx"]) for row in rows], [9, 39])
            self.assertEqual(len(list((root / "output/keyframes/demo").glob("*.jpg"))), 2)
            self.assertAlmostEqual(l1_distance(np.zeros((4, 4, 3), np.uint8), np.full((4, 4, 3), 255, np.uint8)), 1.0)
            validation = validate_keyframes(root, root / "output", root / "output/map-keyframes", ["demo"])
            self.assertTrue(validation[0]["saved_frame_mapping_valid"])
            self.assertEqual(validation[0]["count_ratio"], 1.0)


if __name__ == "__main__":
    unittest.main()
