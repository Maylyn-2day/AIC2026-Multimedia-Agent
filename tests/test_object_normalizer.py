import csv
import json
import tempfile
import unittest
from pathlib import Path

from backend.offline_indexing.object_normalizer import normalize_objects


class ObjectNormalizerTest(unittest.TestCase):
    def test_filters_and_normalizes_detections(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "objects/video").mkdir(parents=True)
            (root / "mappings").mkdir()
            source = {
                "detection_scores": ["0.9", "0.05"],
                "detection_class_names": ["/m/car", "/m/tree"],
                "detection_class_entities": ["Car", "Tree"],
                "detection_class_labels": ["1", "2"],
                "detection_boxes": [["0", "0", "1", "1"], ["0", "0", ".5", ".5"]],
            }
            (root / "objects/video/001.json").write_text(json.dumps(source))
            with (root / "mappings/video.csv").open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=["frame_idx", "pts_time"])
                writer.writeheader()
                writer.writerow({"frame_idx": 42, "pts_time": 1.4})

            normalize_objects(root / "objects", root / "mappings", root / "output", ["video"])
            record = json.loads((root / "output/video.jsonl").read_text())

            self.assertEqual(record["labels"], ["Car"])
            self.assertEqual(record["frame_id"], 42)
            self.assertEqual(len(record["detections"]), 1)


if __name__ == "__main__":
    unittest.main()
