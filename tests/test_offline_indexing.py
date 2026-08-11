import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from backend.offline_indexing import VectorIndex, build_index


class OfflineIndexingTest(unittest.TestCase):
    def test_build_and_search(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset = Path(temporary_directory)
            for directory in (
                "keyframes/L21_V001",
                "keyframes/L22_V001",
                "clip",
                "map-keyframes",
                "objects/L21_V001",
                "media-info",
                "video",
            ):
                (dataset / directory).mkdir(parents=True)
            for keyframe_id in ("0000", "0001"):
                (dataset / "keyframes/L21_V001" / f"{keyframe_id}.jpg").write_bytes(b"jpg")
                (dataset / "objects/L21_V001" / f"{keyframe_id}.json").write_text("[]")
            np.save(dataset / "clip/L21_V001.npy", np.array([[1, 0], [0, 1]], dtype=np.float32))
            with (dataset / "map-keyframes/L21_V001.csv").open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=["frame_idx", "pts_time"])
                writer.writeheader()
                writer.writerows([{"frame_idx": 10, "pts_time": 0.4}, {"frame_idx": 20, "pts_time": 0.8}])
            (dataset / "media-info/L21_V001.json").write_text(json.dumps({"title": "demo"}))
            (dataset / "video/L21_V001.mp4").write_bytes(b"mp4")

            summary = build_index(dataset, dataset / "index", video_prefix="L21")
            result = VectorIndex(dataset / "index").search([1, 0], top_k=1)
            video = json.loads((dataset / "index/videos.jsonl").read_text())

            self.assertEqual(summary, {"videos": 1, "frames": 2, "feature_dim": 2})
            self.assertEqual((result[0]["video_id"], result[0]["frame_id"]), ("L21_V001", 10))
            self.assertAlmostEqual(result[0]["score"], 1.0)
            self.assertEqual(video["video_path"], "video/L21_V001.mp4")


if __name__ == "__main__":
    unittest.main()
