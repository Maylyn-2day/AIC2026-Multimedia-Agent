import csv
import json
import zipfile
from pathlib import Path

from scripts.emergency_submission import _package, _validate_zip


def test_config_driven_packaging_and_validation(tmp_path: Path) -> None:
    kis_root = tmp_path / "kis"
    qa_root = tmp_path / "qa"
    kis_root.mkdir()
    qa_root.mkdir()
    for index in range(20):
        candidates = [{"video_id": f"V{index:02d}", "frame_id": frame_id} for frame_id in range(100)]
        (kis_root / f"query-{index:02d}-kis.json").write_text(json.dumps({"candidates": candidates}), encoding="utf-8")
    answers = {}
    qa_filenames = set()
    for index in range(4):
        filename = f"query-{index:02d}-qa.txt"
        qa_filenames.add(filename)
        answers[filename] = {"candidate_rank": 2, "answer": f"answer {index}"}
        (qa_root / f"query-{index:02d}-qa.json").write_text(
            json.dumps(
                {
                    "candidates": [
                        {"video_id": f"Q{index}", "frame_id": 10},
                        {"video_id": f"Q{index}", "frame_id": 20},
                    ]
                }
            ),
            encoding="utf-8",
        )
    answers_path = tmp_path / "manual-answers.json"
    answers_path.write_text(json.dumps(answers), encoding="utf-8")
    trake_path = tmp_path / "trake.json"
    trake_path.write_text(json.dumps({"sequences": [{"video_id": "T", "frame_ids": [1, 2, 3]}]}), encoding="utf-8")
    final_zip = tmp_path / "submission.zip"
    _package(
        kis_root,
        qa_root,
        trake_path,
        answers_path,
        tmp_path / "work",
        final_zip,
        "query-00-trake.txt",
        qa_filenames,
    )
    result = _validate_zip(final_zip)
    assert result["csv_count"] == 25
    assert result["counts"] == {"kis": 20, "qa": 4, "trake": 1}
    with zipfile.ZipFile(final_zip) as archive:
        names = archive.namelist()
        assert len(names) == 25 and all(name.startswith("submission/") for name in names)
        qa_rows = list(csv.reader(archive.read("submission/query-00-qa.csv").decode().splitlines()))
        assert qa_rows == [["Q0", "20", "answer 0"]]
        trake_rows = list(csv.reader(archive.read("submission/query-00-trake.csv").decode().splitlines()))
        assert trake_rows == [["T", "1", "2", "3"]]
