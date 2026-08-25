from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path

import pytest
from scripts.render_search_results import render_results
from scripts.search_batch1 import (
    build_parser,
    parse_trake_events,
    read_classified_queries,
    read_qa_queries,
    read_query_variants,
    read_query_zip,
    run_kis_batch,
    run_qa_batch,
    validate_variant_coverage,
    write_candidates,
)

from backend.services.batch1_retrieval import SearchCandidate


def test_candidate_outputs_and_offline_html(tmp_path: Path) -> None:
    image = tmp_path / "001.jpg"
    image.write_bytes(b"jpg")
    candidates = [SearchCandidate(1, "L21_V001", 90, "001", 0.9, image, 0)]
    csv_path = tmp_path / "results.csv"
    json_path = tmp_path / "results.json"
    query = "Một người đang phát biểu"
    write_candidates(
        csv_path,
        json_path,
        candidates,
        query_filename="query-p1-1-kis.txt",
        query_text=query,
    )
    assert csv_path.read_bytes().startswith(b"\xef\xbb\xbf")
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row["frame_id"] == "90" and row["keyframe_id"] == "001"
    assert row["query_filename"] == "query-p1-1-kis.txt" and row["query_text"] == query
    assert json_path.is_file()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["query_text"] == query and payload["candidates"][0]["frame_id"] == 90
    html_path = tmp_path / "results.html"
    render_results(csv_path, html_path)
    document = html_path.read_text(encoding="utf-8")
    assert "file:///" in document and "base64" not in document
    assert "http://" not in document and "https://" not in document


def test_reads_unique_top_level_query_as_strict_utf8(tmp_path: Path) -> None:
    archive_path = tmp_path / "queries.zip"
    query = "Người phụ nữ đang nói tiếng Việt."
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("query-p1-1-kis.txt", query.encode("utf-8"))
    assert read_query_zip(archive_path, "query-p1-1-kis.txt") == query


@pytest.mark.parametrize("query_name", ["../query.txt", "folder/query.txt", "folder\\query.txt"])
def test_rejects_non_top_level_query_name(tmp_path: Path, query_name: str) -> None:
    with pytest.raises(ValueError, match="top-level"):
        read_query_zip(tmp_path / "unused.zip", query_name)


def test_rejects_duplicate_missing_directory_and_invalid_utf8(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.zip"
    with zipfile.ZipFile(duplicate, "w") as archive:
        archive.writestr("query.txt", "one")
        archive.writestr("query.txt", "two")
    with pytest.raises(ValueError, match="exactly one"):
        read_query_zip(duplicate, "query.txt")

    invalid = tmp_path / "invalid.zip"
    with zipfile.ZipFile(invalid, "w") as archive:
        archive.writestr("query.txt", b"\xff")
    with pytest.raises(UnicodeDecodeError):
        read_query_zip(invalid, "query.txt")
    with pytest.raises(ValueError, match="exactly one"):
        read_query_zip(invalid, "missing.txt")


def test_cli_rejects_multiple_query_sources() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--query", "one", "--query-zip", "queries.zip"])


def write_variant_config(path: Path, record: object) -> None:
    path.write_text(json.dumps({"query.txt": record}, ensure_ascii=False), encoding="utf-8")


def test_reads_strict_utf8_variants_without_mutation(tmp_path: Path) -> None:
    original = "Tàu vũ trụ và cực quang"
    record = {"original": original, "variants": [" spacecraft ", "four astronauts"]}
    path = tmp_path / "variants.json"
    write_variant_config(path, record)
    snapshot = json.loads(json.dumps(record))
    assert read_query_variants(path, "query.txt", original) == ("spacecraft", "four astronauts")
    assert record == snapshot


@pytest.mark.parametrize(
    "record,match",
    [
        ({"original": "q", "variants": [], "extra": True}, "only"),
        ({"original": "q", "variants": []}, "non-empty"),
        ({"original": "q", "variants": [" "]}, "non-blank"),
        ({"original": "q", "variants": [1]}, "non-blank"),
        ({"original": "q", "variants": ["same", " same "]}, "unique"),
    ],
)
def test_rejects_invalid_variant_schema(tmp_path: Path, record: object, match: str) -> None:
    path = tmp_path / "variants.json"
    write_variant_config(path, record)
    with pytest.raises(ValueError, match=match):
        read_query_variants(path, "query.txt", "q")


def test_rejects_invalid_utf8_variant_json(tmp_path: Path) -> None:
    path = tmp_path / "variants.json"
    path.write_bytes(b"\xff")
    with pytest.raises(UnicodeDecodeError):
        read_query_variants(path, "query.txt", "q")


def test_official_like_query_classification_natural_order_and_multiline(tmp_path: Path) -> None:
    path = tmp_path / "queries.zip"
    with zipfile.ZipFile(path, "w") as archive:
        qa_numbers = {3, 9, 15, 17}
        for index in range(1, 26):
            if index in qa_numbers:
                archive.writestr(f"query-p1-{index}-qa.txt", f"QA {index}\nsecond line")
            elif index == 16:
                archive.writestr(f"query-p1-{index}-trake.txt", "first\nE1 second\nE2 third\nE3 fourth")
            else:
                text = "duplicate" if index in {8, 14} else f"KIS {index}"
                archive.writestr(f"query-p1-{index}-kis.txt", text + ("\nsecond line" if index == 6 else ""))
    grouped = read_classified_queries(path)
    assert {task: len(items) for task, items in grouped.items()} == {"kis": 20, "qa": 4, "trake": 1}
    ordered_numbers = [int(name.split("-")[2]) for task in grouped.values() for name in task]
    assert ordered_numbers[:5] == [1, 2, 4, 5, 6]
    assert grouped["kis"]["query-p1-6-kis.txt"] == "KIS 6\nsecond line"
    assert grouped["qa"]["query-p1-3-qa.txt"] == "QA 3\nsecond line"
    assert grouped["kis"]["query-p1-8-kis.txt"] == grouped["kis"]["query-p1-14-kis.txt"]


def test_trake_parser_preserves_unlabelled_first_event_and_order() -> None:
    text = "First event without a label\nE1 second event\nE2: third event\nE3 - fourth event"
    assert parse_trake_events(text) == (
        "First event without a label",
        "second event",
        "third event",
        "fourth event",
    )
    with pytest.raises(ValueError, match="blank"):
        parse_trake_events("first\n\nE1 second")


def test_variant_coverage_is_exact_and_keeps_duplicate_query_keys(tmp_path: Path) -> None:
    queries = {"query-p1-8-kis.txt": "same", "query-p1-14-kis.txt": "same"}
    path = tmp_path / "variants.json"
    make_batch_config(path, queries)
    variants = validate_variant_coverage(path, queries)
    assert set(variants) == set(queries)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["query-p1-3-qa.txt"] = {"original": "QA", "variants": ["visual"]}
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly match"):
        validate_variant_coverage(path, queries)


class BatchFakeEncoder:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def encode(self, text: str) -> object:
        self.calls.append(text)
        return text


class BatchFakeIndex:
    def __init__(self, image: Path) -> None:
        self.image = image
        self.calls: list[str] = []

    def search(self, vector: object, *, top_k: int) -> list[SearchCandidate]:
        assert top_k == 100
        self.calls.append(str(vector))
        return [
            SearchCandidate(rank, f"V{rank:03d}", rank, f"{rank:03d}", 1 / rank, self.image, rank - 1)
            for rank in range(1, 101)
        ]


def make_batch_config(path: Path, queries: dict[str, str]) -> None:
    payload = {
        name: {"original": text, "variants": [f"visual description {index}"]}
        for index, (name, text) in enumerate(queries.items())
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_batch_resume_overwrite_report_index_and_html_limit(tmp_path: Path) -> None:
    image = tmp_path / "image.jpg"
    image.write_bytes(b"jpg")
    queries = {"query-p1-1-kis.txt": "<script>alert('x')</script>", "query-p1-2-kis.txt": "query two"}
    config = tmp_path / "variants.json"
    make_batch_config(config, queries)
    output = tmp_path / "output"
    index = BatchFakeIndex(image)
    encoder = BatchFakeEncoder()
    report = run_kis_batch(
        index=index,  # type: ignore[arg-type]
        encoder=encoder,  # type: ignore[arg-type]
        queries=queries,
        variants_path=config,
        output_root=output,
        resume=False,
        overwrite=False,
    )
    assert (report["succeeded"], report["failed"], report["skipped"]) == (2, 0, 0)
    assert len(index.calls) == 2 and len(encoder.calls) == 2
    query_html = (output / "query-p1-1-kis.html").read_text(encoding="utf-8")
    assert query_html.count("<article>") == 20 and "<script>alert" not in query_html
    index_html = (output / "index.html").read_text(encoding="utf-8")
    assert index_html.count('.html">') == 2 and "<script>alert" not in index_html

    resumed = run_kis_batch(
        index=index,  # type: ignore[arg-type]
        encoder=encoder,  # type: ignore[arg-type]
        queries=queries,
        variants_path=config,
        output_root=output,
        resume=True,
        overwrite=False,
    )
    assert resumed["skipped"] == 2 and len(index.calls) == 2

    (output / "query-p1-1-kis.json").write_text("broken", encoding="utf-8")
    regenerated = run_kis_batch(
        index=index,  # type: ignore[arg-type]
        encoder=encoder,  # type: ignore[arg-type]
        queries=queries,
        variants_path=config,
        output_root=output,
        resume=True,
        overwrite=False,
    )
    assert regenerated["succeeded"] == 1 and regenerated["skipped"] == 1
    assert len(index.calls) == 3


def test_batch_failure_isolated_and_existing_output_policy(tmp_path: Path) -> None:
    image = tmp_path / "image.jpg"
    image.write_bytes(b"jpg")
    queries = {"query-p1-1-kis.txt": "one", "query-p1-2-kis.txt": "two"}
    config = tmp_path / "variants.json"
    make_batch_config(config, queries)
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["query-p1-2-kis.txt"]["variants"] = []
    config.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "output"
    report = run_kis_batch(
        index=BatchFakeIndex(image),  # type: ignore[arg-type]
        encoder=BatchFakeEncoder(),  # type: ignore[arg-type]
        queries=queries,
        variants_path=config,
        output_root=output,
        resume=False,
        overwrite=False,
    )
    assert report["succeeded"] == 1 and report["failed"] == 1
    assert not (output / "query-p1-2-kis.json").exists()
    make_batch_config(config, {"query-p1-1-kis.txt": "one"})
    second = run_kis_batch(
        index=BatchFakeIndex(image),  # type: ignore[arg-type]
        encoder=BatchFakeEncoder(),  # type: ignore[arg-type]
        queries={"query-p1-1-kis.txt": "one"},
        variants_path=config,
        output_root=output,
        resume=False,
        overwrite=False,
    )
    assert second["failed"] == 1


def test_resume_rejects_output_from_another_query_zip(tmp_path: Path) -> None:
    image = tmp_path / "image.jpg"
    image.write_bytes(b"jpg")
    queries = {"query-p1-1-kis.txt": "one"}
    config = tmp_path / "variants.json"
    make_batch_config(config, queries)
    first_zip = tmp_path / "first.zip"
    second_zip = tmp_path / "second.zip"
    with zipfile.ZipFile(first_zip, "w") as archive:
        archive.writestr("query-p1-1-kis.txt", "first archive")
    with zipfile.ZipFile(second_zip, "w") as archive:
        archive.writestr("query-p1-1-kis.txt", "second archive")
    index = BatchFakeIndex(image)
    encoder = BatchFakeEncoder()
    output = tmp_path / "output"
    run_kis_batch(
        index=index,  # type: ignore[arg-type]
        encoder=encoder,  # type: ignore[arg-type]
        queries=queries,
        variants_path=config,
        output_root=output,
        resume=False,
        overwrite=False,
        query_zip=first_zip,
    )
    assert len(index.calls) == 1
    report = run_kis_batch(
        index=index,  # type: ignore[arg-type]
        encoder=encoder,  # type: ignore[arg-type]
        queries=queries,
        variants_path=config,
        output_root=output,
        resume=True,
        overwrite=False,
        query_zip=second_zip,
    )
    assert report["succeeded"] == 1 and report["skipped"] == 0
    assert len(index.calls) == 2


def test_reads_only_three_qa_entries(tmp_path: Path) -> None:
    path = tmp_path / "queries.zip"
    expected = {"query-p1-15-qa.txt", "query-p1-19-qa.txt", "query-p1-22-qa.txt"}
    with zipfile.ZipFile(path, "w") as archive:
        for name in expected:
            archive.writestr(name, f"question {name}")
        archive.writestr("query-p1-1-kis.txt", "KIS")
        archive.writestr("query-p1-1-trake.txt", "TRAKE")
    assert set(read_qa_queries(path)) == expected


def test_qa_variants_reject_answer_field(tmp_path: Path) -> None:
    path = tmp_path / "variants.json"
    path.write_text(
        json.dumps({"query.txt": {"original": "q", "variants": ["visual scene"], "answer": "guess"}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="only original and variants"):
        read_query_variants(path, "query.txt", "q")


def test_qa_batch_outputs_explicit_vlm_manifests_and_isolates_failure(tmp_path: Path) -> None:
    image = tmp_path / "actual-keyframe-007.jpg"
    image.write_bytes(b"jpg")
    queries = {
        "query-p1-15-qa.txt": "Câu hỏi <b>một</b>",
        "query-p1-19-qa.txt": "Câu hỏi hai",
        "query-p1-22-qa.txt": "Câu hỏi ba",
    }
    config = tmp_path / "qa_variants.json"
    make_batch_config(config, queries)
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["query-p1-22-qa.txt"]["variants"] = []
    config.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    index = BatchFakeIndex(image)
    encoder = BatchFakeEncoder()
    output = tmp_path / "qa"
    report = run_qa_batch(
        index=index,  # type: ignore[arg-type]
        encoder=encoder,  # type: ignore[arg-type]
        queries=queries,
        variants_path=config,
        output_root=output,
    )
    assert report["succeeded"] == 2 and report["failed"] == 1 and report["vlm_loaded"] is False
    assert len(encoder.calls) == 2
    manifest_path = output / "query-p1-15-qa-vlm-input.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["question"] == queries["query-p1-15-qa.txt"]
    assert len(manifest["candidates"]) == 30
    assert [item["candidate_rank"] for item in manifest["candidates"]] == list(range(1, 31))
    assert all(item["image_path"] == str(image.resolve()) for item in manifest["candidates"])
    assert all(Path(item["image_path"]).is_absolute() for item in manifest["candidates"])
    assert not (output / "query-p1-22-qa-vlm-input.json").exists()
    html = (output / "query-p1-15-qa.html").read_text(encoding="utf-8")
    assert html.count("<article>") == 30 and "<b>một</b>" not in html
