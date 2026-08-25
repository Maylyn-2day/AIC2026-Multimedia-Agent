# AIC2026 Standalone Submission Pipeline v1

## Status

This is a standalone offline baseline for preparing AIC2026 submission artifacts. It does not require FastAPI,
Streamlit, Qdrant, or Elasticsearch, does not upload anything to the competition portal, and requires manual QA
review in v1.

## Scope

- KIS text-to-keyframe retrieval.
- QA candidate retrieval and manual candidate/answer selection.
- Ordered temporal alignment for TRAKE.
- Headerless UTF-8 CSV packaging.
- Local ZIP structure and content validation.

## Architecture

```mermaid
flowchart LR
    A[Query ZIP] --> B[Task parsing]
    B --> C[English visual variants]
    C --> D[OpenAI CLIP ViT-B/32 text encoder]
    D --> E[BTC feature shards]
    E --> F[Cosine retrieval]
    F --> G[Reciprocal Rank Fusion]
    G --> H[Mapping resolution]
    H --> I[JSON, CSV, and HTML review artifacts]
    I --> J[Manual selection]
    J --> K[Submission packager]
```

## Data contract

The dense index is a directory of per-video NumPy arrays with shape `(N, 512)`. The arrays must be compatible
with the official OpenAI CLIP ViT-B/32 text encoder. Every video has a mapping CSV with the exact columns
`n,pts_time,fps,frame_idx`, and the number of feature rows must equal the number of mapping rows.

- `n` is the keyframe identifier. The corresponding image is `<keyframe-root>/<video-id>/<n padded to three
  digits>.jpg`.
- `frame_idx` is the true frame identifier used in a submission.
- `pts_time` is the keyframe timestamp used for temporal review and alignment.
- The feature row and `n` must never be submitted as `frame_id`.

Keyframes and mappings are mandatory for this baseline. Image paths are resolved from mapping `n`, never inferred
from `frame_idx`.

## Retrieval

The text encoder produces a finite, non-zero, normalized 512-dimensional vector. The index computes cosine
similarity against each official per-video feature shard. Candidate identity is `(video_id, frame_id)`. Duplicate
frames are collapsed before `top_k`, keeping the occurrence with the highest cosine score and then the earliest
feature row. Final ordering is deterministic: cosine descending, then video ID, frame ID, and feature row.

Standard search accepts `top_k` in `[1, 100]`. A separately bounded pool search, up to 500 candidates, exists only
for temporal sequence assembly.

## Multi-query Reciprocal Rank Fusion

English visual variants are searched independently and fused using rank-only Reciprocal Rank Fusion:

```text
rrf_score = Σ 1 / (k + source_rank)
```

The default is `k=60`. Cosine scores from different text variants are retained for diagnostics but are not added
together. Candidates present in only one variant remain eligible. Fusion deduplicates by `(video_id, frame_id)` and
uses deterministic tie-breaking.

## KIS flow

1. Read the selected UTF-8 query directly from the ZIP without extraction.
2. Validate external English visual variants against the exact query filename and original text.
3. Encode and search each variant.
4. Fuse unique candidates with RRF.
5. Resolve true frame IDs and explicit keyframe paths through mapping metadata.
6. Write review JSON/CSV/HTML outside Git.
7. Package up to 100 real candidates without padding or duplication.

## QA flow

CLIP retrieves visual candidates only. It does not reliably read numbers, names, or other answer text. Reviewers use
the explicit `image_path` stored for each candidate, select a candidate rank, and provide a non-blank manual answer.
The v1 pipeline does not run Qwen or the M2 VLM service. The existing M2 boundary is compatible because
`VLMService.extract_vqa_answer(query, image_path)` accepts an explicit image path; v1 does not invoke it.

Manual selections and answers are local configuration and must remain outside Git.

## TRAKE flow

TRAKE supports `N` ordered semantic events rather than a fixed event count. Each event is retrieved independently.
A valid sequence uses one `video_id` and strictly increasing true frame IDs:

```text
frame_1 < frame_2 < ... < frame_N
```

Temporal assembly is deterministic and can use `pts_time` to prefer reasonable gaps. Retrieval scores do not prove
semantic correctness, so manual sequence review remains required.

## Submission format

Submission files are headerless UTF-8 CSV:

```text
KIS:   video_id,frame_id
QA:    video_id,frame_id,answer
TRAKE: video_id,frame1,...,frameN
```

The ZIP contains only `submission/<query-stem>.csv`. Validation checks the archive, expected task/file count, column
count, non-blank QA answers, unique KIS identities, numeric non-negative frames, and strictly increasing TRAKE
frames. Packaging is local and never uploads the ZIP.

## Repository modules

| Module | Public purpose |
| --- | --- |
| `backend/services/clip_text_encoder.py` | Lazy OpenAI CLIP ViT-B/32 text encoding, deterministic 77-token truncation, and vector normalization. |
| `backend/services/batch1_retrieval.py` | Validated per-video BTC shard search, mapping resolution, candidate models, deduplication, and multi-variant RRF. |
| `scripts/search_batch1.py` | Strict ZIP/config parsing and single-query, KIS-batch, or QA-batch retrieval artifact generation. |
| `scripts/render_search_results.py` | Offline, escaped HTML grid rendering from candidate CSV. |
| `scripts/emergency_submission.py` | Config-driven QA/TRAKE retrieval, temporal assembly, headerless CSV packaging, and ZIP validation for the 20/4/1 v1 submission layout. |
| `scripts/p1_3_two_event_retry.py` | Generic two-event, same-video, ordered-pair review tool whose variants are supplied externally. |
| `tests/test_clip_text_encoder.py` | Encoder validation without loading a real model. |
| `tests/test_batch1_retrieval.py` | Shard search, mapping, deduplication, RRF, and input-integrity tests. |
| `tests/test_batch1_scripts.py` | ZIP/config parsing, artifact rendering, batch isolation, provenance, and explicit-path tests. |
| `tests/test_p1_3_two_event_retry.py` | Two-event fusion, ordering, deterministic pairing, explicit paths, and external variant-config tests. |

## Usage

Keep all data, models, local query configuration, and outputs outside the repository:

```powershell
$env:AIC_FEATURE_ROOT = "<path-to-clip-features-32>"
$env:AIC_MAPPING_ROOT = "<path-to-map-keyframes>"
$env:AIC_KEYFRAME_ROOT = "<path-to-keyframes>"
$env:AIC_QUERY_ZIP = "<path-to-query-zip>"
$env:AIC_MODEL_CACHE = "<path-to-model-cache>"
$env:AIC_OUTPUT_ROOT = "<path-to-output>"
```

Single-query retrieval with an external variant config:

```powershell
python scripts/search_batch1.py `
  --features $env:AIC_FEATURE_ROOT `
  --mappings $env:AIC_MAPPING_ROOT `
  --keyframes $env:AIC_KEYFRAME_ROOT `
  --query-zip $env:AIC_QUERY_ZIP `
  --query-name "<top-level-query-filename>" `
  --query-variants-file "<path-to-query-variants.json>" `
  --top-k 100 `
  --model-cache $env:AIC_MODEL_CACHE `
  --output-csv "$env:AIC_OUTPUT_ROOT/candidates.csv" `
  --output-json "$env:AIC_OUTPUT_ROOT/candidates.json"
```

KIS batch mode uses the same arguments plus `--batch-kis --output-root $env:AIC_OUTPUT_ROOT`; QA batch mode uses
`--batch-qa --output-root $env:AIC_OUTPUT_ROOT`. Batch mode does not accept single-query output arguments.

Render an offline review page:

```powershell
python scripts/render_search_results.py `
  --input "$env:AIC_OUTPUT_ROOT/candidates.csv" `
  --output "$env:AIC_OUTPUT_ROOT/candidates.html" `
  --limit 20
```

The submission helper requires external QA variants, TRAKE variants, and manual answers:

```powershell
python scripts/emergency_submission.py `
  --query-zip $env:AIC_QUERY_ZIP `
  --kis-root "<path-to-validated-kis-json>" `
  --features $env:AIC_FEATURE_ROOT `
  --mappings $env:AIC_MAPPING_ROOT `
  --keyframes $env:AIC_KEYFRAME_ROOT `
  --model-cache $env:AIC_MODEL_CACHE `
  --qa-variants "<path-to-qa-variants.json>" `
  --trake-variants "<path-to-trake-variants.json>" `
  --answers "<path-to-manual-answers.json>" `
  --output-root $env:AIC_OUTPUT_ROOT `
  --final-zip "$env:AIC_OUTPUT_ROOT/submission.zip"
```

The two-event review tool additionally requires `--event-a-variants` and `--event-b-variants`, each pointing to a
strict UTF-8 JSON array of visual-search strings.

## Validation and testing

```powershell
python -m ruff format backend/services/clip_text_encoder.py backend/services/batch1_retrieval.py scripts/search_batch1.py scripts/render_search_results.py scripts/emergency_submission.py scripts/p1_3_two_event_retry.py tests/test_clip_text_encoder.py tests/test_batch1_retrieval.py tests/test_batch1_scripts.py tests/test_p1_3_two_event_retry.py
python -m ruff check backend/services/clip_text_encoder.py backend/services/batch1_retrieval.py scripts/search_batch1.py scripts/render_search_results.py scripts/emergency_submission.py scripts/p1_3_two_event_retry.py tests/test_clip_text_encoder.py tests/test_batch1_retrieval.py tests/test_batch1_scripts.py tests/test_p1_3_two_event_retry.py
python -m pytest tests/test_clip_text_encoder.py tests/test_batch1_retrieval.py tests/test_batch1_scripts.py tests/test_p1_3_two_event_retry.py -q
python -m pytest tests -q
git diff --check
```

These checks do not require model downloads or retrieval runs when tests use their injected fake encoder/index.

## Limitations

- OpenAI CLIP has limited Vietnamese retrieval quality; curated English visual variants are usually required.
- QA selection and answers are manual in v1.
- TRAKE sequences need semantic review.
- There is no OCR, ASR, object-label, or sparse retrieval path.
- Qwen is not loaded or run.
- The standalone path has no API or UI integration.
- SigLIP2 is not integrated into v1.
- Text and image model/data compatibility is mandatory.

## Ownership and integration status

| Member | v1 ownership or dependency | Integration status |
| --- | --- | --- |
| M1 | RRF/fusion, orchestration, validation, and submission packaging. | Standalone v1 code is present. |
| M2 | Existing `VLMService` and API in dev; explicit `image_path` is the compatible boundary. | Qwen/VLM inference was not run or integrated into submission v1. QA uses retrieval plus manual review/answers. |
| M3 | Temporal alignment for `N`-event TRAKE. | v1 uses a standalone temporary implementation. The repository does not establish that M3 code was merged or reused here. |
| M4 | Official offline BTC artifacts: CLIP ViT-B/32 shards, mapping CSV, and keyframes. | v1 does not use an M4 FastAPI indexing endpoint. SigLIP2 output with 96,894 rows is objectively incompatible with the 177,321-row official mappings/keyframes used here. |
| M5 | OpenAI CLIP text encoder, BTC shard cosine search, multi-query retrieval, and RRF candidate fusion. | Standalone retrieval implementation is present. |

## Reproducibility and security

- Store datasets, model caches, query variants, manual answers, review output, and ZIP files outside Git.
- Do not place credentials, API keys, tokens, or portal details in configs or documentation.
- The pipeline performs no automatic upload or network submission.
- Generated outputs are covered by repository ignore rules when created under conventional local output folders.
