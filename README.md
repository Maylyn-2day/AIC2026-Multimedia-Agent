# AIC 2026 Multimedia Agent

Kho mã dùng chung cho hệ thống truy vấn video AIC 2026. Phần hiện có tập trung vào pipeline lập chỉ mục offline: chuẩn hóa artefact BTC, tự trích keyframe từ video, chuẩn hóa object detection và chuẩn bị SigLIP2 feature cho hệ thống tìm kiếm.

## Trạng thái hiện tại

| Hạng mục | Trạng thái |
|---|---|
| Chuẩn hóa Keyframes, CLIP ViT-B/32, mapping, Objects và Metadata BTC | Hoàn thành trên L21 |
| Baseline cosine search/API kiểm tra tích hợp | Hoàn thành |
| PySceneDetect + normalized L1 dedup + frame mapping | Hoàn thành trên 3 video mẫu |
| Kiểm chứng ảnh đã lưu đúng frame video | Hoàn thành |
| Chuẩn hóa Faster R-CNN Objects thành JSONL | Hoàn thành trên 3 video mẫu |
| SigLIP2 global embedding | Code hoàn thành; chưa sinh dữ liệu vì thiếu weights/GPU |
| SigLIP2 dense feature | Code hoàn thành; chỉ chạy cho ứng viên rerank |
| OCR và ASR | Chưa triển khai |

Kết quả đã kiểm tra:

- Index L21: 29 video, 7.800 keyframe, CLIP 512 chiều.
- Keyframe tự sinh: 341, 291 và 263 frame cho `L21_V001`–`L21_V003`.
- Objects mẫu: 855 keyframe, 15.598 detection có score từ 0,1.
- Toàn bộ test hiện có đều đạt.

## Cấu trúc dữ liệu

Dữ liệu lớn nằm ngoài Git. Mỗi máy có thể chọn dataset root riêng nhưng phải giữ cấu trúc bên trong:

```text
<dataset-root>/
├── video/<video_id>.mp4
├── keyframes/<video_id>/*.jpg
├── map-keyframes/<video_id>.csv
├── clip/<video_id>.npy
├── objects/<video_id>/*.json
└── media-info/<video_id>.json
```


## Cài đặt và kiểm tra

```bash
source .venv/bin/activate
python -m unittest discover -s tests -v
python -m backend.offline_indexing.cli --help
```

Các thư viện chính: NumPy, OpenCV, PySceneDetect, PyTorch và `open_clip_torch`.

## 1. Chuẩn hóa artefact BTC

```bash
python -m backend.offline_indexing.cli build \
  /dataset-root \
  --video-prefix L21 \
  --output data/offline_index/l21
```

Đầu ra bàn giao cho Thành viên 5:

```text
data/offline_index/l21/
├── manifest.json     # số video, frame và chiều vector
├── records.jsonl     # video_id, frame_id, timestamp và đường dẫn artefact
├── videos.jsonl      # đường dẫn video và metadata
└── features.npy      # ma trận feature; record.feature_row trỏ tới từng dòng
```

`frame_id` là frame thật của video dùng để nộp bài, không phải số thứ tự keyframe.

## 2. API baseline để kiểm tra tích hợp

API này chỉ giúp kiểm tra index; API database/search chính thức thuộc Thành viên 5.

```bash
python -m backend.offline_indexing.cli serve --index data/offline_index/l21
```

Trong terminal khác:

```bash
curl http://127.0.0.1:8000/health
curl 'http://127.0.0.1:8000/frames?video_id=L21_V001&limit=20'
```

`POST /search` cần vector 512 chiều từ cùng model CLIP ViT-B/32 BTC; vector minh họa hai chiều không hợp lệ.

## 3. Sinh và kiểm chứng keyframe từ video

```bash
python -m backend.offline_indexing.cli preprocess \
  dataset-root/video \
  L21_V001 L21_V002 L21_V003 \
  --output data/processed/l21_sample
```

Pipeline lấy frame giữa mỗi cảnh do PySceneDetect phát hiện, sau đó loại frame gần trùng bằng normalized mean L1. Có thể hiệu chỉnh bằng `--scene-threshold`, `--dedup-threshold` và `--minimum-scene-frames`.

```bash
python -m backend.offline_indexing.cli validate \
  dataset-root/video \
  data/processed/l21_sample \
  dataset-root/map-keyframes \
  L21_V001 L21_V002 L21_V003
```

Đầu ra gồm `keyframes/`, `map-keyframes/`, `preprocessing-report.json` và `validation-report.json`.

## 4. SigLIP2 feature

Backbone duy nhất:

```text
ViT-gopt-16-SigLIP2-384, pretrained=webli
├── global embedding: N × 1536, normalized float16
└── dense feature: N × 1536 × 24 × 24, chỉ dùng khi rerank
```

Model weights 7.5 GB. Nên chạy bằng GPU; CPU chỉ phù hợp kiểm tra một vài ảnh.

```bash
python -m backend.offline_indexing.cli features \
  data/processed/l21_sample/keyframes \
  L21_V001 L21_V002 L21_V003 \
  --output data/processed/l21_sample/siglip2 \
  --device cuda \
  --batch-size 1 \
  --weights /path/to/open_clip_model.safetensors
```

Không thêm `--dense` khi chạy toàn bộ keyframe; chỉ bật cho tập ứng viên nhỏ cần rerank.

## 5. Chuẩn hóa Objects BTC

```bash
python -m backend.offline_indexing.cli objects \
  dataset-root/objects \
  dataset-root/map-keyframes \
  L21_V001 L21_V002 L21_V003 \
  --output data/processed/l21_sample/objects
```

Mỗi dòng JSONL gồm `video_id`, `keyframe_id`, `frame_id`, `timestamp`, danh sách nhãn và detection `{entity, class_name, class_id, score, box}`. Đây là Objects của keyframe BTC, không phải keyframe tự sinh.
