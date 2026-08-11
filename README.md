# AIC2026 Multimedia Agent

## Offline indexing pipeline

Pipeline hiện tại **tái sử dụng artefact BTC**, kiểm tra ánh xạ keyframe → frame thật và tạo đầu vào chung cho các thành viên khác. Chưa chạy lại OCR, ASR, scene detection hoặc model embedding khi chưa có dữ liệu thiếu/benchmark yêu cầu.

### Dữ liệu vào

```text
<dataset>/
├── video/<video_id>.mp4               # tùy chọn khi nạp artefact BTC
├── keyframes/<video_id>/*.jpg
├── map-keyframes/<video_id>.csv       # frame_idx + pts_time (hoặc frame_id/timestamp)
├── clip/<video_id>.npy                # cùng thứ tự với keyframe
├── objects/<video_id>/*.json          # tùy chọn
└── media-info/<video_id>.json         # tùy chọn
```

Tên thư mục thông dụng có dấu cách/gạch nối và chữ hoa/thường cũng được nhận diện; xem `ARTIFACT_DIRECTORY_NAMES` trong `backend/offline_indexing/artifact_indexer.py`.

### Build và bàn giao

```bash
.venv/bin/python -m backend.offline_indexing.cli build /home/depp/AIC/AIC26/batch1 \
  --video-prefix L21 \
  --output data/offline_index/l21
```

Sinh `data/offline_index/{manifest.json,records.jsonl,videos.jsonl,features.npy}`. `feature_row` trong mỗi record trỏ đúng dòng tương ứng trong `features.npy`; `frame_id` là frame video dùng để nộp bài, không phải số thứ tự keyframe.

### API tìm kiếm thô

```bash
.venv/bin/python -m backend.offline_indexing.cli serve
curl http://127.0.0.1:8000/health
curl 'http://127.0.0.1:8000/frames?video_id=L01_V001&limit=20'
curl -X POST http://127.0.0.1:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"vector":[0.1,0.2],"top_k":20}'
```

Vector truy vấn phải cùng model/số chiều với `features.npy`. API giới hạn Top-100 theo luật thi và chỉ là baseline để kiểm tra tích hợp; Thành viên 5 có thể nạp thẳng bốn file đầu ra vào Qdrant/Milvus và Elasticsearch.

### Kiểm tra

```bash
.venv/bin/python -m unittest tests/test_offline_indexing.py
```

### Công việc offline indexing còn lại

- Chạy build trên bộ dữ liệu BTC thật và xử lý sai khác **thực tế** của schema/path nếu có.
- Benchmark độ phủ CLIP BTC trước khi quyết định trích OpenCLIP/SigLIP mới.
- Chỉ chạy PySceneDetect + OpenCV dedup cho video không có keyframe.
- Chỉ chạy Whisper/PaddleOCR cho video thiếu ASR/OCR; lưu timestamp và provenance của model.
- Bàn giao `manifest.json`, schema record và kết quả kiểm tra cho Thành viên 5.
