# ĐẶC TẢ GIAO THỨC API (API CONTRACT)

## 1. Quy chuẩn Kết nối và Quản trị Lỗi Hệ thống
Trong một chiến dịch ngắn ngày, việc thống nhất giao thức RESTful API thông qua FastAPI là yếu tố sống còn để triệt tiêu xung đột giữa Backend (xử lý mô hình AI) và Frontend (Streamlit). FastAPI không chỉ tối ưu về tốc độ mà còn tự động hóa việc tạo tài liệu Swagger/ReDoc, cho phép các thành viên debug độc lập mà không cần chờ đợi module của người khác.

### 1.1. Tiêu chuẩn Giao thức
* **Định dạng:** JSON (UTF-8).
* **Headers:** Bắt buộc `Content-Type: application/json` và `X-Session-ID` để truy vết luồng suy luận của Tác tử (Agent).
* **URL Versioning:** Toàn bộ Endpoint bắt đầu bằng `/v1/` để đảm bảo tính tương thích khi nâng cấp các mô hình nền tảng.

### 1.2. Danh mục Mã lỗi và Chiến lược Phản hồi

| Mã lỗi HTTP | Tên lỗi | Mô tả ngữ cảnh trong AIC | Hướng dẫn xử lý cho Dev |
| :--- | :--- | :--- | :--- |
| **200** | OK | Truy vấn thành công, trả về danh sách frame. | Hiển thị lên lưới SOM (Self-Organizing Map). |
| **400** | Bad Request | Sai Schema JSON hoặc tham số bộ lọc (filters) không hợp lệ. | Kiểm tra lại cấu trúc pydantic trên Backend. |
| **401** | Unauthorized | Thiếu API Key hoặc Session ID đã hết hạn. | Re-login hoặc làm mới session từ UI. |
| **404** | Not Found | Không tìm thấy Keyframe ID hoặc Video ID trong Database. | Kiểm tra đồng bộ hóa giữa Qdrant và Elasticsearch. |
| **500** | Server Error | Lỗi logic mô hình (CLIP/Qwen) hoặc crash tiến trình. | Kiểm tra Logs; Khởi động lại Worker GPU. |
| **503** | Service Unavailable | Lỗi tràn bộ nhớ GPU (OOM) do nạp quá nhiều model. | Giải phóng VRAM; Chuyển sang mô hình nhỏ hơn. |

### 1.3. Cấu trúc Response chuẩn
```json
{
  "status": "success | error",
  "data": {},
  "message": "Thông báo trạng thái/lỗi",
  "execution_time": "0.35s",
  "agent_reasoning": "Chuỗi suy luận System 2 (CoT) nếu có"
}
```
Sự ổn định của tầng kết nối này là tiền đề thực thi các thuật toán lý luận không-thời gian (STAR Framework) phức tạp ở các tầng phía trên.

---

## 2. Nền tảng Thuật toán và Tư duy Tối ưu hóa Phần cứng (VRAM/GPU)
Hệ thống tuân thủ chiến lược "Lean & Mean", tập trung vào hiệu suất thực tế trên Laptop GPU cá nhân bằng cách chia tầng xử lý (Cascading), tránh việc nạp các mô hình khổng lồ cho toàn bộ dataset.

### 2.1. Dung hợp Kết quả bằng RRF (Reciprocal Rank Fusion)
Hệ thống dung hợp kết quả từ luồng Dense Retrieval (Qdrant - Vector) và Sparse Retrieval (Elasticsearch - BM25) theo công thức:

$$RRF\_Score(d) = \sum_{r \in R} rac{1}{k + r(d)}$$

Với hằng số $k=60$. Phương pháp này giúp cân bằng giữa tìm kiếm ngữ nghĩa trừu tượng và tìm kiếm từ khóa chính xác (OCR, Objects).

### 2.2. Công cụ Căn chỉnh Thời gian TRAKE (Multi-Stage Temporal Alignment)
Logic căn chỉnh dựa trên việc xác định các Semantic Keyframes (khung hình mang nghĩa nội dung), phân biệt rõ với I-Frames kỹ thuật.
* **Điều kiện ràng buộc:** $index(r\_p) < index(r\_c) < index(r\_n)$ (Quá khứ < Hiện tại < Tương lai).
* Điểm tích hợp $S_{final}(r_c)$ ưu tiên các video chứa trọn vẹn chuỗi sự kiện trong cùng một dòng thời gian liên tục.

### 2.3. Kiến trúc Lọc phân tầng (Cascading Filtering)
Để bảo vệ VRAM và tối ưu Latency, quy trình xử lý được chia làm 3 luồng:
1. **Luồng 1 (Late-Fusion):** Sử dụng SigLIP 2 (So400m) kết hợp BM25 trên Elasticsearch. Lọc thô Top-50 ứng viên trong <400ms.
2. **Luồng 2 (Re-score):** Kích hoạt Grounding DINO để xác minh tọa độ vật thể (Bounding Box) cho các truy vấn yêu cầu không gian. Thời gian xử lý <200ms.
3. **Luồng 3 (Visual Grounding - System 2):** Sử dụng Qwen2.5-VL (7B variant) để thực hiện lập luận sâu cho Top-5 kết quả. Đây là tầng duy nhất nạp mô hình nặng, xử lý trong <800ms.

### 2.4. Phân tích Hiệu quả VRAM
Bằng cách chỉ gọi Qwen2.5-VL cho 5-10 khung hình cuối cùng, hệ thống tiết kiệm 70% VRAM so với việc inference hàng loạt. Việc sử dụng SigLIP 2 So400m/NaFlex mang lại độ chính xác cao trong việc đọc chữ và nhận diện vật thể nhỏ mà vẫn duy trì mức chiếm dụng bộ nhớ thấp.

---

## 3. Đặc tả Chi tiết 7 Endpoints Cốt lõi

### 3.1. Endpoint 1: GET `/v1/health`
* **Chức năng:** Kiểm tra Heartbeat của Qdrant, Elasticsearch và trạng thái nạp (loaded/unloaded) của mô hình SigLIP 2 & Qwen.
* **Latency Budget:** <50ms.

### 3.2. Endpoint 2: POST `/v1/db/query` (Truy vấn Hỗn hợp)
* **Request Schema:**
```json
{
  "raw_query": "Người phụ nữ mặc áo đỏ tại HTV9",
  "filters": {
    "objects": ["person", "laptop"],
    "ocr": "HTV9",
    "timestamp_range": ["2026-03-14T10:00:00Z", "2026-03-14T12:00:00Z"]
  },
  "use_hippo_rag": true
}
```
`timestamp_range` phải gồm hai ISO 8601 datetime đầy đủ có múi giờ theo thứ tự
`[start, end]`; `start` không được lớn hơn `end`.

* **Response:** Trả về `cluster_id` và `som_coords` (tọa độ SOM) để UI hiển thị lưới kết quả trực quan.
* **Latency Budget:** <400ms.

### 3.3. Endpoint 3: POST `/v1/rerank/early-fusion`
* **Chức năng:** Dùng Qwen2.5-VL xác minh nội dung cho bài toán Q&A.
* **Trường bắt buộc trong Response:** `vqa_answer` (chuỗi văn bản câu trả lời).
* **Latency Budget:** <600ms.

### 3.4. Endpoint 4: POST `/v1/query/image-example`
* **Chức năng:** Image-to-Image search khi người dùng click vào keyframe.
* **Latency Budget:** <100ms.

### 3.5. Endpoint 5: POST `/v1/query/sketch`
* **Chức năng:** Nhận ảnh Base64 từ Canvas, qua ControlNet để ánh xạ đặc trưng đưa vào CLIP/SigLIP.
* **Latency Budget:** <300ms.

### 3.6. Endpoint 6: POST `/v1/temporal/align` (TRAKE)
* **Chức năng:** Phân tách truy vấn thành $Q_{past}$, $Q_{current}$, $Q_{future}$ để tìm chuỗi khung hình ngữ nghĩa.
* **Latency Budget:** <200ms.

### 3.7. Endpoint 7: POST `/v1/submission/submit`
* **Chức năng hiện tại:** Kiểm tra và đóng gói dữ liệu cục bộ theo định dạng BTC. Endpoint chưa gửi dữ liệu đến server cuộc thi và luôn trả `submitted=false`; `validated=true` chỉ có nghĩa là validation cục bộ thành công.
```json
{
  "task_type": "KIS | VQA | TRAKE",
  "results": [
    {"video_id": "L01_V001", "frame_id": 1500, "answer": "màu đỏ"}
  ]
}
```
* **Latency Budget:** <100ms.

### 3.8. Agent routing
* `POST /v1/agent/route` nhận `AgentRequest`; `X-Session-ID` bắt buộc và phải khớp `session_id` trong body. Caller có thể truyền `task_type` (`KIS`, `VQA`, `TRAKE`); khi bỏ trống, provider local dùng heuristic best-effort và ưu tiên KIS nếu không đủ bằng chứng.
* `DELETE /v1/agent/session/{session_id}` chỉ xóa session khớp header của caller.
* `decision_summary` là mô tả công khai ngắn về task và bước xử lý, không phải Chain-of-Thought hay hidden reasoning. `agent_reasoning` luôn là `null`.
* Provider mặc định hiện tại là fallback heuristic chạy local, deterministic và không gọi Gemini/network.

---

## 4. Chiến thuật Nộp bài và Tối ưu hóa Điểm số (Final Score)
Điểm cuối cùng được tính bằng trung bình cộng các chỉ số $R@k$ ($k \in \{1, 5, 20, 50, 100\}$).

### 4.1. Giới hạn kết quả
Mỗi truy vấn được gửi tối đa 100 câu trả lời và không bắt buộc đủ 100. Hệ thống giữ nguyên thứ tự kết quả thực có, không nhân bản candidate để lấp đầy danh sách.

### 4.2. Hệ thống Lý luận System 2
Tác tử AI sẽ tự động phân loại nhiệm vụ:
* **KIS:** Chỉ định vị video/frame.
* **VQA:** Kích hoạt bộ lọc OCR/Object và Qwen2.5-VL để trích xuất answer.
* **TRAKE:** Sử dụng logic căn chỉnh thời gian để kiểm tra thứ tự index. Nếu nộp sai `video_id`, điểm sẽ về 0 ngay lập tức, do đó hệ thống ưu tiên kiểm tra tính hội tụ của video trước khi nộp.

---

## 5. Phân công Vai trò và Quy trình Tích hợp

### 5.2. Quy trình Tích hợp Streamlit UI (ACI - Agent Computer Interface)
Bắt buộc sử dụng Mock API trong 48h đầu tiên: Thành viên 4 phải xây dựng giao diện dựa trên dữ liệu JSON giả lập theo đúng Schema tại Chương 3. Điều này cho phép UI hoàn thiện song song với quá trình tinh chỉnh mô hình AI. Hệ thống UI phải hỗ trợ hai chế độ: Khám phá (Exploration qua SOM) và Khai phá (Exploitation qua Re-query).
