# I. BẢNG PHÂN CÔNG NHÂN SỰ & TECH STACK CHI TIẾT

## THÀNH VIÊN 1: INTEGRATION LEAD & AGENT REASONING
**Phụ trách:** Trợ lý AI & Lập luận và Bộ máy Dung hợp & Nộp bài

### Nhiệm vụ AI & Backend:
* Xây dựng Tác tử AI (Agentic AI) tối giản bằng cơ chế Chain-of-Thought (CoT) truyền thống (hạ cấp từ Tree-of-Thoughts phức tạp để tránh quá tải). Sử dụng Gemini 2.0 Flash hoặc OpenAI o1-mini làm lõi lập luận.
* Lập trình prompt hệ thống để Agent hoạt động như một bộ định tuyến (Router): phân rã câu hỏi của người dùng thành các điều kiện lọc cứng (Metadata, OCR, ASR) và yêu cầu tìm kiếm vector mềm.
* Cài đặt thuật toán Reciprocal Rank Fusion (RRF) ở Mô-đun 5 với hằng số chuẩn hóa $k=60$:

$$RRF\_Score(d) = \sum_{m \in M} rac{1}{60 + r_m(d)}$$

* Quản lý lịch sử hội thoại nhiều lượt (Conversational KIS) thông qua một mảng lưu trữ (Buffer Memory) văn bản thuần đơn giản để truyền ngược lại vào prompt của lượt kế tiếp.

### Nhiệm vụ Frontend (Streamlit):
* Dựng Khung chat Trợ lý ảo (Chatbot Interface) trực quan bằng các component hỗ trợ sẵn của Streamlit.
* Lập trình nút "Submit Answer" (Nộp bài) kết nối trực tiếp đến cổng chấm điểm của Ban tổ chức, tự động đóng gói kết quả theo đúng định dạng: Textual KIS (`<video_id>`, `<frame_id>`), Q&A (`<video_id>`, `<frame_id>`, `<answer>`), hoặc TRAKE (`<video_id>`, `<frame_id_1>`, ..., `<frame_id_n>`).

---

## THÀNH VIÊN 2: AI VISUAL GROUNDING & SKETCH SPECIALIST
**Phụ trách:** Phần Xác thực sâu (Tìm kiếm Thị giác & Vector).

### Nhiệm vụ AI & Backend:
* Cài đặt mô hình OWL-ViT hoặc Grounding DINO để thực hiện dán nhãn ngữ cảnh sâu (Visual Grounding). Các mô hình này có sẵn trên Hugging Face, dễ cài đặt và chạy nhẹ hơn GLIP/UNINEXT rất nhiều.
* Đóng gói thành API `/rerank` (sử dụng FastAPI) chỉ kích hoạt khi danh sách ứng viên từ Database gửi lên đã được lọc thô xuống dưới 50 khung hình để tối ưu hóa tài nguyên phần cứng và đảm bảo thời gian chạy dưới 1 giây.
* Xây dựng API `/sketch` sử dụng mô hình Sketch-CLIP hoặc SDXL-Turbo kết hợp ControlNet (LCM LoRA) để dịch nhanh nét vẽ phác thảo thành ảnh thực tế/vector biểu diễn.

### Nhiệm vụ Frontend (Streamlit):
* Tích hợp Bảng vẽ phác thảo (Sketch Board UI) và bộ mã hóa ảnh kéo thả (Query by Visual Example) lên giao diện chính.
* Hiển thị khung chữ nhật đỏ (Bounding Box) khoanh vùng vật thể được tìm thấy từ OWL-ViT/Grounding DINO trên ảnh kết quả.

---

## THÀNH VIÊN 3: TEMPORAL ENGINE & STAR TOOLS SPECIALIST
**Phụ trách:** Căn chỉnh thời gian - Bài toán TRAKE.

### Nhiệm vụ AI & Backend:
* Lập trình Thuật toán Căn chỉnh Chuỗi Thời gian Đa chặng (Multi-Stage Temporal Alignment Engine) để giải quyết bài toán TRAKE:
  1. Nhận chuỗi phân tách từ Agent: Truy vấn quá khứ ($Q_{previous}$), hiện tại ($Q_{current}$) và tương lai ($Q_{next}$).
  2. Quét chuỗi thời gian tăng dần để tìm các khung hình thuộc cùng một video thỏa mãn: $index(r_p) < index(r_c) < index(r_n)$.
  3. Tính toán lại điểm số tổng hợp theo trọng số:
     $$S_{final}(r_c) = w_c \cdot Score(r_c) + w_p \cdot Score(r_p) + w_n \cdot Score(r_n)$$
* Đóng gói công cụ `temporal-grounding-tool` (xác định phân cảnh video dựa trên mốc thời gian) và công cụ `image-caption-tool` (gọi mô hình Qwen2.5-VL thô để mô tả nhanh phân cảnh tiềm năng).

### Nhiệm vụ Frontend (Streamlit):
* Dựng thanh cuộn dải thời gian (Timeline Expansion Viewer) cho phép người dùng cuộn xem nhanh 30 giây trước và sau của một khung hình bất kỳ để xác thực diễn tiến hành động.
* Thiết kế khung hiển thị album chuỗi thời gian TRAKE theo đúng thứ tự logic.

---

## THÀNH VIÊN 4: DATA PIPELINE & OFFLINE INDEXING ENGINEER
**Chịu trách nhiệm:** Toàn bộ giai đoạn lập chỉ mục offline (Offline Stage).

### Nhiệm vụ AI & Backend (Offline):
* Xây dựng pipeline tiền xử lý video thô: Tách cảnh tự động (AutoShot) kết hợp bộ lọc khoảng cách $L_1$ norm để loại bỏ triệt để các khung hình trùng lặp, giữ lại các khung hình ngữ nghĩa (semantic keyframes) tiêu biểu.
* Chạy trích xuất đặc trưng song song: Vector thị giác bối cảnh toàn cục bằng OpenCLIP ViT-L/14 và vector chi tiết tinh vi bằng SigLIP 2 (Google, 2025).
* Chạy offline bóc tách văn bản OCR tiếng Việt bằng Qwen2.5-VL và bóc tách giọng nói ASR bằng Whisper / PhoWhisper.
* Tận dụng file Objects JSON Faster R-CNN do BTC cung cấp sẵn để gán nhãn vật thể rời rạc.

---

## THÀNH VIÊN 5: SEARCH DATABASE & BENCHMARK ENGINEER
**Chịu trách nhiệm:** Thiết lập hạ tầng cơ sở dữ liệu và tối ưu hóa các API tìm kiếm thô thời gian thực (< 200ms)

### Nhiệm vụ AI & Backend (Online):
* Thiết lập duy nhất một cụm cơ sở dữ liệu dùng chung (Centralized Hybrid DB) chạy Docker bao gồm: Qdrant hoặc Milvus (cho Vector DB, chỉ mục HNSW) và Elasticsearch (cho dữ liệu OCR, ASR, Objects JSON và YouTube Metadata của BTC).
* Xây dựng API tìm kiếm hỗn hợp `/db/query` thực hiện tìm kiếm kết hợp lọc cứng Elasticsearch và so khớp vector mềm để trả về Top-100 ứng viên tiềm năng.
* Áp dụng kỹ thuật hiệu chuẩn trung tâm Mean-Centering / GR-CLIP để giảm khoảng cách miền (Modality Gap) giữa vector chữ và ảnh.
* Xây dựng script tự động tính toán các chỉ số kiểm thử: Recall@K, MAP, Latency trên dữ liệu mẫu của BTC.

### Nhiệm vụ Frontend (Streamlit):
* Dựng bộ lọc thuộc tính thanh bên (Sidebar Filter): Lọc theo khoảng ngày/giờ, kênh YouTube, nhãn vật thể Faster R-CNN và từ khóa OCR/ASR.
* Thiết kế lưới hiển thị kết quả nhóm theo video (Video-Grouped Grid Layout): tự động gom các khung hình cùng video thành các dải thời gian liên tục giúp người dùng quan sát bối cảnh nhanh chóng.
