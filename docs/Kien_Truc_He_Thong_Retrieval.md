# Thiết Kế Hệ Thống Trợ Lý Ảo Thông Minh Phân Tích Và Truy Xuất Dữ Liệu Multimedia Chuẩn AIC 2026, LSC Và VBS

## Bối Cảnh, Bài Toán Và Các Thách Thức Kỹ Thuật Cốt Lõi

Sự bùng nổ của các thiết bị ghi hình cá nhân như camera hành trình, kính thông minh và thiết bị đeo lifelogging đã thúc đẩy sự dịch chuyển mạnh mẽ từ mô hình giám sát truyền thống (Surveillance) sang mô hình nhật ký cá nhân góc nhìn thứ nhất (Sousveillance) [1]. Khác với dữ liệu từ hệ thống camera an ninh công cộng có góc quay cố định và bối cảnh lặp đi lặp lại, dữ liệu sousveillance mang đặc tính góc nhìn thứ nhất (egocentric/POV), chuyển động liên tục, ánh sáng thay đổi bất thường và chứa nhiều thông tin nhiễu nhưng lại giàu giá trị ngữ nghĩa cá nhân.

Thách thức trung tâm của Hội thi Thách thức Trí tuệ Nhân tạo TP.HCM (AIC 2026) là xây dựng một trợ lý ảo thông minh có khả năng phân tích, định vị và truy xuất chuyên sâu các khoảnh khắc chính xác từ hàng nghìn giờ video multimedia không cấu trúc, tuân theo thể thức của các cuộc thi quốc tế như Lifelog Search Challenge (LSC) và Video Browser Showdown (VBS) [1].

### Phân Loại Nhiệm Vụ Truy Xuất Tại Vòng Sơ Tuyển AIC 2026

Bài toán truy xuất tại AIC 2026 được chuẩn hóa thành ba dạng nhiệm vụ chính với các yêu cầu kỹ thuật và định dạng đầu ra nghiêm ngặt:

1. **Tìm kiếm chính xác theo văn bản (Textual Known-Item Search - Textual KIS):** Nhiệm vụ yêu cầu xác định chính xác một phân cảnh duy nhất (Ground Truth) trong kho dữ liệu dựa trên mô tả ngôn ngữ tự nhiên. Hệ thống phải trả về mã nhận dạng video (`video_id`) và chỉ số khung hình (`frame_id`) [2]. Câu trả lời được coi là chính xác nếu `video_id` trùng khớp với đáp án ($v_i = GT_v$) và `frame_id` nằm trong khoảng thời gian quy định ($id_i \in [s, e]$) [2]. Hàm tính điểm R-Score cho tác vụ này được biểu diễn dưới dạng:

   $$R	ext{-}Score(r_i) = \mathbb{I}(v_i = GT_v \wedge id_i \in [s, e])$$

   Trong đó $\mathbb{I}(\cdot)$ là hàm chỉ thị trả về giá trị 1 khi tất cả điều kiện thỏa mãn và 0 trong trường hợp ngược lại.

2. **Truy vấn dạng Hỏi - Đáp (Visual Question Answering - VQA):** Nhiệm vụ này đòi hỏi hệ thống vừa phải định vị khoảnh khắc chứa thông tin vừa thực hiện suy luận để trả lời câu hỏi bằng văn bản. Định dạng đầu ra yêu cầu bộ ba thông tin $<video\_id, frame\_id, answer>$ [2]. Điểm tương quan $R	ext{-}Score$ chỉ đạt giá trị tối đa khi cả vị trí không - thời gian lẫn nội dung câu trả lời đều chính xác về mặt ngữ nghĩa ($a_i = GT_a$) [2]:

   $$R	ext{-}Score(r_i) = \mathbb{I}(v_i = GT_v \wedge id_i \in [s, e] \wedge a_i = GT_a)$$

3. **Truy xuất và căn chỉnh sự kiện video theo thời gian (Temporal Retrieval and Alignment of Key Events - TRAKE):** TRAKE là tác vụ phức hợp nhằm đánh giá khả năng hiểu sâu chuỗi hành động có tính thứ tự thời gian [2]. Hệ thống phải truy xuất đúng $video\_id$ duy nhất và xác định danh sách $N$ khung hình ngữ nghĩa (semantic keyframes) tương ứng với các giai đoạn của chuỗi sự kiện [2]. Nếu nộp sai $video\_id$, hệ thống nhận 0 điểm lập tức. Nếu đúng video, điểm số được tính bằng tỷ lệ phần trăm các khung hình thành phần rơi vào đúng khoảng cho phép $[s_j, e_j]$ [2]:

   $$R	ext{-}Score(r_i) = egin{cases} rac{1}{N}\sum_{j=1}^{N}\mathbb{I}(id_{i,j} \in [s_j, e_j]) & 	ext{nếu } v_i = GT_v \ 0 & 	ext{nếu } v_i 
eq GT_v \end{cases}$$

### Phương Thức Đánh Giá Hệ Thống Và Thách Thức Cốt Lõi

Để khuyến khích các hệ thống không chỉ tìm ra kết quả đúng mà còn phải xếp kết quả đó ở các vị trí đầu tiên, điểm số cuối cùng ($Final\ Score$) cho mỗi câu hỏi được tính bằng trung bình cộng của các chỉ số $R@k$ với $k \in \{1, 5, 20, 50, 100\}$ [2]. Giá trị $R@k$ đại diện cho điểm $R	ext{-}Score$ cao nhất đạt được trong $k$ câu trả lời hàng đầu [2]:

$$R@k = \max_{1 \le i \le k} \{R	ext{-}Score(r_i)\}$$

$$Final\ Score = rac{1}{5} \sum_{k \in \{1, 5, 20, 50, 100\}} R@k$$

Sự thành công của hệ thống phụ thuộc vào khả năng giải quyết ba rào cản kỹ thuật cốt lõi trong xử lý dữ liệu lớn multimedia [1]. Thứ nhất, khoảng cách ngữ nghĩa (**Semantic Gap**) xuất hiện do sự chênh lệch giữa biểu diễn ngôn ngữ tự nhiên trừu tượng của người dùng và các điểm ảnh thô (raw pixels) mà máy tính ghi nhận. Thứ hai, vấn đề thưa thớt dữ liệu (**Data Sparsity**) kết hợp với quy mô hàng nghìn giờ video khiến việc thực hiện suy luận sâu trực tiếp trên toàn bộ dữ liệu trở nên bất khả thi về mặt chi phí và thời gian thực hiện. Cuối cùng, các ràng buộc logic thời gian (**Temporal Logic Constraints**) đòi hỏi mô hình phải nhận biết chính xác thứ tự diễn tiến trước - sau của chuỗi hành động thay vì chỉ so khớp các từ khóa độc lập.

---

## Kiến Trúc Pipeline Hệ Thống Truy Xuất Multimedia Toàn Diện

Một kiến trúc hệ thống hiện đại phục vụ AIC 2026 được thiết kế phân tách rõ ràng thành hai giai đoạn chính: **Giai đoạn Lập chỉ mục ngoại tuyến (Offline Indexing)** nhằm xử lý chuẩn hóa dữ liệu lớn, và **Giai đoạn Truy xuất & Lập luận trực tuyến (Online Retrieval & Reasoning)** chịu trách nhiệm tương tác và phản hồi thời gian thực.

### Giai Đoạn Lập Chỉ Mục Ngoại Tuyến

Quy trình lập chỉ mục ngoại tuyến bắt đầu từ dữ liệu video thô và trải qua các bước xử lý đa phương thức liên tục để trích xuất biểu diễn đặc trưng. Đầu tiên, toàn bộ video được đưa qua tiến trình tách cảnh và lấy mẫu khung hình tối ưu [6]. Thay vì lưu trữ toàn bộ khung hình kỹ thuật (I-Frames) gây dư thừa tài nguyên, hệ thống áp dụng thuật toán AutoShot kết hợp bộ lọc khoảng cách $L_1$-norm để tự động loại bỏ các khung hình trùng lặp, chỉ giữ lại các khung hình ngữ nghĩa tiêu biểu (semantic keyframes) [6].

Sau khi có tập hợp khung hình tối ưu, dữ liệu được phân luồng xử lý song song trên nhiều mô hình deep learning [1]. Ở luồng trích xuất đặc trưng thị giác, hệ thống kết hợp mô hình OpenCLIP (như ViT-L/14) để thu nhận bối cảnh toàn cục và mô hình SigLIP 2 (Google, 2025) để bắt các chi tiết cục bộ mật độ cao. SigLIP 2 với hàm mất mát Sigmoid kết hợp cùng các kỹ thuật tự chưng cất (SILC) và dự đoán mặt nạ patch (TIPS) giúp gia tăng vượt trội khả năng nhận diện vật thể nhỏ, vùng chữ và chi tiết cấu trúc. Nhằm triệt tiêu độ lệch đặc trưng giữa hai miền dữ liệu ảnh và văn bản, thuật toán hiệu chuẩn trung tâm (Mean-Centering / GR-CLIP) được áp dụng để đưa các vector về cùng không gian biểu diễn chuẩn [10].

Ở các luồng bổ trợ, mô hình Large Vision Language Model Qwen2.5-VL thực hiện đọc văn bản xuất hiện trong ảnh (OCR) [6]. Mô hình OpenAI Whisper (hoặc PhoWhisper được tối ưu cho tiếng Việt) đảm nhận việc chuyển đổi toàn bộ băng âm thanh thuyết minh và hội thoại thành văn bản (ASR) gắn kèm mốc thời gian. Dữ liệu phát hiện vật thể (Objects) từ Faster R-CNN (huấn luyện trên Open Images V4) cung cấp danh mục nhãn chi tiết kèm tọa độ hộp bao. Tất cả dữ liệu sau khi xử lý được lưu trữ vào kiến trúc vùng lưu trữ kép: các vector đặc trưng liên tục (Dense Vectors) được nạp vào Vector Database (Milvus hoặc Faiss) hỗ trợ truy vấn ANN thời gian thực dưới 10ms [10], trong khi các dữ liệu văn bản có cấu trúc (Sparse Data như OCR, ASR, Metadata YouTube, nhãn vật thể) được lập chỉ mục trên Elasticsearch [1].

### Giai Đoạn Truy Xuất Và Lập Luận Trực Tuyến

Khi người dùng nhập truy vấn ngôn ngữ tự nhiên, hệ thống kích hoạt Tác tử AI (Agentic AI Engine) làm nhiệm vụ phân tích ý định dựa trên cơ chế tư duy System 2 (Chain-of-Thought) [13]. Tác tử AI tự động tách câu hỏi thành các điều kiện lọc cấu trúc (thời gian, kênh, chữ xuất hiện) và các đoạn mô tả hình ảnh trừu tượng, đồng thời thực hiện mở rộng truy vấn tự động (Generative Query Expansion) để sinh ra các biến thể mô tả đồng nghĩa tiếng Anh và tiếng Việt.

Truy vấn sau khi mở rộng được gửi đồng thời đến Vector Database và Elasticsearch [1]. Kết quả danh sách thứ hạng từ hai luồng tìm kiếm được dung hợp thông qua thuật toán Reciprocal Rank Fusion (RRF) để đạt sự cân bằng giữa độ rộng và độ chính xác:

$$RRF\_Score(d) = \sum_{m \in M} rac{1}{k + r_m(d)}$$

Trong đó $M$ là tập hợp các phương thức tìm kiếm, $r_m(d)$ là thứ hạng của tài liệu hoặc khung hình $d$ trong phương thức $m$, và $k$ là hằng số chuẩn hóa (thường chọn $k=60$) [6].

Đối với các bài toán yêu cầu căn chỉnh thời gian phức tạp như TRAKE, hệ thống chuyển tiếp kết quả RRF qua Thuật toán Căn chỉnh Chuỗi Thời gian Đa chặng (Multi-Stage Temporal Alignment Engine). Truy vấn ban đầu được phân tách thành bộ ba ngữ cảnh: Truy vấn quá khứ ($Q_{previous}$), Truy vấn hiện tại ($Q_{current}$), và Truy vấn tương lai ($Q_{next}$) [6]. Sau khi truy vấn ba danh sách độc lập $R_{previous}, R_{current}, R_{next}$, thuật toán quét qua các ứng viên $r_c \in R_{current}$ để tìm các khung hình $r_p \in R_{previous}$ và $r_n \in R_{next}$ thuộc cùng một $video\_id$ thỏa mãn thứ tự thời gian $index(r_p) < index(r_c) < index(r_n)$ [6]. Điểm số tổng hợp được tính toán lại theo công thức:

$$S_{final}(r_c) = w_c \cdot Score(r_c) + w_p \cdot Score(r_p) + w_n \cdot Score(r_n)$$

Những video chứa trọn vẹn chuỗi diễn tiến theo đúng thứ tự logic sẽ được nâng hạng lên các vị trí Top đầu. Tại chặng cuối cùng, Top-50 ứng viên tốt nhất được đưa qua mô hình Early-Fusion (như Qwen2.5-VL hoặc GLIP) để thực hiện xác minh ngữ nghĩa sâu (visual grounding) và loại bỏ hoàn toàn các kết quả báo động giả (false positives) trước khi xuất ra màn hình người dùng.

---

## Lựa Chọn Mô Hình, Công Nghệ Và Bảng So Sánh Kỹ Thuật

Để xây dựng một hệ thống tối ưu hóa giữa tốc độ truy vấn thực thời, chi phí tài nguyên và độ chính xác phân giải bài toán, các công nghệ hàng đầu được tổng hợp và đề xuất chi tiết theo bảng sau:

| Thành Phần Hệ Thống | Công Nghệ / Mô Hình Đề Xuất | Vai Trò Chính Trong Pipeline | Lý Do Lựa Chọn & Ưu Điểm Kỹ Thuật |
| :--- | :--- | :--- | :--- |
| **Tách Khung Hình** | AutoShot + $L_1$-norm Filtering | Tách cảnh tự động, lọc khung hình trùng lặp. | Giảm 70% dữ liệu dư thừa nhưng bảo toàn trọn vẹn các semantic keyframe [2]. |
| **Visual Encoder (Global)** | OpenCLIP ViT-L/14 | Trích xuất đặc trưng hình ảnh ngữ cảnh toàn cục [2]. | Chuẩn hóa cao, tốc độ trích xuất nhanh, hoạt động ổn định trên các benchmark LSC/VBS [3]. |
| **Visual Encoder (Fine)** | SigLIP 2 (So400m/NaFlex) | Trích xuất đặc trưng chi tiết, vật thể nhỏ và chữ. | Sử dụng Sigmoid loss kết hợp tự chưng cất; vượt trội CLIP ở khả năng đọc chữ và định vị. |
| **Visual Language Model** | Qwen2.5-VL (3B/7B) | Trích xuất OCR, mô tả phân cảnh sâu, giải VQA. | Hiệu năng SOTA trên tác vụ thị giác-ngôn ngữ, hỗ trợ hiểu ngữ cảnh tiếng Việt xuất sắc. |
| **Chuyển Âm Thanh - Chữ** | OpenAI Whisper / PhoWhisper | Chuyển thoại video thành văn bản ASR. | Nhận diện chính xác tiếng Việt trong điều kiện có tiếng ồn môi trường hoặc nhạc nền. |
| **Phát Hiện Vật Thể** | Faster R-CNN (Open Images V4) | Gán nhãn vật thể rời rạc kèm tọa độ bounding box [1]. | Dữ liệu JSON đã được BTC cung cấp sẵn, dễ dàng lập chỉ mục trực tiếp vào Elasticsearch [2]. |
| **Tác Tử Lập Luận (Agent)** | Gemini 2.0 Flash / OpenAI o1-mini | Điều phối System 2, phân tích logic và gọi công cụ [13]. | Khả năng suy luận chuỗi (CoT), tự sửa lỗi và tối ưu hóa câu truy vấn phức tạp [13]. |
| **Vector Database** | Milvus / Faiss | Lưu trữ và tìm kiếm tương đồng vector chuỗi đại số. | Hỗ trợ lưu trữ hàng triệu vector, thời gian truy vấn < 10ms, hỗ trợ lọc kết hợp metadata [12]. |
| **Metadata Engine** | Elasticsearch | Lọc và tìm kiếm văn bản (OCR, ASR, Objects, Metadata) [1]. | Khả năng tìm kiếm BM25 linh hoạt, hỗ trợ ghép nối điều kiện logic Boolean mạnh mẽ. |

---

## Tối Ưu Hóa Dữ Liệu Đặc Thù Tại Việt Nam

Xử lý dữ liệu đặc thù tại Việt Nam đòi hỏi các giải pháp kỹ thuật tinh chỉnh nhằm giải quyết rào cản ngôn ngữ và cấu trúc dữ liệu truyền thông bản địa. Tác tử AI được thiết lập quy trình chuyển đổi tự động các cụm từ mang tính ngữ cảnh văn hóa hoặc từ lóng tiếng Việt sang thuật ngữ chuẩn hóa kép (Anh - Việt) trước khi thực hiện truy vấn vector [1]. Dữ liệu video thi đấu chứa lượng lớn tin tức truyền hình (như HTV9) với các dải chữ tiêu đề tĩnh chạy ở chân màn hình. Việc cấu hình mô hình Qwen2.5-VL tập trung quét vùng địa lý hình ảnh cố định giúp trích xuất chính xác tiêu đề bản tin và mốc thời gian. Đối với dữ liệu âm thanh đời sống có độ nhiễu cao, việc tích hợp bộ lọc tạp âm tiền xử lý trước khi đưa qua PhoWhisper giúp gia tăng đáng kể độ chính xác của chỉ mục từ khóa ASR.

---

## Giao Diện Tương Tác Vô Cùng Hiệu Quả Và Tác Động Của Yếu Tố Con Người

Thực tế thi đấu tại các giải đấu quốc tế như VBS và LSC chứng minh rằng một mô hình truy xuất có độ chính xác cao vẫn có thể thất bại nếu giao diện người dùng (UI/UX) gây cản trở tốc độ quan sát và đưa ra quyết định của chuyên viên thao tác [3].

### Chiến Lược Bố Trí Lưới Hiển Thị Kết Quả Và Tương Tác Phản Hồi

Nghiên cứu đánh giá các hệ thống vô địch LSC và VBS (như NII-UIT, VISIONE, lifeXplore) cho thấy việc sắp xếp kết quả đóng vai trò quyết định đến hiệu suất truy tìm [3]. Hệ thống đề xuất áp dụng **thiết kế lưới nhóm theo video (Video-Grouped Grid Layout)** [3]. Thay vì hiển thị danh sách các khung hình rời rạc sắp xếp đơn thuần theo điểm số tương đồng, giao diện tự động gom các khung hình ứng viên thuộc cùng một `video_id` thành các dải thời gian liên tục [3]. Thiết kế này giúp người dùng lập tức nắm bắt được ngữ cảnh chung của phân cảnh và loại bỏ các video sai lệch mà không mất thời gian xem từng ảnh đơn lẻ [3]. Khi người dùng chọn một khung hình, giao diện cung cấp tính năng mở rộng dải thời gian (**Timeline Expansion**) cho phép cuộn nhanh các khung hình trước và sau đó 30 giây để kiểm tra chuỗi hành động, phục vụ trực tiếp cho các nhiệm vụ KIS và TRAKE.

Để khắc phục các hạn chế của mô hình truy xuất khi gặp câu truy vấn phức tạp, hệ thống tích hợp hai cơ chế phản hồi linh hoạt:

* **Cơ chế Mở Rộng Khám Phá (Exploration):** Khi tìm kiếm ban đầu không trả về kết quả chính xác, hệ thống sử dụng Bản đồ Tự Tổ Chức (**Self-Organizing Maps - SOM**) để phân cụm các ứng viên theo các trường thông tin khác nhau (màu sắc, bối cảnh, đối tượng) [16]. Người dùng chọn cụm thị giác phù hợp nhất để tái định hướng không gian tìm kiếm.
* **Cơ chế Tối Ưu Khai Phá (Exploitation):** Khi người dùng phát hiện một khung hình có nét tương đồng với mục tiêu nhưng chưa hoàn toàn chính xác, tính năng **"Tìm ảnh tương tự" (Re-query)** được kích hoạt. Hệ thống lấy vector thị giác của khung hình đó làm điểm gốc để thực hiện tìm kiếm Image-to-Image kết hợp tinh chỉnh trọng số, đưa các phân cảnh tiệm cận lên Top đầu.

Đối với dạng truy vấn hội thoại (**Conversational KIS**), khi mô tả ban đầu quá mơ hồ, Tác tử AI sẽ chủ động đưa ra **câu hỏi gỡ rối (Clarification Question)** [1]. Ví dụ, khi người dùng nhập câu lệnh "Tìm video tôi đi gặp bạn ở quán cà phê", Trợ lý ảo phân tích không gian vector thấy số lượng ứng viên vượt quá ngưỡng và sẽ hỏi ngược lại: *"Bối cảnh quán cà phê ở trong nhà hay ngoài trời, và người bạn đó mặc áo màu gì?"* [16]. Việc tiếp nhận câu trả lời bổ sung giúp bộ lọc lập tức thu hẹp không gian tìm kiếm về danh sách chính xác [1].

---

## Lộ Trình Triển Khai Cho Đội Thi Mới Bắt Đầu Về Hệ Thống

Dành cho các đội thi mới tiếp cận lĩnh vực Multimedia Retrieval và AI Agent, việc xây dựng hệ thống cần tuân theo lộ trình 4 giai đoạn tiến cấp nhằm đảm bảo tối ưu hóa tài nguyên và tích lũy hiệu quả qua từng vòng thi.

### Các Giai Đoạn Phát Triển Hệ Thống Chuyên Sâu

* **Giai đoạn 1 (Xây dựng Baseline cơ bản):** Đội thi tập trung cài đặt Vector Database (Faiss hoặc Milvus) và nạp toàn bộ dữ liệu CLIP features (`.npy`) cùng tập Keyframes do Ban tổ chức cung cấp [1]. Giai đoạn này hoàn thành khi xây dựng xong một giao diện tìm kiếm Chữ - Ảnh (Text-to-Image Search) cơ bản cho phép nhập văn bản và trả về danh sách khung hình tương đồng nhất.
* **Giai đoạn 2 (Tích hợp Metadata & Phân tích Đa phương thức):** Hệ thống được mở rộng bằng cách lập chỉ mục tập dữ liệu Objects (`.json`) và Metadata vào Elasticsearch [2]. Đội thi tích hợp thêm mô hình Qwen2.5-VL để trích xuất OCR tiếng Việt và mô hình Whisper để bóc tách ASR từ tập video. Thuật toán dung hợp RRF được triển khai để kết hợp điểm số giữa luồng tìm kiếm văn bản và luồng tìm kiếm vector.
* **Giai đoạn 3 (Nâng cấp Tác tử AI & Logic Thời gian):** Hệ thống tích hợp LLM (Gemini 2.0 Flash hoặc OpenAI o1-mini) làm bộ não điều phối Planner để tự động phân tích và mở rộng truy vấn [13]. Thuật toán Căn chỉnh chuỗi thời gian đa chặng được lập trình để xử lý chuyên sâu bài toán TRAKE, đồng thời nạp thêm đặc trưng từ SigLIP 2 nhằm gia tăng độ chính xác tìm kiếm chi tiết tinh vi.
* **Giai đoạn 4 (Tối ưu Giao diện & Chiến thuật Thực chiến):** Giao diện hiển thị được chuyển đổi hoàn toàn sang dạng Video-Grouped Grid tích hợp phím tắt nộp bài tốc độ cao. Đội thi tiến hành diễn tập các kịch bản thi đấu thực tế để tối ưu hóa quy trình tương tác giữa người và máy [1].

### Chiến Thuật Thực Chiến Tối Ưu Điểm Số Trong Cuộc Thi

Thứ nhất, việc tối ưu hóa danh sách nộp bài theo công thức tính điểm $Final\ Score$ là yếu tố then chốt. Do điểm số là trung bình cộng của R@1, R@5, R@20, R@50, R@100, khi hệ thống không chắc chắn 100% về vị trí Top-1, việc nộp đầy đủ danh sách 100 ứng viên tiềm năng nhất xếp theo thứ tự điểm số giảm dần sẽ giúp đảm bảo các chỉ số từ R@20 đến R@100 đạt điểm tuyệt đối (1.0), từ đó kéo điểm Final Score tăng cao vượt trội so với việc chỉ nộp duy nhất một kết quả [2].

Thứ hai, việc áp dụng chiến thuật **lọc phân tầng (Cascading Filtering)** là bắt buộc để tránh tình trạng quá tải hệ thống. Hệ thống luôn sử dụng bộ lọc thô Late-Fusion (CLIP/SigLIP trên Milvus kết hợp BM25 trên Elasticsearch) để thu hẹp hàng triệu khung hình xuống Top-50 ứng viên trong thời gian dưới 50ms, sau đó mới gọi các mô hình Large Vision Language Model đắt đỏ để thực hiện suy luận và xác minh cuối cùng.

---

## Kết Luận

Xây dựng hệ thống trợ lý ảo thông minh cho Hội thi AIC 2026 đòi hỏi một giải pháp tổng thể kết hợp hài hòa giữa kiến trúc xử lý dữ liệu lớn, các mô hình học sâu tiên tiến và thiết kế giao diện tương tác tối ưu. Việc làm chủ hạ tầng lưu trữ kép (Milvus và Elasticsearch), tích hợp mô hình biểu diễn thế hệ mới (SigLIP 2, Qwen2.5-VL), triển khai tác tử AI lập luận System 2 và áp dụng thuật toán căn chỉnh chuỗi thời gian sẽ tạo nên nền tảng kỹ thuật vững chắc. Kết hợp với chiến thuật nộp bài thông minh và giao diện hiển thị nhóm theo video, hệ thống sẽ đạt được hiệu suất tối đa, đáp ứng trọn vẹn các yêu cầu khắt khe của hội thi AIC 2026 cũng như các chuẩn mực quốc tế LSC và VBS [2].

---

## Nguồn Trích Dẫn

1. Tập huấn AIC 2026 - Buổi 1.pptx.pdf
2. Thong tin vong So tuyen AIC2026.pdf
3. lifeXplore at the Lifelog Search Challenge 2024 | Request PDF - ResearchGate, https://www.researchgate.net/publication/381542652_lifeXplore_at_the_Lifelog_Search_Challenge_2024
4. Introduction to the Seventh Annual Lifelog Search Challenge, LSC'24 - ResearchGate, https://www.researchgate.net/publication/381273006_Introduction_to_the_Seventh_Annual_Lifelog_Search_Challenge_LSC24
5. Tập huấn AIC 2026 - Buổi 2.pdf
6. Vortex: Multi-Modal Fusion System for Intelligent Video Retrieval - arXiv, https://arxiv.org/html/2606.19682v1
7. Paper Review: SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features | Andrey Lukyanenko, https://andlukyane.com/blog/paper-review-siglip2
8. SigLIP 2: A better multilingual vision language encoder - Hugging Face, https://huggingface.co/blog/siglip2
9. (PDF) SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features - ResearchGate, https://www.researchgate.net/publication/389207500_SigLIP_2_Multilingual_Vision-Language_Encoders_with_Improved_Semantic_Understanding_Localization_and_Dense_Features
10. Closing the Modality Gap for Mixed Modality Search - OpenReview, https://openreview.net/forum?id=tJE6rcoMPL
11. Daily Papers - Hugging Face, https://huggingface.co/papers?q=egocentric%20wearable%20camera
12. Cathal Gurrin Doctor of Philosophy Lecturer at Dublin City University - ResearchGate, https://www.researchgate.net/profile/Cathal-Gurrin
13. Tập huấn AIC 2026 - Buổi 3.pdf
14. CLIP Statistics 2026: Image Recognition Benchmarks And Multimodal AI Data - Quantumrun, https://www.quantumrun.com/consulting/clip-statistics/
15. SigLIP2 for FiftyOne, https://docs.voxel51.com/plugins/plugins_ecosystem/siglip2.html
16. Vortex: Multi-Modal Fusion System for Intelligent Video Retrieval (pdf), https://paperity.org/p/373793414/vortex-multi-modal-fusion-system-for-intelligent-video-retrieval
