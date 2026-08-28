import numpy as np
import zipfile
import json
import csv
from pathlib import Path

# Tạo cấu trúc thư mục
base = Path("data/mock_v2")
for d in ["features", "mappings", "keyframes/L21_V001", "out"]:
    (base / d).mkdir(parents=True, exist_ok=True)

# 1. Tạo feature vector (3 frames, 512-dim) với giá trị hợp lệ (khác 0)
matrix = np.random.randn(3, 512).astype(np.float16)
np.save(base / "features/L21_V001.npy", matrix)

# 2. Tạo mapping CSV
with open(base / "mappings/L21_V001.csv", "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["n", "pts_time", "fps", "frame_idx"])
    writer.writerows([[1, 0, 30, 10], [2, 0.5, 30, 25], [3, 1.0, 30, 40]])

# 3. Tạo file ảnh JPG rỗng (để bypass check tồn tại file)
for i in range(1, 4):
    (base / f"keyframes/L21_V001/{i:03d}.jpg").write_bytes(b"mock_image")

# 4. Tạo file Query Zip
query_name, query_text = "query-p1-01-kis.txt", "người phụ nữ mặc áo đỏ"
with zipfile.ZipFile(base / "query.zip", "w") as z:
    z.writestr(query_name, query_text.encode("utf-8"))

# 5. Tạo file Query Variants
with open(base / "variants.json", "w", encoding="utf-8") as f:
    json.dump({query_name: {"original": query_text, "variants": ["woman in red shirt"]}}, f)

print(f"[OK] Successfully created mock data at: {base.absolute()}")
