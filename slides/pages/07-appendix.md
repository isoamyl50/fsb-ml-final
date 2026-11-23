<div class="section-tag">PHỤ LỤC Q&amp;A</div>

# Phụ lục A: Phân tích Sai số theo Phân khúc Giá &amp; Khu vực

<div class="card mt-2">
  <p class="text-xs text-slate-600 mb-2">
    Slide dự phòng cho phần hỏi đáp — chi tiết sai số của mô hình tốt nhất (<span class="todo-pill">[TODO: Tên mô hình]</span>) trên từng nhóm dữ liệu tập Test.
  </p>

| Phân khúc / Khu vực | Số mẫu (Test) | RMSE (tỷ) | MAE (tỷ) | MAPE (%) | Nhận xét |
|---|---|---|---|---|---|
| Giá thấp (`< 4 tỷ`) | `--` | `--` | `--` | `--` | `[TODO]` |
| Giá trung bình (`4–7 tỷ`) | `--` | `--` | `--` | `--` | `[TODO]` |
| Giá cao (`≥ 7 tỷ`) | `--` | `--` | `--` | `--` | `[TODO]` |
| TP. Hồ Chí Minh | `--` | `--` | `--` | `--` | `[TODO]` |
| Hà Nội | `--` | `--` | `--` | `--` | `[TODO]` |

</div>

<div class="callout mt-3 text-xs">
  <strong>Nhận xét nhanh:</strong> <span class="todo-pill">[TODO: Phân khúc/khu vực nào có sai số tuyệt đối vs sai số tương đối cao nhất]</span>
</div>

<!--
[Phụ lục Q&A — Không tính vào 8 phút trình bày chính]
- Dùng slide này nếu Thầy/Cô hỏi sâu về độ ổn định của mô hình ở các mức giá khác nhau hoặc giữa TP.HCM và Hà Nội.
-->

---

<div class="section-tag">PHỤ LỤC Q&amp;A</div>

# Phụ lục B: Nhật ký Thử nghiệm &amp; Tinh chỉnh Siêu tham số

<div class="card mt-2">
  <p class="text-xs text-slate-600 mb-2">
    Tổng hợp từ <code>experiments/leaderboard.md</code> — minh chứng quá trình thử nghiệm hệ thống trên tập Validation cố định.
  </p>

| Run ID | Mô hình | Biến đổi Target | Bộ Đặc trưng | Val RMSE | Val R² | Ghi chú |
|---|---|---|---|---|---|---|
| `[TODO]` | `ridge` | `raw` | `P0_numeric` | `--` | `--` | Mốc cơ sở 6 biến số |
| `[TODO]` | `ridge` | `raw` | `P4_geo_priors` | `--` | `--` | Tác động Feature Engineering |
| `[TODO]` | `hgb` | `raw` | `P4_geo_priors` | `--` | `--` | Gradient Boosting trên P4 |
| `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `--` | `--` | Cấu hình tối ưu được chọn |

</div>

<!--
[Phụ lục Q&A — Không tính vào 8 phút trình bày chính]
- Dùng slide này khi được hỏi về quá trình thử nghiệm biến đổi mục tiêu (raw vs log1p vs price_per_m2), siêu tham số hoặc lịch sử các lần chạy.
-->
