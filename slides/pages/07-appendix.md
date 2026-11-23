<div class="section-tag">PHỤ LỤC Q&amp;A</div>

# Phụ lục A: Classification Report Chi tiết theo từng Lớp

<div class="card mt-2">
  <p class="text-xs text-slate-600 mb-2">
    Slide dự phòng cho phần hỏi đáp — ghi chi tiết hiệu năng từng lớp của mô hình tốt nhất (<span class="todo-pill">[TODO: Tên mô hình]</span>) trên tập Test.
  </p>

| Lớp (Class) | Precision | Recall | F1-Score | Support | Ghi chú |
|---|---|---|---|---|---|
| `[TODO: Class 0]` | `--` | `--` | `--` | `--` | `--` |
| `[TODO: Class 1]` | `--` | `--` | `--` | `--` | `--` |
| `[TODO: Class 2]` | `--` | `--` | `--` | `--` | `--` |
| **Macro Avg** | `--` | `--` | `--` | `--` | Trung bình đều theo lớp |
| **Weighted Avg** | `--` | `--` | `--` | `--` | Trung bình có trọng số |

</div>

<div class="callout mt-3 text-xs">
  <strong>Nhận xét nhanh:</strong> <span class="todo-pill">[TODO: Lớp nào đạt F1 cao nhất, lớp nào thấp nhất và nguyên nhân]</span>
</div>

<!--
[Phụ lục Q&A — Không tính vào 8 phút trình bày chính]
- Dùng slide này nếu Thầy/Cô hỏi sâu về hiệu năng từng lớp riêng lẻ hoặc số lượng mẫu kiểm thử (Support) của mỗi lớp.
-->

---

<div class="section-tag">PHỤ LỤC Q&amp;A</div>

# Phụ lục B: Nhật ký Thử nghiệm &amp; Tinh chỉnh Siêu tham số

<div class="card mt-2">
  <p class="text-xs text-slate-600 mb-2">
    Theo dõi các lần chạy thực nghiệm (Experiment Log) nhằm chứng minh quá trình lựa chọn cấu hình tối ưu trên tập Validation.
  </p>

| Run ID | Mô hình | Tiền xử lý / Aug | Siêu tham số chính | Val F1 | Test F1 | Ghi chú |
|---|---|---|---|---|---|---|
| `exp-01` | `[TODO]` | `Baseline` | `lr=1e-3, bs=32` | `--` | `--` | Mốc cơ sở |
| `exp-02` | `[TODO]` | `+ Norm / Aug` | `lr=1e-3, bs=32` | `--` | `--` | Kiểm chứng tiền xử lý |
| `exp-03` | `[TODO]` | `+ Norm / Aug` | `lr=3e-4, dropout=0.3` | `--` | `--` | Cải thiện tổng quát hóa |
| `exp-04` | `[TODO]` | `+ Class Weight` | `lr=3e-4, AdamW` | `--` | `--` | Cấu hình tốt nhất |

</div>

<!--
[Phụ lục Q&A — Không tính vào 8 phút trình bày chính]
- Dùng slide này khi được hỏi về quá trình thử nghiệm hyperparameters, cách nhóm chọn learning rate / regularization hoặc kết quả các lần chạy trung gian.
-->
