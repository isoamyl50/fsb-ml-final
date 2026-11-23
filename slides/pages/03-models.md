<div class="section-tag">03 • MÔ HÌNH &amp; HUẤN LUYỆN (30%)</div>

# Lựa chọn Mô hình &amp; Kiến trúc

<div class="grid grid-cols-2 gap-4 mt-1">
  <div class="card border-t-4 border-t-slate-500">
    <div class="text-xs font-bold uppercase tracking-wider text-slate-500">Mô hình 1 — Baseline</div>
    <h2 class="mt-0.5">[TODO: Tên Mô hình 1]</h2>
    <ul class="text-xs mt-1.5 mb-0">
      <li><strong>Loại mô hình / Kiến trúc:</strong> <span class="todo-pill">[TODO: VD Custom CNN 3-block / Logistic Regression / Random Forest]</span></li>
      <li><strong>Lý do lựa chọn:</strong> <span class="todo-pill">[TODO: Làm mốc chuẩn gọn nhẹ, dễ diễn giải, đo lường ngưỡng cơ sở]</span></li>
      <li><strong>Đặc điểm:</strong> <span class="todo-pill">[TODO: Số tham số / Train from scratch hoặc đặc trưng thủ công]</span></li>
    </ul>
  </div>

  <div class="card border-t-4 border-t-blue-700">
    <div class="text-xs font-bold uppercase tracking-wider text-blue-700">Mô hình 2 — Chính / Nâng cao</div>
    <h2 class="mt-0.5">[TODO: Tên Mô hình 2]</h2>
    <ul class="text-xs mt-1.5 mb-0">
      <li><strong>Loại mô hình / Kiến trúc:</strong> <span class="todo-pill">[TODO: VD ResNet18 / EfficientNet / Transformer / XGBoost]</span></li>
      <li><strong>Lý do lựa chọn:</strong> <span class="todo-pill">[TODO: Khai thác đặc trưng sâu / Transfer Learning phù hợp dữ liệu]</span></li>
      <li><strong>Đặc điểm:</strong> <span class="todo-pill">[TODO: Pretrained weights, fine-tune layer cuối / số tham số]</span></li>
    </ul>
  </div>
</div>

<div class="callout mt-2 py-1.5 text-xs">
  <strong>Giả thuyết so sánh:</strong> <span class="todo-pill">[TODO: Giải thích ngắn vì sao việc đối chiếu hai kiến trúc này giúp làm rõ tác động của độ phức tạp mô hình / Transfer Learning trên bài toán]</span>
  <span class="text-[11px] text-slate-500 block mt-0.5">Ghi chú: Nếu nhóm thử nghiệm 3 mô hình, có thể chuyển lưới phía trên thành <code>grid-cols-3</code>.</span>
</div>

<!--
[2:15 – 3:05 | ~50s]
- Về phần mô hình, nhóm lựa chọn và đối chiếu hai hướng tiếp cận chính.
- Mô hình thứ nhất là [TODO: Mô hình 1], đóng vai trò baseline gọn nhẹ để xác lập ngưỡng hiệu năng cơ sở và kiểm tra độ khó của bài toán.
- Mô hình thứ hai là [TODO: Mô hình 2] với kiến trúc [TODO], kỳ vọng nắm bắt các đặc trưng phức tạp hơn [hoặc tận dụng Transfer Learning] để cải thiện khả năng tổng quát hóa.
-->

---

<div class="section-tag">03 • MÔ HÌNH &amp; HUẤN LUYỆN (30%)</div>

# Thiết lập Huấn luyện &amp; Tinh chỉnh Siêu tham số

| Thành phần cấu hình | Mô hình 1: <span class="todo-pill">[TODO]</span> | Mô hình 2: <span class="todo-pill">[TODO]</span> |
|---|---|---|
| **Hàm mất mát (Loss)** | `[TODO: CrossEntropyLoss]` | `[TODO: CrossEntropy / Weighted CE]` |
| **Bộ tối ưu (Optimizer)** | `[TODO: Adam / SGD]` | `[TODO: AdamW]` |
| **Learning Rate &amp; Scheduler** | `[TODO: 1e-3 / Cosine / StepLR]` | `[TODO: 3e-4 / ReduceLROnPlateau]` |
| **Batch Size &amp; Epochs** | `[TODO: 32 / 30 epochs]` | `[TODO: 32 / 25 epochs]` |
| **Regularization** | `[TODO: Dropout 0.3, Weight Decay]` | `[TODO: Dropout, Weight Decay, Early Stopping]` |
| **Phần cứng / Thời gian train** | `[TODO: GPU/CPU — X phút]` | `[TODO: GPU/CPU — Y phút]` |

<div class="callout mt-3">
  <strong>Chiến lược Hyperparameter Tuning &amp; Chọn mô hình:</strong>
  <ul class="text-sm mt-1 mb-0">
    <li><span class="todo-pill">[TODO: Các siêu tham số đã thử nghiệm: Learning rate, Dropout, Batch size, hoặc Grid/Random Search]</span></li>
    <li><strong>Tiêu chí lưu checkpoint:</strong> Chọn trọng số đạt <code>Validation Loss</code> thấp nhất hoặc <code>Validation Macro F1</code> cao nhất nhằm tránh overfitting.</li>
  </ul>
</div>

<!--
[3:05 – 3:45 | ~40s]
- Bảng trên tổng hợp toàn bộ cấu hình huấn luyện để đảm bảo phép so sánh công bằng giữa các mô hình.
- Nhóm giữ cố định cách chia dữ liệu và hàm mất mát chuẩn, đồng thời tinh chỉnh các siêu tham số quan trọng như Learning rate, Dropout và áp dụng Early Stopping dựa trên tập Validation để chọn checkpoint tốt nhất trước khi đem đánh giá trên tập Test.
-->
