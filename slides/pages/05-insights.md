<div class="section-tag">05 • SO SÁNH &amp; PHÂN TÍCH SÂU (20%)</div>

# So sánh Mô hình &amp; Phân tích Lỗi

<div class="grid grid-cols-12 gap-4 items-start mt-1">
  <div class="col-span-6 space-y-2">
    <div class="card py-1.5">
      <h2>Mô hình 1: <span class="todo-pill">[TODO]</span></h2>
      <ul class="text-xs mb-0">
        <li><strong>Điểm mạnh:</strong> <span class="todo-pill">[TODO: Nhẹ, train nhanh, ít tham số, dễ triển khai]</span></li>
        <li><strong>Điểm yếu:</strong> <span class="todo-pill">[TODO: Khả năng trích xuất đặc trưng hạn chế / dễ nhầm lớp phức tạp]</span></li>
      </ul>
    </div>
    <div class="card py-1.5">
      <h2>Mô hình 2: <span class="todo-pill">[TODO]</span></h2>
      <ul class="text-xs mb-0">
        <li><strong>Điểm mạnh:</strong> <span class="todo-pill">[TODO: Độ chính xác và Macro F1 cao hơn, tổng quát hóa tốt]</span></li>
        <li><strong>Điểm yếu:</strong> <span class="todo-pill">[TODO: Chi phí tính toán lớn hơn, cần tinh chỉnh cẩn thận tránh overfit]</span></li>
      </ul>
    </div>
    <div class="callout py-1.5 text-xs">
      <strong>Đánh đổi (Trade-off):</strong> <span class="todo-pill">[TODO: Hiệu năng vs. thời gian huấn luyện / kích thước mô hình]</span>
    </div>
  </div>

  <div class="col-span-6 space-y-2">
    <FigureSlot
      expectedFile="public/figures/error_analysis.png"
      caption="Hình 5: Các cặp lớp hay nhầm lẫn / mẫu dự đoán sai"
      hint="Hiển thị các mẫu dự đoán sai tiêu biểu (ảnh kèm True/Pred) hoặc biểu đồ F1 theo từng lớp."
      maxHeight="225px"
    />
    <div class="card py-2">
      <ul class="text-xs mb-0">
        <li><strong>Nguyên nhân lỗi chính:</strong> <span class="todo-pill">[TODO: Lớp có ít mẫu / đặc trưng ngoại quan tương đồng / nhiễu nhãn]</span></li>
      </ul>
    </div>
  </div>
</div>

<!--
[5:25 – 6:15 | ~50s]
- Tổng hợp so sánh cho thấy sự đánh đổi rõ rệt giữa độ phức tạp và hiệu năng: Mô hình 1 chạy nhanh nhưng hạn chế ở các lớp khó, trong khi Mô hình 2 cải thiện mạnh chỉ số Recall và F1.
- Khi kiểm tra các ca dự đoán sai trên Hình 5, nhóm nhận thấy phần lớn lỗi tập trung vào [TODO: cặp lớp hay nhầm] do [TODO: nguyên nhân từ dữ liệu hoặc đặc trưng gần giống nhau].
-->

---

<div class="section-tag">05 • SO SÁNH &amp; PHÂN TÍCH SÂU (20%)</div>

# Tác động Tiền xử lý &amp; Thách thức Thực tế

<div class="grid grid-cols-12 gap-4 items-start mt-1">
  <div class="col-span-6 space-y-2">
  <div class="card py-2">
    <h2>Tác động của Tiền xử lý / Augmentation</h2>

| Biến thể thí nghiệm | Accuracy | Macro F1 | Nhận xét |
|---|---|---|---|
| Chưa tiền xử lý / Aug | `--` | `--` | `[TODO]` |
| Đã tiền xử lý + Aug | `--` | `--` | `[TODO]` |

  </div>
  <FigureSlot
      expectedFile="public/figures/preprocessing_impact.png"
      caption="Hình 6: So sánh F1 / Accuracy trước và sau tiền xử lý"
      hint="Biểu đồ cột thể hiện mức cải thiện trước/sau chuẩn hóa hoặc augmentation."
      maxHeight="195px"
    />
  </div>

  <div class="col-span-6 card space-y-2.5">
    <h2>Thách thức gặp phải &amp; Cách khắc phục</h2>
    <div class="callout callout-amber text-xs">
      <strong>1. Thách thức:</strong> <span class="todo-pill">[TODO: VD Mất cân bằng lớp / dữ liệu nhiễu]</span><br/>
      <strong>Giải pháp:</strong> <span class="todo-pill">[TODO: VD Dùng Class Weights / lọc nhiễu / Stratified split]</span>
    </div>
    <div class="callout callout-amber text-xs">
      <strong>2. Thách thức:</strong> <span class="todo-pill">[TODO: VD Overfitting nhanh trên tập Train nhỏ]</span><br/>
      <strong>Giải pháp:</strong> <span class="todo-pill">[TODO: VD Bổ sung Augmentation, Dropout, Early Stopping]</span>
    </div>
    <div class="callout callout-amber text-xs">
      <strong>3. Thách thức:</strong> <span class="todo-pill">[TODO: VD Giới hạn GPU/RAM khi huấn luyện]</span><br/>
      <strong>Giải pháp:</strong> <span class="todo-pill">[TODO: VD Điều chỉnh Batch Size, Resize hợp lý, Mixed Precision]</span>
    </div>
  </div>
</div>

<!--
[6:15 – 7:00 | ~45s]
- Để trả lời câu hỏi tiền xử lý tác động ra sao, nhóm thực hiện đối chứng trước và sau khi áp dụng [chuẩn hóa / tăng cường dữ liệu]. Kết quả cho thấy Macro F1 cải thiện [TODO]%, chứng tỏ chất lượng đầu vào ảnh hưởng trực tiếp tới khả năng học của mô hình.
- Trong quá trình thực hiện, nhóm cũng xử lý ba thách thức chính là [TODO], giúp mô hình ổn định hơn trên tập Test.
-->
