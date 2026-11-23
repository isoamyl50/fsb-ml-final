<div class="section-tag">05 • SO SÁNH &amp; PHÂN TÍCH SÂU (20%)</div>

# So sánh Mô hình &amp; Phân tích Phần dư

<div class="grid grid-cols-12 gap-4 items-start mt-1">
  <div class="col-span-6 space-y-2">
    <div class="card py-1.5">
      <h2>Nhóm Tuyến tính &amp; Mạng Nơ-ron</h2>
      <ul class="text-xs mb-0">
        <li><strong>Ưu điểm:</strong> <span class="todo-pill">[TODO: Ridge cực nhanh, dễ diễn giải; MLP học biểu diễn liên tục mượt]</span></li>
        <li><strong>Hạn chế:</strong> <span class="todo-pill">[TODO: Tuyến tính khó nắm tương tác phi tuyến Diện tích × Vị trí]</span></li>
      </ul>
    </div>
    <div class="card py-1.5">
      <h2>Nhóm Cây (RF / GBDT)</h2>
      <ul class="text-xs mb-0">
        <li><strong>Ưu điểm:</strong> <span class="todo-pill">[TODO: RMSE thấp nhất, chống nhiễu tốt, xử lý tự nhiên giá trị NaN]</span></li>
        <li><strong>Hạn chế:</strong> <span class="todo-pill">[TODO: Không ngoại suy vượt biên nếu gặp tổ hợp giá cực hiếm]</span></li>
      </ul>
    </div>
    <div class="callout py-1.5 text-xs">
      <strong>Đánh đổi (Trade-off):</strong> <span class="todo-pill">[TODO: Độ chính xác (RMSE/R²) vs. tốc độ suy luận và độ phức tạp triển khai]</span>
    </div>
  </div>

  <div class="col-span-6 space-y-2">
    <FigureSlot
      expectedFile="public/figures/residual_analysis.png"
      caption="Hình 5: Phân bố Phần dư (y_pred - y_true) &amp; Feature Importance"
      hint="Biểu đồ phần dư theo mức giá/khu vực và top đặc trưng quan trọng nhất."
      maxHeight="225px"
    />
    <div class="card py-2">
      <ul class="text-xs mb-0">
        <li><strong>Nguyên nhân sai số lớn:</strong> <span class="todo-pill">[TODO: Giá trị thiếu mặt tiền/ngõ, biến động giá rao bán giữa môi giới, đặc trưng vị trí vi mô chưa có trong tin]</span></li>
      </ul>
    </div>
  </div>
</div>

<!--
[5:25 – 6:15 | ~50s]
- So sánh giữa các mô hình cho thấy nhóm mô hình cây (RF/GBDT) xử lý vượt trội các tương tác phi tuyến giữa diện tích và khu vực.
- Khi phân tích phần dư ở Hình 5, sai số tập trung chủ yếu ở các tin thiếu thông tin mặt tiền/đường vào hoặc các căn có mức giá rao bán chủ quan chênh lệch lớn so với mặt bằng khu vực.
-->

---

<div class="section-tag">05 • SO SÁNH &amp; PHÂN TÍCH SÂU (20%)</div>

# Tác động Tiền xử lý &amp; Thách thức Thực tế

<div class="grid grid-cols-12 gap-4 items-start mt-1">
  <div class="col-span-6 space-y-2">
  <div class="card py-2">
    <h2>Tác động của Feature Engineering</h2>

| Giai đoạn tiền xử lý | Val RMSE | Val R² | Nhận xét |
|---|---|---|---|
| `P0_numeric` (6 cột số) | `--` | `--` | Chỉ biết quy mô, thiếu vị trí |
| `P4_geo_priors` (+ Địa chỉ &amp; Priors) | `--` | `--` | `[TODO]` |

  </div>
    <FigureSlot
      expectedFile="public/figures/preprocessing_impact.png"
      caption="Hình 6: Mức giảm RMSE và tăng R² từ P0 đến P4"
      hint="Biểu đồ cột đối chiếu RMSE / R² qua các bước P0 → P2 → P3 → P4."
      maxHeight="195px"
    />
  </div>

  <div class="col-span-6 card space-y-2.5">
    <h2>Thách thức gặp phải &amp; Cách khắc phục</h2>
    <div class="callout callout-amber text-xs">
      <strong>1. Dữ liệu khuyết 40–83% ở các cột phụ:</strong><br/>
      <strong>Giải pháp:</strong> Tránh <code>dropna()</code>; điền khuyết trung vị theo quận (chỉ fit trên Train) kèm biến cờ báo thiếu.
    </div>
    <div class="callout callout-amber text-xs">
      <strong>2. Địa chỉ tiếng Việt không đồng nhất &amp; lỗi Unicode:</strong><br/>
      <strong>Giải pháp:</strong> Chuẩn hóa Unicode NFC, chuẩn hóa tên tỉnh thành và kết hợp OOF Target Encoding + TF-IDF Ridge.
    </div>
    <div class="callout callout-amber text-xs">
      <strong>3. Tin đăng lặp giá nhiễu (1,631 nhóm môi giới):</strong><br/>
      <strong>Giải pháp:</strong> Áp dụng <code>StratifiedGroupKFold</code> theo khóa bất động sản để không rò rỉ sang Val/Test.
    </div>
  </div>
</div>

<!--
[6:15 – 7:00 | ~45s]
- Bảng đối chứng cho thấy phần Feature Engineering từ chuỗi địa chỉ và đơn giá khu vực giúp giảm mạnh RMSE và nâng R² vượt bậc so với việc chỉ dùng 6 biến số ban đầu.
- Bên cạnh đó, nhóm xử lý triệt để ba thách thức thực tế của dữ liệu web-scraped là dữ liệu thiếu nặng, lỗi Unicode địa chỉ và tin đăng lặp gây rò rỉ dữ liệu.
-->
