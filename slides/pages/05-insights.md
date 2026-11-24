<div class="section-tag">05 • SO SÁNH &amp; PHÂN TÍCH SÂU (20%)</div>

# So sánh Mô hình &amp; Phân tích Phần dư

<div class="grid grid-cols-12 gap-4 items-start mt-1">
  <div class="col-span-6 space-y-2">
    <div class="card py-1.5">
      <h2>Hướng 1 (Linear SGD) vs. Hướng 2 (MLP Adam)</h2>
      <ul class="text-xs mb-0">
        <li><strong>Linear (SGD):</strong> Minh bạch theo công thức gradient MSE (346 tham số), nhưng <em>high bias</em> do chỉ ghép tuyến tính (Test RMSE <code>1.2936 tỷ</code>).</li>
        <li><strong>MLP (Adam):</strong> Một tầng ẩn <code>128 ReLU + Dropout(0.1)</code> giảm <strong>0.0636 tỷ RMSE</strong> (<code>1.2300 tỷ</code>) và ổn định đa seed (<code>1.2079 ± 0.0032</code>).</li>
      </ul>
    </div>
    <div class="card py-1.5">
      <h2>Đối chiếu Tham chiếu Cây (HGB / RF)</h2>
      <ul class="text-xs mb-0">
        <li><strong>Hiệu năng:</strong> <code>HGB (1.1955)</code> và <code>RF (1.2026)</code> dẫn trước <code>MLP</code> <strong>0.0345 tỷ</strong> nhờ chia ngưỡng trực tiếp theo đơn giá đường/phường.</li>
        <li><strong>Giới hạn:</strong> Khó giải thích bằng cơ chế backpropagation &amp; tối ưu gradient chuẩn của môn học.</li>
      </ul>
    </div>
    <div class="callout py-1.5 text-xs">
      <strong>Đánh đổi (Trade-off):</strong> <code>MLP + Adam</code> train chỉ <strong>13.23s</strong> trên T4 (nhanh hơn SGD 35.4s vì dừng sớm ep 28) và giữ <strong>97.6% R²</strong> của <code>HGB</code>.
    </div>
  </div>

  <div class="col-span-6 space-y-2">
    <FigureSlot
      src="/figures/residual_analysis.png"
      expectedFile="public/figures/residual_analysis.png"
      caption="Hình 5: Phân bố Phần dư trên Test &amp; Top 8 Đặc trưng Quan trọng (MLP)"
      maxHeight="225px"
    />
    <div class="card py-2">
      <ul class="text-xs mb-0">
        <li><strong>Nguyên nhân sai số lớn:</strong> Phần dư lớn tập trung ở tin thiếu Mặt tiền/Đường vào, nhiễu định giá rao bán và nhà cao cấp <code>≥ 7 tỷ</code> thiếu mô tả nội thất chi tiết.</li>
      </ul>
    </div>
  </div>
</div>

<!--
[5:15 – 6:05 | ~50s]
- So sánh giữa hai hướng chính cho thấy bổ sung phi tuyến qua MLP 128 ReLU giảm rõ rệt sai số so với Linear SGD, thu hẹp khoảng cách với mốc cây HGB xuống chỉ còn 0.0345 tỷ.
- Phân tích phần dư và Permutation Importance ở Hình 5 khẳng định nhóm đặc trưng đơn giá theo phường/đường và độ rộng ngõ vào chi phối mạnh nhất, trong khi sai số lớn tập trung ở tin thiếu thông tin hoặc giá cao thiếu tiện ích vi mô.
-->

---

<div class="section-tag">05 • SO SÁNH &amp; PHÂN TÍCH SÂU (20%)</div>

# Tác động Tiền xử lý &amp; Thách thức Thực tế

<div class="grid grid-cols-12 gap-3 items-start mt-1">
  <div class="col-span-6 space-y-1.5">
  <div class="card py-1.5 px-2.5 compact-table">
    <h2>Tác động Feature Engineering (Val)</h2>

| Tiền xử lý | MLP RMSE | MLP R² | Linear SGD (R²) |
|---|---:|---:|---:|
| P0 (6 biến số) | 1.8541 | 0.3006 | 1.9956 (0.1897) |
| P2 (+ Impute/Tỉ lệ) | 1.3151 | 0.6481 | 1.4210 (0.5891) |
| P4_compact (103) | 1.2173 | 0.6985 | 1.2931 (0.6598) |
| **P4_geo_priors (345)** | **1.2123** | **0.7010** | **1.2830 (0.6651)** |

  </div>
    <FigureSlot
      src="/figures/preprocessing_impact.png"
      expectedFile="public/figures/preprocessing_impact.png"
      caption="Hình 6: Mức giảm Val RMSE từ P0 đến P4 trên Linear SGD, MLP &amp; HGB"
      maxHeight="168px"
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
      <strong>3. Tin đăng lặp giá nhiễu (1,708 cụm môi giới):</strong><br/>
      <strong>Giải pháp:</strong> Áp dụng <code>StratifiedGroupKFold</code> theo khóa bất động sản để không rò rỉ sang Val/Test.
    </div>
  </div>
</div>

<!--
[6:05 – 6:50 | ~45s]
- Bảng đối chứng và Hình 6 cho thấy Feature Engineering từ P0 lên P4 giúp giảm 34.6% RMSE trên MLP và 35.7% trên Linear SGD — tác động mạnh hơn hẳn việc chỉ tinh chỉnh siêu tham số.
- Ngoài ra, nhóm giải quyết triệt để ba bẫy dữ liệu thực tế gồm thiếu dữ liệu nặng, nhiễu Unicode địa chỉ và 1,631 cụm tin đăng lặp của môi giới.
-->
