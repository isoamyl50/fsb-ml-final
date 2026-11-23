<div class="section-tag">03 • MÔ HÌNH &amp; HUẤN LUYỆN (30%)</div>

# Lựa chọn Mô hình Hồi quy

<div class="grid grid-cols-3 gap-3 mt-1">
  <div class="card border-t-4 border-t-slate-500">
    <div class="text-xs font-bold uppercase tracking-wider text-slate-500">1 • Tuyến tính (Baseline)</div>
    <h2 class="mt-0.5">[TODO: Ridge / Lasso]</h2>
    <ul class="text-xs mt-1.5 mb-0">
      <li><strong>Vai trò:</strong> Mốc chuẩn tuyến tính, tốc độ cực nhanh.</li>
      <li><strong>Mục đích:</strong> Định lượng rõ bước nhảy hiệu năng khi thêm đặc trưng địa lý &amp; văn bản.</li>
    </ul>
  </div>

  <div class="card border-t-4 border-t-blue-700">
    <div class="text-xs font-bold uppercase tracking-wider text-blue-700">2 • Bagging &amp; Boosting Trees</div>
    <h2 class="mt-0.5">[TODO: HGB / LightGBM / XGBoost / RF]</h2>
    <ul class="text-xs mt-1.5 mb-0">
      <li><strong>Vai trò:</strong> Mô hình chủ lực cho dữ liệu bảng.</li>
      <li><strong>Lý do:</strong> Nắm bắt tương tác phi tuyến <code>Area × District</code>, xử lý tốt giá trị NaN và cột OHE.</li>
    </ul>
  </div>

  <div class="card border-t-4 border-t-emerald-600">
    <div class="text-xs font-bold uppercase tracking-wider text-emerald-700">3 • Deep Learning</div>
    <h2 class="mt-0.5">[TODO: PyTorch Tabular MLP]</h2>
    <ul class="text-xs mt-1.5 mb-0">
      <li><strong>Kiến trúc:</strong> Linear + BatchNorm + SiLU + Dropout.</li>
      <li><strong>Lý do:</strong> Áp dụng học sâu với <code>SmoothL1Loss</code>, <code>AdamW</code> và theo dõi hội tụ qua từng epoch.</li>
    </ul>
  </div>
</div>

<div class="callout mt-2 py-1.5 text-xs">
  <strong>Giả thuyết thực nghiệm:</strong> Đối chiếu mô hình tuyến tính, cây quyết định (Bagging vs. Gradient Boosting) và mạng nơ-ron đa tầng giúp đánh giá cân bằng giữa <strong>độ chính xác (RMSE / R²)</strong>, <strong>khả năng chống nhiễu</strong> và <strong>chi phí huấn luyện</strong>.
</div>

<!--
[2:15 – 3:05 | ~50s]
- Nhóm so sánh ba nhóm mô hình hồi quy đại diện: hồi quy tuyến tính Ridge làm baseline, các mô hình cây (Random Forest, HistGradientBoosting/LightGBM/XGBoost) cho dữ liệu bảng, và mạng PyTorch Tabular MLP đại diện cho Deep Learning.
- Cách chọn này vừa đáp ứng yêu cầu đa dạng mô hình vừa cho thấy rõ mô hình nào tận dụng tốt nhất đặc trưng địa lý và tương tác phi tuyến.
-->

---

<div class="section-tag">03 • MÔ HÌNH &amp; HUẤN LUYỆN (30%)</div>

# Thiết lập Huấn luyện &amp; Tinh chỉnh Siêu tham số

| Thành phần cấu hình | Mô hình 1: <span class="todo-pill">[TODO]</span> | Mô hình 2: <span class="todo-pill">[TODO]</span> | Mô hình 3: <span class="todo-pill">[TODO]</span> |
|---|---|---|---|
| **Hàm mất mát (Loss)** | `L2 (Ridge)` | `[TODO: Squared / Huber]` | `[TODO: SmoothL1 / MSE]` |
| **Bộ tối ưu (Optimizer)** | `Closed-form` | `Gradient Boosting / Bagging` | `AdamW + CosineLR` |
| **Learning Rate / Alpha** | `[TODO: alpha=10.0]` | `[TODO: lr=0.03–0.04]` | `[TODO: lr=1e-3]` |
| **Regularization** | `L2 penalty` | `[TODO: max_features, l2_reg, min_leaf]` | `BatchNorm, Dropout, WeightDecay` |
| **Early Stopping** | `N/A` | `[TODO: Val RMSE / 15% val]` | `Val RMSE (patience=10)` |
| **Phần cứng / Thời gian** | `[TODO]` | `[TODO]` | `[TODO]` |

<div class="callout mt-3">
  <strong>Chiến lược Tuning &amp; Kỷ luật Đánh giá:</strong>
  <ul class="text-sm mt-1 mb-0">
    <li><span class="todo-pill">[TODO: Các siêu tham số chính đã thử nghiệm: colsample/max_features, độ sâu cây, learning rate, target transform]</span></li>
    <li><strong>Giữ tập Test độc lập:</strong> Mọi quyết định chọn đặc trưng và siêu tham số dựa hoàn toàn trên <code>Validation RMSE / R²</code>; tập <code>Test (4,533 mẫu)</code> chỉ đánh giá một lần sau cùng.</li>
  </ul>
</div>

<!--
[3:05 – 3:45 | ~40s]
- Bảng trên tóm tắt cấu hình huấn luyện của các mô hình chính.
- Toàn bộ quá trình thử nghiệm sử dụng chung một tập Validation cố định; tập Test gồm 4,533 mẫu chỉ được mở khóa để đo kết quả cuối cùng sau khi chốt mô hình và siêu tham số.
-->
