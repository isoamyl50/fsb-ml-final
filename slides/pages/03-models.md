<div class="section-tag">03 • MÔ HÌNH &amp; HUẤN LUYỆN (30%)</div>

# Lựa chọn Mô hình Hồi quy

<div class="grid grid-cols-3 gap-3 mt-1">
  <div class="card border-t-4 border-t-slate-500">
    <div class="text-xs font-bold uppercase tracking-wider text-slate-500">1 • Hướng 1: Tuyến tính</div>
    <h2 class="mt-0.5">Linear (Mini-batch SGD)</h2>
    <ul class="text-xs mt-1.5 mb-0">
      <li><strong>Kiến trúc:</strong> <code>Input(345) → Linear(1)</code> (346 tham số), tối ưu MSE bằng SGD.</li>
      <li><strong>Vai trò:</strong> Baseline minh bạch từ giải tích gradient; đo rõ đóng góp của phi tuyến.</li>
    </ul>
  </div>

  <div class="card border-t-4 border-t-emerald-600">
    <div class="text-xs font-bold uppercase tracking-wider text-emerald-700">2 • Hướng 2: Học sâu Đề xuất</div>
    <h2 class="mt-0.5">MLP (Backprop + Adam)</h2>
    <ul class="text-xs mt-1.5 mb-0">
      <li><strong>Kiến trúc:</strong> <code>Dense(128) → ReLU → Dropout(0.1) → Linear(1)</code> (44,417 tham số).</li>
      <li><strong>Lý do:</strong> Học tương tác phi tuyến <code>Diện tích × Vị trí</code> nhờ ReLU và bước cập nhật thích nghi Adam.</li>
    </ul>
  </div>

  <div class="card border-t-4 border-t-blue-700">
    <div class="text-xs font-bold uppercase tracking-wider text-blue-700">3 • Tham chiếu Bên ngoài</div>
    <h2 class="mt-0.5">HGB &amp; Random Forest</h2>
    <ul class="text-xs mt-1.5 mb-0">
      <li><strong>Vai trò:</strong> Mốc đối chứng từ họ cây quyết định trên cùng tập dữ liệu và cùng <code>P4</code>.</li>
      <li><strong>Ý nghĩa:</strong> Đánh giá khoảng cách hiệu năng giữa mạng MLP và Tree Ensembles trên dữ liệu bảng.</li>
    </ul>
  </div>
</div>

<div class="callout mt-2 py-1.5 text-xs">
  <strong>Chiến lược nghiên cứu:</strong> Trọng tâm là <strong>hai mô hình giải thích được từ nguyên lý đã học (SGD &amp; Backprop/Adam)</strong>, đối chiếu công bằng với <strong>HGB / RF</strong> trên cùng split <code>splits_v1.json</code> và bộ đặc trưng <code>P4_geo_priors</code>.
</div>

<!--
[2:15 – 3:00 | ~45s]
- Nhóm triển khai hai hướng mô hình trọng tâm bám sát kiến thức môn học: hồi quy tuyến tính dùng Mini-batch SGD làm mốc chuẩn tuyến tính, và mạng nơ-ron MLP huấn luyện bằng Backpropagation và Adam làm mô hình chính.
- Song song đó, HistGradientBoosting và Random Forest được giữ làm mốc tham chiếu bên ngoài trên cùng tập dữ liệu và đặc trưng P4 để đánh giá khách quan sức mạnh của mô hình đề xuất.
-->

---

<div class="section-tag">03 • MÔ HÌNH &amp; HUẤN LUYỆN (30%)</div>

# Thiết lập Huấn luyện &amp; Tinh chỉnh Siêu tham số

| Thành phần cấu hình | Hướng 1: `Linear (SGD)` | Hướng 2: `MLP (Adam)` | Tham chiếu: `HGB / RF` |
|---|---|---|---|
| **Hàm mất mát (Loss)** | `MSELoss` | `MSELoss` *(thử Huber, SmoothL1)* | `Squared Error` |
| **Bộ tối ưu (Optimizer)** | `Mini-batch SGD` | `Adam` *(bước thích nghi)* | `Tree Boosting / Bagging` |
| **Batch size &amp; LR** | `bs=128`, `lr=0.01` (`L02`) | `bs=256`, `lr=0.001` (`N05e`) | `HGB lr=0.04`, `max_iter=400` |
| **Regularization** | `grad_clip=5.0`, thử `wd=1e-4` | `Dropout=0.1`, `L2=1e-4` | `l2_reg=2.0`, `max_features=0.4` |
| **Early Stopping** | `Val RMSE`, `patience=20` *(ep 72)* | `Val RMSE`, `patience=15` *(ep 28)* | `15% val nội bộ` (HGB) |
| **Phần cứng / Thời gian** | `Tesla T4` — `35.41s` | `Tesla T4` — `13.23s` | `25.30s` (HGB) / `33.17s` (RF) |

<div class="callout mt-3">
  <strong>Chiến lược Tuning &amp; Kỷ luật Đánh giá:</strong>
  <ul class="text-sm mt-1 mb-0">
    <li><strong>Thử nghiệm có kiểm soát (35 lần chạy trên Colab T4):</strong> Khảo sát <code>L01–L05</code> (batch, LR, L2) cho Linear SGD; <code>N01–N06</code> (width, depth, BatchNorm, batch, LR, Dropout, Huber/SmoothL1, <code>log1p</code>) + kiểm tra <strong>3 seeds (42, 123, 2026)</strong> cho MLP.</li>
    <li><strong>Khóa tập Test độc lập:</strong> Mọi lựa chọn siêu tham số dựa 100% trên <code>Validation RMSE (4,535 mẫu)</code>; tập <code>Test (4,533 mẫu)</code> chỉ chạy đúng một lần sau khi chốt cấu hình.</li>
  </ul>
</div>

<!--
[3:00 – 3:45 | ~45s]
- Bảng trên tóm tắt cấu hình tối ưu của Linear SGD (L02) và MLP Adam (N05e) cùng hai mốc tham chiếu cây HGB và Random Forest.
- Trên Colab Tesla T4, toàn bộ 35 lượt thử nghiệm Validation tiêu tốn chỉ 0.40 compute units; cấu hình MLP tốt nhất được kiểm chứng độ ổn định qua ba seed 42, 123 và 2026 trước khi mở khóa tập Test một lần duy nhất.
-->
