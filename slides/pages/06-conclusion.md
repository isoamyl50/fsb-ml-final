<div class="section-tag">06 • TỔNG KẾT &amp; BÀI HỌC</div>

# Liên hệ Kiến thức Môn học &amp; Kết luận

<div class="grid grid-cols-2 gap-4 mt-1">
  <div class="card">
    <h2>Liên hệ Kiến thức đã học</h2>
    <ul class="text-xs mb-0">
      <li><strong>Bias – Variance Trade-off:</strong> Đối chiếu mô hình tuyến tính (high bias) với Random Forest / GBDT / MLP và vai trò điều chuẩn (L2, subsampling cột).</li>
      <li><strong>Feature Engineering &amp; Domain Priors:</strong> <span class="todo-pill">[TODO: Quy luật Giá ≈ Diện tích × Đơn giá khu vực giúp giảm mạnh sai số hồi quy]</span></li>
      <li><strong>Đánh giá đa chiều:</strong> Kết hợp <code>RMSE</code> (nhạy ngoại lai), <code>MAE</code>/<code>MedAE</code> (sai số điển hình) và <code>R²</code>.</li>
      <li><strong>Kỷ luật Pipeline:</strong> Group-Stratified Split + Out-of-Fold encoding loại bỏ hoàn toàn data leakage.</li>
    </ul>
  </div>

  <div class="space-y-3">
    <div class="callout callout-emerald">
      <h2 class="text-emerald-800 mt-0">Kết luận Chính</h2>
      <ul class="text-xs mb-0">
        <li><strong>Mô hình đề xuất:</strong> <span class="todo-pill">[TODO: Tên mô hình tối ưu]</span> đạt <span class="todo-pill">[TODO: Test RMSE / MAE / R²]</span>.</li>
        <li><span class="todo-pill">[TODO: Nhận định đúc kết quan trọng nhất về dữ liệu &amp; chất lượng dự đoán]</span></li>
      </ul>
    </div>

  <div class="card">
    <h2>Hướng Phát triển Tiếp theo</h2>
    <ul class="text-xs mb-0">
      <li>Tích hợp tọa độ địa lý (Geocoding lat/lon) hoặc mô hình ngôn ngữ tiếng Việt (PhoBERT) trích xuất sâu tiện ích xung quanh từ <code>Address</code>.</li>
      <li>Dự đoán khoảng tin cậy giá (Quantile Regression / Conformal Prediction) thay vì chỉ một giá trị điểm.</li>
    </ul>
  </div>
  </div>
</div>

<!--
[7:00 – 7:45 | ~45s]
- Qua đồ án này, nhóm thấm nhuần tầm quan trọng của kỷ luật chống rò rỉ dữ liệu, đặc biệt khi tạo Target Encoding và chia tập trên dữ liệu có tin đăng lặp.
- Mô hình [TODO] đạt hiệu năng tốt nhất và sẵn sàng mở rộng thêm thông tin tọa độ bản đồ hoặc dự đoán khoảng tin cậy giá trong tương lai.
-->

---
layout: end
hideFooter: true
---

<div class="text-center space-y-4">
  <div class="section-tag">KẾT THÚC BÀI TRÌNH BÀY</div>
  <h1 class="text-4xl font-bold text-slate-900">Cảm ơn Thầy/Cô và các bạn đã lắng nghe!</h1>
  <p class="text-lg text-slate-600">
    Q&amp;A — Rất mong nhận được câu hỏi và góp ý từ Thầy/Cô và các bạn
  </p>
  <div class="inline-block rounded-lg border border-slate-200 bg-white/90 px-5 py-3 text-sm text-slate-600 mt-4">
    <div><strong>Tài liệu kỹ thuật &amp; Mã nguồn:</strong> Jupyter Notebook (<code>.ipynb</code>) &amp; thư mục <code>src/</code> đi kèm</div>
    <div><strong>Bộ dữ liệu:</strong> Vietnam Housing Dataset 2024 (Kaggle — 30,229 mẫu)</div>
  </div>
</div>

<!--
[7:45 – 8:00 | ~15s]
- Phần trình bày của nhóm đến đây là kết thúc. Xin cảm ơn Thầy/Cô và các bạn đã lắng nghe. Nhóm rất mong nhận được câu hỏi và góp ý!
-->
