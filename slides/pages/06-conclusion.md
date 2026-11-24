<div class="section-tag">06 • TỔNG KẾT &amp; BÀI HỌC</div>

# Liên hệ Kiến thức Môn học &amp; Kết luận

<div class="grid grid-cols-2 gap-4 mt-1">
  <div class="card">
    <h2>Liên hệ Kiến thức đã học</h2>
    <ul class="text-xs mb-0">
      <li><strong>Bias – Variance Trade-off:</strong> <code>Linear SGD</code> bị <em>high bias</em> (<code>1.2816</code>); <code>128 ReLU</code> giảm bias (<code>1.2042</code>), còn <code>2 tầng (128,64)+BN</code> sớm overfit (ep 9–10).</li>
      <li><strong>Feature Engineering &amp; Domain Priors:</strong> Quy luật <code>Giá ≈ Diện tích × Đơn giá khu vực</code> (<code>P0 → P4</code>) giúp giảm <strong>34.6–35.7% RMSE</strong>, áp đảo tác động của tuning.</li>
      <li><strong>Tối ưu Gradient:</strong> <code>Adam</code> hội tụ nhanh hơn <code>SGD</code> (28 vs 72 epochs); <code>Dropout 0.1 + L2 1e-4 + Early Stopping</code> giữ ổn định qua 3 seeds.</li>
      <li><strong>Kỷ luật Pipeline:</strong> <code>StratifiedGroupKFold</code> trên <code>1,708</code> cụm lặp + 5-fold OOF encoding + fit scaler riêng trên Train loại bỏ hoàn toàn <em>data leakage</em>.</li>
    </ul>
  </div>

  <div class="space-y-3">
    <div class="callout callout-emerald">
      <h2 class="text-emerald-800 mt-0">Kết luận Chính</h2>
      <ul class="text-xs mb-0">
        <li><strong>Mô hình đề xuất (hướng Deep Learning):</strong> <strong>MLP (<code>128 ReLU, Adam, Dropout=0.1</code>, <code>P4</code>)</strong> đạt Test <strong>RMSE = 1.2300 tỷ</strong>, <strong>MAE = 0.8987 tỷ</strong>, <strong>R² = 0.6896</strong> (đa seed Val: <code>1.2079 ± 0.0032</code>).</li>
        <li>Dù <code>HGB</code> tham chiếu đạt <code>1.1955 tỷ</code>, MLP giữ ưu thế dễ diễn giải từ Backpropagation &amp; Adam và đạt độ chính xác thực tiễn <strong>71.1% trong ±20%</strong>.</li>
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
[6:50 – 7:40 | ~50s]
- Tổng kết lại, đồ án kiểm chứng rõ bốn trụ cột kiến thức môn học: cân bằng Bias-Variance, sức mạnh áp đảo của Feature Engineering, sự khác biệt giữa Adam và SGD cùng Early Stopping, và kỷ luật chống rò rỉ dữ liệu.
- Mô hình MLP + Adam đề xuất đạt Test RMSE 1.2300 tỷ và R² 0.6896, sẵn sàng mở rộng bằng dữ liệu bản đồ và khoảng tin cậy giá.
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
    <div><strong>Tài liệu kỹ thuật &amp; Mã nguồn:</strong> Notebook <code>vietnam_housing_regression.ipynb</code> &amp; thư mục <code>src/</code></div>
    <div><strong>Bộ dữ liệu:</strong> Vietnam Housing Dataset 2024 (Kaggle — 30,229 mẫu)</div>
  </div>
</div>

<!--
[7:40 – 8:00 | ~20s]
- Phần trình bày của nhóm đến đây là kết thúc. Xin cảm ơn Thầy/Cô và các bạn đã lắng nghe. Nhóm rất mong nhận được câu hỏi và góp ý!
-->
