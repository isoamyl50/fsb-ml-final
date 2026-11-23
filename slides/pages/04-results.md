<div class="section-tag">04 • ĐÁNH GIÁ &amp; TRỰC QUAN HÓA (20%)</div>

# Diễn biến Huấn luyện (Loss &amp; RMSE Curves)

<div class="my-2">
  <FigureSlot
    expectedFile="public/figures/training_curves.png"
    caption="Hình 3: Đường cong Train / Validation Loss và RMSE qua các epoch / vòng lặp"
    hint="Hiển thị đường cong hội tụ của PyTorch MLP và các mô hình Boosting (300 DPI)."
    maxHeight="285px"
  />
</div>

<div class="card py-2">
  <ul class="text-sm mb-0">
    <li><strong>Tốc độ hội tụ:</strong> <span class="todo-pill">[TODO: Tốc độ giảm Loss / Val RMSE qua các epoch hoặc iteration]</span></li>
    <li><strong>Khoảng cách Train – Validation:</strong> <span class="todo-pill">[TODO: Dấu hiệu overfitting và tác dụng của Regularization]</span></li>
    <li><strong>Điểm dừng Early Stopping:</strong> <span class="todo-pill">[TODO: Epoch / vòng lặp đạt Val RMSE thấp nhất]</span></li>
  </ul>
</div>

<!--
[3:45 – 4:30 | ~45s]
- Quan sát Hình 3, ta theo dõi được tốc độ hội tụ của hàm mất mát và Validation RMSE qua từng epoch hoặc vòng boosting.
- [TODO: Nêu ngắn gọn điểm hội tụ và khoảng cách giữa đường Train và Validation, khẳng định Early Stopping giúp tránh học vẹt].
-->

---

<div class="section-tag">04 • ĐÁNH GIÁ &amp; TRỰC QUAN HÓA (20%)</div>

# Kết quả Đánh giá &amp; Biểu đồ Dự đoán

<div class="grid grid-cols-12 gap-4 items-start mt-1">
  <div class="col-span-6 space-y-3">
  <div class="card">
    <h2>Bảng Chỉ số trên Tập Test (tỷ VND)</h2>

| Mô hình | RMSE | MAE | R² | MAPE | ±20% |
|---|---|---|---|---|---|
| `[TODO: Model 1]` | `--` | `--` | `--` | `--` | `--` |
| `[TODO: Model 2]` | `--` | `--` | `--` | `--` | `--` |
| `[TODO: Model 3]` | `--` | `--` | `--` | `--` | `--` |

<p class="text-[11px] text-slate-500 mt-1 mb-0">Chú thích: RMSE/MAE càng thấp càng tốt; R² và tỉ lệ dự đoán trong sai số ±20% càng cao càng tốt.</p>
  </div>

  <MetricCard
    label="Mô hình đạt kết quả tốt nhất"
    value="[TODO: Tên Model] — RMSE = [TODO] tỷ"
    subtext="[TODO: R² = X.XX | MAE = X.XX tỷ | ±20% đạt XX%]"
    accent="emerald"
  />
  </div>

  <div class="col-span-6">
    <FigureSlot
      expectedFile="public/figures/pred_vs_actual.png"
      caption="Hình 4: Biểu đồ Giá Thực tế vs. Giá Dự đoán trên tập Test"
      hint="Scatter plot y_true vs y_pred kèm đường chuẩn y=x và dải sai số ±1 tỷ / ±20%."
      maxHeight="270px"
    />
  </div>
</div>

<!--
[4:30 – 5:25 | ~55s]
- Trên tập kiểm thử độc lập, [TODO: Mô hình tốt nhất] đạt RMSE [TODO] tỷ VND, MAE [TODO] tỷ và R² [TODO], giúp [TODO]% số căn nhà dự đoán nằm trong ngưỡng sai số ±20%.
- Hình 4 cho thấy các điểm dự đoán bám sát đường chéo y=x ở phân khúc trung bình, trong khi độ phân tán tăng nhẹ ở các căn giá cao trên 8 tỷ.
-->
