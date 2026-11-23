<div class="section-tag">04 • ĐÁNH GIÁ &amp; TRỰC QUAN HÓA (20%)</div>

# Diễn biến Huấn luyện (Loss &amp; Accuracy Curves)

<div class="my-2">
  <FigureSlot
    expectedFile="public/figures/training_curves.png"
    caption="Hình 3: Đường cong Train / Validation Loss và Accuracy qua các epochs"
    hint="Đặt subplot Loss và Accuracy của các mô hình cạnh nhau (font rõ nét, 300 DPI) để thấy tốc độ hội tụ và khoảng cách Train-Val."
    maxHeight="285px"
  />
</div>

<div class="card py-2">
  <ul class="text-sm mb-0">
    <li><strong>Tốc độ hội tụ:</strong> <span class="todo-pill">[TODO: Mô hình nào hội tụ nhanh và ổn định hơn qua các epoch đầu]</span></li>
    <li><strong>Khoảng cách Train – Validation:</strong> <span class="todo-pill">[TODO: Quan sát dấu hiệu overfitting/underfitting ở từng mô hình]</span></li>
    <li><strong>Hiệu quả Regularization / Early Stopping:</strong> <span class="todo-pill">[TODO: Điểm dừng tối ưu ở epoch thứ mấy]</span></li>
  </ul>
</div>

<!--
[3:45 – 4:30 | ~45s]
- Quan sát đường cong huấn luyện ở Hình 3, ta có thể thấy quá trình hội tụ của hai mô hình qua từng epoch.
- [TODO: Nêu rõ mô hình nào giảm loss nhanh hơn, và khoảng cách giữa đường Train và đường Validation ở giai đoạn cuối ra sao].
- Nhờ cơ chế Early Stopping [và Dropout/Augmentation], mô hình giữ được trọng số tại điểm Validation tốt nhất mà không bị học vẹt quá mức.
-->

---

<div class="section-tag">04 • ĐÁNH GIÁ &amp; TRỰC QUAN HÓA (20%)</div>

# Kết quả Đánh giá &amp; Ma trận Nhầm lẫn

<div class="grid grid-cols-12 gap-4 items-start mt-1">
  <div class="col-span-6 space-y-3">
  <div class="card">
    <h2>Bảng Chỉ số trên Tập Test</h2>

| Mô hình | Acc | Prec (M) | Rec (M) | F1 (M) | F1 (W) |
|---|---|---|---|---|---|
| `[TODO: Model 1]` | `--` | `--` | `--` | `--` | `--` |
| `[TODO: Model 2]` | `--` | `--` | `--` | `--` | `--` |

<p class="text-[11px] text-slate-500 mt-1 mb-0">Chú thích: <strong>(M)</strong> = Macro average, <strong>(W)</strong> = Weighted average.</p>
  </div>

  <MetricCard
    label="Mô hình đạt kết quả tốt nhất"
    value="[TODO: Tên Model] — F1 = [TODO]"
    subtext="[TODO: Chênh lệch +X.X% Macro F1 so với Baseline]"
    accent="emerald"
  />
  </div>

  <div class="col-span-6">
    <FigureSlot
      expectedFile="public/figures/confusion_matrices.png"
      caption="Hình 4: Ma trận nhầm lẫn trên tập Test"
      hint="Hiển thị Confusion Matrix của Mô hình 1 và Mô hình 2 cạnh nhau để đối chiếu trực tiếp."
      maxHeight="270px"
    />
  </div>
</div>

<!--
[4:30 – 5:25 | ~55s]
- Trên tập kiểm thử độc lập, [TODO: Mô hình tốt nhất] đạt Accuracy [TODO] và Macro F1-score [TODO], vượt trội [TODO]% so với mô hình baseline.
- Ma trận nhầm lẫn ở Hình 4 cho thấy đường chéo chính đậm rõ rệt hơn ở mô hình tốt nhất; tuy nhiên vẫn còn sự nhầm lẫn cục bộ giữa lớp [TODO] và lớp [TODO], nhóm sẽ phân tích kỹ nguyên nhân ở slide tiếp theo.
-->
