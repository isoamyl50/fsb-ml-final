<div class="section-tag">04 • ĐÁNH GIÁ &amp; TRỰC QUAN HÓA (20%)</div>

# Diễn biến Huấn luyện (Loss &amp; RMSE Curves)

<div class="my-1">
  <FigureSlot
    src="/figures/training_curves.png"
    expectedFile="public/figures/training_curves.png"
    caption="Hình 3: Đường cong Train / Validation MSE Loss và Val RMSE qua các epoch (P4_geo_priors)"
    maxHeight="230px"
  />
</div>

<div class="card py-1.5">
  <ul class="text-xs mb-0 leading-snug">
    <li><strong>Tốc độ hội tụ:</strong> <code>MLP + Adam (lr=1e-3)</code> đạt đáy <code>Val RMSE = 1.2042 tỷ</code> chỉ sau <strong>28 epochs</strong>, nhanh hơn hẳn <code>Linear + SGD (L02)</code> cần tới <strong>72 epochs</strong> (<code>1.2816 tỷ</code>).</li>
    <li><strong>Khoảng cách Train – Validation:</strong> <code>Linear SGD</code> giữ gap nhỏ (<code>Train 1.61 vs Val 1.65</code> — <em>high bias</em>); ở <code>MLP</code> sau epoch 28, <code>Train MSE</code> tiếp tục giảm (<code>→ 1.185</code>) trong khi <code>Val MSE</code> chững quanh <code>1.45</code>.</li>
    <li><strong>Vai trò Early Stopping:</strong> Tự động khôi phục trọng số tại epoch tối ưu (<strong>ep 28</strong> với <code>MLP</code>, <strong>ep 72</strong> với <code>Linear SGD</code>), chặn kịp thời hiện tượng học vẹt.</li>
  </ul>
</div>

<!--
[3:45 – 4:25 | ~40s]
- Quan sát Hình 3, Adam giúp mạng MLP hội tụ rất nhanh, đạt Val RMSE thấp nhất 1.2042 tỷ ở epoch 28, trong khi Linear SGD cần 72 epoch để đạt 1.2816 tỷ.
- Đồng thời đường Train MSE và Val MSE của MLP bắt đầu tách nhau sau epoch 28, cho thấy Early Stopping đóng vai trò quyết định trong việc giữ điểm dừng tổng quát hóa tốt nhất.
-->

---

<div class="section-tag">04 • ĐÁNH GIÁ &amp; TRỰC QUAN HÓA (20%)</div>

# Kết quả Đánh giá &amp; Biểu đồ Dự đoán

<div class="grid grid-cols-12 gap-3 items-start mt-1">
  <div class="col-span-6 space-y-2.5">
  <div class="card py-2 px-2.5 compact-table">
    <h2>Bảng Chỉ số trên Tập Test (4,533 mẫu)</h2>

| Mô hình | RMSE | MAE | R² | MAPE | ±20% |
|---|---:|---:|---:|---:|---:|
| Linear (SGD) | 1.2936 | 0.9698 | 0.6566 | 19.92% | 66.9% |
| **MLP (Adam) ★** | **1.2300** | **0.8987** | **0.6896** | **17.72%** | **71.1%** |
| HGB (Ref) | 1.1955 | 0.8581 | 0.7067 | 16.89% | 72.6% |
| RF (Ref) | 1.2026 | 0.8561 | 0.7032 | 17.51% | 73.1% |

<p class="text-[11px] text-slate-500 mt-1 mb-0">Đơn vị RMSE/MAE: tỷ VND. Trong ±1 tỷ: MLP <strong>66.8%</strong> vs Linear SGD <strong>62.7%</strong> (HGB 68.8%).</p>
  </div>

  <MetricCard
    label="Mô hình chính tốt nhất (Gradient-based)"
    value="MLP (Adam) — RMSE = 1.2300 tỷ"
    subtext="R² = 0.6896 | MAE = 0.8987 tỷ | Đa seed Val RMSE: 1.2079 ± 0.0032"
    accent="emerald"
  />
  </div>

  <div class="col-span-6">
    <FigureSlot
      src="/figures/pred_vs_actual.png"
      expectedFile="public/figures/pred_vs_actual.png"
      caption="Hình 4: Thực tế vs. Dự đoán trên Tập Test (Linear SGD vs. MLP Adam)"
      maxHeight="265px"
    />
  </div>
</div>

<!--
[4:25 – 5:15 | ~50s]
- Trên tập Test độc lập 4,533 căn nhà, MLP + Adam đạt RMSE 1.2300 tỷ VND, MAE 0.8987 tỷ và R² 0.6896, giảm 0.0636 tỷ so với Linear SGD và giúp 71.1% căn nhà được dự đoán trong sai số ±20%.
- Hình 4 cho thấy MLP bám sát đường chéo lý tưởng hơn ở vùng 2–8 tỷ, trong khi mốc tham chiếu cây HGB và Random Forest đạt khoảng 1.196–1.203 tỷ.
-->
