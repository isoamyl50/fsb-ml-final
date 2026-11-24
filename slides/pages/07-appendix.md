<div class="section-tag">PHỤ LỤC Q&amp;A</div>

# Phụ lục A: Phân tích Sai số theo Phân khúc Giá &amp; Khu vực

<div class="card mt-2">
  <p class="text-xs text-slate-600 mb-2">
    Chi tiết sai số của mô hình đề xuất tốt nhất (<strong>MLP + Adam, N05e, 128 ReLU</strong>) trên từng nhóm dữ liệu tập Test (<code>n = 4,533</code>).
  </p>

| Phân khúc / Khu vực | Số mẫu (Test) | RMSE (tỷ) | MAE (tỷ) | MAPE (%) | Nhận xét chính |
|---|---:|---:|---:|---:|---|
| Giá thấp (`< 4 tỷ`) | `1,019` | `1.1715` | `0.7952` | `30.67%` | Mẫu số giá nhỏ + thiếu Mặt tiền làm MAPE cao |
| Giá trung bình (`4–7 tỷ`) | `2,035` | **`1.0234`** | **`0.7894`** | `14.68%` | Ổn định nhất; <strong>74.0%</strong> căn trong ±20% |
| Giá cao (`≥ 7 tỷ`) | `1,479` | `1.5000` | `1.1205` | **`12.99%`** | MAPE thấp nhất nhưng biến thiên tuyệt đối cao |
| TP. Hồ Chí Minh | `1,759` | **`1.1790`** | `0.8675` | `15.85%` | R² đạt <code>0.6812</code>; <strong>75.7%</strong> dự đoán trong ±20% |
| Hà Nội | `1,583` | `1.2171` | `0.9124` | `16.30%` | R² <code>0.6279</code> do phân hóa sâu theo ngõ/quận |
| Tỉnh/TP khác | `1,191` | `1.3175` | `0.9268` | `22.38%` | MLP vượt trội Linear SGD (<code>1.4530 tỷ</code>) |

</div>

<div class="callout mt-3 text-xs">
  <strong>Nhận xét nhanh:</strong> Phân khúc trung bình (<code>4–7 tỷ</code>) đạt sai số tuyệt đối thấp nhất (<code>1.02 tỷ RMSE</code>); trong khi phân khúc <code>&lt; 4 tỷ</code> có <code>MAPE</code> cao nhất (<code>30.67%</code>) do mẫu số nhỏ và thiếu đặc trưng mặt tiền/ngõ, còn nhà <code>≥ 7 tỷ</code> có sai số tuyệt đối lớn nhất (<code>1.50 tỷ</code>) vì thiếu tiện ích nội thất cao cấp.
</div>

<!--
[Phụ lục Q&A — Không tính vào 8 phút trình bày chính]
- Dùng slide này nếu Thầy/Cô hỏi sâu về độ ổn định của mô hình ở các mức giá khác nhau hoặc giữa TP.HCM, Hà Nội và các tỉnh.
-->

---

<div class="section-tag">PHỤ LỤC Q&amp;A</div>

# Phụ lục B: Nhật ký Thử nghiệm &amp; Tinh chỉnh Siêu tham số

<div class="card mt-1.5 py-2 compact-table">
  <p class="text-xs text-slate-600 mb-1.5">
    Tổng hợp từ <code>experiments/leaderboard.md</code> (<strong>41 lần chạy</strong>) trên cùng tập Validation cố định (<code>4,535 mẫu</code>).
  </p>

| Run | Mô hình &amp; Cấu hình chính | Target | Đặc trưng | Val RMSE | Val R² | Ghi chú |
|---|---|---|---|---:|---:|---|
| P0 | linear_sgd (bs=256, lr=0.01) | raw | P0_numeric | 1.9956 | 0.1897 | Mốc tuyến tính 6 biến số |
| L01 | linear_sgd (bs=256, lr=0.01) | raw | P4_geo_priors | 1.2830 | 0.6651 | Feature Eng. giảm **35.7%** |
| **L02 ★** | **linear_sgd (bs=128, lr=0.01)** | raw | P4_geo_priors | **1.2816** | **0.6658** | Linear tốt nhất (Test 1.2936) |
| N01 | mlp ((64,), Adam lr=1e-3) | raw | P4_geo_priors | 1.2123 | 0.7010 | MLP baseline 1 tầng ẩn |
| N02d | mlp ((128,64) + BatchNorm) | raw | P4_geo_priors | 1.2266 | 0.6939 | 2 tầng + BN overfit sớm (ep 10) |
| N06 | mlp ((128,), Drop 0.1, L2) | log1p | P4_geo_priors | 1.2287 | 0.6928 | Đối chứng target log1p |
| **N05e ★** | **mlp ((128,), Drop 0.1, L2)** | raw | P4_geo_priors | **1.2042** | **0.7050** | MLP tốt nhất (Test 1.2300) |
| HGB Ref | hgb (max_leaf=63, lr=0.04) | raw | P4_geo_priors | 1.1747 | 0.7192 | Cây ngoài (Test 1.1955) |

</div>

<!--
[Phụ lục Q&A — Không tính vào 8 phút trình bày chính]
- Dùng slide này khi được hỏi về quá trình thử nghiệm từng bước L01–L05, N01–N06, so sánh target raw vs log1p và độ ổn định đa seed.
-->
