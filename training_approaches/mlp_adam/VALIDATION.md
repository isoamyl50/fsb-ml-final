# Kiểm chứng Cách 2 — 24/11/2026

Đã triển khai và chạy MLP + Adam bằng TensorFlow CPU. Toàn bộ kết quả dưới đây
thuộc luồng TensorFlow độc lập, cùng dữ liệu với `linear_sgd` TensorFlow;
không so trực tiếp với leaderboard frozen split/P4 trong `experiments/`.
Chưa đánh giá Test thật và chưa chốt mô hình cuối để mở Test.

## Dữ liệu và môi trường

- Đọc trực tiếp `training_approaches/linear_sgd/dataset/normalized/`.
- Train: **21.160 dòng**; Validation: **4.535 dòng**; **423 đặc trưng**.
- CSV có 424 cột, `Price` ở cuối và được tách khỏi X. Không fit thêm scaler.
- `data_split.py`, `data_normalize.py` của MLP giống nguyên byte với Cách 1.
- Không sửa source Cách 1, `src/`, `experiments/` hay `slides/`.
- Python 3.13.5, TensorFlow CPU 2.21.0, Keras 3.15.1, NumPy 2.5.5.
- CPU float32, 6 intra-op threads, 1 inter-op thread, deterministic ops.

SHA256 đã lưu trong JSON của cả 11 run và đối chiếu với CSV tham chiếu:

```text
train.csv      5260c29e883810f6b28cb5098cfc06d6d6e9e7f5e6bec2ee6fb2db5a8a716619
validation.csv 998a4a8665eef53c5299ba554fa221568896e9f4841ad04b7b0850dbb927ca36
```

## Chuỗi cấu hình

Mọi RMSE/MAE đều tính theo tỷ VND, kể cả N06 học target `log1p`.

| Run | Thay đổi so với N01 | Best epoch | Val RMSE | Val MAE | Val R² |
|---|---|---:|---:|---:|---:|
| N01 | 64 neuron, batch 256, LR 0.001, raw, seed 42 | 60 | 1.322415 | 0.990518 | 0.642660 |
| N02a | 32 neuron | 79 | 1.335586 | 1.009149 | 0.635506 |
| N02b | 128 neuron | 39 | **1.315661** | 0.988156 | 0.646300 |
| N03a | Batch 128 | 46 | 1.322927 | 0.994517 | 0.642383 |
| N03b | Batch 512 | 118 | 1.324909 | 0.995668 | 0.641310 |
| N04 | LR 0.0003 | 145 | 1.325990 | 0.996883 | 0.640725 |
| N05 | Dropout 0.1 | 106 | 1.325190 | 0.999592 | 0.641158 |
| N06 | Target log1p | 33 | 1.366214 | 1.015852 | 0.618597 |

N05 được chạy vì sau best epoch của N01, Train RMSE tiếp tục giảm từ 1.217285
xuống 1.203283, trong khi Validation RMSE tăng từ 1.322415 lên 1.325161.
Dropout và log1p chưa cải thiện tiêu chí chọn cấu hình là Val RMSE.
N04 kết thúc đủ 150 epoch; các cấu hình khác dừng bằng early stopping.

## Độ ổn định của N02b

Giữ nguyên 128 neuron, batch 256, LR 0.001, dropout 0, MSE và raw target.
Seed split vẫn là 42; chỉ đổi seed model/shuffle huấn luyện.

| Run | Seed model | Best epoch | Val RMSE | Val MAE | Val R² |
|---|---:|---:|---:|---:|---:|
| N_best_s42 | 42 | 39 | 1.315661 | 0.988156 | 0.646300 |
| N_best_s123 | 123 | 50 | 1.316808 | 0.990381 | 0.645684 |
| N_best_s2026 | 2026 | 46 | 1.312376 | 0.988501 | 0.648064 |

Mean ± độ lệch chuẩn mẫu (ddof=1): RMSE **1.314948 ± 0.002300** tỷ VND;
MAE 0.989013 ± 0.001197; R² 0.646683 ± 0.001236;
tỷ lệ trong ±10% là 0.380522 ± 0.002792.
Seed 42 chạy lại khớp chính xác Val RMSE của N02b ban đầu.

## Bằng chứng thực thi

- `python -m unittest discover -s tests -v`: **21 test đạt** trong môi trường
  `.venv-tf`, gồm 11 test cũ và 10 test mới; không skip TensorFlow.
- Môi trường Python gốc chưa cài TensorFlow vẫn chạy được test cũ và data
  contract; test TensorFlow được skip thay vì buộc cài dependency chung.
- `verify_mlp.py --run-id N01` và `--run-id N_best_s2026` chạy ở các tiến trình
  Python mới: sai số tuyệt đối RMSE sau reload đều **0.0**, dưới ngưỡng 1e-5.
- Cả 11 run có JSON, CSV lịch sử, best `.keras`, hậu kiểm reload đạt và
  `test_metrics: null`. Không có reservation mở Test thật.
- Test dữ liệu giả kiểm tra việc chỉ fit normalizer trên Train, đánh giá Test
  cuối tối đa một lần, early stopping, lưu lịch sử, chặn ghi đè, giữ checkpoint
  hữu hạn khi phát hiện Inf và trả mã lỗi khác 0 khi phân kỳ.
- `git diff --check` sạch; dataset, `.keras`, môi trường và results được ignore.

Các artifact đầy đủ nằm trong `results/runs/`, `results/curves/`,
`results/leaderboard.md` và `output_model/` tại workspace, được gitignore.
File `.keras` nhận feature đã normalize; chưa đóng gói pipeline nhận tin đăng thô.

Chạy lại cấu hình tốt nhất bằng ID mới:

```powershell
.\.venv-tf\Scripts\python.exe training_approaches/mlp_adam/train.py --config training_approaches/mlp_adam/configs/N02b_h128.json --run-id N02b_repeat_01
.\.venv-tf\Scripts\python.exe training_approaches/mlp_adam/verify_mlp.py --run-id N02b_repeat_01
```
