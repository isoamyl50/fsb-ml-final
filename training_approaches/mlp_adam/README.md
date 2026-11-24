# Cách 2 — MLP + Backpropagation + Adam (TensorFlow CPU)

Kết quả chạy và bằng chứng kiểm chứng ngày 24/11/2026: [VALIDATION.md](VALIDATION.md).

MLP có một lớp ẩn ReLU (64 neuron mặc định), HeNormal, đầu ra tuyến tính,
Adam và MSE. Dropout, Huber và target `log1p` là tùy chọn. Mọi metric và
checkpoint đều dùng giá gốc **tỷ VND**, kể cả khi loss học trên thang log.

Luồng này so sánh trực tiếp với `training_approaches/linear_sgd/` TensorFlow.
Đây là split và feature riêng, không phải frozen split/P4 của `src/`.
Kết quả chỉ nằm trong `results/`; không đưa vào `experiments/runs/`.

## Môi trường và chạy train

Chạy từ thư mục gốc repository (Python 3.10–3.13):

```powershell
python -m venv .venv-tf
.\.venv-tf\Scripts\python.exe -m pip install -r training_approaches/mlp_adam/requirements.txt
.\.venv-tf\Scripts\python.exe training_approaches/mlp_adam/train.py --config training_approaches/mlp_adam/configs/N01.json
.\.venv-tf\Scripts\python.exe training_approaches/mlp_adam/verify_mlp.py --run-id N01
```

Mặc định dùng `mlp_adam/dataset/normalized/` nếu đã có, nếu chưa có thì đọc
trực tiếp `linear_sgd/dataset/normalized/`. Có thể chỉ định `--normalized-dir`.
Trước train, SHA256 của Train và Validation phải khớp file tham chiếu trong
`linear_sgd/dataset/normalized/` (`--reference-dir` nếu dữ liệu ở vị trí khác).
Không chuẩn hóa lại trong `train.py`. `Price` ở cột cuối, được tách khỏi X;
X/y float32, y có shape `(N, 1)`, kiểm tra hữu hạn và thứ tự cột.

`data_split.py` và `data_normalize.py` được sao chép **nguyên byte** từ Cách 1.
Nếu cần tạo dữ liệu độc lập, đặt raw CSV tại
`mlp_adam/dataset/raw/vietnam_housing_dataset.csv`, chạy hai script theo thứ tự.
Seed split luôn là 42; `--seed` chỉ đổi seed huấn luyện model.
Hai script gốc không tự tạo thư mục cha `dataset/`, nên cần đặt raw CSV trước.

```powershell
foreach ($cfg in @('N01','N02a_h32','N02b_h128','N03a_bs128','N03b_bs512','N04_lr0003')) {
    .\.venv-tf\Scripts\python.exe training_approaches/mlp_adam/train.py --config "training_approaches/mlp_adam/configs/$cfg.json"
}
```

N01 baseline; N02 đổi hidden 32/128; N03 đổi batch 128/512; N04 đổi LR
0.0003. N05 bật dropout 0.1 khi quan sát overfit; N06 thử target log1p.
Mỗi config chỉ đổi một yếu tố so với N01. JSON hoặc YAML đều được hỗ trợ;
tham số CLI ghi đè config. Ví dụ `--loss huber --huber-delta 1.0`.

Kiểm chứng cấu hình tốt nhất với các seed 42, 123, 2026, mỗi lần đặt ID riêng:

```powershell
foreach ($seed in @(42,123,2026)) {
    .\.venv-tf\Scripts\python.exe training_approaches/mlp_adam/train.py --config training_approaches/mlp_adam/configs/N01.json --seed $seed --run-id "N_best_s$seed"
}
```

Thay N01 bằng config tốt nhất theo Validation. Leaderboard tự tổng hợp
mean ± sample std cho cùng cấu hình, cùng hash dữ liệu, đủ ba seed.

## Checkpoint, log và Test

CPU mặc định 6 intra-op/1 inter-op threads; có thể đổi bằng `--intra-threads`
và `--inter-threads`. Seed Python/NumPy/TensorFlow và deterministic ops được
thiết lập trước train. `PYTHONHASHSEED` được lưu/đặt khi chạy; để cố định hash
ngay lúc Python khởi động, đặt biến môi trường này trước khi gọi Python.
Số liệu thời gian bao gồm suy luận Train/Validation và lưu checkpoint;
không đối chiếu thời gian với luồng PyTorch/P4 hoặc máy khác như cùng benchmark.

- `output_model/<run_id>_best.keras`: ghi tạm rồi atomic replace khi Val RMSE
  giảm hơn 1e-6; patience mặc định 15, tối đa 150 epoch.
- `results/curves/<run_id>_history.csv`: flush/fsync từng epoch; có loss,
  RMSE/MAE/R²/±10%/±20%, gap và thời gian. Dropout tắt khi tính metric.
- `results/runs/<run_id>.json`: atomic JSON, hash/schema dữ liệu, cấu hình,
  phiên bản, metric checkpoint tốt nhất và hậu kiểm reload (sai số < 1e-5).
- `results/leaderboard.md`: xếp theo Val RMSE, ghi rõ phạm vi TensorFlow.

Run ID đã có JSON, checkpoint hoặc curve sẽ bị từ chối. Lock ngăn hai writer
cùng ID; leaderboard có lock riêng. Tiến trình bị kill có thể để lại lock:
kiểm tra PID trong file trước khi dọn thủ công. Ctrl+C giữ CSV/checkpoint,
ghi trạng thái interrupted và trả code 130. NaN/Inf dừng train, giữ checkpoint
hữu hạn trước đó, ghi status diverged và trả code 1. LR lớn nhưng vẫn hữu hạn
không bị gán là phân kỳ; test phân kỳ chủ động chèn Inf để kiểm tra chắc chắn.
Các file dữ liệu, model và results được gitignore.

Workspace đã chạy N01–N06 và ba seed kiểm chứng. Khi chạy lại tại workspace
này, thêm ID mới, ví dụ `--run-id N02b_repeat_01`; không ghi đè các run đã có.

Mặc định không đọc Test và `test_metrics` luôn null. Chỉ sau khi nhóm đã khóa
cấu hình cuối mới thêm **`--eval-test`** vào lệnh train với ID mới. Khi đó:

1. Nạp best checkpoint, fit nguyên `DataNormalizer` trên raw split Train.
2. Xác nhận Train/Validation dựng lại khớp chính xác ma trận đã dùng.
3. Đặt reservation bền vững `results/test_evaluation.json` trước khi đọc Test;
   transform Test bằng normalizer Train và đánh giá một lần.

Reservation ngăn mở Test lần nữa trong cùng thư mục kết quả, kể cả khi tiến
trình bị ngắt sau khi đã mở quyền đọc Test. Không bật Test từ config.
Đường dẫn raw split suy ra từ thư mục anh em `dataset/split/`.
File `.keras` nhận **feature đã normalize**; chưa phải pipeline nhận địa chỉ thô.

## Kiểm thử

```powershell
.\.venv-tf\Scripts\python.exe -m unittest discover -s tests -p test_mlp_adam_tf.py -v
python -m unittest discover -s tests -v
```

Test TensorFlow tự skip khi chưa cài dependency; các kiểm tra data contract
vẫn chạy. Kiểm thử phân kỳ, early stopping, checkpoint, tái lập, thang metric
log1p và tiến trình reload độc lập dùng dữ liệu tổng hợp, không mở Test thật.
