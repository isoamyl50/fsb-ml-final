# Cách 1 — Linear Regression với Mini-batch SGD

Đã triển khai và chạy đủ L01, L02, L03, L04a, L04b, L05 theo đặc tả người dùng cung cấp. Chỉ dùng Validation để chọn checkpoint; không đánh giá Test.

## Pha 0: Đối chiếu kiến trúc hiện có

- Repo dùng module phẳng `src/models.py`, `src/runner.py`, `src/evaluate.py`, không có package `src/models/registry.py`. Giữ nguyên cấu trúc này, đăng ký `linear_sgd` qua `@register_model`; code riêng nằm tại `src/approach_1_linear_sgd/`.
- `src/splits.py` giữ nguyên frozen split `splits_v1`, seed 42: 21.161 Train / 4.535 Validation / 4.533 Test. Runner mới kiểm tra lại disjoint cả chỉ số và repost group khi load split.
- `src/preprocessing.py` đã chuẩn hóa NFC, alias thành phố, tách địa chỉ từ phải sang trái. Không sửa preprocessing hoặc feature recipe chung.
- P4 thực tế là `P4_geo_priors`, trả về `numpy.ndarray`, `float32`, 345 chiều. Wrapper vẫn nhận được sparse matrix và DataFrame.
- P4 fit imputation, vocabulary, TF-IDF/SVD và các prior trên Train; target encoding, prior giá/m² và dự đoán Ridge text cho Train dùng 5-fold OOF. Validation chỉ gọi `transform`.
- `src/targets.py` đã có `raw`; runner riêng ánh xạ `raw_price` trong YAML sang cách biểu diễn này. `P4` được ánh xạ sang `P4_geo_priors`.
- `evaluate_predictions` đã có đủ RMSE, MAE, R², ±10%, ±20%. Thêm `full_precision=True` cho early stopping và kiểm chứng checkpoint; mặc định baseline vẫn làm tròn như trước.
- Baseline lưu JSON atomic, git commit/dirty và phần cứng qua runner chung. Nhánh mới tái sử dụng helper atomic/git, bổ sung thread count, dtype, số CPU, lịch sử epoch và checkpoint. Artifacts riêng nằm dưới `artifacts/approach_1_linear_sgd/`; mỗi run đồng thời xuất JSON tương thích vào `experiments/runs/` và cập nhật leaderboard chung theo `AGENTS.md`.

## Bước 1–5: Pipeline

`wrapper.py` chuyển dense float32, kiểm tra finite/shape, reshape y thành `(N, 1)`, fit StandardScaler duy nhất trên Train. Model gồm một `nn.Linear(d, 1)` có bias, không activation. Trainer dùng CPU, seed 42 cho Python/NumPy/PyTorch, 4 threads, DataLoader shuffle có generator seed 42, `num_workers=0`, MSE, SGD momentum 0.

Mỗi epoch tính metric trên Train/Validation bằng hàm evaluate chung, ghi thời gian và gap. Checkpoint chỉ cập nhật khi Val RMSE giảm hơn `1e-6`; ghi tạm rồi `os.replace`. Patience 20, tối đa 200 epoch. Khi loss hoặc dự đoán không finite, dừng với trạng thái `diverged` và giữ checkpoint tốt nhất đã có. CSV vẫn ghi epoch bị phân kỳ, để trống metric không hợp lệ.

Sau fit, nạp lại checkpoint và kiểm tra Val RMSE sai khác dưới `1e-5`. Runner kiểm tra thêm bằng `wrapper.predict(X_val)`. Nếu phân kỳ trước checkpoint đầu tiên, JSON dùng `null` cho metric, không sinh NaN/Infinity và không cho predict. CLI luôn trả exit code 1 khi trạng thái `diverged`, kể cả khi vẫn còn checkpoint tốt để sử dụng.

CLI hỗ trợ `--approach`, `--config`, `--eval-test`. Test chỉ mở khi có cờ CLI tường minh; trường `eval_test` trong YAML không tự mở Test. Model `linear_sgd` qua `run_and_log` cũng được định tuyến vào nhánh riêng. Run ID trùng bị từ chối để bảo vệ kết quả cũ; muốn chạy lại cần sao chép cấu hình và đổi `run_id`. Khóa atomic tại `experiments/.locks/<run_id>.lock` đặt chỗ trước khi đọc dataset và giữ đến khi xuất xong kết quả/leaderboard, nên hai tiến trình cùng ID không thể cùng train.

## Bước 6: Chạy thí nghiệm

Từ thư mục gốc, dùng PowerShell:

```powershell
foreach ($cfg in 'L01','L02','L03','L04a','L04b','L05') {
    python scripts/run_experiment.py --approach approach_1_linear_sgd --config "configs/approach_1_linear_sgd/$cfg.yaml"
    if ($LASTEXITCODE -ne 0) { throw "Experiment failed: $cfg" }
}
```

Entry point `python run_experiment.py ...` cũng hỗ trợ cùng tham số. Cài dependencies nền bằng `requirements.txt`, dependencies riêng qua `configs/approach_1_linear_sgd/requirements.txt` nếu chưa có PyTorch/PyYAML.

Giữ nguyên cả sáu cấu hình theo bảng đặc tả, gồm L05 so với L01. Không tạo thêm L05_best. Bỏ qua benchmark threads tùy chọn; dùng cố định 4 threads. Máy thực thi báo 12 CPU logic, PyTorch `2.13.0+cpu`; các thời gian đo ở máy này không đại diện cho máy 6 vCPU.

## Kết quả và nghiệm thu

Đầy đủ kết quả tại [leaderboard riêng](../artifacts/approach_1_linear_sgd/leaderboard.md), 6 JSON, 6 CSV và 6 checkpoint tốt nhất. Checkpoint được gitignore.

| Run | Val RMSE (tỷ) | Val MAE (tỷ) | Val R² | Trong ±10% | Best epoch |
|---|---:|---:|---:|---:|---:|
| L01 | 1.283115 | 0.963938 | 0.665029 | 39.32% | 50 |
| L02 | 1.289154 | 0.968264 | 0.661869 | 38.83% | 50 |
| L03 | 1.282574 | 0.963943 | 0.665312 | 39.29% | 50 |
| L04a | 1.284275 | 0.964851 | 0.664423 | 39.12% | 200 |
| L04b | 1.321635 | 0.994315 | 0.644615 | 37.86% | 15 |
| L05 | 1.283121 | 0.963937 | 0.665026 | 39.32% | 50 |

Cả sáu run `completed`; L04b không phân kỳ trên dữ liệu thật. L03 tốt nhất trong nhóm nhưng RMSE vẫn cao hơn HGB + P4 (1.1709 tỷ). Không mở Test hoặc thay thế baseline.

Kiểm chứng:

```powershell
python -m unittest discover -s tests -v
python scripts/verify_linear_sgd.py
python scripts/run_experiment.py --author regression_check --features P0_numeric --model ridge --no-save
python scripts/build_leaderboard.py
git diff --check
```

- 5 bài test đạt: sparse/train-only scaling, checkpoint reload và tái lập, giữ best khi phân kỳ, phân kỳ trước checkpoint đầu tiên/NaN input, baseline fit, khóa Test và chống ghi đè run (một số bài bao gồm nhiều kiểm tra).
- Audit đủ sáu run: JSON hợp lệ, CSV khớp số epoch, checkpoint và Val RMSE khớp dưới `1e-5`, scaler chỉ thấy 21.161 mẫu, Test `null` và `eval_test=false`; repost groups disjoint.
- Lệnh Ridge cũ trên dữ liệu thật không có `--approach` đạt Val RMSE 1.9960, MAE 1.6310, R² 0.1894, khớp baseline cũ. Dùng `--no-save` nên không sinh thêm run baseline.
- Sáu JSON baseline giữ nguyên. Sau khi bổ sung đồng bộ theo `AGENTS.md`, leaderboard chung có thêm các run Linear SGD. Không sửa frozen split, preprocessing, feature recipe, metric kết quả cũ hoặc slides.

## Cải thiện lưu/nạp và theo dõi huấn luyện

Các run mới tự lưu thêm `artifacts/approach_1_linear_sgd/pipelines/<run_id>_pipeline.joblib`, gồm feature recipe đã fit (P4), StandardScaler, cấu hình và trọng số checkpoint tốt nhất. JSON ghi `artifacts.pipeline_path` tương đối với thư mục `artifacts/approach_1_linear_sgd/`. File nhị phân này được gitignore. Sáu run ban đầu chỉ có checkpoint trọng số; không tự bổ sung scaler/P4 bằng cách fit lại rồi gán vào kết quả cũ.

Sau khi export, runner nạp bundle và đối chiếu tất cả dự đoán Validation với mô hình trong bộ nhớ (sai số tuyệt đối tối đa `1e-6`). Có thể nạp bundle để dự đoán từ dictionary tin đăng mà không cần dataset train hoặc file checkpoint riêng:

```python
from src.approach_1_linear_sgd.pipeline import LinearSGDPipeline

pipeline = LinearSGDPipeline.load(
    "artifacts/approach_1_linear_sgd/pipelines/L03_pipeline_v1_pipeline.joblib"
)
rows = [{
    "Address": "Phường Bến Nghé, Quận 1, Hồ Chí Minh",
    "Area": "60", "Frontage": "4", "Access Road": "5",
    "Floors": "3", "Bedrooms": "3", "Bathrooms": "3",
    "Legal status": "Have certificate", "Furniture state": "Basic",
    "House direction": "", "Balcony direction": "",
}]
prices_billion_vnd = pipeline.predict(rows)
```

`predict` loại bỏ trường `Price` nếu có trước khi gọi feature transform. Chỉ nạp bundle joblib từ nguồn tin cậy; bundle sử dụng các lớp Python/scikit-learn của repo, nên giữ cùng môi trường dependencies khi chuyển máy. Phiên bản thư viện được ghi trong bundle.

CSV mở ngay đầu quá trình train; sau mỗi epoch hoàn tất sẽ ghi dòng, flush và fsync xuống đĩa. Console in epoch, Train/Val RMSE, best epoch và thời gian. Khi Ctrl+C, các dòng epoch đã hoàn tất và checkpoint tốt nhất đã ghi vẫn còn; chưa hỗ trợ resume optimizer/DataLoader từ giữa epoch.

Để kiểm chứng bundle trên dữ liệu Validation bằng một tiến trình Python mới:

```powershell
python scripts/verify_linear_sgd.py --run-id L03_pipeline_v1
```

Tiến trình con chỉ nạp bundle, gọi transform/predict trên Validation, so sánh metric với JSON (sai số dưới `1e-5`), kiểm tra scaler sample count và trọng số khớp checkpoint. Không fit lại và không đánh giá Test. Lệnh không có `--run-id` vẫn audit sáu run gốc, đồng thời báo rõ chúng chưa có bundle để kiểm chứng inference đầy đủ.

Run `L03_pipeline_v1` dùng nguyên cấu hình L03 để kiểm chứng thay đổi lưu/nạp, không phải thử siêu tham số mới. Không ghi đè các run cũ. Bộ test bổ sung kiểm tra pipeline qua tiến trình mới khi đã xóa checkpoint rời, lịch sử khi bị KeyboardInterrupt và mã thoát CLI cho cả completed/diverged.

Đã chạy kiểm chứng: 8 bài test đạt; `L03_pipeline_v1` hoàn tất 70 epoch, best epoch 50, Val RMSE 1.2825735834302987 tỷ. Tiến trình Python mới nạp bundle tái hiện metric và trọng số khớp checkpoint. Test vẫn khóa. SHA-256 của 21 file JSON/CSV/checkpoint cũ không thay đổi; lệnh baseline Ridge tiếp tục đạt RMSE 1.9960. `git diff --check` đạt.

## Đồng bộ kết quả và chống xung đột theo AGENTS.md

Run mới ghi JSON riêng và bản tương thích schema chung tại `experiments/runs/<run_id>.json`, giữ nguyên metric, seed, timestamp, git commit, cấu hình và thông tin split. Leaderboard riêng và chung tự cập nhật. Schema chung bổ sung `task=regression`, `outlier_filter=false` và thông tin feature; không chứa trọng số hoặc raw data.

Đồng bộ run đã có mà không train/đánh giá lại:

```powershell
python scripts/sync_linear_sgd_runs.py
python scripts/build_leaderboard.py
```

Đồng bộ có thể chạy lại: nếu JSON chung đã có nội dung tương ứng thì giữ nguyên file; nếu ID trùng nhưng nội dung khác thì báo lỗi và không ghi đè. Các JSON/CSV/checkpoint riêng và sáu JSON baseline được giữ nguyên.

Khóa `run_id` dùng `os.open(..., O_CREAT | O_EXCL)`, không dựa vào kiểm tra `exists()` để phân xử giữa hai tiến trình. Khóa được giải phóng khi thoát bình thường hoặc có exception/Ctrl+C. Nếu tiến trình bị kill hoặc mất điện, khóa có thể còn lại; kiểm tra PID ghi trong file và chỉ xóa khóa khi đã xác nhận tiến trình chủ đã dừng. Không tự xóa khóa theo tuổi. Khóa được gitignore.

Việc rebuild mỗi leaderboard cũng có khóa riêng và ghi file qua đường dẫn tạm rồi replace. Các run phân kỳ trước checkpoint đầu tiên vẫn có JSON chung; leaderboard ghi chúng ở phần không xếp hạng, không tạo metric giả. CSV chung chỉ chứa các run có metric Validation hữu hạn.

Đã kiểm chứng phần đồng bộ: 10 bài test đạt, gồm tranh khóa giữa hai tiến trình, giải phóng khóa khi lỗi, ghi JSON chung, đồng bộ lặp lại không ghi đè, từ chối ID trùng khác nội dung và leaderboard khi run không có metric hợp lệ. Đã đồng bộ đủ 8 run Linear SGD; leaderboard chung có 14 run (6 baseline + 8 Linear SGD). SHA-256 của 31 file cũ trong thư mục artifacts riêng và `experiments/runs/` không đổi. Chạy đồng bộ lần hai vẫn giữ 14 run, metric/seed/split/commit/timestamp khớp nguồn, Test `null`, không để lại khóa run.

## Phạm vi chia sẻ qua Git

Commit source, tests, cấu hình YAML, tài liệu, JSON trong `experiments/runs/` và `experiments/leaderboard.md` theo quy trình chung. Toàn bộ `artifacts/approach_1_linear_sgd/` là đầu ra cục bộ (checkpoint, pipeline, CSV, JSON trùng và leaderboard riêng), được gitignore. Trọng số trong các thư mục con của `artifacts/`, file tạm atomic và khóa cũng được ignore. Không ignore JSON chung, cấu hình của thành viên khác hoặc frozen split.

PyTorch chỉ được import khi dùng nhánh Linear SGD; baseline không bắt buộc cài dependencies của nhánh này. CLI giữ hành vi đọc `eval_test` từ config của model cũ để tương thích; Linear SGD chỉ mở Test khi có `--eval-test` tường minh. Theo quy trình chung, thành viên vẫn cần giữ `eval_test=false` trong config tuning.

Clone mới không có checkpoint/pipeline/CSV riêng. Muốn kiểm chứng nạp pipeline cần train run riêng với ID mới, sau đó dùng `python scripts/verify_linear_sgd.py --run-id <run_id>`. Lệnh sync chỉ dành cho các máy đã có JSON cục bộ cần xuất vào thư mục chung; không train lại.

Rà soát trước khi chia sẻ: 11 bài test đạt, gồm kiểm tra tương thích config baseline và cờ Test của Linear SGD. Đã kiểm tra Ridge import/fit/predict khi chặn import PyTorch. Kiểm tra gitignore xác nhận 11 đường dẫn đầu ra cục bộ bị loại và 10 đường dẫn source/config/kết quả chung vẫn được giữ; các file thay đổi có thể commit không bao gồm dataset, artifacts riêng, trọng số, cache hoặc khóa. `git diff --check` đạt.
