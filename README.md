# AI & Machine Learning Final Course Project

Predict the asking price of a Vietnamese home from its location, area, room counts, and other listing details. Both models predict `Price` as a continuous value in **billion VND**.

- **Instructor** — PhD. Truc Thi Kim, Nguyen
- **Dataset** — [Vietnam Housing Dataset 2024](https://www.kaggle.com/datasets/nguyentiennhan/vietnam-housing-dataset-2024)

## Models

Both approaches use TensorFlow/Keras on CPU with six compute threads, MSE loss, batch size 256, and seed 42. Validation RMSE determines which epoch's weights to save.

| Setting | Linear SGD | MLP + Adam |
| --- | --- | --- |
| Architecture | 423 inputs → one linear output | 423 inputs → 128 ReLU units → 10% dropout → one linear output |
| Optimizer | Mini-batch SGD | Adam |
| Learning rate | `0.05` | `0.001` |
| Maximum epochs | 200 | 150 |
| Early stopping patience | 20 epochs | 15 epochs |

Each approach has its own scripts and local files.

## Data pipeline

The dataset contains 30,229 listings. Both approaches use the same split and feature columns so the results can be compared directly.

| Split | Listings | Purpose |
| --- | ---: | --- |
| Train | 21,160 | Fit preprocessing and learn model weights |
| Validation | 4,535 | Compare configurations and select the best epoch |
| Test | 4,534 | Evaluate the selected model |

`data_split.py` groups matching listings on their input fields and allocates whole groups to Train, Validation, and Test within each price band. `data_normalize.py` fits preprocessing on Train, then applies it to Train and Validation. It fills missing numeric values with training medians, adds missingness indicators, standardizes numeric fields, and one-hot encodes categorical fields and the province and district extracted from the address. The normalized CSVs contain **423 input columns plus `Price`**.

Both `training_approaches/linear_sgd/` and `training_approaches/mlp_adam/` follow this layout.

```text
├── data_split.py
├── data_normalize.py
├── train.py
├── evaluate_test.py
├── dataset/
│   ├── raw/vietnam_housing_dataset.csv
│   ├── split/                 # train.csv, validation.csv, test.csv
│   └── normalized/            # train.csv, validation.csv
└── output_model/              # Saved .keras checkpoint
```

## Run locally on Windows

From the repository root, create a Python 3.11 environment and install the pinned TensorFlow dependency. These commands use `uv`.

```powershell
uv venv --python 3.11 .venv-tf
uv pip install --python .\.venv-tf\Scripts\python.exe -r .\requirements.txt
```

Download and extract the CSV from Kaggle. Copy the **same original CSV** into both raw-data folders. Replace the example source path with the location of the downloaded file.

```powershell
$datasetCsv = "C:\path\to\vietnam_housing_dataset.csv"
$linearRaw = ".\training_approaches\linear_sgd\dataset\raw"
$mlpRaw = ".\training_approaches\mlp_adam\dataset\raw"
New-Item -ItemType Directory -Force -Path $linearRaw, $mlpRaw | Out-Null
Copy-Item -LiteralPath $datasetCsv -Destination "$linearRaw\vietnam_housing_dataset.csv"
Copy-Item -LiteralPath $datasetCsv -Destination "$mlpRaw\vietnam_housing_dataset.csv"
```

Choose an approach and run the pipeline from the repository root. Set `$approach` to `linear_sgd` for the linear model or `mlp_adam` for the neural network.

```powershell
$approach = "linear_sgd"
$python = ".\.venv-tf\Scripts\python.exe"
$folder = ".\training_approaches\$approach"

& $python "$folder\data_split.py"
& $python "$folder\data_normalize.py"
New-Item -ItemType Directory -Force -Path "$folder\output_model" | Out-Null
& $python "$folder\train.py"
```

Once the configuration is final, score the saved model on Test. The evaluator rebuilds preprocessing from the raw Train split and applies it to Test before predicting.

```powershell
& $python "$folder\evaluate_test.py"
```

The dataset and checkpoint folders are gitignored. If you already have matching split and normalized files, you can start with `train.py`.

## Results

| Approach | Validation RMSE (billion VND) | Validation MAE (billion VND) | Validation R² | Within ±10% on Validation | Test RMSE (billion VND) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Linear SGD | 1.6378 | 1.2852 | 0.4519 | 27.78% | 1.6249 |
| Tuned MLP + Adam | **1.3072** | **0.9910** | **0.6508** | **37.29%** | Pending |

The tuned MLP reduced Validation RMSE by **20.2%** compared with Linear SGD on the same split. On the held-out Test set, Linear SGD achieved **1.2608 billion VND MAE**, **0.4597 R²**, and **29.16%** of predictions within ±10% of the listed price.
