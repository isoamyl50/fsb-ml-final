# AI & Machine Learning Final Course Project

This project predicts the **asking price of a Vietnamese home in billion VND** from its location, area, room counts, and other listing details. It compares three TensorFlow/Keras approaches on the same Train, Validation, and Test splits.

- **Instructor** — PhD. Truc Thi Kim, Nguyen
- **Dataset** — [Vietnam Housing Dataset 2024](https://www.kaggle.com/datasets/nguyentiennhan/vietnam-housing-dataset-2024)
- **Group 2 presentation** — [Project slides](https://docs.google.com/presentation/d/1iCMLaqbKC5mSDq8bFqxL-l_Jhoyulr0a3uTqznyQnLU/edit)

## 1. Dataset and preprocessing

The source CSV has **30,229 listings and 12 columns**.

| Role | Raw columns |
| --- | --- |
| Numeric inputs | `Area`, `Frontage`, `Access Road`, `Floors`, `Bedrooms`, `Bathrooms` |
| Categorical inputs | `House direction`, `Balcony direction`, `Legal status`, `Furniture state` |
| Location input | `Address`, parsed into province and district |
| Regression target | `Price`, in billion VND |

### Split first

`data_split.py` groups likely reposts by matching all 11 input fields after text cleanup. It ranks the groups by median price, divides them into 10 price bands, and shuffles within each band with seed 42. It assigns each whole group to a partition to approximate the 70/15/15 row allocation in each band.

| Split | Listings | Purpose |
| --- | ---: | --- |
| Train | 21,160 | Fit preprocessing and train the model |
| Validation | 4,535 | Compare configurations and select the best epoch |
| Test | 4,534 | Evaluate the final saved model |

### Fit preprocessing on Train

`data_normalize.py` fills missing numeric values with the Train median and adds a binary missingness flag for each numeric input. It uses the mean and population standard deviation of the filled Train values to standardize the numeric features. It one-hot encodes the categorical and location values using the vocabulary learned from Train. Each category group has a column for missing values. An unseen category leaves its group of columns at zero.

| Feature group | Columns |
| --- | ---: |
| Standardized numeric values + missingness flags | 12 |
| House direction | 9 |
| Balcony direction | 9 |
| Legal status | 3 |
| Furniture state | 3 |
| Province | 71 |
| District | 316 |
| **Total model inputs** | **423** |

The normalized Train and Validation CSVs have **424 columns** because they also store `Price` as the final column. Training reads that column as the target in its original units. At Test time each evaluator fits the same preprocessing from the Train split, checks the feature order against the normalized Train CSV, and transforms Test in memory.

## 2. Three modeling approaches

All three configurations use **TensorFlow 2.21.0**, **CPU with six compute threads**, `float32`, mini-batches of **256**, and seed **42**. The training scripts monitor **Validation RMSE** after each epoch and save the model with the weights from the best epoch.

| Final setting | Linear SGD | MLP one layer + Adam | MLP three layers + Adam |
| --- | --- | --- | --- |
| Architecture | 423 inputs → 1 linear output | 423 → 128 ReLU → 1 linear output | 423 → 256 ReLU → 128 ReLU → 64 ReLU → 1 linear output |
| Trainable parameters | 424 | 54,401 | 149,761 |
| Optimizer | Mini-batch SGD | Adam | Adam |
| Initial learning rate | 0.05 | 0.001 | 0.001 |
| Learning-rate schedule | Fixed | Fixed | Halve after 6 epochs without Validation RMSE improvement, floor 0.00001 |
| Training loss | MSE | MSE | Huber (δ = 1.0 billion VND) |
| Regularization | Early stopping | Early stopping and 10% dropout after the hidden layer | Early stopping, dropout of 18% / 10% / 5% after the hidden layers, and L2 of 0.002 on hidden-layer weights |
| Maximum epochs | 200 | 150 | 150 |
| Early stopping patience | 20 | 15 | 20 |
| Best epoch in the reported run | 200 | 70 | 59 |

### Approach 1 · Linear SGD

[Linear SGD implementation](training_approaches/linear_sgd) predicts price as a weighted sum of the 423 inputs plus a bias. MSE penalizes larger errors more heavily and SGD updates the weights after each batch. Increasing the learning rate from **0.01** to **0.05** gave the best result among the tested settings under the 200-epoch budget.

### Approach 2 · MLP with one hidden layer and Adam

The [one-layer MLP implementation](training_approaches/mlp_adam) adds a ReLU hidden layer so that the model can learn nonlinear relationships, such as the effect of area varying by location. The baseline has **64 hidden units**. Fine-tuning increased this to **128 units** and added **10% dropout** to help control overfitting. Adam and MSE were used in both configurations.

### Approach 3 · MLP with three hidden layers and Adam

The [three-layer MLP implementation](training_approaches/deep_mlp_adam) stacks **256, 128, and 64 ReLU units** to combine features over multiple layers. The three-layer baseline already used the same dropout, L2, and learning-rate schedule as the final configuration. Fine-tuning switched the loss from **MSE to Huber**. Huber penalizes small errors quadratically and increases the penalty linearly for errors beyond 1 billion VND.

## 3. Results

### Baseline to fine-tuned configuration on Validation

| Approach | Baseline RMSE | Baseline MAE | Fine-tuned RMSE | Fine-tuned MAE | RMSE reduction | Fine-tuned R² | Fine-tuned within ±10% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Linear SGD | 1.6507 | 1.3042 | 1.6378 | 1.2852 | 0.78% | 0.4519 | 27.78% |
| MLP one layer + Adam | 1.3303 | 1.0054 | 1.3072 | 0.9910 | 1.73% | 0.6508 | 37.29% |
| MLP three layers + Adam | 1.3024 | 0.9730 | 1.2862 | 0.9563 | 1.24% | 0.6620 | 39.49% |

### Final configurations on held-out Test

| Approach | RMSE | MAE | R² | Within ±10% | RMSE reduction vs previous approach |
| --- | ---: | ---: | ---: | ---: | ---: |
| Linear SGD | 1.6249 | 1.2608 | 0.4597 | 29.16% | Reference |
| MLP one layer + Adam | 1.3400 | 1.0080 | 0.6326 | 37.69% | 17.5% |
| MLP three layers + Adam | **1.3142** | **0.9741** | **0.6466** | **39.77%** | **1.93%** |

**MSE** averages squared prediction errors and has units of (billion VND)². **RMSE** is its square root, so it emphasizes larger errors and has the same unit as price. **MAE** is the average absolute error, also in billion VND. **R²** compares squared error with a constant prediction equal to the mean target. **Within ±10%** is the share of predictions whose absolute error is at most 10% of the listed price.

The largest Test RMSE reduction came from moving from Linear SGD to the one-layer MLP. The three-layer MLP brought the total reduction to about **19.1%** relative to Linear SGD. The Test comparison measures each full configuration, including architecture, optimizer, loss, and regularization where these differ.

## 4. Run locally on Windows

From the repository root, create a Python 3.11 virtual environment with `uv` and install the pinned dependency.

```powershell
uv venv --python 3.11 .venv-tf
uv pip install --python .\.venv-tf\Scripts\python.exe -r .\requirements.txt
```

Download and extract the dataset. Set `$datasetCsv` to the extracted CSV file and copy the same source file into each approach's local `dataset/raw` folder.

```powershell
$datasetCsv = "C:\path\to\vietnam_housing_dataset.csv"
$approaches = @("linear_sgd", "mlp_adam", "deep_mlp_adam")

foreach ($approach in $approaches) {
    $rawDir = ".\training_approaches\$approach\dataset\raw"
    New-Item -ItemType Directory -Force -Path $rawDir | Out-Null
    Copy-Item -LiteralPath $datasetCsv -Destination (Join-Path $rawDir "vietnam_housing_dataset.csv")
}
```

Choose `linear_sgd`, `mlp_adam`, or `deep_mlp_adam`. Run split, normalization, and training in order. The repository scripts train the final configurations listed above, and training writes the selected model for the chosen approach.

```powershell
$approach = "linear_sgd"
$python = ".\.venv-tf\Scripts\python.exe"
$folder = ".\training_approaches\$approach"

& $python "$folder\data_split.py"
& $python "$folder\data_normalize.py"
New-Item -ItemType Directory -Force -Path "$folder\output_model" | Out-Null
& $python "$folder\train.py"
```

After choosing the final configuration on Validation, evaluate its saved checkpoint on Test.

```powershell
& $python "$folder\evaluate_test.py"
```

### Files produced by each approach

Each approach has its own `data_split.py`, `data_normalize.py`, `train.py`, and `evaluate_test.py`. To reproduce the reported comparisons, start each run with the same source CSV. The processed data and model outputs stay in that approach's folder.

```text
training_approaches/<approach>/
├── dataset/
│   ├── raw/vietnam_housing_dataset.csv
│   ├── split/                    # train.csv, validation.csv, test.csv
│   └── normalized/               # train.csv, validation.csv
└── output_model/                 # <approach>_model.keras
```

The three-layer approach also writes `output_model/deep_mlp_adam_model.json` with its configuration, Validation scores, training time, and learning curves. Each approach's `dataset/` and `output_model/` folders are Git-ignored.
