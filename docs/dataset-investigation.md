# Vietnam Housing Dataset 2024 — EDA Investigation & Regression Recommendations

> **Purpose:** Read this document before writing exploratory code or prompting AI agents on this dataset. Every statistic below was computed directly on `data/raw/vietnam_housing_dataset.csv` to save duplicate exploration work and tokens.

- **Source:** [House Price Prediction Dataset Vietnam - 2024 (Kaggle)](https://www.kaggle.com/datasets/nguyentiennhan/vietnam-housing-dataset-2024)
- **Local File:** `data/raw/vietnam_housing_dataset.csv` (gitignored; `3,532,685` bytes)
- **Dimensions:** `30,229` rows × `12` columns, `0` exact duplicate rows (`0.00%`)
- **Task:** **Continuous House Price Prediction (Regression)** — predict listing `Price` in **billion VND** (`[1.00, 11.50]`).

---

## 1. Exact Schema & Exploratory Statistics

| Column | Type | Missing Count (%) | Summary / Distribution |
|---|---|---:|---|
| `Address` | Text | 0 (0.00%) | Full comma-separated address (`3`–`5` parts in 98.4% of rows); `78` raw last tokens; **73.55%** in `Hồ Chí Minh` (`11,778`, 38.96%) + `Hà Nội` (`10,457`, 34.59%), followed by `Bình Dương` (`1,672`, 5.53%), `Đà Nẵng` (`1,437`, 4.75%); `357` unique raw `(District, City)` pairs |
| `Area` | Float ($\text{m}^2$) | 0 (0.00%) | `min=3.10`, `p01=21.00`, `p05=30.00`, `p25=40.00`, `median=56.00`, `p75=80.00`, `p95=148.62`, `p99=275.72`, `max=595.00`, `mean=68.499`, `std=48.070`, `skew=3.89` |
| `Frontage` | Float (m) | 11,564 (38.25%) | `18,665` non-null; `min=1.00`, `p25=4.00`, `median=4.50`, `p75=5.00`, `p95=9.00`, `p99=29.36`, `max=77.00`, `mean=5.362`, `std=4.346`, `skew=8.09` |
| `Access Road` | Float (m) | 13,297 (43.99%) | `16,932` non-null; `min=1.00`, `p25=4.00`, `median=6.00`, `p75=10.00`, `p95=20.00`, `p99=40.00`, `max=85.00`, `mean=7.854`, `std=7.451`, `skew=3.74` |
| `House direction` | Categorical | 21,239 (70.26%) | `8` cardinal directions (`Đông - Nam` 6.34%, `Đông - Bắc` 3.90%, `Tây - Nam` 3.81%, `Tây - Bắc` 3.72%, `Nam` 3.45%, `Đông` 3.15%, `Bắc` 2.75%, `Tây` 2.62%) |
| `Balcony direction` | Categorical | 24,983 (82.65%) | `8` cardinal directions (`Đông - Nam` 3.60%, `Nam` 2.32%, `Đông - Bắc` 2.22%, etc.) |
| `Floors` | Float | 3,603 (11.92%) | `26,626` non-null; `min=1`, `p25=2`, `median=3`, `p75=4`, `p95=5`, `p99=6`, `max=10`, `mean=3.410`, `std=1.329`, `skew=0.01` |
| `Bedrooms` | Float | 5,162 (17.08%) | `25,067` non-null; `min=1`, `p25=3`, `median=3`, `p75=4`, `p95=6`, `p99=8`, `max=9`, `mean=3.511`, `std=1.309`, `skew=1.05` |
| `Bathrooms` | Float | 7,074 (23.40%) | `23,155` non-null; `min=1`, `p25=2`, `median=3`, `p75=4`, `p95=6`, `p99=8`, `max=9`, `mean=3.347`, `std=1.400`, `skew=0.75` |
| `Legal status` | Categorical | 4,506 (14.91%) | `Have certificate`: `24,774` (81.95%), `Sale contract`: `949` (3.14%) |
| `Furniture state` | Categorical | 14,119 (46.71%) | `Full`: `10,591` (35.04%), `Basic`: `5,519` (18.26%) |
| `Price` *(target)* | Float (tỷ VND) | 0 (0.00%) | **Pre-bounded `[1.00, 11.50]`**: `min=1.00`, `p01=1.45`, `p05=2.20`, `p10=2.85`, `p25=4.20`, `p33.3=4.80`, `median=5.90`, `p66.7=6.90`, `p75=7.50`, `p90=8.95`, `p95=9.50`, `p99=10.00`, `max=11.50` (`mean=5.872`, `std=2.212`, `skew=-0.03`, only `1` row $\ge 11$ tỷ) |

### Derived EDA Metric (Diagnostic Only — Never Use as Input Feature)

- `Price_per_m2 = Price * 1000 / Area` (million VND/$\text{m}^2$): `mean=107.91`, `std=57.46`, `min=2.11`, `p01=13.09`, `p05=28.00`, `p25=64.33`, `median=102.27`, `p75=145.00`, `p95=205.50`, `p99=252.17`, `max=1896.77`.

---

## 2. Verified Data Quality Traps & Handling Rules

1. **Do NOT use `df.dropna()`:**
   - Only **`2,645` rows (`8.75%`)** have all 12 columns non-null; even dropping both direction columns leaves only **`7,920` rows (`26.20%`)**.
   - **Rule:** Impute numerical columns (`Train`-only median or district-grouped median), add binary missingness indicator flags (`Frontage_missing`, `AccessRoad_missing`, etc.), and encode missing categoricals as `"Missing"`.
2. **Vietnamese Unicode & Address Parsing Bugs:**
   - Raw comma-splitting of `Address` yields **78** unique last tokens instead of 63 provinces due to NFD vs. NFC (`'Hà Nội'` × 4 vs. `'Hà Nội'` × 10,457), aliases (`'TP. HCM'` × 2 vs. `'Hồ Chí Minh'` × 11,778), and `5` malformed $\le 2$-part addresses.
   - **Rule:** Always normalize with `unicodedata.normalize("NFC", s)`, map aliases canonically, and parse right-to-left defensively.
3. **Domain Outliers:**
   - `64` rows have `Frontage > Area`, `61` rows have `Area < 15 m²` (`min=3.10`), `216` rows have `Area > 300 m²` (`max=595.00`), `11` rows exceed `400 triệu/m²`, and `165` rows are `< 10 triệu/m²`.
   - **Rule:** Test Train-only `--outlier-filter` cleanly as an ablation in `experiments/`; never filter `Val` or `Test` rows so metrics remain comparable.
4. **Near-Duplicate Broker Reposts (Label Noise + Leakage Risk):**
   - `1,631` groups (`4,004` rows) share exact `(Address, Area, Floors, Bedrooms, Bathrooms)`, and `337` groups differ in `Price` by `> 0.5 tỷ VND`.
   - **Rule:** Use `experiments/splits/splits_v1.json` (`21,161` Train / `4,535` Val / `4,533` Test), built via 20-fold `StratifiedGroupKFold` on continuous price deciles + the broker-repost group key so reposts never leak across splits.

---

## 3. Target Formulations & Evaluation Metrics (`src/targets.py`, `src/evaluate.py`)

Because `Price` is pre-bounded `[1.00, 11.50]` (`skew=-0.03`), while `Area` is right-skewed (`skew=3.89`) and interacts multiplicatively with location ($\text{Price} \approx \text{Area} \times \text{Price}/\text{m}^2$), test **4 registered target transforms** (all inverted back to **billion VND** before scoring):

- **`raw`** *(default)*: Predict `Price` directly in billion VND.
- **`log1p`**: Train on `log1p(Price)`, invert with `expm1`.
- **`price_per_m2`**: Train on clipped `Price * 1000 / Area` (million VND/$\text{m}^2$), invert via `pred_ppm2 * Area / 1000`.
- **`log_price_per_m2`**: Train on `log(Price * 1000 / Area)`, invert via `exp(pred) * Area / 1000`.

**Unified Evaluation Metrics (all in billion VND):**
- Primary ranking metric: **`RMSE`** (billion VND, lower is better)
- Secondary metrics: **`MAE`**, **`MedAE`**, **`R²`**, **`MAPE (%)`**, plus practical error-tolerance accuracies **`±10%`**, **`±20%`**, **`±1 tỷ`**.

---

## 4. Feature Engineering & Leakage Guardrails (`src/features.py`)

Progressive recipes (all fit strictly on `Train`):
- **`P0_numeric`**: 6 raw numerics (`Area`, `Frontage`, `Access Road`, `Floors`, `Bedrooms`, `Bathrooms`).
- **`P1_location`**: `P0` + NFC-normalized `city`, `district`, `ward` & raw categoricals (OHE).
- **`P2_domain`**: `P1` + district-grouped median imputation, missingness flags, `is_project`/`is_alley`, `log1p` transforms (`Area`, `Frontage`, `Access Road`, `usable_area`), structural ratios (`Area/Floors`, `Total_Rooms`, `Area/Bedrooms`, `Frontage/AccessRoad`), and `n_missing_fields`.
- **`P3_tfidf_te`**: `P2` + 5-fold OOF hierarchical continuous Target Encoding (`city`, `district`, `ward`, `street` mean `Price`) + TF-IDF + `TruncatedSVD(20)` on `Address`.
- **`P4_geo_priors`**: `P3` + 5-fold OOF hierarchical `Price/m²` priors (`city → district → ward → street`) + multiplicative `Area × oof_ppm2` priors + `log1p` frequencies + 5-fold OOF TF-IDF Ridge text predictions (`addr_ridge_price`, `addr_ridge_ppm2`).
- **`P4_compact`**: Dense subset of `P4_geo_priors` dropping sparse `district_*` / `ward_*` OHE columns.

**Strict Non-Leakage Rule:** NEVER use `Price`, `Price / Area`, or any target-derived value inside `transform(rows)` or as raw input features.

---

## 5. Recommended Model Lineup & Slide Handoff

Compare diverse regression families on `Val RMSE` / `MAE` / `R²` (`src/models.py`):
1. **Linear Baselines:** `ridge`, `lasso` (clearly shows `P0 → P4` progression).
2. **Bagging & Boosting Trees:** `rf`, `et`, `hgb`, `lgbm`, `xgb`, `catboost` (plus `blend_trees`).
3. **PyTorch Tabular Neural Network (`mlp`):** Residual/BatchNorm + SiLU + Dropout MLP trained with `SmoothL1Loss` or `MSELoss`, `AdamW`, `CosineAnnealingLR`, and `EarlyStopping` (also supplies per-epoch `training_curves.png` for Slide 7).

**6 Slidev Figures (`slides/public/figures/` via `slides/scripts/slide_export.py`):**
1. `price_distribution.png` — Histogram & box/violin of `Price` (and `Price/m²`) overall and across Train/Val/Test (Slide 3)
2. `dataset_samples.png` — Price vs. Area / top cities EDA relationships (Slide 3)
3. `training_curves.png` — PyTorch MLP Train/Val Loss & RMSE per epoch (+ boosting validation curves) (Slide 7)
4. `pred_vs_actual.png` — Predicted vs. Actual `Price` scatter with $y=x$ and $\pm 1$ tỷ / $\pm 20\%$ bands (Slide 8)
5. `residual_analysis.png` — Residual distribution (`y_pred - y_true`), error by price tier / city, and top permutation importances (Slide 9)
6. `preprocessing_impact.png` — `P0 → P4` RMSE & $R^2$ progression bar chart (Slide 10)
