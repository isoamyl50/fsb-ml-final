# Vietnam Housing Dataset 2024 — EDA Investigation & Modeling Recommendations

> **Purpose:** Read this document before writing exploratory code or prompting AI agents on this dataset. Every statistic and benchmark below was computed directly on `data/raw/vietnam_housing_dataset.csv` to save duplicate exploration work and tokens.

- **Source:** [House Price Prediction Dataset Vietnam - 2024 (Kaggle)](https://www.kaggle.com/datasets/nguyentiennhan/vietnam-housing-dataset-2024)
- **Local File:** `data/raw/vietnam_housing_dataset.csv` (gitignored; `3,532,685` bytes)
- **Dimensions:** `30,229` rows × `12` columns, `0` exact duplicate rows (`0.00%`)
- **Task Conversion:** Reformulate continuous `Price` (billion VND) into multi-class price tiers via **fixed domain binning** for the final course project classification task.

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
   - Only **`2,645` rows (`8.75%`)** have all 12 columns non-null.
   - Even after dropping `House direction` and `Balcony direction`, only **`7,920` rows (`26.20%`)** remain complete.
   - **Rule:** Impute numerical columns (train-only median or train-only district/city group median), add binary missingness indicator flags (`has_frontage`, `has_access_road`, etc.), and encode missing categoricals as `"Missing"` / `"Unknown"`. Evaluate dropping vs. keeping `Balcony direction` (82.65% missing) in preprocessing ablation.
2. **Vietnamese Unicode & Address Parsing Bugs:**
   - Raw comma-splitting of `Address` yields **78** unique last tokens instead of 63 provinces due to:
     - **NFD vs. NFC Unicode mismatch:** `'Hà Nội'` (`4` rows, decomposed diacritics) vs. `'Hà Nội'` (`10,457` rows). Always run `unicodedata.normalize("NFC", s)` first.
     - **City aliases:** `'TP. HCM'` (`2` rows) vs. `'Hồ Chí Minh'` (`11,778` rows). Map aliases canonically.
     - **Malformed short addresses:** `5` rows have $\le 2$ comma-separated parts (e.g., `'Đường số 11'`, `'Phòng công chứng Nguyễn Thị Thành'`). Parse right-to-left defensively and fall back to `"Unknown"`.
3. **Domain Outliers:**
   - `64` rows have `Frontage > Area` (impossible or unit-swapped dimensions).
   - `61` rows have `Area < 15 m²` (`min=3.10 m²`); `216` rows have `Area > 300 m²` (`max=595.00 m²`).
   - `11` rows exceed `400 triệu/m²` (up to `1,896.77 triệu/m²`), and `165` rows are `< 10 triệu/m²` (`min=2.11 triệu/m²`).
   - **Rule:** Test a clean outlier-filtering or Winsorization variant against raw retention in preprocessing ablation (`pages/05-insights.md`). Note: when removing target-based `Price/m²` outliers, document clearly that absurd scraping errors (`< 10 triệu/m²` or `> 400 triệu/m²`) were removed as data cleaning, and ensure no `Price`-derived values ever enter features.
4. **Near-Duplicate Broker Reposts (Label Noise + Leakage Risk):**
   - `1,631` groups (`4,004` rows) share exact `(Address, Area, Floors, Bedrooms, Bathrooms)`.
   - `337` of those groups differ in `Price` by `> 0.5 tỷ VND`.
   - **Rule:** Great talking point for Challenges / Error Analysis. Strongly consider grouping by `(Address, Area, Floors, Bedrooms, Bathrooms)` (or deduplicating) before splitting `Train`/`Val`/`Test` so reposts of the same house do not leak across splits.

---

## 3. Binning Strategy Comparison & Recommendation

All benchmarks below were measured on an **80/20 stratified split** (`random_state=42`) using two clean scikit-learn baselines:
- **Numeric only:** `Area`, `Frontage`, `Access Road`, `Floors`, `Bedrooms`, `Bathrooms` (median imputed + scaled for LogisticRegression; raw NaNs for HistGradientBoostingClassifier).
- **+Address/FE:** Above numerics + `log1p(Area)` + `Frontage`/`Access Road` missing flags + `is_project` / `is_alley` flags + top-20 `City`, top-120 `District`, `Legal status`, `Furniture state`, `House direction` (One-Hot Encoded).

| Binning Scheme | Cutoffs (tỷ VND) | Class Counts & Shares | LogReg (Num) Acc / Macro-F1 | HGB (Num) Acc / Macro-F1 | LogReg (+Addr/FE) Acc / Macro-F1 | HGB (+Addr/FE) Acc / Macro-F1 | HGB ±1 Bin Acc |
|---|---|---|---:|---:|---:|---:|---:|
| **3-class Fixed Domain *(Recommended)*** | `< 4` / `4–7` / `≥ 7` | `6,853` (22.67%) / `13,540` (44.79%) / `9,836` (32.54%) | 0.5131 / 0.4771 | 0.5834 / 0.5722 | 0.6968 / 0.7029 | **0.7345 / 0.7397** | **98.68%** |
| **3-class Quantile** | `< 4.8` / `4.8–6.9` / `≥ 6.9` | `10,008` (33.11%) / `9,855` (32.60%) / `10,366` (34.29%) | 0.5002 / 0.4843 | 0.5665 / 0.5563 | 0.6588 / 0.6551 | 0.6947 / 0.6934 | 96.64% |
| **4-class Fixed Domain** | `< 4` / `4–6` / `6–8` / `≥ 8` | `6,853` (22.67%) / `8,748` (28.94%) / `8,637` (28.57%) / `5,991` (19.82%) | 0.4072 / 0.3765 | 0.4719 / 0.4741 | 0.5769 / 0.5803 | 0.6313 / 0.6370 | 94.06% |
| **4-class Quantile** *(dist. only)* | `< 4.2` / `4.2–5.9` / `5.9–7.5` / `≥ 7.5` | `7,422` (24.55%) / `7,634` (25.25%) / `6,928` (22.92%) / `8,245` (27.28%) | — | — | — | — | — |
| **5-class Fixed** *(dist. only)* | `< 3` / `3–5` / `5–7` / `7–9` / `≥ 9` | `3,409` (11.28%) / `7,581` (25.08%) / `9,403` (31.11%) / `6,866` (22.71%) / `2,970` (9.83%) | — | — | — | — | — |

### Why **3-class Fixed Domain (`< 4` / `4–7` / `≥ 7` tỷ VND)** Is Recommended

1. **+4.63 Macro-F1 higher than 3-class quantile (`0.7397` vs. `0.6934`):** Because `Price` is bell-shaped around `5.90` tỷ, quantile tertiles (`4.8` and `6.9` tỷ) slice straight through the densest cluster of listings (`[4, 7)` contains ~45.8% of all rows), maximizing boundary-flip noise. Fixed cuts `< 4` / `4–7` / `≥ 7` yield cleaner separation.
2. **Zero target leakage:** Fixed domain cutoffs do not inspect the dataset distribution before splitting (if using quantile binning instead, compute quantiles on `Train` only).
3. **Clear domain semantics for slides:**
   - Class `0`: **Phổ thông (`< 4 tỷ`)**
   - Class `1`: **Trung cấp (`4–7 tỷ`)**
   - Class `2`: **Cao cấp (`≥ 7 tỷ`)**
4. **Strong error narrative:** With `HGB (+Addr/FE)`, `98.68%` of predictions fall within ±1 adjacent bin (Test confusion matrix rows `[1021, 299, 51]`, `[216, 2040, 452]`, `[29, 558, 1380]`), showing errors concentrate tightly around adjacent price boundaries.

---

## 4. Feature Engineering & Leakage Guardrails

### Already Validated High-Impact Features (+16.8 to +22.6 Macro-F1 Gain)
- Canonical NFC-normalized `City` and `(District, City)` extracted from `Address`.
- Address keyword indicators: `is_project` (`dự án`, `kđt`, `vinhomes`, etc.) and `is_alley` (`ngõ`, `hẻm`, `ngách`).
- Binary missingness flags (`Frontage_missing`, `Access Road_missing`, `Furniture_missing`, `Direction_missing`).
- `log1p(Area)` for heavy right skew (`3.89`) and linear baseline performance.

### High-Leverage Next Features to Try
- `Ward` and `Street` extracted from `Address` using **out-of-fold / Train-only Target Encoding** or **Frequency Encoding**.
- TF-IDF + TruncatedSVD (`10`–`20` components, fit on `Train` only) on `Address` to capture micro-location signals (major streets, urban developments).
- Domain structural ratios (watch `0`/NaN division): `Area / Floors` (estimated footprint), `Bedrooms + Bathrooms`, `Area / Bedrooms`, `Frontage / Access Road`.
- Train-only group-median imputation for `Frontage` and `Access Road` grouped by `(District, City)`.

### Strict Non-Leakage Rules
- **NEVER** include `Price`, `Price / Area`, `log(Price)`, or any price-derived statistic as an input feature.
- Fit all imputers, scalers, One-Hot / Target / Frequency encoders, TF-IDF vectorizers, and SVD reducers strictly on `Train` (`fit_transform(X_train)`, `transform(X_val)`, `transform(X_test)`).

---

## 5. Recommended Model Lineup & Slide Handoff

To score strongly on **Model Selection & Implementation (30%)** and **Performance Analysis (20%)**:

1. **Model 1 — Multinomial Logistic Regression** *(Linear Baseline)*: Fast, interpretable weights; demonstrates huge jump from numeric-only (`0.477` F1) to engineered location features (`0.703+` F1).
2. **Model 2 — Gradient Boosted Trees** (`HistGradientBoostingClassifier` / `XGBoost` / `LightGBM` / `CatBoost`): Strongest tabular learner for non-linear `Area × District` interactions and missing values.
3. **Model 3 — PyTorch Tabular MLP**: Deep learning architecture (`Linear → BatchNorm1d → ReLU → Dropout` blocks + optional entity embeddings for `City`/`District`, `CrossEntropyLoss` with optional class weights, `AdamW`, `EarlyStopping`) — fulfills the course deep learning focus and produces clean per-epoch **Train/Val Loss & Accuracy curves** (`training_curves.png`) for Slide 7.

### Required Artifacts for `slides/`
Use `slides/scripts/slide_export.py` from your training notebook to populate `slides/public/figures/` and `slides/data/results.json`:
1. `class_distribution.png` — Bar chart of the 3 price tiers (Slide 3)
2. `dataset_samples.png` — Visual summary / EDA distributions by tier & city (Slide 3)
3. `training_curves.png` — Train/Val Loss & Accuracy curves for PyTorch MLP / iterative boosting (Slide 7)
4. `confusion_matrices.png` — Side-by-side Confusion Matrices for the compared models (Slide 8)
5. `error_analysis.png` — Per-class F1 comparison + sample misclassified boundary cases (Slide 9)
6. `preprocessing_impact.png` — Before/after ablation bar chart: Numeric-only vs. +Address/FE (+ imputation/outlier cleaning) (Slide 10)
