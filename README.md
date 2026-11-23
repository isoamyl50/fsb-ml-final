# AI & Machine Learning — Final Course Project

**Topic:** House Price Prediction (Continuous Regression) using the Vietnam Housing Dataset 2024 ([Kaggle](https://www.kaggle.com/datasets/nguyentiennhan/vietnam-housing-dataset-2024))  
**Instructor:** PhD. Truc Thi Kim, Nguyen

---

## Repository Structure

```text
final-project/
├── docs/
│   └── dataset-investigation.md   # Full EDA findings, traps, regression targets & recommendations
├── data/
│   ├── README.md                  # Dataset download instructions (local + Drive cache info)
│   ├── raw/                       # Local CSV (gitignored): vietnam_housing_dataset.csv
│   └── processed/                 # Optional cleaned exports (gitignored)
├── src/
│   ├── config.py                  # Auto-resolves local data/raw/ vs Colab Drive cache paths
│   ├── splits.py                  # Shared frozen decile+group-stratified split (splits_v1.json)
│   ├── targets.py                 # Pluggable target transforms (raw, log1p, price_per_m2, log_price_per_m2)
│   ├── features.py                # Pluggable feature recipes (P0_numeric .. P4_geo_priors, P4_compact)
│   ├── models.py                  # Regression model registry (ridge, lasso, hgb, rf, et, lgbm, xgb, catboost, mlp, blend_trees)
│   ├── runner.py                  # Unified train -> invert target -> evaluate -> atomic JSON run log
│   ├── leaderboard.py             # Aggregates runs/*.json into experiments/leaderboard.md
│   ├── preprocessing.py           # Unicode NFC normalization, Address parser, outlier helper
│   └── evaluate.py                # Regression metrics (RMSE, MAE, R², MAPE, ±20%, ±1 tỷ) + Slidev bridge
├── experiments/
│   ├── configs/                   # Reusable JSON/YAML experiment specs
│   ├── splits/                    # Frozen shared split (splits_v1.json)
│   ├── runs/                      # Append-only: one JSON log per run (conflict-free)
│   └── leaderboard.md             # Auto-generated regression leaderboard ranked by Val RMSE
├── scripts/
│   ├── run_experiment.py          # CLI entrypoint for local or Colab GPU trials
│   ├── build_leaderboard.py       # Rebuilds experiments/leaderboard.md from runs/*.json
│   ├── promote_to_slides.py       # Exports finalized winning runs -> slides/data/results.json
│   ├── colab_job.py               # Self-cleaning Colab GPU job runner
│   └── cache_data_on_drive.py     # Verifies Drive FUSE cache & Colab runtime
├── notebooks/                     # Team notebooks (scratch/ + final documentation notebook)
├── artifacts/                     # Saved checkpoints (binary weights gitignored)
├── slides/                        # Modular Slidev presentation deck (Vietnamese slides + notes)
├── requirements.txt               # Python dependencies
├── AGENTS.md                      # AI agent routing & conventions
└── S10. Final Project.pptx.txt    # Course project requirements & grading rubric
```

## Multi-Contributor Experiment Workflow

1. **Read the investigation first:** [docs/dataset-investigation.md](./docs/dataset-investigation.md) summarizes all 12 columns, traps, and regression recommendations.
2. **List available target transforms, feature recipes, and models:**
   ```bash
   python scripts/run_experiment.py --list
   ```
3. **Run an experiment (logs one conflict-free JSON in `experiments/runs/`):**
   ```bash
   python scripts/run_experiment.py \
     --author your_name \
     --target raw \
     --features P4_geo_priors \
     --model hgb \
     --param learning_rate=0.04 \
     --notes "My regression trial"
   ```
   *(Or pass `--config experiments/configs/baseline_hgb_p4.json`.)*
4. **Refresh the team leaderboard:**
   ```bash
   python scripts/build_leaderboard.py
   ```
5. **Only after locking final hyperparameters on `Val`**, run final `--eval-test` and promote winners to slides:
   ```bash
   python scripts/promote_to_slides.py --runs <run_id_1> <run_id_2>
   ```
