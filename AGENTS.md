# AGENTS.md — Root Guide for the Final Project Repository

## 1. Read This First (Save Tokens)

Before inspecting `data/raw/vietnam_housing_dataset.csv` or writing exploratory code, **read [`docs/dataset-investigation.md`](./docs/dataset-investigation.md)**. It already contains:
- Exact schema, missingness rates, and quantiles across all `30,229` rows × `12` columns.
- Verified data-quality traps (`dropna()` retention = 8.75%, Unicode NFD `'Hà Nội'` vs. NFC `'Hà Nội'`, `'TP. HCM'` alias, 5 malformed addresses, domain outliers, and 1,631 near-duplicate broker repost groups).
- Baseline binning benchmarks justifying **3-class fixed domain binning (`< 4` / `4–7` / `≥ 7` tỷ VND)**.
- Feature engineering roadmap and strict leakage guardrails.

Also read [`experiments/leaderboard.md`](./experiments/leaderboard.md) before launching new experiments so you know the current best scores per binning scheme.

## 2. Multi-Contributor Experiment Architecture

- **Do NOT edit shared pipeline defaults just to run a trial.** Register new options via:
  - `src/binning.py` (`@register_binning`)
  - `src/features.py` (`@register_features`)
  - `src/models.py` (`@register_model`)
- **Run trials via `scripts/run_experiment.py`** (or `src.runner.run_and_log` in notebooks). Every run evaluates on the shared frozen `experiments/splits/splits_v1.json` (70/15/15 decile + group-stratified) and writes one conflict-free JSON file under `experiments/runs/`.
- **Refresh `experiments/leaderboard.md`** with `python scripts/build_leaderboard.py`.
- **Keep `Test` locked during tuning:** Do NOT pass `--eval-test` during hyperparameter search or feature ablation; only run `--eval-test` on final locked candidates, then use `scripts/promote_to_slides.py` to export to `slides/data/results.json`.

## 3. Leakage & Git Guardrails

- **NEVER** use `Price`, `Price / Area`, or any price-derived value as an input feature.
- Fit all imputers, scalers, encoders, TF-IDF/SVD transforms, quantile cutoffs, and out-of-fold target encoders strictly on `Train`.
- Raw/processed data (`data/raw/*.csv`, `data/processed/*`), model weights (`artifacts/*.pt`, `*.pkl`, etc.), `experiments/leaderboard.csv`, Python caches, and Slidev build/PDF artifacts are gitignored. Never force-commit data or binary weights.

## 4. Presentation Deck (`slides/`)

When updating slides from promoted results, read and strictly follow [`slides/AGENTS.md`](./slides/AGENTS.md) (Vietnamese slides & speaker notes, 8-minute pacing, strict data integrity from `slides/data/results.json` and `slides/public/figures/`).
