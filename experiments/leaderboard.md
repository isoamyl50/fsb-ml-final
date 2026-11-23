# Experiment Leaderboard

Total logged runs: **5** (grouped by `binning.bin_id`, ranked by `Val Macro-F1`).

## Task: `fixed_3class_[4.0, 7.0]` (3 classes)

Classes: Phổ thông (< 4 tỷ), Trung cấp (4–7 tỷ), Cao cấp (≥ 7 tỷ)

| Rank | Run ID | Author | Model | Features | Val Acc | Val Macro-F1 | Val ±1 Bin | Test Macro-F1 | Time (s) | Notes |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---|
| 1 | `20261123_085948_baseline_hgb_fixed_3class_p3_tfidf_te_2534` | baseline | `hgb` | `P3_tfidf_te` | 0.7537 | **0.7608** | 0.9903 | — | 22.34 | P3 + OOF Target Encoding + TF-IDF SVD(20) HGB |
| 2 | `20261123_085923_baseline_hgb_fixed_3class_p2_domain_a027` | baseline | `hgb` | `P2_domain` | 0.7310 | **0.7384** | 0.9859 | — | 21.69 | P2 location + domain features with HistGradientBoosting |
| 3 | `20261123_085850_baseline_logreg_fixed_3class_p2_domain_4d3c` | baseline | `logreg` | `P2_domain` | 0.7054 | **0.7118** | 0.9843 | — | 2.13 | P2 location + domain features linear baseline |
| 4 | `20261123_085900_baseline_hgb_fixed_3class_p0_numeric_a8f3` | baseline | `hgb` | `P0_numeric` | 0.5828 | **0.5767** | 0.9627 | — | 7.62 | P0 raw numeric HGB baseline |
| 5 | `20261123_085846_baseline_logreg_fixed_3class_p0_numeric_d03b` | baseline | `logreg` | `P0_numeric` | 0.5235 | **0.4942** | 0.9674 | — | 0.03 | P0 raw numeric linear baseline |
