# Regression Experiment Leaderboard

Total logged runs: **14** (all evaluated in **billion VND**, ranked by `Val RMSE` ascending).

| Rank | Run ID | Author | Model | Target | Features | Val RMSE | Val MAE | Val R² | Val MAPE | ±20% Acc | Test RMSE | Time (s) | Notes |
|---:|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `20261123_113555_baseline_hgb_raw_p4_geo_priors_d5ca` | baseline | `hgb` | `raw` | `P4_geo_priors` | **1.1709** | 0.8496 | 0.7210 | 16.98% | 0.7356 | — | 17.04 | P4 location/ppm2 priors + OOF text Ridge with HistGradientBoostingRegressor |
| 2 | `20261123_113626_baseline_rf_raw_p4_geo_priors_1b4b` | baseline | `rf` | `raw` | `P4_geo_priors` | **1.1948** | 0.8623 | 0.7095 | 17.84% | 0.7272 | — | 3.84 | P4 RandomForestRegressor baseline |
| 3 | `20261123_113616_baseline_hgb_log_price_per_m2_p4_geo_priors_2633` | baseline | `hgb` | `log_price_per_m2` | `P4_geo_priors` | **1.1997** | 0.8671 | 0.7072 | 16.58% | 0.7354 | — | 14.13 | P4 HGB predicting log(Price/m2) inverted to billion VND |
| 4 | `20261123_113525_baseline_ridge_raw_p4_geo_priors_dcb8` | baseline | `ridge` | `raw` | `P4_geo_priors` | **1.2794** | 0.9574 | 0.6669 | 19.65% | 0.6798 | — | 0.02 | P4 geo/ppm2 priors + OOF text Ridge regression |
| 5 | `L03` | team | `linear_sgd` | `raw` | `P4_geo_priors` | **1.2826** | 0.9639 | 0.6653 | 19.90% | 0.6750 | — | 8.98 |  |
| 6 | `L03_pipeline_v1` | team | `linear_sgd` | `raw` | `P4_geo_priors` | **1.2826** | 0.9639 | 0.6653 | 19.90% | 0.6750 | — | 9.26 | Fixed L03 config; verify complete pipeline export and fresh-process reload. No Test evaluation. |
| 7 | `L03_repeat` | team | `linear_sgd` | `raw` | `P4_geo_priors` | **1.2826** | 0.9639 | 0.6653 | 19.90% | 0.6750 | — | 8.47 |  |
| 8 | `L01` | team | `linear_sgd` | `raw` | `P4_geo_priors` | **1.2831** | 0.9639 | 0.6650 | 19.91% | 0.6748 | — | 9.00 |  |
| 9 | `L05` | team | `linear_sgd` | `raw` | `P4_geo_priors` | **1.2831** | 0.9639 | 0.6650 | 19.91% | 0.6748 | — | 8.71 |  |
| 10 | `L04a` | team | `linear_sgd` | `raw` | `P4_geo_priors` | **1.2843** | 0.9649 | 0.6644 | 19.84% | 0.6767 | — | 25.76 |  |
| 11 | `L02` | team | `linear_sgd` | `raw` | `P4_geo_priors` | **1.2892** | 0.9683 | 0.6619 | 20.13% | 0.6759 | — | 10.98 |  |
| 12 | `L04b` | team | `linear_sgd` | `raw` | `P4_geo_priors` | **1.3216** | 0.9943 | 0.6446 | 20.60% | 0.6673 | — | 4.33 |  |
| 13 | `20261123_113532_baseline_hgb_raw_p0_numeric_c860` | baseline | `hgb` | `raw` | `P0_numeric` | **1.7936** | 1.4246 | 0.3455 | 32.28% | 0.4926 | — | 4.14 | P0 raw numeric HGB regression baseline |
| 14 | `20261123_113518_baseline_ridge_raw_p0_numeric_9536` | baseline | `ridge` | `raw` | `P0_numeric` | **1.9960** | 1.6310 | 0.1894 | 37.76% | 0.4227 | — | 0.01 | P0 raw numeric Ridge regression baseline |
