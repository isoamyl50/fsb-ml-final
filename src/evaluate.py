"""Regression evaluation metrics and optional export bridge to Slidev."""

from __future__ import annotations

import sys
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    median_absolute_error,
    r2_score,
    root_mean_squared_error,
)

from .config import SLIDES_SCRIPTS_DIR


def evaluate_predictions(
    model_name: str,
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
    export_to_slides: bool = False,
) -> dict[str, Any]:
    """Compute regression metrics in billion VND (RMSE, MAE, MedAE, R², MAPE, tolerances)."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)

    rmse = float(root_mean_squared_error(yt, yp))
    mae = float(mean_absolute_error(yt, yp))
    medae = float(median_absolute_error(yt, yp))
    r2 = float(r2_score(yt, yp))

    abs_err = np.abs(yt - yp)
    rel_err = abs_err / np.maximum(np.abs(yt), 1e-8)
    mape = float(np.mean(rel_err) * 100.0)
    within_10pct = float(np.mean(rel_err <= 0.10))
    within_20pct = float(np.mean(rel_err <= 0.20))
    within_1ty = float(np.mean(abs_err <= 1.0))

    residuals = yp - yt
    qs = np.quantile(residuals, [0.05, 0.25, 0.50, 0.75, 0.95])

    result: dict[str, Any] = {
        "model": model_name,
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "medae": round(medae, 4),
        "r2": round(r2, 4),
        "mape": round(mape, 2),
        "within_10pct": round(within_10pct, 4),
        "within_20pct": round(within_20pct, 4),
        "within_1ty": round(within_1ty, 4),
        "residual_quantiles": {
            "p05": round(float(qs[0]), 4),
            "p25": round(float(qs[1]), 4),
            "p50": round(float(qs[2]), 4),
            "p75": round(float(qs[3]), 4),
            "p95": round(float(qs[4]), 4),
        },
    }

    if export_to_slides:
        scripts_str = str(SLIDES_SCRIPTS_DIR)
        if scripts_str not in sys.path:
            sys.path.insert(0, scripts_str)
        import slide_export

        slide_export.log_regression_metrics(
            model_name=model_name,
            y_true=yt,
            y_pred=yp,
        )

    return result
