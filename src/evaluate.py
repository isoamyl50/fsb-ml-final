"""Evaluation metrics and optional export bridge to the Slidev presentation."""

from __future__ import annotations

import sys
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from .config import SLIDES_SCRIPTS_DIR


def evaluate_predictions(
    model_name: str,
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    class_names: Sequence[str] | None = None,
    export_to_slides: bool = False,
) -> dict[str, Any]:
    """Compute multi-class classification metrics, adjacent-bin accuracy, and confusion matrix."""
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)

    acc = float(accuracy_score(yt, yp))
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(
        yt, yp, average="macro", zero_division=0
    )
    _, _, f1_w, _ = precision_recall_fscore_support(
        yt, yp, average="weighted", zero_division=0
    )
    within_1_bin = float(np.mean(np.abs(yt - yp) <= 1))
    cm = confusion_matrix(yt, yp)

    result: dict[str, Any] = {
        "model": model_name,
        "accuracy": round(acc, 4),
        "precision_macro": round(float(prec_m), 4),
        "recall_macro": round(float(rec_m), 4),
        "f1_macro": round(float(f1_m), 4),
        "f1_weighted": round(float(f1_w), 4),
        "within_1_bin_acc": round(within_1_bin, 4),
        "confusion_matrix": cm.tolist(),
    }

    if export_to_slides:
        scripts_str = str(SLIDES_SCRIPTS_DIR)
        if scripts_str not in sys.path:
            sys.path.insert(0, scripts_str)
        import slide_export

        slide_export.log_test_metrics(
            model_name=model_name,
            y_true=yt,
            y_pred=yp,
            class_names=class_names,
        )

    return result
