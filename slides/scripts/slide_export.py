"""Helper utilities for exporting Jupyter Notebook plots and regression metrics into Slidev."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

SLIDES_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = SLIDES_DIR / "public" / "figures"
RESULTS_PATH = SLIDES_DIR / "data" / "results.json"


def apply_slide_theme() -> None:
    """Configure Matplotlib rcParams for crisp light-mode figures matching the Slidev deck."""
    try:
        import matplotlib as mpl
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required to configure the slide plot theme."
        ) from exc

    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#cbd5e1",
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#cbd5e1",
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.45,
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "legend.frameon": True,
            "legend.facecolor": "white",
            "legend.edgecolor": "#e2e8f0",
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def save_figure(fig: Any, name: str, close: bool = False) -> Path:
    """Save a Matplotlib figure as a 300-DPI PNG inside public/figures/."""
    stem = name[:-4] if name.endswith(".png") else name
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / f"{stem}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f'Saved figure -> {out_path} | Slidev prop: src="/figures/{stem}.png"')

    if close:
        import matplotlib.pyplot as plt

        plt.close(fig)

    return out_path


def load_results(path: Path = RESULTS_PATH) -> dict[str, Any]:
    """Load the shared results JSON payload (returns empty dict if missing)."""
    target = Path(path)
    if not target.exists():
        return {}
    with target.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def update_results(section: str, value: Any, path: Path = RESULTS_PATH) -> Path:
    """Update one top-level section in results.json using UTF-8 encoding."""
    target = Path(path)
    data = load_results(target)
    data[section] = value
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return target


def log_regression_metrics(
    model_name: str,
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
    path: Path = RESULTS_PATH,
) -> dict[str, Any]:
    """Compute regression test metrics in billion VND and upsert into results.json."""
    try:
        from sklearn.metrics import (
            mean_absolute_error,
            median_absolute_error,
            r2_score,
            root_mean_squared_error,
        )
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for log_regression_metrics()."
        ) from exc

    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    abs_err = np.abs(yt - yp)
    rel_err = abs_err / np.maximum(np.abs(yt), 1e-8)

    summary = {
        "model": model_name,
        "rmse": round(float(root_mean_squared_error(yt, yp)), 4),
        "mae": round(float(mean_absolute_error(yt, yp)), 4),
        "medae": round(float(median_absolute_error(yt, yp)), 4),
        "r2": round(float(r2_score(yt, yp)), 4),
        "mape": round(float(np.mean(rel_err) * 100.0), 2),
        "within_10pct": round(float(np.mean(rel_err <= 0.10)), 4),
        "within_20pct": round(float(np.mean(rel_err <= 0.20)), 4),
        "within_1ty": round(float(np.mean(abs_err <= 1.0)), 4),
    }

    target = Path(path)
    data = load_results(target)
    evaluation = data.setdefault("evaluation", {})
    existing_metrics = [
        row
        for row in evaluation.get("test_metrics", [])
        if not str(row.get("model", "")).startswith("[TODO")
        and row.get("model") != model_name
    ]
    existing_metrics.append(summary)
    evaluation["test_metrics"] = existing_metrics

    update_results("evaluation", evaluation, path=target)
    return summary
