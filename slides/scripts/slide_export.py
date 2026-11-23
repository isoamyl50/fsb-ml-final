"""Helper utilities for exporting Jupyter Notebook plots and metrics into the Slidev deck."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

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


def log_test_metrics(
    model_name: str,
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    class_names: Sequence[str] | None = None,
    path: Path = RESULTS_PATH,
) -> dict[str, Any]:
    """Compute multi-class test metrics with scikit-learn and upsert into results.json."""
    try:
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            precision_recall_fscore_support,
        )
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for log_test_metrics()."
        ) from exc

    acc = float(accuracy_score(y_true, y_pred))
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    _, _, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    summary = {
        "model": model_name,
        "accuracy": round(acc, 4),
        "precision_macro": round(float(prec_m), 4),
        "recall_macro": round(float(rec_m), 4),
        "f1_macro": round(float(f1_m), 4),
        "f1_weighted": round(float(f1_w), 4),
    }

    report_kwargs: dict[str, Any] = {"output_dict": True, "zero_division": 0}
    if class_names is not None:
        report_kwargs["target_names"] = list(class_names)
    raw_report = classification_report(y_true, y_pred, **report_kwargs)

    formatted_report: dict[str, Any] = {}
    for key, val in raw_report.items():
        if isinstance(val, dict):
            formatted_report[str(key)] = {
                metric: round(float(score), 4) if metric != "support" else int(score)
                for metric, score in val.items()
            }
        else:
            formatted_report[str(key)] = round(float(val), 4)

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

    per_class = evaluation.setdefault("per_class_report", {})
    per_class[model_name] = formatted_report

    update_results("evaluation", evaluation, path=target)
    return summary
