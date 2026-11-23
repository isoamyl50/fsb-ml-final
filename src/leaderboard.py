"""Aggregate per-run regression JSON logs into unified Markdown and CSV leaderboards."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .config import EXPERIMENTS_RUNS_DIR, LEADERBOARD_CSV_PATH, LEADERBOARD_MD_PATH


def load_all_runs(runs_dir: str | Path = EXPERIMENTS_RUNS_DIR) -> list[dict[str, Any]]:
    """Load all saved JSON run logs from experiments/runs/."""
    target = Path(runs_dir)
    if not target.exists():
        return []
    runs: list[dict[str, Any]] = []
    for path in sorted(target.glob("*.json")):
        with path.open("r", encoding="utf-8") as fh:
            runs.append(json.load(fh))
    return runs


def build_leaderboard(
    runs_dir: str | Path = EXPERIMENTS_RUNS_DIR,
    md_path: str | Path = LEADERBOARD_MD_PATH,
    csv_path: str | Path = LEADERBOARD_CSV_PATH,
) -> str:
    """Rank all regression runs by Val RMSE ascending and write Markdown & CSV tables."""
    runs = load_all_runs(runs_dir)
    ranked = sorted(runs, key=lambda r: float(r["metrics"]["val"]["rmse"]))

    lines: list[str] = [
        "# Regression Experiment Leaderboard",
        "",
        f"Total logged runs: **{len(ranked)}** (all evaluated in **billion VND**, ranked by `Val RMSE` ascending).",
        "",
        "| Rank | Run ID | Author | Model | Target | Features | Val RMSE | Val MAE | Val R² | Val MAPE | ±20% Acc | Test RMSE | Time (s) | Notes |",
        "|---:|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    flat_rows: list[dict[str, Any]] = []
    for rank, run in enumerate(ranked, start=1):
        val_m = run["metrics"]["val"]
        test_m = run["metrics"].get("test")
        test_rmse_str = f"{test_m['rmse']:.4f}" if test_m else "—"
        target_name = run["target"]["name"]
        note_clean = str(run.get("notes", "")).replace("|", "/").replace("\n", " ")

        lines.append(
            f"| {rank} | `{run['run_id']}` | {run['author']} | `{run['model']['name']}` "
            f"| `{target_name}` | `{run['features']['name']}` | **{val_m['rmse']:.4f}** "
            f"| {val_m['mae']:.4f} | {val_m['r2']:.4f} | {val_m['mape']:.2f}% "
            f"| {val_m['within_20pct']:.4f} | {test_rmse_str} | {run['train_seconds']:.2f} | {note_clean} |"
        )
        flat_rows.append(
            {
                "rank": rank,
                "run_id": run["run_id"],
                "author": run["author"],
                "model": run["model"]["name"],
                "target": target_name,
                "features": run["features"]["name"],
                "outlier_filter": run.get("outlier_filter", False),
                "val_rmse": val_m["rmse"],
                "val_mae": val_m["mae"],
                "val_medae": val_m["medae"],
                "val_r2": val_m["r2"],
                "val_mape": val_m["mape"],
                "val_within_10pct": val_m["within_10pct"],
                "val_within_20pct": val_m["within_20pct"],
                "val_within_1ty": val_m["within_1ty"],
                "test_rmse": test_m["rmse"] if test_m else "",
                "test_mae": test_m["mae"] if test_m else "",
                "test_r2": test_m["r2"] if test_m else "",
                "train_seconds": run["train_seconds"],
                "notes": note_clean,
            }
        )

    lines.append("")
    markdown = "\n".join(lines)
    md_target = Path(md_path)
    md_target.parent.mkdir(parents=True, exist_ok=True)
    md_target.write_text(markdown, encoding="utf-8")

    csv_target = Path(csv_path)
    if flat_rows:
        with csv_target.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(flat_rows[0].keys()))
            writer.writeheader()
            writer.writerows(flat_rows)

    return markdown
