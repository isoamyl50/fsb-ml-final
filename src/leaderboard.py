"""Aggregate per-run JSON logs into Markdown and CSV leaderboards grouped by binning."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
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
    """Group runs by binning scheme, rank by Val Macro-F1, and write Markdown & CSV tables."""
    runs = load_all_runs(runs_dir)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[run["binning"]["bin_id"]].append(run)

    lines: list[str] = [
        "# Experiment Leaderboard",
        "",
        f"Total logged runs: **{len(runs)}** (grouped by `binning.bin_id`, ranked by `Val Macro-F1`).",
        "",
    ]

    flat_rows: list[dict[str, Any]] = []
    for bin_id in sorted(grouped.keys()):
        group_runs = sorted(
            grouped[bin_id],
            key=lambda r: float(r["metrics"]["val"]["f1_macro"]),
            reverse=True,
        )
        labels = group_runs[0]["binning"]["labels"]
        lines.extend(
            [
                f"## Task: `{bin_id}` ({len(labels)} classes)",
                "",
                f"Classes: {', '.join(labels)}",
                "",
                "| Rank | Run ID | Author | Model | Features | Val Acc | Val Macro-F1 | Val ±1 Bin | Test Macro-F1 | Time (s) | Notes |",
                "|---:|---|---|---|---|---:|---:|---:|---:|---:|---|",
            ]
        )

        for rank, run in enumerate(group_runs, start=1):
            val_m = run["metrics"]["val"]
            test_m = run["metrics"].get("test")
            test_f1_str = f"{test_m['f1_macro']:.4f}" if test_m else "—"
            note_clean = str(run.get("notes", "")).replace("|", "/").replace("\n", " ")
            lines.append(
                f"| {rank} | `{run['run_id']}` | {run['author']} | `{run['model']['name']}` "
                f"| `{run['features']['name']}` | {val_m['accuracy']:.4f} | **{val_m['f1_macro']:.4f}** "
                f"| {val_m['within_1_bin_acc']:.4f} | {test_f1_str} | {run['train_seconds']:.2f} | {note_clean} |"
            )
            flat_rows.append(
                {
                    "bin_id": bin_id,
                    "rank": rank,
                    "run_id": run["run_id"],
                    "author": run["author"],
                    "model": run["model"]["name"],
                    "features": run["features"]["name"],
                    "outlier_filter": run.get("outlier_filter", False),
                    "val_acc": val_m["accuracy"],
                    "val_f1_macro": val_m["f1_macro"],
                    "val_f1_weighted": val_m["f1_weighted"],
                    "val_within_1_bin": val_m["within_1_bin_acc"],
                    "test_acc": test_m["accuracy"] if test_m else "",
                    "test_f1_macro": test_m["f1_macro"] if test_m else "",
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
