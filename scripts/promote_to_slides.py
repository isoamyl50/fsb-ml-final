#!/usr/bin/env python3
"""Explicitly promote finalized regression runs into slides/data/results.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import EXPERIMENTS_RUNS_DIR, SLIDES_RESULTS_PATH, SLIDES_SCRIPTS_DIR


def _resolve_run_file(token: str, runs_dir: Path) -> Path:
    exact = runs_dir / (token if token.endswith(".json") else f"{token}.json")
    if exact.is_file():
        return exact
    matches = sorted(runs_dir.glob(f"*{token}*.json"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous run selector '{token}': matched {[m.name for m in matches]}")
    raise FileNotFoundError(f"No run log matching '{token}' in {runs_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Promote selected regression runs from experiments/runs/ into slides/data/results.json."
    )
    parser.add_argument("--runs", nargs="+", required=True, help="Run IDs or unique prefixes to promote.")
    parser.add_argument(
        "--allow-val",
        action="store_true",
        help="Allow promoting Validation metrics if Test metrics have not been evaluated yet.",
    )
    args = parser.parse_args()

    scripts_str = str(SLIDES_SCRIPTS_DIR)
    if scripts_str not in sys.path:
        sys.path.insert(0, scripts_str)
    import slide_export

    promoted_metrics: list[dict[str, Any]] = []
    for token in args.runs:
        run_path = _resolve_run_file(token, EXPERIMENTS_RUNS_DIR)
        run = json.loads(run_path.read_text(encoding="utf-8"))
        test_m = run["metrics"].get("test")
        if test_m is not None:
            metric_src = test_m
            split_tag = "test"
        elif args.allow_val:
            metric_src = run["metrics"]["val"]
            split_tag = "val"
        else:
            raise RuntimeError(
                f"Run '{run['run_id']}' does not have Test metrics. "
                "Re-run with --eval-test or pass --allow-val explicitly."
            )

        promoted_metrics.append(
            {
                "model": f"{run['model']['name']} ({run['features']['name']}, {run['target']['name']})",
                "run_id": run["run_id"],
                "split": split_tag,
                "rmse": metric_src["rmse"],
                "mae": metric_src["mae"],
                "medae": metric_src["medae"],
                "r2": metric_src["r2"],
                "mape": metric_src["mape"],
                "within_10pct": metric_src["within_10pct"],
                "within_20pct": metric_src["within_20pct"],
                "within_1ty": metric_src["within_1ty"],
            }
        )

    results = slide_export.load_results(SLIDES_RESULTS_PATH)
    evaluation = results.setdefault("evaluation", {})
    evaluation["test_metrics"] = promoted_metrics
    slide_export.update_results("evaluation", evaluation, path=SLIDES_RESULTS_PATH)

    print(f"Promoted {len(promoted_metrics)} regression run(s) -> {SLIDES_RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
