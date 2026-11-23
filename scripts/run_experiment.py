#!/usr/bin/env python3
"""CLI entrypoint to run one regression experiment from flags or a JSON/YAML config."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_TARGET_TRANSFORM, RANDOM_SEED
from src.features import list_feature_recipes
from src.models import list_models, parse_param_value
from src.runner import run_and_log
from src.targets import list_targets


def _load_config_file(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    raw_text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        return json.loads(raw_text)
    if suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required to load .yaml configs. Use .json or install pyyaml."
            ) from exc
        return yaml.safe_load(raw_text)
    raise ValueError(f"Unsupported config extension '{suffix}'; expected .json, .yaml, or .yml")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run and log a reproducible continuous house-price regression experiment."
    )
    parser.add_argument(
        "--list", action="store_true", help="List registered targets, features, and models."
    )
    parser.add_argument("--config", type=Path, help="Optional JSON/YAML config file.")
    parser.add_argument("--author", help="Contributor name/handle (e.g. hoang, an).")
    parser.add_argument(
        "--target",
        help=f"Registered target transform name (default {DEFAULT_TARGET_TRANSFORM}).",
    )
    parser.add_argument("--features", help="Registered feature recipe name.")
    parser.add_argument("--model", help="Registered model name.")
    parser.add_argument(
        "--param", action="append", default=[], help="Model hyperparameter override key=value."
    )
    parser.add_argument(
        "--outlier-filter", action="store_true", help="Filter domain outliers on Train split only."
    )
    parser.add_argument(
        "--eval-test",
        action="store_true",
        help="Evaluate locked Test split (use only for final candidates).",
    )
    parser.add_argument("--seed", type=int, help=f"Random seed (default {RANDOM_SEED}).")
    parser.add_argument("--notes", help="Short experiment notes.")
    parser.add_argument("--no-save", action="store_true", help="Run without writing JSON log.")
    args = parser.parse_args()

    if args.list:
        print("Registered targets :", ", ".join(list_targets()))
        print("Registered features:", ", ".join(list_feature_recipes()))
        print("Registered models  :", ", ".join(list_models()))
        return 0

    cfg: dict[str, Any] = _load_config_file(args.config) if args.config else {}

    author = args.author or cfg.get("author") or "team"
    target = args.target or cfg.get("target") or DEFAULT_TARGET_TRANSFORM
    features = args.features or cfg.get("features")
    model = args.model or cfg.get("model")
    if not (features and model):
        parser.error("--features and --model are required (via CLI or --config).")

    params = dict(cfg.get("params") or {})
    for item in args.param:
        if "=" not in item:
            parser.error(f"Invalid --param '{item}'; expected key=value")
        k, v = item.split("=", 1)
        params[k.strip()] = parse_param_value(v)

    outlier_filter = bool(args.outlier_filter or cfg.get("outlier_filter", False))
    eval_test = bool(args.eval_test or cfg.get("eval_test", False))
    seed = args.seed if args.seed is not None else int(cfg.get("seed", RANDOM_SEED))
    notes = args.notes if args.notes is not None else str(cfg.get("notes", ""))

    result = run_and_log(
        author=author,
        features=features,
        model=model,
        target=target,
        params=params,
        outlier_filter=outlier_filter,
        eval_test=eval_test,
        seed=seed,
        notes=notes,
        save_run=not args.no_save,
    )

    val = result["metrics"]["val"]
    test = result["metrics"].get("test")
    print(f"Run ID      : {result['run_id']}")
    print(
        f"Task        : regression | Target: {target} | "
        f"Features: {features} ({result['features']['n_features']} cols)"
    )
    print(
        f"Val Metrics : RMSE={val['rmse']:.4f} | MAE={val['mae']:.4f} | "
        f"R²={val['r2']:.4f} | MAPE={val['mape']:.2f}% | ±20%={val['within_20pct']:.4f}"
    )
    if test:
        print(
            f"Test Metrics: RMSE={test['rmse']:.4f} | MAE={test['mae']:.4f} | "
            f"R²={test['r2']:.4f} | MAPE={test['mape']:.2f}% | ±20%={test['within_20pct']:.4f}"
        )
    print(f"Train Time  : {result['train_seconds']:.2f}s ({result['hardware']})")
    if "saved_path" in result:
        print(f"Saved JSON  : {result['saved_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
