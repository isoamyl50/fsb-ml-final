"""Shared regression experiment execution and per-run JSON logging engine."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .config import (
    DEFAULT_TARGET_TRANSFORM,
    DRIVE_MOUNT_ROOT,
    DRIVE_RUNS_DIR,
    EXPERIMENTS_RUNS_DIR,
    EXPERIMENTS_SPLITS_PATH,
    PROJECT_ROOT,
    RANDOM_SEED,
    TARGET_COL,
)
from .evaluate import evaluate_predictions
from .features import build_feature_recipe
from .models import get_model_spec
from .preprocessing import is_valid_domain_row
from .splits import load_frozen_splits, load_raw_records
from .targets import get_target_transform


def _safe_slug(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", text.strip()).strip("-").lower()
    return slug or "anon"


def _detect_git_state() -> tuple[str, bool | None]:
    try:
        rev = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        commit = rev.stdout.strip() if rev.returncode == 0 and rev.stdout.strip() else "uncommitted"
        st = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
        )
        dirty = bool(st.stdout.strip()) if st.returncode == 0 else None
        return commit, dirty
    except Exception:
        return "unknown", None


def _detect_hardware() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return f"GPU ({torch.cuda.get_device_name(0)})"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "GPU (Apple MPS)"
    except ImportError:
        pass
    return "CPU"


def _is_drive_mounted() -> bool:
    mounts = Path("/proc/mounts")
    if mounts.is_file():
        text = mounts.read_text(encoding="utf-8", errors="replace")
        return " /content/drive" in text
    return DRIVE_MOUNT_ROOT.is_mount()


def _filter_train_indices(
    rows: Sequence[Mapping[str, str]],
    train_idx: np.ndarray,
) -> np.ndarray:
    kept: list[int] = []
    for idx in train_idx:
        row = rows[int(idx)]
        area = float(row["Area"])
        price = float(row[TARGET_COL])
        f_raw = str(row.get("Frontage", "")).strip()
        frontage = float(f_raw) if f_raw else None
        if is_valid_domain_row(area=area, price=price, frontage=frontage):
            kept.append(int(idx))
    return np.asarray(kept, dtype=int)


def _fit_model(
    spec_name: str,
    clf: Any,
    X_train: np.ndarray,
    y_train_t: np.ndarray,
    X_val: np.ndarray,
    y_val_t: np.ndarray,
) -> None:
    if spec_name == "lgbm":
        import lightgbm as lgb

        clf.fit(
            X_train,
            y_train_t,
            eval_set=[(X_val, y_val_t)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
    elif spec_name == "xgb":
        clf.fit(X_train, y_train_t, eval_set=[(X_val, y_val_t)], verbose=False)
    elif spec_name == "catboost":
        clf.fit(
            X_train,
            y_train_t,
            eval_set=(X_val, y_val_t),
            early_stopping_rounds=50,
            verbose=False,
        )
    elif spec_name == "mlp":
        clf.fit(X_train, y_train_t, eval_set=(X_val, y_val_t))
    else:
        clf.fit(X_train, y_train_t)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f".tmp.{os.getpid()}.{uuid.uuid4().hex[:6]}")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def run_and_log(
    author: str,
    features: str,
    model: str,
    target: str = DEFAULT_TARGET_TRANSFORM,
    params: Mapping[str, Any] | None = None,
    outlier_filter: bool = False,
    eval_test: bool = False,
    seed: int = RANDOM_SEED,
    notes: str = "",
    save_run: bool = True,
    mirror_to_drive: bool = True,
    csv_path: str | Path | None = None,
    splits_path: str | Path = EXPERIMENTS_SPLITS_PATH,
    runs_dir: str | Path = EXPERIMENTS_RUNS_DIR,
) -> dict[str, Any]:
    """Execute one reproducible regression run on the frozen split and log JSON."""
    rows = load_raw_records(csv_path)
    train_idx, val_idx, test_idx = load_frozen_splits(path=splits_path, rows=rows)
    if outlier_filter:
        train_idx = _filter_train_indices(rows, train_idx)

    train_rows = [rows[int(i)] for i in train_idx]
    val_rows = [rows[int(i)] for i in val_idx]
    test_rows = [rows[int(i)] for i in test_idx]

    train_prices = np.array([float(r[TARGET_COL]) for r in train_rows], dtype=float)
    val_prices = np.array([float(r[TARGET_COL]) for r in val_rows], dtype=float)
    test_prices = np.array([float(r[TARGET_COL]) for r in test_rows], dtype=float)

    train_areas = np.array([float(r["Area"]) for r in train_rows], dtype=float)
    val_areas = np.array([float(r["Area"]) for r in val_rows], dtype=float)
    test_areas = np.array([float(r["Area"]) for r in test_rows], dtype=float)

    tfm = get_target_transform(target)
    y_train_t = tfm.forward(train_prices, train_areas)
    y_val_t = tfm.forward(val_prices, val_areas)

    spec = get_model_spec(model)
    recipe = build_feature_recipe(
        features,
        scale_numeric=spec.needs_scaling,
        preserve_nans=spec.supports_nans,
    )
    X_train = recipe.fit_transform(train_rows, y_train_price=train_prices)
    X_val = recipe.transform(val_rows)

    param_dict = dict(params or {})
    reg = spec.builder(random_state=seed, **param_dict)

    t0 = time.perf_counter()
    _fit_model(spec.name, reg, X_train, y_train_t, X_val, y_val_t)
    train_seconds = round(time.perf_counter() - t0, 3)

    val_pred_t = reg.predict(X_val)
    val_pred_price = tfm.inverse(val_pred_t, val_areas)
    val_metrics = evaluate_predictions(
        model_name=spec.name,
        y_true=val_prices,
        y_pred=val_pred_price,
        export_to_slides=False,
    )

    test_metrics: dict[str, Any] | None = None
    if eval_test:
        X_test = recipe.transform(test_rows)
        test_pred_t = reg.predict(X_test)
        test_pred_price = tfm.inverse(test_pred_t, test_areas)
        test_metrics = evaluate_predictions(
            model_name=spec.name,
            y_true=test_prices,
            y_pred=test_pred_price,
            export_to_slides=False,
        )

    now_utc = datetime.now(timezone.utc)
    ts_compact = now_utc.strftime("%Y%m%d_%H%M%S")
    short_uid = uuid.uuid4().hex[:4]
    run_id = (
        f"{ts_compact}_{_safe_slug(author)}_{_safe_slug(model)}_"
        f"{_safe_slug(target)}_{_safe_slug(features)}_{short_uid}"
    )

    git_commit, git_dirty = _detect_git_state()

    payload: dict[str, Any] = {
        "run_id": run_id,
        "timestamp_utc": now_utc.isoformat(),
        "author": author,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "hardware": _detect_hardware(),
        "seed": seed,
        "outlier_filter": outlier_filter,
        "task": "regression",
        "target": {
            "name": tfm.name,
            "unit": "billion_vnd",
        },
        "features": {
            "name": features,
            "n_features": int(X_train.shape[1]),
            "scaled": spec.needs_scaling,
            "preserve_nans": spec.supports_nans,
        },
        "model": {
            "name": spec.name,
            "params": param_dict,
        },
        "splits": {
            "path": str(Path(splits_path).name),
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "n_test": int(len(test_idx)),
        },
        "metrics": {
            "val": val_metrics,
            "test": test_metrics,
        },
        "train_seconds": train_seconds,
        "notes": notes,
    }

    if hasattr(reg, "history_") and isinstance(reg.history_, dict):
        payload["history"] = reg.history_

    if save_run:
        out_path = Path(runs_dir) / f"{run_id}.json"
        _atomic_write_json(out_path, payload)
        payload["saved_path"] = str(out_path)
        if mirror_to_drive and _is_drive_mounted():
            DRIVE_RUNS_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(out_path, DRIVE_RUNS_DIR / out_path.name)

    return payload
