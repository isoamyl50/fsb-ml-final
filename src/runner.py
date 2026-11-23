"""Shared experiment execution and per-run JSON logging engine."""

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

from .binning import apply_binning
from .config import (
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
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> None:
    if spec_name == "lgbm":
        import lightgbm as lgb

        clf.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
    elif spec_name == "xgb":
        clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    elif spec_name == "mlp":
        clf.fit(X_train, y_train, eval_set=(X_val, y_val))
    else:
        clf.fit(X_train, y_train)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f".tmp.{os.getpid()}.{uuid.uuid4().hex[:6]}")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def run_and_log(
    author: str,
    binning: str,
    features: str,
    model: str,
    params: Mapping[str, Any] | None = None,
    bin_edges: Sequence[float] | None = None,
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
    """Execute one reproducible experiment on the shared frozen split and save a JSON run log."""
    rows = load_raw_records(csv_path)
    train_idx, val_idx, test_idx = load_frozen_splits(path=splits_path, rows=rows)
    if outlier_filter:
        train_idx = _filter_train_indices(rows, train_idx)

    train_rows = [rows[int(i)] for i in train_idx]
    val_rows = [rows[int(i)] for i in val_idx]
    test_rows = [rows[int(i)] for i in test_idx]

    train_prices = [float(r[TARGET_COL]) for r in train_rows]
    val_prices = [float(r[TARGET_COL]) for r in val_rows]
    test_prices = [float(r[TARGET_COL]) for r in test_rows]

    bin_res = apply_binning(
        name=binning,
        train_prices=train_prices,
        val_prices=val_prices,
        test_prices=test_prices,
        custom_edges=bin_edges,
    )

    spec = get_model_spec(model)
    recipe = build_feature_recipe(
        features,
        scale_numeric=spec.needs_scaling,
        preserve_nans=spec.supports_nans,
    )
    X_train = recipe.fit_transform(train_rows, bin_res.y_train)
    X_val = recipe.transform(val_rows)

    param_dict = dict(params or {})
    clf = spec.builder(random_state=seed, **param_dict)

    t0 = time.perf_counter()
    _fit_model(spec.name, clf, X_train, bin_res.y_train, X_val, bin_res.y_val)
    train_seconds = round(time.perf_counter() - t0, 3)

    val_pred = clf.predict(X_val)
    val_metrics = evaluate_predictions(
        model_name=spec.name,
        y_true=bin_res.y_val,
        y_pred=val_pred,
        class_names=bin_res.labels,
        export_to_slides=False,
    )

    test_metrics: dict[str, Any] | None = None
    if eval_test:
        X_test = recipe.transform(test_rows)
        test_pred = clf.predict(X_test)
        test_metrics = evaluate_predictions(
            model_name=spec.name,
            y_true=bin_res.y_test,
            y_pred=test_pred,
            class_names=bin_res.labels,
            export_to_slides=False,
        )

    now_utc = datetime.now(timezone.utc)
    ts_compact = now_utc.strftime("%Y%m%d_%H%M%S")
    short_uid = uuid.uuid4().hex[:4]
    run_id = (
        f"{ts_compact}_{_safe_slug(author)}_{_safe_slug(model)}_"
        f"{_safe_slug(binning)}_{_safe_slug(features)}_{short_uid}"
    )

    git_commit, git_dirty = _detect_git_state()
    class_counts = {
        lbl: int(np.sum(bin_res.y_train == cls_id))
        for cls_id, lbl in enumerate(bin_res.labels)
    }

    payload: dict[str, Any] = {
        "run_id": run_id,
        "timestamp_utc": now_utc.isoformat(),
        "author": author,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "hardware": _detect_hardware(),
        "seed": seed,
        "outlier_filter": outlier_filter,
        "binning": {
            "name": bin_res.name,
            "bin_id": bin_res.bin_id,
            "edges": bin_res.edges,
            "labels": bin_res.labels,
            "n_classes": bin_res.n_classes,
            "train_class_counts": class_counts,
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

    if hasattr(clf, "history_") and isinstance(clf.history_, dict):
        payload["history"] = clf.history_

    if save_run:
        out_path = Path(runs_dir) / f"{run_id}.json"
        _atomic_write_json(out_path, payload)
        payload["saved_path"] = str(out_path)
        if mirror_to_drive and _is_drive_mounted():
            DRIVE_RUNS_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(out_path, DRIVE_RUNS_DIR / out_path.name)

    return payload
