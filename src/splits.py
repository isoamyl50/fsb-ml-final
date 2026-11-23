"""Shared frozen train/val/test split generator and loader across all binning schemes."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from .config import (
    ADDRESS_COL,
    EXPERIMENTS_SPLITS_PATH,
    RANDOM_SEED,
    TARGET_COL,
    resolve_dataset_path,
)
from .preprocessing import normalize_vietnamese_text


def load_raw_records(csv_path: str | Path | None = None) -> list[dict[str, str]]:
    """Load raw CSV listings as a list of string dicts using UTF-8-SIG."""
    resolved = resolve_dataset_path(csv_path)
    with resolved.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def build_repost_group_key(row: Mapping[str, str]) -> str:
    """Build deterministic key to keep near-duplicate broker reposts in the same split."""
    return "|".join(
        [
            normalize_vietnamese_text(row.get(ADDRESS_COL, "")),
            str(row.get("Area", "")).strip(),
            str(row.get("Floors", "")).strip(),
            str(row.get("Bedrooms", "")).strip(),
            str(row.get("Bathrooms", "")).strip(),
        ]
    )


def generate_frozen_splits(
    rows: Sequence[Mapping[str, str]] | None = None,
    seed: int = RANDOM_SEED,
    save_path: str | Path = EXPERIMENTS_SPLITS_PATH,
) -> dict[str, Any]:
    """Create a 70/15/15 group-stratified split stratified on continuous price deciles."""
    records = list(rows) if rows is not None else load_raw_records()
    n_total = len(records)
    prices = np.array([float(r[TARGET_COL]) for r in records], dtype=float)
    groups = np.array([build_repost_group_key(r) for r in records])

    decile_edges = np.quantile(prices, np.linspace(0.1, 0.9, 9))
    price_deciles = np.digitize(prices, bins=decile_edges, right=False)

    all_idx = np.arange(n_total)
    sgkf = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=seed)
    folds = [val_idx for _, val_idx in sgkf.split(all_idx, price_deciles, groups)]

    train_idx = np.sort(np.concatenate(folds[:14])).astype(int)
    val_idx = np.sort(np.concatenate(folds[14:17])).astype(int)
    test_idx = np.sort(np.concatenate(folds[17:20])).astype(int)

    # Strict disjointness checks for both rows and near-duplicate repost groups
    train_set, val_set, test_set = set(train_idx), set(val_idx), set(test_idx)
    assert len(train_set) + len(val_set) + len(test_set) == n_total
    assert not (train_set & val_set) and not (train_set & test_set) and not (val_set & test_set)

    g_train = set(groups[train_idx])
    g_val = set(groups[val_idx])
    g_test = set(groups[test_idx])
    assert not (g_train & g_val) and not (g_train & g_test) and not (g_val & g_test)

    payload: dict[str, Any] = {
        "version": "splits_v1",
        "seed": seed,
        "strategy": "StratifiedGroupKFold(n_splits=20) on 10 price deciles + repost group key",
        "n_total": n_total,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "n_unique_groups": int(len(set(groups))),
        "decile_edges": [round(float(x), 4) for x in decile_edges],
        "train_idx": train_idx.tolist(),
        "val_idx": val_idx.tolist(),
        "test_idx": test_idx.tolist(),
    }

    target = Path(save_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"))
        fh.write("\n")

    return payload


def load_frozen_splits(
    path: str | Path = EXPERIMENTS_SPLITS_PATH,
    rows: Sequence[Mapping[str, str]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load frozen train/val/test index arrays, auto-generating if not yet created."""
    target = Path(path)
    if not target.exists():
        payload = generate_frozen_splits(rows=rows, save_path=target)
    else:
        with target.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)

    if rows is not None and payload["n_total"] != len(rows):
        raise ValueError(
            f"Frozen split row count ({payload['n_total']}) does not match dataset ({len(rows)})."
        )

    return (
        np.asarray(payload["train_idx"], dtype=int),
        np.asarray(payload["val_idx"], dtype=int),
        np.asarray(payload["test_idx"], dtype=int),
    )
