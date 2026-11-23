"""Preprocessing helpers: Vietnamese text normalization, address parsing, binning, and splits."""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Sequence

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from .config import PRICE_BIN_EDGES, RANDOM_SEED, SPLIT_RATIOS

CITY_ALIASES = {
    "TP. HCM": "Hồ Chí Minh",
    "TP.HCM": "Hồ Chí Minh",
    "TP HCM": "Hồ Chí Minh",
    "HCM": "Hồ Chí Minh",
    "Thành phố Hồ Chí Minh": "Hồ Chí Minh",
    "TP. Hà Nội": "Hà Nội",
    "TP Hà Nội": "Hà Nội",
    "HN": "Hà Nội",
}

_PROJECT_KEYWORDS = ("dự án", "kđt", "khu đô thị", "vinhomes", "chung cư")
_ALLEY_KEYWORDS = ("ngõ", "hẻm", "ngách", "kiệt")


def normalize_vietnamese_text(text: str | None) -> str:
    """Normalize Vietnamese text to Unicode NFC with collapsed whitespace."""
    if text is None:
        return ""
    cleaned = unicodedata.normalize("NFC", str(text))
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned.rstrip(".").strip()


def parse_address(address: str | None) -> dict[str, Any]:
    """Parse right-to-left location tokens and property type hints from Address."""
    norm = normalize_vietnamese_text(address)
    if not norm:
        return {
            "city": "Unknown",
            "district": "Unknown",
            "ward": "Unknown",
            "is_project": 0,
            "is_alley": 0,
        }

    parts = [normalize_vietnamese_text(p) for p in norm.split(",") if p.strip()]
    raw_city = parts[-1] if parts else "Unknown"
    city = CITY_ALIASES.get(raw_city, raw_city)

    if len(parts) >= 2:
        district = f"{parts[-2]} | {city}"
    else:
        district = f"Unknown | {city}"

    ward = parts[-3] if len(parts) >= 3 else "Unknown"
    lowered = norm.lower()

    return {
        "city": city,
        "district": district,
        "ward": ward,
        "is_project": int(any(kw in lowered for kw in _PROJECT_KEYWORDS)),
        "is_alley": int(any(kw in lowered for kw in _ALLEY_KEYWORDS)),
    }


def bin_price(
    prices: Sequence[float] | np.ndarray,
    edges: Sequence[float] = PRICE_BIN_EDGES,
) -> np.ndarray:
    """Convert continuous prices (billion VND) into discrete class IDs [0..K-1]."""
    arr = np.asarray(prices, dtype=float)
    return np.digitize(arr, bins=list(edges), right=False)


def is_valid_domain_row(
    area: float,
    price: float,
    frontage: float | None = None,
    min_area: float = 15.0,
    max_area: float = 300.0,
    min_ppm2: float = 10.0,
    max_ppm2: float = 400.0,
) -> bool:
    """Check whether a housing listing passes basic domain sanity bounds."""
    if area <= 0 or price <= 0:
        return False
    if area < min_area or area > max_area:
        return False
    if frontage is not None and not np.isnan(frontage) and frontage > area:
        return False
    ppm2 = (price * 1000.0) / area
    return min_ppm2 <= ppm2 <= max_ppm2


def stratified_train_val_test_split(
    y: Sequence[int] | np.ndarray,
    split_ratios: tuple[float, float, float] = SPLIT_RATIOS,
    random_state: int = RANDOM_SEED,
    groups: Sequence[Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return stratified (and optionally group-aware) train/val/test row indices."""
    train_ratio, val_ratio, test_ratio = split_ratios
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("split_ratios must sum to 1.0")

    y_arr = np.asarray(y)
    n_samples = len(y_arr)
    all_idx = np.arange(n_samples)

    if groups is None:
        holdout_ratio = val_ratio + test_ratio
        train_idx, temp_idx = train_test_split(
            all_idx,
            test_size=holdout_ratio,
            random_state=random_state,
            stratify=y_arr,
        )
        rel_test = test_ratio / holdout_ratio
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=rel_test,
            random_state=random_state,
            stratify=y_arr[temp_idx],
        )
        return train_idx, val_idx, test_idx

    # Group-aware split keeps near-duplicate broker reposts within a single split
    groups_arr = np.asarray(groups)
    n_splits_outer = max(2, round(1.0 / (val_ratio + test_ratio)))
    sgkf_outer = StratifiedGroupKFold(
        n_splits=n_splits_outer, shuffle=True, random_state=random_state
    )
    train_idx, temp_idx = next(sgkf_outer.split(all_idx, y_arr, groups_arr))

    sgkf_inner = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=random_state)
    val_rel, test_rel = next(
        sgkf_inner.split(temp_idx, y_arr[temp_idx], groups_arr[temp_idx])
    )
    return train_idx, temp_idx[val_rel], temp_idx[test_rel]
