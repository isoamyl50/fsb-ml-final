"""Preprocessing helpers: Vietnamese text normalization, address parsing, and outlier bounds."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

import numpy as np

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
