"""Pluggable price-binning registry converting continuous Price (billion VND) into classes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .preprocessing import bin_price

BinningBuilder = Callable[
    [np.ndarray, Sequence[float] | None],
    tuple[str, list[float], list[str]],
]

BINNING_REGISTRY: dict[str, BinningBuilder] = {}


@dataclass(frozen=True)
class BinningResult:
    """Discretized labels and exact cutoff metadata for one experiment run."""

    name: str
    bin_id: str
    edges: list[float]
    labels: list[str]
    n_classes: int
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray


def register_binning(name: str) -> Callable[[BinningBuilder], BinningBuilder]:
    """Decorator registering a named price-binning strategy."""

    def _decorator(fn: BinningBuilder) -> BinningBuilder:
        BINNING_REGISTRY[name] = fn
        return fn

    return _decorator


def list_binnings() -> list[str]:
    """Return sorted registered binning strategy names."""
    return sorted(BINNING_REGISTRY.keys())


def _format_edge_labels(edges: Sequence[float]) -> list[str]:
    if not edges:
        raise ValueError("edges must contain at least one cutoff value.")
    labels = [f"< {edges[0]:g} tỷ"]
    for lo, hi in zip(edges[:-1], edges[1:]):
        labels.append(f"{lo:g}–{hi:g} tỷ")
    labels.append(f"≥ {edges[-1]:g} tỷ")
    return labels


@register_binning("fixed_3class")
def _fixed_3class(
    train_prices: np.ndarray,
    custom_edges: Sequence[float] | None = None,
) -> tuple[str, list[float], list[str]]:
    edges = [4.0, 7.0]
    labels = ["Phổ thông (< 4 tỷ)", "Trung cấp (4–7 tỷ)", "Cao cấp (≥ 7 tỷ)"]
    return f"fixed_3class_{edges}", edges, labels


@register_binning("quantile_3class")
def _quantile_3class(
    train_prices: np.ndarray,
    custom_edges: Sequence[float] | None = None,
) -> tuple[str, list[float], list[str]]:
    edges = [round(float(x), 3) for x in np.quantile(train_prices, [1.0 / 3.0, 2.0 / 3.0])]
    return "quantile_3class", edges, _format_edge_labels(edges)


@register_binning("fixed_4class")
def _fixed_4class(
    train_prices: np.ndarray,
    custom_edges: Sequence[float] | None = None,
) -> tuple[str, list[float], list[str]]:
    edges = [4.0, 6.0, 8.0]
    labels = [
        "Phổ thông (< 4 tỷ)",
        "Trung bình (4–6 tỷ)",
        "Khá cao (6–8 tỷ)",
        "Cao cấp (≥ 8 tỷ)",
    ]
    return f"fixed_4class_{edges}", edges, labels


@register_binning("quantile_4class")
def _quantile_4class(
    train_prices: np.ndarray,
    custom_edges: Sequence[float] | None = None,
) -> tuple[str, list[float], list[str]]:
    edges = [round(float(x), 3) for x in np.quantile(train_prices, [0.25, 0.50, 0.75])]
    return "quantile_4class", edges, _format_edge_labels(edges)


@register_binning("fixed_5class")
def _fixed_5class(
    train_prices: np.ndarray,
    custom_edges: Sequence[float] | None = None,
) -> tuple[str, list[float], list[str]]:
    edges = [3.0, 5.0, 7.0, 9.0]
    labels = ["Dưới 3 tỷ", "3–5 tỷ", "5–7 tỷ", "7–9 tỷ", "Từ 9 tỷ trở lên"]
    return f"fixed_5class_{edges}", edges, labels


@register_binning("custom_fixed")
def _custom_fixed(
    train_prices: np.ndarray,
    custom_edges: Sequence[float] | None = None,
) -> tuple[str, list[float], list[str]]:
    if not custom_edges:
        raise ValueError("custom_fixed binning requires non-empty bin_edges.")
    edges = sorted(float(x) for x in custom_edges)
    if len(set(edges)) != len(edges):
        raise ValueError(f"custom_edges must be strictly increasing, got {edges}")
    return f"custom_fixed_{edges}", edges, _format_edge_labels(edges)


def apply_binning(
    name: str,
    train_prices: Sequence[float] | np.ndarray,
    val_prices: Sequence[float] | np.ndarray,
    test_prices: Sequence[float] | np.ndarray,
    custom_edges: Sequence[float] | None = None,
) -> BinningResult:
    """Apply a registered binning strategy using train-only thresholds when quantile-based."""
    if name not in BINNING_REGISTRY:
        raise KeyError(f"Unknown binning '{name}'. Available: {list_binnings()}")

    tr_arr = np.asarray(train_prices, dtype=float)
    va_arr = np.asarray(val_prices, dtype=float)
    te_arr = np.asarray(test_prices, dtype=float)

    bin_id, edges, labels = BINNING_REGISTRY[name](tr_arr, custom_edges)
    y_train = bin_price(tr_arr, edges=edges)
    y_val = bin_price(va_arr, edges=edges)
    y_test = bin_price(te_arr, edges=edges)

    return BinningResult(
        name=name,
        bin_id=bin_id,
        edges=list(edges),
        labels=list(labels),
        n_classes=len(edges) + 1,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )
