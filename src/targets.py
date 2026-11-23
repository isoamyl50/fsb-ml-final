"""Pluggable target transform registry mapping Price (billion VND) <-> training target."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

ForwardFn = Callable[[np.ndarray, np.ndarray], np.ndarray]
InverseFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class TargetTransform:
    """Bidirectional target transformation always inverting predictions back to billion VND."""

    name: str
    forward_fn: ForwardFn
    inverse_fn: InverseFn

    def forward(
        self,
        prices: Sequence[float] | np.ndarray,
        areas: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        return self.forward_fn(
            np.asarray(prices, dtype=float),
            np.asarray(areas, dtype=float),
        )

    def inverse(
        self,
        preds: Sequence[float] | np.ndarray,
        areas: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        return self.inverse_fn(
            np.asarray(preds, dtype=float),
            np.asarray(areas, dtype=float),
        )


TARGET_REGISTRY: dict[str, TargetTransform] = {}


def register_target(name: str, forward_fn: ForwardFn, inverse_fn: InverseFn) -> None:
    """Register a named target transformation."""
    TARGET_REGISTRY[name] = TargetTransform(
        name=name,
        forward_fn=forward_fn,
        inverse_fn=inverse_fn,
    )


def list_targets() -> list[str]:
    """Return sorted registered target transform names."""
    return sorted(TARGET_REGISTRY.keys())


def get_target_transform(name: str) -> TargetTransform:
    """Look up a registered TargetTransform by name."""
    if name not in TARGET_REGISTRY:
        raise KeyError(f"Unknown target transform '{name}'. Available: {list_targets()}")
    return TARGET_REGISTRY[name]


register_target(
    "raw",
    forward_fn=lambda prices, areas: prices,
    inverse_fn=lambda preds, areas: preds,
)

register_target(
    "log1p",
    forward_fn=lambda prices, areas: np.log1p(np.maximum(prices, 0.0)),
    inverse_fn=lambda preds, areas: np.expm1(np.clip(preds, -1.0, 5.0)),
)

register_target(
    "price_per_m2",
    forward_fn=lambda prices, areas: np.clip(
        prices * 1000.0 / np.maximum(areas, 1.0), 10.0, 400.0
    ),
    inverse_fn=lambda preds, areas: (
        np.maximum(preds, 1.0) * np.maximum(areas, 1.0) / 1000.0
    ),
)

register_target(
    "log_price_per_m2",
    forward_fn=lambda prices, areas: np.log(
        np.clip(prices * 1000.0 / np.maximum(areas, 1.0), 10.0, 400.0)
    ),
    inverse_fn=lambda preds, areas: (
        np.exp(np.clip(preds, 0.0, 7.0)) * np.maximum(areas, 1.0) / 1000.0
    ),
)
