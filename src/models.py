"""Pluggable model registry for continuous house price regression models."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from .config import RANDOM_SEED


@dataclass(frozen=True)
class ModelSpec:
    """Metadata and factory callable for one registered regression model family."""

    name: str
    builder: Callable[..., Any]
    needs_scaling: bool
    supports_nans: bool
    accepts_val: bool = False


MODEL_REGISTRY: dict[str, ModelSpec] = {}


def register_model(
    name: str,
    needs_scaling: bool,
    supports_nans: bool,
    accepts_val: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator registering a regression model factory."""

    def _decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        MODEL_REGISTRY[name] = ModelSpec(
            name=name,
            builder=fn,
            needs_scaling=needs_scaling,
            supports_nans=supports_nans,
            accepts_val=accepts_val,
        )
        return fn

    return _decorator


def list_models() -> list[str]:
    """Return sorted registered regression model names."""
    return sorted(MODEL_REGISTRY.keys())


@register_model("linear_sgd", needs_scaling=False, supports_nans=False, accepts_val=True)
def build_linear_sgd(random_state: int = RANDOM_SEED, **kwargs: Any) -> Any:
    """Build the isolated CPU SGD adapter; scaling is owned by the wrapper."""
    from .approach_1_linear_sgd.wrapper import LinearSGDWrapper

    return LinearSGDWrapper(seed=random_state, **kwargs)


def get_model_spec(name: str) -> ModelSpec:
    """Look up a registered ModelSpec by name."""
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {list_models()}")
    return MODEL_REGISTRY[name]


def parse_param_value(raw: str) -> Any:
    """Parse CLI key=value strings into bool, None, int, float, or str."""
    text = raw.strip()
    low = text.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low in ("none", "null"):
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


@register_model("ridge", needs_scaling=True, supports_nans=False, accepts_val=False)
def build_ridge(random_state: int = RANDOM_SEED, **kwargs: Any) -> Ridge:
    """Build L2-regularized linear Ridge baseline."""
    params: dict[str, Any] = {"alpha": 10.0, "random_state": random_state}
    params.update(kwargs)
    return Ridge(**params)


@register_model("lasso", needs_scaling=True, supports_nans=False, accepts_val=False)
def build_lasso(random_state: int = RANDOM_SEED, **kwargs: Any) -> Lasso:
    """Build L1-regularized sparse linear Lasso baseline."""
    params: dict[str, Any] = {"alpha": 0.005, "max_iter": 3000, "random_state": random_state}
    params.update(kwargs)
    return Lasso(**params)


@register_model("hgb", needs_scaling=False, supports_nans=True, accepts_val=False)
def build_hist_gbdt(
    random_state: int = RANDOM_SEED,
    **kwargs: Any,
) -> HistGradientBoostingRegressor:
    """Build HistGradientBoostingRegressor with column subsampling & regularization."""
    params: dict[str, Any] = {
        "loss": "squared_error",
        "max_iter": 400,
        "learning_rate": 0.04,
        "max_leaf_nodes": 63,
        "min_samples_leaf": 25,
        "l2_regularization": 2.0,
        "max_features": 0.4,
        "early_stopping": True,
        "validation_fraction": 0.15,
        "random_state": random_state,
    }
    params.update(kwargs)
    return HistGradientBoostingRegressor(**params)


@register_model("rf", needs_scaling=False, supports_nans=False, accepts_val=False)
def build_random_forest(
    random_state: int = RANDOM_SEED,
    **kwargs: Any,
) -> RandomForestRegressor:
    """Build RandomForestRegressor."""
    params: dict[str, Any] = {
        "n_estimators": 400,
        "max_features": "sqrt",
        "n_jobs": -1,
        "random_state": random_state,
    }
    params.update(kwargs)
    return RandomForestRegressor(**params)


@register_model("et", needs_scaling=False, supports_nans=False, accepts_val=False)
def build_extra_trees(
    random_state: int = RANDOM_SEED,
    **kwargs: Any,
) -> ExtraTreesRegressor:
    """Build ExtraTreesRegressor."""
    params: dict[str, Any] = {
        "n_estimators": 400,
        "max_features": "sqrt",
        "n_jobs": -1,
        "random_state": random_state,
    }
    params.update(kwargs)
    return ExtraTreesRegressor(**params)


@register_model("lgbm", needs_scaling=False, supports_nans=True, accepts_val=True)
def build_lgbm(random_state: int = RANDOM_SEED, **kwargs: Any) -> Any:
    """Build LightGBM regressor (lazy import)."""
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError(
            "lightgbm is required for model='lgbm'. Run on Colab or install lightgbm."
        ) from exc

    params: dict[str, Any] = {
        "n_estimators": 1500,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "subsample_for_bin": 50000,
        "colsample_bytree": 0.5,
        "subsample": 0.8,
        "subsample_freq": 1,
        "reg_lambda": 2.0,
        "random_state": random_state,
        "n_jobs": -1,
        "verbosity": -1,
    }
    params.update(kwargs)
    return lgb.LGBMRegressor(**params)


@register_model("xgb", needs_scaling=False, supports_nans=True, accepts_val=True)
def build_xgb(random_state: int = RANDOM_SEED, **kwargs: Any) -> Any:
    """Build XGBoost regressor (lazy import; auto-uses CUDA hist when GPU is available)."""
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise ImportError(
            "xgboost is required for model='xgb'. Run on Colab or install xgboost."
        ) from exc

    device = "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            device = "cuda"
    except ImportError:
        pass

    params: dict[str, Any] = {
        "n_estimators": 1500,
        "learning_rate": 0.03,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.5,
        "reg_lambda": 2.0,
        "tree_method": "hist",
        "device": device,
        "early_stopping_rounds": 50,
        "eval_metric": "rmse",
        "random_state": random_state,
        "n_jobs": -1,
    }
    params.update(kwargs)
    return xgb.XGBRegressor(**params)


@register_model("catboost", needs_scaling=False, supports_nans=True, accepts_val=True)
def build_catboost(random_state: int = RANDOM_SEED, **kwargs: Any) -> Any:
    """Build CatBoost regressor (lazy import)."""
    try:
        from catboost import CatBoostRegressor
    except ImportError as exc:
        raise ImportError(
            "catboost is required for model='catboost'. Install via `pip install catboost`."
        ) from exc

    params: dict[str, Any] = {
        "iterations": 2000,
        "learning_rate": 0.04,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "loss_function": "RMSE",
        "random_seed": random_state,
        "verbose": 0,
    }
    params.update(kwargs)
    return CatBoostRegressor(**params)


def build_tabular_mlp(
    input_dim: int,
    hidden_dims: tuple[int, ...] = (128, 64),
    dropout: float = 0.25,
) -> Any:
    """Build a PyTorch Tabular Regression MLP (BatchNorm1d + SiLU + Dropout)."""
    try:
        import torch.nn as nn
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for build_tabular_mlp(). Install via `pip install torch`."
        ) from exc

    layers: list[nn.Module] = []
    in_features = input_dim
    for hidden_dim in hidden_dims:
        layers.extend(
            [
                nn.Linear(in_features, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            ]
        )
        in_features = hidden_dim
    layers.append(nn.Linear(in_features, 1))
    return nn.Sequential(*layers)


class TabularMLPRegressor:
    """Scikit-learn-style PyTorch Tabular MLP regressor with early stopping and history."""

    def __init__(
        self,
        hidden_dims: tuple[int, ...] = (128, 64),
        dropout: float = 0.25,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        epochs: int = 50,
        patience: int = 10,
        loss: str = "smooth_l1",
        random_state: int = RANDOM_SEED,
    ) -> None:
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.loss = loss
        self.random_state = random_state
        self.history_: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_rmse": [],
            "val_mae": [],
            "val_r2": [],
        }
        self._model: Any = None
        self._device: str = "cpu"

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> "TabularMLPRegressor":
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError as exc:
            raise ImportError(
                "PyTorch is required for model='mlp'. Run on Colab or install torch."
            ) from exc

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        if torch.cuda.is_available():
            self._device = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            self._device = "mps"
        else:
            self._device = "cpu"

        self._model = build_tabular_mlp(
            input_dim=X_train.shape[1],
            hidden_dims=tuple(self.hidden_dims),
            dropout=self.dropout,
        ).to(self._device)

        criterion = nn.MSELoss() if self.loss == "mse" else nn.SmoothL1Loss(beta=0.5)
        optimizer = torch.optim.AdamW(
            self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        ds_tr = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )
        loader_tr = DataLoader(ds_tr, batch_size=self.batch_size, shuffle=True)

        best_rmse = float("inf")
        best_state = copy.deepcopy(self._model.state_dict())
        wait = 0

        for _ in range(self.epochs):
            self._model.train()
            tr_loss_sum = 0.0
            tr_total = 0
            for xb, yb in loader_tr:
                xb = xb.to(self._device)
                yb = yb.to(self._device)
                optimizer.zero_grad()
                preds = self._model(xb).squeeze(-1)
                loss_val = criterion(preds, yb)
                loss_val.backward()
                optimizer.step()

                tr_loss_sum += float(loss_val.item()) * len(yb)
                tr_total += len(yb)

            scheduler.step()
            self.history_["train_loss"].append(tr_loss_sum / max(tr_total, 1))

            if eval_set is not None:
                x_va, y_va = eval_set
                self._model.eval()
                with torch.no_grad():
                    vx = torch.tensor(x_va, dtype=torch.float32, device=self._device)
                    vy = torch.tensor(y_va, dtype=torch.float32, device=self._device)
                    v_preds = self._model(vx).squeeze(-1)
                    v_loss = float(criterion(v_preds, vy).item())
                    v_np = v_preds.cpu().numpy()
                v_rmse = float(root_mean_squared_error(y_va, v_np))
                v_mae = float(mean_absolute_error(y_va, v_np))
                v_r2 = float(r2_score(y_va, v_np))
                self.history_["val_loss"].append(v_loss)
                self.history_["val_rmse"].append(v_rmse)
                self.history_["val_mae"].append(v_mae)
                self.history_["val_r2"].append(v_r2)

                if v_rmse < best_rmse:
                    best_rmse = v_rmse
                    best_state = copy.deepcopy(self._model.state_dict())
                    wait = 0
                else:
                    wait += 1
                    if wait >= self.patience:
                        break

        if eval_set is not None:
            self._model.load_state_dict(best_state)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch

        self._model.eval()
        with torch.no_grad():
            tx = torch.tensor(X, dtype=torch.float32, device=self._device)
            return self._model(tx).squeeze(-1).cpu().numpy()


@register_model("mlp", needs_scaling=True, supports_nans=False, accepts_val=True)
def build_mlp_regressor(
    random_state: int = RANDOM_SEED,
    **kwargs: Any,
) -> TabularMLPRegressor:
    """Build TabularMLPRegressor wrapper."""
    return TabularMLPRegressor(random_state=random_state, **kwargs)


class BlendTreesRegressor:
    """Weighted average blend of HistGradientBoosting, RandomForest, and ExtraTrees regressors."""

    def __init__(
        self,
        random_state: int = RANDOM_SEED,
        w_hgb: float = 0.45,
        w_rf: float = 0.35,
        w_et: float = 0.20,
    ) -> None:
        self.random_state = random_state
        total_w = max(float(w_hgb + w_rf + w_et), 1e-12)
        self.weights = (float(w_hgb) / total_w, float(w_rf) / total_w, float(w_et) / total_w)
        self._hgb = build_hist_gbdt(random_state=random_state)
        self._rf = build_random_forest(random_state=random_state + 1, n_estimators=500)
        self._et = build_extra_trees(random_state=random_state + 2, n_estimators=500)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BlendTreesRegressor":
        self._hgb.fit(X, y)
        self._rf.fit(X, y)
        self._et.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        w1, w2, w3 = self.weights
        return (
            w1 * self._hgb.predict(X)
            + w2 * self._rf.predict(X)
            + w3 * self._et.predict(X)
        )


@register_model("blend_trees", needs_scaling=False, supports_nans=False, accepts_val=False)
def build_blend_trees(
    random_state: int = RANDOM_SEED,
    **kwargs: Any,
) -> BlendTreesRegressor:
    """Build weighted HGB + RF + ET regression blend."""
    return BlendTreesRegressor(random_state=random_state, **kwargs)
