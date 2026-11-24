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


def get_model_spec(name: str) -> ModelSpec:
    """Look up a registered ModelSpec by name."""
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {list_models()}")
    return MODEL_REGISTRY[name]


def parse_param_value(raw: str) -> Any:
    """Parse CLI key=value strings into bool, None, int, float, int tuple, or str."""
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
        pass

    has_brackets = (text.startswith("(") and text.endswith(")")) or (
        text.startswith("[") and text.endswith("]")
    )
    if has_brackets or "," in text:
        inner = text[1:-1].strip() if has_brackets else text
        parts = [p.strip() for p in inner.split(",") if p.strip()]
        if parts:
            try:
                return tuple(int(p) for p in parts)
            except ValueError:
                pass
    return text


def _normalize_hidden_dims(
    hidden_dims: tuple[int, ...] | list[int] | int | str,
) -> tuple[int, ...]:
    """Normalize hidden layer sizes from int, str, list, or tuple into a positive int tuple."""
    if isinstance(hidden_dims, str):
        parsed = parse_param_value(hidden_dims)
        if isinstance(parsed, int):
            dims = (parsed,)
        elif isinstance(parsed, tuple):
            dims = parsed
        else:
            raise ValueError(f"Could not parse hidden_dims='{hidden_dims}' into integers.")
    elif isinstance(hidden_dims, int):
        dims = (int(hidden_dims),)
    elif isinstance(hidden_dims, (list, tuple)):
        dims = tuple(int(x) for x in hidden_dims)
    else:
        raise TypeError(f"Unsupported hidden_dims type: {type(hidden_dims).__name__}")

    if not dims or any(d <= 0 for d in dims):
        raise ValueError(f"hidden_dims must contain positive integers, got {dims}")
    return dims


def _resolve_torch_device() -> str:
    """Select the fastest available PyTorch compute device."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _build_loss_fn(loss: str) -> Any:
    """Build PyTorch regression loss function by registered name."""
    import torch.nn as nn

    key = loss.strip().lower()
    if key == "mse":
        return nn.MSELoss()
    if key == "smooth_l1":
        return nn.SmoothL1Loss(beta=0.5)
    if key == "huber":
        return nn.HuberLoss(delta=1.0)
    raise ValueError(f"Unsupported loss '{loss}'. Expected 'mse', 'smooth_l1', or 'huber'.")


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


def _build_mean_fallback_state(model: Any, mean_target: float) -> dict[str, Any]:
    """Create a finite constant-mean fallback state dict if epoch 1 diverges."""
    import torch.nn as nn

    state = copy.deepcopy(model.state_dict())
    last_linear_prefix: str | None = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            last_linear_prefix = name
    if last_linear_prefix is not None:
        weight_key = f"{last_linear_prefix}.weight" if last_linear_prefix else "weight"
        bias_key = f"{last_linear_prefix}.bias" if last_linear_prefix else "bias"
        if weight_key in state:
            state[weight_key].zero_()
        if bias_key in state:
            state[bias_key].fill_(float(mean_target))
    return state


class _TorchTabularBase:
    """Shared deterministic PyTorch tabular regression training & early-stopping engine."""

    def __init__(
        self,
        lr: float,
        weight_decay: float,
        batch_size: int,
        epochs: int,
        patience: int,
        loss: str,
        grad_clip: float | None,
        random_state: int,
        use_scheduler: bool = False,
        drop_last_if_single: bool = False,
    ) -> None:
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.patience = int(patience)
        self.loss = str(loss)
        self.grad_clip = float(grad_clip) if grad_clip is not None else None
        self.random_state = int(random_state)
        self.use_scheduler = bool(use_scheduler)
        self._drop_last_if_single = bool(drop_last_if_single)
        self.history_: dict[str, Any] = {
            "train_loss": [],
            "val_loss": [],
            "val_rmse": [],
            "val_mae": [],
            "val_r2": [],
            "best_epoch": 0,
            "stopped_epoch": 0,
            "best_val_rmse": None,
            "diverged": False,
            "n_params": 0,
        }
        self._model: Any = None
        self._device: str = "cpu"

    def _build_module(self, input_dim: int) -> Any:
        raise NotImplementedError

    def _build_optimizer(self) -> Any:
        raise NotImplementedError

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
        inverse_fn: Callable[[np.ndarray], np.ndarray] | None = None,
        y_val_orig: np.ndarray | None = None,
    ) -> "_TorchTabularBase":
        try:
            import torch
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError as exc:
            raise ImportError(
                "PyTorch is required for 'linear_sgd' and 'mlp'. Run on Colab or install torch."
            ) from exc

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        self._device = _resolve_torch_device()
        if self._device == "cuda":
            torch.cuda.manual_seed_all(self.random_state)

        self._model = self._build_module(X_train.shape[1]).to(self._device)
        n_params = int(sum(p.numel() for p in self._model.parameters()))
        fallback_state = _build_mean_fallback_state(self._model, float(np.mean(y_train)))

        criterion = _build_loss_fn(self.loss)
        optimizer = self._build_optimizer()
        scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
            if self.use_scheduler
            else None
        )

        ds_tr = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )
        generator = torch.Generator()
        generator.manual_seed(self.random_state)
        drop_last = (
            self._drop_last_if_single
            and len(ds_tr) > self.batch_size
            and (len(ds_tr) % self.batch_size == 1)
        )
        loader_tr = DataLoader(
            ds_tr,
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
            drop_last=drop_last,
        )

        history: dict[str, Any] = {
            "train_loss": [],
            "val_loss": [],
            "val_rmse": [],
            "val_mae": [],
            "val_r2": [],
            "best_epoch": 0,
            "stopped_epoch": 0,
            "best_val_rmse": None,
            "diverged": False,
            "n_params": n_params,
        }

        best_rmse = float("inf")
        best_state = fallback_state
        wait = 0

        for epoch_idx in range(1, self.epochs + 1):
            self._model.train()
            tr_loss_sum = 0.0
            tr_total = 0
            epoch_diverged = False

            for xb, yb in loader_tr:
                xb = xb.to(self._device)
                yb = yb.to(self._device)
                optimizer.zero_grad()
                preds = self._model(xb).squeeze(-1)
                loss_val = criterion(preds, yb)
                if not torch.isfinite(loss_val):
                    epoch_diverged = True
                    break
                loss_val.backward()
                if self.grad_clip is not None and self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self._model.parameters(), self.grad_clip)
                optimizer.step()

                tr_loss_sum += float(loss_val.item()) * len(yb)
                tr_total += len(yb)

            if epoch_diverged:
                history["diverged"] = True
                break

            tr_loss = tr_loss_sum / max(tr_total, 1)
            if not np.isfinite(tr_loss):
                history["diverged"] = True
                break

            if scheduler is not None:
                scheduler.step()

            if eval_set is None:
                history["train_loss"].append(round(float(tr_loss), 6))
                history["stopped_epoch"] = epoch_idx
                continue

            x_va, y_va = eval_set
            self._model.eval()
            with torch.no_grad():
                vx = torch.tensor(x_va, dtype=torch.float32, device=self._device)
                vy = torch.tensor(y_va, dtype=torch.float32, device=self._device)
                v_preds = self._model(vx).squeeze(-1)
                v_loss = float(criterion(v_preds, vy).item())
                v_np = v_preds.cpu().numpy()

            if not np.isfinite(v_loss) or not np.all(np.isfinite(v_np)):
                history["diverged"] = True
                break

            if inverse_fn is not None and y_val_orig is not None:
                pred_eval = np.asarray(inverse_fn(v_np), dtype=float)
                y_eval = np.asarray(y_val_orig, dtype=float)
            else:
                pred_eval = np.asarray(v_np, dtype=float)
                y_eval = np.asarray(y_va, dtype=float)

            if not np.all(np.isfinite(pred_eval)):
                history["diverged"] = True
                break

            v_rmse = float(root_mean_squared_error(y_eval, pred_eval))
            v_mae = float(mean_absolute_error(y_eval, pred_eval))
            v_r2 = float(r2_score(y_eval, pred_eval))

            history["train_loss"].append(round(float(tr_loss), 6))
            history["val_loss"].append(round(float(v_loss), 6))
            history["val_rmse"].append(round(v_rmse, 6))
            history["val_mae"].append(round(v_mae, 6))
            history["val_r2"].append(round(v_r2, 6))
            history["stopped_epoch"] = epoch_idx

            if v_rmse < best_rmse:
                best_rmse = v_rmse
                history["best_epoch"] = epoch_idx
                history["best_val_rmse"] = round(v_rmse, 6)
                best_state = copy.deepcopy(self._model.state_dict())
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    break

        if eval_set is not None:
            self._model.load_state_dict(best_state)
        self.history_ = history
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch

        self._model.eval()
        with torch.no_grad():
            tx = torch.tensor(X, dtype=torch.float32, device=self._device)
            return self._model(tx).squeeze(-1).cpu().numpy()


class LinearSGDRegressor(_TorchTabularBase):
    """PyTorch Linear Regression trained via Mini-batch SGD with early stopping."""

    def __init__(
        self,
        lr: float = 0.01,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
        batch_size: int = 256,
        epochs: int = 200,
        patience: int = 20,
        loss: str = "mse",
        grad_clip: float | None = 5.0,
        random_state: int = RANDOM_SEED,
    ) -> None:
        super().__init__(
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            loss=loss,
            grad_clip=grad_clip,
            random_state=random_state,
            use_scheduler=False,
            drop_last_if_single=False,
        )
        self.momentum = float(momentum)

    def _build_module(self, input_dim: int) -> Any:
        import torch.nn as nn

        return nn.Linear(input_dim, 1)

    def _build_optimizer(self) -> Any:
        import torch

        return torch.optim.SGD(
            self._model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            momentum=self.momentum,
        )


@register_model("linear_sgd", needs_scaling=True, supports_nans=False, accepts_val=True)
def build_linear_sgd_regressor(
    random_state: int = RANDOM_SEED,
    **kwargs: Any,
) -> LinearSGDRegressor:
    """Build LinearSGDRegressor wrapper."""
    return LinearSGDRegressor(random_state=random_state, **kwargs)


def build_tabular_mlp(
    input_dim: int,
    hidden_dims: tuple[int, ...] = (64,),
    dropout: float = 0.0,
    use_batchnorm: bool = False,
    activation: str = "relu",
) -> Any:
    """Build a configurable PyTorch Tabular Regression MLP."""
    try:
        import torch.nn as nn
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for build_tabular_mlp(). Install via `pip install torch`."
        ) from exc

    act_key = activation.strip().lower()
    if act_key not in ("relu", "silu"):
        raise ValueError(f"Unsupported activation '{activation}'. Expected 'relu' or 'silu'.")

    layers: list[nn.Module] = []
    in_features = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(in_features, hidden_dim))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU() if act_key == "relu" else nn.SiLU())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        in_features = hidden_dim
    layers.append(nn.Linear(in_features, 1))
    return nn.Sequential(*layers)


class TabularMLPRegressor(_TorchTabularBase):
    """Configurable PyTorch Tabular MLP regressor with Adam/AdamW, early stopping & history."""

    def __init__(
        self,
        hidden_dims: tuple[int, ...] | list[int] | int | str = (64,),
        use_batchnorm: bool = False,
        activation: str = "relu",
        dropout: float = 0.0,
        optimizer: str = "adam",
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int = 256,
        epochs: int = 150,
        patience: int = 15,
        loss: str = "mse",
        use_scheduler: bool = False,
        grad_clip: float | None = 5.0,
        random_state: int = RANDOM_SEED,
    ) -> None:
        self.hidden_dims = _normalize_hidden_dims(hidden_dims)
        self.use_batchnorm = bool(use_batchnorm)
        self.activation = str(activation)
        self.dropout = float(dropout)
        self.optimizer = str(optimizer)
        super().__init__(
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            loss=loss,
            grad_clip=grad_clip,
            random_state=random_state,
            use_scheduler=use_scheduler,
            drop_last_if_single=self.use_batchnorm,
        )

    def _build_module(self, input_dim: int) -> Any:
        return build_tabular_mlp(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
            use_batchnorm=self.use_batchnorm,
            activation=self.activation,
        )

    def _build_optimizer(self) -> Any:
        import torch

        opt_key = self.optimizer.strip().lower()
        if opt_key == "adam":
            return torch.optim.Adam(
                self._model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        if opt_key == "adamw":
            return torch.optim.AdamW(
                self._model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        raise ValueError(f"Unsupported optimizer '{self.optimizer}'. Expected 'adam' or 'adamw'.")


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
