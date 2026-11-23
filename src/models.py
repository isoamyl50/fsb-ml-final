"""Pluggable model registry for linear, tree-based, and PyTorch neural classifiers."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from .config import RANDOM_SEED


@dataclass(frozen=True)
class ModelSpec:
    """Metadata and factory callable for one registered model family."""

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
    """Decorator registering a model factory function."""

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
    """Return sorted registered model names."""
    return sorted(MODEL_REGISTRY.keys())


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


@register_model("logreg", needs_scaling=True, supports_nans=False, accepts_val=False)
def build_logistic_regression(
    random_state: int = RANDOM_SEED,
    **kwargs: Any,
) -> LogisticRegression:
    """Build multinomial Logistic Regression baseline."""
    params: dict[str, Any] = {
        "max_iter": 1000,
        "C": 1.0,
        "random_state": random_state,
    }
    params.update(kwargs)
    return LogisticRegression(**params)


@register_model("hgb", needs_scaling=False, supports_nans=True, accepts_val=False)
def build_hist_gbdt(
    random_state: int = RANDOM_SEED,
    **kwargs: Any,
) -> HistGradientBoostingClassifier:
    """Build HistGradientBoostingClassifier (supports native NaNs in numeric features)."""
    params: dict[str, Any] = {
        "max_iter": 250,
        "learning_rate": 0.05,
        "early_stopping": True,
        "validation_fraction": 0.15,
        "random_state": random_state,
    }
    params.update(kwargs)
    return HistGradientBoostingClassifier(**params)


@register_model("rf", needs_scaling=False, supports_nans=False, accepts_val=False)
def build_random_forest(
    random_state: int = RANDOM_SEED,
    **kwargs: Any,
) -> RandomForestClassifier:
    """Build RandomForestClassifier baseline."""
    params: dict[str, Any] = {
        "n_estimators": 300,
        "n_jobs": -1,
        "random_state": random_state,
    }
    params.update(kwargs)
    return RandomForestClassifier(**params)


@register_model("lgbm", needs_scaling=False, supports_nans=True, accepts_val=True)
def build_lgbm(
    random_state: int = RANDOM_SEED,
    **kwargs: Any,
) -> Any:
    """Build LightGBM classifier (lazy import so local envs without lightgbm stay clean)."""
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError(
            "lightgbm is required for model='lgbm'. Run on Colab or install lightgbm."
        ) from exc

    params: dict[str, Any] = {
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "subsample_for_bin": 50000,
        "colsample_bytree": 0.8,
        "subsample": 0.8,
        "subsample_freq": 1,
        "random_state": random_state,
        "n_jobs": -1,
        "verbosity": -1,
    }
    params.update(kwargs)
    return lgb.LGBMClassifier(**params)


@register_model("xgb", needs_scaling=False, supports_nans=True, accepts_val=True)
def build_xgb(
    random_state: int = RANDOM_SEED,
    **kwargs: Any,
) -> Any:
    """Build XGBoost classifier (lazy import; auto-uses CUDA hist when GPU is available)."""
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
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "device": device,
        "early_stopping_rounds": 50,
        "eval_metric": "mlogloss",
        "random_state": random_state,
        "n_jobs": -1,
    }
    params.update(kwargs)
    return xgb.XGBClassifier(**params)


def build_tabular_mlp(
    input_dim: int,
    num_classes: int,
    hidden_dims: tuple[int, ...] = (128, 64),
    dropout: float = 0.25,
) -> Any:
    """Build a PyTorch Tabular MLP (BatchNorm + ReLU + Dropout) when PyTorch is installed."""
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
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        )
        in_features = hidden_dim
    layers.append(nn.Linear(in_features, num_classes))
    return nn.Sequential(*layers)


class TabularMLPClassifier:
    """Scikit-learn-style PyTorch Tabular MLP with validation early stopping and history."""

    def __init__(
        self,
        hidden_dims: tuple[int, ...] = (128, 64),
        dropout: float = 0.25,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        epochs: int = 40,
        patience: int = 8,
        random_state: int = RANDOM_SEED,
    ) -> None:
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.random_state = random_state
        self.history_: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "val_f1_macro": [],
        }
        self._model: Any = None
        self._device: str = "cpu"

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> "TabularMLPClassifier":
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

        n_classes = int(np.max(y_train)) + 1
        self._model = build_tabular_mlp(
            input_dim=X_train.shape[1],
            num_classes=n_classes,
            hidden_dims=tuple(self.hidden_dims),
            dropout=self.dropout,
        ).to(self._device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        ds_tr = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        )
        loader_tr = DataLoader(ds_tr, batch_size=self.batch_size, shuffle=True)

        best_score = -1.0
        best_state = copy.deepcopy(self._model.state_dict())
        wait = 0

        for _ in range(self.epochs):
            self._model.train()
            tr_loss_sum = 0.0
            tr_correct = 0
            tr_total = 0
            for xb, yb in loader_tr:
                xb = xb.to(self._device)
                yb = yb.to(self._device)
                optimizer.zero_grad()
                logits = self._model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

                tr_loss_sum += float(loss.item()) * len(yb)
                tr_correct += int((logits.argmax(dim=1) == yb).sum().item())
                tr_total += len(yb)

            scheduler.step()
            self.history_["train_loss"].append(tr_loss_sum / max(tr_total, 1))
            self.history_["train_acc"].append(tr_correct / max(tr_total, 1))

            if eval_set is not None:
                x_va, y_va = eval_set
                self._model.eval()
                with torch.no_grad():
                    vx = torch.tensor(x_va, dtype=torch.float32, device=self._device)
                    vy = torch.tensor(y_va, dtype=torch.long, device=self._device)
                    v_logits = self._model(vx)
                    v_loss = float(criterion(v_logits, vy).item())
                    v_pred = v_logits.argmax(dim=1).cpu().numpy()
                v_acc = float(np.mean(v_pred == y_va))
                v_f1 = float(f1_score(y_va, v_pred, average="macro", zero_division=0))
                self.history_["val_loss"].append(v_loss)
                self.history_["val_acc"].append(v_acc)
                self.history_["val_f1_macro"].append(v_f1)

                if v_f1 > best_score:
                    best_score = v_f1
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
            logits = self._model(tx)
            return logits.argmax(dim=1).cpu().numpy()


@register_model("mlp", needs_scaling=True, supports_nans=False, accepts_val=True)
def build_mlp_classifier(
    random_state: int = RANDOM_SEED,
    **kwargs: Any,
) -> TabularMLPClassifier:
    """Build TabularMLPClassifier wrapper."""
    return TabularMLPClassifier(random_state=random_state, **kwargs)
