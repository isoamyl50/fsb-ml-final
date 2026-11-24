"""Train-only scaling and strict float32/shape contracts for CPU SGD."""

from types import SimpleNamespace

import numpy as np
import torch
from scipy import sparse
from sklearn.preprocessing import StandardScaler

from .model import LinearRegressionModel
from .trainer import set_seed, train


class LinearSGDWrapper:
    def __init__(self, **hyperparams):
        defaults = dict(seed=42, num_threads=4, num_workers=0, dtype="float32",
                        batch_size=256, lr=0.01, weight_decay=0.0, momentum=0.0,
                        loss="mse", optimizer="sgd", scaler="standard",
                        max_epochs=200, patience=20)
        unknown = hyperparams.keys() - defaults.keys()
        if unknown:
            raise ValueError(f"Unknown hyperparameters: {sorted(unknown)}")
        defaults.update(hyperparams)
        for key, expected in dict(num_workers=0, dtype="float32", momentum=0.0,
                                  loss="mse", optimizer="sgd", scaler="standard").items():
            if defaults[key] != expected:
                raise ValueError(f"{key} must be {expected!r}")
        for key in ("num_threads", "batch_size", "max_epochs", "patience"):
            if not isinstance(defaults[key], int) or defaults[key] <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if not np.isfinite(defaults["lr"]) or defaults["lr"] <= 0:
            raise ValueError("lr must be finite and positive")
        if not np.isfinite(defaults["weight_decay"]) or defaults["weight_decay"] < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        self.config = SimpleNamespace(**defaults)
        self.scaler = StandardScaler()

    @staticmethod
    def _dense(X):
        return np.asarray(X.toarray() if sparse.issparse(X) else X, dtype=np.float32)

    def fit(self, X_train, y_train, X_val, y_val, artifact_dir, run_id):
        X_train, X_val = self._dense(X_train), self._dense(X_val)
        y_train = np.asarray(y_train, dtype=np.float32).reshape(-1, 1)
        y_val = np.asarray(y_val, dtype=np.float32).reshape(-1, 1)
        for name, value in dict(X_train=X_train, X_val=X_val,
                                y_train=y_train, y_val=y_val).items():
            assert np.isfinite(value).all(), f"{name} contains NaN or Inf"
        assert X_train.ndim == X_val.ndim == 2
        assert X_train.shape[1] == X_val.shape[1]
        assert len(X_train) == len(y_train) and len(X_val) == len(y_val)
        assert len(X_train) > 1 and len(X_val) > 1
        X_train = self.scaler.fit_transform(X_train).astype(np.float32)
        X_val = self.scaler.transform(X_val).astype(np.float32)
        assert self.scaler.n_samples_seen_ == len(X_train)
        assert np.isfinite(X_train).all() and np.isfinite(X_val).all()
        set_seed(self.config.seed)
        torch.set_num_threads(self.config.num_threads)
        self.model = LinearRegressionModel(X_train.shape[1]).cpu().float()
        self.train_summary_ = train(self.model, X_train, y_train, X_val, y_val,
                                    self.config, artifact_dir, run_id)
        self.train_summary_["scaler_n_samples_seen"] = int(self.scaler.n_samples_seen_)
        return self.train_summary_

    def predict(self, X):
        if not self.train_summary_["best_epoch"]:
            raise RuntimeError("Training diverged before producing a valid checkpoint")
        X = self._dense(X)
        assert np.isfinite(X).all(), "X contains NaN or Inf"
        X = self.scaler.transform(X).astype(np.float32)
        assert np.isfinite(X).all(), "Scaled X contains NaN or Inf"
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.from_numpy(X)).numpy().reshape(-1)
