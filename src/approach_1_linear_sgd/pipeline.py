"""Portable inference bundle: fitted feature recipe, scaler, and best CPU weights."""

import os
import uuid
from pathlib import Path

import joblib
import numpy as np
import sklearn
import torch

from .model import LinearRegressionModel
from .wrapper import LinearSGDWrapper


class LinearSGDPipeline:
    """Predict prices in billion VND from raw listing dictionaries, without fitting."""

    def __init__(self, recipe, regressor):
        self.recipe = recipe
        self.regressor = regressor

    def predict(self, rows):
        # Labels must never reach a feature transform, even if callers supply them.
        features = self.recipe.transform([{k: v for k, v in row.items() if k != "Price"}
                                          for row in rows])
        return self.regressor.predict(features)

    def save(self, path):
        reg = self.regressor
        if not getattr(reg, "train_summary_", {}).get("best_epoch"):
            raise RuntimeError("Cannot export a pipeline without a valid best checkpoint")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(format_version=1, target_unit="billion_vnd", recipe=self.recipe,
                       scaler=reg.scaler, config=vars(reg.config).copy(),
                       input_dim=reg.model.linear.in_features,
                       state_dict={k: v.detach().cpu().numpy().copy()
                                   for k, v in reg.model.state_dict().items()},
                       train_summary=reg.train_summary_.copy(),
                       versions=dict(torch=torch.__version__, sklearn=sklearn.__version__,
                                     numpy=np.__version__))
        tmp = path.with_suffix(f".tmp.{uuid.uuid4().hex}")
        try:
            joblib.dump(payload, tmp, compress=3)
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)
        return path

    @classmethod
    def load(cls, path):
        """Load a trusted locally generated joblib bundle; no dataset/refitting required."""
        payload = joblib.load(path)
        if payload.get("format_version") != 1 or payload.get("target_unit") != "billion_vnd":
            raise ValueError("Unsupported linear SGD pipeline format or target unit")
        reg = LinearSGDWrapper(**payload["config"])
        torch.set_num_threads(reg.config.num_threads)
        reg.scaler = payload["scaler"]
        reg.model = LinearRegressionModel(payload["input_dim"]).cpu().float()
        reg.model.load_state_dict({k: torch.from_numpy(v) for k, v in payload["state_dict"].items()})
        reg.model.eval()
        reg.train_summary_ = payload["train_summary"]
        if reg.scaler.n_features_in_ != payload["input_dim"]:
            raise ValueError("Scaler and model feature dimensions do not match")
        return cls(payload["recipe"], reg)
