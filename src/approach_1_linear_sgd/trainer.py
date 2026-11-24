"""Deterministic CPU SGD, validation early stopping, and atomic checkpoints."""

import csv
import os
import random
import time
import uuid
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..evaluate import evaluate_predictions


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def score(y, pred):
    return evaluate_predictions("linear_sgd", y.reshape(-1), pred.reshape(-1),
                                full_precision=True)


def train(model, X_train, y_train, X_val, y_val, config, artifact_dir, run_id):
    root = Path(artifact_dir)
    checkpoint = root / "checkpoints" / f"{run_id}_best.pt"
    curve = root / "curves" / f"{run_id}_history.csv"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    curve.parent.mkdir(parents=True, exist_ok=True)
    xt, yt, xv = map(torch.from_numpy, (X_train, y_train, X_val))
    loader = DataLoader(TensorDataset(xt, yt), batch_size=config.batch_size,
                        shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(config.seed))
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr,
                                weight_decay=config.weight_decay, momentum=0.0)
    best, best_epoch, stale = float("inf"), 0, 0
    history = []
    status = "completed"
    started = time.perf_counter()
    fields = ["epoch", "train_mse", "train_rmse", "val_rmse", "val_mae",
              "val_r2", "val_within_10pct", "train_val_gap", "epoch_sec"]
    with curve.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        fh.flush()
        for epoch in range(1, config.max_epochs + 1):
            tick = time.perf_counter()
            model.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                if not torch.isfinite(loss):
                    status = "diverged"
                    break
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                train_pred = model(xt).numpy().reshape(-1)
                val_pred = model(xv).numpy().reshape(-1)
            if not (np.isfinite(train_pred).all() and np.isfinite(val_pred).all()):
                status = "diverged"
            entry = dict.fromkeys(["train_mse", "train_rmse", "val_rmse", "val_mae",
                                   "val_r2", "val_within_10pct", "train_val_gap"])
            if status != "diverged":
                tm, vm = score(y_train, train_pred), score(y_val, val_pred)
                entry.update(train_mse=tm["rmse"] ** 2, train_rmse=tm["rmse"],
                             val_rmse=vm["rmse"], val_mae=vm["mae"], val_r2=vm["r2"],
                             val_within_10pct=vm["within_10pct"],
                             train_val_gap=vm["rmse"] - tm["rmse"])
                if vm["rmse"] < best - 1e-6:
                    best, best_epoch, stale = vm["rmse"], epoch, 0
                    tmp = checkpoint.with_suffix(f".tmp.{uuid.uuid4().hex}")
                    try:
                        torch.save(dict(state_dict=model.state_dict(), input_dim=X_train.shape[1],
                                        best_epoch=epoch, best_val_rmse=best), tmp)
                        os.replace(tmp, checkpoint)
                    finally:
                        tmp.unlink(missing_ok=True)
                else:
                    stale += 1
            row = dict(epoch=epoch, **entry, epoch_sec=time.perf_counter() - tick)
            history.append(row)
            writer.writerow(row)
            fh.flush()
            os.fsync(fh.fileno())
            if status == "diverged":
                print(f"[{run_id}] epoch={epoch} status=diverged best_epoch={best_epoch}", flush=True)
            else:
                print(f"[{run_id}] epoch={epoch}/{config.max_epochs} "
                      f"train_rmse={tm['rmse']:.6f} val_rmse={vm['rmse']:.6f} "
                      f"best_epoch={best_epoch} epoch_sec={row['epoch_sec']:.3f}", flush=True)
            if status == "diverged" or stale >= config.patience:
                break
    total = time.perf_counter() - started
    tm = vm = None
    # Never reload a checkpoint left by a different invocation.
    if best_epoch:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(saved["state_dict"])
        model.eval()
        with torch.no_grad():
            tm, vm = score(y_train, model(xt).numpy()), score(y_val, model(xv).numpy())
        assert abs(vm["rmse"] - best) < 1e-5
    return dict(status=status, best_epoch=best_epoch,
                best_val_rmse=best if best_epoch else None,
                total_train_sec=total, sec_per_epoch=total / len(history),
                train_metrics=tm, val_metrics=vm,
                train_val_gap=vm["rmse"] - tm["rmse"] if vm else None,
                checkpoint_path=str(checkpoint) if best_epoch else None,
                curve_path=str(curve), epochs_ran=len(history))
