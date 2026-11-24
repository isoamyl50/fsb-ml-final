"""Approach-specific orchestration; preserves the shared baseline execution path."""

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..config import PROJECT_ROOT, TARGET_COL, EXPERIMENTS_SPLITS_PATH
from ..features import build_feature_recipe
from ..models import get_model_spec
from ..runner import _atomic_write_json, _detect_git_state
from ..splits import build_repost_group_key, load_frozen_splits, load_raw_records
from .trainer import score
from .pipeline import LinearSGDPipeline
from .logging import publish_shared_run, run_lock_path
from ..experiment_io import file_lock
from ..leaderboard import build_leaderboard as build_shared_leaderboard

APPROACH = "approach_1_linear_sgd"


def build_leaderboard(root):
    with file_lock(root / "leaderboard.lock", timeout=30):
        return _build_leaderboard(root)


def _build_leaderboard(root):
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in (root / "runs").glob("*.json")]
    runs.sort(key=lambda r: r["val_metrics"]["rmse"] if r["val_metrics"] else float("inf"))
    lines = ["# Cách 1 — Linear Regression / Mini-batch SGD", "",
             "Đơn vị: tỷ VND. Xếp hạng theo Val RMSE; Test khóa mặc định.", "",
             "Mốc tham chiếu HGB + P4: RMSE 1.1709, MAE 0.8496, R² 0.7210, Trong ±10% 45.73%.", "",
             "| Run | Status | RMSE (tỷ) | MAE (tỷ) | R² | Trong ±10% | Train-Val gap | Giây/epoch | Tổng giây | Best epoch |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in runs:
        m, t = r["val_metrics"], r["timing"]
        values = (f"{m['rmse']:.6f} | {m['mae']:.6f} | {m['r2']:.6f} | "
                  f"{m['within_10pct']:.2%} | {r['train_val_gap']:.6f}" if m else "— | — | — | — | —")
        lines.append(f"| {r['run_id']} | {r['status']} | {values} | {t['sec_per_epoch']:.4f} | "
                     f"{t['total_train_sec']:.3f} | {r['best_epoch']} |")
    path = root / "leaderboard.md"
    tmp = path.with_suffix(f".tmp.{uuid.uuid4().hex}")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def run_linear_sgd(*, author, features, target, params, cpu_config, eval_test,
                   seed, notes, csv_path, splits_path, run_id):
    run_id = run_id or f"linear_sgd_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"
    with file_lock(run_lock_path(PROJECT_ROOT, run_id)):
        return _run_linear_sgd(author=author, features=features, target=target, params=params,
                               cpu_config=cpu_config, eval_test=eval_test, seed=seed, notes=notes,
                               csv_path=csv_path, splits_path=splits_path, run_id=run_id)


def _run_linear_sgd(*, author, features, target, params, cpu_config, eval_test,
                    seed, notes, csv_path, splits_path, run_id):
    if features not in ("P4", "P4_geo_priors") or target not in ("raw", "raw_price"):
        raise ValueError("Approach 1 requires P4_geo_priors and raw Price in billion VND")
    if seed != 42 or Path(splits_path).resolve() != EXPERIMENTS_SPLITS_PATH.resolve():
        raise ValueError("Approach 1 requires the shared splits_v1.json and seed=42")
    frozen = json.loads(Path(splits_path).read_text(encoding="utf-8"))
    if frozen.get("version") != "splits_v1" or frozen.get("seed") != 42:
        raise ValueError("Frozen split metadata must be splits_v1 with seed=42")
    root = PROJECT_ROOT / "artifacts" / APPROACH
    shared_path = PROJECT_ROOT / "experiments" / "runs" / f"{run_id}.json"
    if shared_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing shared run {run_id}")
    for directory, filename in (("runs", f"{run_id}.json"), ("curves", f"{run_id}_history.csv"),
                                ("checkpoints", f"{run_id}_best.pt"),
                                ("pipelines", f"{run_id}_pipeline.joblib")):
        if (root / directory / filename).exists():
            raise FileExistsError(f"Refusing to overwrite existing run {run_id}")
    rows = load_raw_records(csv_path)
    ti, vi, si = load_frozen_splits(splits_path, rows)
    sets = [set(idx.tolist()) for idx in (ti, vi, si)]
    assert all(len(indices) == len(unique) for indices, unique in zip((ti, vi, si), sets))
    assert len(set.union(*sets)) == len(rows)
    groups = [{build_repost_group_key(rows[int(i)]) for i in idx} for idx in (ti, vi, si)]
    for i, j in ((0, 1), (0, 2), (1, 2)):
        assert not sets[i] & sets[j], "Overlapping split indices"
        assert not groups[i] & groups[j], "Repost group leakage"
    train_rows, val_rows = [rows[int(i)] for i in ti], [rows[int(i)] for i in vi]
    yp = np.array([float(r[TARGET_COL]) for r in train_rows])
    yv = np.array([float(r[TARGET_COL]) for r in val_rows])
    recipe = build_feature_recipe("P4_geo_priors", scale_numeric=False, preserve_nans=False)
    xt, xv = recipe.fit_transform(train_rows, yp), recipe.transform(val_rows)
    options = dict(params or {})
    options.update(cpu_config or {})
    reg = get_model_spec("linear_sgd").builder(random_state=seed, **options)
    summary = reg.fit(xt, yp, xv, yv, root, run_id)
    vm = score(yv, reg.predict(xv)) if summary["best_epoch"] else None
    if vm:
        assert abs(vm["rmse"] - summary["best_val_rmse"]) < 1e-5
    pipeline_path = None
    if vm:
        pipeline_path = LinearSGDPipeline(recipe, reg).save(
            root / "pipelines" / f"{run_id}_pipeline.joblib")
        restored = LinearSGDPipeline.load(pipeline_path)
        np.testing.assert_allclose(restored.predict(val_rows), reg.predict(xv), rtol=0, atol=1e-6)
    test_metrics = None
    if eval_test and vm:
        test_rows = [rows[int(i)] for i in si]
        test_metrics = score(np.array([float(r[TARGET_COL]) for r in test_rows]),
                             reg.predict(recipe.transform(test_rows)))
    commit, dirty = _detect_git_state()
    config = vars(reg.config).copy()
    cpu = {key: config.pop(key) for key in ("num_threads", "num_workers", "dtype")}
    config.pop("seed")
    payload = dict(run_id=run_id, approach=APPROACH, model_name="linear_sgd",
                   status=summary["status"], seed=seed, git_commit=commit, git_dirty=dirty,
                   timestamp_utc=datetime.now(timezone.utc).isoformat(), author=author,
                   hardware="CPU", cpu_count=os.cpu_count(), feature_version="P4",
                   input_dim=int(xt.shape[1]), cpu_config=cpu, hyperparameters=config,
                   timing={k: summary[k] for k in ("sec_per_epoch", "total_train_sec")},
                   best_epoch=summary["best_epoch"], best_val_rmse=summary["best_val_rmse"],
                   train_metrics=summary["train_metrics"], val_metrics=vm,
                   train_val_gap=summary["train_val_gap"], epochs_ran=summary["epochs_ran"],
                   scaler_n_samples_seen=summary["scaler_n_samples_seen"],
                   artifacts={k: summary[k] for k in ("checkpoint_path", "curve_path")},
                   target={"name": "raw", "unit": "billion_vnd"},
                   features={"name": "P4_geo_priors", "n_features": int(xt.shape[1]), "scaled": True},
                   model={"name": "linear_sgd", "params": config},
                   splits=dict(path=str(splits_path), n_train=len(ti), n_val=len(vi), n_test=len(si)),
                   metrics={"val": vm, "test": test_metrics}, eval_test=eval_test,
                   train_seconds=summary["total_train_sec"], notes=notes)
    payload["artifacts"]["pipeline_path"] = (
        pipeline_path.relative_to(root).as_posix() if pipeline_path else None)
    path = root / "runs" / f"{run_id}.json"
    _atomic_write_json(path, payload)
    shared_path = publish_shared_run(payload, PROJECT_ROOT)
    build_leaderboard(root)
    build_shared_leaderboard(runs_dir=shared_path.parent,
                             md_path=PROJECT_ROOT / "experiments" / "leaderboard.md",
                             csv_path=PROJECT_ROOT / "experiments" / "leaderboard.csv")
    payload["saved_path"] = str(path)
    payload["shared_saved_path"] = str(shared_path)
    return payload
