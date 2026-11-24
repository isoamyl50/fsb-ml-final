#!/usr/bin/env python3
"""Unattended Colab T4 regression experiment suite for Linear SGD, MLP Adam, and tree refs.

Designed to run via `colab exec -f` (IPython does not define `__file__`) after
uploading `/content/colab_bundle.tar.gz`. Evaluates Validation strictly across all
tuning runs, locks winning configurations on Val RMSE, evaluates Test ONCE, computes
segment breakdowns and Val permutation importance, and writes `/content/colab_results.tar.gz`.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

import numpy as np

BUNDLE_PATH = Path("/content/colab_bundle.tar.gz")
CONTENT_PROJECT = Path("/content/project")
RESULTS_TAR = Path("/content/colab_results.tar.gz")

if BUNDLE_PATH.is_file():
    for stale_dir in (
        CONTENT_PROJECT / "src",
        CONTENT_PROJECT / "scripts",
        CONTENT_PROJECT / "experiments",
        CONTENT_PROJECT / "artifacts",
    ):
        if stale_dir.exists():
            shutil.rmtree(stale_dir)
    CONTENT_PROJECT.mkdir(parents=True, exist_ok=True)
    with tarfile.open(BUNDLE_PATH, "r:gz") as tf:
        tf.extractall(CONTENT_PROJECT)
    PROJECT_DIR = CONTENT_PROJECT
elif (CONTENT_PROJECT / "src").is_dir():
    PROJECT_DIR = CONTENT_PROJECT
else:
    PROJECT_DIR = Path.cwd()

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
os.chdir(PROJECT_DIR)

import torch

from src.evaluate import evaluate_predictions
from src.preprocessing import parse_address
from src.runner import run_and_log
from src.splits import load_frozen_splits, load_raw_records
from src.targets import get_target_transform

CSV_PATH = PROJECT_DIR / "data" / "raw" / "vietnam_housing_dataset.csv"
SPLITS_PATH = PROJECT_DIR / "experiments" / "splits" / "splits_v1.json"
RUNS_DIR = PROJECT_DIR / "experiments" / "runs"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"

RUNS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def _log(msg: str) -> None:
    print(f"[overnight] {msg}", flush=True)


def _run_val(
    tag: str,
    features: str,
    model: str,
    target: str = "raw",
    params: dict[str, Any] | None = None,
    seed: int = 42,
    notes: str = "",
) -> dict[str, Any]:
    t0 = time.perf_counter()
    res = run_and_log(
        author="overnight_colab",
        features=features,
        model=model,
        target=target,
        params=params or {},
        outlier_filter=False,
        eval_test=False,
        seed=seed,
        notes=f"[{tag}] {notes}".strip(),
        save_run=True,
        mirror_to_drive=False,
        csv_path=CSV_PATH,
        splits_path=SPLITS_PATH,
        runs_dir=RUNS_DIR,
    )
    vm = res["metrics"]["val"]
    hist = res.get("history", {})
    ep_str = ""
    if hist:
        ep_str = f" best_ep={hist.get('best_epoch')}/{hist.get('stopped_epoch')} div={hist.get('diverged')}"
    _log(
        f"{tag:10s} | {model:10s} | {features:13s} | {target:4s} | "
        f"RMSE={vm['rmse']:.4f} MAE={vm['mae']:.4f} R2={vm['r2']:.4f} ±10%={vm['within_10pct']:.4f} "
        f"| {time.perf_counter()-t0:.2f}s{ep_str}"
    )
    return {
        "tag": tag,
        "run_id": res["run_id"],
        "features": features,
        "model": model,
        "target": target,
        "params": dict(params or {}),
        "seed": seed,
        "val": vm,
        "history_meta": {
            k: hist.get(k)
            for k in ("best_epoch", "stopped_epoch", "best_val_rmse", "diverged", "n_params")
        }
        if hist
        else {},
        "train_seconds": res["train_seconds"],
    }


def _perm_importance(
    reg: Any,
    X_val: np.ndarray,
    val_prices: np.ndarray,
    val_areas: np.ndarray,
    feature_names: list[str],
    target_name: str,
    n_repeats: int = 3,
    top_k: int = 15,
    seed: int = 42,
) -> list[dict[str, Any]]:
    tfm = get_target_transform(target_name)
    base_pred = tfm.inverse(reg.predict(X_val), val_areas)
    base_rmse = float(np.sqrt(np.mean((val_prices - base_pred) ** 2)))
    rng = np.random.default_rng(seed)

    # Zero-variance features cannot change predictions, skip extra forward passes safely
    X_work = X_val.copy()
    importances: list[dict[str, Any]] = []
    n_cols = X_work.shape[1]
    for col_idx in range(n_cols):
        col_orig = X_work[:, col_idx].copy()
        if float(np.std(col_orig)) < 1e-12:
            continue
        deltas: list[float] = []
        for _ in range(n_repeats):
            X_work[:, col_idx] = rng.permutation(col_orig)
            perm_pred = tfm.inverse(reg.predict(X_work), val_areas)
            perm_rmse = float(np.sqrt(np.mean((val_prices - perm_pred) ** 2)))
            deltas.append(perm_rmse - base_rmse)
        X_work[:, col_idx] = col_orig
        importances.append(
            {
                "feature": feature_names[col_idx] if col_idx < len(feature_names) else f"f_{col_idx}",
                "delta_rmse_mean": round(float(np.mean(deltas)), 5),
                "delta_rmse_std": round(float(np.std(deltas)), 5),
            }
        )

    importances.sort(key=lambda item: item["delta_rmse_mean"], reverse=True)
    return importances[:top_k]


def _compute_segment_report(
    test_rows: list[dict[str, str]],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_label: str,
) -> list[dict[str, Any]]:
    cities = [parse_address(r.get("Address"))["city"] for r in test_rows]
    masks = [
        ("Giá thấp (< 4 tỷ)", y_true < 4.0),
        ("Giá trung bình (4–7 tỷ)", (y_true >= 4.0) & (y_true < 7.0)),
        ("Giá cao (≥ 7 tỷ)", y_true >= 7.0),
        ("TP. Hồ Chí Minh", np.array([c == "Hồ Chí Minh" for c in cities], dtype=bool)),
        ("Hà Nội", np.array([c == "Hà Nội" for c in cities], dtype=bool)),
        (
            "Tỉnh/TP khác",
            np.array([c not in ("Hồ Chí Minh", "Hà Nội") for c in cities], dtype=bool),
        ),
    ]
    rows_out: list[dict[str, Any]] = []
    for seg_name, mask in masks:
        cnt = int(np.sum(mask))
        if cnt == 0:
            continue
        m = evaluate_predictions(model_label, y_true[mask], y_pred[mask], export_to_slides=False)
        rows_out.append(
            {
                "segment": seg_name,
                "count": cnt,
                "rmse": m["rmse"],
                "mae": m["mae"],
                "medae": m["medae"],
                "r2": m["r2"],
                "mape": m["mape"],
                "within_10pct": m["within_10pct"],
                "within_20pct": m["within_20pct"],
                "within_1ty": m["within_1ty"],
            }
        )
    return rows_out


def main() -> int:
    _log(f"PROJECT_DIR={PROJECT_DIR}")
    _log(
        f"Python={sys.version.split()[0]} | Torch={torch.__version__} | "
        f"CUDA={torch.cuda.is_available()} | "
        f"GPU={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
    )
    assert CSV_PATH.is_file(), f"Missing dataset at {CSV_PATH}"
    assert SPLITS_PATH.is_file(), f"Missing splits at {SPLITS_PATH}"

    suite_t0 = time.perf_counter()

    # ------------------------------------------------------------------
    # Phase A: Preprocessing Ablation (P0 -> P2 -> P4_compact / P4_geo_priors)
    # ------------------------------------------------------------------
    _log("=== Phase A: Preprocessing Ablation ===")
    lin_base_params = {
        "batch_size": 256,
        "lr": 0.01,
        "weight_decay": 0.0,
        "epochs": 200,
        "patience": 20,
        "loss": "mse",
        "grad_clip": 5.0,
    }
    mlp_base_params = {
        "hidden_dims": (64,),
        "activation": "relu",
        "use_batchnorm": False,
        "dropout": 0.0,
        "optimizer": "adam",
        "loss": "mse",
        "batch_size": 256,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "epochs": 150,
        "patience": 15,
        "use_scheduler": False,
        "grad_clip": 5.0,
    }

    ablation: dict[str, dict[str, dict[str, Any]]] = {"linear_sgd": {}, "mlp": {}, "hgb": {}}
    for feat in ("P0_numeric", "P2_domain", "P4_compact", "P4_geo_priors"):
        ablation["linear_sgd"][feat] = _run_val(
            tag=f"ABL_LIN_{feat}",
            features=feat,
            model="linear_sgd",
            params=lin_base_params,
            notes=f"Preprocessing ablation Linear SGD on {feat}",
        )
        ablation["mlp"][feat] = _run_val(
            tag=f"ABL_MLP_{feat}",
            features=feat,
            model="mlp",
            params=mlp_base_params,
            notes=f"Preprocessing ablation MLP Adam on {feat}",
        )

    for feat in ("P2_domain", "P4_compact"):
        ablation["hgb"][feat] = _run_val(
            tag=f"ABL_HGB_{feat}",
            features=feat,
            model="hgb",
            params={},
            notes=f"Preprocessing ablation HGB ref on {feat}",
        )

    # Also evaluate HGB P0/P4_geo_priors without duplicate save_run if desired (or record in summary)
    for feat in ("P0_numeric", "P4_geo_priors"):
        h_res = run_and_log(
            author="overnight_colab",
            features=feat,
            model="hgb",
            target="raw",
            params={},
            eval_test=False,
            seed=42,
            notes=f"[ABL_HGB_{feat}] Reference check",
            save_run=False,
            mirror_to_drive=False,
            csv_path=CSV_PATH,
            splits_path=SPLITS_PATH,
            runs_dir=RUNS_DIR,
        )
        ablation["hgb"][feat] = {
            "tag": f"ABL_HGB_{feat}",
            "run_id": None,
            "features": feat,
            "model": "hgb",
            "target": "raw",
            "params": {},
            "seed": 42,
            "val": h_res["metrics"]["val"],
            "history_meta": {},
            "train_seconds": h_res["train_seconds"],
        }

    best_p4_linear = min(
        ("P4_geo_priors", "P4_compact"),
        key=lambda f: ablation["linear_sgd"][f]["val"]["rmse"],
    )
    best_p4_mlp = min(
        ("P4_geo_priors", "P4_compact"),
        key=lambda f: ablation["mlp"][f]["val"]["rmse"],
    )
    _log(f"Selected P4 for Linear SGD: {best_p4_linear} | for MLP Adam: {best_p4_mlp}")

    # ------------------------------------------------------------------
    # Phase B: Track 1 — Linear Regression + Mini-batch SGD (L01..L05)
    # ------------------------------------------------------------------
    _log("=== Phase B: Track 1 — Linear SGD Tuning (L01–L05) ===")
    track1: list[dict[str, Any]] = []
    l01 = dict(ablation["linear_sgd"][best_p4_linear])
    l01["tag"] = "L01"
    track1.append(l01)

    l02 = _run_val(
        "L02", best_p4_linear, "linear_sgd",
        params={**lin_base_params, "batch_size": 128, "lr": 0.01},
        notes="Track1 L02 batch_size=128 lr=0.01",
    )
    l03 = _run_val(
        "L03", best_p4_linear, "linear_sgd",
        params={**lin_base_params, "batch_size": 512, "lr": 0.01},
        notes="Track1 L03 batch_size=512 lr=0.01",
    )
    l04a = _run_val(
        "L04a", best_p4_linear, "linear_sgd",
        params={**lin_base_params, "batch_size": 256, "lr": 0.001},
        notes="Track1 L04a batch_size=256 lr=0.001",
    )
    l04b = _run_val(
        "L04b", best_p4_linear, "linear_sgd",
        params={**lin_base_params, "batch_size": 256, "lr": 0.003},
        notes="Track1 L04b batch_size=256 lr=0.003",
    )
    l04c = _run_val(
        "L04c", best_p4_linear, "linear_sgd",
        params={**lin_base_params, "batch_size": 256, "lr": 0.05},
        notes="Track1 L04c batch_size=256 lr=0.05 (grad_clip=5.0)",
    )
    track1.extend([l02, l03, l04a, l04b, l04c])

    best_l_pre_wd = min(track1, key=lambda r: r["val"]["rmse"])
    l05 = _run_val(
        "L05",
        best_p4_linear,
        "linear_sgd",
        params={**best_l_pre_wd["params"], "weight_decay": 1e-4},
        notes=f"Track1 L05 weight_decay=1e-4 on {best_l_pre_wd['tag']}",
    )
    track1.append(l05)
    best_linear = min(track1, key=lambda r: r["val"]["rmse"])
    _log(
        f"Winning Linear SGD: {best_linear['tag']} ({best_linear['features']}, "
        f"bs={best_linear['params']['batch_size']}, lr={best_linear['params']['lr']}, "
        f"wd={best_linear['params']['weight_decay']}) Val RMSE={best_linear['val']['rmse']:.4f}"
    )

    # ------------------------------------------------------------------
    # Phase C: Track 2 — MLP + Backprop + Adam (N01..N06 + Multi-seed)
    # ------------------------------------------------------------------
    _log("=== Phase C: Track 2 — MLP + Adam Tuning (N01–N06) ===")
    track2: list[dict[str, Any]] = []
    n01 = dict(ablation["mlp"][best_p4_mlp])
    n01["tag"] = "N01"
    track2.append(n01)

    n02a = _run_val(
        "N02a", best_p4_mlp, "mlp",
        params={**mlp_base_params, "hidden_dims": (32,), "use_batchnorm": False},
        notes="Track2 N02a 1-layer (32,)",
    )
    n02b = _run_val(
        "N02b", best_p4_mlp, "mlp",
        params={**mlp_base_params, "hidden_dims": (128,), "use_batchnorm": False},
        notes="Track2 N02b 1-layer (128,)",
    )
    n02c = _run_val(
        "N02c", best_p4_mlp, "mlp",
        params={**mlp_base_params, "hidden_dims": (128, 64), "use_batchnorm": False},
        notes="Track2 N02c 2-layer (128, 64) no BN",
    )
    n02d = _run_val(
        "N02d", best_p4_mlp, "mlp",
        params={**mlp_base_params, "hidden_dims": (128, 64), "use_batchnorm": True},
        notes="Track2 N02d 2-layer (128, 64) + BatchNorm1d",
    )
    arch_candidates = [n01, n02a, n02b, n02c, n02d]
    track2.extend([n02a, n02b, n02c, n02d])
    best_arch_run = min(arch_candidates, key=lambda r: r["val"]["rmse"])
    cur_mlp = dict(best_arch_run["params"])
    cur_best_run = best_arch_run

    n03a = _run_val(
        "N03a", best_p4_mlp, "mlp",
        params={**cur_mlp, "batch_size": 128},
        notes=f"Track2 N03a batch_size=128 on {best_arch_run['tag']}",
    )
    n03b = _run_val(
        "N03b", best_p4_mlp, "mlp",
        params={**cur_mlp, "batch_size": 512},
        notes=f"Track2 N03b batch_size=512 on {best_arch_run['tag']}",
    )
    track2.extend([n03a, n03b])
    cur_best_run = min([cur_best_run, n03a, n03b], key=lambda r: r["val"]["rmse"])
    cur_mlp = dict(cur_best_run["params"])

    n04 = _run_val(
        "N04", best_p4_mlp, "mlp",
        params={**cur_mlp, "lr": 3e-4},
        notes=f"Track2 N04 lr=3e-4 on {cur_best_run['tag']}",
    )
    track2.append(n04)
    cur_best_run = min([cur_best_run, n04], key=lambda r: r["val"]["rmse"])
    cur_mlp = dict(cur_best_run["params"])

    n05a = _run_val(
        "N05a", best_p4_mlp, "mlp",
        params={**cur_mlp, "dropout": 0.1},
        notes=f"Track2 N05a dropout=0.1 on {cur_best_run['tag']}",
    )
    n05b = _run_val(
        "N05b", best_p4_mlp, "mlp",
        params={**cur_mlp, "dropout": 0.2},
        notes=f"Track2 N05b dropout=0.2 on {cur_best_run['tag']}",
    )
    track2.extend([n05a, n05b])
    cur_best_run = min([cur_best_run, n05a, n05b], key=lambda r: r["val"]["rmse"])
    cur_mlp = dict(cur_best_run["params"])

    n05c = _run_val(
        "N05c", best_p4_mlp, "mlp",
        params={**cur_mlp, "loss": "huber"},
        notes=f"Track2 N05c loss=huber on {cur_best_run['tag']}",
    )
    n05d = _run_val(
        "N05d", best_p4_mlp, "mlp",
        params={**cur_mlp, "loss": "smooth_l1"},
        notes=f"Track2 N05d loss=smooth_l1 on {cur_best_run['tag']}",
    )
    track2.extend([n05c, n05d])
    cur_best_run = min([cur_best_run, n05c, n05d], key=lambda r: r["val"]["rmse"])
    cur_mlp = dict(cur_best_run["params"])

    n05e = _run_val(
        "N05e", best_p4_mlp, "mlp",
        params={**cur_mlp, "weight_decay": 1e-4},
        notes=f"Track2 N05e weight_decay=1e-4 on {cur_best_run['tag']}",
    )
    track2.append(n05e)
    cur_best_run = min([cur_best_run, n05e], key=lambda r: r["val"]["rmse"])
    cur_mlp = dict(cur_best_run["params"])

    n06 = _run_val(
        "N06", best_p4_mlp, "mlp",
        target="log1p",
        params=cur_mlp,
        notes=f"Track2 N06 target=log1p(price) on {cur_best_run['tag']}",
    )
    track2.append(n06)

    best_mlp = min(track2, key=lambda r: r["val"]["rmse"])
    _log(
        f"Winning MLP: {best_mlp['tag']} ({best_mlp['features']}, target={best_mlp['target']}, "
        f"params={best_mlp['params']}) Val RMSE={best_mlp['val']['rmse']:.4f}"
    )

    _log("=== Multi-Seed MLP Stability Check (42, 123, 2026) ===")
    mlp_seed123 = _run_val(
        "MLP_S123", best_p4_mlp, "mlp",
        target=best_mlp["target"],
        params=best_mlp["params"],
        seed=123,
        notes=f"Multi-seed stability check (seed=123) on {best_mlp['tag']}",
    )
    mlp_seed2026 = _run_val(
        "MLP_S2026", best_p4_mlp, "mlp",
        target=best_mlp["target"],
        params=best_mlp["params"],
        seed=2026,
        notes=f"Multi-seed stability check (seed=2026) on {best_mlp['tag']}",
    )
    multiseed_runs = [best_mlp, mlp_seed123, mlp_seed2026]
    ms_stats: dict[str, Any] = {"seeds": [42, 123, 2026], "runs": multiseed_runs}
    for metric_key in ("rmse", "mae", "r2", "mape", "within_10pct", "within_20pct", "within_1ty"):
        vals = [float(r["val"][metric_key]) for r in multiseed_runs]
        ms_stats[metric_key] = {
            "values": vals,
            "mean": round(float(np.mean(vals)), 4),
            "std": round(float(np.std(vals, ddof=1)), 4),
        }
    _log(
        f"Multi-seed Val RMSE: mean={ms_stats['rmse']['mean']:.4f} ± {ms_stats['rmse']['std']:.4f} "
        f"| R2={ms_stats['r2']['mean']:.4f} ± {ms_stats['r2']['std']:.4f}"
    )

    # ------------------------------------------------------------------
    # Phase D: Final Locked Test Evaluation (RUN ONCE on Test = 4,533 rows)
    # ------------------------------------------------------------------
    _log("=== Phase D: FINAL LOCKED TEST EVALUATION (Test = 4,533 rows) ===")
    raw_rows = load_raw_records(CSV_PATH)
    _, _, test_idx = load_frozen_splits(path=SPLITS_PATH, rows=raw_rows)
    test_rows = [raw_rows[int(i)] for i in test_idx]

    locked_specs = [
        {
            "key": "linear_sgd",
            "label": f"Linear (Mini-batch SGD, {best_p4_linear})",
            "model": "linear_sgd",
            "features": best_p4_linear,
            "target": "raw",
            "params": best_linear["params"],
        },
        {
            "key": "mlp",
            "label": f"MLP (Adam, {best_p4_mlp}, {best_mlp['target']})",
            "model": "mlp",
            "features": best_p4_mlp,
            "target": best_mlp["target"],
            "params": best_mlp["params"],
        },
        {
            "key": "hgb",
            "label": "HGB (P4_geo_priors, External Tree Ref)",
            "model": "hgb",
            "features": "P4_geo_priors",
            "target": "raw",
            "params": {},
        },
        {
            "key": "rf",
            "label": "Random Forest (P4_geo_priors, External Tree Ref)",
            "model": "rf",
            "features": "P4_geo_priors",
            "target": "raw",
            "params": {},
        },
    ]

    final_models: dict[str, Any] = {}
    segment_reports: dict[str, list[dict[str, Any]]] = {}
    perm_importances: dict[str, list[dict[str, Any]]] = {}
    npz_payload: dict[str, np.ndarray] = {}

    for spec in locked_specs:
        payload, details = run_and_log(
            author="final_locked",
            features=spec["features"],
            model=spec["model"],
            target=spec["target"],
            params=spec["params"],
            outlier_filter=False,
            eval_test=True,
            seed=42,
            notes=f"FINAL LOCKED TEST | {spec['label']}",
            save_run=True,
            mirror_to_drive=False,
            csv_path=CSV_PATH,
            splits_path=SPLITS_PATH,
            runs_dir=RUNS_DIR,
            return_details=True,
        )
        tm = payload["metrics"]["test"]
        vm = payload["metrics"]["val"]
        _log(
            f"FINAL {spec['key']:10s} | Val RMSE={vm['rmse']:.4f} | "
            f"TEST RMSE={tm['rmse']:.4f} MAE={tm['mae']:.4f} R2={tm['r2']:.4f} "
            f"MAPE={tm['mape']:.2f}% ±10%={tm['within_10pct']:.4f} ±20%={tm['within_20pct']:.4f} ±1ty={tm['within_1ty']:.4f}"
        )

        npz_payload["y_val"] = np.asarray(details["val_prices"], dtype=np.float32)
        npz_payload["y_test"] = np.asarray(details["test_prices"], dtype=np.float32)
        npz_payload["val_areas"] = np.asarray(details["val_areas"], dtype=np.float32)
        npz_payload["test_areas"] = np.asarray(details["test_areas"], dtype=np.float32)
        npz_payload[f"val_pred_{spec['key']}"] = np.asarray(details["val_pred"], dtype=np.float32)
        npz_payload[f"test_pred_{spec['key']}"] = np.asarray(details["test_pred"], dtype=np.float32)

        final_models[spec["key"]] = {
            "run_id": payload["run_id"],
            "label": spec["label"],
            "model": spec["model"],
            "features": spec["features"],
            "n_features": payload["features"]["n_features"],
            "target": spec["target"],
            "params": spec["params"],
            "hardware": payload["hardware"],
            "train_seconds": payload["train_seconds"],
            "val": vm,
            "test": tm,
            "history": payload.get("history", {}),
        }
        segment_reports[spec["key"]] = _compute_segment_report(
            test_rows=test_rows,
            y_true=np.asarray(details["test_prices"], dtype=float),
            y_pred=np.asarray(details["test_pred"], dtype=float),
            model_label=spec["label"],
        )

        if spec["key"] in ("mlp", "hgb", "linear_sgd"):
            _log(f"Computing Val Permutation Importance for {spec['key']}...")
            perm_importances[spec["key"]] = _perm_importance(
                reg=details["model"],
                X_val=details["X_val"],
                val_prices=details["val_prices"],
                val_areas=details["val_areas"],
                feature_names=list(details["recipe"].feature_names),
                target_name=spec["target"],
                n_repeats=3,
                top_k=15,
                seed=42,
            )

    npz_path = ARTIFACTS_DIR / "final_test_predictions.npz"
    np.savez_compressed(npz_path, **npz_payload)

    summary = {
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": bool(torch.cuda.is_available()),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "total_suite_seconds": round(time.perf_counter() - suite_t0, 2),
        },
        "best_features": {
            "linear_sgd": best_p4_linear,
            "mlp": best_p4_mlp,
        },
        "ablation": ablation,
        "track1_linear_sgd": track1,
        "track2_mlp": track2,
        "best_linear_config": best_linear,
        "best_mlp_config": best_mlp,
        "multiseed_mlp": ms_stats,
        "final_models": final_models,
        "segment_reports": segment_reports,
        "val_permutation_importance": perm_importances,
    }
    summary_path = ARTIFACTS_DIR / "final_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    with tarfile.open(RESULTS_TAR, "w:gz") as tf:
        for run_file in sorted(RUNS_DIR.glob("*.json")):
            tf.add(run_file, arcname=f"experiments/runs/{run_file.name}")
        tf.add(npz_path, arcname="artifacts/final_test_predictions.npz")
        tf.add(summary_path, arcname="artifacts/final_summary.json")

    _log(
        f"Done in {summary['environment']['total_suite_seconds']:.1f}s! "
        f"Archived {len(list(RUNS_DIR.glob('*.json')))} runs + artifacts -> {RESULTS_TAR}"
    )
    return 0


if __name__ == "__main__":
    main()
