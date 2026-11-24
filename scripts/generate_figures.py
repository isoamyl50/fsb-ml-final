#!/usr/bin/env python3
"""Export 6 light-mode 300-DPI PNG figures from verified runs and artifacts into slides/public/figures/."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SLIDES_SCRIPTS = PROJECT_ROOT / "slides" / "scripts"
if str(SLIDES_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SLIDES_SCRIPTS))

import slide_export
from src.config import ARTIFACTS_DIR
from src.preprocessing import parse_address
from src.splits import load_frozen_splits, load_raw_records

PALETTE = {
    "blue": "#1d4ed8",
    "slate": "#475569",
    "emerald": "#059669",
    "amber": "#d97706",
    "rose": "#e11d48",
    "indigo": "#4338ca",
    "muted": "#94a3b8",
}


def make_price_distribution(
    prices: np.ndarray,
    areas: np.ndarray,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    te_idx: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1))

    ax = axes[0]
    bins = np.linspace(1.0, 11.5, 29)
    ax.hist(
        prices[tr_idx], bins=bins, density=True, alpha=0.30,
        color=PALETTE["blue"], edgecolor=PALETTE["blue"], linewidth=0.8,
        label=f"Train (n={len(tr_idx):,})",
    )
    ax.hist(
        prices[va_idx], bins=bins, density=True, histtype="step",
        linewidth=1.8, color=PALETTE["amber"],
        label=f"Val (n={len(va_idx):,})",
    )
    ax.hist(
        prices[te_idx], bins=bins, density=True, histtype="step",
        linewidth=1.8, linestyle="--", color=PALETTE["emerald"],
        label=f"Test (n={len(te_idx):,})",
    )
    med_p = float(np.median(prices))
    mean_p = float(np.mean(prices))
    ax.axvline(med_p, color="#0f172a", linestyle=":", linewidth=1.5, label=f"Trung vị = {med_p:.2f} tỷ")
    ax.axvline(mean_p, color=PALETTE["rose"], linestyle="-.", linewidth=1.3, label=f"TB = {mean_p:.2f} tỷ")
    ax.set_title("Phân bố Giá bán (Price) trên 3 tập")
    ax.set_xlabel("Giá bán (tỷ VND)")
    ax.set_ylabel("Mật độ xác suất")
    ax.legend(loc="upper right", fontsize=8.5)

    ax2 = axes[1]
    ppm2 = (prices * 1000.0) / np.maximum(areas, 1.0)
    ppm2_clip = ppm2[ppm2 <= 300.0]
    ax2.hist(
        ppm2_clip, bins=35, color=PALETTE["emerald"], alpha=0.65,
        edgecolor="white", linewidth=0.6,
    )
    med_u = float(np.median(ppm2))
    mean_u = float(np.mean(ppm2))
    ax2.axvline(med_u, color="#0f172a", linestyle=":", linewidth=1.6, label=f"Trung vị = {med_u:.1f} tr/m²")
    ax2.axvline(mean_u, color=PALETTE["rose"], linestyle="-.", linewidth=1.4, label=f"TB = {mean_u:.1f} tr/m²")
    ax2.set_title(f"Phân bố Đơn giá (triệu/m²) — {100*len(ppm2_clip)/len(ppm2):.1f}% mẫu ≤ 300")
    ax2.set_xlabel("Đơn giá (triệu VND / m²)")
    ax2.set_ylabel("Số lượng tin đăng")
    ax2.legend(loc="upper right", fontsize=8.8)

    fig.tight_layout(pad=1.4)
    slide_export.save_figure(fig, "price_distribution", close=True)


def make_dataset_samples(
    prices: np.ndarray,
    areas: np.ndarray,
    cities: list[str],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.1))

    rng = np.random.default_rng(42)
    valid_mask = areas <= 250.0
    valid_indices = np.where(valid_mask)[0]
    sample_idx = rng.choice(valid_indices, size=min(3600, len(valid_indices)), replace=False)

    ax = axes[0]
    city_arr = np.array(cities, dtype=object)
    groups = [
        ("Tỉnh/TP khác", np.array([c not in ("Hồ Chí Minh", "Hà Nội") for c in city_arr[sample_idx]]), PALETTE["muted"], 0.35),
        ("Hà Nội", city_arr[sample_idx] == "Hà Nội", PALETTE["amber"], 0.42),
        ("TP. Hồ Chí Minh", city_arr[sample_idx] == "Hồ Chí Minh", PALETTE["blue"], 0.40),
    ]
    for label, mask, color, alpha in groups:
        pts = sample_idx[mask]
        ax.scatter(
            areas[pts], prices[pts], s=11, alpha=alpha, color=color,
            edgecolors="none", label=label, rasterized=True,
        )
    ax.set_title(f"Diện tích vs. Giá bán (≤ 250 m², {100*np.mean(valid_mask):.1f}% mẫu)")
    ax.set_xlabel("Diện tích (m²)")
    ax.set_ylabel("Giá bán (tỷ VND)")
    ax.set_xlim(5, 255)
    ax.set_ylim(0.5, 12.0)
    ax.legend(loc="lower right", fontsize=8.8, markerscale=1.6)

    ax2 = axes[1]
    counts = Counter(cities)
    top6 = [c for c, _ in counts.most_common(6)][::-1]
    by_city: dict[str, list[float]] = defaultdict(list)
    for c, p in zip(cities, prices):
        if c in counts:
            by_city[c].append(float(p))
    data_boxes = [by_city[c] for c in top6]
    labels = [f"{c}\n(n={counts[c]:,})" for c in top6]

    bp = ax2.boxplot(
        data_boxes, orientation="horizontal", patch_artist=True, showfliers=False, widths=0.58,
        medianprops={"color": "#0f172a", "linewidth": 1.8},
        whiskerprops={"color": PALETTE["slate"], "linewidth": 1.1},
        capprops={"color": PALETTE["slate"], "linewidth": 1.1},
    )
    box_colors = ["#cbd5e1", "#cbd5e1", "#93c5fd", "#a7f3d0", "#fde68a", "#93c5fd"]
    for patch, col in zip(bp["boxes"], box_colors):
        patch.set_facecolor(col)
        patch.set_edgecolor(PALETTE["slate"])
        patch.set_alpha(0.88)

    ax2.set_yticks(range(1, len(top6) + 1))
    ax2.set_yticklabels(labels, fontsize=8.8)
    ax2.set_xlabel("Giá bán (tỷ VND)")
    ax2.set_title("Phân bố Giá theo Top 6 Tỉnh/Thành phố")
    ax2.set_xlim(0.5, 12.0)

    fig.tight_layout(pad=1.4)
    slide_export.save_figure(fig, "dataset_samples", close=True)


def make_training_curves(summary: dict) -> None:
    lin = summary["final_models"]["linear_sgd"]
    mlp = summary["final_models"]["mlp"]
    hgb = summary["final_models"]["hgb"]

    lh = lin["history"]
    mh = mlp["history"]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))

    # Left panel: Train vs Val MSE Loss (epochs 2+ zoomed cleanly)
    ax = axes[0]
    ep_lin = np.arange(1, len(lh["train_loss"]) + 1)
    ep_mlp = np.arange(1, len(mh["train_loss"]) + 1)

    ax.plot(
        ep_lin[1:], lh["train_loss"][1:], linestyle="--", linewidth=1.6,
        color="#60a5fa", label=f"Linear SGD Train MSE (ep 1: {lh['train_loss'][0]:.2f})",
    )
    ax.plot(
        ep_lin[1:], lh["val_loss"][1:], linestyle="-", linewidth=2.0,
        color=PALETTE["blue"], label="Linear SGD Val MSE",
    )
    ax.plot(
        ep_mlp[1:], mh["train_loss"][1:], linestyle="--", linewidth=1.6,
        color="#34d399", label=f"MLP Adam Train MSE (ep 1: {mh['train_loss'][0]:.2f})",
    )
    ax.plot(
        ep_mlp[1:], mh["val_loss"][1:], linestyle="-", linewidth=2.1,
        color=PALETTE["emerald"], label="MLP Adam Val MSE",
    )
    ax.set_ylim(1.12, 2.25)
    ax.set_xlim(1, max(len(ep_lin), len(ep_mlp)) + 2)
    ax.set_title("Đường cong Mất mát MSE (từ Epoch 2)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (tỷ²)")
    ax.legend(loc="upper right", fontsize=8.2)

    # Right panel: Validation RMSE (billion VND) & Early Stopping best checkpoints
    ax2 = axes[1]
    ax2.plot(
        ep_lin, lh["val_rmse"], linewidth=2.0, color=PALETTE["blue"],
        label=f"Linear SGD (L02, best ep {lh['best_epoch']})",
    )
    ax2.plot(
        ep_mlp, mh["val_rmse"], linewidth=2.2, color=PALETTE["emerald"],
        label=f"MLP Adam (N05e, best ep {mh['best_epoch']})",
    )

    best_lin_ep = int(lh["best_epoch"])
    best_lin_rmse = float(lh["val_rmse"][best_lin_ep - 1])
    best_mlp_ep = int(mh["best_epoch"])
    best_mlp_rmse = float(mh["val_rmse"][best_mlp_ep - 1])

    ax2.scatter([best_lin_ep], [best_lin_rmse], color=PALETTE["blue"], s=65, zorder=5, edgecolor="#0f172a")
    ax2.annotate(
        f"Best Ep {best_lin_ep}\n{best_lin_rmse:.4f} tỷ",
        xy=(best_lin_ep, best_lin_rmse),
        xytext=(best_lin_ep - 24, best_lin_rmse + 0.055),
        fontsize=8.5, fontweight="bold", color=PALETTE["blue"],
        arrowprops={"arrowstyle": "->", "color": PALETTE["blue"], "lw": 1.1},
    )

    ax2.scatter([best_mlp_ep], [best_mlp_rmse], color=PALETTE["emerald"], s=65, zorder=5, edgecolor="#0f172a")
    ax2.annotate(
        f"Best Ep {best_mlp_ep}\n{best_mlp_rmse:.4f} tỷ",
        xy=(best_mlp_ep, best_mlp_rmse),
        xytext=(best_mlp_ep + 10, best_mlp_rmse + 0.036),
        fontsize=8.5, fontweight="bold", color=PALETTE["emerald"], va="center",
        arrowprops={"arrowstyle": "->", "color": PALETTE["emerald"], "lw": 1.1},
    )

    hgb_val_rmse = float(hgb["val"]["rmse"])
    ax2.axhline(
        hgb_val_rmse, color=PALETTE["amber"], linestyle="--", linewidth=1.6,
        label=f"HGB + P4 tham chiếu ({hgb_val_rmse:.4f} tỷ)",
    )
    ax2.set_ylim(1.14, 1.56)
    ax2.set_xlim(0, max(len(ep_lin), len(ep_mlp)) + 3)
    ax2.set_title("Validation RMSE (tỷ VND) & Early Stopping")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Val RMSE (tỷ VND)")
    ax2.legend(loc="upper right", fontsize=8.2)

    fig.tight_layout(pad=1.4)
    slide_export.save_figure(fig, "training_curves", close=True)


def make_pred_vs_actual(summary: dict, preds: np.lib.npyio.NpzFile) -> None:
    y_test = np.asarray(preds["y_test"], dtype=float)
    lin_pred = np.asarray(preds["test_pred_linear_sgd"], dtype=float)
    mlp_pred = np.asarray(preds["test_pred_mlp"], dtype=float)

    lin_m = summary["final_models"]["linear_sgd"]["test"]
    mlp_m = summary["final_models"]["mlp"]["test"]

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.3), sharey=True)
    x_grid = np.array([0.5, 12.0])

    specs = [
        (axes[0], "Linear (Mini-batch SGD)", lin_pred, lin_m, PALETTE["blue"]),
        (axes[1], "MLP (Adam, 128 ReLU)", mlp_pred, mlp_m, PALETTE["emerald"]),
    ]

    for ax, title, yp, tm, color in specs:
        ax.fill_between(
            x_grid, x_grid - 1.0, x_grid + 1.0,
            color="#e2e8f0", alpha=0.55, label="Vùng ±1 tỷ VND",
        )
        ax.plot(x_grid, x_grid, color="#0f172a", linestyle="--", linewidth=1.4, label="Lý tưởng (y = x)")
        ax.scatter(
            y_test, yp, s=9, alpha=0.28, color=color, edgecolors="none", rasterized=True
        )
        ax.text(
            0.04, 0.95,
            f"RMSE = {tm['rmse']:.4f} tỷ\nMAE  = {tm['mae']:.4f} tỷ\nR²   = {tm['r2']:.4f}\n±20% = {tm['within_20pct']*100:.1f}%",
            transform=ax.transAxes, va="top", ha="left", fontsize=8.4,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.94},
        )
        ax.set_title(title, fontsize=11.5)
        ax.set_xlabel("Giá Thực tế (tỷ VND)")
        ax.set_xlim(0.7, 11.8)
        ax.set_ylim(0.5, 11.8)
        ax.legend(loc="lower right", fontsize=8.0)

    axes[0].set_ylabel("Giá Dự đoán (tỷ VND)")
    fig.suptitle(f"Thực tế vs. Dự đoán trên Tập Test độc lập (n = {len(y_test):,})", fontsize=12.5, fontweight="bold", y=1.01)
    fig.tight_layout(pad=1.1)
    slide_export.save_figure(fig, "pred_vs_actual", close=True)


def make_residual_analysis(summary: dict, preds: np.lib.npyio.NpzFile) -> None:
    y_test = np.asarray(preds["y_test"], dtype=float)
    res_lin = np.asarray(preds["test_pred_linear_sgd"], dtype=float) - y_test
    res_mlp = np.asarray(preds["test_pred_mlp"], dtype=float) - y_test
    res_hgb = np.asarray(preds["test_pred_hgb"], dtype=float) - y_test

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.3), gridspec_kw={"width_ratios": [1.0, 1.15]})

    ax = axes[0]
    bins = np.linspace(-4.5, 4.5, 37)
    ax.hist(res_lin, bins=bins, density=True, histtype="step", linewidth=1.6, color=PALETTE["blue"], label=f"Linear SGD (std={np.std(res_lin):.2f})")
    ax.hist(res_mlp, bins=bins, density=True, alpha=0.28, color=PALETTE["emerald"], edgecolor=PALETTE["emerald"], linewidth=1.4, label=f"MLP Adam (std={np.std(res_mlp):.2f})")
    ax.hist(res_hgb, bins=bins, density=True, histtype="step", linestyle="--", linewidth=1.7, color=PALETTE["amber"], label=f"HGB Ref (std={np.std(res_hgb):.2f})")
    ax.axvline(0.0, color="#0f172a", linestyle=":", linewidth=1.4)
    ax.set_title("Phân bố Phần dư (y_pred − y_true)", fontsize=11.2)
    ax.set_xlabel("Sai số dự đoán trên Test (tỷ VND)")
    ax.set_ylabel("Mật độ")
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(0.0, 0.56)
    ax.legend(loc="upper left", fontsize=8.0)

    ax2 = axes[1]
    top_imp = summary["val_permutation_importance"]["mlp"][:8][::-1]
    nice_names = {
        "log1p_est_price_ward": "log1p(Est. Price Ward)",
        "log1p_AccessRoad": "log1p(Access Road)",
        "Furniture state_Full": "Nội thất Đầy đủ",
        "city_Hồ Chí Minh": "TP. Hồ Chí Minh",
        "Furniture state_Basic": "Nội thất Cơ bản",
        "addr_ridge_ppm2": "Ridge TF-IDF Đơn giá/m²",
        "oof_street_ppm2": "OOF Street Price/m²",
        "Legal status_Have certificate": "Đã có Sổ hồng/đỏ",
        "city_Hà Nội": "Hà Nội",
        "Floors": "Số tầng (Floors)",
    }
    f_labels = [nice_names.get(item["feature"], item["feature"]) for item in top_imp]
    means = [item["delta_rmse_mean"] for item in top_imp]
    stds = [item["delta_rmse_std"] for item in top_imp]

    y_pos = np.arange(len(f_labels))
    bars = ax2.barh(
        y_pos, means, xerr=stds, color=PALETTE["emerald"], alpha=0.85,
        edgecolor="#065f46", linewidth=0.7, capsize=2.5, height=0.64,
    )
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(f_labels, fontsize=8.4)
    ax2.set_xlabel("Δ Val RMSE khi hoán vị đặc trưng (tỷ VND)")
    ax2.set_title("Top 8 Đặc trưng Quan trọng (MLP Adam)", fontsize=11.2)
    ax2.set_xlim(0, max(means) * 1.28)
    for bar, val, err in zip(bars, means, stds):
        ax2.text(
            val + err + 0.010,
            bar.get_y() + bar.get_height() / 2.0,
            f"+{val:.3f}",
            va="center", ha="left", fontsize=8.0, fontweight="bold", color="#0f172a",
        )

    fig.tight_layout(pad=1.2)
    slide_export.save_figure(fig, "residual_analysis", close=True)


def make_preprocessing_impact(summary: dict) -> None:
    abl = summary["ablation"]
    stages = ["P0_numeric", "P2_domain", "P4_compact", "P4_geo_priors"]
    stage_labels = ["P0_numeric\n(6 biến số)", "P2_domain\n(Impute+Tỉ lệ)", "P4_compact\n(Bỏ OHE thưa)", "P4_geo_priors\n(Full 345 cột)"]

    lin_rmse = [abl["linear_sgd"][s]["val"]["rmse"] for s in stages]
    mlp_rmse = [abl["mlp"][s]["val"]["rmse"] for s in stages]
    hgb_rmse = [abl["hgb"][s]["val"]["rmse"] for s in stages]

    fig, ax = plt.subplots(figsize=(8.8, 3.7))
    x = np.arange(len(stages))
    w = 0.25

    b1 = ax.bar(x - w, lin_rmse, width=w, color=PALETTE["blue"], alpha=0.90, label="Linear (Mini-batch SGD)")
    b2 = ax.bar(x, mlp_rmse, width=w, color=PALETTE["emerald"], alpha=0.90, label="MLP (Adam)")
    b3 = ax.bar(x + w, hgb_rmse, width=w, color=PALETTE["amber"], alpha=0.90, label="HGB (Tree Ref)")

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + 0.025,
                f"{h:.3f}",
                ha="center", va="bottom", fontsize=7.8, fontweight="bold", color="#0f172a",
            )

    drop_pct = (mlp_rmse[0] - mlp_rmse[-1]) / mlp_rmse[0] * 100.0
    ax.annotate(
        f"MLP: −{mlp_rmse[0] - mlp_rmse[-1]:.3f} tỷ (−{drop_pct:.1f}%)",
        xy=(3.0, mlp_rmse[-1] + 0.085),
        xytext=(1.15, 1.68),
        fontsize=8.6, fontweight="bold", color=PALETTE["emerald"],
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#ecfdf5", "edgecolor": PALETTE["emerald"]},
        arrowprops={"arrowstyle": "->", "color": PALETTE["emerald"], "lw": 1.3},
    )

    ax.set_xticks(x)
    ax.set_xticklabels(stage_labels, fontsize=8.8)
    ax.set_ylabel("Validation RMSE (tỷ VND)")
    ax.set_title("Tác động Tiền xử lý & Feature Engineering (P0 → P4)", fontsize=11.8)
    ax.set_ylim(0.85, 2.18)
    ax.legend(loc="upper right", fontsize=8.4)

    fig.tight_layout(pad=1.1)
    slide_export.save_figure(fig, "preprocessing_impact", close=True)


def main() -> int:
    slide_export.apply_slide_theme()
    rows = load_raw_records()
    tr_idx, va_idx, te_idx = load_frozen_splits(rows=rows)
    prices = np.array([float(r["Price"]) for r in rows], dtype=float)
    areas = np.array([float(r["Area"]) for r in rows], dtype=float)
    cities = [parse_address(r.get("Address"))["city"] for r in rows]

    summary = json.loads((ARTIFACTS_DIR / "final_summary.json").read_text(encoding="utf-8"))
    preds = np.load(ARTIFACTS_DIR / "final_test_predictions.npz")

    make_price_distribution(prices, areas, tr_idx, va_idx, te_idx)
    make_dataset_samples(prices, areas, cities)
    make_training_curves(summary)
    make_pred_vs_actual(summary, preds)
    make_residual_analysis(summary, preds)
    make_preprocessing_impact(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
