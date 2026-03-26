#!/usr/bin/env python3
"""Create composite grid images from individual figures."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import numpy as np

FIG = Path("/home/user/D/CIDL_project1/outputs/figures")
OUT = FIG  # save composites to same folder


def make_grid(image_paths, title, output_name, ncols=4, figsize_per=(4, 3)):
    """Combine multiple images into a single grid figure."""
    paths = [p for p in image_paths if p.exists()]
    if not paths:
        print(f"  SKIP {output_name}: no images found")
        return
    n = len(paths)
    nrows = (n + ncols - 1) // ncols
    fw = figsize_per[0] * ncols
    fh = figsize_per[1] * nrows + 0.8
    fig, axes = plt.subplots(nrows, ncols, figsize=(fw, fh), facecolor="white")
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.99)
    if nrows == 1:
        axes = [axes] if ncols == 1 else [axes]
    axes_flat = np.array(axes).flatten()
    for i, ax in enumerate(axes_flat):
        if i < n:
            img = mpimg.imread(str(paths[i]))
            ax.imshow(img)
            label = paths[i].stem.replace("_", " ").replace("equity curve ", "").replace("predictions overlay ", "").replace("residual analysis ", "").replace("scatter ", "")
            # Shorten label
            parts = label.split()
            short = " ".join(parts[:5]) if len(parts) > 5 else label
            ax.set_title(short, fontsize=7, pad=2)
        ax.axis("off")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = OUT / output_name
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {output_name} ({n} images in {nrows}x{ncols} grid)")


# =========================================================================
# EQUITY CURVES — all 16 XAUUSD in one grid, all 16 BTCUSD in one grid
# =========================================================================
print("Creating equity curve composites...")
models = ["simple_lstm", "stacked_lstm", "bilstm", "lstm_attention"]
lookbacks = [30, 60]
horizons = [1, 5]

for asset, asset_label in [("xauusd_1h", "XAUUSD 1H"), ("btcusd_1d", "BTCUSD 1D")]:
    paths = []
    for m in models:
        for lb in lookbacks:
            for h in horizons:
                paths.append(FIG / f"equity_curve_{m}_lb{lb}_h{h}_{asset}.png")
    make_grid(paths, f"Equity Curves — {asset_label} (All Models & Configs)",
              f"composite_equity_{asset}.png", ncols=4, figsize_per=(4, 3))

# =========================================================================
# PREDICTIONS OVERLAY — same grid approach
# =========================================================================
print("Creating prediction overlay composites...")
for asset, asset_label in [("xauusd_1h", "XAUUSD 1H"), ("btcusd_1d", "BTCUSD 1D")]:
    paths = []
    for m in models:
        for lb in lookbacks:
            for h in horizons:
                paths.append(FIG / f"predictions_overlay_{m}_lb{lb}_h{h}_{asset}.png")
    make_grid(paths, f"Predicted vs Actual Returns — {asset_label}",
              f"composite_predictions_{asset}.png", ncols=4, figsize_per=(4, 3))

# =========================================================================
# SCATTER PLOTS — all in one
# =========================================================================
print("Creating scatter plot composites...")
for asset, asset_label in [("xauusd_1h", "XAUUSD 1H"), ("btcusd_1d", "BTCUSD 1D")]:
    paths = []
    for m in models:
        for lb in lookbacks:
            for h in horizons:
                paths.append(FIG / f"scatter_{m}_lb{lb}_h{h}_{asset}.png")
    make_grid(paths, f"Scatter: Predicted vs Actual — {asset_label}",
              f"composite_scatter_{asset}.png", ncols=4, figsize_per=(4, 3))

# =========================================================================
# RESIDUAL ANALYSIS — all in one
# =========================================================================
print("Creating residual analysis composites...")
for asset, asset_label in [("xauusd_1h", "XAUUSD 1H"), ("btcusd_1d", "BTCUSD 1D")]:
    paths = []
    for m in models:
        for lb in lookbacks:
            for h in horizons:
                paths.append(FIG / f"residual_analysis_{m}_lb{lb}_h{h}_{asset}.png")
    make_grid(paths, f"Residual Analysis — {asset_label}",
              f"composite_residuals_{asset}.png", ncols=4, figsize_per=(4, 3))

# =========================================================================
# MODEL ARCHITECTURES — all 4 Mermaid diagrams in 2x2
# =========================================================================
print("Creating architecture composite...")
arch_paths = [
    FIG / "mermaid_simple_lstm.png",
    FIG / "mermaid_stacked_lstm.png",
    FIG / "mermaid_bilstm.png",
    FIG / "mermaid_lstm_attention.png",
]
make_grid(arch_paths, "LSTM Model Architectures",
          "composite_architectures.png", ncols=2, figsize_per=(6, 7))

# =========================================================================
# FUZZY — all 3 in one row
# =========================================================================
print("Creating fuzzy composite...")
fuzzy_paths = [
    FIG / "fuzzy_input_membership_functions.png",
    FIG / "fuzzy_output_membership_functions.png",
    FIG / "fuzzy_grade_surface.png",
]
make_grid(fuzzy_paths, "Fuzzy Logic: Student Performance Evaluation System",
          "composite_fuzzy.png", ncols=3, figsize_per=(5, 4))

# =========================================================================
# CLASSIFICATION — ROC + Confusion in 2x2
# =========================================================================
print("Creating classification composite...")
cls_paths = [
    FIG / "roc_curves_xauusd_1h.png",
    FIG / "roc_curves_btcusd_1d.png",
    FIG / "confusion_matrices_xauusd_1h.png",
    FIG / "confusion_matrices_btcusd_1d.png",
]
make_grid(cls_paths, "Classification Analysis: ROC Curves & Confusion Matrices",
          "composite_classification.png", ncols=2, figsize_per=(6, 5))

# =========================================================================
# DATA EXPLORATION — price/returns + feature correlation in 2x2
# =========================================================================
print("Creating data exploration composite...")
data_paths = [
    FIG / "price_returns_xauusd.png",
    FIG / "price_returns_btcusd.png",
    FIG / "feature_correlation_xauusd.png",
    FIG / "feature_correlation_btcusd.png",
]
make_grid(data_paths, "Data Exploration: Price History, Returns & Feature Correlations",
          "composite_data_exploration.png", ncols=2, figsize_per=(6, 5))

# =========================================================================
# COMPARISON CHARTS — model, lookback, horizon in 1x3
# =========================================================================
print("Creating comparison composite...")
comp_paths = [
    FIG / "model_comparison.png",
    FIG / "lookback_comparison.png",
    FIG / "horizon_comparison.png",
]
make_grid(comp_paths, "Model, Lookback & Horizon Comparisons",
          "composite_comparisons.png", ncols=3, figsize_per=(5, 4))

print("\nAll composite figures created!")
