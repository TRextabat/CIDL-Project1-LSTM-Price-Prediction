"""All 15 figure types for the CIDL LSTM project.

Each function saves its figure to the specified directory at 300 DPI.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve, auc

# Professional style
STYLE = {
    "figure.figsize": (12, 6),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
}
plt.rcParams.update(STYLE)
sns.set_palette("husl")


def _savefig(fig, save_dir: str | Path, name: str) -> str:
    """Save figure and close. Returns the file path."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"{name}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(path)


# 1. Price and Returns
def plot_price_and_returns(
    df: pd.DataFrame, symbol: str, save_dir: str | Path
) -> str:
    """Plot price series and log-returns distribution side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Price
    axes[0].plot(df.index, df["close"], linewidth=0.8, color="#2196F3")
    axes[0].set_title(f"{symbol.upper()} — Close Price")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Price")
    axes[0].grid(True, alpha=0.3)

    # Returns histogram
    log_ret = np.log(df["close"] / df["close"].shift(1)).dropna()
    axes[1].hist(log_ret, bins=100, density=True, alpha=0.7, color="#4CAF50", edgecolor="none")
    axes[1].axvline(0, color="red", linestyle="--", alpha=0.5)
    axes[1].set_title(f"{symbol.upper()} — Log-Returns Distribution")
    axes[1].set_xlabel("Log Return")
    axes[1].set_ylabel("Density")
    axes[1].grid(True, alpha=0.3)

    # Add stats text
    stats_text = f"Mean: {log_ret.mean():.6f}\nStd: {log_ret.std():.6f}\nSkew: {log_ret.skew():.3f}\nKurt: {log_ret.kurtosis():.3f}"
    axes[1].text(0.02, 0.98, stats_text, transform=axes[1].transAxes,
                 verticalalignment="top", fontsize=10,
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.suptitle(f"{symbol.upper()} Data Overview", fontsize=16, fontweight="bold")
    fig.tight_layout()
    return _savefig(fig, save_dir, f"price_returns_{symbol}")


# 2. Feature Correlation
def plot_feature_correlation(
    features_df: pd.DataFrame, save_dir: str | Path, symbol: str = ""
) -> str:
    """Plot correlation heatmap of features."""
    fig, ax = plt.subplots(figsize=(14, 12))
    corr = features_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, vmin=-1, vmax=1, square=True, ax=ax,
        annot_kws={"size": 8}, linewidths=0.5,
    )
    ax.set_title(f"Feature Correlation Matrix{' — ' + symbol.upper() if symbol else ''}", fontsize=14)
    fig.tight_layout()
    return _savefig(fig, save_dir, f"feature_correlation_{symbol}" if symbol else "feature_correlation")


# 3. Walk-Forward Splits
def plot_walk_forward_splits(
    n_samples: int, n_folds: int, save_dir: str | Path, purge: int = 30
) -> str:
    """Visualize walk-forward cross-validation splits."""
    fig, ax = plt.subplots(figsize=(14, 4))
    test_size = n_samples // (n_folds + 1)
    colors_train = plt.cm.Blues(np.linspace(0.3, 0.7, n_folds))
    colors_test = plt.cm.Oranges(np.linspace(0.4, 0.8, n_folds))

    for fold in range(n_folds):
        train_end = test_size * (fold + 1)
        test_start = train_end + purge
        test_end = min(test_start + test_size, n_samples)
        y = n_folds - fold - 1

        ax.barh(y, train_end, left=0, height=0.6, color=colors_train[fold],
                label="Train" if fold == 0 else None, edgecolor="white")
        ax.barh(y, purge, left=train_end, height=0.6, color="#BDBDBD",
                label="Purge" if fold == 0 else None, edgecolor="white")
        ax.barh(y, test_end - test_start, left=test_start, height=0.6,
                color=colors_test[fold], label="Test" if fold == 0 else None,
                edgecolor="white")
        ax.text(train_end / 2, y, f"Train: {train_end}", ha="center", va="center", fontsize=9)
        ax.text((test_start + test_end) / 2, y, f"Test: {test_end - test_start}",
                ha="center", va="center", fontsize=9)

    ax.set_yticks(range(n_folds))
    ax.set_yticklabels([f"Fold {i}" for i in range(n_folds - 1, -1, -1)])
    ax.set_xlabel("Sample Index")
    ax.set_title("Walk-Forward Cross-Validation Splits", fontsize=14)
    ax.legend(loc="upper right")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    return _savefig(fig, save_dir, "walk_forward_splits")


# 4. Training Curves
def plot_training_curves(results_dict: dict, save_dir: str | Path) -> str:
    """Plot training and validation loss curves for all models."""
    n_models = len(results_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5), squeeze=False)
    axes = axes.flatten()

    for i, (name, result) in enumerate(results_dict.items()):
        ax = axes[i]
        curves = result.get("training_curves", {})
        train_l = curves.get("train_losses", [])
        val_l = curves.get("val_losses", [])
        epochs = range(1, len(train_l) + 1)

        ax.plot(epochs, train_l, label="Train", linewidth=1.5, color="#2196F3")
        ax.plot(epochs, val_l, label="Val", linewidth=1.5, color="#F44336")
        best_ep = result.get("fold_details", [{}])[0].get("best_epoch", 0)
        if best_ep < len(val_l):
            ax.axvline(best_ep + 1, color="green", linestyle="--", alpha=0.5, label=f"Best: {best_ep + 1}")
        ax.set_title(name, fontsize=12)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Training Curves", fontsize=16, fontweight="bold")
    fig.tight_layout()
    return _savefig(fig, save_dir, "training_curves")


# 5. Predictions Overlay
def plot_predictions_overlay(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: np.ndarray | None,
    model_name: str,
    asset: str,
    save_dir: str | Path,
) -> str:
    """Plot predicted vs actual values as overlaid time series."""
    fig, ax = plt.subplots(figsize=(16, 6))
    x = dates if dates is not None else range(len(y_true))

    ax.plot(x, y_true, label="Actual", linewidth=0.8, alpha=0.8, color="#2196F3")
    ax.plot(x, y_pred, label="Predicted", linewidth=0.8, alpha=0.8, color="#F44336")
    ax.fill_between(
        range(len(y_true)),
        y_true - np.abs(y_true - y_pred),
        y_true + np.abs(y_true - y_pred),
        alpha=0.1, color="gray",
    )
    ax.set_title(f"{model_name} — Predictions vs Actuals ({asset})", fontsize=14)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Log Return")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _savefig(fig, save_dir, f"predictions_overlay_{model_name}_{asset}")


# 6. Prediction Scatter
def plot_prediction_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_dir: str | Path,
) -> str:
    """Scatter plot of predicted vs actual with 45-degree line."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true, y_pred, alpha=0.3, s=10, color="#2196F3")

    lims = [
        min(y_true.min(), y_pred.min()),
        max(y_true.max(), y_pred.max()),
    ]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Log Return")
    ax.set_ylabel("Predicted Log Return")
    ax.set_title(f"{model_name} — Predicted vs Actual Scatter", fontsize=14)
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _savefig(fig, save_dir, f"scatter_{model_name}")


# 7. ROC Curves
def plot_roc_curves(results_dict: dict, save_dir: str | Path, asset: str = "") -> str:
    """Plot ROC curves for all models overlaid, with AUC in legend."""
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = plt.cm.Set1(np.linspace(0, 1, len(results_dict)))

    for (name, result), color in zip(results_dict.items(), colors):
        y_true = np.array(result.get("directions_actual", []))
        y_pred = np.array(result.get("predictions", []))
        if len(y_true) == 0 or len(y_pred) == 0:
            continue
        try:
            from scipy.special import expit
            y_score = expit(y_pred)
            fpr, tpr, _ = roc_curve(y_true.astype(int), y_score)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{name} (AUC={roc_auc:.3f})")
        except Exception:
            continue

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves{' — ' + asset.upper() if asset else ''}", fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _savefig(fig, save_dir, f"roc_curves_{asset}" if asset else "roc_curves")


# 8. Confusion Matrices
def plot_confusion_matrices(results_dict: dict, save_dir: str | Path, asset: str = "") -> str:
    """Plot 2x2 confusion matrices for each model."""
    n = len(results_dict)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), squeeze=False)

    for i, (name, result) in enumerate(results_dict.items()):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        y_true = np.array(result.get("directions_actual", [])).astype(int)
        y_pred = np.array(result.get("directions_pred", [])).astype(int)
        if len(y_true) == 0:
            continue
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        ConfusionMatrixDisplay(cm, display_labels=["Down", "Up"]).plot(ax=ax, cmap="Blues")
        ax.set_title(name, fontsize=12)

    # Hide unused axes
    for j in range(i + 1, rows * cols):
        r, c = divmod(j, cols)
        axes[r][c].set_visible(False)

    fig.suptitle(f"Confusion Matrices{' — ' + asset.upper() if asset else ''}", fontsize=16, fontweight="bold")
    fig.tight_layout()
    return _savefig(fig, save_dir, f"confusion_matrices_{asset}" if asset else "confusion_matrices")


# 9. Model Comparison
def plot_model_comparison(metrics_df: pd.DataFrame, save_dir: str | Path) -> str:
    """Grouped bar chart comparing models across key metrics."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    key_metrics = ["rmse", "direction_accuracy", "sharpe_ratio", "f1", "profit_factor", "max_drawdown"]
    colors = sns.color_palette("husl", n_colors=metrics_df["model"].nunique())

    for ax, metric in zip(axes.flatten(), key_metrics):
        if metric not in metrics_df.columns:
            ax.set_visible(False)
            continue
        data = metrics_df.pivot_table(values=metric, index="asset", columns="model")
        data.plot(kind="bar", ax=ax, color=colors, edgecolor="white", width=0.8)
        ax.set_title(metric.replace("_", " ").title(), fontsize=12)
        ax.set_xlabel("")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=0)

    fig.suptitle("Model Comparison Across Assets", fontsize=16, fontweight="bold")
    fig.tight_layout()
    return _savefig(fig, save_dir, "model_comparison")


# 10. Lookback Comparison
def plot_lookback_comparison(metrics_df: pd.DataFrame, save_dir: str | Path) -> str:
    """Line plot showing metric vs lookback window."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = ["rmse", "direction_accuracy", "sharpe_ratio"]

    for ax, metric in zip(axes, metrics):
        if metric not in metrics_df.columns:
            continue
        for model in metrics_df["model"].unique():
            subset = metrics_df[metrics_df["model"] == model]
            grouped = subset.groupby("lookback")[metric].mean()
            ax.plot(grouped.index, grouped.values, marker="o", linewidth=2, label=model)
        ax.set_title(metric.replace("_", " ").title(), fontsize=12)
        ax.set_xlabel("Lookback Window")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Effect of Lookback Window", fontsize=16, fontweight="bold")
    fig.tight_layout()
    return _savefig(fig, save_dir, "lookback_comparison")


# 11. Horizon Comparison
def plot_horizon_comparison(metrics_df: pd.DataFrame, save_dir: str | Path) -> str:
    """Line plot showing metric vs prediction horizon."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = ["rmse", "direction_accuracy", "sharpe_ratio"]

    for ax, metric in zip(axes, metrics):
        if metric not in metrics_df.columns:
            continue
        for model in metrics_df["model"].unique():
            subset = metrics_df[metrics_df["model"] == model]
            grouped = subset.groupby("horizon")[metric].mean()
            ax.plot(grouped.index, grouped.values, marker="s", linewidth=2, label=model)
        ax.set_title(metric.replace("_", " ").title(), fontsize=12)
        ax.set_xlabel("Prediction Horizon")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Effect of Prediction Horizon", fontsize=16, fontweight="bold")
    fig.tight_layout()
    return _savefig(fig, save_dir, "horizon_comparison")


# 12. Attention Weights
def plot_attention_weights(
    weights: np.ndarray,
    timestamps: np.ndarray | None,
    save_dir: str | Path,
    title: str = "Attention Weights",
) -> str:
    """Heatmap of attention weights over time steps."""
    fig, ax = plt.subplots(figsize=(16, 6))
    if weights.ndim == 1:
        weights = weights.reshape(1, -1)

    # Show at most 50 samples
    n_show = min(weights.shape[0], 50)
    sns.heatmap(
        weights[:n_show], ax=ax, cmap="YlOrRd",
        xticklabels=5, yticklabels=False,
    )
    ax.set_xlabel("Time Step (lookback)")
    ax.set_ylabel("Sample")
    ax.set_title(title, fontsize=14)
    fig.tight_layout()
    return _savefig(fig, save_dir, "attention_weights")


# 13. Residual Analysis
def plot_residual_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_dir: str | Path,
    model_name: str = "",
) -> str:
    """Residual histogram and Q-Q plot."""
    residuals = np.asarray(y_true).flatten() - np.asarray(y_pred).flatten()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Histogram
    axes[0].hist(residuals, bins=80, density=True, alpha=0.7, color="#9C27B0", edgecolor="none")
    axes[0].axvline(0, color="red", linestyle="--", alpha=0.5)
    axes[0].set_title("Residual Distribution", fontsize=12)
    axes[0].set_xlabel("Residual")
    axes[0].set_ylabel("Density")
    axes[0].grid(True, alpha=0.3)

    # Q-Q plot
    from scipy import stats as sp_stats
    sp_stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].set_title("Q-Q Plot (Normal)", fontsize=12)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"Residual Analysis{' — ' + model_name if model_name else ''}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return _savefig(fig, save_dir, f"residual_analysis_{model_name}" if model_name else "residual_analysis")


# 14. Equity Curve
def plot_equity_curve(
    strategy_returns: np.ndarray,
    buy_hold_returns: np.ndarray,
    save_dir: str | Path,
    label: str = "",
) -> str:
    """Cumulative returns: strategy vs buy-and-hold."""
    fig, ax = plt.subplots(figsize=(14, 6))

    cum_strategy = np.cumsum(strategy_returns)
    cum_bh = np.cumsum(buy_hold_returns)

    ax.plot(cum_strategy, linewidth=1.5, label="LSTM Strategy", color="#4CAF50")
    ax.plot(cum_bh, linewidth=1.5, label="Buy & Hold", color="#2196F3", alpha=0.7)
    ax.fill_between(range(len(cum_strategy)), cum_strategy, alpha=0.1, color="#4CAF50")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
    ax.set_xlabel("Trade")
    ax.set_ylabel("Cumulative Log Return")
    ax.set_title(f"Equity Curve{' — ' + label if label else ''}", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _savefig(fig, save_dir, f"equity_curve_{label}" if label else "equity_curve")


# 15. Monthly Returns Heatmap
def plot_monthly_returns(
    strategy_returns: np.ndarray,
    dates: pd.DatetimeIndex | np.ndarray,
    save_dir: str | Path,
    label: str = "",
) -> str:
    """Monthly returns heatmap."""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Create series
    returns_series = pd.Series(strategy_returns, index=pd.DatetimeIndex(dates[:len(strategy_returns)]))
    monthly = returns_series.resample("ME").sum()

    # Pivot to year x month
    pivot = pd.DataFrame({
        "year": monthly.index.year,
        "month": monthly.index.month,
        "return": monthly.values,
    }).pivot_table(values="return", index="year", columns="month")
    pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][:len(pivot.columns)]

    sns.heatmap(
        pivot, annot=True, fmt=".3f", cmap="RdYlGn", center=0,
        ax=ax, linewidths=0.5, annot_kws={"size": 9},
    )
    ax.set_title(f"Monthly Returns{' — ' + label if label else ''}", fontsize=14)
    ax.set_ylabel("Year")
    fig.tight_layout()
    return _savefig(fig, save_dir, f"monthly_returns_{label}" if label else "monthly_returns")
