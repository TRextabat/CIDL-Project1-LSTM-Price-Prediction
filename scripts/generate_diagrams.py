#!/usr/bin/env python3
"""Generate architecture and strategy diagrams for the report."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

OUT = Path("outputs/figures")
OUT.mkdir(parents=True, exist_ok=True)

# Color palette
C = {
    "lstm": "#4A90D9",
    "fc": "#E8913A",
    "attn": "#D94A6B",
    "input": "#5CB85C",
    "output": "#9B59B6",
    "bg": "#FAFBFC",
    "border": "#DEE2E6",
    "text": "#2C3E50",
    "arrow": "#7F8C8D",
}


def box(ax, x, y, w, h, text, color, fontsize=9, bold=False):
    rect = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.1", facecolor=color, edgecolor="#555", linewidth=1.2, alpha=0.9
    )
    ax.add_patch(rect)
    weight = "bold" if bold else "normal"
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, weight=weight, color="white")


def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=C["arrow"], lw=1.5))


# =========================================================================
# 1. All 4 Model Architectures - Side by Side
# =========================================================================
def plot_architectures():
    fig, axes = plt.subplots(1, 4, figsize=(20, 8), facecolor=C["bg"])
    fig.suptitle("LSTM Model Architectures", fontsize=18, fontweight="bold", color=C["text"], y=0.98)

    models = [
        {
            "name": "Simple LSTM",
            "params": "76,417",
            "blocks": [
                ("Input\n(batch, seq, 20)", C["input"], 0.5),
                ("LSTM\n1 layer, h=128", C["lstm"], 0.38),
                ("Last Hidden\nState", C["arrow"], 0.28),
                ("FC Layer\n128 -> 1", C["fc"], 0.18),
                ("Output\nlog-return", C["output"], 0.08),
            ],
        },
        {
            "name": "Stacked LSTM",
            "params": "357,121",
            "blocks": [
                ("Input\n(batch, seq, 20)", C["input"], 0.55),
                ("LSTM Layer 1\nh=128", C["lstm"], 0.45),
                ("LSTM Layer 2\nh=128, drop=0.2", C["lstm"], 0.35),
                ("LSTM Layer 3\nh=128, drop=0.2", C["lstm"], 0.25),
                ("FC -> ReLU -> FC\n128->64->1", C["fc"], 0.15),
                ("Output\nlog-return", C["output"], 0.05),
            ],
        },
        {
            "name": "Bidirectional LSTM",
            "params": "580,865",
            "blocks": [
                ("Input\n(batch, seq, 20)", C["input"], 0.55),
                ("BiLSTM Layer 1\nfwd+bwd, h=128", C["lstm"], 0.43),
                ("BiLSTM Layer 2\nfwd+bwd, drop=0.2", C["lstm"], 0.31),
                ("Concat\nh=256 (128x2)", C["arrow"], 0.21),
                ("FC -> ReLU -> FC\n256->128->1", C["fc"], 0.11),
                ("Output\nlog-return", C["output"], 0.02),
            ],
        },
        {
            "name": "LSTM + Attention",
            "params": "233,346",
            "blocks": [
                ("Input\n(batch, seq, 20)", C["input"], 0.58),
                ("LSTM Layer 1\nh=128", C["lstm"], 0.48),
                ("LSTM Layer 2\nh=128, drop=0.2", C["lstm"], 0.38),
                ("Bahdanau\nAttention", C["attn"], 0.28),
                ("Context Vector\nWeighted Sum", C["attn"], 0.18),
                ("FC Layer\n128 -> 1", C["fc"], 0.08),
                ("Output\nlog-return", C["output"], -0.01),
            ],
        },
    ]

    for idx, (ax, model) in enumerate(zip(axes, models)):
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.05, 0.65)
        ax.set_facecolor(C["bg"])
        ax.axis("off")
        ax.set_title(f"{model['name']}\n({model['params']} params)", fontsize=11,
                     fontweight="bold", color=C["text"], pad=10)

        blocks = model["blocks"]
        for i, (text, color, y_pos) in enumerate(blocks):
            box(ax, 0.5, y_pos, 0.7, 0.07, text, color, fontsize=7.5, bold=("LSTM" in text or "Attention" in text))
            if i < len(blocks) - 1:
                next_y = blocks[i + 1][2]
                arrow(ax, 0.5, y_pos - 0.035, 0.5, next_y + 0.035)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = OUT / "architecture_all_models.png"
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor=C["bg"])
    plt.close()
    print(f"Saved: {path}")


# =========================================================================
# 2. Walk-Forward Validation Strategy
# =========================================================================
def plot_walk_forward_strategy():
    fig, ax = plt.subplots(figsize=(14, 5), facecolor=C["bg"])
    ax.set_facecolor(C["bg"])

    n_folds = 3
    total_bars = 100
    colors_train = "#4A90D9"
    colors_test = "#E8913A"
    colors_purge = "#E74C3C"

    fold_labels = []
    for fold in range(n_folds):
        y = n_folds - fold - 1
        train_end = (fold + 1) * 25
        purge_start = train_end
        purge_end = train_end + 3
        test_start = purge_end
        test_end = test_start + 25

        # Train
        ax.barh(y, train_end, left=0, height=0.6, color=colors_train, alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.text(train_end / 2, y, f"TRAIN ({train_end}%)", ha="center", va="center", fontsize=9, color="white", fontweight="bold")

        # Purge
        ax.barh(y, purge_end - purge_start, left=purge_start, height=0.6, color=colors_purge, alpha=0.7, edgecolor="white")
        ax.text((purge_start + purge_end) / 2, y, "P", ha="center", va="center", fontsize=7, color="white", fontweight="bold")

        # Test
        ax.barh(y, test_end - test_start, left=test_start, height=0.6, color=colors_test, alpha=0.85, edgecolor="white")
        ax.text((test_start + test_end) / 2, y, f"TEST ({25}%)", ha="center", va="center", fontsize=9, color="white", fontweight="bold")

        # Unused
        if test_end < total_bars:
            ax.barh(y, total_bars - test_end, left=test_end, height=0.6, color="#ECF0F1", edgecolor="white")

        fold_labels.append(f"Fold {fold + 1}")

    ax.set_yticks(range(n_folds))
    ax.set_yticklabels(fold_labels[::-1], fontsize=11, fontweight="bold", color=C["text"])
    ax.set_xlabel("Data Timeline (%)", fontsize=12, color=C["text"])
    ax.set_title("Walk-Forward Cross-Validation Strategy\n(Expanding Window with Purge Gap)", fontsize=14, fontweight="bold", color=C["text"], pad=15)
    ax.set_xlim(0, total_bars)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=colors_train, alpha=0.85, label="Training Set (expanding)"),
        mpatches.Patch(facecolor=colors_purge, alpha=0.7, label="Purge Gap (prevents leakage)"),
        mpatches.Patch(facecolor=colors_test, alpha=0.85, label="Test Set (fixed size)"),
        mpatches.Patch(facecolor="#ECF0F1", label="Unused data"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    path = OUT / "architecture_walk_forward.png"
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor=C["bg"])
    plt.close()
    print(f"Saved: {path}")


# =========================================================================
# 3. Training Pipeline Flowchart
# =========================================================================
def plot_training_pipeline():
    fig, ax = plt.subplots(figsize=(16, 4), facecolor=C["bg"])
    ax.set_facecolor(C["bg"])
    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(-0.3, 0.8)
    ax.axis("off")
    ax.set_title("End-to-End Training Pipeline", fontsize=15, fontweight="bold", color=C["text"], pad=15)

    steps = [
        ("Load\nParquet Data", C["input"]),
        ("Compute 20\nFeatures", "#17A589"),
        ("Walk-Forward\nSplit", "#2E86C1"),
        ("RobustScaler\n(per fold)", "#8E44AD"),
        ("Create PyTorch\nDatasets", C["lstm"]),
        ("Train LSTM\non GPU", C["attn"]),
        ("Early Stop\n+ Best Model", C["fc"]),
        ("Compute All\nMetrics", "#27AE60"),
        ("Generate\nFigures + PDF", C["output"]),
    ]

    for i, (text, color) in enumerate(steps):
        box(ax, i, 0.3, 0.85, 0.35, text, color, fontsize=8, bold=True)
        if i < len(steps) - 1:
            arrow(ax, i + 0.43, 0.3, i + 0.57, 0.3)

    plt.tight_layout()
    path = OUT / "architecture_training_pipeline.png"
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor=C["bg"])
    plt.close()
    print(f"Saved: {path}")


# =========================================================================
# 4. Attention Mechanism Detail
# =========================================================================
def plot_attention_detail():
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=C["bg"])
    ax.set_facecolor(C["bg"])
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.1, 1.1)
    ax.axis("off")
    ax.set_title("Bahdanau Attention Mechanism", fontsize=14, fontweight="bold", color=C["text"], pad=15)

    # LSTM hidden states
    for i in range(5):
        x = 0.1 + i * 0.2
        box(ax, x, 0.85, 0.15, 0.08, f"h_{i+1}", C["lstm"], fontsize=8)
        arrow(ax, x, 0.81, x, 0.7)

    # Attention scores
    box(ax, 0.5, 0.65, 0.85, 0.08, "score_i = V * tanh(W * h_i)    then    alpha_i = softmax(scores)", "#E67E22", fontsize=8, bold=True)
    arrow(ax, 0.5, 0.61, 0.5, 0.48)

    # Weighted sum
    box(ax, 0.5, 0.43, 0.7, 0.08, "context = SUM(alpha_i * h_i)", C["attn"], fontsize=9, bold=True)
    arrow(ax, 0.5, 0.39, 0.5, 0.26)

    # FC
    box(ax, 0.5, 0.21, 0.5, 0.08, "FC Layer -> Output", C["fc"], fontsize=9, bold=True)
    arrow(ax, 0.5, 0.17, 0.5, 0.06)
    box(ax, 0.5, 0.02, 0.4, 0.06, "Predicted log-return", C["output"], fontsize=8)

    # Labels
    ax.text(0.5, 0.95, "LSTM Hidden States (all timesteps)", ha="center", fontsize=10, color=C["text"], style="italic")

    plt.tight_layout()
    path = OUT / "architecture_attention_detail.png"
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor=C["bg"])
    plt.close()
    print(f"Saved: {path}")


if __name__ == "__main__":
    plot_architectures()
    plot_walk_forward_strategy()
    plot_training_pipeline()
    plot_attention_detail()
    print("\nAll architecture diagrams generated!")
