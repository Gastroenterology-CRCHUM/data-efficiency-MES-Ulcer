"""
src/evaluation/plots.py
-----------------------
Pure visualisation — no MLflow, no file management, no metric computation.

Every function returns the matplotlib Figure so the caller decides
what to do with it (save, log to MLflow, display in a notebook…).

Public API
----------
plot_roc_curve(labels, probs, threshold, title)         -> Figure
plot_confusion_matrix(cm, threshold, class_names)       -> Figure
plot_roc_curves(roc_data, title)                        -> Figure
plot_delong_heatmap(p_matrix, df_summary, alpha)        -> Figure
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

# ---------------------------------------------------------------------------
# Single-model ROC curve
# ---------------------------------------------------------------------------


def plot_roc_curve(
    name: str,
    labels: np.ndarray,
    probs: np.ndarray,
    threshold: float | None = None,
) -> plt.Figure:
    """Plot a single ROC curve with an optional operating-point marker.

    Args:
        name:      Model name.
        labels:    Ground-truth binary labels.
        probs:     Predicted probabilities for the positive class.
                   Accepts shape (N,) or (N, 2) — second column is used.
        threshold: If provided, marks the closest point on the curve in red.
        title:     Figure title.

    Returns:
        matplotlib Figure (not yet saved or shown).
    """
    if isinstance(probs, np.ndarray) and probs.ndim == 2:
        probs = probs[:, 1]

    fpr, tpr, thresholds = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("1 − Specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title(f"ROC Curve — Ulcer detection with {name}")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    if threshold is not None:
        idx = np.argmin(np.abs(thresholds - threshold))
        ax.scatter(
            fpr[idx],
            tpr[idx],
            color="red",
            zorder=5,
            label=f"Threshold = {threshold:.2f}",
        )
        ax.legend(loc="lower right")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------


def plot_confusion_matrix(
    cm: np.ndarray,
    threshold: float,
    class_names: tuple[str, str] = ("No ulcer", "Ulcer"),
    title: str = "Confusion matrix — Ulcer detection",
) -> plt.Figure:
    """Plot a colour-coded normalised confusion matrix.

    Cell colours:
    - Dark blue → high rate (row-normalised)
    - White → low rate
    - Grey → zero samples (avoid misleading white cells)

    Args:
        cm:          2×2 confusion matrix (sklearn convention: rows = true,
                     columns = predicted).
        threshold:   Decision threshold shown in the title.
        class_names: (negative_label, positive_label).
        title:       Figure title.

    Returns:
        matplotlib Figure (not yet saved or shown).
    """
    # Row-normalised rates + raw counts for annotations
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)  # Avoid division by zero

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for i in range(2):
        for j in range(2):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(
                j,
                i,
                f"{cm_norm[i, j] * 100:.1f}%\n({cm[i, j]})",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color=color,
            )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title + f"  (threshold = {threshold:.2f})")

    fig.tight_layout()
    return fig


def plot_confusion_matrix_multiclass(
    cm: np.ndarray,
    class_names: list[str],
    title: str = "Confusion matrix",
) -> plt.Figure:
    """Normalised confusion matrix for N classes.

    Args:
        cm:          N×N confusion matrix (sklearn convention).
        class_names: List of class labels (length N).
        title:       Figure title.

    Returns:
        matplotlib Figure.
    """
    n = len(class_names)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(n * 2, n * 1.8))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for i in range(n):
        for j in range(n):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(
                j,
                i,
                f"{cm_norm[i, j] * 100:.1f}%\n({cm[i, j]})",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color=color,
            )

    ax.set_xticks(range(n))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Multi-model ROC curves
# ---------------------------------------------------------------------------


def plot_roc_curves(
    roc_data: list[dict],
    title: str = "ROC curves — Ulcer detection",
) -> plt.Figure:
    """Overlay ROC curves for several models on the same axes.

    Args:
        roc_data: List of dicts, each with keys:
                    "name"  (str)   — label shown in the legend,
                    "fpr"   (array) — false positive rates,
                    "tpr"   (array) — true positive rates,
                    "auc"   (float) — AUROC value.
        title:    Figure title.

    Returns:
        matplotlib Figure (not yet saved or shown).
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    for entry in roc_data:
        ax.plot(
            entry["fpr"],
            entry["tpr"],
            linewidth=2,
            label=f"{entry['name']}  (AUC = {entry['auc']:.3f})",
        )

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Chance")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("1 − Specificity  (FPR)")
    ax.set_ylabel("Sensitivity  (TPR)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# DeLong heatmap
# ---------------------------------------------------------------------------


def _delong_cell_color(p: float, alpha: float) -> list[float]:
    """RGBA color for an off-diagonal DeLong heatmap cell.

    Green (significant) when p < alpha; red otherwise; dark background for NaN.
    """
    if np.isnan(p):
        return [0.05, 0.05, 0.1, 1.0]
    if p < alpha:
        intensity = max(0.4, 1 - p / alpha * 0.6)
        return [0.0, intensity * 0.8, intensity * 0.4, 1.0]
    intensity = max(0.4, 1 - (p - alpha) / (1 - alpha) * 0.5)
    return [intensity * 0.9, 0.1, 0.1, 1.0]


def plot_delong_heatmap(
    p_matrix: pd.DataFrame,
    df_summary: pd.DataFrame,
    alpha: float = 0.05,
) -> plt.Figure:
    """Visualise DeLong p-value matrix and pairwise comparison table.

    Colour coding:
      - Green  → significant difference  (p < alpha)
      - Red    → non-significant difference
      - Grey   → diagonal (self-comparison)

    Args:
        p_matrix:   N×N DataFrame of p-values (output of delong_matrix).
        df_summary: Pairwise comparison DataFrame (output of delong_matrix).
        alpha:      Significance threshold.

    Returns:
        matplotlib Figure (not yet saved or shown).
    """
    names = p_matrix.index.tolist()
    n = len(names)

    fig, (ax_heat, ax_table) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#1a1a2e")

    # ── Heatmap ──────────────────────────────────────────────────────────────
    ax_heat.set_facecolor("#16213e")

    display = p_matrix.copy().astype(float)
    for i in range(n):
        for j in range(i + 1):  # lower triangle + diagonal → NaN
            display.iloc[i, j] = np.nan

    colors = np.zeros((n, n, 4))
    for i in range(n):
        for j in range(n):
            if i == j:
                colors[i, j] = [0.2, 0.2, 0.3, 1]
            else:
                p = display.iloc[i, j] if i < j else float("nan")
                colors[i, j] = _delong_cell_color(p, alpha)

    ax_heat.imshow(colors, aspect="auto")

    for i in range(n):
        for j in range(i + 1, n):
            p = display.iloc[i, j]
            if not np.isnan(p):
                txt = f"p={p:.3f}" if p >= 0.001 else "p<0.001"
                ax_heat.text(
                    j,
                    i,
                    txt,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                    fontweight="bold",
                )
        ax_heat.text(i, i, "—", ha="center", va="center", fontsize=10, color="#666688")

    ax_heat.set_xticks(range(n))
    ax_heat.set_xticklabels(names, rotation=30, ha="right", color="#aaaacc", fontsize=9)
    ax_heat.set_yticks(range(n))
    ax_heat.set_yticklabels(names, color="#aaaacc", fontsize=9)
    ax_heat.set_title(f"DeLong test p-values  (α = {alpha})", color="white", fontsize=11, pad=10)

    legend_elems = [
        mpatches.Patch(facecolor="#00cc66", label=f"Significant  (p < {alpha})"),
        mpatches.Patch(facecolor="#cc1111", label=f"Non-significant  (p ≥ {alpha})"),
    ]
    ax_heat.legend(
        handles=legend_elems, loc="lower right", facecolor="#1a1a2e", labelcolor="white", fontsize=8
    )

    # ── Summary table ─────────────────────────────────────────────────────────
    ax_table.set_facecolor("#16213e")
    ax_table.axis("off")

    y = 0.97
    ax_table.text(
        0.05,
        y,
        "DeLong pairwise comparisons",
        color="white",
        fontsize=11,
        fontweight="bold",
        transform=ax_table.transAxes,
        va="top",
    )
    y -= 0.06

    headers = ["Model A", "Model B", "ΔAUC", "z", "p-value", "sig."]
    col_x = [0.00, 0.28, 0.56, 0.70, 0.83, 0.96]

    for hdr, x in zip(headers, col_x):
        ax_table.text(
            x + 0.02,
            y,
            hdr,
            color="#aaaacc",
            fontsize=8,
            fontweight="bold",
            transform=ax_table.transAxes,
            va="top",
        )
    y -= 0.04
    ax_table.axhline(y, color="#333355", linewidth=0.8, xmin=0.02, xmax=0.98)
    y -= 0.01

    for _, row in df_summary.iterrows():
        color = "#2ecc71" if row["significant"] else "#e74c3c"
        vals = [
            str(row["Model A"])[:18],
            str(row["Model B"])[:18],
            f"{row['ΔAUC']:+.4f}",
            f"{row['z']:.2f}",
            f"{row['p-value']:.4f}" if row["p-value"] >= 0.001 else "<0.001",
            "Yes" if row["significant"] else "No",
        ]
        for val, x in zip(vals, col_x):
            ax_table.text(
                x + 0.02, y, val, color=color, fontsize=7.5, transform=ax_table.transAxes, va="top"
            )
        y -= 0.055
        if y < 0.02:
            break

    fig.tight_layout(pad=1.5)
    plt.grid(False)
    return fig


# ----------------------------------------------------------------------------
# Learning curves
# ---------------------------------------------------------------------------


def plot_learning_curves(
    results_df: pd.DataFrame,
    metric: str = "f1",
    title: str = "Data efficiency — learning curves",
) -> plt.Figure:

    head_types = results_df["head_type"].unique()
    n_heads = len(head_types)

    fig, axes = plt.subplots(1, n_heads, figsize=(6 * n_heads, 5), sharey=True)
    if n_heads == 1:
        axes = [axes]

    for ax, head in zip(axes, head_types):
        sub = results_df[results_df["head_type"] == head]
        for model_name, grp in sub.groupby("model"):
            grp = grp.sort_values("subset_ratio")
            x = grp["subset_ratio"] * 100
            y = grp[f"{metric}_mean"]
            # Ensure the std column exists (from the np.std we added earlier)
            y_std = grp[f"{metric}_std"] if f"{metric}_std" in grp.columns else 0

            # 1. Plot the main line
            (line,) = ax.plot(x, y, marker="o", label=model_name, linewidth=2)

            # 2. Add the shadow (Standard Deviation)
            # Use the same color as the line with low alpha (transparency)
            ax.fill_between(x, y - y_std, y + y_std, color=line.get_color(), alpha=0.15)

        ax.set_title(f"Head : {head}", fontweight="bold")
        ax.set_xlabel("Training set size (%)")
        ax.set_ylabel(metric.upper() if ax == axes[0] else "")
        ax.set_xticks([10, 25, 50, 75, 100])
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend(loc="lower right", fontsize=8, frameon=True)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig
