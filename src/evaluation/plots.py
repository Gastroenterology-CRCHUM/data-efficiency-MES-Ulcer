"""Per-run figures of the training runners: ROC curves, confusion matrices, learning curves.

Every function returns the matplotlib Figure; the backend is never changed here (the
runners select a non-interactive one themselves).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from src.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Single-model ROC curve
# ---------------------------------------------------------------------------


def plot_roc_curve(
    name: str,
    labels: np.ndarray,
    probs: np.ndarray,
    threshold: float | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Plot a single ROC curve with an optional operating-point marker.

    Args:
        name:      Model name.
        labels:    Ground-truth binary labels.
        probs:     Predicted probabilities for the positive class.
                   Accepts shape (N,) or (N, 2) — second column is used.
        threshold: If provided, marks the closest point on the curve in red.
        title:     Figure title (default: "ROC Curve — Ulcer detection with <name>").

    Returns:
        matplotlib Figure (not yet saved or shown).
    """
    probs = np.asarray(probs)
    if probs.ndim == 2:
        probs = probs[:, 1]

    fpr, tpr, thresholds = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("1 − Specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title(title if title is not None else f"ROC Curve — Ulcer detection with {name}")
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

    Cell colours (row-normalised rates):
    - Dark blue → high rate
    - White → low rate
    Each cell is annotated with the rate and the raw count.

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


# ----------------------------------------------------------------------------
# Learning curves
# ---------------------------------------------------------------------------

# Columns holding the nominal (planned) training fraction
_NOMINAL_X = ("subset_ratio", "pct_data")
# Label for a missing model / freeze / head_type value (such rows are still drawn)
_MISSING_KEY = "n/a"


def _is_fraction(values: pd.Series) -> bool:
    """True when a column holds fractions in [0, 1] (plotted as %)."""
    arr = values.to_numpy(dtype=float)
    finite = arr[np.isfinite(arr)]
    return bool(finite.size and finite.min() >= 0.0 and finite.max() <= 1.0)


def _x_values(values: pd.Series, as_percent: bool) -> np.ndarray:
    """Plotting values of the x column (x100 when shown as %)."""
    arr = values.to_numpy(dtype=float)
    return arr * 100.0 if as_percent else arr


def _default_x_label(x: str, as_percent: bool) -> str:
    if x in _NOMINAL_X:
        return "Nominal training fraction (%)"
    if x.startswith("n_train"):
        return "Training frames (n)"
    if "realized" in x or "realised" in x:
        return "Realised training fraction (%)" if as_percent else "Realised training size"
    return f"{x} (%)" if as_percent else x


def plot_learning_curves(
    results_df: pd.DataFrame,
    metric: str = "f1",
    title: str = "Data efficiency — learning curves",
    x: str = "subset_ratio",
    *,
    x_label: str | None = None,
    logx: bool = False,
    ylim: tuple[float, float] | None = (0, 1.05),
) -> plt.Figure:
    """Plot mean ± std learning curves from an aggregated results table.

    One panel per ``head_type`` (a single panel if the column is absent) and
    one line per (model, freeze) configuration, so frozen and fine-tuned runs
    of the same backbone are never merged into one line.  Rows with a missing
    ``model`` / ``freeze`` / ``head_type`` are drawn under the label "n/a"
    (with a logged warning) rather than silently dropped.

    Args:
        results_df: Aggregated results with ``{metric}_mean`` (and optionally
                    ``{metric}_std``) plus ``model``, ``freeze``, ``head_type``
                    and the x column.
        metric:     Metric prefix, e.g. "f1" or "auroc".
        title:      Figure title.
        x:          Column used for the x-axis.  Default "subset_ratio" (the
                    nominal fraction).  Any other column works, e.g. a realised
                    fraction or "n_train_mean".  Values in [0, 1] are shown as %.
        x_label:    Axis label override.
        logx:       Log-scale x-axis (useful for 1 %…100 % fractions).
        ylim:       y-axis limits, or None for automatic limits.

    Returns:
        matplotlib Figure (not yet saved or shown).
    """
    y_col, std_col = f"{metric}_mean", f"{metric}_std"
    for col in (x, y_col, "model"):
        if col not in results_df.columns:
            raise KeyError(f"Column '{col}' not found in results_df.")

    df = results_df.copy()
    if "head_type" not in df.columns:
        df["head_type"] = "all"
    group_keys = ["model", "freeze"] if "freeze" in df.columns else ["model"]

    # NaN keys would be dropped by groupby and never match a head panel
    missing = df[[*group_keys, "head_type"]].isna()
    if missing.to_numpy().any():
        counts = {col: int(n) for col, n in missing.sum().items() if n}
        logger.warning(
            f"plot_learning_curves: missing values {counts} in the grouping columns; "
            f"those rows are drawn under '{_MISSING_KEY}'."
        )
        for col in missing.columns[missing.any()]:
            df[col] = df[col].astype(object).where(~missing[col], _MISSING_KEY)

    head_types = df["head_type"].unique()
    n_heads = len(head_types)
    # One scale for the whole figure: fractions in [0, 1] are shown as %
    as_percent = _is_fraction(df[x])

    fig, axes = plt.subplots(1, n_heads, figsize=(6 * n_heads, 5), sharey=True, squeeze=False)
    axes = axes[0]

    for ax, head in zip(axes, head_types):
        sub = df[df["head_type"] == head]
        for key, grp in sub.groupby(group_keys, sort=True, dropna=False):
            key = key if isinstance(key, tuple) else (key,)
            label = str(key[0]) if len(key) == 1 else f"{key[0]} ({key[1]})"
            grp = grp.sort_values(x)
            xs = _x_values(grp[x], as_percent)
            ys = grp[y_col].to_numpy(dtype=float)
            y_std = (
                grp[std_col].fillna(0.0).to_numpy(dtype=float)
                if std_col in grp.columns
                else np.zeros_like(ys)
            )

            (line,) = ax.plot(xs, ys, marker="o", label=label, linewidth=2)
            # Shaded band = ± 1 std across seeds, same colour as the line
            ax.fill_between(xs, ys - y_std, ys + y_std, color=line.get_color(), alpha=0.15)

        ax.set_title(f"Head : {head}", fontweight="bold")
        ax.set_xlabel(x_label if x_label is not None else _default_x_label(x, as_percent))
        ax.set_ylabel(metric.upper() if ax is axes[0] else "")
        if logx:
            ax.set_xscale("log")
        if x in _NOMINAL_X:
            ticks = _x_values(pd.Series(np.sort(sub[x].dropna().unique())), as_percent)
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{t:g}" for t in ticks])
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend(loc="lower right", fontsize=8, frameon=True)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig
