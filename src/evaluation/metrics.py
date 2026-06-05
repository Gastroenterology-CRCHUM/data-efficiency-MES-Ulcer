"""
src/evaluation/metrics.py
-------------------------
Pure metric computation with bootstrap confidence intervals.

No I/O, no MLflow, no plotting — only numerical results.

Public API
----------
compute_metrics_with_ci(labels, preds, probs, n_bootstrap) -> dict
    Frame-level metrics with 95% bootstrap CIs.
    Returns keys "Metric" (formatted str) and "_Metric_mean" (float).


bootstrap_ci(labels, preds, probs, metric_fn, n_bootstrap, ci, seed)
    -> (mean, lower, upper)

fmt(mean, lower, upper) -> "0.876 (0.851–0.901)"
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def fmt(mean: float, lower: float, upper: float) -> str:
    """Format a metric with its confidence interval: '0.876 (0.851–0.901)'."""
    if np.isnan(lower) or np.isnan(upper):
        return f"{mean} (CI unavailable)"
    return f"{mean} ({lower}–{upper})"


# ---------------------------------------------------------------------------
# Scalar metric functions  (uniform signature for bootstrap_ci)
# ---------------------------------------------------------------------------


def _auroc(labels, preds, probs):
    if probs.ndim == 2:
        probs = np.asarray(probs, dtype=float)
        present_classes = np.unique(labels)
        n_present = len(present_classes)

        if n_present < 2:
            return float("nan")  # cannot compute an AUROC

        if n_present < probs.shape[1]:
            probs_partial = probs[:, present_classes]
            row_sums = probs_partial.sum(axis=1, keepdims=True)
            row_sums[row_sums <= 0] = 1.0
            probs_partial = probs_partial / row_sums

            if n_present == 2:
                # Degraded binary case — take the probability of the highest class
                return float(
                    roc_auc_score(
                        (labels == present_classes[1]).astype(int),
                        probs_partial[:, 1],
                    )
                )
            # 3+ classes but not all present
            return float(
                roc_auc_score(
                    labels,
                    probs_partial,
                    multi_class="ovr",
                    average="macro",
                    labels=present_classes,
                )
            )

        row_sums = probs.sum(axis=1, keepdims=True)
        row_sums[row_sums <= 0] = 1.0
        probs = probs / row_sums
        return float(roc_auc_score(labels, probs, multi_class="ovr", average="macro"))

    return float(roc_auc_score(labels, probs))


def _accuracy(labels: np.ndarray, preds: np.ndarray, probs: np.ndarray) -> float:
    return float(accuracy_score(labels, preds))


def _sensitivity(labels, preds, probs):
    avg = "binary" if probs.ndim == 1 else "macro"
    return float(recall_score(labels, preds, average=avg, zero_division=0))


def _precision(labels, preds, probs):
    avg = "binary" if probs.ndim == 1 else "macro"
    return float(precision_score(labels, preds, average=avg, zero_division=0))


def _f1(labels, preds, probs):
    avg = "binary" if probs.ndim == 1 else "macro"
    return float(f1_score(labels, preds, average=avg, zero_division=0))


def _specificity(labels, preds, probs):
    if probs.ndim == 2:
        return float("nan")  # not defined in multiclass
    cm = confusion_matrix(labels, preds)
    if cm.shape == (2, 2):
        tn, fp, _, _ = cm.ravel()
        return float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    return 0.0


METRIC_FNS: dict[str, Callable] = {
    "Accuracy": _accuracy,
    "AUROC": _auroc,
    "Sensitivity": _sensitivity,
    "Specificity": _specificity,
    "Precision": _precision,
    "F1": _f1,
}


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------


def bootstrap_ci(
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    metric_fn: Callable,
    n_bootstrap: int = 1000,
    ci: int = 95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute mean ± bootstrap CI for a single metric.

    Args:
        labels:      Ground-truth binary labels.
        preds:       Binary predictions.
        probs:       Predicted probabilities for the positive class.
        metric_fn:   Callable with signature (labels, preds, probs) -> float.
        n_bootstrap: Number of resampling iterations.
        ci:          Confidence level in % (default 95).
        seed:        Random seed for reproducibility.

    Returns:
        (mean, lower, upper) rounded to 3 decimal places.
    """
    rng = np.random.default_rng(seed)
    n = len(labels)
    scores: list[float] = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        min_classes = 2
        if len(np.unique(labels[idx])) < min_classes:
            continue
        scores.append(metric_fn(labels[idx], preds[idx], probs[idx]))

    if not scores:
        point_estimate = round(float(metric_fn(labels, preds, probs)), 3)
        print("  [warn] bootstrap_ci: no valid samples — CI unavailable, returning point estimate.")
        return point_estimate, float("nan"), float("nan")

    alpha = (100 - ci) / 2
    return (
        round(float(np.mean(scores)), 3),
        round(float(np.percentile(scores, alpha)), 3),
        round(float(np.percentile(scores, 100 - alpha)), 3),
    )


def bootstrap_ci_aggregated(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn,
    n: int = 10_000,
    ci: float = 95,
    seed: int = 42,
) -> tuple[float, float]:
    """Unified bootstrap CI for aggregated metrics.

    Used for clip-level or patient-level aggregated metrics where we need
    to resample from the aggregated data rather than frame-level data.

    Args:
        y_true: Ground-truth labels (aggregated level).
        y_score: Predicted scores (aggregated level).
        metric_fn: Metric function callable(y_true, y_score) -> float.
        n: Number of bootstrap resamples.
        ci: Confidence interval level (default 95%).
        seed: Random seed for reproducibility.

    Returns:
        (lower_bound, upper_bound) at specified confidence level.
    """
    rng = np.random.default_rng(seed)
    scores, size = [], len(y_true)
    for _ in range(n):
        idx = rng.integers(0, size, size=size)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            scores.append(metric_fn(y_true[idx], y_score[idx]))
        except Exception:
            continue

    if not scores:
        return float("nan"), float("nan")

    lower = np.percentile(scores, (100 - ci) / 2)
    upper = np.percentile(scores, 100 - (100 - ci) / 2)
    return lower, upper


# ---------------------------------------------------------------------------
# Frame-level metrics with CI
# ---------------------------------------------------------------------------


def compute_metrics_with_ci(
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    """Compute all metrics in METRIC_FNS with bootstrap CIs.

    Returns:
        Dict with:
          - "Metric"        -> str  formatted as "mean (lower–upper)"
          - "_Metric_mean"  -> float  raw mean (for plots and comparisons)
    """
    result: dict = {}
    for name, fn in METRIC_FNS.items():
        mean, lower, upper = bootstrap_ci(labels, preds, probs, fn, n_bootstrap, seed=seed)
        if np.isnan(mean):
            continue  # ← Specificity silently skipped in multiclass
        result[name] = fmt(mean, lower, upper)
        result[f"_{name}_mean"] = mean
        result[f"_{name}_lower"] = lower
        result[f"_{name}_upper"] = upper
    return result


