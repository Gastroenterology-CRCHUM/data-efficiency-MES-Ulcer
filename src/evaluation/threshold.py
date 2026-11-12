"""Threshold sweeping for binary classifiers.

    results = sweep_thresholds(probs, labels)            # every unique validation score
    best = find_best_threshold(results, metric="f1")     # row maximising the metric

``n_thresholds=N`` sweeps a uniform grid of N points inside (0, 1) instead.
``find_best_threshold`` flags an optimum at the smallest or largest candidate (``at_edge``).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.metrics import roc_auc_score

from src.utils.logging import get_logger

logger = get_logger(__name__)

def _binary_scores(probs: np.ndarray, num_classes: int) -> np.ndarray:
    """Return 1-D positive-class scores (column 1 of an (N, 2) matrix)."""
    if num_classes > 2:
        raise ValueError(
            f"sweep_thresholds is binary-only, got num_classes={num_classes}. "
            "Multiclass predictions use argmax, not a threshold."
        )
    probs = np.asarray(probs, dtype=float)
    if probs.ndim == 2:
        if probs.shape[1] == 2:
            probs = probs[:, 1]
        elif probs.shape[1] == 1:
            probs = probs[:, 0]
        else:
            raise ValueError(f"Expected (N,), (N, 1) or (N, 2) scores, got {probs.shape}.")
    if probs.ndim != 1:
        raise ValueError(f"Expected 1-D scores, got shape {probs.shape}.")
    return probs


def sweep_thresholds(
    probs: np.ndarray,
    labels: np.ndarray,
    n_thresholds: int | None = None,
    num_classes: int = 1,
) -> list[dict]:
    """Evaluate binary metrics at every candidate threshold (predict 1 iff prob >= t).

    AUROC is threshold-independent: it is computed once and repeated in every
    result dict.

    Args:
        probs:        Positive-class scores, shape (N,) — or (N, 2), in which
                      case column 1 is used.
        labels:       Ground-truth binary labels (0/1), shape (N,).
        n_thresholds: None (default) → candidates are the unique values of
                      ``probs``: every distinct positive-prediction operating
                      point represented by the scores, with no grid truncation.
                      int n → uniform grid of n points strictly inside (0, 1):
                      ``np.linspace(0, 1, n + 2)[1:-1]``
                      (99 → step 0.01; 9999 → step 0.0001).
        num_classes:  1 (sigmoid) or 2 (softmax); > 2 raises ValueError.

    Returns:
        List of dicts sorted by increasing threshold, each containing:
            threshold, accuracy, precision, recall, f1, roc_auc,
            confusion_matrix (2×2, sklearn layout [[tn, fp], [fn, tp]]).
        Precision / F1 use zero_division=0 (as sklearn).  probs / labels are
        NOT stored per entry.
    """
    probs = _binary_scores(probs, num_classes)
    labels = np.asarray(labels)
    if labels.shape[0] != probs.shape[0]:
        raise ValueError(f"probs has {probs.shape[0]} rows, labels have {labels.shape[0]}.")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError(f"labels must be binary 0/1, got {np.unique(labels).tolist()}.")
    labels = labels.astype(int)

    if n_thresholds is None:
        thresholds = np.unique(probs)
    else:
        if n_thresholds < 1:
            raise ValueError(f"n_thresholds must be >= 1, got {n_thresholds}.")
        # Rounded so that e.g. 0.07 is stored and applied as exactly 0.07
        thresholds = np.round(np.linspace(0.0, 1.0, n_thresholds + 2)[1:-1], 12)

    # Compute AUROC once — it is threshold-independent
    roc_auc = float(roc_auc_score(labels, probs)) if np.unique(labels).size == 2 else float("nan")

    # Vectorised confusion counts: predicted positive iff prob >= t
    pos_sorted = np.sort(probs[labels == 1])
    neg_sorted = np.sort(probs[labels == 0])
    n_pos, n_neg = pos_sorted.size, neg_sorted.size
    tp = n_pos - np.searchsorted(pos_sorted, thresholds, side="left")
    fp = n_neg - np.searchsorted(neg_sorted, thresholds, side="left")
    fn = n_pos - tp
    tn = n_neg - fp

    results: list[dict] = []
    for k, t in enumerate(thresholds):
        tp_k, fp_k, fn_k, tn_k = int(tp[k]), int(fp[k]), int(fn[k]), int(tn[k])
        precision = tp_k / (tp_k + fp_k) if (tp_k + fp_k) > 0 else 0.0
        recall = tp_k / n_pos if n_pos > 0 else 0.0
        denom_f1 = 2 * tp_k + fp_k + fn_k
        results.append(
            {
                "threshold": float(t),
                "accuracy": (tp_k + tn_k) / labels.size,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(2 * tp_k / denom_f1) if denom_f1 > 0 else 0.0,
                "roc_auc": roc_auc,
                "confusion_matrix": np.array([[tn_k, fp_k], [fn_k, tp_k]]),
            }
        )

    return results


# ---------------------------------------------------------------------------
# 2. Best-threshold selection
# ---------------------------------------------------------------------------

Metric = Literal["f1", "accuracy", "precision", "recall"]


def find_best_threshold(
    results: list[dict],
    metric: Metric = "f1",
) -> dict:
    """Return the result entry that maximises *metric*.

    Ties resolve to the lowest threshold.  When the best threshold is the
    smallest or largest candidate, a warning is logged: on a grid the optimum
    may lie outside it; with the default unique-score candidates the rule is
    degenerate (predict every sample positive, or every sample negative).

    Args:
        results: Output of :func:`sweep_thresholds`.
        metric:  One of ``"f1"``, ``"accuracy"``, ``"precision"``, ``"recall"``.

    Returns:
        A copy of the dict from *results* with the highest value for *metric*,
        plus ``"at_edge"`` (bool).
    """
    if not results:
        raise ValueError("results list is empty.")
    if metric not in results[0]:
        raise KeyError(f"Metric '{metric}' not found. Available: {list(results[0].keys())}")

    best = dict(max(results, key=lambda r: r[metric]))
    thresholds = [r["threshold"] for r in results]
    lo, hi = min(thresholds), max(thresholds)
    best["at_edge"] = bool(best["threshold"] <= lo or best["threshold"] >= hi)
    if best["at_edge"]:
        rule = ""
        cm = best.get("confusion_matrix")
        if cm is not None:
            n_pred_pos, n_total = int(np.asarray(cm)[:, 1].sum()), int(np.asarray(cm).sum())
            if n_pred_pos == 0:
                rule = " (predicts every sample negative)"
            elif n_pred_pos == n_total:
                rule = " (predicts every sample positive)"
        logger.warning(
            f"find_best_threshold: best {metric} threshold {best['threshold']:.6g} is at the "
            f"edge of the candidate range [{lo:.6g}, {hi:.6g}]{rule}."
        )
    return best
