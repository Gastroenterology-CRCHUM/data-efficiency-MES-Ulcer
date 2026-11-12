"""Metrics with bootstrap confidence intervals (no I/O, no plotting).

* Every reported value is the point estimate on the full sample; the bootstrap is used only
  for the percentile CI. Nothing is rounded here (see :func:`fmt`).
* The class set is explicit (``range(probs.shape[1])``, ``{0, 1}`` for 1-D scores, or
  ``range(n_classes)``); resamples that miss a class are skipped and counted.
* ``groups`` switches to a cluster bootstrap: whole groups (clips, patients) are resampled.

    compute_metrics_with_ci(labels, preds, probs, ...) -> dict    every metric of METRIC_FNS
    bootstrap_ci(labels, preds, probs, metric_fn, ...) -> (point, lower, upper)
    bootstrap_indices(n, n_bootstrap, seed, groups=None)          row indices of each resample
    fmt(point, lower, upper, decimals=3) -> "0.876 (0.851–0.901)"
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def fmt(mean: float, lower: float, upper: float, decimals: int = 3) -> str:
    """Format a metric with its confidence interval: '0.876 (0.851–0.901)'.

    This is the only place where values are rounded (for display).
    """
    point = f"{float(mean):.{decimals}f}"
    if np.isnan(lower) or np.isnan(upper):
        return f"{point} (CI unavailable)"
    return f"{point} ({float(lower):.{decimals}f}–{float(upper):.{decimals}f})"


# ---------------------------------------------------------------------------
# Scalar metric functions  (uniform signature for bootstrap_ci)
#
# 1-D ``probs``  -> binary task, positive class = 1.
# 2-D ``probs``  -> multiclass task with classes 0..C-1 (C = probs.shape[1]);
#                   macro averages always run over all C classes.
# ---------------------------------------------------------------------------


def _classes_from_probs(probs: np.ndarray) -> np.ndarray:
    """Explicit class set implied by the score array."""
    return np.arange(probs.shape[1]) if probs.ndim == 2 else np.array([0, 1])


def _auroc(labels: np.ndarray, preds: np.ndarray, probs: np.ndarray) -> float:
    """Binary AUROC, or macro one-vs-rest AUROC over all C classes.

    Returns NaN when a class is absent from ``labels`` (the AUROC over the
    full class set is undefined there — it is *not* silently recomputed over
    the classes that happen to be present).
    """
    probs = np.asarray(probs, dtype=float)
    if probs.ndim == 2:
        n_classes = probs.shape[1]
        if n_classes == 2:
            probs = probs[:, 1]
        else:
            if np.unique(labels).size < n_classes:
                return float("nan")
            # Guard against float32 softmax rows that do not sum exactly to 1
            row_sums = probs.sum(axis=1, keepdims=True)
            row_sums[row_sums <= 0] = 1.0
            return float(
                roc_auc_score(
                    labels,
                    probs / row_sums,
                    multi_class="ovr",
                    average="macro",
                    labels=np.arange(n_classes),
                )
            )

    if np.unique(labels).size < 2:
        return float("nan")
    return float(roc_auc_score(labels, probs))


def _accuracy(labels: np.ndarray, preds: np.ndarray, probs: np.ndarray) -> float:
    return float(accuracy_score(labels, preds))


def _averaging_kwargs(probs: np.ndarray) -> dict:
    """sklearn kwargs: binary for 1-D scores, macro over the explicit class set otherwise."""
    if probs.ndim == 1:
        return {"average": "binary", "pos_label": 1}
    return {"average": "macro", "labels": _classes_from_probs(probs)}


def _sensitivity(labels: np.ndarray, preds: np.ndarray, probs: np.ndarray) -> float:
    return float(recall_score(labels, preds, zero_division=0, **_averaging_kwargs(probs)))


def _precision(labels: np.ndarray, preds: np.ndarray, probs: np.ndarray) -> float:
    return float(precision_score(labels, preds, zero_division=0, **_averaging_kwargs(probs)))


def _f1(labels: np.ndarray, preds: np.ndarray, probs: np.ndarray) -> float:
    return float(f1_score(labels, preds, zero_division=0, **_averaging_kwargs(probs)))


def _specificity(labels: np.ndarray, preds: np.ndarray, probs: np.ndarray) -> float:
    if probs.ndim == 2:
        return float("nan")  # not defined in multiclass
    tn, fp, _, _ = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")


METRIC_FNS: dict[str, Callable] = {
    "Accuracy": _accuracy,
    "AUROC": _auroc,
    "Sensitivity": _sensitivity,
    "Specificity": _specificity,
    "Precision": _precision,
    "F1": _f1,
}

# Metrics that are structurally undefined for multiclass scores (omitted from the output)
_BINARY_ONLY = {"Specificity"}


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------


def _normalize_ci(ci: float) -> float:
    """Accept a confidence level as a fraction (0.95) or a percentage (95)."""
    level = float(ci)
    if level > 1:
        level /= 100.0
    if not 0.0 < level < 1.0:
        raise ValueError(f"ci must be in (0, 1) (or (0, 100) as a percentage), got {ci!r}.")
    return level


def _as_int_labels(labels: np.ndarray, name: str = "labels") -> np.ndarray:
    """Return integer class labels, rejecting non-integral values."""
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {labels.shape}.")
    if labels.dtype.kind == "f":
        if not np.all(np.isfinite(labels)) or not np.all(labels == np.round(labels)):
            raise ValueError(f"{name} must contain integer class labels.")
    elif labels.dtype.kind not in "iub":
        raise ValueError(f"{name} must contain integer class labels, got dtype {labels.dtype}.")
    return labels.astype(int)


def _prepare_inputs(
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    n_classes: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate the three aligned arrays and resolve the explicit class set."""
    labels = _as_int_labels(labels)
    preds = np.asarray(preds)
    probs = np.asarray(probs, dtype=float)
    n = labels.shape[0]
    if preds.shape[0] != n or probs.shape[0] != n:
        raise ValueError(
            f"labels, preds and probs must be aligned: got {n}, {preds.shape[0]}, {probs.shape[0]}."
        )
    if probs.ndim not in (1, 2):
        raise ValueError(f"probs must be 1-D (binary) or 2-D (N, C), got shape {probs.shape}.")

    classes = _classes_from_probs(probs)
    if n_classes is not None:
        if probs.ndim == 2 and probs.shape[1] != n_classes:
            raise ValueError(f"n_classes={n_classes} but probs has {probs.shape[1]} columns.")
        if probs.ndim == 1 and n_classes not in (1, 2):
            raise ValueError(f"1-D probs imply a binary task, got n_classes={n_classes}.")
    unknown = np.setdiff1d(np.unique(labels), classes)
    if unknown.size:
        raise ValueError(f"labels contain values {unknown.tolist()} outside {classes.tolist()}.")
    return labels, preds, probs, classes


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------


def _group_members(groups, n: int) -> list[np.ndarray]:
    """Row indices of each distinct group, in first-sorted group order."""
    groups = np.asarray(groups)
    if groups.shape[0] != n:
        raise ValueError(
            f"groups must have one entry per row: got {groups.shape[0]}, expected {n}."
        )
    _, inverse = np.unique(groups, return_inverse=True)
    inverse = inverse.ravel()
    order = np.argsort(inverse, kind="stable")
    counts = np.bincount(inverse)
    return np.split(order, np.cumsum(counts)[:-1])


def bootstrap_indices(
    n: int,
    n_bootstrap: int,
    seed: int = 42,
    groups: np.ndarray | list | None = None,
) -> Iterator[np.ndarray]:
    """Yield the row indices of each bootstrap resample.

    Args:
        n:           Number of rows.
        n_bootstrap: Number of resamples.
        seed:        Seed of the numpy Generator (same seed -> same resamples).
        groups:      Optional group id per row (e.g. clip or patient id).
                     When given, G group ids are drawn with replacement
                     (G = number of distinct groups) and every row of each
                     drawn group is included, so the resample size varies.

    Yields:
        int arrays of row indices.
    """
    rng = np.random.default_rng(seed)
    if groups is None:
        for _ in range(n_bootstrap):
            yield rng.integers(0, n, size=n)
        return

    members = _group_members(groups, n)
    n_groups = len(members)
    for _ in range(n_bootstrap):
        draw = rng.integers(0, n_groups, size=n_groups)
        yield np.concatenate([members[g] for g in draw])


def _bootstrap_distribution(
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    metric_fns: dict[str, Callable],
    classes: np.ndarray,
    n_bootstrap: int,
    seed: int,
    groups,
) -> tuple[dict[str, np.ndarray], int]:
    """Bootstrap distributions of several metrics on shared resamples.

    Resamples that do not contain every class in ``classes`` are skipped.

    Returns:
        ({metric_name: finite bootstrap scores}, n_skipped)
    """
    scores: dict[str, list[float]] = {name: [] for name in metric_fns}
    n_skipped = 0
    for idx in bootstrap_indices(len(labels), n_bootstrap, seed, groups):
        y = labels[idx]
        if np.unique(y).size < classes.size:
            n_skipped += 1
            continue
        p, s = preds[idx], probs[idx]
        for name, fn in metric_fns.items():
            scores[name].append(fn(y, p, s))

    out = {}
    for name, vals in scores.items():
        arr = np.asarray(vals, dtype=float)
        out[name] = arr[np.isfinite(arr)]
    return out, n_skipped


def _percentile_ci(scores: np.ndarray, level: float) -> tuple[float, float]:
    """Percentile interval of a bootstrap distribution (NaN when empty)."""
    if scores.size == 0:
        return float("nan"), float("nan")
    tail = (1.0 - level) / 2.0
    lower, upper = np.percentile(scores, [100.0 * tail, 100.0 * (1.0 - tail)])
    return float(lower), float(upper)


def _log_resampling(n_bootstrap: int, n_skipped: int, n_groups: int | None) -> None:
    unit = f"{n_groups} groups" if n_groups is not None else "rows"
    if n_bootstrap and n_skipped == n_bootstrap:
        logger.warning(
            f"bootstrap: all {n_bootstrap} resamples (of {unit}) missed a class — "
            "CI unavailable, only the point estimate is reported."
        )
    elif n_skipped:
        logger.info(
            f"bootstrap: skipped {n_skipped}/{n_bootstrap} resamples (of {unit}) "
            "that missed a class."
        )


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------


def bootstrap_ci(
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    metric_fn: Callable,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
    *,
    groups: np.ndarray | list | None = None,
    n_classes: int | None = None,
    return_info: bool = False,
) -> tuple[float, float, float] | tuple[float, float, float, dict]:
    """Full-sample point estimate and percentile bootstrap CI for one metric.

    Args:
        labels:      Ground-truth class labels, shape (N,).
        preds:       Hard predictions, shape (N,).
        probs:       Scores: (N,) positive-class probability for a binary task,
                     or (N, C) class probabilities for a multiclass task.
        metric_fn:   Callable with signature (labels, preds, probs) -> float.
        n_bootstrap: Number of resampling iterations.
        ci:          Confidence level, as a fraction (0.95) or a percentage (95).
        seed:        Random seed for reproducibility.
        groups:      Optional cluster id per row (e.g. clip ids).  When given,
                     whole clusters are resampled (cluster bootstrap).
        n_classes:   Size of the explicit class set (default: inferred from
                     ``probs`` — C columns, or 2 for 1-D scores).
        return_info: Also return a dict with the resampling bookkeeping.

    Returns:
        (point, lower, upper) where ``point = metric_fn(labels, preds, probs)``
        on the full sample and (lower, upper) is the percentile CI.  Values are
        not rounded.  The CI is NaN when no resample contains every class.
        With ``return_info=True`` a fourth element is appended:
        {"n_bootstrap", "n_valid", "n_skipped", "ci", "n_groups"}.
    """
    labels, preds, probs, classes = _prepare_inputs(labels, preds, probs, n_classes)
    level = _normalize_ci(ci)
    n_groups = None if groups is None else len(np.unique(np.asarray(groups)))

    point = float(metric_fn(labels, preds, probs))
    dist, n_skipped = _bootstrap_distribution(
        labels, preds, probs, {"metric": metric_fn}, classes, n_bootstrap, seed, groups
    )
    _log_resampling(n_bootstrap, n_skipped, n_groups)
    lower, upper = _percentile_ci(dist["metric"], level)

    if not return_info:
        return point, lower, upper
    info = {
        "n_bootstrap": int(n_bootstrap),
        "n_valid": int(n_bootstrap - n_skipped),
        "n_skipped": int(n_skipped),
        "ci": level,
        "n_groups": n_groups,
    }
    return point, lower, upper, info



def compute_metrics_with_ci(
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
    *,
    ci: float = 0.95,
    groups: np.ndarray | list | None = None,
    n_classes: int | None = None,
) -> dict:
    """Compute all metrics in METRIC_FNS with percentile bootstrap CIs.

    Every metric is evaluated on the same resamples.  Specificity is omitted
    for multiclass scores (2-D ``probs``), where it is not defined.

    Args:
        labels:      Ground-truth class labels, shape (N,).
        preds:       Hard predictions, shape (N,).
        probs:       (N,) positive-class probability (binary) or (N, C)
                     class probabilities (multiclass, classes 0..C-1).
        n_bootstrap: Number of bootstrap resamples.
        seed:        Seed of the resampling generator.
        ci:          Confidence level, fraction (0.95) or percentage (95).
        groups:      Optional cluster id per row (e.g. clip ids) for a
                     cluster bootstrap.
        n_classes:   Size of the explicit class set (default: inferred).

    Returns:
        Dict with, for every metric name ``M``:
          - "M"          -> str    "point (lower–upper)", 3 decimals, for display
          - "_M_mean"    -> float  full-sample point estimate (kept name; NOT a
                                   bootstrap mean)
          - "_M_point"   -> float  same value as "_M_mean"
          - "_M_lower"   -> float  lower percentile bound (NaN if unavailable)
          - "_M_upper"   -> float  upper percentile bound (NaN if unavailable)
        and "_bootstrap" -> {"n_bootstrap", "n_valid", "n_skipped", "ci",
        "n_groups"}.  No value is rounded.
    """
    labels, preds, probs, classes = _prepare_inputs(labels, preds, probs, n_classes)
    level = _normalize_ci(ci)
    n_groups = None if groups is None else len(np.unique(np.asarray(groups)))

    missing = np.setdiff1d(classes, np.unique(labels))
    if missing.size:
        logger.warning(
            f"compute_metrics_with_ci: classes {missing.tolist()} are absent from the "
            "evaluated labels — class-dependent metrics are undefined and no CI can be built."
        )

    fns = {
        name: fn
        for name, fn in METRIC_FNS.items()
        if not (probs.ndim == 2 and name in _BINARY_ONLY)
    }
    dist, n_skipped = _bootstrap_distribution(
        labels, preds, probs, fns, classes, n_bootstrap, seed, groups
    )
    _log_resampling(n_bootstrap, n_skipped, n_groups)

    result: dict = {}
    for name, fn in fns.items():
        point = float(fn(labels, preds, probs))
        lower, upper = _percentile_ci(dist[name], level)
        result[name] = fmt(point, lower, upper)
        result[f"_{name}_mean"] = point
        result[f"_{name}_point"] = point
        result[f"_{name}_lower"] = lower
        result[f"_{name}_upper"] = upper
    result["_bootstrap"] = {
        "n_bootstrap": int(n_bootstrap),
        "n_valid": int(n_bootstrap - n_skipped),
        "n_skipped": int(n_skipped),
        "ci": level,
        "n_groups": n_groups,
    }
    return result
