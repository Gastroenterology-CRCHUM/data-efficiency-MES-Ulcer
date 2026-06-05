"""
src/evaluation/aggregation.py
==============================
Clip-level prediction aggregation for ulcer detection.

Aggregation methods
-------------------
    majority_vote       : clip positive if >50% of frame predictions are 1
    mean_prob-T         : clip positive if mean(frame_probs) >= T
    threshold_ratio-T   : clip positive if ratio of positive frames >= T

Public API
----------
    aggregate_by_clip(...)       → dict with clip-level metrics + raw arrays
    compare_aggregation_methods(...)  → DataFrame with all methods ranked by F1
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.evaluation.metrics import bootstrap_ci_aggregated

# Thresholds swept for parametric methods
_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]
# Top-K ratios for temporal smoothing methods
_TOPK_RATIOS = [0.2, 0.3, 0.5]


def _all_methods() -> list[str]:
    methods = ["majority_vote"]
    for base in ("mean_prob", "threshold_ratio", "conf_weighted"):
        for t in _THRESHOLDS:
            methods.append(f"{base}-{t}")
    for ratio in _TOPK_RATIOS:
        methods.append(f"moving_avg_topk-{ratio}")
        methods.append(f"gaussian_topk-{ratio}")
    return methods


def _smooth_moving_avg(probs: np.ndarray) -> np.ndarray:
    """Uniform moving-average smoothing. Window = max(1, n//5)."""
    n = len(probs)
    if n <= 1:
        return probs.copy()
    window = max(1, n // 5)
    kernel = np.ones(window) / window
    return np.convolve(probs, kernel, mode="same")


def _smooth_gaussian(probs: np.ndarray) -> np.ndarray:
    """Gaussian smoothing with sigma = max(1, n/6)."""
    n = len(probs)
    if n <= 1:
        return probs.copy()
    sigma = max(1.0, n / 6.0)
    x = np.arange(n)
    center = (n - 1) / 2
    kernel = np.exp(-0.5 * ((x - center) / sigma) ** 2)
    kernel /= kernel.sum()
    return np.convolve(probs, kernel, mode="same")


# ---------------------------------------------------------------------------
# Core aggregation
# ---------------------------------------------------------------------------


def aggregate_frame_to_clip(
    probabilities: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    clip_ids: list[str],
    method: str = "mean_prob-0.5",
) -> dict:
    """
    Aggregate frame-level outputs to clip-level predictions.

    Args:
        probabilities : 1-D float array — frame-level predicted probabilities.
        predictions   : 1-D int array   — frame-level binary predictions.
        labels        : 1-D int array   — frame-level ground-truth labels.
        clip_ids      : list aligned with the three arrays above.
        method        : aggregation method string (see module docstring).

    Returns:
        Dict with keys:
            clip_ids, y_true, y_pred, y_prob_clip,
            accuracy, precision, recall, f1, roc_auc,
            confusion_matrix, method, n_clips, n_frames.
    """
    # ── Group frames by clip ──────────────────────────────────────────
    clip_data: dict[str, dict] = defaultdict(lambda: {"preds": [], "labels": [], "probs": []})
    for i, cid in enumerate(clip_ids):
        clip_data[cid]["preds"].append(int(predictions[i]))
        clip_data[cid]["labels"].append(int(labels[i]))
        clip_data[cid]["probs"].append(float(probabilities[i]))

    # ── Parse method string ───────────────────────────────────────────
    parts = method.split("-")
    base = parts[0]
    param = float(parts[1]) if len(parts) > 1 else 0.5

    _valid_bases = (
        "majority_vote",
        "mean_prob",
        "threshold_ratio",
        "conf_weighted",
        "moving_avg_topk",
        "gaussian_topk",
    )
    if base not in _valid_bases:
        raise ValueError(f"Unknown method '{method}'. Valid bases: {_valid_bases}.")

    # ── Aggregate per clip ────────────────────────────────────────────
    clip_pred: dict[str, int] = {}
    clip_label: dict[str, int] = {}
    clip_prob: dict[str, float] = {}  # clip score used for AUROC in all methods

    for cid, d in clip_data.items():
        clip_label[cid] = int(np.round(np.mean(d["labels"])))
        probs_arr = np.array(d["probs"])

        if base == "majority_vote":
            clip_prob[cid] = float(np.mean(probs_arr))
            clip_pred[cid] = int(np.round(np.mean(d["preds"])))

        elif base == "mean_prob":
            clip_prob[cid] = float(np.mean(probs_arr))
            clip_pred[cid] = int(clip_prob[cid] >= param)

        elif base == "threshold_ratio":
            clip_prob[cid] = float(np.mean(probs_arr))
            clip_pred[cid] = int(np.mean(d["preds"]) >= param)

        elif base == "conf_weighted":
            # Weight each frame by its distance from 0.5 (confidence)
            weights = np.abs(probs_arr - 0.5) * 2.0
            w_sum = weights.sum()
            score = (
                float(np.dot(probs_arr, weights) / w_sum)
                if w_sum > 1e-9
                else float(np.mean(probs_arr))
            )
            clip_prob[cid] = score
            clip_pred[cid] = int(score >= param)

        elif base == "moving_avg_topk":
            # Temporal smoothing (moving average) then top-K mean
            ratio = param
            smoothed = _smooth_moving_avg(probs_arr)
            k = max(1, int(np.ceil(len(probs_arr) * ratio)))
            top_k_mean = float(np.sort(smoothed)[-k:].mean())
            clip_prob[cid] = top_k_mean
            clip_pred[cid] = int(top_k_mean >= 0.5)

        elif base == "gaussian_topk":
            # Temporal smoothing (Gaussian) then top-K mean
            ratio = param
            smoothed = _smooth_gaussian(probs_arr)
            k = max(1, int(np.ceil(len(probs_arr) * ratio)))
            top_k_mean = float(np.sort(smoothed)[-k:].mean())
            clip_prob[cid] = top_k_mean
            clip_pred[cid] = int(top_k_mean >= 0.5)

    # ── Build output arrays ───────────────────────────────────────────
    ids = sorted(clip_pred.keys())
    y_true = np.array([clip_label[c] for c in ids])
    y_pred = np.array([clip_pred[c] for c in ids])
    y_prob = np.array([clip_prob[c] for c in ids])

    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = float("nan")

    return {
        "clip_ids": ids,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob_clip": y_prob,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc,
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "method": method,
        "n_clips": len(ids),
        "n_frames": len(predictions),
    }


def compare_aggregation_methods(
    probabilities: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    clip_ids: list[str],
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run all aggregation methods and return a ranked DataFrame.

    Columns: method, f1, f1_95ci, auroc, auroc_95ci,
             sensitivity, specificity, precision, recall, n_clips.

    Args:
        probabilities : 1-D frame probability array.
        predictions   : 1-D frame binary prediction array.
        labels        : 1-D frame ground-truth array.
        clip_ids      : clip identifier for every frame.
        n_bootstrap   : bootstrap iterations for CIs.

    Returns:
        DataFrame sorted by F1 descending.
    """
    rows = []
    for method in _all_methods():
        res = aggregate_frame_to_clip(probabilities, predictions, labels, clip_ids, method)
        y_true = res["y_true"]
        y_pred = res["y_pred"]
        y_prob = res["y_prob_clip"]

        f1_lo, f1_hi = bootstrap_ci_aggregated(
            y_true,
            y_pred,
            lambda a, b: f1_score(a, b, zero_division=0),
            n=n_bootstrap,
            seed=seed,
        )
        auc_lo, auc_hi = bootstrap_ci_aggregated(
            y_true, y_prob, roc_auc_score, n=n_bootstrap, seed=seed
        )

        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
            specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
        else:
            sensitivity = specificity = float("nan")

        rows.append(
            {
                "method": method,
                "f1": round(res["f1"], 4),
                "f1_95ci": f"({f1_lo:.4f}-{f1_hi:.4f})",
                "auroc": round(res["roc_auc"], 4),
                "auroc_95ci": f"({auc_lo:.4f}-{auc_hi:.4f})",
                "sensitivity": round(sensitivity, 4),
                "specificity": round(specificity, 4),
                "precision": round(res["precision"], 4),
                "recall": round(res["recall"], 4),
                "n_clips": res["n_clips"],
            }
        )

    return pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
