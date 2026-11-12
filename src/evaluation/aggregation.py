"""Clip-level aggregation of binary frame predictions (mean probability).

A clip is positive when the mean of its frame probabilities is >= T
(``method="mean_prob-T"``). Metrics are point estimates over clips.
"""

from __future__ import annotations

from collections import defaultdict

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


def aggregate_frame_to_clip(
    probabilities: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    clip_ids: list[str],
    method: str = "mean_prob-0.5",
) -> dict:
    """Clip-level predictions and metrics from frame outputs.

    Args:
        probabilities, predictions, labels: 1-D frame arrays.
        clip_ids: Clip of every frame.
        method:   ``"mean_prob-T"``.

    Returns:
        clip_ids, y_true, y_pred, y_prob_clip, accuracy, precision, recall, f1,
        roc_auc, confusion_matrix, method, n_clips, n_frames.
    """
    base, _, param = method.partition("-")
    if base != "mean_prob":
        raise ValueError(f"Unknown method '{method}'; use 'mean_prob-T'.")
    threshold = float(param) if param else 0.5

    clip_probs: dict[str, list[float]] = defaultdict(list)
    clip_labels: dict[str, list[int]] = defaultdict(list)
    for cid, p, y in zip(clip_ids, probabilities, labels):
        clip_probs[cid].append(float(p))
        clip_labels[cid].append(int(y))

    mixed = sum(len(set(v)) > 1 for v in clip_labels.values())
    if mixed:
        logger.warning(f"{mixed} clip(s) have mixed frame labels; clip label = rounded mean.")

    ids = sorted(clip_probs)
    y_true = np.array([int(np.round(np.mean(clip_labels[c]))) for c in ids])
    y_prob = np.array([float(np.mean(clip_probs[c])) for c in ids])
    y_pred = (y_prob >= threshold).astype(int)
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
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]),
        "method": method,
        "n_clips": len(ids),
        "n_frames": len(predictions),
    }
