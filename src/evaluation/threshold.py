"""
src/evaluation/threshold.py
---------------------------
Threshold sweeping for binary classifiers.

Public API
----------
collect_probabilities(model, dataloader, device, num_classes)
    -> tuple[np.ndarray, np.ndarray]
    Run inference once and return (probs, labels).

sweep_thresholds(probs, labels, n_thresholds=99)
    -> list[dict]
    Evaluate every threshold without re-running the model.
    Suitable for both scripts and notebooks (vary n_thresholds).

find_best_threshold(results, metric="f1")
    -> dict
    Return the result dict that maximises the chosen metric.

Typical usage
-------------
# In a script (coarse sweep, default 99 thresholds):
probs, labels = collect_probabilities(model, val_loader, device, num_classes)
results       = sweep_thresholds(probs, labels)
best          = find_best_threshold(results, metric="f1")

# In a notebook (fine-grained sweep):
results = sweep_thresholds(probs, labels, n_thresholds=9999)
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
import tqdm
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# 1. Inference
# ---------------------------------------------------------------------------


def collect_probabilities(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device | str,
    num_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a single inference pass and return per-sample ulcer probabilities.

    Args:
        model:       Trained classifier (supports HuggingFace wrapper).
        dataloader:  Yields (images, labels, *extras) batches.
        device:      Torch device.
        num_classes: 1 → sigmoid output; >1 → softmax, class-1 probability.

    Returns:
        probs:  float32 array of shape (N,) — P(ulcer) for each sample.
        labels: int array   of shape (N,).
    """
    model.eval()
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for images, labels, *_ in tqdm.tqdm(dataloader, desc="Collecting probabilities"):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            if hasattr(logits, "logits"):  # HuggingFace wrapper
                logits = logits.logits

            if num_classes == 1:
                probs = torch.sigmoid(logits.squeeze(1))
                labels = labels.float()
                prob_ulcer = probs
            else:
                probs = torch.softmax(logits, dim=1)
                prob_ulcer = probs[:, 1]

            all_probs.append(prob_ulcer.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    return np.concatenate(all_probs).astype(np.float32), np.concatenate(all_labels).astype(int)


# ---------------------------------------------------------------------------
# 2. Threshold sweep (pure computation — no model, no dataloader)
# ---------------------------------------------------------------------------


def sweep_thresholds(
    probs: np.ndarray, labels: np.ndarray, n_thresholds: int = 80, num_classes: int = 1
) -> list[dict]:
    """Evaluate binary metrics across a uniform threshold grid.

    AUROC and the raw arrays (probs / labels) are computed once and shared
    across all result dicts to avoid redundant computation and memory waste.

    Args:
        probs:        Float array of shape (N,) — predicted positive probabilities.
        labels:       Int array   of shape (N,) — ground-truth binary labels.
        n_thresholds: Number of evenly-spaced thresholds in (0, 1).
                      99  → step 0.01  (fast, for scripts).
                      9999 → step 0.0001 (fine, for notebooks).

    Returns:
        List of dicts, one per threshold, each containing:
            threshold, accuracy, precision, recall, f1,
            roc_auc, confusion_matrix.
        probs / labels are NOT stored per-entry (retrieve from input arrays).
    """
    thresholds = np.linspace(0.1, 0.9, n_thresholds + 1)  # exclude extreme values

    # Compute AUROC once — it is threshold-independent
    # For multiclass (>2 classes), must specify multi_class parameter
    if num_classes > 2:
        roc_auc = roc_auc_score(labels, probs, multi_class="ovr")
    elif num_classes == 2:
        roc_auc = roc_auc_score(labels, probs[:, 1])
    else:
        roc_auc = roc_auc_score(labels, probs)

    results: list[dict] = []
    for t in thresholds:
        preds = (probs >= t).astype(int)

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary", zero_division=0
        )
        results.append(
            {
                "threshold": float(round(t, 6)),
                "accuracy": float(accuracy_score(labels, preds)),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "roc_auc": float(roc_auc),
                "confusion_matrix": confusion_matrix(labels, preds),
            }
        )

    return results


# ---------------------------------------------------------------------------
# 3. Best-threshold selection
# ---------------------------------------------------------------------------

Metric = Literal["f1", "accuracy", "precision", "recall"]


def find_best_threshold(
    results: list[dict],
    metric: Metric = "f1",
) -> dict:
    """Return the result entry that maximises *metric*.

    Args:
        results: Output of :func:`sweep_thresholds`.
        metric:  One of ``"f1"``, ``"accuracy"``, ``"precision"``, ``"recall"``.

    Returns:
        The dict from *results* with the highest value for *metric*.
    """
    if not results:
        raise ValueError("results list is empty.")
    if metric not in results[0]:
        raise KeyError(f"Metric '{metric}' not found. Available: {list(results[0].keys())}")

    return max(results, key=lambda r: r[metric])
