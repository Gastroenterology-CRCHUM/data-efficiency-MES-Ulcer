"""Saved test predictions of the seeds of a model, and their confusion matrices."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.analysis.catalog import (
    FULL_PCT,
    MES_NUM_CLASSES,
    MODEL_ORDER,
    SECONDARY_MODELS,
    predictions_dir,
    results_dir,
)


def load_seed_predictions(
    task: str, model: str
) -> tuple[pd.DataFrame, list[np.ndarray], np.ndarray]:
    """Per-seed rows at 100 % (sorted by seed), their test probabilities and the labels."""
    per_seed = pd.read_csv(results_dir(task, model) / "results_per_seed.csv")
    full = per_seed[
        per_seed["model"].isin([*MODEL_ORDER, *SECONDARY_MODELS])
        & (per_seed["pct_data"] == FULL_PCT)
        & (per_seed["freeze"] == "finetune")
        & (per_seed["head_type"] == "linear")
    ]
    rows = full[full["model"] == model].sort_values("seed")
    expected = sorted(full["seed"].unique().tolist())  # seeds of the task at 100 %
    if rows["seed"].tolist() != expected:
        raise ValueError(f"{task}/{model}: seeds {rows['seed'].tolist()} at 100%, expected {expected}")
    labels = None
    probs = []
    for seed in rows["seed"]:
        d = predictions_dir(task, model, int(seed))
        y = np.load(d / "test_labels.npy").astype(int).ravel()
        if labels is None:
            labels = y
        elif not np.array_equal(labels, y):
            raise AssertionError(f"{d.name}: test labels differ between seeds")
        probs.append(np.load(d / "test_probs.npy").astype(float))
    return rows.reset_index(drop=True), probs, labels


def seed_confusions(task: str, model: str) -> dict:
    """Row-normalised test confusion matrix of every seed at 100 %, checked against results_per_seed.csv.

    Ulcer: each seed at its own validation-selected threshold; MES: argmax. Returns seeds, fractions
    (n_seeds, k, k), labels and the ulcer thresholds.
    """
    rows, probs, labels = load_seed_predictions(task, model)
    k = 2 if task == "ulcer" else MES_NUM_CLASSES
    fractions, thresholds = [], []
    for (_, r), p in zip(rows.iterrows(), probs, strict=True):
        if task == "ulcer":
            thr = float(r["tuned_threshold"])
            pred = ((p[:, 1] if p.ndim == 2 else p).ravel() >= thr).astype(int)
            thresholds.append(thr)
        else:
            pred = p.argmax(axis=1)
        cm = confusion_matrix(labels, pred, labels=list(range(k)))
        if not np.array_equal(cm, np.bincount(k * labels + pred, minlength=k * k).reshape(k, k)):
            raise AssertionError(f"{task}/{model}: confusion matrix mismatch")
        frac = cm / cm.sum(axis=1, keepdims=True)
        expected = ([r["tuned_specificity"], r["tuned_sensitivity"]] if task == "ulcer"
                    else [r[f"sens_mayo{i}"] for i in range(k)])
        if not np.allclose(np.diag(frac), expected, atol=1e-12):
            raise AssertionError(f"{task}/{model} seed {r['seed']}: CM disagrees with results_per_seed.csv")
        fractions.append(frac)
    return {"seeds": rows["seed"].astype(int).tolist(), "fractions": np.stack(fractions),
            "labels": labels, "thresholds": thresholds}
