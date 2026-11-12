"""Threshold sweep and clip aggregation (src.evaluation.threshold, src.evaluation.aggregation)."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import confusion_matrix, f1_score

from src.evaluation.aggregation import aggregate_frame_to_clip
from src.evaluation.threshold import find_best_threshold, sweep_thresholds


def _scores(n=120, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    p = np.clip(0.5 * y + rng.normal(0.25, 0.2, size=n), 0, 1).round(2)  # with ties
    return y, p


def test_sweep_uses_the_unique_scores_and_matches_sklearn():
    y, p = _scores()
    results = sweep_thresholds(p, y)
    assert [r["threshold"] for r in results] == sorted(np.unique(p).tolist())
    for r in results[::7]:
        pred = (p >= r["threshold"]).astype(int)
        want = confusion_matrix(y, pred, labels=[0, 1])
        np.testing.assert_array_equal(r["confusion_matrix"], want)
        assert r["f1"] == pytest.approx(f1_score(y, pred, zero_division=0))
    assert len({r["roc_auc"] for r in results}) == 1


def test_sweep_grid_and_two_column_input():
    y, p = _scores()
    grid = sweep_thresholds(p, y, n_thresholds=99)
    assert len(grid) == 99
    assert grid[0]["threshold"] == 0.01 and grid[-1]["threshold"] == 0.99
    two_col = sweep_thresholds(np.column_stack([1 - p, p]), y, num_classes=2)
    assert [r["f1"] for r in two_col] == [r["f1"] for r in sweep_thresholds(p, y)]


def test_sweep_rejects_bad_inputs():
    y, p = _scores()
    with pytest.raises(ValueError):
        sweep_thresholds(np.random.default_rng(0).random((y.size, 3)), y, num_classes=3)
    with pytest.raises(ValueError, match="binary"):
        sweep_thresholds(p, np.where(y == 1, 2, 0))
    with pytest.raises(ValueError, match="rows"):
        sweep_thresholds(p[:-1], y)


def test_find_best_threshold_ties_go_to_the_lowest_and_edges_are_flagged():
    results = [
        {"threshold": 0.2, "f1": 0.5},
        {"threshold": 0.4, "f1": 0.8},
        {"threshold": 0.6, "f1": 0.8},
        {"threshold": 0.8, "f1": 0.1},
    ]
    best = find_best_threshold(results)
    assert best["threshold"] == 0.4 and best["at_edge"] is False
    assert "at_edge" not in results[1]  # a copy is returned
    results[-1]["f1"] = 0.9
    edge = find_best_threshold(results)
    assert edge["threshold"] == 0.8 and edge["at_edge"] is True
    with pytest.raises(ValueError):
        find_best_threshold([])
    with pytest.raises(KeyError):
        find_best_threshold(results, metric="auc")


def test_best_threshold_is_the_exact_optimum():
    y, p = _scores(seed=3)
    best = find_best_threshold(sweep_thresholds(p, y))
    brute = max(f1_score(y, (p >= t).astype(int), zero_division=0) for t in np.unique(p))
    assert best["f1"] == pytest.approx(brute)


def test_aggregate_frame_to_clip_mean_prob():
    probs = np.array([0.9, 0.7, 0.2, 0.1, 0.6, 0.3])
    preds = (probs >= 0.5).astype(int)
    labels = np.array([1, 1, 0, 0, 1, 1])
    clips = ["a", "a", "b", "b", "c", "c"]
    out = aggregate_frame_to_clip(probs, preds, labels, clips, "mean_prob-0.5")

    by_clip = dict(zip(out["clip_ids"], zip(out["y_true"], out["y_pred"], out["y_prob_clip"])))
    assert by_clip["a"][:2] == (1, 1) and by_clip["a"][2] == pytest.approx(0.8)
    assert by_clip["b"][:2] == (0, 0)
    assert by_clip["c"][:2] == (1, 0)  # mean 0.45 < 0.5
    assert out["n_clips"] == 3
    np.testing.assert_array_equal(out["confusion_matrix"], [[1, 0], [1, 1]])


# Recent sklearn warns (instead of raising) on a single-class AUROC; the value is NaN either way.
@pytest.mark.filterwarnings("ignore:Only one class is present")
def test_aggregate_single_class_clips():
    probs = np.array([0.9, 0.8, 0.7])
    out = aggregate_frame_to_clip(probs, np.ones(3, int), np.ones(3, int), ["a", "b", "c"])
    assert out["confusion_matrix"].shape == (2, 2)  # labels [0, 1] are explicit
    assert np.isnan(out["roc_auc"])
