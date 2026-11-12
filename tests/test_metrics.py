"""Point estimates and bootstrap CIs (src.evaluation.metrics)."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from src.evaluation.metrics import (
    METRIC_FNS,
    bootstrap_ci,
    bootstrap_indices,
    compute_metrics_with_ci,
    fmt,
)


def _binary(n=300, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    p = 1 / (1 + np.exp(-(2 * y - 1 + rng.normal(size=n))))
    return y, (p >= 0.5).astype(int), p


def _multiclass(n=300, n_classes=4, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_classes, size=n)
    logits = rng.normal(size=(n, n_classes))
    logits[np.arange(n), y] += 1.0
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    return y, probs.argmax(axis=1), probs


def _sklearn_binary(y, pred, p):
    tn = np.sum((y == 0) & (pred == 0))
    fp = np.sum((y == 0) & (pred == 1))
    return {
        "Accuracy": accuracy_score(y, pred),
        "AUROC": roc_auc_score(y, p),
        "Sensitivity": recall_score(y, pred),
        "Specificity": tn / (tn + fp),
        "Precision": precision_score(y, pred),
        "F1": f1_score(y, pred),
    }


def _sklearn_multiclass(y, pred, probs):
    labels = np.arange(probs.shape[1])
    return {
        "Accuracy": accuracy_score(y, pred),
        "AUROC": roc_auc_score(y, probs, multi_class="ovr", average="macro", labels=labels),
        "Sensitivity": recall_score(y, pred, average="macro", labels=labels, zero_division=0),
        "Precision": precision_score(y, pred, average="macro", labels=labels, zero_division=0),
        "F1": f1_score(y, pred, average="macro", labels=labels, zero_division=0),
    }


# ---------------------------------------------------------------------------
# Point estimates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(METRIC_FNS))
def test_bootstrap_point_equals_full_sample_metric_binary(name):
    y, pred, p = _binary()
    point, lower, upper = bootstrap_ci(y, pred, p, METRIC_FNS[name], n_bootstrap=200, seed=1)
    assert point == pytest.approx(_sklearn_binary(y, pred, p)[name], abs=1e-12)
    assert lower <= upper


@pytest.mark.parametrize("name", ["Accuracy", "AUROC", "Sensitivity", "Precision", "F1"])
def test_bootstrap_point_equals_full_sample_metric_multiclass(name):
    y, pred, probs = _multiclass()
    point, lower, upper = bootstrap_ci(y, pred, probs, METRIC_FNS[name], n_bootstrap=200)
    assert point == pytest.approx(_sklearn_multiclass(y, pred, probs)[name], abs=1e-12)
    assert lower <= upper


def test_compute_metrics_with_ci_binary_keys_and_points():
    y, pred, p = _binary()
    res = compute_metrics_with_ci(y, pred, p, n_bootstrap=100, seed=3)
    ref = _sklearn_binary(y, pred, p)

    for name, value in ref.items():
        assert res[f"_{name}_mean"] == res[f"_{name}_point"]
        assert res[f"_{name}_point"] == pytest.approx(value, abs=1e-12)
        assert res[f"_{name}_lower"] <= res[f"_{name}_upper"]
        assert res[name] == fmt(value, res[f"_{name}_lower"], res[f"_{name}_upper"])
    assert res["_bootstrap"] == {
        "n_bootstrap": 100,
        "n_valid": 100,
        "n_skipped": 0,
        "ci": 0.95,
        "n_groups": None,
    }


def test_compute_metrics_with_ci_multiclass_has_no_specificity():
    y, pred, probs = _multiclass()
    res = compute_metrics_with_ci(y, pred, probs, n_bootstrap=100)
    assert "Specificity" not in res and "_Specificity_mean" not in res
    for name, value in _sklearn_multiclass(y, pred, probs).items():
        assert res[f"_{name}_mean"] == pytest.approx(value, abs=1e-12)


def test_point_estimates_do_not_depend_on_the_bootstrap():
    y, pred, p = _binary()
    a = compute_metrics_with_ci(y, pred, p, n_bootstrap=20, seed=0)
    b = compute_metrics_with_ci(y, pred, p, n_bootstrap=30, seed=9, groups=np.arange(y.size) // 5)
    for name in METRIC_FNS:
        assert a[f"_{name}_mean"] == b[f"_{name}_mean"]


def test_ci_accepts_fraction_or_percentage():
    y, pred, p = _binary()
    frac = bootstrap_ci(y, pred, p, METRIC_FNS["AUROC"], n_bootstrap=100, ci=0.9, seed=2)
    pct = bootstrap_ci(y, pred, p, METRIC_FNS["AUROC"], n_bootstrap=100, ci=90, seed=2)
    assert frac == pct
    wide = bootstrap_ci(y, pred, p, METRIC_FNS["AUROC"], n_bootstrap=100, ci=0.99, seed=2)
    assert wide[1] <= frac[1] and wide[2] >= frac[2]
    with pytest.raises(ValueError, match="ci"):
        bootstrap_ci(y, pred, p, METRIC_FNS["AUROC"], ci=0)


# ---------------------------------------------------------------------------
# Class-incomplete resamples
# ---------------------------------------------------------------------------


def test_class_incomplete_resamples_are_skipped_and_counted():
    # Class 3 has a single frame among 30: many row resamples miss it.
    rng = np.random.default_rng(0)
    y = np.r_[rng.integers(0, 3, size=29), 3]
    probs = rng.dirichlet(np.ones(4), size=30)
    pred = probs.argmax(axis=1)
    n_boot, seed = 150, 7

    expected_skipped = sum(
        np.unique(y[idx]).size < 4 for idx in bootstrap_indices(y.size, n_boot, seed)
    )
    assert 0 < expected_skipped < n_boot

    *_, info = bootstrap_ci(
        y, pred, probs, METRIC_FNS["F1"], n_bootstrap=n_boot, seed=seed, return_info=True
    )
    assert info["n_skipped"] == expected_skipped
    assert info["n_valid"] == n_boot - expected_skipped

    res = compute_metrics_with_ci(y, pred, probs, n_bootstrap=n_boot, seed=seed)
    assert res["_bootstrap"]["n_skipped"] == expected_skipped


def test_skipped_resamples_do_not_enter_the_ci():
    # Metric that returns 999 when a class is missing: it must never reach the CI.
    rng = np.random.default_rng(1)
    y = np.r_[np.zeros(20, dtype=int), 1]
    p = rng.random(21)
    pred = (p >= 0.5).astype(int)

    def sentinel(labels, preds, probs):
        return 999.0 if np.unique(labels).size < 2 else 0.5

    point, lower, upper, info = bootstrap_ci(
        y, pred, p, sentinel, n_bootstrap=200, seed=0, return_info=True
    )
    assert info["n_skipped"] > 0
    assert point == 0.5 and lower == 0.5 and upper == 0.5


def test_ci_is_nan_when_every_resample_misses_a_class():
    # 3-column probabilities but class 2 never occurs: every resample is skipped.
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=40)
    probs = rng.dirichlet(np.ones(3), size=40)
    pred = probs.argmax(axis=1)
    res = compute_metrics_with_ci(y, pred, probs, n_bootstrap=50)

    assert res["_bootstrap"]["n_valid"] == 0
    assert np.isnan(res["_Accuracy_lower"]) and np.isnan(res["_Accuracy_upper"])
    assert res["_Accuracy_mean"] == pytest.approx(accuracy_score(y, pred))
    assert res["Accuracy"].endswith("(CI unavailable)")
    # Macro OVR AUROC over the declared class set is undefined without class 2.
    assert np.isnan(res["_AUROC_mean"])


def test_labels_outside_the_class_set_raise():
    y, pred, p = _binary()
    with pytest.raises(ValueError, match="outside"):
        compute_metrics_with_ci(np.r_[y[:-1], 2], pred, p, n_bootstrap=10)
    with pytest.raises(ValueError, match="aligned"):
        compute_metrics_with_ci(y[:-1], pred, p, n_bootstrap=10)
    with pytest.raises(ValueError, match="integer"):
        compute_metrics_with_ci(y + 0.5, pred, p, n_bootstrap=10)
    with pytest.raises(ValueError, match="n_classes"):
        compute_metrics_with_ci(y, pred, p, n_bootstrap=10, n_classes=4)


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------


def test_bootstrap_indices_rows_and_clusters():
    rows = list(bootstrap_indices(10, 5, seed=3))
    assert len(rows) == 5 and all(r.shape == (10,) for r in rows)
    again = list(bootstrap_indices(10, 5, seed=3))
    assert all(np.array_equal(a, b) for a, b in zip(rows, again))

    groups = np.array(["a", "a", "b", "b", "b", "c", "d", "d", "e", "e"])
    members = {g: set(np.flatnonzero(groups == g)) for g in np.unique(groups)}
    for idx in bootstrap_indices(10, 50, seed=1, groups=groups):
        drawn = [groups[i] for i in idx]
        counts = {g: drawn.count(g) for g in set(drawn)}
        for g, c in counts.items():  # whole groups, possibly repeated
            assert c % len(members[g]) == 0
        assert sum(c // len(members[g]) for g, c in counts.items()) == 5  # G draws


def test_cluster_bootstrap_counts_groups():
    y, pred, p = _binary(n=120)
    groups = np.repeat(np.arange(30), 4)
    *_, info = bootstrap_ci(
        y, pred, p, METRIC_FNS["AUROC"], n_bootstrap=100, groups=groups, return_info=True
    )
    assert info["n_groups"] == 30
    with pytest.raises(ValueError, match="groups"):
        bootstrap_ci(y, pred, p, METRIC_FNS["AUROC"], n_bootstrap=10, groups=groups[:-1])



def test_fmt():
    assert fmt(0.87654, 0.8, 0.9) == "0.877 (0.800–0.900)"
    assert fmt(0.5, float("nan"), float("nan")) == "0.500 (CI unavailable)"
    assert fmt(0.5, 0.25, 0.75, decimals=1) == "0.5 (0.2–0.8)"
