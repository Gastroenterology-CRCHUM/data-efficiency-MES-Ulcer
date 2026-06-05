"""
src/evaluation/delong.py
------------------------
DeLong test for comparing correlated AUROCs.

Reference: DeLong et al. (1988) — Comparing the areas under two or more
correlated receiver operating characteristic curves: a nonparametric approach.
Biometrics, 44(3), 837-845.

Principle:
  The AUC is equivalent to the Wilcoxon-Mann-Whitney statistic.
  For two models evaluated on the SAME test set, their AUCs are
  correlated (same samples). DeLong accounts for this correlation
  via "placement values" (V10 and V01) to build the variance-covariance
  matrix, then performs a z-test.

Public API
----------
delong_test(labels, probs_a, probs_b)   → (auc_a, auc_b, z_stat, p_value)
delong_matrix(labels, model_probs)      → (p_matrix_df, summary_df)
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _placement_values(
    labels: np.ndarray,
    probs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute placement values V10 and V01 for a classifier.

    V10[i] = fraction of negatives whose score is lower than positive i
             (+ 0.5 × fraction of ties — standard tie-breaking rule).
    V01[j] = fraction of positives whose score is higher than negative j.

    AUC = mean(V10) = mean(V01).

    Args:
        labels: Binary array (0/1), shape (N,).
        probs:  Predicted probabilities, shape (N,).

    Returns:
        V10: shape (n_positives,)
        V01: shape (n_negatives,)
    """
    pos = probs[labels == 1]
    neg = probs[labels == 0]
    n0 = len(neg)
    n1 = len(pos)

    V10 = np.array([((neg < p).sum() + 0.5 * (neg == p).sum()) / n0 for p in pos])
    V01 = np.array([((pos > p).sum() + 0.5 * (pos == p).sum()) / n1 for p in neg])
    return V10, V01


# ---------------------------------------------------------------------------
# DeLong test (pairwise)
# ---------------------------------------------------------------------------


def delong_test(
    labels: np.ndarray,
    probs_a: np.ndarray,
    probs_b: np.ndarray,
) -> tuple[float, float, float, float]:
    """Compare two AUROCs on the same test set (two-sided test).

    H0 : AUC_A = AUC_B
    H1 : AUC_A ≠ AUC_B

    Args:
        labels:  Binary labels (0/1), shape (N,).
        probs_a: Probabilities from model A, shape (N,).
        probs_b: Probabilities from model B, shape (N,).

    Returns:
        auc_a:   AUROC of model A.
        auc_b:   AUROC of model B.
        z_stat:  z-statistic of the test.
        p_value: Two-sided p-value (p < 0.05 → significant difference).
    """
    labels = np.asarray(labels)
    probs_a = np.asarray(probs_a)
    probs_b = np.asarray(probs_b)

    n1 = (labels == 1).sum()
    n0 = (labels == 0).sum()

    V10_a, V01_a = _placement_values(labels, probs_a)
    V10_b, V01_b = _placement_values(labels, probs_b)

    auc_a = float(V10_a.mean())
    auc_b = float(V10_b.mean())

    # Covariance matrix S (2×2)
    # S[i,j] = (1/n1)*cov(V10_i, V10_j) + (1/n0)*cov(V01_i, V01_j)
    V10 = np.stack([V10_a, V10_b])  # (2, n1)
    V01 = np.stack([V01_a, V01_b])  # (2, n0)

    S = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            S[i, j] = (
                np.cov(V10[i], V10[j], bias=False)[0, 1] / n1
                + np.cov(V01[i], V01[j], bias=False)[0, 1] / n0
            )

    # Contrast vector L = [1, -1] tests H0: AUC_A - AUC_B = 0
    L = np.array([1.0, -1.0])
    se = np.sqrt(L @ S @ L)

    if se == 0:
        return auc_a, auc_b, 0.0, 1.0

    z_stat = (auc_a - auc_b) / se
    p_value = 2 * float(stats.norm.sf(abs(z_stat)))

    return auc_a, auc_b, float(z_stat), p_value


# ---------------------------------------------------------------------------
# DeLong matrix (all pairs)
# ---------------------------------------------------------------------------


def delong_matrix(
    labels: np.ndarray,
    model_probs: dict[str, np.ndarray],
    alpha: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the DeLong test to all pairs of models.

    Args:
        labels:      Binary labels (0/1), shape (N,).
        model_probs: {model_name: probs_array}.
        alpha:       Significance threshold (default 0.05).

    Returns:
        p_matrix:   N×N DataFrame of p-values (upper triangle, NaN elsewhere).
        df_summary: DataFrame of pairs sorted by p-value, columns:
                    [Model A, Model B, AUC A, AUC B, ΔAUC, z, p-value, significant].
    """
    names = list(model_probs.keys())
    p_matrix = pd.DataFrame(np.nan, index=names, columns=names)
    rows: list[dict] = []

    for name_a, name_b in itertools.combinations(names, 2):
        auc_a, auc_b, z, p = delong_test(labels, model_probs[name_a], model_probs[name_b])
        p_matrix.loc[name_a, name_b] = round(p, 4)
        rows.append(
            {
                "Model A": name_a,
                "Model B": name_b,
                "AUC A": round(auc_a, 4),
                "AUC B": round(auc_b, 4),
                "ΔAUC": round(auc_a - auc_b, 4),
                "z": round(z, 3),
                "p-value": round(p, 4),
                "significant": p < alpha,
            }
        )

    df_summary = pd.DataFrame(rows).sort_values("p-value").reset_index(drop=True)
    return p_matrix, df_summary
