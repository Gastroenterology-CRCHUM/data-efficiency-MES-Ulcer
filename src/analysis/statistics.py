"""Post-hoc AUROC comparisons of the trained runs, from their saved test predictions.

A run is one configuration ``(model, pct)`` trained once per seed; its predictions are in
``{model}_{freeze}_{head}[_lr<lr>]_{pct}pct_seed{seed}/test_{labels,probs}.npy``.

``bootstrap_run_aurocs`` computes the AUROC of every network on the full test set and on
seed x patient resamples shared by all runs: every contrast is paired by patient and by
seed, and its interval covers the training variability and the test-set sampling.
Δ = challenger − baseline; positive = challenger better.

    load_test_clip_ids, weighted_aucs, bootstrap_run_aurocs    predictions and bootstrap
    auroc_contrasts, aulc_contrasts                            contrasts, Holm-adjusted p
    auroc_learning_curves                                      label efficiency
    seed_deltas, seed_wilcoxon                                 per-seed sensitivity check
    adjust_pvalues, active_disease_endpoint                    multiplicity, MES endpoint
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from src.data.manifest_utils import load_manifest
from src.utils.logging import get_logger

logger = get_logger(__name__)

RUN_DIR_PATTERN = re.compile(
    r"^(?P<model>.+?)_(?P<freeze>finetune|frozen|freeze\d+)_(?P<head>[A-Za-z0-9]+)"
    r"(?:_lr[^_]+)?_(?P<pct>\d+)pct_seed(?P<seed>\d+)$"
)
"""{model}_{freeze}_{head}[_lr<lr>]_{pct}pct_seed{seed}"""

ENDPOINTS = ("binary", "active")
"""binary: labels and P(class 1); active: MES Mayo >= 2 vs < 2 with score p2 + p3."""


def run_label(model: str, pct: int) -> str:
    """Label of a run: '<model>_<pct>pct'."""
    return f"{model}_{int(pct)}pct"


# ---------------------------------------------------------------------------
# Multiplicity and MES endpoint
# ---------------------------------------------------------------------------


def adjust_pvalues(p_values, method: str | None = "holm") -> np.ndarray:
    """Adjust a family of p-values for multiple comparisons.

    Args:
        p_values: Iterable of raw p-values (NaNs are kept as NaN and excluded
                  from the family).
        method:   Any ``statsmodels.stats.multitest.multipletests`` method
                  ("holm", "bonferroni", "fdr_bh", ...), or None / "none"
                  for no adjustment.

    Returns:
        Array of adjusted p-values (same order as the input).
    """
    p = np.asarray(p_values, dtype=float)
    out = p.copy()
    if method is None or str(method).lower() == "none":
        return out
    mask = np.isfinite(p)
    if mask.any():
        out[mask] = multipletests(p[mask], method=method)[1]
    return out


def _as_prob_matrix(probs: np.ndarray, n: int, name: str) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    if probs.ndim != 2 or probs.shape[0] != n:
        raise ValueError(f"{name} must be an (N={n}, C) probability matrix, got {probs.shape}.")
    return probs


def _check_class_labels(labels: np.ndarray, n_classes: int) -> np.ndarray:
    """Integer class labels in [0, n_classes - 1] (multiclass endpoints)."""
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1-D, got shape {labels.shape}.")
    if not np.isin(labels, np.arange(n_classes)).all():
        raise ValueError(
            f"labels must be integers in [0, {n_classes - 1}], got {np.unique(labels).tolist()}."
        )
    return labels.astype(int)


def active_disease_endpoint(
    labels: np.ndarray,
    probs: np.ndarray,
    cutoff: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Binary 'active disease' endpoint from MES labels and class probabilities.

    Positive = Mayo >= ``cutoff`` (default 2, i.e. Mayo 2-3 vs 0-1);
    score = P(Mayo >= cutoff) = probs[:, cutoff:].sum(axis=1).

    Returns:
        (y_bin, score): int array (N,) and float array (N,).
    """
    probs = _as_prob_matrix(probs, np.shape(labels)[0], "probs")
    labels = _check_class_labels(labels, probs.shape[1])
    if not 1 <= cutoff < probs.shape[1]:
        raise ValueError(f"cutoff must be in [1, {probs.shape[1] - 1}], got {cutoff}.")
    return (labels >= cutoff).astype(int), probs[:, cutoff:].sum(axis=1)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _load_run_predictions(
    preds_dir: Path,
    runs: list[tuple[str, int]],
    head: str,
    freeze: str,
    transform,
) -> tuple[dict[str, dict[int, np.ndarray]], np.ndarray | None]:
    """``transform(labels, probs)`` of every (model, pct) in ``runs``, per training seed.

    All runs must have the same test labels (same test set, same order); labels is None
    when no run is found.
    """
    target = {(m, int(p)) for m, p in runs}
    out: dict[str, dict[int, np.ndarray]] = {}
    ref_labels = None
    for run_dir in sorted(Path(preds_dir).iterdir()):
        m = RUN_DIR_PATTERN.match(run_dir.name)
        if not run_dir.is_dir() or m is None:
            continue
        if m.group("head") != head or m.group("freeze") != freeze:
            continue
        model, pct = m.group("model"), int(m.group("pct"))
        if (model, pct) not in target:
            continue
        labels = np.load(run_dir / "test_labels.npy").astype(int).ravel()
        probs = np.load(run_dir / "test_probs.npy")
        if ref_labels is None:
            ref_labels = labels
        elif not np.array_equal(labels, ref_labels):
            raise ValueError(f"{run_dir.name}: test labels differ from the other runs.")
        label, seed = run_label(model, pct), int(m.group("seed"))
        if seed in out.setdefault(label, {}):
            raise ValueError(f"Several prediction dirs for {label} seed {seed} "
                             "(e.g. learning-rate variants): filter them first.")
        out[label][seed] = transform(labels, probs)
    return out, ref_labels


def load_test_clip_ids(manifest_path: Path, level: str = "patient") -> np.ndarray | None:
    """Cluster id of every test frame in prediction order, or None without manifest.

    level "patient" (default): ``patient_id`` (the procedure folder without
    ``video_data.csv``); "procedure": ``procedure_id`` if present, else ``patient_id``;
    "clip": ``video_id``.
    """
    if level not in ("patient", "procedure", "clip"):
        raise ValueError(f"level must be 'patient', 'procedure' or 'clip', got {level!r}")
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        logger.warning(f"Manifest not found: {manifest_path} (clustered analyses skipped)")
        return None
    test = load_manifest(manifest_path)
    test = test[test["split"] == "test"].reset_index(drop=True)
    if level == "patient":
        return test["patient_id"].astype(str).to_numpy()
    if level == "procedure":
        col = "procedure_id" if "procedure_id" in test.columns else "patient_id"
        return test[col].astype(str).to_numpy()
    return test["video_id"].astype(str).to_numpy()


def _aligned_clip_ids(clip_ids, n_frames: int, task: str) -> np.ndarray | None:
    """``clip_ids`` when they match the predictions, else None (with a warning)."""
    if clip_ids is None:
        return None
    clip_ids = np.asarray(clip_ids)
    if clip_ids.size != n_frames:
        logger.warning(f"[{task}] {clip_ids.size} clip ids for {n_frames} test frames "
                       "(manifest of another split?): clip-level analysis skipped.")
        return None
    return clip_ids


def _endpoint_labels(labels: np.ndarray, endpoint: str) -> np.ndarray:
    if endpoint == "active":
        return (np.asarray(labels) >= 2).astype(int)
    if endpoint == "binary":
        return np.asarray(labels).astype(int)
    raise ValueError(f"endpoint must be one of {ENDPOINTS}, got {endpoint!r}")


def _endpoint_score(labels: np.ndarray, probs: np.ndarray, endpoint: str) -> np.ndarray:
    if endpoint == "active":
        return active_disease_endpoint(labels, probs)[1]
    probs = np.asarray(probs, dtype=float)
    return probs[:, 1] if probs.ndim == 2 else probs


# ---------------------------------------------------------------------------
# Seed x clip bootstrap of the network AUROCs
# ---------------------------------------------------------------------------


def weighted_aucs(y: np.ndarray, score: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """AUROC of ``score`` under each row of frame weights (B, N), mid-ranks for ties.

    Integer weights (bootstrap counts) give the AUROC of the resampled data, weights of 1
    give sklearn's roc_auc_score and weights of 0 drop frames. NaN for a row without
    positive or without negative weight.
    """
    weights = np.atleast_2d(np.asarray(weights, dtype=float))
    order = np.argsort(score, kind="stable")
    s = np.asarray(score, dtype=float)[order]
    starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])  # first frame of each tied score
    w = weights[:, order]
    pos = np.asarray(y)[order] == 1
    w_pos = np.add.reduceat(w * pos, starts, axis=1)
    w_neg = np.add.reduceat(w * ~pos, starts, axis=1)
    below = np.cumsum(w_neg, axis=1) - w_neg  # negative weight strictly below each score
    num = (w_pos * (below + 0.5 * w_neg)).sum(axis=1)
    den = w_pos.sum(axis=1) * w_neg.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


@dataclass
class AurocBootstrap:
    """Test AUROC of every network, on the full test set and on shared seed x patient resamples."""

    task: str
    endpoint: str
    seeds: list[int]
    auroc: dict[str, np.ndarray]  # run -> (n_seeds,) full-test AUROC, in ``seeds`` order
    boot: dict[str, np.ndarray]  # run -> (n_boot, n_seeds) AUROC on each clip resample
    seed_draws: np.ndarray  # (n_boot, n_seeds) indices of the seeds drawn in each resample
    valid: np.ndarray  # (n_boot,) resamples that contain both classes
    scores: dict[str, dict[int, np.ndarray]] = field(repr=False)  # run -> {seed: score (N,)}
    y: np.ndarray = field(repr=False)
    clip_ids: np.ndarray = field(repr=False)

    def run_mean(self, run: str) -> float:
        """Seed-mean AUROC of ``run`` on the full test set."""
        return float(self.auroc[run].mean())

    def boot_mean(self, run: str) -> np.ndarray:
        """Seed-mean AUROC of ``run`` on each resample (over the drawn seeds), (n_boot,)."""
        return np.take_along_axis(self.boot[run], self.seed_draws, axis=1).mean(axis=1)


def bootstrap_run_aurocs(
    preds_dir: Path,
    runs: list[tuple[str, int]],
    clip_ids: np.ndarray | None,
    *,
    task: str = "",
    endpoint: str = "binary",
    head: str = "linear",
    freeze: str = "finetune",
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> AurocBootstrap | None:
    """Test AUROC of every network of ``runs``, on the full test set and on bootstrap resamples.

    Each resample draws the test clips with replacement (every frame of a drawn clip) and
    the seeds with replacement. Both draws are shared by all runs, so every contrast is
    paired by clip and by seed: for one seed, the training subsets are nested across the
    fractions and shared by all models. Resamples without both classes are marked invalid.
    All runs must have the same seeds. None without predictions or aligned ``clip_ids``.
    """
    if endpoint not in ENDPOINTS:
        raise ValueError(f"endpoint must be one of {ENDPOINTS}, got {endpoint!r}")
    scores, labels = _load_run_predictions(
        preds_dir, runs, head, freeze, lambda y, p: _endpoint_score(y, p, endpoint)
    )
    if not scores:
        logger.warning(f"[{task}] No predictions found for the requested runs.")
        return None
    missing = [run_label(m, p) for m, p in runs if run_label(m, p) not in scores]
    if missing:
        logger.warning(f"[{task}] Runs not found: {missing}")
    seeds = sorted(next(iter(scores.values())))
    uneven = [run for run, by_seed in scores.items() if sorted(by_seed) != seeds]
    if uneven:
        raise ValueError(f"Runs without the seeds {seeds} (the bootstrap pairs by seed): {uneven}")
    y = _endpoint_labels(labels, endpoint)
    clip_ids = _aligned_clip_ids(clip_ids, y.size, task)
    if clip_ids is None:
        logger.warning(f"[{task}] No clip ids aligned with the predictions: AUROC bootstrap skipped.")
        return None

    rng = np.random.default_rng(seed)
    _, inv = np.unique(clip_ids, return_inverse=True)
    inv = inv.ravel()
    n_clips = inv.max() + 1
    counts = rng.multinomial(n_clips, np.full(n_clips, 1.0 / n_clips), size=n_bootstrap)
    weights = counts[:, inv].astype(float)  # (n_boot, n_frames)
    valid = (weights @ y > 0) & (weights @ (1 - y) > 0)
    seed_draws = rng.integers(0, len(seeds), size=(n_bootstrap, len(seeds)))

    full = np.ones((1, y.size))
    auroc, boot = {}, {}
    for run, by_seed in scores.items():
        auroc[run] = np.array([weighted_aucs(y, by_seed[s], full)[0] for s in seeds])
        b = np.column_stack([weighted_aucs(y, by_seed[s], weights) for s in seeds])
        b[~valid] = np.nan
        boot[run] = b
    if not valid.all():
        logger.info(f"[{task}] {int((~valid).sum())}/{n_bootstrap} resamples without both classes skipped.")
    return AurocBootstrap(task, endpoint, seeds, auroc, boot, seed_draws, valid, scores, y, clip_ids)


# ---------------------------------------------------------------------------
# Contrasts
# ---------------------------------------------------------------------------


def _inverted_percentile_p(d: np.ndarray) -> float:
    """Two-sided p of H0 Δ = 0 by inverting the percentile interval."""
    return float(min(1.0, 2.0 * min(np.sum(d <= 0) + 1, np.sum(d >= 0) + 1) / (d.size + 1)))


def _finish_contrasts(table: pd.DataFrame, delta: str, bs: AurocBootstrap, method: str,
                      mc_correction: str | None, alpha: float, test: bool = True) -> pd.DataFrame:
    """Adjusted p-values, verdict and attrs of a contrast table (``test=False``: estimates only)."""
    if test:
        table["p_sup_adj"] = adjust_pvalues(table["p_sup"].to_numpy(), mc_correction)
        better = (table["p_sup_adj"] < alpha) & (table[delta] > 0)
        worse = (table["p_sup_adj"] < alpha) & (table[delta] < 0)
        table["verdict"] = np.select([better, worse], ["better", "worse"], "inconclusive")
        p_note = f"{mc_correction}-adjusted over {len(table)} pairs" if mc_correction else "uncorrected"
    else:
        table["verdict"] = "estimate"
        p_note = None
    table.attrs = {
        "method": method,
        "task": bs.task,
        "endpoint": bs.endpoint,
        "baseline": ", ".join(sorted(table["baseline"].unique())),
        "alpha": alpha,
        "tested": test,
        "p_note": p_note,
        "n_boot": int(bs.valid.sum()),
        "n_clips": int(np.unique(bs.clip_ids).size),
        "n_seeds": len(bs.seeds),
    }
    return table


def auroc_contrasts(
    bs: AurocBootstrap,
    pairs: list[tuple[str, str]],
    *,
    mc_correction: str | None = "holm",
    alpha: float = 0.05,
    test: bool = True,
) -> pd.DataFrame:
    """ΔAUROC = challenger − baseline (positive = better) for each (challenger, baseline) run pair.

    Point estimate: difference of the seed-mean AUROCs on the full test set. Interval:
    (1 − alpha) percentile interval of the seed x patient bootstrap.
    Test: two-sided p of H0 ΔAUROC = 0 (inverted percentile interval), adjusted over the
    pairs of the family; verdict "better" / "worse" / "inconclusive".
    ``test=False``: estimates and intervals only (descriptive contrasts, e.g. every label
    fraction vs the full-label baseline): no p-value, verdict "estimate", and the pairs do
    not form a multiplicity family.
    """
    tail = alpha / 2
    rows = []
    for chal, base in pairs:
        absent = [r for r in (chal, base) if r not in bs.auroc]
        if absent:
            logger.warning(f"[{bs.task}] Runs not in the bootstrap: {absent}")
            continue
        d = (bs.boot_mean(chal) - bs.boot_mean(base))[bs.valid]
        lo, hi = np.percentile(d, [100 * tail, 100 * (1 - tail)])
        row = {
            "challenger": chal, "baseline": base,
            "AUROC challenger": bs.run_mean(chal), "AUROC baseline": bs.run_mean(base),
            "ΔAUROC": bs.run_mean(chal) - bs.run_mean(base), "ci_lo": lo, "ci_hi": hi,
        }
        if test:
            row["p_sup"] = _inverted_percentile_p(d)
        rows.append(row)
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return _finish_contrasts(table, "ΔAUROC", bs, "auroc_bootstrap", mc_correction, alpha, test)


# ---------------------------------------------------------------------------
# Learning curves
# ---------------------------------------------------------------------------


def _logit(a: np.ndarray) -> np.ndarray:
    a = np.clip(a, 1e-4, 1 - 1e-4)
    return np.log(a / (1 - a))


def _expit(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def _curve_axis(bs: AurocBootstrap, fractions) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray]:
    """Sorted pcts, log2 realised fraction per (pct, seed) (n_pct, n_seeds), the same for the
    seeds drawn in each valid resample (n_valid, n_pct, n_seeds), and the seed draws.

    ``fractions``: {pct: fraction} or {pct: {seed: fraction}} (realised fraction of each seed).
    """
    pcts = sorted(fractions)
    per_seed = {p: (fractions[p] if isinstance(fractions[p], dict) else dict.fromkeys(bs.seeds, fractions[p]))
                for p in pcts}
    x_seed = np.log2([[per_seed[p][s] for s in bs.seeds] for p in pcts])
    draws = bs.seed_draws[bs.valid]
    return pcts, x_seed, x_seed[:, draws].transpose(1, 0, 2), draws


def _model_seed_aurocs(bs: AurocBootstrap, model: str, pcts: list[int], draws: np.ndarray):
    """AUROC per (pct, seed) on the full test set (n_pct, n_seeds) and, on each valid resample,
    of the drawn seeds (n_valid, n_pct, n_seeds). None when a fraction is missing."""
    runs = [run_label(model, p) for p in pcts]
    if any(r not in bs.auroc for r in runs):
        logger.warning(f"[{bs.task}] {model}: not every fraction in the bootstrap, skipped.")
        return None
    full = np.stack([bs.auroc[r] for r in runs])
    boot = np.stack([bs.boot[r][bs.valid] for r in runs], axis=1)
    return full, np.take_along_axis(boot, np.broadcast_to(draws[:, None, :], boot.shape), axis=2)


def _row_ols(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Intercept and slope of y on x, one least-squares fit per row (..., n_points)."""
    xm, ym = x.mean(axis=-1, keepdims=True), y.mean(axis=-1, keepdims=True)
    slope = ((x - xm) * (y - ym)).sum(axis=-1) / ((x - xm) ** 2).sum(axis=-1)
    return ym[..., 0] - slope * xm[..., 0], slope


def _aulc(x: np.ndarray, auc: np.ndarray) -> np.ndarray:
    """Mean AUROC over the log2-fraction range (trapezoids), one per row (..., n_points)."""
    area = (0.5 * (auc[..., 1:] + auc[..., :-1]) * np.diff(x, axis=-1)).sum(axis=-1)
    return area / (x[..., -1] - x[..., 0])


def _model_aulc(full, boot, x_seed, x_drawn, per_seed: bool) -> tuple[float, np.ndarray]:
    """AULC on the full test set and per resample: of each seed's curve on its own realised
    fractions, then averaged over the seeds (``per_seed``), or of the seed-mean curve placed at
    the seed-mean fractions."""
    if per_seed:
        return (float(_aulc(x_seed.T, full.T).mean()),
                _aulc(x_drawn.transpose(0, 2, 1), boot.transpose(0, 2, 1)).mean(axis=1))
    return float(_aulc(x_seed.mean(axis=1), full.mean(axis=1))), _aulc(x_drawn.mean(axis=2), boot.mean(axis=2))


def aulc_contrasts(
    bs: AurocBootstrap,
    model_pairs: list[tuple[str, str]],
    fractions,
    *,
    per_seed: bool = True,
    mc_correction: str | None = "holm",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """ΔAULC = challenger − baseline for each (challenger, baseline) *model* pair.

    AULC (area under the learning curve) = mean AUROC over the log2-fraction range of
    ``fractions`` ({pct: fraction} or {pct: {seed: fraction}}), by trapezoids on the observed
    points: one number per model over the whole label range, free of any curve shape. With
    ``per_seed`` (default), each seed's curve is integrated over its own realised fractions and
    the AULCs are averaged; otherwise the seed-mean curve is integrated at the seed-mean
    fractions. Interval and two-sided p from the seed x patient bootstrap (the same seeds are drawn
    for both models), adjusted over the pairs.
    """
    pcts, x_seed, x_drawn, draws = _curve_axis(bs, fractions)
    tail = alpha / 2
    cache: dict[str, tuple[float, np.ndarray] | None] = {}

    def aulc(model: str):
        if model not in cache:
            arrays = _model_seed_aurocs(bs, model, pcts, draws)
            cache[model] = None if arrays is None else _model_aulc(*arrays, x_seed, x_drawn, per_seed)
        return cache[model]

    rows = []
    for chal, base in model_pairs:
        a, b = aulc(chal), aulc(base)
        if a is None or b is None:
            continue
        d = a[1] - b[1]
        lo, hi = np.percentile(d, [100 * tail, 100 * (1 - tail)])
        rows.append({"challenger": chal, "baseline": base, "AULC challenger": a[0], "AULC baseline": b[0],
                     "ΔAULC": a[0] - b[0], "ci_lo": lo, "ci_hi": hi, "p_sup": _inverted_percentile_p(d)})
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table = _finish_contrasts(table, "ΔAULC", bs, "aulc_bootstrap", mc_correction, alpha)
    table.attrs["fractions"] = f"{pcts[0]}–{pcts[-1]} %"
    table.attrs["per_seed"] = per_seed
    return table


@dataclass
class LearningCurves:
    """AUROC learning curves of several models (see :func:`auroc_learning_curves`)."""

    summary: pd.DataFrame  # one row per model
    curves: pd.DataFrame  # fitted curve and 95 % band on a grid of fractions, per model
    observed: pd.DataFrame  # seed-mean AUROC per model and fraction, fitted value, residual
    attrs: dict = field(default_factory=dict)


def _log2_crossing(a, b, target_logit, x_lo, x_hi) -> np.ndarray:
    """log2(fraction) where a + b·x reaches ``target_logit``: −inf if the fitted curve is
    already above it at the smallest fraction, +inf if it is still below at the largest."""
    a, b, t = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float), np.asarray(target_logit, float))
    x_lo, x_hi = np.broadcast_to(x_lo, a.shape), np.broadcast_to(x_hi, a.shape)
    out = np.full(a.shape, np.inf)
    out[a + b * x_lo >= t] = -np.inf
    inside = np.isposinf(out) & (a + b * x_hi >= t) & (b > 0)
    out[inside] = (t[inside] - a[inside]) / b[inside]
    return out


def _interp_crossing(x: np.ndarray, y: np.ndarray, target) -> np.ndarray | float:
    """log2(fraction) where the piecewise-linear curve through (x, y) first reaches ``target``.

    Rows of (..., n_points) curves, x increasing. −inf when the curve is already at the target
    at the smallest fraction, +inf when it never reaches it (no extrapolation).
    """
    y2 = np.atleast_2d(np.asarray(y, dtype=float))
    x2 = np.broadcast_to(np.atleast_2d(np.asarray(x, dtype=float)), y2.shape)
    t = np.broadcast_to(np.asarray(target, dtype=float).reshape(-1), (y2.shape[0],))
    reached = y2 >= t[:, None]
    first = reached.argmax(axis=1)
    out = np.full(y2.shape[0], np.inf)
    out[reached.any(axis=1) & (first == 0)] = -np.inf
    mid = np.flatnonzero(reached.any(axis=1) & (first > 0))
    i = first[mid]
    x0, x1, y0, y1 = x2[mid, i - 1], x2[mid, i], y2[mid, i - 1], y2[mid, i]
    out[mid] = x0 + (t[mid] - y0) * (x1 - x0) / (y1 - y0)
    return float(out[0]) if np.ndim(y) == 1 else out


def _median_crossing(c: np.ndarray, axis: int = -1) -> np.ndarray:
    """Median of per-seed log2 crossings (±inf = outside the range). When the two middle seeds
    are −inf and +inf the median is undefined and reported as +inf (not shown in the range)."""
    with np.errstate(invalid="ignore"):
        m = np.median(c, axis=axis)
    return np.where(np.isnan(m), np.inf, m)


def _fraction_text(log2_f: float, x_lo: float, x_hi: float) -> str:
    if np.isneginf(log2_f):
        return f"<{2 ** x_lo:.0%}"
    if np.isposinf(log2_f):
        return f">{2 ** x_hi:.0%}"
    return f"{2 ** log2_f:.0%}"


def auroc_learning_curves(
    bs: AurocBootstrap,
    models: list[str],
    fractions: dict[int, float] | dict[int, dict[int, float]],
    *,
    baseline_model: str | None = None,
    reference_pct: int | None = None,
    crossing: str = "interp",
    per_seed: bool = True,
    mc_correction: str | None = "holm",
    alpha: float = 0.05,
    n_grid: int = 60,
) -> LearningCurves:
    """AUROC learning curve of each model and the labels needed to reach the baseline.

    ``fractions`` gives the realised training fraction of each pct, one value per pct
    ({pct: fraction}) or one per seed ({pct: {seed: fraction}}). With ``per_seed`` (default)
    every (seed, fraction) point is used at its own realised fraction; otherwise the seed-mean
    AUROC of each pct is placed at the seed-mean fraction. Everything is recomputed on every
    seed x patient resample of ``bs`` for the intervals.

    summary (one row per model):
      * auroc_fit_max, slope — continuous curve logit(AUROC) = a + b · log2(fraction), fitted by
        least squares to every (seed, fraction) point (``per_seed``) or to the seed-mean curve;
        slope = logit(AUROC) gain per doubling of the labels. Check the fit with
        ``observed.logit_residual``: curves that plateau leave a systematic residual pattern;
      * aulc — area under the learning curve, mean AUROC over the log2-fraction range
        (trapezoids on the observed points, no curve shape): per seed on its own fractions,
        then averaged (``per_seed``), or of the seed-mean curve;
      * with ``baseline_model``: Δslope and Δaulc vs the baseline (two-sided p, adjusted over
        the models), and the fraction at which each model reaches the baseline's AUROC at
        ``reference_pct`` (default: the largest pct; target = seed-mean logit AUROC), with the
        label saving factor. ``crossing="interp"`` (default) interpolates linearly between the
        observed fractions (logit AUROC vs log2 fraction, first crossing): per seed on its own
        realised fractions, median over the seeds (``per_seed``), or on the seed-mean curve.
        ``crossing="fit"`` uses the fitted lines, the target being the baseline's fitted AUROC.
        Only crossings inside the observed range are estimated; outside it the text reads
        '<min' or '>max' (no extrapolation). The baseline row is the reference.
    curves: fitted AUROC and 95 % band on ``n_grid`` fractions of the observed range.
    observed: seed-mean AUROC per (model, pct) at the seed-mean fraction, fitted value and the
    mean logit residual of the seeds (fit check).
    """
    if crossing not in ("interp", "fit"):
        raise ValueError(f"crossing must be 'interp' or 'fit', got {crossing!r}")
    pcts, x_seed, x_drawn, draws = _curve_axis(bs, fractions)
    x_mean = x_seed.mean(axis=1)
    x_lo, x_hi = x_mean[0], x_mean[-1]
    n_pct = len(pcts)
    tail = alpha / 2

    fits, rows, obs_rows, curve_rows = {}, [], [], []
    grid = np.linspace(x_lo, x_hi, n_grid)
    for model in models:
        arrays = _model_seed_aurocs(bs, model, pcts, draws)
        if arrays is None:
            continue
        full, boot = arrays
        lf, lb = _logit(full), _logit(boot)  # (n_pct, n_seeds), (n_valid, n_pct, n_seeds)
        if per_seed:  # every (seed, fraction) point at its own realised fraction
            a, b = _row_ols(x_seed.ravel(), lf.ravel())
            a_boot, b_boot = _row_ols(x_drawn.reshape(len(lb), -1), lb.reshape(len(lb), -1))
        else:  # the seed-mean curve at the seed-mean fractions
            a, b = _row_ols(x_mean, lf.mean(axis=1))
            a_boot, b_boot = _row_ols(x_drawn.mean(axis=2), lb.mean(axis=2))
        aulc, aulc_boot = _model_aulc(full, boot, x_seed, x_drawn, per_seed)
        fits[model] = {"a": float(a), "b": float(b), "a_boot": a_boot, "b_boot": b_boot,
                       "aulc_boot": aulc_boot, "lf": lf, "lb": lb}
        rows.append({
            "model": model,
            "auroc_fit_max": float(_expit(a + b * x_hi)),
            "slope": float(b),
            "slope_lo": np.percentile(b_boot, 100 * tail),
            "slope_hi": np.percentile(b_boot, 100 * (1 - tail)),
            "aulc": aulc,
            "aulc_lo": np.percentile(aulc_boot, 100 * tail),
            "aulc_hi": np.percentile(aulc_boot, 100 * (1 - tail)),
        })
        residual = (lf - (a + b * x_seed)).mean(axis=1)
        obs_rows += [{"model": model, "pct": p, "fraction": float(2 ** x), "auroc": float(auc),
                      "auroc_fit": float(_expit(a + b * x)), "logit_residual": float(r)}
                     for p, x, auc, r in zip(pcts, x_mean, full.mean(axis=1), residual)]
        band = _expit(a_boot[:, None] + b_boot[:, None] * grid[None, :])
        curve_rows += [{"model": model, "fraction": float(2 ** g), "auroc_fit": float(_expit(a + b * g)),
                        "lo": lo, "hi": hi}
                       for g, lo, hi in zip(grid, np.percentile(band, 100 * tail, axis=0),
                                            np.percentile(band, 100 * (1 - tail), axis=0))]
    summary = pd.DataFrame(rows)
    fit_on = "every (seed, fraction) point" if per_seed else "the seed-mean curve"
    attrs = {"task": bs.task, "endpoint": bs.endpoint, "baseline": baseline_model, "alpha": alpha,
             "x_range": (float(2 ** x_lo), float(2 ** x_hi)), "per_seed": per_seed,
             "model": f"logit(AUROC) = a + b · log2(realised fraction), fitted to {fit_on}"}

    if baseline_model is not None and baseline_model in fits and not summary.empty:
        ref_pct = pcts[-1] if reference_pct is None else reference_pct
        ref_run = run_label(baseline_model, ref_pct)
        ref_idx = pcts.index(ref_pct)
        base = fits[baseline_model]
        if crossing == "interp":  # target = the baseline's seed-mean logit AUROC at the reference
            ref_logit = base["lf"][ref_idx].mean()
            ref_logit_boot = base["lb"][:, ref_idx, :].mean(axis=1)
        else:  # target = its fitted AUROC: both sides smoothed by the same curve model
            ref_logit = base["a"] + base["b"] * x_mean[ref_idx]
            ref_logit_boot = base["a_boot"] + base["b_boot"] * x_drawn[:, ref_idx, :].mean(axis=1)
        for col, key in (("Δslope", "b"), ("Δaulc", "aulc")):
            point = summary.set_index("model")["slope" if key == "b" else "aulc"]
            boot_key = "b_boot" if key == "b" else "aulc_boot"
            d = {m: fits[m][boot_key] - base[boot_key] for m in fits}
            summary[col] = [point[m] - point[baseline_model] for m in summary["model"]]
            summary[f"{col}_lo"] = [np.percentile(d[m], 100 * tail) for m in summary["model"]]
            summary[f"{col}_hi"] = [np.percentile(d[m], 100 * (1 - tail)) for m in summary["model"]]
            p = np.array([np.nan if m == baseline_model else _inverted_percentile_p(d[m])
                          for m in summary["model"]])
            summary[f"p_{col}_adj"] = adjust_pvalues(p, mc_correction)

        ref_f = float(2 ** x_mean[ref_idx])
        texts, f_eq, saving = [], [], []
        for m in summary["model"]:
            if m == baseline_model:
                texts.append("reference")
                f_eq.append(ref_f)
                saving.append(1.0)
                continue
            fm = fits[m]
            if crossing == "fit":
                point = float(_log2_crossing(fm["a"], fm["b"], ref_logit, x_lo, x_hi))
                boot = _log2_crossing(fm["a_boot"], fm["b_boot"], ref_logit_boot, x_lo, x_hi)
            elif per_seed:  # each seed's own curve, then the median over the seeds
                point = float(_median_crossing(_interp_crossing(x_seed.T, fm["lf"].T, ref_logit)))
                n_valid, n_seeds = fm["lb"].shape[0], fm["lb"].shape[2]
                c = _interp_crossing(x_drawn.transpose(0, 2, 1).reshape(-1, n_pct),
                                     fm["lb"].transpose(0, 2, 1).reshape(-1, n_pct),
                                     np.repeat(ref_logit_boot, n_seeds))
                boot = _median_crossing(c.reshape(n_valid, n_seeds), axis=1)
            else:
                point = _interp_crossing(x_mean, fm["lf"].mean(axis=1), ref_logit)
                boot = _interp_crossing(x_drawn.mean(axis=2), fm["lb"].mean(axis=2), ref_logit_boot)
            lo, hi = np.percentile(boot, [100 * tail, 100 * (1 - tail)], method="nearest")
            texts.append(f"{_fraction_text(point, x_lo, x_hi)} [{_fraction_text(lo, x_lo, x_hi)}, "
                         f"{_fraction_text(hi, x_lo, x_hi)}]")
            finite = np.isfinite(point)
            f_eq.append(2 ** point if finite else np.nan)
            saving.append(ref_f / 2 ** point if finite else np.nan)
        summary[f"fraction to reach {ref_run}"] = texts
        summary["fraction_eq"] = f_eq
        summary["label_saving"] = saving
        attrs.update({"reference": ref_run, "reference_auroc": float(_expit(ref_logit)),
                      "reference_auroc_observed": bs.run_mean(ref_run), "crossing": crossing,
                      "reference_fraction": ref_f, "p_note": f"{mc_correction}-adjusted over the models"
                      if mc_correction else "uncorrected"})
    return LearningCurves(summary, pd.DataFrame(curve_rows), pd.DataFrame(obs_rows), attrs)


# ---------------------------------------------------------------------------
# Per-seed sensitivity check
# ---------------------------------------------------------------------------


def seed_deltas(bs: AurocBootstrap, pairs: list[tuple[str, str]], fractions=None) -> pd.DataFrame:
    """Per-seed Δ = challenger − baseline on the full test set, one row per (pair, seed).

    Without ``fractions``: (challenger, baseline) runs, Δ = per-seed AUROC. With ``fractions``
    ({pct: fraction} or {pct: {seed: fraction}}): model pairs, Δ = per-seed AULC (as
    :func:`aulc_contrasts`). Columns: challenger, baseline, measure, seed, Δ.
    """
    if fractions is None:
        measure = "ΔAUROC"

        def values(run: str):
            return bs.auroc.get(run)
    else:
        measure = "ΔAULC"
        pcts, x_seed, _, _ = _curve_axis(bs, fractions)

        def values(model: str):
            runs = [run_label(model, p) for p in pcts]
            if any(r not in bs.auroc for r in runs):
                return None
            return _aulc(x_seed.T, np.stack([bs.auroc[r] for r in runs]).T)

    rows = []
    for chal, base in pairs:
        a, b = values(chal), values(base)
        if a is None or b is None:
            logger.warning(f"[{bs.task}] Not in the bootstrap: {chal} / {base}")
            continue
        rows += [{"challenger": chal, "baseline": base, "measure": measure, "seed": s, "Δ": float(d)}
                 for s, d in zip(bs.seeds, a - b, strict=True)]
    return pd.DataFrame(rows, columns=["challenger", "baseline", "measure", "seed", "Δ"])


def seed_wilcoxon(
    bs: AurocBootstrap,
    pairs: list[tuple[str, str]],
    fractions=None,
    *,
    mc_correction: str | None = "holm",
) -> pd.DataFrame:
    """Exact seed-paired Wilcoxon signed-rank test of :func:`seed_deltas`: a sensitivity check of
    the bootstrap contrasts. The test patients are fixed; with n seeds, p >= 2 / 2**n.
    """
    rows = []
    deltas = seed_deltas(bs, pairs, fractions)
    for (chal, base, measure), g in deltas.groupby(["challenger", "baseline", "measure"], sort=False):
        d = g["Δ"].to_numpy()
        rows.append({"challenger": chal, "baseline": base, "measure": measure,
                     "mean Δ": float(d.mean()), "median Δ": float(np.median(d)),
                     "n_pos": int((d > 0).sum()), "n_seeds": int(d.size),
                     "p_exact": float(wilcoxon(d, method="exact").pvalue),
                     "p_floor": 2.0 / 2 ** d.size})
    table = pd.DataFrame(rows)
    if not table.empty:
        table["p_adj"] = adjust_pvalues(table["p_exact"].to_numpy(), mc_correction)
    return table
