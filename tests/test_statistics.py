"""AUROC comparisons from saved test predictions (src.analysis.statistics)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

from src.analysis.statistics import (
    active_disease_endpoint,
    adjust_pvalues,
    aulc_contrasts,
    auroc_contrasts,
    auroc_learning_curves,
    bootstrap_run_aurocs,
    seed_deltas,
    seed_wilcoxon,
    weighted_aucs,
)

SEEDS = (42, 84, 128, 256, 364, 420)
N_CLIPS, FRAMES_PER_CLIP = 12, 10


def _clip_ids() -> np.ndarray:
    return np.repeat([f"clip{c:02d}" for c in range(N_CLIPS)], FRAMES_PER_CLIP)


def _labels() -> np.ndarray:
    return np.repeat(np.arange(N_CLIPS) % 2, FRAMES_PER_CLIP)  # one label per clip


def _probs(signal: float, seed: int) -> np.ndarray:
    """P(class 1): larger ``signal`` = better separated classes, higher AUROC."""
    rng = np.random.default_rng(seed)
    y = _labels()
    return 1 / (1 + np.exp(-(signal * (2 * y - 1) + rng.normal(scale=0.5, size=y.size))))


def _write_run(root, model, pct, seed, probs, labels=None, freeze="finetune", head="linear", lr=""):
    d = root / f"{model}_{freeze}_{head}{lr}_{pct}pct_seed{seed}"
    d.mkdir(parents=True)
    np.save(d / "test_labels.npy", _labels() if labels is None else labels)
    np.save(d / "test_probs.npy", probs)
    return d


@pytest.fixture()
def preds_dir(tmp_path):
    """A weak baseline, a strong challenger at the same fraction and one at 25 %."""
    root = tmp_path / "predictions"
    for seed in SEEDS:
        _write_run(root, "base", 100, seed, _probs(0.5, seed))
        _write_run(root, "good", 100, seed, _probs(3.0, 1000 + seed))
        _write_run(root, "good", 25, seed, _probs(2.5, 2000 + seed))
        _write_run(root, "base", 50, seed, _probs(0.5, 3000 + seed))  # never requested: ignored
    return root


# ---------------------------------------------------------------------------
# AUROC
# ---------------------------------------------------------------------------


def test_weighted_aucs():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=80)
    score = np.round(y * 0.8 + rng.normal(size=80), 1)  # rounded: many ties
    assert weighted_aucs(y, score, np.ones(80))[0] == pytest.approx(roc_auc_score(y, score))

    counts = rng.integers(0, 4, size=80)  # bootstrap-like integer weights
    rep = np.repeat(np.arange(80), counts)
    assert weighted_aucs(y, score, counts)[0] == pytest.approx(roc_auc_score(y[rep], score[rep]))

    keep = np.arange(80) % 3 != 0  # zero weights drop frames
    w = np.vstack([np.ones(80), keep.astype(float)])
    assert weighted_aucs(y, score, w)[1] == pytest.approx(roc_auc_score(y[keep], score[keep]))
    assert np.isnan(weighted_aucs(y, score, (y == 1).astype(float))[0])  # no negative left


def _bs(preds_dir, runs, n_bootstrap=300, **kw):
    return bootstrap_run_aurocs(preds_dir, runs, _clip_ids(), task="Toy", n_bootstrap=n_bootstrap, **kw)


def test_bootstrap_run_aurocs(preds_dir):
    bs = _bs(preds_dir, [("base", 100), ("good", 100), ("good", 25)])
    assert sorted(bs.auroc) == ["base_100pct", "good_100pct", "good_25pct"]
    assert bs.seeds == sorted(SEEDS) and bs.boot["good_25pct"].shape == (300, 6)
    assert bs.auroc["base_100pct"][0] == pytest.approx(roc_auc_score(_labels(), _probs(0.5, 42)))
    assert np.isnan(bs.boot["base_100pct"][~bs.valid]).all()
    assert bs.boot_mean("good_100pct").shape == (300,)

    # Same resamples and seed draws for every run: paired by clip and by seed
    again = _bs(preds_dir, [("base", 100), ("good", 100)])
    assert np.array_equal(again.seed_draws, bs.seed_draws)
    assert np.allclose(again.boot["base_100pct"], bs.boot["base_100pct"], equal_nan=True)


def test_bootstrap_run_aurocs_active_endpoint_and_errors(tmp_path):
    rng = np.random.default_rng(1)
    labels = np.repeat(np.arange(N_CLIPS) % 4, FRAMES_PER_CLIP)  # Mayo 0-3, one per clip
    root = tmp_path / "mes"
    for seed in SEEDS:
        probs = rng.dirichlet(np.ones(4), size=labels.size)
        _write_run(root, "m", 100, seed, probs, labels=labels)
    bs = bootstrap_run_aurocs(root, [("m", 100)], _clip_ids(), endpoint="active", n_bootstrap=50)
    probs = np.load(root / "m_finetune_linear_100pct_seed42" / "test_probs.npy")
    assert np.array_equal(bs.y, (labels >= 2).astype(int))
    assert bs.auroc["m_100pct"][0] == pytest.approx(roc_auc_score(labels >= 2, probs[:, 2:].sum(axis=1)))

    with pytest.raises(ValueError, match="endpoint"):
        bootstrap_run_aurocs(root, [("m", 100)], _clip_ids(), endpoint="macro")
    _write_run(root, "short", 100, 42, rng.dirichlet(np.ones(4), size=labels.size), labels=labels)
    with pytest.raises(ValueError, match="seeds"):
        bootstrap_run_aurocs(root, [("m", 100), ("short", 100)], _clip_ids(), endpoint="active")
    assert bootstrap_run_aurocs(root, [("m", 100)], None, endpoint="active") is None


def test_loading_checks(tmp_path):
    bad = tmp_path / "bad"
    _write_run(bad, "m", 100, 42, _probs(1, 0))
    _write_run(bad, "m", 100, 84, _probs(1, 0), labels=1 - _labels())
    with pytest.raises(ValueError, match="test labels differ"):
        bootstrap_run_aurocs(bad, [("m", 100)], _clip_ids())

    dup = tmp_path / "dup"
    _write_run(dup, "m", 100, 42, _probs(1, 0))
    _write_run(dup, "m", 100, 42, _probs(1, 0), lr="_lr1e-4")
    with pytest.raises(ValueError, match="Several prediction dirs"):
        bootstrap_run_aurocs(dup, [("m", 100)], _clip_ids())

    other = tmp_path / "other"  # another head / freeze label is ignored
    _write_run(other, "m", 100, 42, _probs(1, 0))
    _write_run(other, "m", 100, 84, _probs(1, 1), head="mlp1")
    _write_run(other, "m", 100, 128, _probs(1, 2), freeze="frozen")
    assert bootstrap_run_aurocs(other, [("m", 100)], _clip_ids(), n_bootstrap=10).seeds == [42]


def test_auroc_contrasts_test_and_estimates(preds_dir):
    for seed in SEEDS:  # an exact copy of the baseline: Δ = 0 on every resample
        _write_run(preds_dir, "same", 100, seed, _probs(0.5, seed))
    bs = _bs(preds_dir, [("base", 100), ("good", 100), ("same", 100)])
    pairs = [("good_100pct", "base_100pct"), ("same_100pct", "base_100pct")]
    table = auroc_contrasts(bs, pairs)
    rows = table.set_index("challenger")
    good, same = rows.loc["good_100pct"], rows.loc["same_100pct"]
    assert good["ΔAUROC"] > 0 and good["ci_lo"] > 0 and good["verdict"] == "better"
    assert same["ΔAUROC"] == 0 and same["p_sup"] == 1 and same["verdict"] == "inconclusive"
    assert (rows["p_sup_adj"] >= rows["p_sup"]).all()
    assert table.attrs["n_boot"] == int(bs.valid.sum()) and table.attrs["tested"]
    assert table.attrs["endpoint"] == "binary" and table.attrs["p_note"] == "holm-adjusted over 2 pairs"

    # Descriptive: same estimates and intervals, no p-value and no multiplicity family
    est = auroc_contrasts(bs, pairs, test=False)
    assert not {"p_sup", "p_sup_adj"} & set(est.columns) and (est["verdict"] == "estimate").all()
    pd.testing.assert_frame_equal(est[["ΔAUROC", "ci_lo", "ci_hi"]], table[["ΔAUROC", "ci_lo", "ci_hi"]])
    assert not est.attrs["tested"] and est.attrs["p_note"] is None


def _curve_runs(tmp_path):
    """Learning curves of known shape (AUROC grows with the signal of _probs)."""
    root = tmp_path / "curves"
    fractions = {10: 0.1, 25: 0.25, 50: 0.5, 100: 1.0}
    signals = {
        "steep": lambda f: 0.05 + 0.5 * f,   # baseline: AUROC 0.61 -> 0.94
        "eff": lambda f: 0.35 + 0.25 * f,    # reaches the baseline's 100 % level near 80 %
        "flat": lambda f: 0.15,              # never reaches it
        "top": lambda f: 0.8,                # above it from the smallest fraction
    }
    for k, (model, sig) in enumerate(signals.items()):
        for seed in SEEDS:
            for pct, frac in fractions.items():
                _write_run(root, model, pct, seed, _probs(sig(frac), 10_000 * k + 10 * pct + seed))
    runs = [(m, p) for m in signals for p in fractions]
    return bootstrap_run_aurocs(root, runs, _clip_ids(), n_bootstrap=300), list(signals), fractions


def test_auroc_learning_curves(tmp_path):
    bs, models, fractions = _curve_runs(tmp_path)
    lc = auroc_learning_curves(bs, models, fractions, baseline_model="steep", n_grid=20)
    s = lc.summary.set_index("model")

    assert s.loc["steep", "slope"] > 0 and abs(s.loc["flat", "slope"]) < s.loc["steep", "slope"]
    assert s.loc["steep", "slope_lo"] < s.loc["steep", "slope"] < s.loc["steep", "slope_hi"]
    assert s.loc["flat", "Δslope"] < 0 and s.loc["steep", "Δslope"] == 0
    assert np.isnan(s.loc["steep", "p_Δslope_adj"]) and np.isnan(s.loc["steep", "p_Δaulc_adj"])
    assert s.loc["top", "aulc"] > s.loc["eff", "aulc"] > s.loc["flat", "aulc"]
    assert s.loc["steep", "aulc_lo"] < s.loc["steep", "aulc"] < s.loc["steep", "aulc_hi"]

    # Fraction needed to reach the baseline's AUROC at 100 %: estimated only inside 10-100 %
    col = "fraction to reach steep_100pct"
    assert s.loc["top", col].startswith("<10%") and np.isnan(s.loc["top", "fraction_eq"])
    assert s.loc["flat", col].startswith(">100%") and np.isnan(s.loc["flat", "label_saving"])
    obs = lc.observed.pivot(index="model", columns="pct", values="auroc")
    assert obs.loc["eff", 50] < obs.loc["steep", 100] <= obs.loc["eff", 100]
    assert 0.5 < s.loc["eff", "fraction_eq"] < 1.0  # interpolated between 50 and 100 %
    assert s.loc["eff", "label_saving"] == pytest.approx(1.0 / s.loc["eff", "fraction_eq"])
    assert s.loc["steep", "fraction_eq"] == pytest.approx(1.0)  # the baseline reaches its own level at 100 %
    assert lc.attrs["crossing"] == "interp" and lc.attrs["reference"] == "steep_100pct"
    assert abs(lc.attrs["reference_auroc"] - lc.attrs["reference_auroc_observed"]) < 0.01
    assert lc.attrs["x_range"] == pytest.approx((0.1, 1.0))

    # crossing="fit": fitted lines, the target being the baseline's fitted AUROC
    fit = auroc_learning_curves(bs, models, fractions, baseline_model="steep", crossing="fit").summary
    fit = fit.set_index("model")
    assert fit.loc["steep", "fraction_eq"] == pytest.approx(1.0)
    assert np.isnan(fit.loc["top", "fraction_eq"]) and fit.loc["top", col].startswith("<10%")
    with pytest.raises(ValueError, match="crossing"):
        auroc_learning_curves(bs, models, fractions, crossing="spline")

    assert len(lc.curves) == 4 * 20 and len(lc.observed) == 4 * len(fractions)
    assert lc.curves["fraction"].min() == pytest.approx(0.1) and lc.curves["fraction"].max() == pytest.approx(1.0)
    obs = lc.observed.set_index(["model", "pct"])
    assert obs.loc[("steep", 100), "auroc"] == pytest.approx(bs.run_mean("steep_100pct"))
    assert np.isfinite(lc.observed["logit_residual"]).all()


def test_auroc_learning_curves_per_seed_fractions(tmp_path):
    bs, models, fractions = _curve_runs(tmp_path)
    same = {p: dict.fromkeys(SEEDS, f) for p, f in fractions.items()}
    a = auroc_learning_curves(bs, models, fractions, baseline_model="steep").summary
    b = auroc_learning_curves(bs, models, same, baseline_model="steep").summary
    assert np.allclose(a["slope"], b["slope"]) and np.allclose(a["aulc_lo"], b["aulc_lo"])

    shifted = {p: {s: f * (1.2 if s == 420 else 1.0) for s in SEEDS} for p, f in fractions.items()}
    shifted[100] = dict.fromkeys(SEEDS, 1.0)
    c = auroc_learning_curves(bs, models, shifted, baseline_model="steep").summary
    assert not np.allclose(a["slope"], c["slope"])  # the realised fraction of each seed is used


def test_learning_curves_per_seed_vs_seed_mean(tmp_path):
    bs, models, fractions = _curve_runs(tmp_path)
    kw = dict(baseline_model="steep", mc_correction=None)
    seeds_curve = auroc_learning_curves(bs, models, fractions, per_seed=True, **kw)
    mean_curve = auroc_learning_curves(bs, models, fractions, per_seed=False, **kw)
    a, b = seeds_curve.summary.set_index("model"), mean_curve.summary.set_index("model")
    # Same fractions for every seed: integrating each seed's curve and averaging equals
    # integrating the mean curve, and the pooled fit has the slope of the fit to the means
    assert np.allclose(a["aulc"], b["aulc"]) and np.allclose(a["aulc_lo"], b["aulc_lo"])
    assert np.allclose(a["slope"], b["slope"])
    assert a.loc["steep", "fraction to reach steep_100pct"] == "reference"
    assert a.loc["steep", "fraction_eq"] == 1.0 and a.loc["steep", "label_saving"] == 1.0
    assert seeds_curve.attrs["per_seed"] and not mean_curve.attrs["per_seed"]
    assert "every (seed, fraction) point" in seeds_curve.attrs["model"]

    # Per-seed crossing = median over the seeds of each seed's own crossing
    from src.analysis.statistics import _interp_crossing, _logit

    x = np.log2([fractions[p] for p in sorted(fractions)])
    lf = np.stack([_logit(bs.auroc[f"eff_{p}pct"]) for p in sorted(fractions)])
    target = np.mean(_logit(bs.auroc["steep_100pct"]))
    per_seed = [_interp_crossing(x, lf[:, k], target) for k in range(len(SEEDS))]
    assert a.loc["eff", "fraction_eq"] == pytest.approx(2 ** np.median(per_seed))


def test_learning_curves_use_each_seeds_realised_fraction(tmp_path):
    bs, models, fractions = _curve_runs(tmp_path)
    shifted = {p: {s: f * (1.3 if s == 420 and p < 100 else 1.0) for s in SEEDS} for p, f in fractions.items()}
    seed_mean = aulc_contrasts(bs, [("eff", "steep")], {p: float(np.exp2(np.mean(np.log2(list(v.values())))))
                                                       for p, v in shifted.items()})
    per_seed = aulc_contrasts(bs, [("eff", "steep")], shifted)
    legacy = aulc_contrasts(bs, [("eff", "steep")], shifted, per_seed=False)
    assert per_seed.attrs["per_seed"] and not legacy.attrs["per_seed"]
    # The seed-mean option only sees the mean fraction of each pct; per seed uses each one
    assert legacy["ΔAULC"].iloc[0] == pytest.approx(seed_mean["ΔAULC"].iloc[0])
    assert per_seed["ΔAULC"].iloc[0] != pytest.approx(legacy["ΔAULC"].iloc[0])


def test_aulc_contrasts(tmp_path):
    bs, models, fractions = _curve_runs(tmp_path)
    table = aulc_contrasts(bs, [("eff", "steep"), ("flat", "steep"), ("top", "eff")], fractions)
    rows = table.set_index("challenger")
    lc = auroc_learning_curves(bs, models, fractions, baseline_model="steep").summary.set_index("model")
    assert rows.loc["eff", "ΔAULC"] == pytest.approx(lc.loc["eff", "aulc"] - lc.loc["steep", "aulc"])
    assert rows.loc["eff", "ci_lo"] == pytest.approx(lc.loc["eff", "Δaulc_lo"])  # same resamples
    assert rows.loc["eff", "verdict"] == "better" and rows.loc["top", "ΔAULC"] > 0
    assert (rows["p_sup_adj"] >= rows["p_sup"]).all()
    assert table.attrs["method"] == "aulc_bootstrap" and table.attrs["fractions"] == "10–100 %"


def test_seed_wilcoxon(preds_dir, tmp_path):
    bs = _bs(preds_dir, [("base", 100), ("good", 100), ("good", 25)], n_bootstrap=20)
    t = seed_wilcoxon(bs, [("good_100pct", "base_100pct"), ("good_25pct", "base_100pct")])
    assert list(t["n_pos"]) == [6, 6]  # every seed better: the smallest exact p
    assert t["p_exact"].tolist() == pytest.approx([2 / 64, 2 / 64])
    assert t["p_adj"].tolist() == pytest.approx([4 / 64, 4 / 64])  # Holm over 2
    assert (t["p_floor"] == 2 / 64).all() and (t["measure"] == "ΔAUROC").all()

    d = bs.auroc["base_100pct"] - bs.auroc["good_100pct"]
    d[:2] *= -1  # mixed signs
    bs.auroc["mixed"] = bs.auroc["good_100pct"] + d
    p = seed_wilcoxon(bs, [("mixed", "good_100pct")], mc_correction=None)["p_exact"].iloc[0]
    assert p == pytest.approx(wilcoxon(d, method="exact").pvalue)

    cbs, _, fractions = _curve_runs(tmp_path)
    aulc = seed_wilcoxon(cbs, [("eff", "steep")], fractions).iloc[0]
    x = np.log2([fractions[k] for k in sorted(fractions)])
    per_seed = [np.trapezoid([cbs.auroc[f"{m}_{k}pct"][i] for k in sorted(fractions)], x) / (x[-1] - x[0])
                for m in ("eff", "steep") for i in range(len(SEEDS))]
    a, b = np.array(per_seed[: len(SEEDS)]), np.array(per_seed[len(SEEDS):])
    assert aulc["measure"] == "ΔAULC" and aulc["mean Δ"] == pytest.approx((a - b).mean())


def test_seed_deltas(preds_dir, tmp_path):
    bs = _bs(preds_dir, [("base", 100), ("good", 100)], n_bootstrap=20)
    d = seed_deltas(bs, [("good_100pct", "base_100pct"), ("good_100pct", "missing")])  # missing: skipped
    assert list(d.columns) == ["challenger", "baseline", "measure", "seed", "Δ"]
    assert d["seed"].tolist() == sorted(SEEDS) and (d["measure"] == "ΔAUROC").all()
    assert d["Δ"].to_numpy() == pytest.approx(bs.auroc["good_100pct"] - bs.auroc["base_100pct"])

    cbs, _, fractions = _curve_runs(tmp_path)
    aulc = seed_deltas(cbs, [("eff", "steep")], fractions)
    test = seed_wilcoxon(cbs, [("eff", "steep")], fractions).iloc[0]
    assert (aulc["measure"] == "ΔAULC").all() and aulc["Δ"].mean() == pytest.approx(test["mean Δ"])


def test_plot_comparison_forest_auroc(preds_dir):
    import matplotlib.pyplot as plt

    from src.analysis.figures import plot_comparison_forest

    plt.switch_backend("Agg")
    bs = _bs(preds_dir, [("base", 100), ("good", 100), ("good", 25)], n_bootstrap=100)
    pairs = [("good_25pct", "base_100pct"), ("good_100pct", "base_100pct")]
    fig = plot_comparison_forest(auroc_contrasts(bs, pairs))
    ax = fig.axes[0]
    assert len(ax.patches) == 2 and "seed × patient bootstrap, p holm-adjusted" in ax.get_title()
    plt.close(fig)
    fig = plot_comparison_forest(auroc_contrasts(bs, pairs, test=False))
    assert "descriptive, no test" in fig.axes[0].get_title()
    plt.close(fig)


def test_plot_seed_deltas(tmp_path):
    import matplotlib.pyplot as plt

    from src.analysis.figures import plot_seed_deltas

    plt.switch_backend("Agg")
    bs, _, fractions = _curve_runs(tmp_path)
    pairs = [("eff", "steep"), ("flat", "steep")]
    runs = [(f"{c}_100pct", f"{b}_100pct") for c, b in pairs]
    deltas = pd.concat([seed_deltas(bs, pairs, fractions), seed_deltas(bs, runs)]).assign(family="toy")
    tests = pd.concat([seed_wilcoxon(bs, pairs, fractions), seed_wilcoxon(bs, runs)])
    fig = plot_seed_deltas(deltas, tests, [aulc_contrasts(bs, pairs, fractions), auroc_contrasts(bs, runs)])
    left, right = fig.axes
    assert [t.get_text() for t in left.get_yticklabels()] == ["toy", "eff  vs  steep", "flat  vs  steep"]
    assert right.get_title() == "ΔAUROC at 100 % of the labels"
    assert len(left.collections) == 2 + 2 + 2  # per pair: interval, seed dots, median tick
    assert [t.get_text().split(" ·")[0] for t in left.texts] == [f"{n}/6" for n in tests["n_pos"].iloc[:2]]
    plt.close(fig)

    clash = seed_deltas(bs, [("eff_100pct", "steep_100pct"), ("eff_50pct", "steep_100pct")])
    with pytest.raises(ValueError, match="share a model pair"):
        plot_seed_deltas(clash, seed_wilcoxon(bs, [("eff_100pct", "steep_100pct")]))


def test_plot_learning_curve_fits(tmp_path):
    import matplotlib.pyplot as plt

    from src.analysis.figures import plot_learning_curve_fits

    plt.switch_backend("Agg")
    bs, models, fractions = _curve_runs(tmp_path)
    lc = auroc_learning_curves(bs, models, fractions, baseline_model="steep", n_grid=10)
    fig = plot_learning_curve_fits(lc, ["steep", "eff"])
    ax = fig.axes[0]
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels[:2] == ["steep", "eff"] and labels[2].startswith("steep_100pct")
    assert [t.get_text() for t in ax.get_xticklabels()] == ["10%", "25%", "50%", "100%"]
    plt.close(fig)


def test_adjust_pvalues_holm_matches_statsmodels_and_keeps_nans():
    raw = np.array([0.01, 0.04, np.nan, 0.03, 0.2])
    adj = adjust_pvalues(raw, "holm")
    finite = np.isfinite(raw)
    np.testing.assert_allclose(adj[finite], multipletests(raw[finite], method="holm")[1])
    assert np.isnan(adj[2])
    assert np.all(adj[finite] >= raw[finite])
    np.testing.assert_array_equal(adjust_pvalues(raw, None), raw)
    np.testing.assert_array_equal(adjust_pvalues(raw, "none"), raw)


def _mes_probs(n=240, seed=0):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 4, size=n)
    logits = rng.normal(size=(n, 4))
    logits[np.arange(n), labels] += 1.5
    e = np.exp(logits)
    return labels, e / e.sum(axis=1, keepdims=True)


def test_active_disease_endpoint_definition():
    labels, probs = _mes_probs()
    y_bin, score = active_disease_endpoint(labels, probs)
    np.testing.assert_array_equal(y_bin, (labels >= 2).astype(int))
    np.testing.assert_allclose(score, probs[:, 2] + probs[:, 3])
    y3, s3 = active_disease_endpoint(labels, probs, cutoff=3)
    np.testing.assert_array_equal(y3, (labels == 3).astype(int))
    np.testing.assert_allclose(s3, probs[:, 3])
    with pytest.raises(ValueError, match="cutoff"):
        active_disease_endpoint(labels, probs, cutoff=4)
    with pytest.raises(ValueError):
        active_disease_endpoint(labels, probs[:, 0])  # not a probability matrix
    with pytest.raises(ValueError, match="labels"):
        active_disease_endpoint(np.full(len(labels), 5), probs)  # label outside the classes
