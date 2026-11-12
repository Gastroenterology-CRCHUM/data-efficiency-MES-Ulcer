"""Figure helpers of the analysis notebooks: learning curves, confusion matrices, contrasts.

Every ``plot_*`` function returns the matplotlib Figure; the backend is never changed here.
"""

from __future__ import annotations

import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from src.analysis.catalog import COLORS, MARKERS, model_label, short_arch
from src.config.models import MODEL_REGISTRY

# ---------------------------------------------------------------------------
# Learning curves (mean ± SD over the seeds)
# ---------------------------------------------------------------------------


def model_style(key: str) -> dict:
    """Line2D style of a model: colour, architecture marker, ImageNet dashed with open markers."""
    e = MODEL_REGISTRY[key]
    imagenet = e.pretrain_data == "ImageNet"
    return {
        "color": COLORS[key],
        "marker": MARKERS.get(short_arch(e.architecture), "o"),
        "linestyle": "--" if imagenet else "-",
        "markerfacecolor": "white" if imagenet else COLORS[key],
        "markeredgecolor": COLORS[key],
    }


def draw_curve(ax, sub: pd.DataFrame, key: str, metric: str, band_alpha: float) -> None:
    """Mean ± SD over the seeds of one model, at its mean realised fraction (%)."""
    st = model_style(key)
    sub = sub.sort_values("pct_data")
    x = 100 * sub["realized_fraction_mean"].to_numpy(dtype=float)
    mean = sub[f"{metric}_mean"].to_numpy(dtype=float)
    sd = sub[f"{metric}_std"].to_numpy(dtype=float)
    ax.fill_between(x, mean - sd, mean + sd, color=st["color"], alpha=band_alpha, linewidth=0, zorder=2)
    ax.plot(x, mean, linewidth=2, markersize=5.5, zorder=3, label=model_label(key), **st)


def x_ticks(df: pd.DataFrame) -> tuple[list[float], list[str]]:
    """Ticks at the mean realised fraction of each nominal fraction ('realised (nominal)' when they differ).

    Raises when the models do not share the realised fraction of a nominal fraction (the training
    subsets of a seed are shared by every model).
    """
    r = df.groupby("pct_data")["realized_fraction_mean"].agg(["mean", "min", "max"]).sort_index()
    if (r["max"] - r["min"]).max() > 1e-9:
        raise AssertionError("realised fractions differ between models at one fraction")
    ticks = (100 * r["mean"]).tolist()
    labels = [f"{t:.0f}" if round(t) == p else f"{t:.0f}\n({p})" for p, t in zip(r.index, ticks, strict=True)]
    return ticks, labels


def y_limits(df: pd.DataFrame, metric: str) -> tuple[float, float]:
    """Zoom on the curves and their SD bands (padding 5 % of the range), within [0, 1]."""
    lo = float((df[f"{metric}_mean"] - df[f"{metric}_std"]).min())
    hi = float((df[f"{metric}_mean"] + df[f"{metric}_std"]).max())
    pad = 0.05 * (hi - lo)
    return max(0.0, lo - pad), min(1.0, hi + pad)


def curve_axes(ax, title: str, xlabel: str, ylabel: str | None, ticks, ylim,
               chance: float | None = 0.5) -> None:
    """Log x axis at the realised fractions, chance line, title and labels of a learning-curve panel."""
    ax.set_xscale("log")
    ax.set_xticks(ticks[0], ticks[1], fontsize=8.5)
    ax.minorticks_off()
    ax.set_xlim(min(ticks[0]) * 0.8, max(ticks[0]) * 1.15)
    ax.set_ylim(*ylim)
    if chance is not None and ylim[0] < chance < ylim[1]:
        ax.axhline(chance, color="#8C8C8C", linestyle=":", linewidth=1, zorder=0)
        ax.annotate("chance", xy=(1, chance), xycoords=("axes fraction", "data"), xytext=(-2, 3),
                    textcoords="offset points", ha="right", fontsize=8, color="#6E6E6E")
    ax.set_title(title, fontsize=10.5, fontweight="bold")
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(linestyle="--", color="#D9D9D9", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


# ---------------------------------------------------------------------------
# Confusion matrices (mean ± SD over the seeds)
# ---------------------------------------------------------------------------


def _class_ticks(ax, names: list[str], n: np.ndarray, show_y: bool) -> None:
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([f"{c}\n(n = {int(v):,})" for c, v in zip(names, n, strict=True)] if show_y else [])
    ax.set_xlabel("Predicted")
    if show_y:
        ax.set_ylabel("True")


def draw_mean_cm(ax, mean: np.ndarray, sd: np.ndarray, names, n, title: str, fs: float, show_y: bool):
    """Row-normalised confusion matrix: mean % ± SD over the seeds in each cell."""
    im = ax.imshow(mean, cmap="Blues", vmin=0, vmax=1)
    for i in range(mean.shape[0]):
        for j in range(mean.shape[1]):
            ax.text(j, i, f"{100 * mean[i, j]:.1f}%\n± {100 * sd[i, j]:.1f}", ha="center", va="center",
                    fontsize=fs, fontweight="bold", color="white" if mean[i, j] > 0.5 else "black")
    _class_ticks(ax, names, n, show_y)
    ax.set_title(title, fontsize=10)
    ax.grid(False)
    return im


# ---------------------------------------------------------------------------
# Run comparison forest plot
# ---------------------------------------------------------------------------

_VERDICT_COLORS = {"better": "#2ecc71", "worse": "#e74c3c", "estimate": "#5B9BE0"}


def plot_comparison_forest(table: pd.DataFrame) -> plt.Figure:
    """Forest plot of an ``auroc_contrasts`` or ``aulc_contrasts`` table.

    One bar per (challenger, baseline) pair: Δ = challenger − baseline (positive = better),
    coloured by the adjusted verdict (blue for descriptive estimates, ``test=False``), with
    its 95 % seed × patient bootstrap interval.

    Args:
        table: Output of ``auroc_contrasts`` or ``aulc_contrasts`` (uses its ``attrs``).

    Returns:
        matplotlib Figure (not yet saved or shown).
    """
    a = table.attrs
    delta = "ΔAULC" if "ΔAULC" in table.columns else "ΔAUROC"
    df = table.sort_values(delta).reset_index(drop=True)
    several = df["baseline"].nunique() > 1
    names = (df["challenger"] + "  vs  " + df["baseline"]) if several else df["challenger"]
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(9, max(3, len(df) * 0.55 + 1.5)))
    ax.barh(y, df[delta], color=[_VERDICT_COLORS.get(v, "#95a5a6") for v in df["verdict"]],
            alpha=0.75, zorder=3)
    ax.errorbar(df[delta], y, xerr=[df[delta] - df["ci_lo"], df["ci_hi"] - df[delta]],
                fmt="none", color="black", capsize=4, linewidth=1.5, zorder=4,
                label="95% CI, seeds × clips (bootstrap)")
    ax.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(names.values, fontsize=9)
    ax.set_xlabel(f"{delta}  (challenger − baseline)  —  positive = better")
    title = f"{a.get('task', '')} — {delta[1:]}" + ("" if several else f" vs {a.get('baseline', '')}")
    inference = f"p {a.get('p_note', '')}" if a.get("tested", True) else "descriptive, no test"
    ax.set_title(f"{title}\n(seed × patient bootstrap, {inference})", fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Per-seed contrasts (seed-paired Wilcoxon)
# ---------------------------------------------------------------------------

_SIGN_COLORS = ("#2a78d6", "#e34948", "#898781")  # Δ > 0 (challenger better), Δ < 0, Δ = 0
_INK, _INK_2, _GRID, _RANGE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"


def _pair_model(name: str) -> str:
    """Model of a pair member: an AUROC run 'resnet50_5M_100pct' -> 'resnet50_5M'."""
    return re.sub(r"_\d+pct$", "", name)


def _measure_title(measure: str, challengers: pd.Series) -> str:
    pcts = set(challengers.str.extract(r"_(\d+)pct$", expand=False).dropna())
    if measure == "ΔAUROC" and len(pcts) == 1:
        return f"ΔAUROC at {pcts.pop()} % of the labels"
    return "ΔAULC, over the label fractions" if measure == "ΔAULC" else measure


def plot_seed_deltas(
    deltas: pd.DataFrame,
    tests: pd.DataFrame,
    intervals: list[pd.DataFrame] | None = None,
    *,
    alpha: float = 0.05,
    title: str = "Per-seed Δ, exact seed-paired Wilcoxon",
) -> plt.Figure:
    """Per-seed Δ of each pair (``seed_deltas``) with its Wilcoxon test (``seed_wilcoxon``).

    One panel per measure (ΔAULC, ΔAUROC), one row per model pair (an AUROC run is matched to
    its model), grouped by ``family`` when that column exists: a dot per seed (blue: challenger
    better, red: baseline better), a black tick at the median and, on the right, the seeds
    favouring the challenger and ``p_adj`` (bold below ``alpha``). Grey bar: the 95 % seed ×
    patient bootstrap interval of ``intervals`` (``aulc_contrasts`` / ``auroc_contrasts``
    tables), when given.

    Returns:
        matplotlib Figure (not yet saved or shown).
    """
    deltas = deltas.assign(_group=deltas.get("family", ""),
                           _chal=deltas["challenger"].map(_pair_model),
                           _base=deltas["baseline"].map(_pair_model))
    pairs = deltas.drop_duplicates(["measure", "challenger", "baseline"])
    if pairs.duplicated(["measure", "_group", "_chal", "_base"]).any():
        raise ValueError("Two pairs of one measure share a model pair (other fractions): plot them apart.")
    rows = list(dict.fromkeys(zip(pairs["_group"], pairs["_chal"], pairs["_base"], strict=True)))
    ticks: list[tuple[str, bool]] = []  # y tick labels: (text, family header)
    ypos = {}
    for i, (group, chal, base) in enumerate(rows):
        if group and (i == 0 or rows[i - 1][0] != group):
            ticks.append((group, True))
        ypos[group, chal, base] = len(ticks)
        ticks.append((f"{chal}  vs  {base}", False))

    measures = list(dict.fromkeys(deltas["measure"]))
    test = tests.set_index(["challenger", "baseline", "measure"])
    ci = {}
    for table in intervals or []:
        measure = next((m for m in measures if m in table.columns), None)
        ci.update({(r.challenger, r.baseline, measure): (r.ci_lo, r.ci_hi) for r in table.itertuples()})

    fig, axes = plt.subplots(1, len(measures), figsize=(4.6 * len(measures) + 3.6, 0.36 * len(ticks) + 2),
                             sharey=True, squeeze=False, layout="constrained")
    for ax, measure in zip(axes[0], measures, strict=True):
        sub = deltas[deltas["measure"] == measure]
        for (group, chal, base, run_c, run_b), g in sub.groupby(
                ["_group", "_chal", "_base", "challenger", "baseline"], sort=False):
            y, v = ypos[group, chal, base], g["Δ"].to_numpy(dtype=float)
            if (run_c, run_b, measure) in ci:
                ax.hlines(y, *ci[run_c, run_b, measure], color=_RANGE, linewidth=6, zorder=1)
            offsets = np.linspace(-0.25, 0.25, v.size) if v.size > 1 else np.zeros(1)
            ax.scatter(v, y + offsets, s=30, c=np.select([v > 0, v < 0], _SIGN_COLORS[:2], _SIGN_COLORS[2]),
                       edgecolors="white", linewidths=0.8, zorder=3)
            ax.vlines(np.median(v), y - 0.35, y + 0.35, color=_INK, linewidth=1.8, zorder=4)
            if (run_c, run_b, measure) in test.index:
                t = test.loc[(run_c, run_b, measure)]
                significant = bool(t["p_adj"] < alpha)
                ax.annotate(f"{int(t['n_pos'])}/{int(t['n_seeds'])} · p = {t['p_adj']:.3f}", xy=(1, y),
                            xycoords=("axes fraction", "data"), xytext=(8, 0), textcoords="offset points",
                            va="center", fontsize=8.5, color=_INK if significant else _INK_2,
                            fontweight="bold" if significant else "normal", annotation_clip=False)
        ax.axvline(0, color=_INK_2, linewidth=1, zorder=2)
        ax.grid(axis="x", color=_GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel(f"{measure} (challenger − baseline)")
        ax.set_title(_measure_title(measure, sub["challenger"]), fontsize=10.5, fontweight="bold")

    ax = axes[0][0]
    ax.set_yticks(range(len(ticks)), [text for text, _ in ticks])
    for label, (_, header) in zip(ax.get_yticklabels(), ticks, strict=True):
        label.set_fontweight("bold" if header else "normal")
        label.set_color(_INK if header else _INK_2)
    ax.set_ylim(len(ticks) - 0.5, -0.5)  # first row on top

    dot = {"linestyle": "none", "marker": "o", "markersize": 6, "markeredgecolor": "white"}
    handles = [Line2D([], [], markerfacecolor=_SIGN_COLORS[0], label="seed: challenger better", **dot),
               Line2D([], [], markerfacecolor=_SIGN_COLORS[1], label="seed: baseline better", **dot),
               Line2D([], [], linestyle="none", marker="|", markersize=11, markeredgewidth=1.8, color=_INK,
                      label="median over the seeds")]
    if ci:
        handles.append(Line2D([], [], color=_RANGE, linewidth=6, label="95 % CI, seed × patient bootstrap"))
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False, fontsize=8.5)
    fig.suptitle(f"{title}\nright: seeds favouring the challenger, adjusted p (bold: p < {alpha:g})",
                 fontsize=11)
    return fig


# ---------------------------------------------------------------------------
# Fitted learning curves
# ---------------------------------------------------------------------------


def plot_learning_curve_fits(lc, models: list[str] | None = None) -> plt.Figure:
    """Fitted AUROC learning curves (``auroc_learning_curves``) with their 95 % bands.

    Dotted line and points: observed seed-mean AUROC at each realised fraction. Line and
    band: fitted logit-linear curve and its seed x patient bootstrap band. Dashed line: the
    baseline's AUROC at the reference fraction (observed, or fitted with crossing='fit'); triangles: the fraction where each other curve reaches it
    (only when inside the observed range).

    Args:
        lc:     ``LearningCurves`` from ``auroc_learning_curves``.
        models: Models to draw (default: every model of ``lc.summary``).

    Returns:
        matplotlib Figure (not yet saved or shown).
    """
    a = lc.attrs
    models = list(lc.summary["model"]) if models is None else models
    colors = plt.get_cmap("tab10").colors
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, m in enumerate(models):
        color = colors[i % len(colors)]
        cur = lc.curves[lc.curves["model"] == m]
        obs = lc.observed[lc.observed["model"] == m]
        ax.fill_between(cur["fraction"], cur["lo"], cur["hi"], color=color, alpha=0.12, linewidth=0)
        ax.plot(cur["fraction"], cur["auroc_fit"], color=color, linewidth=2, label=m)
        ax.plot(obs["fraction"], obs["auroc"], color=color, linewidth=1, linestyle=":", marker="o",
                markersize=5, zorder=3)
    if "reference" in a:
        ax.axhline(a["reference_auroc"], color="black", linestyle="--", linewidth=1,
                   label=f"{a['reference']}{', fitted' if a.get('crossing') == 'fit' else ''} ({a['reference_auroc']:.3f})")
        summary = lc.summary.set_index("model")
        for i, m in enumerate(models):
            f = summary["fraction_eq"].get(m, np.nan)
            if m != a.get("baseline") and np.isfinite(f):
                ax.plot([f], [a["reference_auroc"]], marker="v", markersize=9, zorder=4,
                        color=colors[i % len(colors)])
    ax.set_xscale("log", base=2)
    ticks = sorted(lc.observed["fraction"].unique())
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:.0%}" for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("Training fraction (realised, log scale)")
    ax.set_ylabel("AUROC, active disease (Mayo 2-3 vs 0-1)" if a.get("endpoint") == "active" else "AUROC")
    ax.set_title(f"{a.get('task', '')} — fitted learning curves\n"
                 "logit(AUROC) = a + b · log2(fraction); band = 95% seed × patient bootstrap",
                 fontsize=10, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig
