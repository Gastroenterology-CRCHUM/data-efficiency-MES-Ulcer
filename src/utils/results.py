"""Result tables of the data-efficiency experiments: per-seed rows and their aggregate.

Per-seed columns (one row per run)
----------------------------------
Keys:        model, freeze, head_type, learning_rate, subset_ratio, pct_data, seed
Metrics:     f1, auroc (frame level; ulcer f1 at the fixed 0.5 threshold, MES f1 = macro
             F1 of the argmax), clip_f1, clip_auroc (NaN when no clip-level result),
             val_auroc (best validation AUROC reached by fit(), the checkpoint criterion)
MES:         micro_f1, macro_f1, sens_mayo{i}, spec_mayo{i}, clin_sensitivity,
             clin_specificity, clin_f1, clin_auroc (active disease = Mayo >= 2), qwk
Ulcer:       tuned_threshold (val-F1-optimal), tuned_threshold_at_edge, tuned_sensitivity,
             tuned_specificity, tuned_f1, tuned_mcc (test metrics at the tuned threshold)
Subset:      n_train, n_clips, n_patients, realized_fraction (n_train / full train frames)
Provenance:  freeze_layers, head_lr_scale, epochs, patience, es_start_epoch, batch_size,
             loader_workers
             (DataLoader workers; the augmentation RNG streams depend on it), img_size,
             deterministic, best_epoch, stopped_epoch, stop_reason,
             git_sha, checkpoint_dir (repo-relative), run_name (``predictions/<run_name>``;
             ``--resume`` matches on it)

Optional columns may be missing (e.g. task-specific metrics).
"""

from __future__ import annotations

import contextlib
import os
import re
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]

TASKS = ("mes", "ulcer")

GROUP_KEYS = ("model", "freeze", "head_type", "learning_rate", "pct_data", "subset_ratio")
"""Columns identifying one configuration; rows that differ only by ``seed`` are aggregated."""

# (per-seed column, aggregated prefix). Aggregates are "<prefix>_mean" / "<prefix>_std".
_COMMON_METRICS = (
    ("f1", "f1"),
    ("auroc", "auroc"),
    ("clip_f1", "clip_f1"),
    ("clip_auroc", "clip_auroc"),
)
_MES_METRICS = (
    ("micro_f1", "micro_f1"),
    ("macro_f1", "macro_f1"),
    ("clin_sensitivity", "clin_sens"),
    ("clin_specificity", "clin_spec"),
    ("clin_f1", "clin_f1"),
    ("clin_auroc", "clin_auroc"),
    ("qwk", "qwk"),
)
_ULCER_METRICS = (
    ("tuned_threshold", "tuned_threshold"),
    ("tuned_sensitivity", "tuned_sensitivity"),
    ("tuned_specificity", "tuned_specificity"),
    ("tuned_f1", "tuned_f1"),
    ("tuned_mcc", "tuned_mcc"),
)
_RUN_METRICS = (
    ("val_auroc", "val_auroc"),
    ("realized_fraction", "realized_fraction"),
)
_PER_CLASS_RE = re.compile(r"^(sens|spec)_mayo(\d+)$")


# ---------------------------------------------------------------------------
# Labels and paths
# ---------------------------------------------------------------------------


def freeze_label(freeze_layers: int) -> str:
    """Label of a ``freeze_layers`` value: 0 → "finetune", -1 → "frozen", N → "freezeN"."""
    n = int(freeze_layers)
    return {0: "finetune", -1: "frozen"}.get(n, f"freeze{n}")


def pct_label(subset_ratio: float) -> int:
    """Integer percentage of a subset ratio, rounded (0.29 → 29, never truncated to 28)."""
    return int(round(float(subset_ratio) * 100))


def portable_path(
    path: Path | str | None,
    root: Path | str | None = None,
    *,
    anchors: Mapping[str, Path | str] | None = None,
) -> str | None:
    """Path string that is safe to write into committed result files.

    Returns the POSIX path relative to ``root`` (default: the repository root)
    when ``path`` lies inside it. Otherwise returns ``"<label>/<relative>"`` for
    the first ``anchors`` entry ``{label: directory}`` that contains it, and
    finally ``"<external>/<name>"``, so absolute local paths (user names, drive
    layouts) never end up in CSV or JSON outputs. Relative inputs are resolved
    against the current working directory, like any CLI path.
    """
    if path is None:
        return None
    p = Path(path).resolve()
    bases: list[tuple[Path, str | None]] = [(Path(root) if root else REPO_ROOT, None)]
    bases += [(Path(d), label) for label, d in (anchors or {}).items()]
    for base, label in bases:
        try:
            rel = p.relative_to(base.resolve())
        except ValueError:
            continue
        rel_str = rel.as_posix()
        if label is None:
            return rel_str
        return label if rel_str == "." else f"{label}/{rel_str}"
    return f"<external>/{p.name}"


def run_dir_of(per_seed: pd.DataFrame, model: str, pct: int, seed: int) -> Path:
    """Checkpoint directory of the (model, pct, seed) row of a ``results_per_seed.csv``."""
    row = per_seed[(per_seed["model"] == model) & (per_seed["pct_data"] == pct)
                   & (per_seed["seed"] == seed)]
    if len(row) != 1:
        raise LookupError(f"{len(row)} runs of {model} at {pct} %, seed {seed} (expected 1)")
    return REPO_ROOT / str(row["checkpoint_dir"].iloc[0]).replace("\\", "/")


# Home directories of any user, including Windows 8.3 short names (C:/Users/ABCDEF~1).
_USER_HOME_RE = re.compile(
    r"(?<![\w.\-])(?:[A-Za-z]:[\\/]+(?:Users|Documents and Settings)|/home|/Users)"
    r"[\\/]+[^\\/\s'\"<>|:]+",
    re.IGNORECASE,
)
# A prefix only matches whole path components: next comes a separator, a delimiter or the end.
_PATH_END = r"(?![^\\/\s'\"<>|:,;()\[\]{}])"


def _prefix_pattern(prefix: str) -> re.Pattern | None:
    """Regex matching ``prefix`` with either separator (case-insensitive on Windows).

    None for a filesystem root ("/", "C:/"), which would match every absolute path.
    """
    parts = [part for part in re.split(r"[\\/]+", prefix) if part]
    if not parts or (len(parts) == 1 and parts[0].endswith(":")):
        return None
    lead = r"[\\/]+" if prefix[:1] in ("/", "\\") else ""
    body = r"[\\/]+".join(re.escape(part) for part in parts)
    flags = re.IGNORECASE if os.name == "nt" else 0
    return re.compile(rf"(?<![\w.\-]){lead}{body}{_PATH_END}", flags)


def redact_local_paths(text: str | None) -> str | None:
    """``text`` with the machine's absolute path prefixes replaced by placeholders.

    For free text that may end up in committed files, such as the exception
    messages and tracebacks of ``failures.json``. Replaced, longest prefix first:
    the repository root by ".", the Python environment (``sys.prefix``) by
    "<env>", the base interpreter (``sys.base_prefix``) by "<python>" and the home
    directory by "~". Any remaining user home (``C:/Users/<name>``, including
    8.3 short names, ``/home/<name>``, ``/Users/<name>``) also becomes "~". Both
    separators are matched, case-insensitively on Windows. None is returned as is.
    """
    if not text:
        return text
    prefixes: dict[str, str] = {}
    for placeholder, prefix in (
        (".", REPO_ROOT),
        ("<env>", sys.prefix),
        ("<env>", sys.exec_prefix),
        ("<python>", sys.base_prefix),
    ):
        if prefix:
            prefixes.setdefault(str(Path(prefix)), placeholder)
    with contextlib.suppress(RuntimeError, KeyError):  # no resolvable home directory
        prefixes.setdefault(str(Path.home()), "~")
    out = str(text)
    for prefix in sorted(prefixes, key=len, reverse=True):
        pattern = _prefix_pattern(prefix)
        if pattern is not None:
            out = pattern.sub(lambda _m, rep=prefixes[prefix]: rep, out)
    return _USER_HOME_RE.sub("~", out)


def remove_run_outputs(
    dirs: Sequence[Path | str],
    protected: Sequence[Path | str | None] = (),
    *,
    dry_run: bool = False,
) -> list[Path]:
    """Delete the existing folders among ``dirs`` (a runner's ``--overwrite``) and return them.

    Raises ValueError, before deleting anything, for a folder that is the repository, one
    of its parents or one of its top-level folders (``results``, ``output``, ...), or that
    is or contains a ``protected`` path (dataset, weights, config; None entries skipped).
    ``dry_run`` only lists.
    """
    root = REPO_ROOT.resolve()
    keep = [root, *(Path(p).resolve() for p in protected if p is not None)]
    targets = [Path(d).resolve() for d in dirs]
    for d in targets:
        if d.parent == root or any(k == d or d in k.parents for k in keep):
            raise ValueError(f"--overwrite refuses to delete {portable_path(d)}")
    existing = [d for d in dict.fromkeys(targets) if d.is_dir()]
    if not dry_run:
        for d in existing:
            shutil.rmtree(d)
    return existing


# ---------------------------------------------------------------------------
# Per-run metrics (point estimates)
# ---------------------------------------------------------------------------


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else float("nan")


def binary_metrics_at_threshold(
    labels: Sequence[int] | np.ndarray,
    probs: Sequence[float] | np.ndarray,
    threshold: float,
) -> dict:
    """Binary metrics when predicting 1 iff ``prob >= threshold``.

    Returns:
        dict with ``confusion_matrix`` (2x2, labels [0, 1]), ``f1``, ``sensitivity``,
        ``specificity`` and ``mcc`` (NaN when undefined).
    """
    y = np.asarray(labels).astype(int).ravel()
    p = np.asarray(probs, dtype=float)
    if p.ndim == 2:
        p = p[:, 1]
    pred = (p >= float(threshold)).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "confusion_matrix": cm,
        "f1": float(f1_score(y, pred, labels=[0, 1], zero_division=0)),
        "sensitivity": _safe_div(tp, tp + fn),
        "specificity": _safe_div(tn, tn + fp),
        "mcc": _safe_div(float(tp) * tn - float(fp) * fn,
                         float(np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))),
    }


def _quadratic_kappa(cm: np.ndarray) -> float:
    """Quadratic weighted kappa of a confusion matrix (rows = true); NaN when undefined."""
    cm = np.asarray(cm, dtype=float)
    k = np.arange(cm.shape[0])
    w = (k[:, None] - k[None, :]) ** 2
    expected = np.outer(cm.sum(axis=1), cm.sum(axis=0)) / cm.sum()
    return 1.0 - _safe_div((w * cm).sum(), (w * expected).sum())


def mes_extra_metrics(
    labels: Sequence[int] | np.ndarray,
    probs: np.ndarray,
    num_classes: int | None = None,
    *,
    active_cutoff: int = 2,
) -> dict:
    """MES metrics beyond frame F1/AUROC, from (N,) labels and (N, C) probabilities.

    Predictions are the argmax. Keys:
        micro_f1, macro_f1                      over all C classes (labels passed explicitly)
        sens_mayo{i}, spec_mayo{i}              one-vs-rest, NaN when undefined
        clin_sensitivity, clin_specificity,
        clin_f1                                 active disease (Mayo >= ``active_cutoff``)
                                                vs remission, from the argmax class
        clin_auroc                              same endpoint, score = sum of the active
                                                classes' probabilities (NaN if one class)
        qwk                                     quadratic weighted kappa of the argmax
    """
    y = np.asarray(labels).astype(int).ravel()
    prob = np.asarray(probs, dtype=float)
    if prob.ndim != 2:
        raise ValueError(f"MES probabilities must be (N, C), got shape {prob.shape}")
    n_cls = int(num_classes) if num_classes is not None else prob.shape[1]
    classes = list(range(n_cls))
    pred = prob.argmax(axis=1)

    out: dict = {
        "micro_f1": float(f1_score(y, pred, labels=classes, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y, pred, labels=classes, average="macro", zero_division=0)),
    }
    cm = confusion_matrix(y, pred, labels=classes)
    total = cm.sum()
    for i in classes:
        tp = cm[i, i]
        fn = cm[i].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        out[f"sens_mayo{i}"] = _safe_div(tp, tp + fn)
        out[f"spec_mayo{i}"] = _safe_div(tn, tn + fp)

    y_act = (y >= active_cutoff).astype(int)
    clin = binary_metrics_at_threshold(y_act, (pred >= active_cutoff).astype(float), 0.5)
    out["clin_sensitivity"] = clin["sensitivity"]
    out["clin_specificity"] = clin["specificity"]
    out["clin_f1"] = clin["f1"]
    score = prob[:, active_cutoff:].sum(axis=1)
    out["clin_auroc"] = (
        float(roc_auc_score(y_act, score)) if len(np.unique(y_act)) == 2 else float("nan")
    )
    out["qwk"] = _quadratic_kappa(cm)
    return out


# ---------------------------------------------------------------------------
# Aggregation across seeds
# ---------------------------------------------------------------------------


def _normalize_task(task: str) -> str:
    key = str(task).strip().lower()
    for t in TASKS:
        if key == t or key.startswith(t + "_"):
            return t
    raise ValueError(f"task must be one of {TASKS} (or e.g. 'mes_multiclass'), got {task!r}")


def _metric_columns(df: pd.DataFrame, task: str) -> list[tuple[str, str]]:
    specs = list(_COMMON_METRICS)
    specs += list(_MES_METRICS) if task == "mes" else list(_ULCER_METRICS)
    if task == "mes":
        per_class = sorted(
            (c for c in df.columns if _PER_CLASS_RE.match(c)),
            key=lambda c: (_PER_CLASS_RE.match(c).group(1), int(_PER_CLASS_RE.match(c).group(2))),
        )
        specs += [(c, c) for c in per_class]
    specs += list(_RUN_METRICS)
    return [(src, dst) for src, dst in specs if src in df.columns]


def aggregate_seed_results(per_seed_df: pd.DataFrame, task: str) -> pd.DataFrame:
    """Aggregate per-seed rows into one row per configuration (``results.csv``).

    Groups by the ``GROUP_KEYS`` columns that are present. Output columns, when
    their inputs exist:

    - the group keys;
    - ``<m>_mean`` / ``<m>_std`` for every metric: f1, auroc, clip_f1, clip_auroc,
      then MES micro_f1, macro_f1, clin_sens, clin_spec, clin_f1, clin_auroc, qwk,
      sens_mayo{i}, spec_mayo{i}, or ulcer tuned_threshold / tuned_sensitivity /
      tuned_specificity / tuned_f1 / tuned_mcc; then val_auroc and realized_fraction.
    - ``n_train_mean`` / ``n_train_min`` / ``n_train_max``, ``n_clips_mean``,
      ``n_patients_mean``;
    - ``n_seeds``: number of distinct seeds that produced a row (failed runs are
      missing, so it can be lower than the planned number of seeds);
    - ``best_seed``, ``best_checkpoint_dir``, ``best_seed_val_auroc``: the seed with
      the highest **validation** AUROC (ties → lowest seed). Only when a
      ``val_auroc`` column exists; the test set is never used for this choice.

    Args:
        per_seed_df: One row per run (see the module docstring).
        task:        "mes" or "ulcer" (aliases like "mes_multiclass" accepted);
                     selects the task-specific metrics.

    Returns:
        DataFrame sorted by model, freeze, head_type, learning_rate, subset_ratio.
    """
    task = _normalize_task(task)
    df = per_seed_df.copy()
    keys = [k for k in GROUP_KEYS if k in df.columns]
    if not keys:
        raise ValueError(f"per-seed results need at least one of {GROUP_KEYS}")

    metrics = _metric_columns(df, task)
    numeric = [src for src, _ in metrics] + [
        c for c in ("n_train", "n_clips", "n_patients") if c in df.columns
    ]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    grouped = df.groupby(keys, dropna=False, sort=False)
    agg_spec: dict[str, tuple[str, str]] = {}
    for src, dst in metrics:
        agg_spec[f"{dst}_mean"] = (src, "mean")
        agg_spec[f"{dst}_std"] = (src, "std")
    if "n_train" in df.columns:
        agg_spec["n_train_mean"] = ("n_train", "mean")
        agg_spec["n_train_min"] = ("n_train", "min")
        agg_spec["n_train_max"] = ("n_train", "max")
    for col in ("n_clips", "n_patients"):
        if col in df.columns:
            agg_spec[f"{col}_mean"] = (col, "mean")
    if "seed" in df.columns:
        agg_spec["n_seeds"] = ("seed", "nunique")

    if agg_spec:
        agg = grouped.agg(**agg_spec).reset_index()
    else:
        agg = grouped.size().rename("n_rows").reset_index()
    if "seed" not in df.columns:
        agg["n_seeds"] = grouped.size().to_numpy()

    if "val_auroc" in df.columns and "seed" in df.columns:
        cols = keys + ["seed", "val_auroc"] + (["checkpoint_dir"] if "checkpoint_dir" in df else [])
        cand = df.loc[np.isfinite(df["val_auroc"].to_numpy(dtype=float)), cols]
        best = (
            cand.sort_values(["val_auroc", "seed"], ascending=[False, True], kind="mergesort")
            .drop_duplicates(subset=keys, keep="first")
            .rename(
                columns={
                    "seed": "best_seed",
                    "checkpoint_dir": "best_checkpoint_dir",
                    "val_auroc": "best_seed_val_auroc",
                }
            )
        )
        agg = agg.merge(best, on=keys, how="left")
        agg["best_seed"] = agg["best_seed"].astype("Int64")

    sort_cols = [
        c for c in ("model", "freeze", "head_type", "learning_rate", "subset_ratio") if c in agg
    ]
    return agg.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
