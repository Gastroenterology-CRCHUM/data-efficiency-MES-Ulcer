"""MES grading (4 classes) learning curves: every model of the plan at several label fractions.

Usage
-----
    python -m scripts.mes.run_data_efficiency                  # full plan
    python -m scripts.mes.run_data_efficiency --dry-run
    python -m scripts.mes.run_data_efficiency --model vits16_imagenet --subset-ratios 0.1 1.0
    python -m scripts.mes.run_data_efficiency --parallel-jobs 8 --resume

Settings: CLI flags > plan entry > ``--config`` (default ``configs/base.yaml``) > defaults.
Every run is seeded (``seed_everything``): training subset, loaders and model.

Outputs (``--results-dir``, default ``results/mes/data_efficiency``)
    results_per_seed.csv     one row per run
    results.csv              mean/std across seeds
    experiment_meta.json     plan, seeds, split summary, git SHA, versions
    predictions/<run_name>/  test and validation labels / probabilities, subset_info.json
    figures/                 per-run ROC and confusion matrix, learning curves

The tables are written once every planned run has finished. Until then the finished rows are
in ``results_per_seed.partial.csv`` (failures in ``failures.json``) and ``--resume`` runs only
the missing ones. A run that narrows the plan (``--model``, ``--max-runs``, another manifest,
...) writes to ``results/mes/_scratch/<timestamp>/`` unless ``--results-dir`` is given.
``--overwrite`` deletes the results and models directories first.
"""

from __future__ import annotations

import os

# Headless runner: select the Agg backend before matplotlib / pyplot is first imported
# (same effect as matplotlib.use("Agg"); the variable is inherited by spawned workers).
os.environ["MPLBACKEND"] = "Agg"

import argparse
import concurrent.futures
import contextlib
import gc
import json
import multiprocessing
import platform
import sys
import traceback
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize

from src.config import (
    MODEL_REGISTRY,
    Config,
    apply_overrides,
    get_img_size,
    get_norm_stats,
    legacy_dict_to_config,
    load_config,
    validate_config,
)
from src.data.dataloader import get_split_loaders, get_test_loader
from src.data.manifest_utils import load_manifest, split_summary
from src.evaluation.plots import (
    plot_confusion_matrix_multiclass,
    plot_learning_curves,
    plot_roc_curves,
)
from src.models.classifier import ClassifierModel, _TrainingInterrupted
from src.utils import (
    get_device,
    get_git_sha,
    ignore_sigint,
    seed_everything,
    setup_logging,
    sigterm_as_interrupt,
    terminate_pool,
)
from src.utils.results import (
    aggregate_seed_results,
    freeze_label,
    mes_extra_metrics,
    pct_label,
    portable_path,
    redact_local_paths,
    remove_run_outputs,
)

TASK = "mes"
TASK_NAME = "mes_multiclass"
LABEL_COL = "label"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLAN = REPO_ROOT / "configs" / "experiments" / "data_efficiency.yaml"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "base.yaml"
DEFAULT_SUBSETS = [0.10, 0.25, 0.50, 0.75, 1.00]
DEFAULT_SEEDS = [42]
DEFAULT_HEAD_TYPES = ["linear"]
DEFAULT_HEAD_LR_SCALES = {
    "linear": 1.0,
    "mlp1": 1.0,
    "mlp2": 1.0,
}
PLAN_KEYS = ("models", "subset_ratios", "seeds", "head_types", "head_lr_scales")
# Fields the runner loop sets for every job (plan head_types / seeds / subset_ratios, and
# the class count of the manifest). A plan run entry that sets them is an error, instead
# of being silently overwritten; so is a run entry with ``paths`` (the runner uses the
# base config's paths).
LOOP_KEYS = {"model": ("head_type", "num_classes"), "training": ("random_seed", "subset_ratio")}
PARTIAL_CSV = "results_per_seed.partial.csv"
PARTIAL_META = "experiment_meta.partial.json"
FINAL_META = "experiment_meta.json"

logger = setup_logging("scripts.mes.run_data_efficiency")


# ---------------------------------------------------------------------------
# Environment and paths
# ---------------------------------------------------------------------------


def _utf8_stdio() -> None:
    """Write stdout/stderr as UTF-8 so redirected output never fails on cp1252 consoles."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        with contextlib.suppress(ValueError, OSError):
            reconfigure(encoding="utf-8", errors="replace")


def _repo_path(path: Path | str) -> Path:
    """Resolve a configured default path against the repository root."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _cli_path(value: str | None, default: Path | str) -> Path:
    """CLI path (relative to the working directory) or the repo-rooted default."""
    return Path(value).resolve() if value else _repo_path(default)


def _sanitize_paths(obj):
    """Portable copy of a JSON-able structure for committed files.

    Strings that are absolute paths become ``portable_path`` values; local path
    prefixes inside other strings (exception messages, tracebacks) are redacted.
    """
    if isinstance(obj, dict):
        return {k: _sanitize_paths(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_paths(v) for v in obj]
    if isinstance(obj, str):
        if len(obj) > 2 and "\n" not in obj and Path(obj).is_absolute():
            return portable_path(obj)
        return redact_local_paths(obj)
    return obj


def _versions() -> dict:
    out = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
    }
    for name in ("torchvision", "timm", "numpy", "pandas", "sklearn"):
        try:
            out[name] = __import__(name).__version__
        except Exception:  # noqa: BLE001
            out[name] = None
    return out


# ---------------------------------------------------------------------------
# Plan loading
# ---------------------------------------------------------------------------


def _get_plan(args: argparse.Namespace, cfg: Config) -> dict:
    """Load the experiment plan and apply the CLI filters.

    A ``--plan`` that does not exist raises FileNotFoundError. Only when
    ``--plan`` is omitted and the default plan is missing does the runner fall
    back to one run of the config model (flagged as a partial run).

    Returns:
        dict with ``path``, ``run_specs``, ``subset_ratios``, ``seeds``,
        ``head_types``, ``head_lr_scales`` (after the CLI filters), ``full``
        (the same lists before filtering) and ``partial_reasons`` (why the
        run does not reproduce the whole plan; empty for a full run).
    """
    reasons: list[str] = []
    if args.plan:
        plan_path: Path | None = Path(args.plan).resolve()
        if not plan_path.is_file():
            raise FileNotFoundError(f"Plan file not found: {args.plan}")
        if plan_path != DEFAULT_PLAN.resolve():
            reasons.append(f"non-default --plan {portable_path(plan_path)}")
    else:
        plan_path = DEFAULT_PLAN

    if plan_path.is_file():
        with open(plan_path, encoding="utf-8") as fh:
            plan = yaml.safe_load(fh) or {}
        if not isinstance(plan, dict):
            raise ValueError(f"Plan {plan_path} must be a mapping with keys {PLAN_KEYS}")
        unknown = sorted(set(plan) - set(PLAN_KEYS))
        if unknown:
            raise ValueError(f"Unknown plan key(s) {unknown} in {plan_path}; allowed: {PLAN_KEYS}")
        run_specs = list(plan.get("models") or [])
        subsets = plan.get("subset_ratios", DEFAULT_SUBSETS)
        seeds = plan.get("seeds", DEFAULT_SEEDS)
        head_types = plan.get("head_types", DEFAULT_HEAD_TYPES)
        head_lr_scales = plan.get("head_lr_scales", DEFAULT_HEAD_LR_SCALES)
    else:
        logger.warning(
            f"Default plan {portable_path(plan_path)} not found: falling back to one run of "
            f"'{cfg.model.model}'."
        )
        reasons.append("default plan missing (single fallback run)")
        plan_path = None
        run_specs = [
            {
                "model": cfg.model.model,
                "freeze_layers": cfg.model.freeze_layers,
                "learning_rate": cfg.training.learning_rate,
            }
        ]
        subsets = DEFAULT_SUBSETS
        seeds = DEFAULT_SEEDS
        head_types = DEFAULT_HEAD_TYPES
        head_lr_scales = DEFAULT_HEAD_LR_SCALES

    if not run_specs:
        raise ValueError("The plan has no 'models'.")
    for spec in run_specs:
        if not isinstance(spec, dict) or not isinstance(spec.get("model"), str):
            raise ValueError(f"Every plan run needs a 'model' name, got {spec!r}")
        if spec["model"] not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model '{spec['model']}' in plan.")
        _check_run_entry(spec)
    subsets = [float(x) for x in subsets]
    seeds = [int(x) for x in seeds]
    head_types = [str(h) for h in head_types]
    head_lr_scales = {str(k): float(v) for k, v in (head_lr_scales or {}).items()}
    full = {
        "run_specs": [dict(s) for s in run_specs],
        "subset_ratios": list(subsets),
        "seeds": list(seeds),
        "head_types": list(head_types),
        "head_lr_scales": dict(head_lr_scales),
    }

    if args.model:
        run_specs = [spec for spec in run_specs if spec.get("model") == args.model]
        if not run_specs:
            raise ValueError(f"Model '{args.model}' not found in plan.")
        if len(run_specs) < len(full["run_specs"]):
            reasons.append(f"--model {args.model}")
    if args.subset_ratios:
        subsets = [float(x) for x in args.subset_ratios]
        if sorted(set(subsets)) != sorted(set(full["subset_ratios"])):
            reasons.append(f"--subset-ratios {' '.join(f'{r:g}' for r in subsets)}")
    if args.seeds:
        seeds = [int(x) for x in args.seeds]
        if sorted(set(seeds)) != sorted(set(full["seeds"])):
            reasons.append(f"--seeds {' '.join(str(s) for s in seeds)}")
    if args.epochs is not None:
        reasons.append(f"--epochs {args.epochs}")

    for r in subsets:
        if not (0 < r <= 1):
            raise ValueError(f"subset ratio must be in (0, 1], got {r}")
    pcts = [pct_label(r) for r in subsets]
    if min(pcts) < 1:
        raise ValueError(f"subset ratios below 0.005 have no integer-percent label: {subsets}")
    if len(set(pcts)) != len(pcts):
        raise ValueError(f"subset ratios {subsets} map to duplicate percent labels {pcts}")
    if len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise ValueError(f"seeds must be distinct non-negative integers, got {seeds}")

    n_planned = len(run_specs) * len(head_types) * len(subsets) * len(seeds)
    if args.max_runs is not None:
        if args.max_runs < 1:
            raise ValueError(f"--max-runs must be >= 1, got {args.max_runs}")
        if args.max_runs < n_planned:
            reasons.append(f"--max-runs {args.max_runs}")

    return {
        "path": plan_path,
        "run_specs": run_specs,
        "subset_ratios": subsets,
        "seeds": seeds,
        "head_types": head_types,
        "head_lr_scales": head_lr_scales,
        "full": full,
        "partial_reasons": reasons,
    }


def _check_run_entry(spec: dict) -> None:
    """Reject run-entry keys that the runner loop would silently overwrite (``LOOP_KEYS``)."""
    bad = [key for keys in LOOP_KEYS.values() for key in keys if key in spec]
    for section, keys in LOOP_KEYS.items():
        sub = spec.get(section)
        if isinstance(sub, dict):
            bad += [f"{section}.{key}" for key in keys if key in sub]
    if "paths" in spec:
        bad.append("paths")
    if bad:
        raise ValueError(
            f"Plan run entry for '{spec['model']}' sets {bad}, which the runner sets for "
            "every job: use the plan's head_types / seeds / subset_ratios (num_classes "
            "comes from the manifest, paths from --config)."
        )


def _dataset_reasons(manifest_path: Path, data_dir: Path) -> list[str]:
    """Partial-run reasons for inputs other than the default dataset (``Config()`` paths).

    The committed results belong to the default manifest and image directory, so
    another dataset (``--manifest`` / ``--data-dir``, or a ``--config`` with other
    paths) never writes into them unless ``--results-dir`` says so.
    """
    paths = Config().paths
    reasons = []
    default_manifest = _repo_path(paths.mes_splits_dir / "dataset_manifest.csv")
    if manifest_path.resolve() != default_manifest.resolve():
        reasons.append(f"manifest {portable_path(manifest_path)} is not the default")
    if data_dir.resolve() != _repo_path(paths.mes_processed_dir).resolve():
        reasons.append(f"data dir {portable_path(data_dir)} is not the default")
    return reasons


# ---------------------------------------------------------------------------
# Per-run configuration
# ---------------------------------------------------------------------------


def _run_config(
    cfg: Config,
    spec: dict,
    *,
    head_type: str,
    seed: int,
    subset_ratio: float,
    num_classes: int,
    args: argparse.Namespace,
) -> Config:
    """Config of one run: ``cfg`` < plan run entry < runner loop values < CLI overrides."""
    run_cfg = legacy_dict_to_config(spec, base=cfg, strict=True)
    training: dict = {"random_seed": seed, "subset_ratio": subset_ratio}
    for key, value in (
        ("epochs", args.epochs),
        ("batch_size", args.batch_size),
        ("deterministic", args.deterministic),
    ):
        if value is not None:
            training[key] = value
    run_cfg = apply_overrides(
        run_cfg,
        {"model": {"head_type": head_type, "num_classes": num_classes}, "training": training},
        strict=True,
    )
    validate_config(run_cfg, img_size=get_img_size(run_cfg.model.model))
    return run_cfg


def _run_name(run_cfg: Config, *, with_lr: bool = False) -> str:
    m, t = run_cfg.model, run_cfg.training
    lr = f"_lr{t.learning_rate:g}" if with_lr else ""
    return (
        f"{m.model}_{freeze_label(m.freeze_layers)}_{m.head_type}{lr}"
        f"_{pct_label(t.subset_ratio)}pct_seed{t.random_seed}"
    )


def _build_jobs(
    cfg: Config,
    plan: dict,
    args: argparse.Namespace,
    *,
    num_classes: int,
    parallel_jobs: int,
    context: dict,
) -> list[dict]:
    """One picklable job per run, in plan order (fraction > seed > run entry > head type)."""
    jobs: list[dict] = []
    for subset_ratio in plan["subset_ratios"]:
        for seed in plan["seeds"]:
            for spec in plan["run_specs"]:
                for head_type in plan["head_types"]:
                    run_cfg = _run_config(
                        cfg,
                        spec,
                        head_type=head_type,
                        seed=seed,
                        subset_ratio=subset_ratio,
                        num_classes=num_classes,
                        args=args,
                    )
                    n_loader = min(run_cfg.training.num_workers, os.cpu_count() or 8)
                    jobs.append(
                        {
                            **context,
                            "run_cfg": run_cfg,
                            "run_name": _run_name(run_cfg),
                            "head_lr_scale": float(plan["head_lr_scales"].get(head_type, 1.0)),
                            "loader_workers": max(1, n_loader // parallel_jobs)
                            if parallel_jobs > 1
                            else n_loader,
                        }
                    )
    # Plan entries that differ only by learning rate get the LR in their run name.
    counts = pd.Series([j["run_name"] for j in jobs]).value_counts()
    for job in jobs:
        if counts[job["run_name"]] > 1:
            job["run_name"] = _run_name(job["run_cfg"], with_lr=True)
    names = [j["run_name"] for j in jobs]
    if len(set(names)) != len(names):
        dup = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(f"Duplicate plan entries (same model/freeze/head/lr): {dup[:5]}")
    if args.max_runs is not None:
        jobs = jobs[: args.max_runs]
    return jobs


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------


def _build_model(
    run_cfg: Config, *, head_lr_scale: float, models_root: Path, pretrained_dir: Path
) -> ClassifierModel:
    m, t, e = run_cfg.model, run_cfg.training, run_cfg.evaluation
    entry = MODEL_REGISTRY[m.model]
    gastronet_path = pretrained_dir / entry.gastronet if entry.gastronet is not None else None
    return ClassifierModel(
        base_model=m.model,
        num_classes=m.num_classes,
        class_weights=t.class_weights,
        optimizer=t.optimizer,
        learning_rate=t.learning_rate,
        threshold=m.threshold,
        dropout_rate=m.dropout_rate,
        drop_path_rate=m.drop_path_rate,
        num_epochs=t.epochs,
        freeze_layers=m.freeze_layers,
        gastronet_path=gastronet_path,
        es_patience=t.es_patience,
        es_start_epoch=t.es_start_epoch,
        grad_clip_norm=t.grad_clip_norm,
        weight_decay=t.weight_decay,
        label_smoothing=t.label_smoothing,
        active_loss_weight=t.active_loss_weight,
        head_type=m.head_type,
        label_col=LABEL_COL,
        random_seed=t.random_seed,
        warmup_epochs=t.warmup_epochs,
        min_lr=t.min_lr,
        n_bootstrap=e.bootstrap_samples,
        ci=e.confidence_interval,
        head_lr_scale=head_lr_scale,
        layer_decay=t.layer_decay,
        use_amp=t.use_amp,
        models_dir=models_root,
    )


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------


def _loader_kwargs(run_cfg: Config, job: dict) -> dict:
    """Keyword arguments shared by get_split_loaders and get_test_loader for one run."""
    t = run_cfg.training
    mean, std = get_norm_stats(run_cfg.model.model)
    return {
        "manifest_path": Path(job["manifest_path"]),
        "data_dir": Path(job["data_dir"]),
        "batch_size": t.batch_size,
        "img_size": get_img_size(run_cfg.model.model),
        "label_col": LABEL_COL,
        "num_workers": job["loader_workers"],
        "equalize": t.equalize,
        "mean": mean,
        "std": std,
        "seed": t.random_seed,
        "use_randaugment": t.use_randaugment,
        "randaugment_m": t.randaugment_m,
        "use_random_erasing": t.use_random_erasing,
        "random_erasing_p": t.random_erasing_p,
    }


def _nan_if_missing(d: dict | None, key: str) -> float:
    if not d or d.get(key) is None:
        return float("nan")
    return float(d[key])


def _execute_run(job: dict) -> dict:
    """Seed, build loaders and model, train, evaluate and save one run; return its row.

    Every loader belongs to its run: its persistent worker processes (~1 GB of commit
    each on Windows) end with the run, the training ones as soon as fit() returns and
    the validation ones after the validation predictions.
    """
    run_cfg: Config = job["run_cfg"]
    m, t = run_cfg.model, run_cfg.training
    run_name = job["run_name"]
    seed_everything(t.random_seed, deterministic=t.deterministic)

    lkw = _loader_kwargs(run_cfg, job)
    train_loader, val_loader = get_split_loaders(
        subset_ratio=t.subset_ratio, random_seed=t.random_seed, **lkw
    )
    test_loader = get_test_loader(**lkw)
    subset = dict(train_loader.subset_info)
    device = get_device(t.device_id)
    models_root = Path(job["models_root"])
    results_dir = Path(job["results_dir"])
    num_classes = m.num_classes

    print("\n" + "-" * 72)
    print(
        f"{run_name} | n_train={subset['n_train']} ({subset['realized_fraction']:.1%} of "
        f"train frames, {subset['n_clips']} clips, {subset['n_patients']} patients)"
    )
    print(
        f"  lr={t.learning_rate:g} head_lr_scale={job['head_lr_scale']:g} "
        f"batch={t.batch_size} img={lkw['img_size']} sampler=frame_budget "
        f"deterministic={t.deterministic} device={device}"
    )
    print("-" * 72)

    model = _build_model(
        run_cfg,
        head_lr_scale=job["head_lr_scale"],
        models_root=models_root,
        pretrained_dir=Path(job["pretrained_dir"]),
    )
    run_metadata = {
        "task": TASK_NAME,
        "run_name": run_name,
        "seed": t.random_seed,
        "subset_ratio": t.subset_ratio,
        "pct_data": pct_label(t.subset_ratio),
        "deterministic": t.deterministic,
        "equalize": t.equalize,
        "img_size": lkw["img_size"],
        "batch_size": t.batch_size,
        "loader_workers": lkw["num_workers"],
        "head_lr_scale": job["head_lr_scale"],
        "n_train": subset["n_train"],
        "n_clips": subset["n_clips"],
        "n_patients": subset["n_patients"],
        "realized_fraction": subset["realized_fraction"],
        "manifest": job["manifest_rel"],
        "plan": job["plan_rel"],
        "git_sha": job["git_sha"],
    }
    fit_interrupted = False
    try:
        model.fit(
            train_loader,
            val_loader,
            t.epochs,
            device,
            checkpoint_root=models_root,
            es_patience=t.es_patience,
            run_name=run_name,
            run_metadata=run_metadata,
        )
    except _TrainingInterrupted:
        fit_interrupted = True
        print("[!] Training interrupted - evaluating the best checkpoint.")
    checkpoint_dir = model.checkpoint_dir
    del train_loader  # its workers exit now (not used after fit)

    # ── Validation predictions of the restored best model ─────────────────
    val_labels, val_probs = model.get_labels_and_probs(val_loader, device)
    del val_loader  # its workers exit before the test loader starts its own

    test_result = model.test_evaluation(
        test_loader,
        device,
        threshold=m.threshold,
        aggregate_by_clip=True,
    )
    y_true = np.asarray(test_result["labels"])
    y_prob = np.asarray(test_result["probabilities"])
    class_names = [f"Mayo {i}" for i in range(num_classes)]

    # ── Save figures ──────────────────────────────────────────────────────
    figs_dir = results_dir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    cm_fig = plot_confusion_matrix_multiclass(
        test_result["frame_level"]["confusion_matrix"],
        class_names=class_names,
        title=f"Confusion matrix - Mayo classification ({run_name})",
    )
    cm_fig.savefig(figs_dir / f"cm_{run_name}.png", dpi=150, bbox_inches="tight")
    plt.close(cm_fig)

    y_bin = np.asarray(label_binarize(y_true, classes=list(range(num_classes))))
    roc_data = []
    for i, name in enumerate(class_names):
        if y_bin[:, i].sum() in (0, len(y_bin)):
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        auc = roc_auc_score(y_bin[:, i], y_prob[:, i])
        roc_data.append({"name": name, "fpr": fpr, "tpr": tpr, "auc": auc})
    if roc_data:
        roc_fig = plot_roc_curves(roc_data, title=f"ROC (OVR) - {run_name}")
        roc_fig.savefig(figs_dir / f"roc_{run_name}.png", dpi=150, bbox_inches="tight")
        plt.close(roc_fig)

    # ── Save predictions ──────────────────────────────────────────────────
    preds_dir = results_dir / "predictions" / run_name
    preds_dir.mkdir(parents=True, exist_ok=True)
    np.save(preds_dir / "test_labels.npy", y_true)
    np.save(preds_dir / "test_probs.npy", y_prob)
    np.save(preds_dir / "val_labels.npy", np.asarray(val_labels))
    np.save(preds_dir / "val_probs.npy", np.asarray(val_probs))
    (preds_dir / "subset_info.json").write_text(json.dumps(subset, indent=2), encoding="utf-8")

    # ── Metrics (full-sample point estimates) ─────────────────────────────
    frame = test_result["frame_level"]
    clip = test_result.get("clip_level")  # None → clip metrics are NaN, never frame copies
    f1 = float(frame["f1"])
    auroc = float(frame["roc_auc"])
    clip_f1 = _nan_if_missing(clip, "f1")
    clip_auroc = _nan_if_missing(clip, "roc_auc")
    extra = mes_extra_metrics(y_true, y_prob, num_classes)
    val_auroc = float(model.best_val_auroc)

    print(
        f"  -> F1={f1:.4f} AUROC={auroc:.4f}  clip_F1={clip_f1:.4f} clip_AUROC={clip_auroc:.4f}"
        f"  val_AUROC={val_auroc:.4f}"
    )
    print(
        f"  micro_F1={extra['micro_f1']:.3f}  macro_F1={extra['macro_f1']:.3f}  QWK={extra['qwk']:.3f}"
        f"  clin_sens={extra['clin_sensitivity']:.3f}  clin_spec={extra['clin_specificity']:.3f}"
    )

    per_class = {k: v for k, v in extra.items() if k.startswith(("sens_mayo", "spec_mayo"))}
    result = {
        "model": m.model,
        "freeze": freeze_label(m.freeze_layers),
        "head_type": m.head_type,
        "subset_ratio": t.subset_ratio,
        "pct_data": pct_label(t.subset_ratio),
        "seed": t.random_seed,
        "n_train": subset["n_train"],
        "f1": f1,
        "auroc": auroc,
        "clip_f1": clip_f1,
        "clip_auroc": clip_auroc,
        "micro_f1": extra["micro_f1"],
        "macro_f1": extra["macro_f1"],
        "clin_sensitivity": extra["clin_sensitivity"],
        "clin_specificity": extra["clin_specificity"],
        "clin_f1": extra["clin_f1"],
        "checkpoint_dir": portable_path(checkpoint_dir, anchors={"<models-dir>": models_root}),
        **dict(sorted(per_class.items(), key=lambda kv: (kv[0][:4], int(kv[0][9:])))),
        "clin_auroc": extra["clin_auroc"],
        "qwk": extra["qwk"],
        "val_auroc": val_auroc,
        "freeze_layers": m.freeze_layers,
        "learning_rate": t.learning_rate,
        "active_loss_weight": t.active_loss_weight,
        "head_lr_scale": job["head_lr_scale"],
        "epochs": t.epochs,
        "patience": t.es_patience,
        "es_start_epoch": t.es_start_epoch,
        "batch_size": t.batch_size,
        "loader_workers": lkw["num_workers"],
        "img_size": lkw["img_size"],
        "realized_fraction": subset["realized_fraction"],
        "n_clips": subset["n_clips"],
        "n_patients": subset["n_patients"],
        "best_epoch": model.best_epoch,
        "stopped_epoch": model.stopped_epoch,
        "stop_reason": model.stop_reason,
        "deterministic": t.deterministic,
        "git_sha": job["git_sha"],
        "run_name": run_name,
    }
    if fit_interrupted:
        raise _TrainingInterrupted(checkpoint_dir, result=result)
    return result


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def _failure(job: dict, exc: BaseException, tb: str | None = None) -> dict:
    """failures.json record; local path prefixes in the message and traceback are redacted."""
    t = job["run_cfg"].training
    return {
        "run_name": job["run_name"],
        "model": job["run_cfg"].model.model,
        "pct_data": pct_label(t.subset_ratio),
        "seed": t.random_seed,
        "error": redact_local_paths(f"{type(exc).__name__}: {exc}"),
        "traceback": redact_local_paths(tb if tb is not None else traceback.format_exc()),
    }


def _parallel_worker(job: dict) -> dict:
    """Execute one run in a spawned process; never raises (failures are returned)."""
    _utf8_stdio()
    try:
        return {"status": "ok", "result": _execute_run(job)}
    except _TrainingInterrupted as exc:
        return {"status": "interrupted", "result": exc.result}
    except Exception as exc:  # noqa: BLE001
        return {"status": "failed", "failure": _failure(job, exc)}


def _plan_order(jobs: list[dict]) -> dict[str, int]:
    """Position of every run name in the plan."""
    return {job["run_name"]: i for i, job in enumerate(jobs)}


def _rows_frame(rows: list[dict], order: dict[str, int]) -> pd.DataFrame:
    """Per-seed rows in plan order, whatever order they finished in (parallel mode)."""
    df = pd.DataFrame(rows)
    if df.empty or "run_name" not in df.columns:
        return df
    rank = df["run_name"].map(order).fillna(len(order)).to_numpy()
    return df.iloc[np.argsort(rank, kind="stable")].reset_index(drop=True)


def _write_partial(rows: list[dict], results_dir: Path, order: dict[str, int]) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    _rows_frame(rows, order).to_csv(results_dir / PARTIAL_CSV, index=False)


def _job_settings(job: dict) -> dict:
    """Settings a resumed row must share with its job (per-seed column names)."""
    m, t = job["run_cfg"].model, job["run_cfg"].training
    return {
        "model": m.model,
        "freeze_layers": m.freeze_layers,
        "head_type": m.head_type,
        "learning_rate": t.learning_rate,
        "active_loss_weight": t.active_loss_weight,
        "subset_ratio": t.subset_ratio,
        "seed": t.random_seed,
        "epochs": t.epochs,
        "patience": t.es_patience,
        "es_start_epoch": t.es_start_epoch,
        "batch_size": t.batch_size,
        "img_size": get_img_size(m.model),
        "head_lr_scale": job["head_lr_scale"],
        "deterministic": t.deterministic,
    }


def _same_value(a, b) -> bool:
    if isinstance(a, (bool, np.bool_)) or isinstance(b, (bool, np.bool_)):
        return str(a).strip().lower() == str(b).strip().lower()
    try:
        return bool(np.isclose(float(a), float(b), rtol=1e-9, atol=0.0))
    except (TypeError, ValueError):
        return str(a) == str(b)


def _load_resume(results_dir: Path, jobs: list[dict], git_sha: str | None) -> list[dict]:
    """Finished rows of an earlier invocation to keep (``--resume``).

    Reads ``results_per_seed.partial.csv`` of ``results_dir``. Rows of runs that
    were interrupted (``stop_reason == "interrupted"``) or that are not in the
    current plan are dropped, so those runs are rerun or ignored.

    Raises:
        ValueError: A row has no ``run_name``, or was run with other settings than
            its job (another plan, --epochs, --batch-size, ...).
    """
    path = results_dir / PARTIAL_CSV
    if not path.is_file():
        print(f"[!] --resume: {portable_path(path)} not found; every planned run will run.")
        return []
    df = pd.read_csv(path)
    if "run_name" not in df.columns:
        raise ValueError(f"{portable_path(path)} has no run_name column; it cannot be resumed.")
    df = df.drop_duplicates(subset="run_name", keep="last")
    by_name = {job["run_name"]: job for job in jobs}
    rows: list[dict] = []
    n_dropped = 0
    for rec in df.to_dict(orient="records"):
        job = by_name.get(rec["run_name"])
        if job is None or rec.get("stop_reason") == "interrupted":
            n_dropped += 1
            continue
        diff = [k for k, v in _job_settings(job).items() if k in rec and not _same_value(rec[k], v)]
        if diff:
            raise ValueError(
                f"--resume: {rec['run_name']} in {portable_path(path)} was run with other "
                f"settings ({', '.join(diff)}). Remove the file or run without --resume."
            )
        rows.append(rec)
    if n_dropped:
        print(f"[!] --resume: {n_dropped} interrupted or unplanned row(s) dropped (rerun).")
    other_sha = sorted({str(r.get("git_sha")) for r in rows} - {str(git_sha)})
    if other_sha:
        logger.warning(f"--resume: kept rows come from other commits {other_sha[:3]}.")
    return rows


def _release_cuda_caches() -> None:
    """Return the cached CUDA device memory and pinned host memory to the driver."""
    if not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()
    empty_host_cache = getattr(torch.cuda, "empty_host_cache", None) or getattr(
        torch._C, "_host_emptyCache", None
    )
    if empty_host_cache is not None:
        empty_host_cache()


def _run_isolated(job: dict) -> dict:
    """``_parallel_worker(job)`` in a fresh spawned process that exits after the run.

    Nothing a run allocates (CUDA context and caches, pinned host memory, DataLoader
    workers) outlives it, so a long plan cannot accumulate memory in this process.
    The child ignores Ctrl+C: this process kills it and re-raises KeyboardInterrupt.
    """
    ctx = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=1, mp_context=ctx, initializer=ignore_sigint
    ) as ex:
        fut = ex.submit(_parallel_worker, job)
        try:
            while True:
                try:
                    return fut.result(timeout=1.0)  # short waits: Ctrl+C is seen at once
                except concurrent.futures.TimeoutError:
                    continue
        except KeyboardInterrupt:
            terminate_pool(ex)
            raise
        except Exception as exc:  # noqa: BLE001  (e.g. BrokenProcessPool)
            return {"status": "failed", "failure": _failure(job, exc, tb="")}


def _run_sequential(
    jobs: list[dict],
    results_dir: Path,
    rows: list[dict],
    failures: list[dict],
    order: dict[str, int],
    *,
    isolate: bool = False,
) -> bool:
    """Run jobs one by one. Returns True when interrupted.

    ``isolate`` runs every job in its own spawned process (``_run_isolated``), which
    Ctrl+C kills at once; in-process, fit() evaluates its best checkpoint first.
    """
    try:
        for i, job in enumerate(jobs, 1):
            print(f"\n[{i}/{len(jobs)}] {job['run_name']}", flush=True)
            if isolate:
                out = _run_isolated(job)
                if out["status"] == "failed":
                    failures.append(out["failure"])
                    print(f"[!] Run failed - {job['run_name']}: {out['failure']['error']}")
                    print(out["failure"]["traceback"], file=sys.stderr)
                elif out["result"] is not None:
                    rows.append(out["result"])
                    _write_partial(rows, results_dir, order)
                if out["status"] == "interrupted":
                    print("\n[!] Run interrupted - proceeding to aggregation.")
                    return True
                continue
            try:
                rows.append(_execute_run(job))
                _write_partial(rows, results_dir, order)
            except _TrainingInterrupted as exc:
                if exc.result is not None:
                    rows.append(exc.result)
                    _write_partial(rows, results_dir, order)
                print("\n[!] Run interrupted - proceeding to aggregation.")
                return True
            except Exception as exc:  # noqa: BLE001
                failures.append(_failure(job, exc))
                print(f"[!] Run failed - {job['run_name']}: {type(exc).__name__}: {exc}")
                traceback.print_exc()
            finally:
                gc.collect()
                _release_cuda_caches()
    except KeyboardInterrupt:
        print("\n[!] Interrupted - proceeding to aggregation.")
        return True
    return False


def _run_parallel(
    jobs: list[dict],
    n_procs: int,
    results_dir: Path,
    rows: list[dict],
    failures: list[dict],
    order: dict[str, int],
) -> bool:
    """Run jobs in spawned processes. Returns True when interrupted (Ctrl+C kills the running jobs)."""
    print(f"\n  Parallel mode: {len(jobs)} runs x {n_procs} workers")
    mp_ctx = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=n_procs, mp_context=mp_ctx, initializer=ignore_sigint
    ) as ex:
        fut_to_job = {ex.submit(_parallel_worker, job): job for job in jobs}
        pending = set(fut_to_job)
        try:
            while pending:
                done, pending = concurrent.futures.wait(
                    pending, timeout=1.0, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for fut in done:
                    job = fut_to_job[fut]
                    try:
                        out = fut.result()
                    except Exception as exc:  # noqa: BLE001  (e.g. BrokenProcessPool)
                        out = {"status": "failed", "failure": _failure(job, exc, tb="")}
                    if out["status"] == "failed":
                        failures.append(out["failure"])
                        print(f"  [!] FAILED {job['run_name']}: {out['failure']['error']}")
                        continue
                    result = out.get("result")
                    if result is None:
                        continue
                    rows.append(result)
                    _write_partial(rows, results_dir, order)
                    print(
                        f"  OK {result['model']} {result['pct_data']}%"
                        f" seed={result['seed']}  F1={result['f1']:.4f}"
                        f"  AUROC={result['auroc']:.4f}"
                    )
        except KeyboardInterrupt:
            print("\n[!] Interrupted - stopping the running jobs.")
            terminate_pool(ex)
            return True
    return False


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def _execution_mode(n_procs: int, isolate: bool) -> str:
    if n_procs > 1:
        return f"parallel, {n_procs} worker processes"
    return "sequential, one spawned process per run" if isolate else "sequential, in-process"


def _write_meta(meta: dict, results_dir: Path, name: str = FINAL_META) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / name).write_text(
        json.dumps(_sanitize_paths(meta), indent=2, default=str), encoding="utf-8"
    )


def main(args: argparse.Namespace) -> int:
    _utf8_stdio()
    config_path = _cli_path(args.config, DEFAULT_CONFIG)
    cfg = load_config(config_path)

    manifest_path = _cli_path(args.manifest, cfg.paths.mes_splits_dir / "dataset_manifest.csv")
    data_dir = _cli_path(args.data_dir, cfg.paths.mes_processed_dir)
    if not manifest_path.exists():
        raise FileNotFoundError(f"MES manifest not found: {manifest_path}")

    manifest_df = load_manifest(manifest_path)
    if LABEL_COL not in manifest_df.columns:
        raise ValueError(f"Manifest must contain '{LABEL_COL}' column: {manifest_path}")
    labels = sorted(int(v) for v in manifest_df[LABEL_COL].unique())
    num_classes = len(labels)
    if num_classes < 2:
        raise ValueError(f"Expected multiclass labels in MES manifest, got {num_classes} class.")
    if labels != list(range(num_classes)):
        raise ValueError(f"MES labels must be 0..{num_classes - 1}, got {labels}")

    plan = _get_plan(args, cfg)
    reasons = plan["partial_reasons"]
    reasons += _dataset_reasons(manifest_path, data_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scratch = _repo_path(cfg.paths.results_mes_dir) / "_scratch" / stamp
    if args.results_dir:
        results_dir = Path(args.results_dir).resolve()
    elif reasons:
        results_dir = scratch
    else:
        results_dir = _repo_path(cfg.paths.results_mes_data_efficiency_dir)
    if args.models_dir:
        models_root = Path(args.models_dir).resolve()
    elif reasons:
        models_root = _repo_path(cfg.paths.output_mes_dir) / "_scratch" / stamp / "models"
    else:
        models_root = _repo_path(cfg.paths.mes_models_dir)

    git_sha = get_git_sha()
    context = {
        "manifest_path": str(manifest_path),
        "data_dir": str(data_dir),
        "models_root": str(models_root),
        "results_dir": str(results_dir),
        "pretrained_dir": str(_repo_path(cfg.paths.pretrained_dir)),
        "manifest_rel": portable_path(manifest_path),
        "plan_rel": portable_path(plan["path"]),
        "git_sha": git_sha,
    }
    n_procs = max(1, int(args.parallel_jobs))
    jobs = _build_jobs(
        cfg, plan, args, num_classes=num_classes, parallel_jobs=n_procs, context=context
    )
    deterministic = sorted({j["run_cfg"].training.deterministic for j in jobs})

    print("\n" + "=" * 72)
    print("MES DATA EFFICIENCY")
    print("=" * 72)
    print(f"Manifest       : {portable_path(manifest_path)}")
    print(f"Data dir       : {portable_path(data_dir)}")
    print(f"Plan           : {portable_path(plan['path'])}")
    print(f"Config         : {portable_path(config_path)}")
    print(f"Classes        : {num_classes}")
    print(f"Models         : {[r.get('model') for r in plan['run_specs']]}")
    print(f"Subset ratios  : {plan['subset_ratios']}")
    print(f"Seeds          : {plan['seeds']}")
    print(f"Head types     : {plan['head_types']}")
    print(f"Deterministic  : {', '.join(str(d) for d in deterministic)}")
    print(f"Execution      : {_execution_mode(n_procs, args.isolate_runs)}")
    print(f"Planned runs   : {len(jobs)}")
    partial_csv = results_dir / PARTIAL_CSV
    resumed = _load_resume(results_dir, jobs, git_sha) if args.resume else []
    done = {row["run_name"] for row in resumed}
    todo = [job for job in jobs if job["run_name"] not in done]
    if args.resume:
        print(f"Resumed runs   : {len(resumed)} kept, {len(todo)} to run")
    elif partial_csv.is_file() and not args.overwrite:
        print(
            f"[!] {portable_path(partial_csv)} of an earlier run will be replaced; "
            "pass --resume to continue it instead."
        )
    print(f"Results dir    : {portable_path(results_dir)}")
    print(f"Models dir     : {portable_path(models_root)}")
    protected = [manifest_path, data_dir, _repo_path(cfg.paths.pretrained_dir), config_path, plan["path"]]
    overwritten = []
    if args.overwrite:
        overwritten = remove_run_outputs([results_dir, models_root], protected, dry_run=True)
        listed = ", ".join(portable_path(d) for d in overwritten) or "nothing to delete"
        print(f"Overwrite      : {listed}")
    if reasons:
        where = "" if args.results_dir else " (scratch; pass --results-dir to choose)"
        print(f"Partial run    : {'; '.join(reasons)}{where}")
    for job in jobs[:3]:
        print(f"  e.g. {job['run_name']}")
    if args.dry_run:
        print("[DRY RUN] No training launched.")
        return 0
    if not data_dir.is_dir():
        raise FileNotFoundError(f"MES data dir not found: {data_dir}")
    remove_run_outputs(overwritten, protected)
    device_name = (
        torch.cuda.get_device_name(cfg.training.device_id)
        if torch.cuda.is_available() and cfg.training.device_id >= 0
        else "cpu"
    )
    print(f"Device         : {device_name}")

    meta = {
        "task": TASK_NAME,
        "status": "running",
        "created": datetime.now().isoformat(timespec="seconds"),
        "git_sha": git_sha,
        "manifest": portable_path(manifest_path),
        "data_dir": portable_path(data_dir),
        "results_dir": portable_path(results_dir),
        "models_dir": portable_path(models_root),
        "num_classes": num_classes,
        "plan": {"path": portable_path(plan["path"]), **plan["full"]},
        "config": portable_path(config_path),
        "partial_run_reasons": reasons,
        "models": [r.get("model") for r in plan["run_specs"]],
        "subset_ratios": plan["subset_ratios"],
        "seeds": plan["seeds"],
        "head_types": plan["head_types"],
        "head_lr_scales": plan["head_lr_scales"],
        "deterministic": deterministic[0] if len(deterministic) == 1 else deterministic,
        "n_runs_planned": len(jobs),
        "n_runs_resumed": len(resumed),
        "split_summary": split_summary(manifest_df, label_col=LABEL_COL),
        "base_config": cfg.to_dict(),
        "versions": _versions(),
        "device": device_name,
        "metrics": {
            "f1 / auroc": "frame level; f1 = macro F1 of the argmax, auroc = macro one-vs-rest",
            "clip_*": "clip level (mean probabilities, argmax); NaN without clip ids",
            "clin_*": "active disease (Mayo >= 2) vs remission; clin_auroc score = p2 + p3",
            "qwk": "quadratic weighted kappa of the argmax grade",
            "val_auroc": "best validation AUROC reached by fit() (checkpoint criterion)",
            "best_seed": "results.csv: seed with the highest val_auroc (never test)",
            "estimates": "point estimates on the full test set; CIs are not stored here",
        },
    }
    _write_meta(meta, results_dir, PARTIAL_META)

    rows: list[dict] = list(resumed)
    failures: list[dict] = []
    order = _plan_order(jobs)
    if n_procs > 1:
        interrupted = _run_parallel(todo, n_procs, results_dir, rows, failures, order)
    else:
        interrupted = _run_sequential(
            todo, results_dir, rows, failures, order, isolate=args.isolate_runs
        )
    return _finalize(rows, failures, jobs, interrupted, results_dir=results_dir, meta=meta)


def _finalize(
    rows: list[dict],
    failures: list[dict],
    jobs: list[dict],
    interrupted: bool,
    *,
    results_dir: Path,
    meta: dict,
) -> int:
    """Write the outputs of one invocation and return the exit code.

    ``results_per_seed.csv``, ``results.csv``, ``experiment_meta.json`` and the
    learning curves are replaced only when every planned job has a finished row
    (``rows`` include the ``--resume`` rows). Otherwise existing tables are left
    untouched: the finished rows stay in ``results_per_seed.partial.csv`` and the
    status goes to ``experiment_meta.partial.json``.

    Returns:
        0 when complete; 1 with failed or unrun runs; 130 when interrupted.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    order = _plan_order(jobs)
    results_df = _rows_frame(rows, order)
    if results_df.empty:
        finished_df = results_df
    elif "stop_reason" in results_df.columns:
        finished_df = results_df[results_df["stop_reason"] != "interrupted"]
    else:
        finished_df = results_df
    finished = set(finished_df["run_name"]) if "run_name" in finished_df else set()
    n_interrupted = len(results_df) - len(finished_df)
    missing = [job["run_name"] for job in jobs if job["run_name"] not in finished]
    complete = not missing
    n_not_run = max(0, len(missing) - len(failures) - n_interrupted)
    status = "completed" if complete else ("interrupted" if interrupted else "incomplete")
    meta.update(
        {
            "status": status,
            "finished": datetime.now().isoformat(timespec="seconds"),
            "n_runs": len(finished),
            "n_failed": len(failures),
            "n_interrupted": n_interrupted,
            "n_not_run": n_not_run,
            "failed_runs": [f["run_name"] for f in failures],
        }
    )
    failures_path = results_dir / "failures.json"
    if failures:
        failures_path.write_text(json.dumps(_sanitize_paths(failures), indent=2), encoding="utf-8")
    elif failures_path.exists():
        failures_path.unlink()

    if failures:
        print(f"\n[!] {len(failures)} run(s) failed (details in {portable_path(failures_path)}):")
        for f in failures:
            print(f"    {f['run_name']}: {f['error']}")
    if n_interrupted:
        print(f"[!] {n_interrupted} run(s) were interrupted before the end of training.")
    if n_not_run:
        print(f"[!] {n_not_run} planned run(s) did not run.")

    partial_csv = results_dir / PARTIAL_CSV
    if not complete:
        _write_meta(meta, results_dir, PARTIAL_META)
        if not results_df.empty:
            results_df.to_csv(partial_csv, index=False)
        print(
            f"\n[!] Incomplete run ({len(finished)}/{len(jobs)} runs finished): "
            f"results_per_seed.csv, results.csv and {FINAL_META} in "
            f"{portable_path(results_dir)} were left untouched."
        )
        if not results_df.empty:
            print(
                f"    Finished rows -> {portable_path(partial_csv)}. Rerun the same command "
                "with --resume to run only the missing runs."
            )
        if not finished_df.empty:
            preview = aggregate_seed_results(finished_df, TASK)
            cols = ["model", "freeze", "pct_data", "n_seeds", "f1_mean", "auroc_mean"]
            print("    Preview of the finished runs:")
            print(preview[[c for c in cols if c in preview.columns]].to_string(index=False))
        return 1 if failures or not interrupted else 130

    agg_df = aggregate_seed_results(results_df, TASK)

    print("\n" + "=" * 72)
    print("AGGREGATED RESULTS")
    print("=" * 72)
    show = [
        "model",
        "freeze",
        "pct_data",
        "n_seeds",
        "f1_mean",
        "f1_std",
        "auroc_mean",
        "auroc_std",
        "macro_f1_mean",
        "clin_sens_mean",
        "clin_spec_mean",
        "val_auroc_mean",
        "n_train_mean",
    ]
    print(agg_df[[c for c in show if c in agg_df.columns]].to_string(index=False))

    raw_path = results_dir / "results_per_seed.csv"
    agg_path = results_dir / "results.csv"
    results_df.to_csv(raw_path, index=False)
    agg_df.to_csv(agg_path, index=False)
    _write_meta(meta, results_dir)
    partial_csv.unlink(missing_ok=True)
    (results_dir / PARTIAL_META).unlink(missing_ok=True)
    print(f"\nPer-seed results -> {portable_path(raw_path)}")
    print(f"Aggregated results -> {portable_path(agg_path)}")

    # ── Save learning curve plots ─────────────────────────────────────────
    figs_dir = results_dir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)
    for metric in ("f1", "auroc"):
        for x, suffix, label in (
            ("subset_ratio", "", "nominal fraction"),
            ("realized_fraction_mean", "_realized", "realised fraction of train frames"),
        ):
            if x not in agg_df.columns:
                continue
            fig = plot_learning_curves(
                results_df=agg_df,
                metric=metric,
                title=f"MES data efficiency - {metric.upper()} vs training set size ({label})",
                x=x,
            )
            out_path = figs_dir / f"learning_curve_{metric}{suffix}.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Learning curve ({metric}, {label}) -> {portable_path(out_path)}")

    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MES data-efficiency experiment runner")
    parser.add_argument(
        "--plan",
        default=None,
        help="Experiment YAML plan (default: configs/experiments/data_efficiency.yaml; "
        "a given path that does not exist is an error)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Base YAML/JSON config (default: configs/base.yaml); plan entries and CLI flags override it",
    )
    parser.add_argument("--model", default=None, help="Run a single model from the plan")
    parser.add_argument(
        "--manifest",
        default=None,
        help="MES manifest CSV (default: <paths.mes>/splits/dataset_manifest.csv)",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Root directory of the images the manifest points to (default: data/mes/processed)",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Output directory (default: results/mes/data_efficiency for a full plan run, "
        "results/mes/_scratch/<timestamp> for a partial run)",
    )
    parser.add_argument(
        "--models-dir",
        default=None,
        help="Checkpoint root (default: output/mes/models, or output/mes/_scratch/<timestamp>/"
        "models for a partial run)",
    )
    parser.add_argument(
        "--subset-ratios", type=float, nargs="+", default=None, help="Override subset ratios"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Override random seeds")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Override batch size (every run, any mode)"
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Deterministic cuDNN/torch algorithms (default: on, or training.deterministic "
        "of --config)",
    )
    parser.add_argument(
        "--max-runs", type=int, default=None, help="Stop after N runs (for smoke tests)"
    )
    fresh = parser.add_mutually_exclusive_group()
    fresh.add_argument(
        "--resume",
        action="store_true",
        help="Keep the finished runs of <results-dir>/results_per_seed.partial.csv (left by an "
        "interrupted or failed invocation with the same settings) and run only the others",
    )
    fresh.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete <results-dir> and <models-dir> before training (a fresh run)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print plan and exit")
    parser.add_argument(
        "--parallel-jobs",
        type=int,
        default=1,
        help="Number of parallel training jobs (default: 1 = sequential).",
    )
    parser.add_argument(
        "--isolate-runs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sequential mode: run every job in a fresh spawned process, so that no memory "
        "accumulates over a long plan (default: on; --no-isolate-runs runs them all in "
        "this process). Parallel mode always uses worker processes.",
    )
    return parser


if __name__ == "__main__":
    sigterm_as_interrupt()
    sys.exit(main(build_parser().parse_args()))
