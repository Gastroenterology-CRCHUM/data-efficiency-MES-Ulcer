"""
scripts/ulcer/run_data_efficiency.py
=====================================
Data efficiency experiment — ulcer detection learning curves.

Trains each model at multiple training set fractions and saves all results
as CSV files. Produces learning curve plots at the end.

Usage
-----
    python -m scripts.ulcer.run_data_efficiency
    python -m scripts.ulcer.run_data_efficiency --plan configs/experiments/data_efficiency.yaml
    python -m scripts.ulcer.run_data_efficiency --dry-run
    python -m scripts.ulcer.run_data_efficiency --model vits16_imagenet
    python -m scripts.ulcer.run_data_efficiency --manifest data/ulcer/splits/dataset_manifest.csv
    python -m scripts.ulcer.run_data_efficiency --subset-ratios 0.1 0.5 1.0
    python -m scripts.ulcer.run_data_efficiency --epochs 50 --max-runs 4
    python -m scripts.ulcer.run_data_efficiency --num-workers 2 --batch-size 32
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import confusion_matrix as _sklearn_cm
from sklearn.metrics import f1_score as _sklearn_f1

from src.config import MODEL_REGISTRY, Config, get_img_size, legacy_dict_to_config, load_config
from src.data.dataloader import get_split_loaders, get_test_loader, get_val_loader
from src.evaluation.plots import plot_learning_curves, plot_roc_curve
from src.evaluation.threshold import collect_probabilities, find_best_threshold, sweep_thresholds
from src.models.classifier import ClassifierModel, _TrainingInterrupted
from src.utils import get_device, loader_dataset_size

DEFAULT_PLAN = Path("configs/experiments/data_efficiency.yaml")

DEFAULT_SUBSETS = [1.00]
DEFAULT_SEEDS = [42]
DEFAULT_HEAD_TYPES = ["linear"]
DEFAULT_HEAD_LR_SCALES = {
    "linear": 1.0,
    "mlp1": 1.0,
    "mlp2": 1.0,
}


# ---------------------------------------------------------------------------
# Plan loading
# ---------------------------------------------------------------------------


def _get_plan(
    args: argparse.Namespace, cfg: Config
) -> tuple[list[dict], list[float], list[int], list[str], dict[str, float]]:
    plan_path = Path(args.plan) if args.plan else DEFAULT_PLAN
    if plan_path.exists():
        with open(plan_path, encoding="utf-8") as fh:
            plan = yaml.safe_load(fh) or {}
        run_specs = plan.get("runs", [])
        subsets = plan.get("subset_ratios", DEFAULT_SUBSETS)
        seeds = plan.get("seeds", DEFAULT_SEEDS)
        head_types = plan.get("head_types", DEFAULT_HEAD_TYPES)
        head_lr_scales = plan.get("head_lr_scales", DEFAULT_HEAD_LR_SCALES)
    else:
        print(f"Plan {plan_path} not found — using defaults.")
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

    if args.model:
        run_specs = [r for r in run_specs if r.get("model") == args.model]
        if not run_specs:
            raise ValueError(f"Model '{args.model}' not found in plan.")

    if args.subset_ratios:
        subsets = args.subset_ratios
    if args.seeds:
        seeds = args.seeds

    subsets = [float(x) for x in subsets]
    seeds = [int(x) for x in seeds]
    for r in subsets:
        if not (0 < r <= 1):
            raise ValueError(f"subset ratio must be in (0, 1], got {r}")

    return run_specs, subsets, seeds, head_types, head_lr_scales


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------


def _build_model(run_cfg: Config) -> ClassifierModel:
    model_entry = MODEL_REGISTRY.get(run_cfg.model.model)
    gastronet_path = model_entry.gastronet if model_entry else None
    return ClassifierModel(
        base_model=run_cfg.model.model,
        num_classes=run_cfg.model.num_classes,
        class_weights=run_cfg.training.class_weights,
        optimizer=run_cfg.training.optimizer,
        learning_rate=run_cfg.training.learning_rate,
        threshold=run_cfg.model.threshold,
        dropout_rate=run_cfg.model.dropout_rate,
        num_epochs=run_cfg.training.epochs,
        freeze_layers=run_cfg.model.freeze_layers,
        gastronet_path=gastronet_path,
        es_patience=run_cfg.training.es_patience,
        lr_patience=run_cfg.training.lr_patience,
        lr_factor=run_cfg.training.lr_factor,
        weight_decay=run_cfg.training.weight_decay,
        label_smoothing=run_cfg.training.label_smoothing,
        head_type=run_cfg.model.head_type,
        label_col="label",
    )


def _loader_kwargs(cfg: Config, manifest_path: Path) -> dict:
    return {
        "manifest_path": manifest_path,
        "data_dir": cfg.paths.ulcer_processed_dir,
        "img_size": get_img_size(cfg.model.model),
        "batch_size": cfg.training.batch_size,
        "num_workers": min(cfg.training.num_workers, os.cpu_count() or 8),
        "equalize": cfg.training.equalize,
    }


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------


def _run_one(
    run_cfg: Config,
    train_loader,
    val_loader,
    test_loader,
    device: torch.device,
    results_dir: Path,
) -> dict:
    model_name = run_cfg.model.model
    freeze = run_cfg.model.freeze_layers
    freeze_str = "frozen" if freeze == -1 else "finetune"
    pct = int(run_cfg.training.subset_ratio * 100)
    head_type = run_cfg.model.head_type
    seed = run_cfg.training.random_seed
    subset_ratio = run_cfg.training.subset_ratio
    run_name = f"{model_name}_{freeze_str}_{head_type}_{pct}pct_seed{seed}"

    print(f"\n  {'─' * 70}")
    print(f"  {model_name} | {freeze_str} | head={head_type} | {pct}% | seed={seed}")
    print(f"  n_train={loader_dataset_size(train_loader)}")
    print(f"  {'─' * 70}")

    model = _build_model(run_cfg)
    _fit_interrupted = False
    try:
        _, _, checkpoint_dir = model.fit(
            train_loader,
            val_loader,
            epochs=run_cfg.training.epochs,
            device=device,
            use_amp=device.type == "cuda",
            es_patience=run_cfg.training.es_patience,
        )
    except _TrainingInterrupted as _exc:
        _fit_interrupted = True
        checkpoint_dir = _exc.checkpoint_dir
        print("[!] Training interrupted — evaluating best checkpoint.")

    # ── Threshold tuning on validation set ───────────────────────────────
    val_probs, val_labels = collect_probabilities(
        model.base_model, val_loader, device, num_classes=1
    )
    val_sweep = sweep_thresholds(val_probs, val_labels)
    best_thr = find_best_threshold(val_sweep, metric="f1")
    tuned_threshold = best_thr["threshold"]
    print(f"  Tuned threshold (val F1={best_thr['f1']:.4f}): {tuned_threshold:.4f}")

    test_result = model.test_evaluation(
        test_loader,
        device,
        threshold=run_cfg.model.threshold,
        aggregate_by_clip=True,
    )

    # ── Extra metrics at tuned threshold ─────────────────────────────────
    _test_probs = np.asarray(test_result["probabilities_1d"])
    _test_labels = np.asarray(test_result["labels"])
    _test_preds = (_test_probs >= tuned_threshold).astype(int)
    _cm = _sklearn_cm(_test_labels, _test_preds)
    _tn, _fp, _fn, _tp = _cm.ravel()
    tuned_f1 = _sklearn_f1(_test_labels, _test_preds)
    tuned_sensitivity = _tp / (_tp + _fn) if (_tp + _fn) > 0 else 0.0
    tuned_specificity = _tn / (_tn + _fp) if (_tn + _fp) > 0 else 0.0

    # ── Save figures ──────────────────────────────────────────────────────
    figs_dir = results_dir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)
    roc_fig = plot_roc_curve(model_name, test_result["labels"], test_result["probabilities_1d"])
    roc_fig.savefig(figs_dir / f"roc_{run_name}.png", dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(roc_fig)

    # ── Save predictions ──────────────────────────────────────────────────
    preds_dir = results_dir / "predictions" / run_name
    preds_dir.mkdir(parents=True, exist_ok=True)
    np.save(preds_dir / "test_labels.npy", np.asarray(test_result["labels"]))
    np.save(preds_dir / "test_probs.npy", np.asarray(test_result["probabilities_1d"]))

    f1 = test_result["frame_level"]["f1"]
    auroc = test_result["frame_level"]["roc_auc"]
    print(f"  → F1={f1:.4f}  AUROC={auroc:.4f}")

    result = {
        "model": model_name,
        "freeze": freeze_str,
        "head_type": head_type,
        "subset_ratio": subset_ratio,
        "pct_data": pct,
        "seed": seed,
        "n_train": loader_dataset_size(train_loader),
        "f1": f1,
        "auroc": auroc,
        "tuned_threshold": tuned_threshold,
        "tuned_sensitivity": tuned_sensitivity,
        "tuned_specificity": tuned_specificity,
        "tuned_f1": tuned_f1,
        "clip_f1": test_result["clip_level"]["f1"],
        "clip_auroc": test_result["clip_level"]["roc_auc"],
        "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir else None,
    }
    if _fit_interrupted:
        raise _TrainingInterrupted(result=result)
    return result


# ---------------------------------------------------------------------------
# Parallel worker (module-level for multiprocessing spawn compatibility)
# ---------------------------------------------------------------------------


def _parallel_worker(job: dict) -> dict | None:
    """Execute one training run in a subprocess. Called by ProcessPoolExecutor."""
    from pathlib import Path

    import torch

    from src.config import get_img_size, legacy_dict_to_config
    from src.data.dataloader import get_split_loaders, get_test_loader, get_val_loader
    from src.models.classifier import _TrainingInterrupted

    run_cfg = legacy_dict_to_config(job["base_spec"])
    run_cfg.model.head_type = job["head_type"]
    run_cfg.training.random_seed = job["seed"]
    run_cfg.training.subset_ratio = job["subset_ratio"]
    if job["epochs"] is not None:
        run_cfg.training.epochs = job["epochs"]
    run_cfg.training.learning_rate = float(run_cfg.training.learning_rate) * job["lr_scale"]
    run_cfg.training.batch_size = job["batch_size"]

    img_size = get_img_size(run_cfg.model.model)
    lkw = {
        "manifest_path": Path(job["manifest_path"]),
        "data_dir": Path(job["data_dir"]),
        "img_size": img_size,
        "batch_size": run_cfg.training.batch_size,
        "num_workers": job["loader_num_workers"],
        "equalize": job["equalize"],
    }
    train_loader, _ = get_split_loaders(
        subset_ratio=job["subset_ratio"], random_seed=job["seed"], **lkw
    )
    val_loader = get_val_loader(**lkw)
    test_loader = get_test_loader(**lkw)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    try:
        return _run_one(
            run_cfg=run_cfg,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            results_dir=Path(job["results_dir"]),
        )
    except _TrainingInterrupted as exc:
        return exc.result
    except Exception as exc:  # noqa: BLE001
        model = job["base_spec"].get("model", "?")
        pct = int(job["subset_ratio"] * 100)
        print(f"[!] Worker failed — {model} {pct}% seed={job['seed']}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace) -> None:
    cfg = load_config()

    manifest_path = (
        Path(args.manifest)
        if args.manifest
        else cfg.paths.ulcer_splits_dir / "dataset_manifest.csv"
    )
    if not manifest_path.exists():
        raise FileNotFoundError(f"Ulcer manifest not found: {manifest_path}")

    run_specs, subset_ratios, seeds, head_types, head_lr_scales = _get_plan(args, cfg)

    n_total = len(run_specs) * len(head_types) * len(subset_ratios) * len(seeds)
    results_dir = cfg.paths.get_task_output_config("ulcer_data_efficiency")["results_dir"]

    print(f"\n{'=' * 60}")
    print("DATA EFFICIENCY EXPERIMENT — ULCER DETECTION")
    print(f"{'=' * 60}")
    print(f"  Manifest       : {manifest_path}")
    print(f"  Models         : {[r.get('model') for r in run_specs]}")
    print(f"  Fractions      : {[int(r * 100) for r in subset_ratios]}%")
    print(f"  Seeds          : {seeds}")
    print(f"  Head types     : {head_types}")
    print(f"  Planned runs   : {n_total}")
    print(f"  Results dir    : {results_dir}")
    if args.dry_run:
        print("  [DRY RUN] — no training will be run")
        return

    num_workers = args.num_workers
    all_results: list[dict] = []

    if num_workers > 1:
        # ── Parallel path ────────────────────────────────────────────
        dl_workers = max(1, min(cfg.training.num_workers, os.cpu_count() or 8) // num_workers)
        batch_size = args.batch_size or cfg.training.batch_size

        jobs: list[dict] = []
        run_counter = 0
        for subset_ratio in subset_ratios:
            for seed in seeds:
                for base_spec in run_specs:
                    spec_dict = base_spec if isinstance(base_spec, dict) else vars(base_spec)
                    for head_type in head_types:
                        run_counter += 1
                        if args.max_runs and run_counter > args.max_runs:
                            break
                        jobs.append(
                            {
                                "base_spec": spec_dict,
                                "head_type": head_type,
                                "seed": seed,
                                "subset_ratio": float(subset_ratio),
                                "epochs": args.epochs,
                                "lr_scale": float(head_lr_scales.get(head_type, 1.0)),
                                "manifest_path": str(manifest_path),
                                "data_dir": str(cfg.paths.ulcer_processed_dir),
                                "loader_num_workers": dl_workers,
                                "equalize": cfg.training.equalize,
                                "batch_size": batch_size,
                                "results_dir": str(results_dir),
                            }
                        )

        print(f"\n  Parallel mode: {len(jobs)} runs × {num_workers} workers")
        print(f"  batch_size={batch_size}  DataLoader workers/job={dl_workers}")

        mp_ctx = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers, mp_context=mp_ctx
        ) as executor:
            futures = [executor.submit(_parallel_worker, job) for job in jobs]
            try:
                for fut in concurrent.futures.as_completed(futures):
                    result = fut.result()
                    if result is not None:
                        all_results.append(result)
                        print(
                            f"  ✓ {result['model']} {result['pct_data']}%"
                            f" seed={result['seed']}  F1={result['f1']:.4f}"
                            f"  AUROC={result['auroc']:.4f}"
                        )
            except KeyboardInterrupt:
                print("\n[!] Interrupted — cancelling pending jobs.")
                for f in futures:
                    f.cancel()

    else:
        # ── Sequential path ───────────────────────────────────────────
        device = get_device()
        lkw = _loader_kwargs(cfg, manifest_path)

        test_loader = get_test_loader(**lkw)
        val_loader = get_val_loader(**lkw)
        print(f"\n  test : {loader_dataset_size(test_loader)} frames")
        print(f"  val  : {loader_dataset_size(val_loader)} frames")

        run_counter = 0
        _interrupted = False

        try:
            for subset_ratio in subset_ratios:
                if _interrupted:
                    break
                for seed in seeds:
                    if _interrupted:
                        break

                    print(f"\n{'─' * 60}")
                    print(f"  Fraction {int(subset_ratio * 100)}%  |  seed={seed}")

                    train_loader, _ = get_split_loaders(
                        subset_ratio=subset_ratio,
                        random_seed=seed,
                        **lkw,
                    )
                    print(f"  train : {loader_dataset_size(train_loader)} frames")

                    for base_spec in run_specs:
                        if _interrupted:
                            break
                        base_cfg = (
                            legacy_dict_to_config(base_spec)
                            if isinstance(base_spec, dict)
                            else deepcopy(base_spec)
                        )
                        for head_type in head_types:
                            if _interrupted:
                                break

                            run_counter += 1
                            if args.max_runs and run_counter > args.max_runs:
                                print(f"Reached --max-runs={args.max_runs}, stopping.")
                                _interrupted = True
                                break

                            run_cfg = deepcopy(base_cfg)
                            run_cfg.model.head_type = head_type
                            run_cfg.training.random_seed = seed
                            run_cfg.training.subset_ratio = subset_ratio
                            if args.epochs is not None:
                                run_cfg.training.epochs = args.epochs
                            base_lr = float(base_cfg.training.learning_rate)
                            run_cfg.training.learning_rate = base_lr * float(
                                head_lr_scales.get(head_type, 1.0)
                            )

                            try:
                                result = _run_one(
                                    run_cfg=run_cfg,
                                    train_loader=train_loader,
                                    val_loader=val_loader,
                                    test_loader=test_loader,
                                    device=device,
                                    results_dir=results_dir,
                                )
                                all_results.append(result)
                            except _TrainingInterrupted as exc:
                                if exc.result is not None:
                                    all_results.append(exc.result)
                                print("\n[!] Run interrupted — moving to aggregation.")
                                _interrupted = True
                                break
                            except KeyboardInterrupt:
                                print("\n[!] Interrupted between runs — aggregating.")
                                _interrupted = True
                                break

        except KeyboardInterrupt:
            print("\n[!] Interrupted — moving to aggregation.")

    if not all_results:
        print("No results to aggregate.")
        return

    results_df = pd.DataFrame(all_results)

    # Best checkpoint per config = seed with highest AUROC
    best_ckpt = (
        results_df.sort_values("auroc", ascending=False)
        .drop_duplicates(subset=["model", "freeze", "head_type", "pct_data", "subset_ratio"])[
            [
                "model",
                "freeze",
                "head_type",
                "pct_data",
                "subset_ratio",
                "checkpoint_dir",
                "seed",
                "auroc",
            ]
        ]
        .rename(
            columns={
                "checkpoint_dir": "best_checkpoint_dir",
                "seed": "best_seed",
                "auroc": "best_seed_auroc",
            }
        )
    )
    agg_df = (
        results_df.groupby(["model", "freeze", "head_type", "pct_data", "subset_ratio"])
        .agg(
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            auroc_mean=("auroc", "mean"),
            auroc_std=("auroc", "std"),
            tuned_threshold_mean=("tuned_threshold", "mean"),
            tuned_threshold_std=("tuned_threshold", "std"),
            tuned_sensitivity_mean=("tuned_sensitivity", "mean"),
            tuned_sensitivity_std=("tuned_sensitivity", "std"),
            tuned_specificity_mean=("tuned_specificity", "mean"),
            tuned_specificity_std=("tuned_specificity", "std"),
            tuned_f1_mean=("tuned_f1", "mean"),
            tuned_f1_std=("tuned_f1", "std"),
            clip_f1_mean=("clip_f1", "mean"),
            clip_f1_std=("clip_f1", "std"),
            clip_auroc_mean=("clip_auroc", "mean"),
            clip_auroc_std=("clip_auroc", "std"),
            n_train=("n_train", "first"),
        )
        .reset_index()
        .merge(best_ckpt, on=["model", "freeze", "head_type", "pct_data", "subset_ratio"])
    )

    print(f"\n{'=' * 60}\nAGGREGATED RESULTS\n{'=' * 60}")
    print(
        agg_df[["model", "pct_data", "f1_mean", "f1_std", "auroc_mean", "auroc_std"]].to_string(
            index=False
        )
    )

    results_dir.mkdir(parents=True, exist_ok=True)
    raw_path = results_dir / "results_per_seed.csv"
    agg_path = results_dir / "results.csv"
    results_df.to_csv(raw_path, index=False)
    agg_df.to_csv(agg_path, index=False)
    print(f"\nPer-seed results → {raw_path}")
    print(f"Aggregated results → {agg_path}")

    # ── Save learning curve plots ─────────────────────────────────────────
    import matplotlib.pyplot as plt

    figs_dir = results_dir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)
    for metric in ("f1", "auroc"):
        fig = plot_learning_curves(
            results_df=agg_df,
            metric=metric,
            title=f"Data efficiency — {metric.upper()} vs training set size (Ulcer)",
        )
        out_path = figs_dir / f"learning_curve_{metric}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Learning curve ({metric}) → {out_path}")

    # ── Save experiment metadata ──────────────────────────────────────────
    meta = {
        "task": "ulcer_detection",
        "manifest": str(manifest_path),
        "subset_ratios": subset_ratios,
        "seeds": seeds,
        "head_types": head_types,
        "n_runs": len(all_results),
    }
    (results_dir / "experiment_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data efficiency experiment — ulcer detection")
    parser.add_argument("--plan", default=None, help="Path to experiment plan YAML")
    parser.add_argument("--model", default=None, help="Filter to a single model")
    parser.add_argument("--manifest", default=None, help="Path to manifest CSV")
    parser.add_argument(
        "--subset-ratios", type=float, nargs="+", default=None, help="Override subset ratios"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Override random seeds")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument(
        "--max-runs", type=int, default=None, help="Stop after N runs (for smoke tests)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without training")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel training jobs (default: 1 = sequential).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size (parallel mode only; sequential uses config value)",
    )
    args = parser.parse_args()
    main(args)
