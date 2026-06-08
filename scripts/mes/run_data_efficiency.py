"""
scripts/mes/run_data_efficiency.py
====================================
Data efficiency experiment — MES multiclass learning curves.

Trains MES models on multiple train-set fractions and aggregates results
across seeds. Saves all results as CSV files and learning curve plots.

Usage
-----
    python -m scripts.mes.run_data_efficiency
    python -m scripts.mes.run_data_efficiency --plan configs/experiments/data_efficiency.yaml
    python -m scripts.mes.run_data_efficiency --dry-run
    python -m scripts.mes.run_data_efficiency --model vits16_imagenet
    python -m scripts.mes.run_data_efficiency --manifest data/mes/splits/dataset_manifest.csv
    python -m scripts.mes.run_data_efficiency --subset-ratios 0.1 0.25 0.5 0.75 1.0
    python -m scripts.mes.run_data_efficiency --epochs 50 --num-workers 2
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
from sklearn.metrics import (
    confusion_matrix as _sklearn_cm,
)
from sklearn.metrics import (
    f1_score as _sklearn_f1,
)
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from src.config import MODEL_REGISTRY, Config, get_img_size, legacy_dict_to_config, load_config
from src.data.dataloader import get_split_loaders, get_test_loader
from src.evaluation.plots import (
    plot_confusion_matrix_multiclass,
    plot_learning_curves,
    plot_roc_curves,
)
from src.models.classifier import ClassifierModel, _TrainingInterrupted
from src.utils import get_device, loader_dataset_size

DEFAULT_PLAN = Path("configs/experiments/data_efficiency.yaml")
LABEL_COL = "label"
DEFAULT_SUBSETS = [0.10, 0.25, 0.50, 0.75, 1.00]
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
        print(f"Plan file not found: {plan_path}. Falling back to defaults.")
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
        run_specs = [spec for spec in run_specs if spec.get("model") == args.model]
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


def _build_model(run_cfg: Config, num_classes: int) -> ClassifierModel:
    model_entry = MODEL_REGISTRY.get(run_cfg.model.model)
    gastronet_path = model_entry.gastronet if model_entry else None
    return ClassifierModel(
        base_model=run_cfg.model.model,
        num_classes=num_classes,
        class_weights=run_cfg.training.class_weights,
        optimizer=run_cfg.training.optimizer,
        learning_rate=run_cfg.training.learning_rate,
        threshold=0.5,
        dropout_rate=run_cfg.training.dropout_rate,
        num_epochs=run_cfg.training.epochs,
        freeze_layers=run_cfg.model.freeze_layers,
        gastronet_path=gastronet_path,
        es_patience=run_cfg.training.es_patience,
        lr_patience=run_cfg.training.lr_patience,
        lr_factor=run_cfg.training.lr_factor,
        weight_decay=run_cfg.training.weight_decay,
        label_smoothing=run_cfg.training.label_smoothing,
        head_type=run_cfg.model.head_type,
        label_col=LABEL_COL,
    )


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------


def _run_one(
    run_cfg: Config,
    *,
    device: torch.device,
    train_loader,
    val_loader,
    test_loader,
    num_classes: int,
    models_root: Path,
    results_dir: Path,
) -> dict:
    model_name = run_cfg.model.model
    freeze = run_cfg.model.freeze_layers
    freeze_str = "frozen" if freeze == -1 else "finetune"
    head_type = run_cfg.model.head_type
    seed = run_cfg.training.random_seed
    subset_ratio = run_cfg.training.subset_ratio
    pct = int(subset_ratio * 100)

    run_name = f"{model_name}_{freeze_str}_{head_type}_{pct}pct_seed{seed}"
    print("\n" + "-" * 72)
    print(f"{run_name} | n_train={loader_dataset_size(train_loader)}")
    print("-" * 72)

    model = _build_model(run_cfg, num_classes=num_classes)
    _fit_interrupted = False
    try:
        _, _, checkpoint_dir = model.fit(
            train_loader,
            val_loader,
            run_cfg.training.epochs,
            device,
            checkpoint_root=models_root,
        )
    except _TrainingInterrupted as _exc:
        _fit_interrupted = True
        checkpoint_dir = _exc.checkpoint_dir
        print("[!] Training interrupted — evaluating best checkpoint.")

    test_result = model.test_evaluation(
        test_loader,
        device,
        threshold=0.5,
        aggregate_by_clip=True,
    )

    class_names = [f"Mayo {i}" for i in range(num_classes)]

    # ── Save figures ──────────────────────────────────────────────────────
    import matplotlib.pyplot as plt

    figs_dir = results_dir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    cm_fig = plot_confusion_matrix_multiclass(
        test_result["frame_level"]["confusion_matrix"],
        class_names=class_names,
        title=f"Confusion matrix - Mayo classification ({run_name})",
    )
    cm_fig.savefig(figs_dir / f"cm_{run_name}.png", dpi=150, bbox_inches="tight")
    plt.close(cm_fig)

    y_true = test_result["labels"]
    y_prob = test_result["probabilities"]
    classes = list(range(num_classes))
    y_bin = np.asarray(label_binarize(y_true, classes=classes))
    roc_data = []
    for i, name in enumerate(class_names):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        auc = roc_auc_score(y_bin[:, i], y_prob[:, i])
        roc_data.append({"name": name, "fpr": fpr, "tpr": tpr, "auc": auc})
    if roc_data:
        roc_fig = plot_roc_curves(roc_data, title=f"ROC (OVR) — {run_name}")
        roc_fig.savefig(figs_dir / f"roc_{run_name}.png", dpi=150, bbox_inches="tight")
        plt.close(roc_fig)

    # ── Save predictions ──────────────────────────────────────────────────
    preds_dir = results_dir / "predictions" / run_name
    preds_dir.mkdir(parents=True, exist_ok=True)
    np.save(preds_dir / "test_labels.npy", np.asarray(test_result["labels"]))
    np.save(preds_dir / "test_probs.npy", np.asarray(test_result["probabilities"]))

    f1 = float(test_result["frame_level"]["f1"])
    auroc = float(test_result["frame_level"]["roc_auc"])
    _clip = test_result.get("clip_level") or test_result["frame_level"]
    clip_f1 = float(_clip["f1"])
    clip_auroc = float(_clip["roc_auc"])
    print(f"  -> F1={f1:.4f} AUROC={auroc:.4f}  clip_F1={clip_f1:.4f} clip_AUROC={clip_auroc:.4f}")

    # ── Extended MES metrics ──────────────────────────────────────────────
    _y_true = np.asarray(test_result["labels"])
    _y_prob = np.asarray(test_result["probabilities"])
    _y_pred = _y_prob.argmax(axis=1)

    micro_f1 = float(_sklearn_f1(_y_true, _y_pred, average="micro"))
    macro_f1 = float(_sklearn_f1(_y_true, _y_pred, average="macro"))

    _cm = _sklearn_cm(_y_true, _y_pred, labels=list(range(num_classes)))
    _row_sums = _cm.sum(axis=1)
    per_sens = {
        f"sens_mayo{i}": float(_cm[i, i] / _row_sums[i]) if _row_sums[i] > 0 else 0.0
        for i in range(num_classes)
    }
    per_spec = {}
    for i in range(num_classes):
        _tn = _cm.sum() - _cm[i].sum() - _cm[:, i].sum() + _cm[i, i]
        _fp = _cm[:, i].sum() - _cm[i, i]
        per_spec[f"spec_mayo{i}"] = float(_tn / (_tn + _fp)) if (_tn + _fp) > 0 else 0.0

    # Clinical binary: Active (Mayo 2-3) vs Remission (Mayo 0-1)
    _y_clin_true = (_y_true >= 2).astype(int)
    _y_clin_pred = (_y_pred >= 2).astype(int)
    _cm_clin = _sklearn_cm(_y_clin_true, _y_clin_pred)
    _tn_c, _fp_c, _fn_c, _tp_c = _cm_clin.ravel()
    clin_sens = float(_tp_c / (_tp_c + _fn_c)) if (_tp_c + _fn_c) > 0 else 0.0
    clin_spec = float(_tn_c / (_tn_c + _fp_c)) if (_tn_c + _fp_c) > 0 else 0.0
    clin_f1 = float(_sklearn_f1(_y_clin_true, _y_clin_pred))

    print(
        f"  micro_F1={micro_f1:.3f}  macro_F1={macro_f1:.3f}"
        f"  clin_sens={clin_sens:.3f}  clin_spec={clin_spec:.3f}"
    )

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
        "clip_f1": clip_f1,
        "clip_auroc": clip_auroc,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "clin_sensitivity": clin_sens,
        "clin_specificity": clin_spec,
        "clin_f1": clin_f1,
        "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir else None,
        **per_sens,
        **per_spec,
    }
    if _fit_interrupted:
        raise _TrainingInterrupted(result=result)
    return result


# ---------------------------------------------------------------------------
# Parallel worker
# ---------------------------------------------------------------------------


def _parallel_worker(job: dict) -> dict | None:
    """Execute one MES training run in a subprocess."""
    from pathlib import Path

    import torch

    from src.config import get_img_size, legacy_dict_to_config
    from src.data.dataloader import get_split_loaders, get_test_loader
    from src.models.classifier import _TrainingInterrupted

    run_cfg = legacy_dict_to_config(job["base_spec"])
    run_cfg.model.num_classes = job["num_classes"]
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
        "label_col": LABEL_COL,
    }
    train_loader, val_loader = get_split_loaders(
        subset_ratio=job["subset_ratio"], random_seed=job["seed"], **lkw
    )
    test_loader = get_test_loader(**lkw)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    try:
        return _run_one(
            run_cfg,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            num_classes=job["num_classes"],
            models_root=Path(job["models_root"]),
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
        else (cfg.paths.mes_splits_dir / "dataset_manifest.csv")
    )
    data_dir = Path(args.data_dir) if args.data_dir else cfg.paths.mes_processed_dir
    models_root = cfg.paths.get_task_output_config("mes")["models_dir"]
    results_dir = cfg.paths.get_task_output_config("mes_data_efficiency")["results_dir"]

    if not manifest_path.exists():
        raise FileNotFoundError(f"MES manifest not found: {manifest_path}")

    manifest_df = pd.read_csv(manifest_path)
    if LABEL_COL not in manifest_df.columns:
        raise ValueError(f"Manifest must contain '{LABEL_COL}' column: {manifest_path}")

    num_classes = int(manifest_df[LABEL_COL].nunique())
    if num_classes < 2:
        raise ValueError(f"Expected multiclass labels in MES manifest, got {num_classes} class.")

    run_specs, subset_ratios, seeds, head_types, head_lr_scales = _get_plan(args, cfg)

    planned_runs = len(run_specs) * len(head_types) * len(subset_ratios) * len(seeds)
    print("\n" + "=" * 72)
    print("MES DATA EFFICIENCY")
    print("=" * 72)
    print(f"Manifest       : {manifest_path}")
    print(f"Data dir       : {data_dir}")
    print(f"Classes        : {num_classes}")
    print(f"Models         : {[r.get('model') for r in run_specs]}")
    print(f"Subset ratios  : {subset_ratios}")
    print(f"Seeds          : {seeds}")
    print(f"Head types     : {head_types}")
    print(f"Planned runs   : {planned_runs}")
    print(f"Results dir    : {results_dir}")
    if args.dry_run:
        print("[DRY RUN] No training launched.")
        return

    num_workers = args.num_workers
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(cfg.training.device_id))
    torch.backends.cudnn.benchmark = True

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
                                "data_dir": str(data_dir),
                                "models_root": str(models_root),
                                "num_classes": num_classes,
                                "loader_num_workers": dl_workers,
                                "equalize": cfg.training.equalize,
                                "batch_size": batch_size,
                                "results_dir": str(results_dir),
                            }
                        )

        print(f"\n  Parallel mode: {len(jobs)} runs × {num_workers} workers")

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
        _base_lkw = {
            "manifest_path": manifest_path,
            "data_dir": data_dir,
            "num_workers": min(cfg.training.num_workers, os.cpu_count() or 8),
            "equalize": cfg.training.equalize,
            "label_col": LABEL_COL,
        }

        run_counter = 0
        _interrupted = False

        try:
            for subset_ratio in subset_ratios:
                if _interrupted:
                    break
                for seed in seeds:
                    if _interrupted:
                        break
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
                            run_cfg.model.num_classes = num_classes
                            run_cfg.model.head_type = head_type
                            run_cfg.training.random_seed = seed
                            run_cfg.training.subset_ratio = subset_ratio
                            if args.epochs is not None:
                                run_cfg.training.epochs = args.epochs
                            base_lr = float(base_cfg.training.learning_rate)
                            run_cfg.training.learning_rate = base_lr * float(
                                head_lr_scales.get(head_type, 1.0)
                            )

                            img_size = get_img_size(run_cfg.model.model)
                            lkw = {
                                **_base_lkw,
                                "batch_size": run_cfg.training.batch_size,
                                "img_size": img_size,
                            }
                            train_loader, val_loader = get_split_loaders(
                                subset_ratio=subset_ratio,
                                random_seed=seed,
                                **lkw,
                            )
                            test_loader = get_test_loader(**lkw)

                            print(f"\n{'─' * 60}")
                            print(f"  Fraction {int(subset_ratio * 100)}%  |  seed={seed}")
                            print(f"  train : {loader_dataset_size(train_loader)} frames")

                            try:
                                result = _run_one(
                                    run_cfg,
                                    device=device,
                                    train_loader=train_loader,
                                    val_loader=val_loader,
                                    test_loader=test_loader,
                                    num_classes=num_classes,
                                    models_root=models_root,
                                    results_dir=results_dir,
                                )
                                all_results.append(result)
                            except _TrainingInterrupted as exc:
                                if exc.result is not None:
                                    all_results.append(exc.result)
                                print("\n[!] Run interrupted — proceeding to aggregation.")
                                _interrupted = True
                                break
                            except KeyboardInterrupt:
                                _interrupted = True
                                break

        except KeyboardInterrupt:
            print("\n[!] Interrupted — moving to aggregation.")

    if not all_results:
        print("No run results to aggregate.")
        return

    results_df = pd.DataFrame(all_results)
    agg_df = (
        results_df.groupby(["model", "freeze", "head_type", "pct_data", "subset_ratio"])
        .agg(
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            auroc_mean=("auroc", "mean"),
            auroc_std=("auroc", "std"),
            clip_f1_mean=("clip_f1", "mean"),
            clip_f1_std=("clip_f1", "std"),
            clip_auroc_mean=("clip_auroc", "mean"),
            clip_auroc_std=("clip_auroc", "std"),
            micro_f1_mean=("micro_f1", "mean"),
            micro_f1_std=("micro_f1", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            cli_sens_mean=("clin_sensitivity", "mean"),
            cli_sens_std=("clin_sensitivity", "std"),
            cli_spec_mean=("clin_specificity", "mean"),
            cli_spec_std=("clin_specificity", "std"),
            cli_f1_mean=("clin_f1", "mean"),
            cli_f1_std=("clin_f1", "std"),
            n_train=("n_train", "first"),
        )
        .reset_index()
        .sort_values(["model", "subset_ratio"])
    )

    print("\n" + "=" * 72)
    print("AGGREGATED RESULTS")
    print("=" * 72)
    print(
        agg_df[
            [
                "model",
                "pct_data",
                "f1_mean",
                "f1_std",
                "auroc_mean",
                "auroc_std",
                "micro_f1_mean",
                "macro_f1_mean",
                "cli_sens_mean",
                "cli_spec_mean",
            ]
        ].to_string(index=False)
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
            title=f"MES data efficiency — {metric.upper()} vs training set size",
        )
        out_path = figs_dir / f"learning_curve_{metric}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Learning curve ({metric}) → {out_path}")

    # ── Save experiment metadata ──────────────────────────────────────────
    meta = {
        "task": "mes_multiclass",
        "manifest": str(manifest_path),
        "data_dir": str(data_dir),
        "num_classes": num_classes,
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
    parser = argparse.ArgumentParser(description="MES data-efficiency experiment runner")
    parser.add_argument("--plan", default=None, help="Path to experiment YAML plan")
    parser.add_argument("--model", default=None, help="Run a single model from the plan")
    parser.add_argument("--manifest", default=None, help="Path to MES manifest CSV")
    parser.add_argument(
        "--data-dir", default=None, help="Root directory of images referenced by manifest"
    )
    parser.add_argument(
        "--subset-ratios", type=float, nargs="+", default=None, help="Override subset ratios"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Override random seeds")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument(
        "--max-runs", type=int, default=None, help="Stop after N runs (for smoke tests)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Print plan and exit")
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
