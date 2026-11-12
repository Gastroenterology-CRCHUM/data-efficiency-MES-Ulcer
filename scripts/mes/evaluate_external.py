"""External test of the MES checkpoints, inference only: internal-trained runs on the LIMUC
test split and LIMUC-trained runs on the internal test split.

    python -m scripts.mes.evaluate_external                        # both directions
    python -m scripts.mes.evaluate_external --direction internal_to_limuc
    python -m scripts.mes.evaluate_external --dry-run
    python -m scripts.mes.evaluate_external --vitb                 # the exploratory ViT-B runs

Every run of the source ``results_per_seed.csv`` is reloaded with its own preprocessing
(``load_run_model``) and evaluated with the metrics of ``run_data_efficiency.py``. Checked
first: every checkpoint exists, the target test labels match the target's in-domain
predictions, and a source run reproduces its stored test metrics.

Outputs (``<source results dir>/external_<TARGET>``)
    results_per_seed.csv     source rows with the target-test metrics
    results.csv              mean/std across seeds
    experiment_meta.json     direction, paths, target test summary, checks, git SHA
    predictions/<run_name>/  test_{labels,probs}.npy

The tables are written once every source run has a row; the next invocation resumes from
``results_per_seed.partial.csv``. ``--overwrite`` deletes the output folder first.
"""

from __future__ import annotations

import argparse
import gc
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.mes.run_data_efficiency import (
    DEFAULT_CONFIG,
    FINAL_META,
    LABEL_COL,
    PARTIAL_CSV,
    TASK,
    TASK_NAME,
    _nan_if_missing,
    _repo_path,
    _utf8_stdio,
    _versions,
    _write_meta,
)
from scripts.run_all import LIMUC_DEFAULTS, VITB_DEFAULTS
from src.config import load_config
from src.data.dataloader import get_test_loader
from src.data.manifest_utils import load_manifest, split_summary
from src.models.checkpoint import load_run_model
from src.utils import get_device, get_git_sha, seed_everything, sigterm_as_interrupt
from src.utils.results import (
    aggregate_seed_results,
    mes_extra_metrics,
    portable_path,
    remove_run_outputs,
    run_dir_of,
)

DIRECTIONS = {"internal_to_limuc": ("internal", "LIMUC"), "limuc_to_internal": ("LIMUC", "internal")}
"""Direction -> (dataset of the trained runs, dataset of the test split)."""
REPRO_KEYS = ("auroc", "clin_auroc", "qwk")
REPRO_TOL = 1e-4
PER_SEED_CSV = "results_per_seed.csv"


# ---------------------------------------------------------------------------
# Datasets and checks
# ---------------------------------------------------------------------------


def _datasets(vitb: bool = False) -> dict[str, dict[str, Path]]:
    """Manifest, image root and results folder: internal data (configs/base.yaml), LIMUC (run_all).

    ``vitb``: the results folders of the exploratory ViT-B runs (same data).
    """
    paths = load_config(DEFAULT_CONFIG).paths
    if vitb:
        sets = _datasets()
        for name, task in (("internal", "mes"), ("LIMUC", "limuc")):
            sets[name]["results"] = _repo_path(VITB_DEFAULTS[task]["results-dir"])
        return sets
    return {
        "internal": {
            "manifest": _repo_path(paths.mes_splits_dir) / "dataset_manifest.csv",
            "data_dir": _repo_path(paths.mes_processed_dir),
            "results": _repo_path(paths.results_mes_data_efficiency_dir),
        },
        "LIMUC": {
            "manifest": _repo_path(LIMUC_DEFAULTS["manifest"]),
            "data_dir": _repo_path(LIMUC_DEFAULTS["data-dir"]),
            "results": _repo_path(LIMUC_DEFAULTS["results-dir"]),
        },
    }


def _test_summary(manifest_df: pd.DataFrame) -> dict:
    """Frames, clips and patients of the test split, and frames per grade."""
    s = split_summary(manifest_df, label_col=LABEL_COL)
    return {
        **{key: s[key]["test"] for key in ("n_frames", "n_clips", "n_patients")},
        "frames_per_grade": {grade: d["test"]["frames"] for grade, d in s["per_class"].items()},
    }


def _missing_checkpoints(run_dirs: dict[str, Path]) -> list[str]:
    return [
        name
        for name, d in run_dirs.items()
        if not ((d / "best.pt").is_file() and (d / "run_config.json").is_file())
    ]


def _check_target_labels(test_labels: np.ndarray, target_results: Path) -> str:
    """Return "match" when the target's in-domain test_labels.npy equal ``test_labels``, else "skipped"
    when the target has no predictions.

    Raises:
        ValueError: the target manifest changed since its runs, so the external predictions
            would not be paired with the in-domain ones.
    """
    ref_path = next(iter(sorted((target_results / "predictions").glob("*/test_labels.npy"))), None)
    if ref_path is None:
        print("  [!] No in-domain predictions of the target: label alignment not checked.")
        return "skipped"
    if not np.array_equal(np.load(ref_path).astype(int).ravel(), test_labels):
        raise ValueError(
            f"Target test labels differ from {portable_path(ref_path)}: the target manifest changed "
            "since its runs, so external and in-domain predictions would not be paired."
        )
    return "match"


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _load(run_dir: Path):
    model = load_run_model(run_dir)
    model.n_bootstrap = 0  # point estimates; the CIs come from the bootstrap of src.analysis.statistics
    return model


def _preprocessing(model) -> tuple:
    return int(model.img_size), bool(model.equalize), tuple(model.norm_mean), tuple(model.norm_std)


def _test_loader(dataset: dict, prep: tuple, args: argparse.Namespace):
    img_size, equalize, mean, std = prep
    return get_test_loader(
        dataset["manifest"],
        dataset["data_dir"],
        args.batch_size,
        img_size,
        LABEL_COL,
        num_workers=args.num_workers,
        equalize=equalize,
        mean=mean,
        std=std,
    )


def _evaluate(model, loader, device: torch.device) -> dict:
    return model.test_evaluation(loader, device, threshold=None, aggregate_by_clip=True)


def _metrics(result: dict) -> dict:
    """Per-seed metric columns of run_data_efficiency.py from a ``test_evaluation`` result."""
    frame, clip = result["frame_level"], result.get("clip_level")
    probs = np.asarray(result["probabilities"])
    return {
        "f1": float(frame["f1"]),
        "auroc": float(frame["roc_auc"]),
        "clip_f1": _nan_if_missing(clip, "f1"),
        "clip_auroc": _nan_if_missing(clip, "roc_auc"),
        **mes_extra_metrics(result["labels"], probs, probs.shape[1]),
    }


def _release(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _reproduction_check(
    row: dict, run_dir: Path, source: dict, args: argparse.Namespace, device: torch.device
) -> dict:
    """Re-evaluate ``row`` on its own test split; its stored auroc, clin_auroc and qwk must come back.

    Raises:
        RuntimeError: a metric differs by more than REPRO_TOL (checkpoint, preprocessing or
            source manifest no longer those of the run).
    """
    model = _load(run_dir)
    got = _metrics(_evaluate(model, _test_loader(source, _preprocessing(model), args), device))
    del model
    _release(device)
    diff = {k: abs(got[k] - float(row[k])) for k in REPRO_KEYS}
    if not all(d <= REPRO_TOL for d in diff.values()):  # NaN fails too
        raise RuntimeError(
            f"{row['run_name']} does not reproduce its test metrics on its own test split "
            f"(|diff| {diff}): checkpoint, preprocessing or source manifest changed."
        )
    print(f"  Reproduction check passed ({row['run_name']}, max |diff| {max(diff.values()):.1e}).")
    return {"run_name": row["run_name"], "max_abs_diff": max(diff.values())}


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def _finished_rows(out_dir: Path, runs: list[dict]) -> dict[str, dict]:
    """Rows already evaluated (final, then partial table) whose checkpoint is still the source's."""
    checkpoint = {r["run_name"]: r["checkpoint_dir"] for r in runs}
    rows: dict[str, dict] = {}
    for name in (PER_SEED_CSV, PARTIAL_CSV):
        if (out_dir / name).is_file():
            for r in pd.read_csv(out_dir / name).to_dict("records"):
                if checkpoint.get(r["run_name"]) == r["checkpoint_dir"]:
                    rows[r["run_name"]] = r
    return rows


def _table(rows: dict[str, dict], runs: list[dict]) -> pd.DataFrame:
    """Evaluated rows in source order."""
    return pd.DataFrame([rows[r["run_name"]] for r in runs if r["run_name"] in rows])


def _save_predictions(preds_dir: Path, result: dict) -> None:
    preds_dir.mkdir(parents=True, exist_ok=True)
    np.save(preds_dir / "test_labels.npy", np.asarray(result["labels"]))
    np.save(preds_dir / "test_probs.npy", np.asarray(result["probabilities"]))


# ---------------------------------------------------------------------------
# One direction
# ---------------------------------------------------------------------------


def _run_direction(
    name: str, args: argparse.Namespace, device: torch.device, git_sha: str | None
) -> None:
    src_name, tgt_name = DIRECTIONS[name]
    sets = _datasets(args.vitb)
    source, target = sets[src_name], sets[tgt_name]
    out_dir = source["results"] / f"external_{tgt_name}"
    per_seed_path = source["results"] / PER_SEED_CSV
    if not per_seed_path.is_file():  # runs not finished (e.g. the optional ViT-B runs)
        print(f"\n=== {name}: no finished {src_name} runs ({portable_path(per_seed_path)}): skipped ===")
        return
    per_seed = pd.read_csv(per_seed_path)
    runs = per_seed.to_dict("records")
    run_dirs = {r["run_name"]: run_dir_of(per_seed, r["model"], r["pct_data"], r["seed"]) for r in runs}
    manifest_df = load_manifest(target["manifest"])
    test_labels = manifest_df.loc[manifest_df["split"] == "test", LABEL_COL].to_numpy(dtype=int)
    summary = _test_summary(manifest_df)

    done = {} if args.overwrite else _finished_rows(out_dir, runs)
    todo = [r for r in runs if r["run_name"] not in done]
    missing = _missing_checkpoints({r["run_name"]: run_dirs[r["run_name"]] for r in todo})
    kind = "ViT-B runs" if args.vitb else "runs"
    print(f"\n=== {name}: {src_name}-trained {kind} on the {tgt_name} test split ===")
    print(f"Source runs    : {len(runs)} ({portable_path(per_seed_path)})")
    print(
        f"Target test    : {summary['n_frames']} frames, {summary['n_patients']} patients, "
        f"frames per grade {summary['frames_per_grade']}"
    )
    print(f"Output         : {portable_path(out_dir)}" + (" (deleted first)" if args.overwrite else ""))
    print(f"To evaluate    : {len(todo)} ({len(done)} already done)")
    print(f"Checkpoints    : {f'{len(missing)} missing, e.g. {missing[:3]}' if missing else 'all found'}")
    if args.dry_run:
        print("[DRY RUN] Nothing evaluated.")
        return
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} run(s) without best.pt / run_config.json, e.g. {missing[:3]}"
        )
    for data_dir in (source["data_dir"], target["data_dir"]):
        if not data_dir.is_dir():
            raise FileNotFoundError(f"Image folder not found: {portable_path(data_dir)}")
    if not todo and (out_dir / PER_SEED_CSV).is_file() and not (out_dir / PARTIAL_CSV).is_file():
        print("Up to date (--overwrite to evaluate again).")
        return
    if args.overwrite:
        remove_run_outputs([out_dir])

    # Against the target's main runs, which always have predictions (the ViT-B ones may not yet).
    checks = {"target_labels": _check_target_labels(test_labels, _datasets()[tgt_name]["results"])}
    first = (todo or runs)[0]
    checks["reproduction"] = _reproduction_check(first, run_dirs[first["run_name"]], source, args, device)

    # Grouped by backbone, so that one test loader serves all the runs of a backbone.
    batch = sorted(todo, key=lambda r: r["model"])[: args.max_runs]
    rows = dict(done)
    loader, prep = None, None
    for i, row in enumerate(batch, 1):
        model = _load(run_dirs[row["run_name"]])
        if _preprocessing(model) != prep:
            loader = None  # its workers exit before the next loader starts its own
            prep = _preprocessing(model)
            loader = _test_loader(target, prep, args)
        result = _evaluate(model, loader, device)
        del model
        _save_predictions(out_dir / "predictions" / row["run_name"], result)
        metrics = _metrics(result)
        rows[row["run_name"]] = {
            **row, **metrics, "source": src_name, "target": tgt_name, "eval_git_sha": git_sha,
        }
        _table(rows, runs).to_csv(out_dir / PARTIAL_CSV, index=False)
        print(
            f"[{i}/{len(batch)}] {row['run_name']}  auroc={metrics['auroc']:.4f}  "
            f"clin_auroc={metrics['clin_auroc']:.4f}  qwk={metrics['qwk']:.4f}"
        )
        _release(device)
    loader = None

    if len(rows) < len(runs):
        print(f"{len(runs) - len(rows)} run(s) left: run the same command again to continue.")
        return
    df = _table(rows, runs)
    agg = aggregate_seed_results(df, TASK)
    df.to_csv(out_dir / PER_SEED_CSV, index=False)
    agg.to_csv(out_dir / "results.csv", index=False)
    meta = {
        "task": TASK_NAME,
        "direction": name,
        "source": src_name,
        "target": tgt_name,
        "target_split": "test",
        "created": datetime.now().isoformat(timespec="seconds"),
        "git_sha": git_sha,
        "source_results": portable_path(per_seed_path),
        "target_manifest": portable_path(target["manifest"]),
        "target_data_dir": portable_path(target["data_dir"]),
        "target_test": summary,
        "n_runs": len(df),
        "checks": checks,
        "versions": _versions(),
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "metrics": {
            "f1, auroc, clip_*, clin_*, qwk, sens / spec": "as in the source results, on the target "
            "test split",
            "val_auroc": "source validation AUROC (checkpoint criterion); the training columns "
            "are the source run's",
            "estimates": "point estimates on the full target test set; CIs are not stored here",
        },
    }
    _write_meta(meta, out_dir, FINAL_META)
    (out_dir / PARTIAL_CSV).unlink(missing_ok=True)
    cols = ["model", "pct_data", "n_seeds", "auroc_mean", "clin_auroc_mean", "qwk_mean"]
    print(agg[[c for c in cols if c in agg.columns]].to_string(index=False))
    print(f"Results -> {portable_path(out_dir)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="External test of the MES checkpoints, internal <-> LIMUC (inference only)"
    )
    parser.add_argument(
        "--direction", choices=[*DIRECTIONS, "both"], default="both", help="Default: both"
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Evaluate at most N pending runs per direction (smoke test; rerun to continue)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Delete the output folder and evaluate every run"
    )
    parser.add_argument("--dry-run", action="store_true", help="List the runs and exit")
    parser.add_argument(
        "--vitb", action="store_true", help="Evaluate the exploratory ViT-B runs (their own results folders)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Default: training.batch_size of base.yaml"
    )
    parser.add_argument(
        "--num-workers", type=int, default=None, help="Default: training.num_workers of base.yaml"
    )
    parser.add_argument(
        "--device-id", type=int, default=None, help="Default: training.device_id (< 0 = CPU)"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    _utf8_stdio()
    args = build_parser().parse_args(argv)
    if args.max_runs is not None and args.max_runs < 1:
        raise SystemExit("--max-runs must be >= 1")
    training = load_config(DEFAULT_CONFIG).training
    for key in ("batch_size", "num_workers", "device_id"):
        if getattr(args, key) is None:
            setattr(args, key, getattr(training, key))
    device = get_device(args.device_id)
    if not args.dry_run:
        seed_everything(training.random_seed, deterministic=True)
    git_sha = get_git_sha()
    for name in DIRECTIONS if args.direction == "both" else [args.direction]:
        _run_direction(name, args, device, git_sha)
    return 0


if __name__ == "__main__":
    sigterm_as_interrupt()
    sys.exit(main())
