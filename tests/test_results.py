"""Result tables: src.utils.results and how the runners write them.

- src.utils.results: labels, portable / redacted paths, per-run metrics and the
  aggregation across seeds.
- scripts/{mes,ulcer}/run_data_efficiency.py (no training): per-run jobs (run names,
  input size, batch size), plan-entry checks, the dataset check that keeps another
  manifest out of the committed results, failures.json redaction, plan-ordered rows,
  the end-of-run handling of incomplete runs and ``--resume``, and the one process per
  run of ``--isolate-runs``.

The runners are imported inside the tests: importing them sets MPLBACKEND and
pulls in torch, which collection should not do.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import Config
from src.utils.results import (
    REPO_ROOT,
    aggregate_seed_results,
    binary_metrics_at_threshold,
    freeze_label,
    mes_extra_metrics,
    pct_label,
    portable_path,
    redact_local_paths,
    remove_run_outputs,
)
from tests.conftest import make_rows

SEEDS = (42, 84, 128)


def _mes_per_seed(with_new_columns: bool = True) -> pd.DataFrame:
    """Two configurations x three seeds of MES per-seed rows."""
    rng = np.random.default_rng(0)
    rows = []
    for model in ("efficientnetb0", "resnet50_5M"):
        for i, seed in enumerate(SEEDS):
            row = {
                "model": model,
                "freeze": "finetune",
                "head_type": "linear",
                "subset_ratio": 0.5,
                "pct_data": 50,
                "seed": seed,
                "n_train": 1000 + 10 * i,
                "f1": rng.random(),
                "auroc": rng.random(),
                "clip_f1": rng.random(),
                "clip_auroc": rng.random(),
                "micro_f1": rng.random(),
                "macro_f1": rng.random(),
                "clin_sensitivity": rng.random(),
                "clin_specificity": rng.random(),
                "clin_f1": rng.random(),
                "checkpoint_dir": f"output/mes/models/{model}/0/linear/run_{seed}",
            }
            for k in range(4):
                row[f"sens_mayo{k}"] = rng.random()
                row[f"spec_mayo{k}"] = rng.random()
            if with_new_columns:
                row.update(
                    learning_rate=2e-5,
                    clin_auroc=rng.random(),
                    qwk=rng.random(),
                    val_auroc=rng.random(),
                    realized_fraction=0.5 + 0.01 * i,
                    n_clips=10 + i,
                    n_patients=5,
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _ulcer_per_seed() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rows = []
    for ratio in (0.1, 1.0):
        for seed in SEEDS:
            rows.append(
                {
                    "model": "efficientnetb0",
                    "freeze": "finetune",
                    "head_type": "linear",
                    "learning_rate": 2e-5,
                    "subset_ratio": ratio,
                    "pct_data": pct_label(ratio),
                    "seed": seed,
                    "n_train": int(5000 * ratio),
                    "f1": rng.random(),
                    "auroc": rng.random(),
                    "tuned_threshold": rng.random(),
                    "tuned_sensitivity": rng.random(),
                    "tuned_specificity": rng.random(),
                    "tuned_f1": rng.random(),
                    "tuned_mcc": rng.random(),
                    "clip_f1": np.nan,  # no clip-level result
                    "clip_auroc": np.nan,
                    "val_auroc": rng.random(),
                    "checkpoint_dir": f"output/ulcer/models/run_{seed}",
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# aggregate_seed_results
# ---------------------------------------------------------------------------


def test_mes_aggregation_means_stds_and_counts():
    per_seed = _mes_per_seed()
    agg = aggregate_seed_results(per_seed, "mes")

    assert len(agg) == 2
    for _, row in agg.iterrows():
        src = per_seed[per_seed["model"] == row["model"]]
        for metric in ("f1", "auroc", "clip_auroc", "macro_f1", "qwk", "sens_mayo2", "val_auroc"):
            assert row[f"{metric}_mean"] == pytest.approx(src[metric].mean())
            assert row[f"{metric}_std"] == pytest.approx(src[metric].std(ddof=1))
        assert row["clin_sens_mean"] == pytest.approx(src["clin_sensitivity"].mean())
        assert row["clin_spec_mean"] == pytest.approx(src["clin_specificity"].mean())
        assert row["clin_auroc_mean"] == pytest.approx(src["clin_auroc"].mean())
        assert row["n_train_mean"] == pytest.approx(1010)
        assert (row["n_train_min"], row["n_train_max"]) == (1000, 1020)
        assert row["n_seeds"] == 3
        assert row["n_clips_mean"] == pytest.approx(11)
        assert row["realized_fraction_mean"] == pytest.approx(0.51)
    assert not [c for c in agg.columns if c.startswith("cli_")]
    assert "n_train" not in agg.columns and "best_seed_auroc" not in agg.columns
    assert list(agg["model"]) == sorted(agg["model"])


def test_mes_per_class_columns_are_ordered():
    agg = aggregate_seed_results(_mes_per_seed(), "mes_multiclass")
    sens = [c for c in agg.columns if c.startswith("sens_mayo") and c.endswith("_mean")]
    spec = [c for c in agg.columns if c.startswith("spec_mayo") and c.endswith("_mean")]
    assert sens == [f"sens_mayo{k}_mean" for k in range(4)]
    assert spec == [f"spec_mayo{k}_mean" for k in range(4)]


def test_best_seed_is_chosen_on_validation_not_test():
    per_seed = _mes_per_seed()
    one = per_seed["model"] == "efficientnetb0"
    per_seed.loc[one, "auroc"] = [0.99, 0.50, 0.60]  # seed 42 wins on test ...
    per_seed.loc[one, "val_auroc"] = [0.70, 0.90, 0.80]  # ... seed 84 wins on val
    row = aggregate_seed_results(per_seed, "mes").set_index("model").loc["efficientnetb0"]

    assert row["best_seed"] == 84
    assert row["best_seed_val_auroc"] == pytest.approx(0.90)
    assert row["best_checkpoint_dir"].endswith("run_84")


def test_best_seed_ties_and_nans():
    per_seed = _mes_per_seed()
    one = per_seed["model"] == "efficientnetb0"
    per_seed.loc[one, "val_auroc"] = [np.nan, 0.8, 0.8]
    row = aggregate_seed_results(per_seed, "mes").set_index("model").loc["efficientnetb0"]
    assert row["best_seed"] == 84  # tie -> lowest seed; NaN never wins

    per_seed.loc[one, "val_auroc"] = np.nan
    agg = aggregate_seed_results(per_seed, "mes").set_index("model")
    assert pd.isna(agg.loc["efficientnetb0", "best_seed"])
    assert agg["best_seed"].dtype == "Int64"


def test_ulcer_aggregation():
    per_seed = _ulcer_per_seed()
    agg = aggregate_seed_results(per_seed, "ulcer")

    assert list(agg["subset_ratio"]) == [0.1, 1.0]
    for metric in ("tuned_threshold", "tuned_sensitivity", "tuned_specificity", "tuned_f1", "tuned_mcc"):
        assert f"{metric}_mean" in agg.columns and f"{metric}_std" in agg.columns
    assert agg["clip_f1_mean"].isna().all()
    assert not [c for c in agg.columns if c.startswith(("clin_", "micro_", "sens_mayo"))]
    assert set(agg["best_seed"]) <= set(SEEDS)


def test_learning_rates_are_separate_groups():
    per_seed = pd.concat(
        [_ulcer_per_seed(), _ulcer_per_seed().assign(learning_rate=1e-5)], ignore_index=True
    )
    agg = aggregate_seed_results(per_seed, "ulcer")
    assert len(agg) == 4
    assert set(agg["n_seeds"]) == {3}
    assert list(agg["learning_rate"]) == [1e-5, 1e-5, 2e-5, 2e-5]


def test_missing_group_values_are_kept():
    per_seed = _ulcer_per_seed()
    per_seed.loc[per_seed["subset_ratio"] == 0.1, "learning_rate"] = np.nan
    agg = aggregate_seed_results(per_seed, "ulcer")
    assert len(agg) == 2
    assert agg["learning_rate"].isna().sum() == 1
    assert agg["best_seed"].notna().all()  # NaN keys still get their best seed


def test_historical_columns_are_tolerated():
    # historical run per-seed CSVs: no learning_rate, val_auroc, realized_fraction, n_clips.
    per_seed = _mes_per_seed(with_new_columns=False)
    agg = aggregate_seed_results(per_seed, "mes")

    assert len(agg) == 2
    assert {"f1_mean", "clin_sens_mean", "n_train_mean", "n_seeds"} <= set(agg.columns)
    assert not {"best_seed", "val_auroc_mean", "clin_auroc_mean"} & set(agg.columns)


def test_failed_runs_lower_n_seeds():
    per_seed = _mes_per_seed().iloc[1:]  # seed 42 of the first model failed
    agg = aggregate_seed_results(per_seed, "mes").set_index("model")
    assert agg.loc["efficientnetb0", "n_seeds"] == 2
    assert agg.loc["resnet50_5M", "n_seeds"] == 3


def test_bad_task_and_missing_keys_raise():
    with pytest.raises(ValueError, match="task"):
        aggregate_seed_results(_mes_per_seed(), "polyp")
    with pytest.raises(ValueError, match="at least one"):
        aggregate_seed_results(pd.DataFrame({"seed": [1], "f1": [0.5]}), "mes")


@pytest.mark.integration
@pytest.mark.parametrize(
    ("task", "experiment"),
    [
        (task, f"{experiment}{vitb}{external}")
        for task, experiment, external in [
            ("ulcer", "data_efficiency", ""),
            ("mes", "data_efficiency", ""),
            ("mes", "data_efficiency", "/external_LIMUC"),
            ("mes", "data_efficiency_LIMUC", ""),
            ("mes", "data_efficiency_LIMUC", "/external_internal"),
        ]
        for vitb in ("", "_vitb")
    ],
)
def test_committed_result_tables_are_reproduced(task, experiment):
    # The committed results.csv must be the aggregate of the committed per-seed CSV.
    folder = REPO_ROOT / "results" / task / experiment
    per_seed_path, table_path = folder / "results_per_seed.csv", folder / "results.csv"
    if not (per_seed_path.exists() and table_path.exists()):
        pytest.skip("committed result tables not present")
    table = pd.read_csv(table_path)
    table = table.rename(columns=lambda c: c.replace("cli_", "clin_", 1))  # historical run spelling
    agg = aggregate_seed_results(pd.read_csv(per_seed_path), task)

    keys = [
        k
        for k in ("model", "freeze", "head_type", "learning_rate", "subset_ratio")
        if k in table.columns and k in agg.columns
    ]
    merged = table.merge(agg, on=keys, suffixes=("_committed", ""), validate="one_to_one")
    assert len(merged) == len(table) == len(agg)
    metrics = [c for c in table.columns if c.endswith(("_mean", "_std")) and c in agg.columns]
    assert metrics
    for col in metrics:
        np.testing.assert_allclose(
            merged[f"{col}_committed"], merged[col], rtol=1e-9, atol=1e-12, err_msg=col
        )


# ---------------------------------------------------------------------------
# Labels and paths
# ---------------------------------------------------------------------------


def test_freeze_and_pct_labels():
    assert freeze_label(0) == "finetune"
    assert freeze_label(-1) == "frozen"
    assert freeze_label(3) == "freeze3"
    assert [pct_label(r) for r in (0.1, 0.25, 0.29, 0.57, 1.0)] == [10, 25, 29, 57, 100]


def test_portable_path_is_never_absolute(tmp_path):
    inside = REPO_ROOT / "results" / "mes" / "x.csv"
    assert portable_path(inside) == "results/mes/x.csv"
    assert portable_path(REPO_ROOT) == "."
    assert portable_path(None) is None

    outside = tmp_path / "deep" / "weights.pth"
    assert portable_path(outside) == "<external>/weights.pth"
    assert portable_path(outside, anchors={"pretrained": tmp_path}) == "pretrained/deep/weights.pth"
    assert portable_path(tmp_path, anchors={"pretrained": tmp_path}) == "pretrained"
    assert portable_path(outside, root=tmp_path) == "deep/weights.pth"
    for value in (portable_path(inside), portable_path(outside)):
        assert not Path(value).is_absolute() and "\\" not in value


def test_redact_local_paths_replaces_machine_prefixes():
    assert redact_local_paths(None) is None
    assert redact_local_paths("") == ""
    assert redact_local_paths("results/mes/x.csv") == "results/mes/x.csv"

    repo = redact_local_paths(f"open {REPO_ROOT / 'src' / 'x.py'} failed")
    assert repo.replace("\\", "/") == "open ./src/x.py failed"
    assert redact_local_paths(f'"{REPO_ROOT.as_posix()}/data"') == '"./data"'
    env = redact_local_paths(f'File "{Path(sys.prefix) / "lib" / "site.py"}", line 1')
    assert env.replace("\\", "/") == 'File "<env>/lib/site.py", line 1'
    home = redact_local_paths(str(Path.home() / "weights.pth"))
    assert home.replace("\\", "/") == "~/weights.pth"
    # Any user's home, with either separator and 8.3 short names.
    for other, expected in (
        ("C:\\Users\\someone\\a.txt", "~\\a.txt"),
        ("C:/Users/ABCDEF~1/a.txt", "~/a.txt"),
        ("/home/someone/a.txt", "~/a.txt"),
        ("/Users/someone/a.txt", "~/a.txt"),
    ):
        assert redact_local_paths(f"x {other}") == f"x {expected}"
    # Prefixes match whole path components only.
    assert not redact_local_paths(f"{REPO_ROOT}-other/x").startswith(".-other")
    if os.name == "nt":  # Windows paths are case-insensitive
        assert redact_local_paths(str(REPO_ROOT).upper() + "\\A") == ".\\A"


def test_remove_run_outputs_deletes_the_existing_folders(tmp_path):
    results, models, missing = tmp_path / "res", tmp_path / "models", tmp_path / "none"
    (results / "predictions").mkdir(parents=True)
    (models / "resnet50_5M").mkdir(parents=True)
    existing = [results.resolve(), models.resolve()]
    assert remove_run_outputs([results, models, missing], dry_run=True) == existing
    assert results.is_dir() and models.is_dir()
    assert remove_run_outputs([results, models, missing]) == existing
    assert not results.exists() and not models.exists()


@pytest.mark.parametrize("target", [".", "..", "results", "output"])
def test_remove_run_outputs_refuses_the_repository_and_its_top_level(target):
    with pytest.raises(ValueError, match="refuses"):
        remove_run_outputs([REPO_ROOT / target], dry_run=True)


def test_remove_run_outputs_refuses_a_folder_holding_a_protected_path(tmp_path):
    frames = tmp_path / "data" / "processed"
    frames.mkdir(parents=True)
    (tmp_path / "res").mkdir()
    with pytest.raises(ValueError, match="refuses"):
        remove_run_outputs([tmp_path / "res", tmp_path / "data"], [frames, None])
    assert frames.is_dir() and (tmp_path / "res").is_dir()  # nothing deleted


# ---------------------------------------------------------------------------
# Per-run metrics
# ---------------------------------------------------------------------------


def test_binary_metrics_at_threshold_hand_case():
    labels = [0, 0, 0, 1, 1]
    probs = [0.1, 0.6, 0.3, 0.6, 0.9]
    out = binary_metrics_at_threshold(labels, probs, 0.6)  # predict 1 iff p >= 0.6
    np.testing.assert_array_equal(out["confusion_matrix"], [[2, 1], [0, 2]])
    assert out["sensitivity"] == 1.0
    assert out["specificity"] == pytest.approx(2 / 3)
    assert out["f1"] == pytest.approx(0.8)
    assert out["mcc"] == pytest.approx(2 / 3)  # (2*2 - 1*0) / sqrt(3*2*3*2)

    two_col = np.column_stack([1 - np.array(probs), probs])
    assert binary_metrics_at_threshold(labels, two_col, 0.6)["f1"] == pytest.approx(0.8)


def test_binary_metrics_single_class_is_nan_not_zero():
    out = binary_metrics_at_threshold([0, 0, 0], [0.1, 0.7, 0.2], 0.5)
    assert out["confusion_matrix"].shape == (2, 2)
    assert np.isnan(out["sensitivity"])
    assert out["specificity"] == pytest.approx(2 / 3)
    assert np.isnan(out["mcc"])


def test_mes_extra_metrics_hand_case():
    labels = np.array([0, 1, 2, 3, 2, 0])
    probs = np.array(
        [
            [0.7, 0.1, 0.1, 0.1],  # 0 -> 0
            [0.1, 0.2, 0.6, 0.1],  # 1 -> 2 (remission called active)
            [0.1, 0.1, 0.7, 0.1],  # 2 -> 2
            [0.1, 0.1, 0.2, 0.6],  # 3 -> 3
            [0.5, 0.3, 0.1, 0.1],  # 2 -> 0 (active called remission)
            [0.6, 0.2, 0.1, 0.1],  # 0 -> 0
        ]
    )
    out = mes_extra_metrics(labels, probs, 4)

    assert out["micro_f1"] == pytest.approx(4 / 6)
    assert out["sens_mayo0"] == 1.0 and out["sens_mayo1"] == 0.0
    assert out["sens_mayo2"] == 0.5 and out["sens_mayo3"] == 1.0
    assert out["spec_mayo0"] == pytest.approx(3 / 4)
    assert out["spec_mayo2"] == pytest.approx(3 / 4)
    # Active disease (Mayo >= 2): y = [0,0,1,1,1,0], argmax active = [0,1,1,1,0,0].
    assert out["clin_sensitivity"] == pytest.approx(2 / 3)
    assert out["clin_specificity"] == pytest.approx(2 / 3)
    assert out["clin_f1"] == pytest.approx(2 / 3)
    # Score p2 + p3: actives 0.8 / 0.8 / 0.2 vs remission 0.2 / 0.7 / 0.2 -> 7 of 9 pairs.
    assert out["clin_auroc"] == pytest.approx(7 / 9)
    assert out["qwk"] == pytest.approx(34 / 49)  # = sklearn cohen_kappa_score(weights="quadratic")


def test_mes_extra_metrics_absent_class_is_nan():
    labels = np.array([0, 0, 1, 1])
    probs = np.array([[0.9, 0.1, 0, 0], [0.8, 0.2, 0, 0], [0.2, 0.8, 0, 0], [0.4, 0.6, 0, 0]])
    out = mes_extra_metrics(labels, probs, 4)
    assert np.isnan(out["sens_mayo2"]) and np.isnan(out["sens_mayo3"])
    assert out["spec_mayo3"] == 1.0
    assert np.isnan(out["clin_auroc"])  # no active-disease frame
    with pytest.raises(ValueError, match=r"\(N, C\)"):
        mes_extra_metrics(labels, probs[:, 0])


# ---------------------------------------------------------------------------
# Runners: per-run jobs and plan-entry checks (no training)
# ---------------------------------------------------------------------------

TASKS = ("mes", "ulcer")
NUM_CLASSES = {"mes": 4, "ulcer": 1}
MIXED_RUNS = (
    "  - {model: efficientnetb0, freeze_layers: 3, learning_rate: 1.0e-5, batch_size: 16}\n"
    "  - {model: efficientnetb0, freeze_layers: 3, learning_rate: 1.0e-4}\n"
    "  - {model: vits16_dinov3, freeze_layers: 0, learning_rate: 2.0e-5}\n"
)


def _runner(task: str):
    return importlib.import_module(f"scripts.{task}.run_data_efficiency")


def _write_plan(tmp_path: Path, models: str, *, seeds: str = "[1]") -> Path:
    path = tmp_path / "plan.yaml"
    path.write_text(
        f"subset_ratios: [0.5]\nseeds: {seeds}\nhead_types: [linear]\nmodels:\n{models}",
        encoding="utf-8",
    )
    return path


def _jobs(task: str, tmp_path: Path, plan_path: Path, *extra: str, parallel_jobs: int = 1):
    """The runner's jobs for a plan file, as main() builds them (no data needed)."""
    runner = _runner(task)
    args = runner.build_parser().parse_args(["--plan", str(plan_path), *extra])
    cfg = Config()
    context = {
        "manifest_path": str(tmp_path / "manifest.csv"),
        "data_dir": str(tmp_path / "frames"),
        "models_root": str(tmp_path / "models"),
        "results_dir": str(tmp_path / "results"),
        "pretrained_dir": str(tmp_path / "pretrained"),
        "manifest_rel": "manifest.csv",
        "plan_rel": "plan.yaml",
        "git_sha": None,
    }
    return runner._build_jobs(
        cfg,
        runner._get_plan(args, cfg),
        args,
        num_classes=NUM_CLASSES[task],
        parallel_jobs=parallel_jobs,
        context=context,
    )


@pytest.mark.parametrize("task", TASKS)
def test_runner_jobs_use_each_runs_model_and_batch_size(task, tmp_path):
    runner = _runner(task)
    plan_path = _write_plan(tmp_path, MIXED_RUNS)
    jobs = _jobs(task, tmp_path, plan_path)

    # freezeN label; entries that differ only by learning rate get an _lr suffix.
    assert [j["run_name"] for j in jobs] == [
        "efficientnetb0_freeze3_linear_lr1e-05_50pct_seed1",
        "efficientnetb0_freeze3_linear_lr0.0001_50pct_seed1",
        "vits16_dinov3_finetune_linear_50pct_seed1",
    ]
    loaders = [runner._loader_kwargs(j["run_cfg"], j) for j in jobs]
    assert [kw["img_size"] for kw in loaders] == [256, 256, 256]  # registry size of each model
    default_bs = Config().training.batch_size
    assert [kw["batch_size"] for kw in loaders] == [16, default_bs, default_bs]
    assert all(kw["seed"] == 1 and kw["manifest_path"].name == "manifest.csv" for kw in loaders)

    # --batch-size applies to every run, in sequential and parallel mode alike.
    for n_procs in (1, 2):
        jobs = _jobs(task, tmp_path, plan_path, "--batch-size", "8", parallel_jobs=n_procs)
        assert [runner._loader_kwargs(j["run_cfg"], j)["batch_size"] for j in jobs] == [8] * 3
    n_loader = min(Config().training.num_workers, os.cpu_count() or 8)
    assert all(j["loader_workers"] == max(1, n_loader // 2) for j in jobs)


@pytest.mark.parametrize("task", TASKS)
def test_runner_rejects_duplicate_and_inconsistent_entries(task, tmp_path):
    same = "  - {model: efficientnetb0, freeze_layers: 3, learning_rate: 1.0e-5}\n"
    with pytest.raises(ValueError, match="Duplicate plan entries"):
        _jobs(task, tmp_path, _write_plan(tmp_path, same * 2))
    # Entries that differ only by batch size would share a run name (and predictions/).
    by_batch = (
        "  - {model: efficientnetb0, learning_rate: 1.0e-5, batch_size: 16}\n"
        "  - {model: efficientnetb0, learning_rate: 1.0e-5, batch_size: 32}\n"
    )
    with pytest.raises(ValueError, match="Duplicate plan entries"):
        _jobs(task, tmp_path, _write_plan(tmp_path, by_batch))
    dropout = "  - {model: efficientnetb0, training: {dropout_rate: 0.3}}\n"
    with pytest.raises(Exception, match="training.dropout_rate"):  # model-only key
        _jobs(task, tmp_path, _write_plan(tmp_path, dropout))


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize(
    ("entry", "key"),
    [
        ("head_type: mlp1", "head_type"),
        ("num_classes: 3", "num_classes"),
        ("random_seed: 7", "random_seed"),
        ("subset_ratio: 0.2", "subset_ratio"),
        ("training: {random_seed: 7}", "training.random_seed"),
        ("paths: {results: x}", "paths"),
    ],
)
def test_runner_rejects_loop_controlled_keys_in_run_entries(task, entry, key, tmp_path):
    plan_path = _write_plan(tmp_path, f"  - {{model: efficientnetb0, {entry}}}\n")
    with pytest.raises(ValueError, match=rf"sets \['{re.escape(key)}'\]"):
        _jobs(task, tmp_path, plan_path)


# ---------------------------------------------------------------------------
# Runners: where results go
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task", TASKS)
def test_runner_dataset_other_than_the_default_is_a_partial_run(task, tmp_path):
    runner = _runner(task)
    paths = Config().paths
    manifest = runner._repo_path(getattr(paths, f"{task}_splits_dir") / "dataset_manifest.csv")
    data_dir = runner._repo_path(getattr(paths, f"{task}_processed_dir"))
    assert runner._dataset_reasons(manifest, data_dir) == []

    other = runner._dataset_reasons(tmp_path / "m.csv", tmp_path / "frames")
    assert len(other) == 2 and "manifest" in other[0] and "data dir" in other[1]
    assert str(tmp_path) not in " ".join(other)  # reasons end up in experiment_meta.json


@pytest.mark.parametrize("task", TASKS)
def test_runner_other_manifest_goes_to_scratch(task, tmp_path, capsys):
    runner = _runner(task)
    rows = make_rows(12, 4 if task == "mes" else 2, seed=0)
    for row in rows:
        row["split"] = ("train", "val", "test")[int(row["patient_id"][1:]) % 3]
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)

    args = runner.build_parser().parse_args(["--manifest", str(manifest), "--dry-run"])
    assert runner.main(args) == 0
    out = capsys.readouterr().out
    assert f"Results dir    : results/{task}/_scratch/" in out
    assert "Partial run    : manifest <external>/manifest.csv is not the default" in out


def _row(task: str, run_name: str, seed: int, stop_reason: str = "max_epochs") -> dict:
    row = {
        "model": "efficientnetb0",
        "freeze": "finetune",
        "head_type": "linear",
        "learning_rate": 2e-5,
        "subset_ratio": 0.5,
        "pct_data": 50,
        "seed": seed,
        "n_train": 100,
        "f1": 0.5 + seed / 100,
        "auroc": 0.7 + seed / 100,
        "val_auroc": 0.6 + seed / 100,
        "realized_fraction": 0.5,
        "stop_reason": stop_reason,
        "run_name": run_name,
    }
    if task == "mes":
        row.update({"macro_f1": 0.4, "clin_sensitivity": 0.8, "clin_specificity": 0.7})
    return row


@pytest.mark.parametrize("task", TASKS)
def test_runner_rows_follow_the_plan_order(task):
    runner = _runner(task)
    order = {"a": 0, "b": 1, "c": 2}
    rows = [_row(task, name, i) for i, name in enumerate(["c", "x", "a", "b"])]
    assert runner._rows_frame(rows, order)["run_name"].tolist() == ["a", "b", "c", "x"]
    assert runner._rows_frame([], order).empty


def _committed_tables(results_dir: Path) -> dict[str, str]:
    results_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "results_per_seed.csv": "old per-seed\n",
        "results.csv": "old aggregate\n",
        "experiment_meta.json": '{"status": "completed"}',
    }
    for name, text in tables.items():
        (results_dir / name).write_text(text, encoding="utf-8")
    return tables


@pytest.mark.parametrize("task", TASKS)
def test_runner_incomplete_run_leaves_the_tables_untouched(task, tmp_path):
    runner = _runner(task)
    results_dir = tmp_path / "results"
    tables = _committed_tables(results_dir)
    jobs = [{"run_name": name} for name in ("a", "b", "c")]
    local = f"FileNotFoundError: {REPO_ROOT / 'pretrained' / 'w.pth'} in {Path.home()}"
    failures = [{"run_name": "a", "error": local, "traceback": local}]
    rows = [_row(task, "c", 2), _row(task, "b", 1, stop_reason="interrupted")]

    code = runner._finalize(
        rows, failures, jobs, True, results_dir=results_dir, meta={"status": "running"}
    )
    assert code == 1  # a failed run outranks the interruption
    for name, text in tables.items():
        assert (results_dir / name).read_text(encoding="utf-8") == text
    assert pd.read_csv(results_dir / runner.PARTIAL_CSV)["run_name"].tolist() == ["b", "c"]
    meta = json.loads((results_dir / runner.PARTIAL_META).read_text(encoding="utf-8"))
    assert meta["status"] == "interrupted"
    assert (meta["n_runs"], meta["n_failed"], meta["n_interrupted"]) == (1, 1, 1)
    saved = json.loads((results_dir / "failures.json").read_text(encoding="utf-8"))
    assert saved[0]["run_name"] == "a"
    for prefix in (str(REPO_ROOT), str(Path.home())):
        assert prefix not in saved[0]["error"] and prefix not in saved[0]["traceback"]

    # Interrupted without failures -> 130; nothing planned ran -> 1. Tables stay untouched.
    finished_c = [_row(task, "c", 2)]
    assert runner._finalize(finished_c, [], jobs, True, results_dir=results_dir, meta={}) == 130
    assert runner._finalize([], [], jobs, False, results_dir=results_dir, meta={}) == 1
    for name, text in tables.items():
        assert (results_dir / name).read_text(encoding="utf-8") == text


@pytest.fixture
def agg_backend():
    """Headless figures. The runners set MPLBACKEND=Agg on import, which is too late
    when another test module imported pyplot first (TkAgg on Windows)."""
    import matplotlib.pyplot as plt

    plt.switch_backend("Agg")


@pytest.mark.parametrize("task", TASKS)
def test_runner_complete_run_replaces_the_tables(task, tmp_path, agg_backend):
    runner = _runner(task)
    results_dir = tmp_path / "results"
    _committed_tables(results_dir)
    (results_dir / "failures.json").write_text("[]", encoding="utf-8")  # stale
    (results_dir / runner.PARTIAL_CSV).write_text("run_name\na\n", encoding="utf-8")
    jobs = [{"run_name": name} for name in ("a", "b")]
    rows = [_row(task, "b", 1), _row(task, "a", 2)]

    assert runner._finalize(rows, [], jobs, False, results_dir=results_dir, meta={}) == 0
    per_seed = pd.read_csv(results_dir / "results_per_seed.csv")
    assert per_seed["run_name"].tolist() == ["a", "b"]  # plan order, not finishing order
    agg = pd.read_csv(results_dir / "results.csv")
    assert agg["n_seeds"].tolist() == [2] and agg["best_seed"].tolist() == [2]
    meta = json.loads((results_dir / "experiment_meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "completed" and meta["n_runs"] == 2
    for leftover in ("failures.json", runner.PARTIAL_CSV, runner.PARTIAL_META):
        assert not (results_dir / leftover).exists()


@pytest.mark.parametrize("task", TASKS)
def test_runner_resume_keeps_only_matching_finished_rows(task, tmp_path):
    runner = _runner(task)
    models = "  - {model: efficientnetb0, freeze_layers: 0, learning_rate: 2.0e-5}\n"
    jobs = _jobs(task, tmp_path, _write_plan(tmp_path, models, seeds="[1, 2, 3]"))
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    assert runner._load_resume(results_dir, jobs, None) == []  # no partial file

    def rec(job: dict, **changes) -> dict:
        base = {**runner._job_settings(job), "run_name": job["run_name"]}
        return {**base, "stop_reason": "max_epochs", **changes}

    records = [
        rec(jobs[0]),
        rec(jobs[1], stop_reason="interrupted"),  # dropped: rerun
        rec(jobs[2], run_name="not_in_plan"),  # dropped: not planned
    ]
    pd.DataFrame(records).to_csv(results_dir / runner.PARTIAL_CSV, index=False)
    kept = runner._load_resume(results_dir, jobs, None)
    assert [r["run_name"] for r in kept] == [jobs[0]["run_name"]]

    pd.DataFrame([rec(jobs[0], epochs=1)]).to_csv(results_dir / runner.PARTIAL_CSV, index=False)
    with pytest.raises(ValueError, match="other settings"):
        runner._load_resume(results_dir, jobs, None)


@pytest.mark.parametrize("task", TASKS)
def test_runner_failure_record_has_no_local_paths(task, tmp_path):
    runner = _runner(task)
    job = _jobs(task, tmp_path, _write_plan(tmp_path, "  - {model: efficientnetb0}\n"))[0]
    exc = FileNotFoundError(f"GastroNet weights not found: {REPO_ROOT / 'pretrained' / 'w.pth'}")
    tb = f'File "{Path.home() / "x.py"}", line 1\nFile "{Path(sys.prefix) / "torch.py"}"'
    record = runner._failure(job, exc, tb=tb)
    assert record["run_name"] == job["run_name"] and record["seed"] == 1
    assert record["error"].replace("\\", "/") == (
        "FileNotFoundError: GastroNet weights not found: ./pretrained/w.pth"
    )
    for prefix in (str(REPO_ROOT), str(Path.home()), sys.prefix):
        assert prefix not in record["traceback"]


@pytest.mark.parametrize("task", TASKS)
def test_runner_isolated_runs_record_rows_failures_and_interrupts(task, tmp_path, monkeypatch):
    """--isolate-runs: each job's outcome (from its own process) is recorded as in-process."""
    runner = _runner(task)
    jobs = [{"run_name": name} for name in ("a", "b", "c", "d")]
    order = runner._plan_order(jobs)
    failure = {"run_name": "b", "error": "RuntimeError: boom", "traceback": "tb"}
    interrupted_row = _row(task, "c", 2, stop_reason="interrupted")
    outcomes = {
        "a": {"status": "ok", "result": _row(task, "a", 1)},
        "b": {"status": "failed", "failure": failure},
        "c": {"status": "interrupted", "result": interrupted_row},
    }
    started = []

    def fake_isolated(job):
        started.append(job["run_name"])
        return outcomes[job["run_name"]]

    monkeypatch.setattr(runner, "_run_isolated", fake_isolated)
    rows, failures = [], []
    assert runner._run_sequential(jobs, tmp_path, rows, failures, order, isolate=True)
    assert started == ["a", "b", "c"]  # the interrupted run stops the plan
    assert [r["run_name"] for r in rows] == ["a", "c"] and failures == [failure]
    assert pd.read_csv(tmp_path / runner.PARTIAL_CSV)["run_name"].tolist() == ["a", "c"]

    # Ctrl+C: _run_isolated kills the child and re-raises; no row, the plan stops.
    def interrupted(job):
        raise KeyboardInterrupt

    monkeypatch.setattr(runner, "_run_isolated", interrupted)
    rows = []
    assert runner._run_sequential(jobs, tmp_path, rows, [], order, isolate=True)
    assert rows == []


@pytest.mark.parametrize("task", TASKS)
def test_runner_isolated_run_failure_comes_back_from_its_process(task, tmp_path):
    """A job that fails in its spawned process becomes a failure record with its traceback."""
    runner = _runner(task)
    job = _jobs(task, tmp_path, _write_plan(tmp_path, "  - {model: efficientnetb0}\n"))[0]
    out = runner._run_isolated(job)  # the job's manifest.csv does not exist
    assert out["status"] == "failed"
    assert out["failure"]["run_name"] == job["run_name"]
    assert out["failure"]["error"].startswith("FileNotFoundError")
    assert "_execute_run" in out["failure"]["traceback"]
