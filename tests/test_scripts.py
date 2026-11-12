"""Command-line entry points: run_all forwarding and runner plan handling (no training).

The runners are imported inside the tests: importing them sets MPLBACKEND and
pulls in torch, which collection should not do.
"""

from __future__ import annotations

import concurrent.futures
import importlib
import multiprocessing
import signal
import time
from pathlib import Path

import pandas as pd
import pytest

from src.config import Config
from src.utils import ignore_sigint, terminate_pool
from tests.conftest import make_rows

REPO_ROOT = Path(__file__).resolve().parents[1]
PLANS = REPO_ROOT / "configs" / "experiments"


# ---------------------------------------------------------------------------
# run_all
# ---------------------------------------------------------------------------


@pytest.fixture
def run_all():
    return importlib.import_module("scripts.run_all")


def test_run_all_forwards_shared_and_task_flags(run_all, tmp_path):
    args = run_all.build_parser().parse_args(
        [
            "--subset-ratios",
            "0.1",
            "0.25",
            "--seeds",
            "42",
            "84",
            "--epochs",
            "2",
            "--no-deterministic",
            "--dry-run",
            "--mes-manifest",
            str(tmp_path / "m.csv"),
            "--ulcer-results-dir",
            str(tmp_path / "res"),
        ]
    )
    mes = run_all.forwarded_args(args, "mes")
    ulcer = run_all.forwarded_args(args, "ulcer")
    shared = [
        "--subset-ratios",
        "0.1",
        "0.25",
        "--seeds",
        "42",
        "84",
        "--epochs",
        "2",
        "--no-deterministic",
        "--dry-run",
    ]
    assert mes == [*shared, "--manifest", str((tmp_path / "m.csv").resolve())]
    assert ulcer == [*shared, "--results-dir", str((tmp_path / "res").resolve())]


@pytest.mark.parametrize("flag", ["--manifest", "--data-dir", "--results-dir", "--models-dir"])
def test_run_all_rejects_task_specific_flags(run_all, flag, capsys):
    with pytest.raises(SystemExit):
        run_all.build_parser().parse_args([flag, "x"])
    assert f"--mes-{flag[2:]}" in capsys.readouterr().err


def test_run_all_rejects_abbreviations(run_all):
    with pytest.raises(SystemExit):
        run_all.build_parser().parse_args(["--subset", "0.1"])


def test_run_all_forwards_default_limuc_configuration(run_all):
    args = run_all.build_parser().parse_args(["--limuc"])
    forwarded = run_all.forwarded_args(args, "limuc")
    assert forwarded == [
        "--plan",
        str((REPO_ROOT / "configs/experiments/data_efficiency_limuc.yaml").resolve()),
        "--manifest",
        str((REPO_ROOT / "data_LIMUC/mes/splits/dataset_manifest.csv").resolve()),
        "--data-dir",
        str((REPO_ROOT / "data_LIMUC/mes/processed").resolve()),
        "--results-dir",
        str((REPO_ROOT / "results/mes/data_efficiency_LIMUC").resolve()),
        "--models-dir",
        str((REPO_ROOT / "output/limuc/models").resolve()),
    ]


def test_run_all_limuc_is_opt_in(run_all):
    args = run_all.build_parser().parse_args([])
    assert args.limuc is False


def test_run_all_vitb_runs_use_their_own_plans_and_folders(run_all, tmp_path):
    args = run_all.build_parser().parse_args(
        ["--limuc", "--vitb-only", "--mes-manifest", str(tmp_path / "m.csv"), "--mes-results-dir", str(tmp_path)]
    )
    resolved = lambda p: str((REPO_ROOT / p).resolve())  # noqa: E731
    assert run_all.forwarded_args(args, "mes_vitb") == [  # the main task's data, never its results
        "--plan", resolved("configs/experiments/data_efficiency_vitb.yaml"),
        "--manifest", str((tmp_path / "m.csv").resolve()),
        "--results-dir", resolved("results/mes/data_efficiency_vitb"),
        "--models-dir", resolved("output/mes/models_vitb"),
    ]
    assert run_all.forwarded_args(args, "limuc_vitb") == [
        "--plan", resolved("configs/experiments/data_efficiency_limuc_vitb.yaml"),
        "--manifest", resolved("data_LIMUC/mes/splits/dataset_manifest.csv"),
        "--data-dir", resolved("data_LIMUC/mes/processed"),
        "--results-dir", resolved("results/mes/data_efficiency_LIMUC_vitb"),
        "--models-dir", resolved("output/limuc/models_vitb"),
    ]


class _DonePopen:
    started: list = []

    def __init__(self, cmd, cwd=None):
        _DonePopen.started.append(cmd)

    def wait(self):
        return 0


@pytest.mark.parametrize(("flags", "tasks"), [
    ([], ["ulcer", "mes", "limuc"]),
    (["--vitb"], ["ulcer", "mes", "limuc", "ulcer_vitb", "mes_vitb", "limuc_vitb"]),
    (["--vitb-only"], ["ulcer_vitb", "mes_vitb", "limuc_vitb"]),
])
def test_run_all_vitb_is_opt_in(run_all, monkeypatch, flags, tasks):
    _DonePopen.started = []
    monkeypatch.setattr(run_all.subprocess, "Popen", _DonePopen)
    assert run_all.main(["--limuc", "--dry-run", *flags]) == 0
    plans = [cmd[cmd.index("--plan") + 1] if "--plan" in cmd else None for cmd in _DonePopen.started]
    assert ["_vitb" if p and p.endswith("_vitb.yaml") else "" for p in plans] == [
        "_vitb" if t.endswith("_vitb") else "" for t in tasks
    ]


def test_run_all_rejects_plan_with_vitb(run_all, capsys):
    with pytest.raises(SystemExit):
        run_all.main(["--vitb", "--plan", "configs/experiments/data_efficiency.yaml"])
    assert "--vitb-only" in capsys.readouterr().err


def test_run_all_forwards_overwrite_but_not_with_resume(run_all, capsys):
    args = run_all.build_parser().parse_args(["--overwrite", "--limuc"])
    assert run_all.forwarded_args(args, "ulcer") == ["--overwrite"]
    assert "--overwrite" in run_all.forwarded_args(args, "limuc")
    with pytest.raises(SystemExit):
        run_all.build_parser().parse_args(["--overwrite", "--resume"])
    assert "not allowed with" in capsys.readouterr().err


def test_run_all_smoke_only_overrides_full_plan(run_all):
    args = run_all.build_parser().parse_args(["--limuc", "--smoke-only", "--epochs", "100"])
    assert run_all.forwarded_args(args, "ulcer") == ["--epochs", "5", "--max-runs", "1"]
    assert run_all.forwarded_args(args, "limuc") == [
        "--plan",
        str((REPO_ROOT / "configs/experiments/data_efficiency_limuc.yaml").resolve()),
        "--epochs",
        "5",
        "--max-runs",
        "1",
        "--manifest",
        str((REPO_ROOT / "data_LIMUC/mes/splits/dataset_manifest.csv").resolve()),
        "--data-dir",
        str((REPO_ROOT / "data_LIMUC/mes/processed").resolve()),
        "--results-dir",
        str((REPO_ROOT / "results/mes/data_efficiency_LIMUC").resolve()),
        "--models-dir",
        str((REPO_ROOT / "output/limuc/models").resolve()),
    ]


class _FakePopen:
    """Popen whose first wait() is interrupted by Ctrl+C."""

    started: list = []

    def __init__(self, cmd, cwd=None):
        self.cmd, self.waits, self.killed = cmd, 0, False
        _FakePopen.started.append(cmd)

    def wait(self):
        self.waits += 1
        if self.waits == 1:
            raise KeyboardInterrupt
        return 130

    def kill(self):
        self.killed = True

    def send_signal(self, sig):
        pass


def test_run_all_waits_for_the_runner_and_stops_on_ctrl_c(run_all, monkeypatch):
    _FakePopen.started = []
    monkeypatch.setattr(run_all.subprocess, "Popen", _FakePopen)
    assert run_all.main(["--tasks", "ulcer", "mes", "--dry-run"]) == 130
    assert len(_FakePopen.started) == 1  # MES never starts


# ---------------------------------------------------------------------------
# Stopping worker pools
# ---------------------------------------------------------------------------


def _pool() -> concurrent.futures.ProcessPoolExecutor:
    ctx = multiprocessing.get_context("spawn")
    return concurrent.futures.ProcessPoolExecutor(max_workers=1, mp_context=ctx, initializer=ignore_sigint)


def test_pool_workers_ignore_ctrl_c():
    with _pool() as ex:
        assert ex.submit(signal.getsignal, signal.SIGINT).result(timeout=60) == signal.SIG_IGN


def test_terminate_pool_stops_a_running_job():
    ex = _pool()
    fut = ex.submit(time.sleep, 60)
    while not fut.running():
        time.sleep(0.05)
    workers = list(ex._processes.values())
    start = time.monotonic()
    terminate_pool(ex)
    assert time.monotonic() - start < 10
    assert not any(p.is_alive() for p in workers)


# ---------------------------------------------------------------------------
# Runners (plan parsing and dry runs)
# ---------------------------------------------------------------------------


def _runner(task: str):
    return importlib.import_module(f"scripts.{task}.run_data_efficiency")


def _manifest(tmp_path: Path, n_classes: int) -> Path:
    rows = make_rows(20, n_classes, seed=0)
    for row in rows:
        row["split"] = ("train", "val", "test")[int(row["patient_id"][1:]) % 3]
    path = tmp_path / f"manifest_{n_classes}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.mark.parametrize("task", ["mes", "ulcer"])
def test_default_plan_is_a_full_run(task):
    runner = _runner(task)
    plan = runner._get_plan(runner.build_parser().parse_args([]), Config())
    assert plan["partial_reasons"] == []
    assert plan["path"] == runner.DEFAULT_PLAN
    assert plan["seeds"] == [42, 84, 128, 256, 364, 420, 512, 1024, 2048, 4096]


@pytest.mark.parametrize("task", ["mes", "ulcer"])
def test_default_config_is_base_yaml(task):
    runner = _runner(task)
    assert runner.DEFAULT_CONFIG == REPO_ROOT / "configs" / "base.yaml"
    assert runner.DEFAULT_CONFIG.is_file()


@pytest.mark.parametrize("task", ["mes", "ulcer"])
def test_cli_filters_mark_a_partial_run(task):
    runner = _runner(task)
    argv = ["--seeds", "42", "--subset-ratios", "0.1", "--epochs", "1", "--max-runs", "1"]
    plan = runner._get_plan(runner.build_parser().parse_args(argv), Config())
    reasons = " ".join(plan["partial_reasons"])
    for flag in ("--seeds", "--subset-ratios", "--epochs", "--max-runs"):
        assert flag in reasons
    assert plan["seeds"] == [42] and plan["subset_ratios"] == [0.1]


@pytest.mark.parametrize("task", ["mes", "ulcer"])
def test_plan_errors(task, tmp_path):
    runner = _runner(task)
    parse = runner.build_parser().parse_args
    with pytest.raises(FileNotFoundError):
        runner._get_plan(parse(["--plan", str(tmp_path / "missing.yaml")]), Config())
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        runner._get_plan(parse(["--subset-ratios", "0.5", "1.5"]), Config())
    with pytest.raises(ValueError, match="duplicate percent"):
        runner._get_plan(parse(["--subset-ratios", "0.101", "0.104"]), Config())
    with pytest.raises(ValueError, match="distinct"):
        runner._get_plan(parse(["--seeds", "1", "1"]), Config())
    with pytest.raises(ValueError, match="not found in plan"):
        runner._get_plan(parse(["--model", "not_a_model"]), Config())
    bad = tmp_path / "bad.yaml"
    bad.write_text("models: [{model: efficientnetb0}]\nepochs: 3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown plan key"):
        runner._get_plan(parse(["--plan", str(bad)]), Config())


@pytest.mark.parametrize(
    ("task", "n_classes", "plan", "n_runs"),
    [
        ("mes", 4, "data_efficiency.yaml", 9 * 5 * 10),
        ("mes", 4, "data_efficiency_limuc.yaml", 9 * 7 * 10),
        ("ulcer", 2, "data_efficiency.yaml", 9 * 5 * 10),
        ("mes", 4, "data_efficiency_vitb.yaml", 3 * 5 * 10),
        ("mes", 4, "data_efficiency_limuc_vitb.yaml", 3 * 7 * 10),
        ("ulcer", 2, "data_efficiency_vitb.yaml", 3 * 5 * 10),
    ],
)
def test_dry_run_builds_every_job_and_writes_nothing(
    task, n_classes, plan, n_runs, tmp_path, capsys
):
    runner = _runner(task)
    results, models = tmp_path / "results", tmp_path / "models"
    args = runner.build_parser().parse_args(
        [
            "--manifest",
            str(_manifest(tmp_path, n_classes)),
            "--data-dir",
            str(tmp_path / "frames"),
            "--plan",
            str(PLANS / plan),
            "--results-dir",
            str(results),
            "--models-dir",
            str(models),
            "--dry-run",
        ]
    )
    assert runner.main(args) == 0
    out = capsys.readouterr().out
    assert f"Planned runs   : {n_runs}" in out
    assert not results.exists() and not models.exists()


@pytest.mark.parametrize(("task", "n_classes"), [("mes", 4), ("ulcer", 2)])
def test_overwrite_dry_run_lists_and_keeps_the_run_folders(task, n_classes, tmp_path, capsys):
    runner = _runner(task)
    results, models = tmp_path / "results", tmp_path / "models"
    (results / "predictions").mkdir(parents=True)
    models.mkdir()
    args = runner.build_parser().parse_args(
        [
            "--manifest",
            str(_manifest(tmp_path, n_classes)),
            "--data-dir",
            str(tmp_path / "frames"),
            "--results-dir",
            str(results),
            "--models-dir",
            str(models),
            "--max-runs",
            "1",
            "--overwrite",
            "--dry-run",
        ]
    )
    assert runner.main(args) == 0
    out = capsys.readouterr().out
    assert "Overwrite      : <external>/results, <external>/models" in out
    assert results.is_dir() and models.is_dir()
    with pytest.raises(SystemExit):
        runner.build_parser().parse_args(["--overwrite", "--resume"])
