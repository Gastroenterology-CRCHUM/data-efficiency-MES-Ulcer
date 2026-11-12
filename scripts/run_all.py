"""Run the ulcer then MES data-efficiency experiments, optionally LIMUC and the ViT-B runs.

    python -m scripts.run_all --limuc --dry-run
    python -m scripts.run_all --limuc --parallel-jobs 8
    python -m scripts.run_all --tasks mes --resume
    python -m scripts.run_all --limuc --vitb-only --parallel-jobs 3
    python -m scripts.run_all --limuc --smoke-only

``--tasks`` selects the tasks (default: ulcer then MES); ``--limuc`` adds the LIMUC
replication; ``--vitb`` adds the exploratory ViT-B runs (own plans and folders,
``VITB_DEFAULTS``) and ``--vitb-only`` runs only those. ``--smoke-only`` runs one five-epoch
run per task instead of the plan.

Flags shared by both runners are forwarded as they are (``--plan --config --model
--subset-ratios --seeds --epochs --batch-size --deterministic --max-runs --parallel-jobs
--resume --overwrite --dry-run``). Paths are task-specific:
``--{ulcer,mes,limuc}-{manifest,data-dir,results-dir,models-dir}``.

The first task that exits with a non-zero code stops the sequence. Ctrl+C stops the running
task and starts no other; continue with ``--resume`` and ``--tasks`` limited to the unfinished
tasks (a finished task has no partial table, so ``--resume`` would retrain it).
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TASKS = ("ulcer", "mes")
LIMUC_DEFAULTS = {
    "plan": "configs/experiments/data_efficiency_limuc.yaml",
    "manifest": "data_LIMUC/mes/splits/dataset_manifest.csv",
    "data-dir": "data_LIMUC/mes/processed",
    "results-dir": "results/mes/data_efficiency_LIMUC",
    "models-dir": "output/limuc/models",
}
VITB_DEFAULTS = {  # exploratory ViT-B runs (--vitb, --vitb-only): own plan, results and models folders
    "ulcer": {"plan": "configs/experiments/data_efficiency_vitb.yaml",
              "results-dir": "results/ulcer/data_efficiency_vitb", "models-dir": "output/ulcer/models_vitb"},
    "mes": {"plan": "configs/experiments/data_efficiency_vitb.yaml",
            "results-dir": "results/mes/data_efficiency_vitb", "models-dir": "output/mes/models_vitb"},
    "limuc": {"plan": "configs/experiments/data_efficiency_limuc_vitb.yaml",
              "results-dir": "results/mes/data_efficiency_LIMUC_vitb", "models-dir": "output/limuc/models_vitb"},
}
AMBIGUOUS_FLAGS = ("--manifest", "--data-dir", "--results-dir", "--models-dir")
TASK_PATH_FLAGS = ("manifest", "data-dir", "results-dir", "models-dir")


class _RejectAmbiguous(argparse.Action):
    """Refuse a task-specific flag that would be given to both tasks."""

    def __call__(self, parser, namespace, values, option_string=None):
        name = option_string.lstrip("-")
        parser.error(
            f"{option_string} is task-specific and cannot be shared by both tasks; "
            f"use --ulcer-{name} and/or --mes-{name}."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the ulcer then the MES data-efficiency experiments.",
        allow_abbrev=False,
    )
    shared = parser.add_argument_group("shared (forwarded to both tasks)")
    shared.add_argument("--plan", default=None, help="Experiment YAML plan")
    shared.add_argument("--config", default=None, help="Base YAML/JSON config (load_config)")
    shared.add_argument("--model", default=None, help="Run a single model from the plan")
    shared.add_argument("--subset-ratios", type=float, nargs="+", default=None)
    shared.add_argument("--seeds", type=int, nargs="+", default=None)
    shared.add_argument("--epochs", type=int, default=None)
    shared.add_argument("--batch-size", type=int, default=None)
    shared.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=None)
    shared.add_argument("--max-runs", type=int, default=None, help="Per task")
    shared.add_argument("--parallel-jobs", type=int, default=None, help="Runs trained at once on the GPU")
    fresh = shared.add_mutually_exclusive_group()
    fresh.add_argument(
        "--resume",
        action="store_true",
        help="Continue each task from its results_per_seed.partial.csv",
    )
    fresh.add_argument(
        "--overwrite",
        action="store_true",
        help="Each task deletes its results and models directories before training",
    )
    shared.add_argument("--dry-run", action="store_true")
    shared.add_argument(
        "--smoke-only",
        action="store_true",
        help="Run only one five-epoch smoke run per selected task",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=TASKS,
        default=list(TASKS),
        help="Tasks to run, in the order given (default: ulcer mes)",
    )
    parser.add_argument(
        "--limuc",
        action="store_true",
        help="Also run the LIMUC MES replication after the selected tasks",
    )
    vitb = parser.add_mutually_exclusive_group()
    vitb.add_argument(
        "--vitb",
        action="store_true",
        help="Then run the exploratory ViT-B plans of the selected tasks (own results folders)",
    )
    vitb.add_argument(
        "--vitb-only", action="store_true", help="Run only the exploratory ViT-B plans of the selected tasks"
    )

    for task in (*TASKS, "limuc"):
        group = parser.add_argument_group(f"{task} only")
        for flag in TASK_PATH_FLAGS:
            group.add_argument(f"--{task}-{flag}", default=None, metavar="PATH")

    for flag in AMBIGUOUS_FLAGS:
        parser.add_argument(flag, action=_RejectAmbiguous, help=argparse.SUPPRESS)
    return parser


def _abs(path: str) -> str:
    return str(Path(path).resolve())


def forwarded_args(args: argparse.Namespace, task: str) -> list[str]:
    """CLI arguments of one task's runner (``<task>_vitb``: the task's ViT-B runs)."""
    out: list[str] = []
    prefix = task.removesuffix("_vitb")
    vitb = VITB_DEFAULTS[prefix] if task != prefix else None
    for flag in ("plan", "config"):
        value = getattr(args, flag)
        if flag == "plan" and value is None:
            value = vitb["plan"] if vitb else LIMUC_DEFAULTS["plan"] if task == "limuc" else None
        if value is not None:
            out += [f"--{flag}", _abs(value)]
    if args.model is not None:
        out += ["--model", args.model]
    if args.smoke_only:
        out += ["--epochs", "5", "--max-runs", "1"]
    if not args.smoke_only and args.subset_ratios:
        out += ["--subset-ratios", *(f"{r:g}" for r in args.subset_ratios)]
    if not args.smoke_only and args.seeds:
        out += ["--seeds", *(str(s) for s in args.seeds)]
    for flag in ("batch_size", "parallel_jobs"):
        value = getattr(args, flag)
        if value is not None:
            out += [f"--{flag.replace('_', '-')}", str(value)]
    if not args.smoke_only:
        for flag in ("epochs", "max_runs"):
            value = getattr(args, flag)
            if value is not None:
                out += [f"--{flag.replace('_', '-')}", str(value)]
    if args.deterministic is not None:
        out.append("--deterministic" if args.deterministic else "--no-deterministic")
    if args.resume:
        out.append("--resume")
    if args.overwrite:
        out.append("--overwrite")
    if args.dry_run:
        out.append("--dry-run")
    for flag in TASK_PATH_FLAGS:
        value = getattr(args, f"{prefix}_{flag.replace('-', '_')}")
        if vitb and flag in vitb:  # results and models: always the ViT-B folders
            out += [f"--{flag}", _abs(vitb[flag])]
        elif value is not None:
            out += [f"--{flag}", _abs(value)]
        elif prefix == "limuc" and flag in LIMUC_DEFAULTS:
            out += [f"--{flag}", _abs(LIMUC_DEFAULTS[flag])]
    return out


def _run_task(cmd: list[str]) -> int:
    """Run one task's runner; on Ctrl+C or SIGTERM, wait for it to stop (twice: kill it)."""
    proc = subprocess.Popen(cmd, cwd=REPO_ROOT)

    def _forward(signum, frame):  # kill <pid of run_all>: stop the runner as Ctrl+C would
        proc.terminate() if os.name == "nt" else proc.send_signal(signal.SIGINT)

    previous = signal.signal(signal.SIGTERM, _forward)
    try:
        try:
            return proc.wait()
        except KeyboardInterrupt:  # the runner got the same Ctrl+C and stops its runs
            print("\n[!] Interrupted - waiting for the runner to stop (Ctrl+C again to kill it).",
                  flush=True)
            try:
                proc.wait()
            except KeyboardInterrupt:
                proc.kill()
                proc.wait()
            return 130
    finally:
        signal.signal(signal.SIGTERM, previous)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.vitb and args.plan:
        parser.error("--plan with --vitb would give the main plan to the ViT-B runs; use --vitb-only")
    tasks = list(dict.fromkeys(args.tasks))
    if args.limuc:
        tasks.append("limuc")
    if args.vitb_only:
        tasks = [f"{t}_vitb" for t in tasks]
    elif args.vitb:
        tasks += [f"{t}_vitb" for t in tasks]

    for task in tasks:
        base = task.removesuffix("_vitb")
        module = f"scripts.{'mes' if base == 'limuc' else base}.run_data_efficiency"
        cmd = [sys.executable, "-m", module, *forwarded_args(args, task)]

        sep = "=" * 64
        print(f"\n{sep}")
        print(f"  TASK : {task.upper()}")
        print(f"  CMD  : python -m {module} {' '.join(cmd[3:])}")
        print(sep, flush=True)

        code = _run_task(cmd)
        if code != 0:
            print(f"\n[!] {task} exited with code {code} - stopping.")
            return code

    print(f"\nCompleted successfully: {', '.join(tasks)}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
