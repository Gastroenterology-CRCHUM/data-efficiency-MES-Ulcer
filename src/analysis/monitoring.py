"""Progress and settings of a data-efficiency plan, read from the runner outputs (notebook 02)."""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.config import MODEL_REGISTRY, Config, get_img_size, legacy_dict_to_config
from src.utils.results import REPO_ROOT, redact_local_paths

KEYS = ["model", "head_type", "pct", "seed"]
_STAMP = re.compile(r"\d{8}_\d{6}")
_EXPECTED: dict = {}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


def read_plan(path: Path | None, meta: dict) -> dict:
    """Run entries, fractions (%), seeds and heads of the plan YAML (of experiment_meta without one)."""
    plan = (yaml.safe_load(path.read_text(encoding="utf-8")) if path else meta) or {}
    return {
        "specs": [m if isinstance(m, dict) else {"model": m} for m in plan.get("models") or []],
        "pcts": [round(float(r) * 100) for r in plan.get("subset_ratios") or []],
        "seeds": [int(s) for s in plan.get("seeds") or []],
        "heads": list(plan.get("head_types") or ["linear"]),
    }


def run_info(run_dir: Path | None) -> dict:
    """run_config.json, start, last write and duration (start to last write) of a run folder."""
    if run_dir is None or not run_dir.is_dir():
        return {"cfg": {}, "start": None, "end": None, "has_history": False, "duration_min": np.nan}
    cfg = read_json(run_dir / "run_config.json")
    if cfg.get("created"):
        start = datetime.fromisoformat(cfg["created"])
    else:
        m = _STAMP.search(run_dir.name)
        start = datetime.strptime(m.group(0), "%Y%m%d_%H%M%S") if m else None
    files = [f for f in run_dir.iterdir() if f.is_file()]
    end = datetime.fromtimestamp(max(f.stat().st_mtime for f in files)) if files else None
    done = (run_dir / "history.json").is_file()
    duration = (end - start).total_seconds() / 60 if done and start and end else np.nan
    return {"cfg": cfg, "start": start, "end": end, "has_history": done, "duration_min": duration}


def resolve_run_dir(ckpt, models_dir: Path | None, root: Path = REPO_ROOT) -> Path | None:
    """Run folder of a per-seed row (checkpoint_dir is repo-relative, or '<models-dir>/...')."""
    s = str(ckpt).replace("\\", "/")
    if s.startswith("<models-dir>/") and models_dir is not None:
        return models_dir / s.split("/", 1)[1]
    return None if s in ("", "nan", "None") or s.startswith("<") else root / s


def load_task(
    spec: dict, *, now: datetime | None = None, active_window_min: float = 120, root: Path = REPO_ROOT
) -> dict:
    """Progress of one plan: finished rows, planned grid, issues and a summary.

    ``spec``: ``results`` folder, ``plan`` YAML (None: the planned runs of experiment_meta) and
    default ``models`` folder. A running plan without a write for ``active_window_min`` minutes
    is reported as stalled.
    """
    now = now or datetime.now()
    res = spec["results"]
    f = {k: res / k for k in ("experiment_meta.json", "experiment_meta.partial.json",
                              "results_per_seed.csv", "results_per_seed.partial.csv")}
    partial = (max(_mtime(f["experiment_meta.partial.json"]), _mtime(f["results_per_seed.partial.csv"]))
               > max(_mtime(f["experiment_meta.json"]), _mtime(f["results_per_seed.csv"])))
    sfx = ".partial" if partial else ""
    meta, csv = read_json(f[f"experiment_meta{sfx}.json"]), f[f"results_per_seed{sfx}.csv"]
    rows = pd.read_csv(csv).rename(columns={"pct_data": "pct"}) if csv.is_file() else pd.DataFrame(columns=KEYS)
    for col in ("head_type", "stop_reason", "checkpoint_dir", "best_epoch", "run_name"):
        if col not in rows:
            rows[col] = np.nan
    rows["head_type"] = rows["head_type"].fillna("linear")
    md = str(meta.get("models_dir") or "")
    models_dir = root / md if md and not md.startswith("<") else spec.get("models")
    rows["run_dir"] = [resolve_run_dir(c, models_dir, root) for c in rows["checkpoint_dir"]]
    info = [run_info(d) for d in rows["run_dir"]]
    rows = rows.join(pd.DataFrame(info, index=rows.index, columns=list(run_info(None))))
    finished = (rows[rows["stop_reason"].ne("interrupted")].astype({"pct": int, "seed": int})
                .drop_duplicates(KEYS, keep="last"))
    plan = read_plan(spec.get("plan"), meta)

    # Planned grid vs finished runs
    grid = pd.MultiIndex.from_product([[s["model"] for s in plan["specs"]], plan["heads"], plan["pcts"],
                                       plan["seeds"]], names=KEYS).to_frame(index=False)
    status = grid.merge(finished[KEYS].assign(done=True), on=KEYS, how="left")
    status["done"] = status["done"].eq(True)
    n_extra = len(finished) - int(status["done"].sum())

    # Run folders without a result row: in progress (no history.json yet) or aborted
    listed = {d.resolve() for d in rows["run_dir"].dropna()}
    run_dirs = sorted(d for d in models_dir.glob("*/*/*/*") if d.is_dir()) if models_dir and models_dir.is_dir() else []
    others = [{"run_dir": d, **run_info(d)} for d in run_dirs if d.resolve() not in listed]
    created = datetime.fromisoformat(meta["created"]) if meta.get("created") else datetime.min
    running = meta.get("status") == "running"
    in_progress = [o for o in others if not o["has_history"] and running and o["start"] and o["start"] >= created
                   and "stop_reason" not in o["cfg"]]
    aborted = [o for o in others if not o["has_history"] and o not in in_progress]

    writes = [e for e in [*rows["end"], *(o["end"] for o in others)] if e is not None and not pd.isna(e)]
    writes += [datetime.fromtimestamp(_mtime(p)) for p in f.values() if p.exists()]
    last = max(writes, default=None)
    idle_min = (now - last).total_seconds() / 60 if last else np.inf

    # ETA from the durations of the finished runs
    durations = finished.dropna(subset=["duration_min"])

    def expected(model, pct) -> float:
        for sel in (durations[(durations["model"] == model) & (durations["pct"] == pct)],
                    durations[durations["pct"] == pct], durations):
            if len(sel):
                return float(sel["duration_min"].median())
        return np.nan

    missing = status[~status["done"]]
    remaining = sum(expected(m, p) for m, p in zip(missing["model"], missing["pct"], strict=True))
    elapsed = sum(min((now - o["start"]).total_seconds() / 60, expected(o["cfg"].get("model"), o["cfg"].get("pct_data")))
                  for o in in_progress)
    eta = max(remaining - elapsed, 0) / max(1, len(in_progress))

    # Failed / interrupted / in progress / aborted
    failures = read_json(res / "failures.json") or [{"run_name": r, "error": "-"} for r in meta.get("failed_runs", [])]
    done_names = set(finished["run_name"].dropna())
    issues = [{"kind": "failed", "run": x.get("run_name"),
               "detail": (redact_local_paths(str(x.get("error"))) or "-").splitlines()[0][:160]}
              for x in failures if x.get("run_name") not in done_names]
    issues += [{"kind": "interrupted", "run": r, "detail": "stop_reason = interrupted (rerun with --resume)"}
               for r in rows.loc[rows["stop_reason"].eq("interrupted"), "run_name"]]
    issues += [{"kind": "in progress", "run": o["run_dir"].name,
                "detail": f"started {o['start']:%a %H:%M}, {(now - o['start']).total_seconds() / 60:.0f} min "
                          f"(expected ≈ {expected(o['cfg'].get('model'), o['cfg'].get('pct_data')):.0f})"}
               for o in in_progress]
    issues += [{"kind": "aborted", "run": o["run_dir"].name, "detail": "run folder without history.json"}
               for o in aborted]

    state = meta.get("status", "no results")
    if running and idle_min > active_window_min:
        state = f"running? (no write for {idle_min / 60:.1f} h)"
    summary = {
        "plan": spec["plan"].name if spec.get("plan") else "experiment_meta",
        "results file": csv.name if csv.is_file() else "-",
        "status": state,
        "planned": len(status),
        "done": int(status["done"].sum()),
        "missing": len(missing),
        "rows outside the plan": n_extra,
        "failed": sum(i["kind"] == "failed" for i in issues),
        "interrupted": sum(i["kind"] == "interrupted" for i in issues),
        "in progress": len(in_progress),
        "aborted run folders": len(aborted),
        "last write": f"{last:%Y-%m-%d %H:%M}" if last else "-",
        "median run (min)": round(durations["duration_min"].median(), 1) if len(durations) else np.nan,
        "remaining (h, 1 job)": round(remaining / 60, 1),
        "ETA": (f"{eta / 60:.1f} h (≈ {now + timedelta(minutes=eta):%a %d %b %H:%M})"
                if running and idle_min <= active_window_min and len(missing) and np.isfinite(eta) else "-"),
    }
    return {"summary": summary, "status": status, "rows": finished, "plan": plan, "issues": issues}


def expected_protocol(model: str, plan: dict, base: Config) -> dict:
    """Current protocol of a model: the ``base`` config, overridden by its plan entry."""
    key = (model, json.dumps(plan["specs"], sort_keys=True), id(base))
    if key not in _EXPECTED:
        spec = next((s for s in plan["specs"] if s.get("model") == model), None)
        known = model in MODEL_REGISTRY
        cfg = legacy_dict_to_config(spec, base=base, strict=True) if spec and known else base
        t = cfg.training
        vit = known and MODEL_REGISTRY[model].is_vit  # ViT-only settings are off for the CNNs
        _EXPECTED[key] = {
            "img_size": get_img_size(model) if known else np.nan,
            "optimizer": t.optimizer, "learning_rate": t.learning_rate, "weight_decay": t.weight_decay,
            "warmup_epochs": t.warmup_epochs, "min_lr": t.min_lr, "epochs": t.epochs,
            "es_patience": t.es_patience, "batch_size": t.batch_size, "grad_clip_norm": t.grad_clip_norm,
            "use_amp": t.use_amp, "label_smoothing": t.label_smoothing, "freeze_layers": cfg.model.freeze_layers,
            "equalize": t.equalize, "deterministic": t.deterministic,
            "layer_decay": t.layer_decay if vit else 1.0, "drop_path_rate": cfg.model.drop_path_rate if vit else 0.0,
        }
    return _EXPECTED[key]
