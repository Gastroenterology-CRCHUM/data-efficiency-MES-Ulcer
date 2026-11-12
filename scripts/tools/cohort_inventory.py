"""Cohort inventory: patients, procedures, clips and frames per split and grade.

    python -m scripts.tools.cohort_inventory            # writes results/inventory/

Reads data/{mes,ulcer}/splits/dataset_manifest.csv and, if present,
data/video_data.csv (demographics over patients, gender 1 = female, computed only
on procedures found in both). Output is aggregate only. Exits with status 1 when
a patient spans splits or cohorts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from src.data.manifest_utils import SPLITS, load_manifest, load_video_data

TASKS = {"mes": "UC (MES)", "ulcer": "CD (ulcer)"}


def _procedures(df: pd.DataFrame) -> pd.Series:
    return df["procedure_id"] if "procedure_id" in df.columns else df["patient_id"]


def split_table(df: pd.DataFrame) -> pd.DataFrame:
    """Patients / procedures / clips / frames per split and label."""
    df = df.assign(procedure=_procedures(df))
    rows = []
    for (split, label), g in df.groupby(["split", "label"]):
        rows.append({"split": split, "label": int(label), "patients": g["patient_id"].nunique(),
                     "procedures": g["procedure"].nunique(), "clips": g["video_id"].nunique(),
                     "frames": len(g)})
    for split, g in df.groupby("split"):
        rows.append({"split": split, "label": "all", "patients": g["patient_id"].nunique(),
                     "procedures": g["procedure"].nunique(), "clips": g["video_id"].nunique(),
                     "frames": len(g)})
    order = {s: i for i, s in enumerate(SPLITS)}
    out = pd.DataFrame(rows)
    return out.sort_values(["split", "label"], key=lambda c: c.map(order) if c.name == "split" else c.astype(str))


def _markdown(table: pd.DataFrame) -> str:
    cols = list(table.columns)
    rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    rows += ["| " + " | ".join(str(v) for v in r) + " |" for r in table.itertuples(index=False)]
    return "\n".join(rows)


def demographics(df: pd.DataFrame, video_data: pd.DataFrame | None, dataset: str) -> dict:
    """Age / sex over patients (first procedure) on procedures present in both sources."""
    if video_data is None:
        return {"available": False}
    vd = video_data[video_data["dataset"] == dataset]
    procs = set(_procedures(df))
    both = vd[vd.index.isin(procs)].copy()
    out = {"available": True, "procedures_matched": len(both),
           "procedures_only_in_manifest": len(procs - set(vd.index)),
           "procedures_only_in_video_data": int((~vd.index.isin(procs)).sum())}
    if both.empty:
        return out
    if "date_of_procedure" in both.columns:
        both = both.sort_values("date_of_procedure")
    first = both.groupby("patient_id").first()
    out["patients"] = len(first)
    if "age" in first.columns:
        age = pd.to_numeric(first["age"], errors="coerce").dropna()
        out["age"] = {"mean": round(age.mean(), 1), "sd": round(age.std(), 1),
                      "median": float(age.median()), "min": float(age.min()), "max": float(age.max())}
    if "gender" in first.columns:
        female = (pd.to_numeric(first["gender"], errors="coerce") == 1).sum()
        out["female"] = {"n": int(female), "pct": round(100 * female / len(first), 1)}
    return out


def checks(manifests: dict[str, pd.DataFrame]) -> list[str]:
    """Errors (patient across splits or cohorts) and warnings (grade with < 2 patients in a split)."""
    msgs = []
    for task, df in manifests.items():
        n_multi = int((df.groupby("patient_id")["split"].nunique() > 1).sum())
        if n_multi:
            msgs.append(f"ERROR {task}: {n_multi} patient(s) in more than one split")
        for (split, label), g in df.groupby(["split", "label"]):
            if g["patient_id"].nunique() < 2:
                msgs.append(f"WARNING {task}: grade {label} has {g['patient_id'].nunique()} patient in {split}")
    if len(manifests) == 2:
        shared = set(manifests["mes"]["patient_id"]) & set(manifests["ulcer"]["patient_id"])
        if shared and all("procedure_id" in d.columns for d in manifests.values()):
            msgs.append(f"ERROR: {len(shared)} patient(s) in both UC and CD")
    return msgs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out-dir", default="results/inventory")
    args = parser.parse_args(argv)
    root, out_dir = Path(args.data_root), Path(args.out_dir)

    manifests = {t: load_manifest(root / t / "splits" / "dataset_manifest.csv")
                 for t in TASKS if (root / t / "splits" / "dataset_manifest.csv").is_file()}
    video_data = load_video_data(root / "video_data.csv")
    report, lines = {}, []
    for task, df in manifests.items():
        table = split_table(df)
        demo = demographics(df, video_data, task)
        report[task] = {"splits": table.to_dict("records"), "demographics": demo}
        lines += [f"## {TASKS[task]}", "", _markdown(table), "",
                  f"Demographics: {json.dumps(demo)}", ""]
    msgs = checks(manifests)
    report["checks"] = msgs
    lines += ["## Checks", ""] + [f"- {m}" for m in msgs or ["none"]]

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "inventory.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "inventory.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("\n".join(lines))
    return 1 if any(m.startswith("ERROR") for m in msgs) else 0


if __name__ == "__main__":
    sys.exit(main())
