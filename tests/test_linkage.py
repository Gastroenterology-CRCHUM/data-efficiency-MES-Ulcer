"""Patient linkage (video_data.csv) and the temporal split."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts.mes import create_manifest as mes_manifest
from src.analysis.statistics import load_test_clip_ids
from src.data.manifest_utils import link_patients, load_video_data
from src.data.splits import temporal_patient_split
from tests.conftest import write_png

# procedure -> (patient, date, highest grade); P2 straddles the 2025-06-01 cut-off.
PROCEDURES = {
    "vid_03_1": ("P1", "2024-01-10", 0), "vid_03_2": ("P2", "2024-02-10", 1),
    "vid_03_3": ("P3", "2024-03-10", 2), "vid_03_4": ("P4", "2024-04-10", 3),
    "vid_03_5": ("P5", "2024-05-10", 0), "vid_03_6": ("P6", "2024-06-10", 2),
    "vid_03_7": ("P7", "2025-07-10", 1), "vid_03_8": ("P2", "2025-08-10", 2),
    "vid_03_9": ("P9", "2025-09-10", 3),
}
CUTOFF = "2025-06-01"


def _video_data(path, rows=None):
    rows = rows or [
        {"video_id": v, "patient_id": p, "dataset": "mes", "date_of_procedure": d}
        for v, (p, d, _) in PROCEDURES.items()
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.fixture
def tree(tmp_path):
    """data/mes/processed/{label}/{procedure}/c0/*.png with data/video_data.csv."""
    rng = np.random.default_rng(0)
    raw = tmp_path / "data" / "mes" / "processed"
    for proc, (_, _, grade) in PROCEDURES.items():
        for label in sorted({0, grade}):
            for f in range(2):
                write_png(raw / f"{label}/{proc}/c{label}/f{f}.png", rng)
    _video_data(tmp_path / "data" / "video_data.csv")
    return raw


def _run(raw, out, *extra):
    args = mes_manifest.build_parser().parse_args(
        ["--input-dir", str(raw), "--output-dir", str(out), "--skip-duplicate-check", *extra]
    )
    mes_manifest.main(args)
    return pd.read_csv(out / "dataset_manifest.csv"), json.loads((out / "split_info.json").read_text())


def _rows():
    return [{"patient_id": v, "label": g} for v, (_, _, g) in PROCEDURES.items()]


def test_absent_file_keeps_todays_manifest(tree, tmp_path):
    linked, _ = _run(tree, tmp_path / "a", "--split-strategy", "coverage")
    assert "procedure_id" in linked.columns and linked["patient_id"].str.startswith("P").all()
    off, _ = _run(tree, tmp_path / "b", "--split-strategy", "coverage", "--video-data", "none")
    (tmp_path / "data" / "video_data.csv").unlink()
    absent, _ = _run(tree, tmp_path / "c", "--split-strategy", "coverage")
    assert list(absent.columns) == ["filepath", "label", "split", "patient_id", "video_id"]
    assert absent["patient_id"].str.startswith("vid_03_").all()
    assert absent.equals(off)


def test_temporal_split_end_to_end(tree, tmp_path):
    m, info = _run(tree, tmp_path / "s", "--split-strategy", "temporal", "--test-cutoff", CUTOFF)
    split_of = m.groupby("patient_id")["split"].agg(set)
    assert (split_of.map(len) == 1).all()  # no patient spans splits
    assert {p for p, s in split_of.items() if s == {"test"}} == {"P7", "P9"}
    assert split_of["P2"] != {"test"}  # earlier procedure keeps P2 in development
    assert info["split_strategy"] == "temporal" and info["test_cutoff"] == CUTOFF
    assert len(info["video_data_sha256"]) == 64 and info["n_procedures"] == 9


def test_temporal_split_is_deterministic_and_stratified():
    rows = [dict(r, date_of_procedure=PROCEDURES[r["patient_id"]][1]) for r in _rows()]
    a, _ = temporal_patient_split([dict(r) for r in rows], CUTOFF, 0.5, 3, verbose=False)
    b, _ = temporal_patient_split([dict(r) for r in rows], CUTOFF, 0.5, 3, verbose=False)
    assert [r["split"] for r in a] == [r["split"] for r in b]
    val_grades = {r["label"] for r in a if r["split"] == "val"}
    assert val_grades == {0, 2}  # strata with >= 2 development patients


def test_loader_rules(tmp_path):
    assert load_video_data(tmp_path / "missing.csv") is None
    base = [{"video_id": "vid_03_1", "patient_id": "P1", "dataset": "mes"}]
    with pytest.raises(ValueError, match="missing required"):
        load_video_data(_video_data(tmp_path / "a.csv", [{"video_id": "vid_03_1", "dataset": "mes"}]))
    vd = load_video_data(_video_data(tmp_path / "b.csv", base * 2))  # exact duplicate dropped
    assert len(vd) == 1
    with pytest.raises(ValueError, match="conflicting"):
        load_video_data(_video_data(tmp_path / "c.csv", base + [dict(base[0], patient_id="P9")]))
    with pytest.raises(ValueError, match="raw hospital"):
        load_video_data(_video_data(tmp_path / "d.csv", [dict(base[0], patient_id="1234567")]))


def test_link_patients_errors(tmp_path):
    vd = load_video_data(_video_data(tmp_path / "v.csv"))
    with pytest.raises(ValueError, match="no video_data.csv row"):
        link_patients([{"patient_id": "vid_03_99", "label": 0}], vd, "mes")
    with pytest.raises(ValueError, match="not 'ulcer'"):
        link_patients(_rows(), vd, "ulcer")
    rows = link_patients(_rows(), vd, "mes")
    assert rows[0]["procedure_id"] == "vid_03_1" and rows[0]["patient_id"] == "P1"


def test_cluster_levels(tree, tmp_path):
    _run(tree, tmp_path / "s", "--split-strategy", "temporal", "--test-cutoff", CUTOFF)
    manifest = tmp_path / "s" / "dataset_manifest.csv"
    assert set(load_test_clip_ids(manifest)) == {"P7", "P9"}
    assert set(load_test_clip_ids(manifest, level="procedure")) == {"vid_03_7", "vid_03_9"}


def test_inventory_reports_and_flags_patients_across_splits(tree, tmp_path):
    from scripts.tools import cohort_inventory

    data_root = tmp_path / "data"
    _run(tree, data_root / "mes" / "splits", "--split-strategy", "temporal", "--test-cutoff", CUTOFF)
    out = tmp_path / "inventory"
    assert cohort_inventory.main(["--data-root", str(data_root), "--out-dir", str(out)]) == 0
    report = json.loads((out / "inventory.json").read_text())
    assert report["mes"]["demographics"]["patients"] == 8  # P1-P9 without P8 (P2 has two procedures)

    manifest = data_root / "mes" / "splits" / "dataset_manifest.csv"
    df = pd.read_csv(manifest)
    df.loc[df["procedure_id"] == "vid_03_8", "split"] = "test"  # P2 now spans splits
    df.to_csv(manifest, index=False)
    assert cohort_inventory.main(["--data-root", str(data_root), "--out-dir", str(out)]) == 1
