"""Manifest creation and normalisation (src.data.manifest_utils, scripts/*/create_manifest.py)."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd
import pytest

from src.data.manifest_utils import (
    IMAGE_EXTENSIONS,
    collect_frames,
    find_duplicate_frames,
    load_manifest,
    normalize_manifest,
    report_duplicate_frames,
)
from tests.conftest import write_png


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def clip_tree(tmp_path, rng):
    """{label}/{patient}/{clip}/<image> tree with non-images, hidden entries and odd folders."""
    root = tmp_path / "processed"
    for rel in [
        "0/pA/clip1/f001.png",
        "0/pA/clip1/f002.JPG",  # upper-case extension
        "0/pA/clip2/f001.jpeg",
        "score_1/pB/clip1/f001.bmp",  # score_N label folder
        "score_1/pB/clip1/f002.png",
        "2/pC/clip9/f001.png",
    ]:
        write_png(root / rel, rng)
    # Not frames: other files, hidden entries, a non-label folder.
    (root / "0/pA/clip1/notes.txt").write_text("x", encoding="utf-8")
    (root / "0/pA/clip1/Thumbs.db").write_bytes(b"\x00")
    write_png(root / "0/pA/clip1/.hidden.png", rng)
    write_png(root / "0/pA/.ipynb_checkpoints/f001.png", rng)
    write_png(root / "0/.ipynb_checkpoints/clip/f001.png", rng)
    write_png(root / "extras/pZ/clip/f001.png", rng)
    (root / "README.md").write_text("x", encoding="utf-8")
    return root


# ---------------------------------------------------------------------------
# collect_frames
# ---------------------------------------------------------------------------


def test_collect_frames_clip_layout(clip_tree, capsys):
    rows = collect_frames(clip_tree)
    got = {r["filepath"]: (r["label"], r["patient_id"], r["video_id"]) for r in rows}

    assert got == {
        "0/pA/clip1/f001.png": (0, "pA", "pA__clip1"),
        "0/pA/clip1/f002.JPG": (0, "pA", "pA__clip1"),
        "0/pA/clip2/f001.jpeg": (0, "pA", "pA__clip2"),
        "score_1/pB/clip1/f001.bmp": (1, "pB", "pB__clip1"),
        "score_1/pB/clip1/f002.png": (1, "pB", "pB__clip1"),
        "2/pC/clip9/f001.png": (2, "pC", "pC__clip9"),
    }
    assert all("\\" not in r["filepath"] for r in rows)
    assert "hidden entr" in capsys.readouterr().out


def test_image_extensions():
    assert {".jpg", ".jpeg", ".png", ".bmp"} == IMAGE_EXTENSIONS


def test_collect_frames_flat_limuc_layout(tmp_path, rng):
    # score_X/{patient}/<image>: every image is its own single-frame clip.
    root = tmp_path / "limuc"
    for rel in ["score_0/101/a.png", "score_0/101/b.png", "score_3/102/a.bmp", "score_2/7/x.png"]:
        write_png(root / rel, rng)
    write_png(root / "score_0/101/.ipynb_checkpoints/a-checkpoint.png", rng)

    rows = collect_frames(root)
    got = {r["filepath"]: (r["label"], r["patient_id"], r["video_id"]) for r in rows}
    assert got == {
        "score_0/101/a.png": (0, "101", "101__a"),
        "score_0/101/b.png": (0, "101", "101__b"),
        "score_2/7/x.png": (2, "7", "7__x"),
        "score_3/102/a.bmp": (3, "102", "102__a"),
    }
    assert collect_frames(root) == rows


def test_collect_frames_layout_mismatch_raises(tmp_path, clip_tree, rng):
    flat = tmp_path / "flat"
    write_png(flat / "score_0/101/a.png", rng)
    assert collect_frames(flat)
    assert collect_frames(clip_tree)


def test_collect_frames_empty_tree(tmp_path, capsys):
    (tmp_path / "0").mkdir()
    assert collect_frames(tmp_path) == []
    assert "no image frames" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Duplicate frames
# ---------------------------------------------------------------------------


@pytest.fixture
def dup_tree(tmp_path, rng):
    root = tmp_path / "dups"
    write_png(root / "0/pA/clip1/f001.png", rng)
    write_png(root / "0/pA/clip1/f002.png", rng)
    write_png(root / "1/pB/clip1/f001.png", rng)
    # Byte-identical copies: one in another patient's clip, one inside the same clip.
    shutil.copyfile(root / "0/pA/clip1/f001.png", root / "1/pB/clip1/f009.png")
    shutil.copyfile(root / "0/pA/clip1/f002.png", root / "0/pA/clip1/f003.png")
    return root


def test_find_duplicate_frames(dup_tree):
    rows = collect_frames(dup_tree, check_duplicates=False)
    dups = find_duplicate_frames(rows, dup_tree)

    assert len(dups) == 4  # two groups of two
    assert set(dups["dup_group_size"]) == {2}
    cross = dups[dups["dup_n_clips"] == 2]
    assert set(cross["relative_path"]) == {"0/pA/clip1/f001.png", "1/pB/clip1/f009.png"}
    assert set(cross["dup_n_patients"]) == {2} and set(cross["dup_n_labels"]) == {2}
    # Only the copy in the later clip is flagged for removal; within-clip repeats are kept.
    assert list(dups.loc[dups["is_cross_clip_copy"], "relative_path"]) == ["1/pB/clip1/f009.png"]
    # The original row index is kept.
    assert all(rows[i]["filepath"] == dups.loc[i, "relative_path"] for i in dups.index)

    stats = report_duplicate_frames(dups)
    assert stats["n_duplicate_rows"] == 4
    assert stats["n_distinct_images"] == 2
    assert stats["n_excess_rows"] == 2
    assert stats["n_cross_clip_copies"] == 1
    assert stats["n_groups_cross_patient"] == 1
    assert stats["n_groups_cross_label"] == 1


def test_find_duplicate_frames_with_split_column(dup_tree):
    rows = collect_frames(dup_tree, check_duplicates=False)
    for r in rows:
        r["split"] = "test" if r["patient_id"] == "pB" else "train"
    dups = find_duplicate_frames(pd.DataFrame(rows), dup_tree)
    assert "dup_n_splits" in dups.columns
    assert report_duplicate_frames(dups)["n_groups_cross_split"] == 1


def test_no_duplicates(clip_tree):
    rows = collect_frames(clip_tree, check_duplicates=False)
    dups = find_duplicate_frames(rows, clip_tree)
    assert dups.empty
    assert {"md5", "dup_group_size", "is_cross_clip_copy"} <= set(dups.columns)
    assert report_duplicate_frames(dups)["n_duplicate_rows"] == 0


def test_collect_frames_warns_and_can_drop_cross_clip_copies(dup_tree, capsys):
    rows = collect_frames(dup_tree)
    assert "byte-identical" in capsys.readouterr().out
    assert len(rows) == 5

    kept = collect_frames(dup_tree, drop_cross_clip_duplicates=True)
    assert "1/pB/clip1/f009.png" not in {r["filepath"] for r in kept}
    assert len(kept) == 4


def test_missing_files_are_skipped(dup_tree, capsys):
    rows = collect_frames(dup_tree, check_duplicates=False)
    rows.append(
        {"filepath": "0/pA/clip1/gone.png", "label": 0, "patient_id": "pA", "video_id": "pA__clip1"}
    )
    dups = find_duplicate_frames(rows, dup_tree)
    assert "gone.png" not in " ".join(dups["relative_path"])
    assert "missing" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# normalize_manifest / load_manifest
# ---------------------------------------------------------------------------


def test_normalize_new_format_with_backslashes():
    df = pd.DataFrame(
        {
            "filepath": ["0\\pA\\clip1\\f001.png", "1/pB/clip1/f001.png"],
            "label": [0, 1],
            "patient_id": ["pA", "pB"],
            "video_id": ["pA__clip1", "pB__clip1"],
        }
    )
    out = normalize_manifest(df)

    assert "filepath" in df.columns  # input untouched
    assert list(out["relative_path"]) == ["0/pA/clip1/f001.png", "1/pB/clip1/f001.png"]
    assert list(out["clip_key"]) == list(out["video_id"]) == ["pA__clip1", "pB__clip1"]
    pd.testing.assert_frame_equal(normalize_manifest(out), out)  # idempotent




def test_load_manifest_normalises_windows_paths(tmp_path):
    path = tmp_path / "m.csv"
    path.write_text(
        "filepath,label,split,patient_id,video_id\n0\\pA\\c\\f.png,0,train,pA,pA__c\n",
        encoding="utf-8",
    )
    df = load_manifest(path)
    assert df.loc[0, "relative_path"] == "0/pA/c/f.png"
    assert df.loc[0, "clip_key"] == "pA__c"


# ---------------------------------------------------------------------------
# create_manifest scripts (end to end on a synthetic tree)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task", ["mes", "ulcer"])
def test_create_manifest_script(task, tmp_path, rng):
    import importlib

    module = importlib.import_module(f"scripts.{task}.create_manifest")
    raw = tmp_path / "processed"
    labels = (0, 1, 2, 3) if task == "mes" else (0, 1)
    for p in range(16):
        label = labels[p % len(labels)]
        for f in range(3):
            write_png(raw / f"{label}/p{p:02d}/c0/f{f}.png", rng)
    out = tmp_path / "splits"
    args = module.build_parser().parse_args(
        ["--input-dir", str(raw), "--output-dir", str(out), "--seed", "1",
         "--split-strategy", "stratified"]
    )
    module.main(args)

    manifest = pd.read_csv(out / "dataset_manifest.csv")
    assert list(manifest.columns) == ["filepath", "label", "split", "patient_id", "video_id"]
    assert len(manifest) == 48
    assert not manifest["filepath"].str.contains("\\\\", regex=False).any()
    assert (manifest.groupby("patient_id")["split"].nunique() == 1).all()
    info = json.loads((out / "split_info.json").read_text(encoding="utf-8"))
    assert info["total_frames"] == 48 and info["split_strategy"] == "stratified"

    with pytest.raises(FileExistsError):
        module.main(args)  # refuses to overwrite without --overwrite
    module.main(
        module.build_parser().parse_args(
            ["--input-dir", str(raw), "--output-dir", str(out), "--overwrite",
             "--split-strategy", "coverage"]
        )
    )


@pytest.mark.parametrize(
    ("task", "description", "input_dir", "output_dir"),
    [
        ("mes", "Create MES scoring manifest", "data/mes/processed", "data/mes/splits"),
        ("ulcer", "Create ulcer detection manifest", "data/ulcer/processed", "data/ulcer/splits"),
    ],
)
def test_create_manifest_wrappers_preserve_task_cli(task, description, input_dir, output_dir):
    import importlib

    parser = importlib.import_module(f"scripts.{task}.create_manifest").build_parser()
    assert parser.description == description
    with pytest.raises(SystemExit):
        parser.parse_args([])  # --split-strategy is required
    args = parser.parse_args(["--split-strategy", "coverage"])
    assert (args.input_dir, args.output_dir) == (input_dir, output_dir)
