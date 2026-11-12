"""LIMUC preprocessing (scripts/mes/prepare_limuc.py) on a tiny synthetic source tree."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.data.manifest_utils import collect_frames

prepare_limuc = pytest.importorskip("scripts.mes.prepare_limuc")


def _bmp(path, rng, size=(12, 10)):
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(pixels).save(path, format="BMP")


@pytest.fixture
def limuc_src(tmp_path):
    """{patient}/Mayo {X}/<frame>.bmp plus a stray file."""
    rng = np.random.default_rng(0)
    src = tmp_path / "patient_based_classified_images"
    for rel in ["101/Mayo 0/a.bmp", "101/Mayo 2/b.bmp", "102/Mayo 3/c.bmp", "103/mayo 1/d.bmp"]:
        _bmp(src / rel, rng)
    (src / "102/Mayo 3/notes.txt").write_text("x", encoding="utf-8")
    return src


def test_plan_conversion_maps_to_the_flat_layout(limuc_src):
    items, counts = prepare_limuc.plan_conversion(limuc_src)
    got = {(it["dst"], it["label"], it["patient_id"]) for it in items}

    assert got == {
        ("score_0/101/a.png", 0, "101"),
        ("score_2/101/b.png", 2, "101"),
        ("score_3/102/c.png", 3, "102"),
        ("score_1/103/d.png", 1, "103"),
    }
    assert counts == {"skipped_other": 1}


def test_check_sources_flags_empty_and_truncated_bmps(limuc_src):
    good = limuc_src / "101/Mayo 0/a.bmp"
    empty = limuc_src / "101/Mayo 2/b.bmp"
    truncated = limuc_src / "102/Mayo 3/c.bmp"
    empty.write_bytes(b"")
    truncated.write_bytes(good.read_bytes()[:40])  # header says more bytes follow
    not_bmp = limuc_src / "103/mayo 1/d.bmp"
    not_bmp.write_bytes(b"PNG-not-a-bitmap")

    items, _ = prepare_limuc.plan_conversion(limuc_src)
    problems = {p.name: why for p, why in prepare_limuc.check_sources(items)}
    assert problems["b.bmp"] == "empty"
    assert problems["c.bmp"].startswith("truncated")
    assert problems["d.bmp"] == "no BMP signature"
    assert "a.bmp" not in problems


def test_convert_one_and_collect_frames(limuc_src, tmp_path):
    items, _ = prepare_limuc.plan_conversion(limuc_src)
    dst_root = tmp_path / "processed"
    for it in items:
        assert prepare_limuc.convert_one(it["src"], dst_root / it["dst"], (8, 6))
    with Image.open(dst_root / "score_0/101/a.png") as im:
        assert im.size == (8, 6) and im.format == "PNG"

    # The converted tree is a flat score_X/{patient}/<frame> layout for create_manifest.
    rows = collect_frames(dst_root, check_duplicates=False)
    assert {r["video_id"] for r in rows} == {"101__a", "101__b", "102__c", "103__d"}

    broken = tmp_path / "broken.bmp"
    broken.write_bytes(b"BM\x00\x00")
    assert not prepare_limuc.convert_one(broken, tmp_path / "out.png", None)
