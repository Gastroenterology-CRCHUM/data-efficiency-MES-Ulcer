"""Shared fixtures and synthetic-data builders for the test suite.

Everything here is synthetic: patient / clip identifiers are made up
(``p000``, ``p000__c0``), images are small random PNGs written under pytest's
``tmp_path``. No test reads the real datasets, downloads weights or needs a GPU.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.data.manifest_utils import normalize_manifest, patient_level_split

# ---------------------------------------------------------------------------
# Row builders
# ---------------------------------------------------------------------------


def make_rows(
    n_patients: int = 30,
    n_classes: int = 4,
    *,
    clips_per_patient: tuple[int, int] = (1, 3),
    frames_per_clip: tuple[int, int] = (3, 12),
    class_probs: list[float] | None = None,
    single_label_patients: bool = False,
    seed: int = 0,
) -> list[dict]:
    """Frame rows in the ``collect_frames`` format (filepath, label, patient_id, video_id).

    Every patient has 1-3 clips; each clip has one label drawn from
    ``class_probs`` (the same label for all clips of a patient when
    ``single_label_patients``). Clip keys are ``{patient}__c{j}``.
    """
    rng = np.random.default_rng(seed)
    probs = class_probs or [1.0 / n_classes] * n_classes
    rows: list[dict] = []
    for i in range(n_patients):
        pid = f"p{i:03d}"
        n_clips = int(rng.integers(clips_per_patient[0], clips_per_patient[1] + 1))
        patient_label = int(rng.choice(n_classes, p=probs))
        for j in range(n_clips):
            label = patient_label if single_label_patients else int(rng.choice(n_classes, p=probs))
            clip = f"c{j}"
            n_frames = int(rng.integers(frames_per_clip[0], frames_per_clip[1] + 1))
            for k in range(n_frames):
                rows.append(
                    {
                        "filepath": f"{label}/{pid}/{clip}/f{k:03d}.png",
                        "label": label,
                        "patient_id": pid,
                        "video_id": f"{pid}__{clip}",
                    }
                )
    return rows


def rows_with_labels(patients_per_class: dict[int, int], frames: int = 4) -> list[dict]:
    """One single-clip patient per entry: ``{label: n_patients}`` -> rows."""
    rows: list[dict] = []
    idx = 0
    for label, n_pat in patients_per_class.items():
        for _ in range(n_pat):
            pid = f"p{idx:03d}"
            idx += 1
            for k in range(frames):
                rows.append(
                    {
                        "filepath": f"{label}/{pid}/c0/f{k:03d}.png",
                        "label": label,
                        "patient_id": pid,
                        "video_id": f"{pid}__c0",
                    }
                )
    return rows


def make_train_frames(
    clips_per_class: dict[int, int],
    *,
    frames_per_clip: tuple[int, int] = (5, 60),
    n_patients: int = 15,
    seed: int = 0,
) -> pd.DataFrame:
    """Normalised frame-level train manifest with ``clips_per_class[c]`` clips of class c."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for label, n_clips in clips_per_class.items():
        for j in range(n_clips):
            pid = f"p{int(rng.integers(n_patients)):03d}"
            clip_key = f"{pid}__l{label}c{j:03d}"
            n_frames = int(rng.integers(frames_per_clip[0], frames_per_clip[1] + 1))
            for k in range(n_frames):
                rows.append(
                    {
                        "filepath": f"{label}/{pid}/l{label}c{j:03d}/f{k:03d}.png",
                        "label": label,
                        "patient_id": pid,
                        "video_id": clip_key,
                        "split": "train",
                    }
                )
    return normalize_manifest(pd.DataFrame(rows))


def frames_from_clip_sizes(clip_sizes: dict[int, list[int]]) -> pd.DataFrame:
    """Normalised train manifest with explicit clip sizes: ``{label: [n_frames, ...]}``.

    Clip ``j`` of class ``c`` gets key ``{c-letter}{j}`` (``a0, a1, ...`` for
    class 0, ``b0, ...`` for class 1) and its own patient, so the clip-key
    order equals the list order.
    """
    rows: list[dict] = []
    for label, sizes in clip_sizes.items():
        letter = "abcdefgh"[label]
        for j, n_frames in enumerate(sizes):
            key = f"{letter}{j}"
            for k in range(n_frames):
                rows.append(
                    {
                        "filepath": f"{label}/{key}/clip/f{k:03d}.png",
                        "label": label,
                        "patient_id": f"pat_{key}",
                        "video_id": key,
                        "split": "train",
                    }
                )
    return normalize_manifest(pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# Image trees
# ---------------------------------------------------------------------------


def write_png(path: Path, rng: np.random.Generator, size: int = 16) -> None:
    """Write a small random RGB PNG (random content, so no two files are identical)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(path)


def write_frames(root: Path, rows: list[dict], *, size: int = 16, seed: int = 0) -> None:
    """Materialise ``rows`` (``filepath`` relative to ``root``) as random PNGs."""
    rng = np.random.default_rng(seed)
    for row in rows:
        write_png(root / row["filepath"], rng, size=size)


@pytest.fixture
def image_manifest(tmp_path: Path):
    """Synthetic binary dataset on disk: (manifest_path, data_dir, manifest_df).

    24 patients, 1-2 clips each, 2-4 frames per clip, 16 px PNGs, train / val / test.
    """
    rows = make_rows(
        24,
        2,
        clips_per_patient=(1, 2),
        frames_per_clip=(2, 4),
        class_probs=[0.6, 0.4],
        seed=3,
    )
    rows = patient_level_split(rows, 0.15, 0.2, seed=1, verbose=False)
    data_dir = tmp_path / "processed"
    write_frames(data_dir, rows)
    df = pd.DataFrame(rows)[["filepath", "label", "split", "patient_id", "video_id"]]
    manifest_path = tmp_path / "dataset_manifest.csv"
    df.to_csv(manifest_path, index=False)
    return manifest_path, data_dir, df


# ---------------------------------------------------------------------------
# Global torch state
# ---------------------------------------------------------------------------


@pytest.fixture
def restore_torch_flags(monkeypatch):
    """Restore torch determinism flags (and CUBLAS_WORKSPACE_CONFIG) after a test."""
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    state = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
    )
    yield
    enabled, warn_only, benchmark, deterministic = state
    torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
    torch.backends.cudnn.benchmark = benchmark
    torch.backends.cudnn.deterministic = deterministic
