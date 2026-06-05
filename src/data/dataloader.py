"""
DataLoader factory — patient-level train/val split.

Public API
----------
    get_split_loaders(...)  → (train_loader, val_loader)
    get_test_loader(...)    → test_loader
    get_val_loader(...)     → val_loader (fixed, independent of subset_ratio)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.data.dataset import UlcerDataset
from src.data.transforms import get_transforms

# ---------------------------------------------------------------------------
# Patient-level split helpers (merged from src/data/splits.py)
# ---------------------------------------------------------------------------


def _modal_patient_label(
    patient_id: str,
    df: pd.DataFrame,
    patient_col: str = "patient_id",
    label_col: str = "label",
) -> str:
    mask = df[patient_col].astype(str) == str(patient_id)
    vals = df.loc[mask, label_col].dropna()
    if vals.empty:
        return "unknown"
    mode_vals = vals.astype(int).mode()
    return str(int(mode_vals.iloc[0])) if not mode_vals.empty else "unknown"


def assign_val_split(
    train_df: pd.DataFrame,
    val_ratio: float,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carve a patient-level val set from train_df. Falls back to unstratified."""
    patients = train_df["patient_id"].unique().tolist()
    label_map = {pid: _modal_patient_label(pid, train_df) for pid in patients}
    strat_bins = [label_map[p] for p in patients]
    try:
        train_patients, val_patients = train_test_split(
            patients, test_size=val_ratio, random_state=random_seed, stratify=strat_bins
        )
    except ValueError:
        train_patients, val_patients = train_test_split(
            patients, test_size=val_ratio, random_state=random_seed
        )
    return (
        train_df[train_df["patient_id"].isin(train_patients)].copy(),
        train_df[train_df["patient_id"].isin(val_patients)].copy(),
    )


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _normalize_manifest(df: pd.DataFrame) -> pd.DataFrame:
    """Align column names between old and new manifest formats.

    Old format: relative_path, video_id (patient vid), segment_id, clip_key
    New format: filepath (= clip-relative path), video_id (= clip key, no segment_id)
    """
    if "relative_path" not in df.columns and "filepath" in df.columns:
        df = df.rename(columns={"filepath": "relative_path"})
    if "clip_key" not in df.columns and "video_id" in df.columns:
        df = df.copy()
        df["clip_key"] = df["video_id"]
    if "segment_id" not in df.columns:
        df = df.copy() if "clip_key" in df.columns else df
        df["segment_id"] = ""
    return df


def _make_loader(
    df: pd.DataFrame,
    data_dir: Path,
    transform,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    label_col: str = "label",
) -> DataLoader:
    dataset = UlcerDataset(df, data_dir, transform=transform, label_col=label_col)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=torch.cuda.is_available(),
    )


def _sampling_train(
    train_df: pd.DataFrame,
    subset_ratio: float,
    label_col: str = "label",
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Stratified clip-level sampling that preserves:
      1. Class ratio (modal label per clip) from the full train set.
      2. Frame-count distribution per class (tertile bins: few / medium / many).

    Works on the frame-level manifest — groups by clip_key, samples clips,
    then returns all frames belonging to sampled clips.
    Supports binary and multiclass labels.
    """
    # ── 1. Build clip-level summary ────────────────────────────────────────
    clip_df = train_df.groupby("clip_key", as_index=False).agg(
        _label_val=(label_col, lambda x: int(x.mode().iloc[0])),
        n_frames=(label_col, "count"),
        patient_id=("patient_id", "first"),
    )

    # Rename the temporary label column to the actual label_col name
    clip_df = clip_df.rename(columns={"_label_val": label_col})

    n_clips_total = len(clip_df)
    assert n_clips_total > 0, "clip_df is empty — check clip_key construction."

    # ── 2. Frame-count tertile bins (computed on full train set) ───────────
    clip_df["frame_bin"] = pd.Categorical(
        [""] * len(clip_df), categories=["few", "medium", "many"], ordered=True
    )
    for cls_val in clip_df[label_col].unique():
        mask = clip_df[label_col] == cls_val
        cls_frames = clip_df.loc[mask, "n_frames"]
        q33, q66 = cls_frames.quantile([1 / 3, 2 / 3])
        if q33 == q66:
            q33 = q66 - 1
        clip_df.loc[mask, "frame_bin"] = pd.cut(
            cls_frames,
            bins=[-np.inf, q33, q66, np.inf],
            labels=["few", "medium", "many"],
        )

    # ── 3. Stratum = class × frame_bin  ────────────────────────────────────
    clip_df["_stratum"] = clip_df[label_col].astype(str) + "_" + clip_df["frame_bin"].astype(str)

    # Sanity check: no clip should have a NaN stratum
    n_nan_strata = clip_df["_stratum"].isna().sum()
    if n_nan_strata > 0:
        raise ValueError(
            f"_sampling_train: {n_nan_strata} clips have NaN stratum. "
            "Check n_frames column for NaN or zero values."
        )

    # ── 4. Stratified sampling — proportional allocation ───────────────────
    # Compute global target first, then distribute across strata.
    rng = np.random.default_rng(random_seed)
    n_target = max(1, round(n_clips_total * subset_ratio))

    strata_sizes = clip_df.groupby("_stratum", observed=True).size()
    raw_alloc = strata_sizes / strata_sizes.sum() * n_target
    floor_alloc = raw_alloc.apply(np.floor).astype(int)
    remainders = raw_alloc - floor_alloc
    n_remaining = n_target - int(floor_alloc.sum())
    if n_remaining > 0:
        top_strata = remainders.nlargest(n_remaining).index
        floor_alloc[top_strata] += 1

    # Guarantee at least 1 clip per class — purely proportional allocation
    # gives 0 clips to rare classes at low ratios (e.g. Mayo 1 at 10%).
    for cls_val in clip_df[label_col].unique():
        cls_strata = clip_df[clip_df[label_col] == cls_val]["_stratum"].unique()
        if floor_alloc.reindex(cls_strata, fill_value=0).sum() == 0:
            # Add 1 to the stratum with the most clips for this class
            best = strata_sizes.reindex(cls_strata, fill_value=0).idxmax()
            floor_alloc[best] = floor_alloc.get(best, 0) + 1

    sampled_parts = []
    for stratum_name, g in clip_df.groupby("_stratum", observed=True):
        quota = int(floor_alloc.get(stratum_name, 0))
        if quota > 0:
            sampled_parts.append(
                g.sample(n=min(quota, len(g)), random_state=int(rng.integers(0, 2**31)))
            )
    sampled_clips = (
        pd.concat(sampled_parts).reset_index(drop=True) if sampled_parts else clip_df.iloc[:0]
    )

    if subset_ratio == 1.0 and len(sampled_clips) != n_clips_total:
        raise ValueError(
            f"_sampling_train: expected {n_clips_total} clips at ratio=1.0, "
            f"got {len(sampled_clips)}. Investigate stratum assignments."
        )

    # ── 5. Filter frame-level manifest ─────────────────────────────────────
    sampled_df = train_df[train_df["clip_key"].isin(sampled_clips["clip_key"])].copy()

    # ── 6. Diagnostics ─────────────────────────────────────────────────────
    n_frames_total = len(train_df)
    n_frames_subset = len(sampled_df)
    n_clips_subset = len(sampled_clips)

    class_counts = sampled_clips[label_col].value_counts().sort_index()
    class_ratio_full = clip_df[label_col].value_counts(normalize=True).sort_index()
    class_ratio_subset = sampled_clips[label_col].value_counts(normalize=True).sort_index()

    all_classes = sorted(class_counts.index)
    col_w = 8
    header = (
        f"  {'label':<10}"
        + f"{'clips':>{col_w}}"
        + f"{'full %':>{col_w}}"
        + f"{'subset %':>{col_w + 2}}"
    )
    print(
        f"\n[subset {subset_ratio:.0%}]  "
        f"clips: {n_clips_subset}/{n_clips_total}  "
        f"frames: {n_frames_subset}/{n_frames_total} "
        f"({n_frames_subset / n_frames_total:.1%})"
    )
    print(header)
    print("  " + "-" * (10 + col_w * 2 + col_w + 2))
    for cls in all_classes:
        n = class_counts.get(cls, 0)
        rf = class_ratio_full.get(cls, 0) * 100
        rs = class_ratio_subset.get(cls, 0) * 100
        print(f"  {cls!s:<10}{n:>{col_w}}{rf:>{col_w}.1f}%{rs:>{col_w + 1}.1f}%")

    bin_summary = (
        sampled_clips.groupby([label_col, "frame_bin"], observed=True).size().unstack(fill_value=0)
    )
    print(f"  frame bins:\n{bin_summary.to_string()}\n")

    return sampled_df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_split_loaders(
    manifest_path: Path,
    data_dir: Path,
    batch_size: int,
    img_size: int,
    subset_ratio: float = 1.0,
    label_col: str = "label",
    *,
    val_ratio: float = 0.15,
    num_workers: int = 8,
    equalize: bool = True,
    random_seed: int = 42,
    **augmentation_params,
) -> tuple[DataLoader, DataLoader]:
    """
    Return (train_loader, val_loader) for a classic single split.

    If the manifest has 'val' rows, they are used directly.
    Otherwise a patient-level val set is carved from 'train' using val_ratio.

    Args:
        val_ratio: Fraction of train patients used for val when no 'val'
                   rows exist in the manifest.
        subset_ratio: If <1.0, randomly subsample the train pool before
                   splitting (for faster experiments).

    Returns:
        (train_loader, val_loader)
    """
    manifest = _normalize_manifest(pd.read_csv(manifest_path))

    if "val" in manifest["split"].values:
        train_df = manifest[manifest["split"] == "train"].copy()
        val_df = manifest[manifest["split"] == "val"].copy()
    else:
        all_train = manifest[manifest["split"] == "train"].copy()
        train_df, val_df = assign_val_split(all_train, val_ratio=val_ratio, random_seed=random_seed)
        print(
            "[!] Validation split not found in manifest — splitting train randomly into train/val."
        )

    if subset_ratio < 1.0:
        train_df = _sampling_train(train_df, subset_ratio, label_col, random_seed)

    train_transform = get_transforms(
        img_size, is_training=True, equalize=equalize, **augmentation_params
    )
    val_transform = get_transforms(img_size, is_training=False, equalize=equalize)

    return (
        _make_loader(
            train_df,
            data_dir,
            train_transform,
            batch_size,
            num_workers,
            label_col=label_col,
            shuffle=True,
        ),
        _make_loader(
            val_df,
            data_dir,
            val_transform,
            batch_size,
            num_workers,
            label_col=label_col,
            shuffle=False,
        ),
    )


def get_test_loader(
    manifest_path: Path,
    data_dir: Path,
    batch_size: int,
    img_size: int,
    label_col: str = "label",
    *,
    num_workers: int = 8,
    equalize: bool = True,
) -> DataLoader:
    """
    Return the held-out test DataLoader.
    Call only once, after all training and model selection are complete.

    Returns:
        test_loader
    """
    manifest = _normalize_manifest(pd.read_csv(manifest_path))
    test_df = manifest[manifest["split"] == "test"].copy()
    transform = get_transforms(img_size, is_training=False, equalize=equalize)
    return _make_loader(
        test_df, data_dir, transform, batch_size, num_workers, label_col=label_col, shuffle=False
    )


def get_val_loader(
    manifest_path: Path,
    data_dir: Path,
    batch_size: int,
    img_size: int,
    label_col: str = "label",
    *,
    val_ratio: float = 0.15,
    num_workers: int = 8,
    equalize: bool = True,
    random_seed: int = 42,
) -> DataLoader:
    """Fixed val loader — independent of any subset_ratio."""
    manifest = _normalize_manifest(pd.read_csv(manifest_path))
    if "val" in manifest["split"].values:
        val_df = manifest[manifest["split"] == "val"].copy()
    else:
        all_train = manifest[manifest["split"] == "train"].copy()
        _, val_df = assign_val_split(all_train, val_ratio=val_ratio, random_seed=random_seed)
        print(
            "[!] Validation split not found in manifest — splitting train randomly into train/val."
        )
    transform = get_transforms(img_size, is_training=False, equalize=equalize)
    return _make_loader(
        val_df, data_dir, transform, batch_size, num_workers, label_col=label_col, shuffle=False
    )
