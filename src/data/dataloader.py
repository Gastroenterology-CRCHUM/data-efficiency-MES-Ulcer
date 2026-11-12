"""
DataLoader factory — patient-level train/val split, seeded loaders, subset sampling.

Public API
----------
    get_split_loaders(...)  → (train_loader, val_loader)
    get_test_loader(...)    → test_loader
    describe_subset(...)    → realised size / class mix of a training subset (dict)

Both loader factories accept the same keyword arguments (train-only keys are
ignored by get_test_loader), so a runner can pass one kwargs dict to each.

Reproducibility
---------------
Every DataLoader gets a seeded ``torch.Generator`` (shuffle order, worker base
seeds) and ``worker_init_fn=seed_worker``. Call
``src.utils.seed_everything(seed)`` once per run as well, so that model
initialisation and main-process augmentations are seeded too.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Generator, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.dataset import UlcerDataset
from src.data.frame_cache import cache_dir_for, cache_is_complete
from src.data.manifest_utils import load_manifest, normalize_manifest
from src.data.transforms import get_transforms
from src.utils.seed import make_generator, seed_worker

# Keys accepted by every loader factory but only used by get_split_loaders.
_TRAIN_ONLY_KWARGS = frozenset(
    {
        "subset_ratio",
        "random_seed",
        "use_randaugment",
        "randaugment_m",
        "use_random_erasing",
        "random_erasing_p",
        "verbose",
    }
)


def _frame_cache(manifest: pd.DataFrame, data_dir: Path, img_size: int, equalize: bool):
    """Cache folder when it holds every manifest frame, else None (originals are decoded)."""
    cache_dir = cache_dir_for(data_dir, img_size, equalize)
    if cache_is_complete(cache_dir, manifest["relative_path"]):
        return cache_dir
    if cache_dir.is_dir():
        print(f"[!] Incomplete frame cache {cache_dir.name}: decoding original frames.")
    return None

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _check_shared_kwargs(func_name: str, kwargs: dict) -> None:
    """Reject unknown keyword arguments (train-only keys are accepted and ignored)."""
    unknown = sorted(set(kwargs) - _TRAIN_ONLY_KWARGS)
    if unknown:
        raise TypeError(f"{func_name}() got unexpected keyword argument(s): {unknown}")


def _split_frames(manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(train, val) frames of a normalised manifest; the manifest must have val rows."""
    if "val" not in manifest["split"].values:
        raise ValueError("manifest has no 'val' rows; rebuild it with create_manifest")
    return (
        manifest[manifest["split"] == "train"].copy(),
        manifest[manifest["split"] == "val"].copy(),
    )


# Environment of the DataLoader worker processes while they start. numpy and scipy each
# bundle an OpenBLAS that commits per-thread buffers for every core when it is imported
# (~0.7 GB each on a 32-thread CPU). On Windows every worker is a spawned process that
# imports both again, so each worker committed ~1.5 GB it never uses: the workers only
# decode images and apply CLAHE / torchvision transforms, none of which calls BLAS.
_WORKER_ENV = {"OPENBLAS_NUM_THREADS": "1"}


@contextlib.contextmanager
def _worker_env() -> Generator[None, None, None]:
    """Set ``_WORKER_ENV`` while worker processes are started, then restore os.environ.

    Only processes started meanwhile see it: the OpenBLAS of this process is already
    initialised, so its thread count (and every result computed here) is unchanged.
    """
    saved = {key: os.environ.get(key) for key in _WORKER_ENV}
    os.environ.update(_WORKER_ENV)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class _LeanWorkerLoader(DataLoader):
    """DataLoader whose worker processes start with one OpenBLAS thread (``_WORKER_ENV``).

    Batches, their order and the worker seeds are those of a plain DataLoader.
    """

    def _get_iterator(self):
        if self.num_workers == 0:
            return super()._get_iterator()
        with _worker_env():  # the workers are started by the multiprocessing iterator
            return super()._get_iterator()


def _make_loader(
    df: pd.DataFrame,
    data_dir: Path,
    transform,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    label_col: str = "label",
    *,
    seed: int = 42,
    cache_dir: Path | None = None,
) -> DataLoader:
    """DataLoader with a seeded generator and seeded workers.

    ``seed`` fixes the shuffle order and the per-worker RNG seeds (numpy,
    random and torch inside each worker, see ``src.utils.seed.seed_worker``).
    Worker processes start with a single OpenBLAS thread (``_LeanWorkerLoader``).
    """
    dataset = UlcerDataset(df, data_dir, transform=transform, label_col=label_col, cache_dir=cache_dir)
    return _LeanWorkerLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=torch.cuda.is_available(),
        generator=make_generator(seed),
        worker_init_fn=seed_worker,
    )


def _clip_table(train_df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """One row per clip (sorted by clip_key): modal label, n_frames, patient_id."""
    clip_df = train_df.groupby("clip_key", as_index=False).agg(
        _label_val=(label_col, lambda x: int(x.mode().iloc[0])),
        n_frames=(label_col, "count"),
        patient_id=("patient_id", "first"),
    )
    return clip_df.rename(columns={"_label_val": label_col})


# ---------------------------------------------------------------------------
# Training-subset sampling
# ---------------------------------------------------------------------------


def _sampling_train(
    train_df: pd.DataFrame,
    subset_ratio: float,
    label_col: str = "label",
    random_seed: int = 42,
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Sample a training subset holding ``subset_ratio`` of every class's frames.

    Works on the frame-level manifest: clips are identified by ``clip_key`` and
    each clip's class is its modal frame label. The clips of class ``c``
    (sorted by ``clip_key``) are put in one random order that depends only on
    ``(random_seed, c)``, never on ``subset_ratio``, and their frames are listed
    clip after clip, in path (temporal) order within a clip. The subset takes
    the first ``B_c = max(1, round(subset_ratio * F_c))`` frames of that list,
    ``F_c`` being the frames of class-``c`` clips:

    * every class holds exactly its budget, so for every seed the realised
      fraction equals ``subset_ratio`` (to one frame per class) and the class
      mix is that of the full train split;
    * clips are taken whole except the last one, cut to its first frames: at
      most one partial clip per class;
    * subsets are **nested**: for a given seed, the subset at a smaller ratio
      is contained in the subset at any larger ratio;
    * ``subset_ratio == 1`` returns every frame.

    Args:
        train_df:     Normalised frame-level train manifest (needs ``clip_key``,
                      ``relative_path``, ``patient_id`` and ``label_col``).
        subset_ratio: Target fraction in (0, 1].
        label_col:    Label column.
        random_seed:  Seed of the subset draw.
        verbose:      Print the realised subset composition.

    Returns:
        The selected frames (a copy, original index and row order kept).
    """
    if not 0 < subset_ratio <= 1:
        raise ValueError(f"subset_ratio must be in (0, 1], got {subset_ratio}")
    return _sampling_train_frame_budget(
        train_df, subset_ratio, label_col, random_seed, verbose=verbose
    )


def _class_rng(random_seed: int, cls, cls_index: int) -> np.random.Generator:
    """Per-(seed, class) generator, independent of the subset ratio."""
    try:
        token = int(cls)
    except (TypeError, ValueError):
        token = cls_index
    if token < 0:
        token = cls_index
    return np.random.default_rng([int(random_seed) % 2**32, token])


def _sampling_train_frame_budget(
    train_df: pd.DataFrame,
    subset_ratio: float,
    label_col: str,
    random_seed: int,
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    """Nested per-class frame-budget sampling (see ``_sampling_train``)."""
    clip_df = _clip_table(train_df, label_col)
    assert len(clip_df) > 0, "clip_df is empty - check clip_key construction."

    # One row per train_df position, so a non-unique index cannot select extra rows.
    frames = pd.DataFrame(
        {"clip_key": train_df["clip_key"].to_numpy(), "path": train_df["relative_path"].to_numpy()}
    )
    keep = np.zeros(len(frames), dtype=bool)
    for cls_index, cls in enumerate(sorted(clip_df[label_col].unique())):
        cls_clips = clip_df[clip_df[label_col] == cls].sort_values("clip_key", kind="stable")
        order = _class_rng(random_seed, cls, cls_index).permutation(len(cls_clips))
        clip_rank = pd.Series(np.arange(len(order)), index=cls_clips["clip_key"].to_numpy()[order])
        cls_frames = frames[frames["clip_key"].isin(clip_rank.index)]
        cls_frames = cls_frames.assign(rank=cls_frames["clip_key"].map(clip_rank))
        cls_frames = cls_frames.sort_values(["rank", "path"])
        # First B_c frames of the ratio-independent order -> exact budget, nested.
        budget = max(1, int(round(subset_ratio * len(cls_frames))))
        keep[cls_frames.index[:budget]] = True

    sampled_df = train_df[keep].copy()
    if subset_ratio == 1.0 and len(sampled_df) != len(train_df):
        raise ValueError(
            f"_sampling_train: expected all {len(train_df)} frames at ratio=1.0, "
            f"got {len(sampled_df)}."
        )
    if verbose:
        _print_subset(sampled_df, train_df, subset_ratio, label_col)
    return sampled_df


def describe_subset(
    train_df: pd.DataFrame,
    full_train_df: pd.DataFrame,
    label_col: str = "label",
) -> dict:
    """Realised size and class mix of a training subset (JSON-serialisable).

    Args:
        train_df:      Frames actually used for training (e.g.
                       ``train_loader.dataset.df``).
        full_train_df: The full train split the subset was drawn from.
        label_col:     Label column.

    Returns:
        dict with
            n_train, n_clips, n_patients            subset sizes
            realized_fraction                       n_train / full train frames
            class_frac                              {label: share of subset frames}
            n_train_full, n_clips_full, n_patients_full
            clip_fraction                           n_clips / full train clips
            class_frac_full                         {label: share of full train frames}
            class_realized_fraction                 {label: subset / full frames of that class}
        Labels are string keys (``"0"``, ``"1"``, ...).
    """
    sub = normalize_manifest(train_df)
    full = normalize_manifest(full_train_df)
    n_full = len(full)

    def _key(v) -> str:
        try:
            return str(int(v))
        except (TypeError, ValueError):
            return str(v)

    labels = sorted(full[label_col].unique())
    sub_counts = sub[label_col].value_counts()
    full_counts = full[label_col].value_counts()
    n_sub = len(sub)
    return {
        "n_train": int(n_sub),
        "n_clips": int(sub["clip_key"].nunique()),
        "n_patients": int(sub["patient_id"].nunique()),
        "realized_fraction": float(n_sub / n_full) if n_full else float("nan"),
        "class_frac": {
            _key(c): float(sub_counts.get(c, 0) / n_sub) if n_sub else 0.0 for c in labels
        },
        "n_train_full": int(n_full),
        "n_clips_full": int(full["clip_key"].nunique()),
        "n_patients_full": int(full["patient_id"].nunique()),
        "clip_fraction": (
            float(sub["clip_key"].nunique() / full["clip_key"].nunique())
            if n_full
            else float("nan")
        ),
        "class_frac_full": {
            _key(c): float(full_counts.get(c, 0) / n_full) if n_full else 0.0 for c in labels
        },
        "class_realized_fraction": {
            _key(c): float(sub_counts.get(c, 0) / full_counts[c]) for c in labels
        },
    }


def _print_subset(
    sampled_df: pd.DataFrame,
    train_df: pd.DataFrame,
    subset_ratio: float,
    label_col: str,
) -> None:
    """Print the realised subset size and per-class composition."""
    info = describe_subset(sampled_df, train_df, label_col)
    print(
        f"\n[subset {subset_ratio:.0%} | frame_budget]  "
        f"clips: {info['n_clips']}/{info['n_clips_full']}  "
        f"frames: {info['n_train']}/{info['n_train_full']} ({info['realized_fraction']:.1%})  "
        f"patients: {info['n_patients']}/{info['n_patients_full']}"
    )
    sub_clips = sampled_df.groupby(label_col)["clip_key"].nunique()
    sub_frames = sampled_df[label_col].value_counts()
    full_frames = train_df[label_col].value_counts()
    n_sub, n_full = max(len(sampled_df), 1), max(len(train_df), 1)
    print(f"  {'label':<8}{'clips':>7}{'frames':>9}{'full %':>9}{'subset %':>10}{'of class':>10}")
    for cls in sorted(full_frames.index):
        n_f = int(sub_frames.get(cls, 0))
        print(
            f"  {cls!s:<8}{int(sub_clips.get(cls, 0)):>7}{n_f:>9}"
            f"{full_frames[cls] / n_full:>9.1%}{n_f / n_sub:>10.1%}"
            f"{n_f / full_frames[cls]:>10.1%}"
        )


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
    num_workers: int = 8,
    equalize: bool = True,
    random_seed: int = 42,
    seed: int | None = None,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
    use_randaugment: bool = False,
    randaugment_m: int = 9,
    use_random_erasing: bool = False,
    random_erasing_p: float = 0.25,
    verbose: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """
    Return (train_loader, val_loader) from the manifest's train and val rows.

    Args:
        subset_ratio:    Fraction in (0, 1]; if < 1.0, train on that fraction
                         of every class's train frames (see ``_sampling_train``).
                         Values outside (0, 1] raise ``ValueError``.
        random_seed:     Seed of the subset draw.
        seed:            Seed of the DataLoader generators (shuffle order and
                         worker seeds). Defaults to ``random_seed``.
        mean, std:       Input normalisation of the backbone (default ImageNet);
                         see ``src.config.models.get_norm_stats``.
        use_randaugment, randaugment_m, use_random_erasing, random_erasing_p:
                         Training augmentations (``TrainingConfig`` fields).
        verbose:         Print the realised subset composition.

    Returns:
        (train_loader, val_loader). ``train_loader.subset_info`` holds
        ``describe_subset(train frames, full train frames)``.
    """
    # Validate before loading anything: a ratio >= 1 skips _sampling_train, so
    # an out-of-range value (e.g. 1.5) would otherwise train on the full split.
    if not 0 < subset_ratio <= 1:
        raise ValueError(f"subset_ratio must be in (0, 1], got {subset_ratio}")
    manifest = load_manifest(manifest_path)
    cache_dir = _frame_cache(manifest, Path(data_dir), img_size, equalize)
    full_train_df, val_df = _split_frames(manifest)

    if subset_ratio < 1.0:
        train_df = _sampling_train(
            full_train_df,
            subset_ratio,
            label_col,
            random_seed,
            verbose=verbose,
        )
    else:
        train_df = full_train_df

    loader_seed = random_seed if seed is None else seed
    train_transform = get_transforms(
        img_size,
        is_training=True,
        equalize=equalize,
        cached=cache_dir is not None,
        use_randaugment=use_randaugment,
        randaugment_m=randaugment_m,
        use_random_erasing=use_random_erasing,
        random_erasing_p=random_erasing_p,
        mean=mean,
        std=std,
    )
    val_transform = get_transforms(
        img_size, is_training=False, equalize=equalize, cached=cache_dir is not None,
        mean=mean, std=std,
    )

    train_loader = _make_loader(
        train_df,
        data_dir,
        train_transform,
        batch_size,
        num_workers,
        label_col=label_col,
        shuffle=True,
        seed=loader_seed,
        cache_dir=cache_dir,
    )
    train_loader.subset_info = describe_subset(train_df, full_train_df, label_col)
    val_loader = _make_loader(
        val_df,
        data_dir,
        val_transform,
        batch_size,
        num_workers,
        label_col=label_col,
        shuffle=False,
        seed=loader_seed,
        cache_dir=cache_dir,
    )
    return train_loader, val_loader


def get_test_loader(
    manifest_path: Path,
    data_dir: Path,
    batch_size: int,
    img_size: int,
    label_col: str = "label",
    *,
    num_workers: int = 8,
    equalize: bool = True,
    seed: int = 42,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
    **train_only_kwargs,
) -> DataLoader:
    """
    Return the held-out test DataLoader.
    Call only once, after all training and model selection are complete.

    Args:
        seed:      Seed of the loader generator (worker seeds; no shuffling).
        mean, std: Input normalisation of the backbone (default ImageNet).
        **train_only_kwargs: train-only loader keys, accepted and ignored.

    Returns:
        test_loader
    """
    _check_shared_kwargs("get_test_loader", train_only_kwargs)
    manifest = load_manifest(manifest_path)
    cache_dir = _frame_cache(manifest, Path(data_dir), img_size, equalize)
    test_df = manifest[manifest["split"] == "test"].copy()
    transform = get_transforms(
        img_size, is_training=False, equalize=equalize, cached=cache_dir is not None,
        mean=mean, std=std,
    )
    return _make_loader(
        test_df,
        data_dir,
        transform,
        batch_size,
        num_workers,
        label_col=label_col,
        shuffle=False,
        seed=seed,
        cache_dir=cache_dir,
    )
