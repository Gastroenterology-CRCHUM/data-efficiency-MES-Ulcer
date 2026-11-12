"""Loader factories on a tiny synthetic image tree (src.data.dataloader, src.data.dataset)."""

from __future__ import annotations

import pandas as pd
import pytest
import torch

from src.data.dataloader import get_split_loaders, get_test_loader
from src.data.dataset import UlcerDataset


def _kwargs(manifest_path, data_dir, **extra):
    """One kwargs dict shared by both factories (as the runners use them)."""
    kw = dict(
        manifest_path=manifest_path,
        data_dir=data_dir,
        batch_size=4,
        img_size=32,
        num_workers=0,
        equalize=True,
        label_col="label",
        seed=42,
        use_randaugment=False,
        randaugment_m=9,
        use_random_erasing=False,
        random_erasing_p=0.25,
    )
    kw.update(extra)
    return kw


def _frame_ids(loader) -> list[str]:
    return list(loader.dataset.df["relative_path"])


def _patients(loader) -> set[str]:
    return set(loader.dataset.df["patient_id"])


def test_val_set_is_the_manifest_val_rows_for_every_seed(image_manifest):
    manifest_path, data_dir, df = image_manifest
    expected = list(df.loc[df["split"] == "val", "filepath"])
    for seed in (42, 84):
        train_loader, val_loader = get_split_loaders(
            subset_ratio=0.5, random_seed=seed, verbose=False, **_kwargs(manifest_path, data_dir, seed=seed)
        )
        assert _frame_ids(val_loader) == expected
        assert not _patients(train_loader) & _patients(val_loader)


def test_val_rows_in_the_manifest_are_used_as_is(image_manifest, tmp_path):
    manifest_path, data_dir, df = image_manifest
    df = df.copy()
    first_train_patient = df.loc[df["split"] == "train", "patient_id"].iloc[0]
    df.loc[df["patient_id"] == first_train_patient, "split"] = "val"
    path = tmp_path / "with_val.csv"
    df.to_csv(path, index=False)

    _, val_loader = get_split_loaders(
        subset_ratio=1.0, random_seed=0, verbose=False, **_kwargs(path, data_dir)
    )
    original_val = set(df.loc[df["split"] == "val", "patient_id"]) - {first_train_patient}
    assert _patients(val_loader) == original_val | {first_train_patient}


def test_manifest_without_val_rows_raises(image_manifest, tmp_path):
    manifest_path, data_dir, df = image_manifest
    path = tmp_path / "no_val.csv"
    df.assign(split=df["split"].replace({"val": "train"})).to_csv(path, index=False)
    with pytest.raises(ValueError, match="no 'val' rows"):
        get_split_loaders(**_kwargs(path, data_dir), verbose=False)


def test_subset_info_matches_the_train_loader(image_manifest):
    manifest_path, data_dir, _ = image_manifest
    train_loader, _ = get_split_loaders(
        subset_ratio=0.5, random_seed=42, verbose=False, **_kwargs(manifest_path, data_dir)
    )
    info = train_loader.subset_info
    full_train = pd.read_csv(manifest_path).query("split == 'train'")

    assert info["n_train"] == len(train_loader.dataset)
    assert info["n_train_full"] == len(full_train)
    assert info["realized_fraction"] == pytest.approx(info["n_train"] / info["n_train_full"])
    assert set(info["class_frac"]) == {"0", "1"}


def test_shuffle_order_is_reproducible_for_a_seed(image_manifest):
    manifest_path, data_dir, _ = image_manifest

    def epoch_order(seed):
        train_loader, _ = get_split_loaders(
            subset_ratio=1.0,
            random_seed=seed,
            verbose=False,
            **_kwargs(manifest_path, data_dir, seed=seed),
        )
        return [fid for _, _, _, frame_ids in train_loader for fid in frame_ids]

    first = epoch_order(1)
    assert first == epoch_order(1)
    assert first != epoch_order(2)
    assert sorted(first) == sorted(epoch_order(2))


def test_batches_have_the_expected_structure(image_manifest):
    manifest_path, data_dir, _ = image_manifest
    test_loader = get_test_loader(
        **_kwargs(manifest_path, data_dir), subset_ratio=0.5, random_seed=3, verbose=False
    )
    images, labels, clip_ids, frame_ids = next(iter(test_loader))

    assert images.shape[1:] == (3, 32, 32) and images.dtype == torch.float32
    assert labels.dtype == torch.long
    assert all("__" in c for c in clip_ids)
    assert all("\\" not in f and not f.startswith("/") for f in frame_ids)
    assert set(test_loader.dataset.df["split"]) == {"test"}


def test_factories_reject_unknown_kwargs(image_manifest):
    manifest_path, data_dir, _ = image_manifest
    with pytest.raises(TypeError, match="unexpected"):
        get_test_loader(**_kwargs(manifest_path, data_dir), not_a_loader_option=1)


@pytest.mark.parametrize("ratio", [0.0, 1.5])
def test_out_of_range_subset_ratio_raises(image_manifest, ratio):
    # A ratio above 1 must not silently fall back to the full train set.
    manifest_path, data_dir, _ = image_manifest
    with pytest.raises(ValueError, match="subset_ratio"):
        get_split_loaders(
            subset_ratio=ratio, random_seed=0, verbose=False, **_kwargs(manifest_path, data_dir)
        )


def test_dataset_accepts_filepath_manifests_with_backslashes(image_manifest):
    _, data_dir, df = image_manifest
    windows_style = df.assign(filepath=df["filepath"].str.replace("/", "\\", regex=False))
    ds = UlcerDataset(windows_style, data_dir)
    image, label, clip_id, frame_id = ds[0]

    assert frame_id == df["filepath"].iloc[0]
    assert clip_id == df["video_id"].iloc[0]
    assert int(label) == int(df["label"].iloc[0])
    assert image.size == (16, 16)


def test_dataset_load_error_names_the_relative_path(image_manifest):
    _, data_dir, df = image_manifest
    missing = pd.DataFrame(
        [
            {
                "filepath": "0/nobody/c0/missing.png",
                "label": 0,
                "split": "test",
                "patient_id": "nobody",
                "video_id": "nobody__c0",
            }
        ]
    )
    with pytest.raises(RuntimeError, match="0/nobody/c0/missing.png"):
        UlcerDataset(missing, data_dir)[0]
