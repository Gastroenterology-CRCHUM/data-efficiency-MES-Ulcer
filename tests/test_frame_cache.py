"""Frame cache: identical tensors, automatic use when complete (src.data.frame_cache)."""

from __future__ import annotations

import torch

from src.data.dataloader import get_split_loaders, get_test_loader
from src.data.frame_cache import build_frame_cache, cache_dir_for, cached_path

KW = dict(batch_size=4, img_size=32, num_workers=0, equalize=True, seed=0)


def _batches(loader):
    return [(x.clone(), list(f)) for x, _, _, f in loader]


def test_cached_frames_give_identical_tensors(image_manifest):
    manifest_path, data_dir, df = image_manifest
    plain_test = _batches(get_test_loader(manifest_path, data_dir, **KW))
    torch.manual_seed(5)
    plain_train = _batches(get_split_loaders(manifest_path, data_dir, verbose=False, **KW)[0])

    cache_dir, n = build_frame_cache(manifest_path, data_dir, 32, True, workers=2)
    assert n == len(df) and cache_dir == cache_dir_for(data_dir, 32, True)
    assert build_frame_cache(manifest_path, data_dir, 32, True)[1] == 0  # nothing left to write

    test_loader = get_test_loader(manifest_path, data_dir, **KW)
    assert test_loader.dataset.cache_dir == cache_dir
    for (a, fa), (b, fb) in zip(plain_test, _batches(test_loader)):
        assert fa == fb and torch.equal(a, b)
    torch.manual_seed(5)  # same augmentation draws
    for (a, _), (b, _) in zip(plain_train, _batches(get_split_loaders(manifest_path, data_dir, verbose=False, **KW)[0])):
        assert torch.equal(a, b)


def test_incomplete_cache_is_ignored(image_manifest):
    manifest_path, data_dir, df = image_manifest
    cache_dir, _ = build_frame_cache(manifest_path, data_dir, 32, True)
    cached_path(cache_dir, df["filepath"].iloc[0]).unlink()
    assert get_test_loader(manifest_path, data_dir, **KW).dataset.cache_dir is None


def test_run_config_records_cached_preprocessing(image_manifest):
    from src.models.classifier import _describe_loader

    manifest_path, data_dir, _ = image_manifest
    plain = _describe_loader(get_test_loader(manifest_path, data_dir, **KW))
    build_frame_cache(manifest_path, data_dir, 32, True)
    cached = _describe_loader(get_test_loader(manifest_path, data_dir, **KW))
    assert (plain["img_size"], plain["equalize"]) == (cached["img_size"], cached["equalize"]) == (32, True)
    assert cached["frame_cache"] is True and cached["mean"] == plain["mean"]
