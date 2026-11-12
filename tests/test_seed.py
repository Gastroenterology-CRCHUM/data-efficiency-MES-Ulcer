"""Global seeding and seeded DataLoaders (src.utils.seed)."""

from __future__ import annotations

import os
import random

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from src.utils import make_generator, seed_everything, seed_worker

pytestmark = pytest.mark.usefixtures("restore_torch_flags")


def _draws() -> tuple:
    return (
        random.random(),
        tuple(np.random.rand(3)),
        tuple(torch.rand(3).tolist()),
        tuple(torch.nn.Linear(4, 2).weight.flatten().tolist()),  # model init
    )


def test_seed_everything_reproduces_all_rngs():
    seed_everything(123)
    first = _draws()
    seed_everything(123)
    assert _draws() == first
    seed_everything(124)
    other = _draws()
    assert all(a != b for a, b in zip(first, other))


def test_seed_everything_accepts_large_seeds():
    seed_everything(2**40 + 5)
    first = _draws()
    seed_everything(2**40 + 5)
    assert _draws() == first


def test_seed_everything_determinism_flags():
    seed_everything(0, deterministic=True)
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.is_deterministic_algorithms_warn_only_enabled()
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"

    seed_everything(0, deterministic=False)
    assert not torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.deterministic is False
    assert torch.backends.cudnn.benchmark is True


def test_make_generator():
    assert torch.equal(
        torch.randperm(20, generator=make_generator(5)),
        torch.randperm(20, generator=make_generator(5)),
    )
    assert not torch.equal(
        torch.randperm(20, generator=make_generator(5)),
        torch.randperm(20, generator=make_generator(6)),
    )


def _loader_order(seed: int, **kw) -> list[int]:
    ds = TensorDataset(torch.arange(64))
    loader = DataLoader(ds, batch_size=8, shuffle=True, generator=make_generator(seed), **kw)
    return [int(v) for (batch,) in loader for v in batch]


def test_dataloader_shuffle_order_is_reproducible():
    assert _loader_order(11) == _loader_order(11)
    assert _loader_order(11) != _loader_order(12)
    assert sorted(_loader_order(11)) == list(range(64))


def test_seed_everything_fixes_an_unseeded_dataloader():
    # Without an explicit generator, DataLoader draws its shuffle from the global torch RNG.
    def order() -> list[int]:
        loader = DataLoader(TensorDataset(torch.arange(32)), batch_size=4, shuffle=True)
        return [int(v) for (batch,) in loader for v in batch]

    seed_everything(21)
    first = order()
    seed_everything(21)
    assert order() == first
    seed_everything(22)
    assert order() != first


def test_shuffle_order_ignores_the_global_torch_rng():
    torch.manual_seed(0)
    a = _loader_order(3)
    torch.manual_seed(999)
    torch.rand(10)
    assert _loader_order(3) == a


def test_seed_worker_propagates_the_torch_seed():
    torch.manual_seed(2024)
    seed_worker(0)
    got = (np.random.rand(), random.random())
    np.random.seed(2024)
    random.seed(2024)
    assert got == (np.random.rand(), random.random())


class _RandomAugDataset(Dataset):
    """Returns numpy / random / torch draws, like an augmentation would (module-level: picklable)."""

    def __len__(self) -> int:
        return 16

    def __getitem__(self, idx: int):
        return torch.tensor([idx, np.random.rand(), random.random(), torch.rand(1).item()])


def _worker_draws(seed: int) -> torch.Tensor:
    loader = DataLoader(
        _RandomAugDataset(),
        batch_size=4,
        shuffle=True,
        num_workers=2,
        generator=make_generator(seed),
        worker_init_fn=seed_worker,
    )
    return torch.cat(list(loader))


@pytest.mark.slow
def test_worker_augmentations_are_reproducible():
    # Spawns worker processes (a few seconds on Windows).
    first = _worker_draws(7)
    assert torch.equal(first, _worker_draws(7))
    assert not torch.equal(first, _worker_draws(8))
    # numpy / random differ between workers (not one shared stream).
    assert first[:, 1].unique().numel() == first.shape[0]
