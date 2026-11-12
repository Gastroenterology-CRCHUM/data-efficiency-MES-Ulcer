"""Global RNG seeding for reproducible training runs.

Exports
-------
    seed_everything : seed python / numpy / torch (CPU + CUDA) and set determinism flags
    seed_worker     : DataLoader ``worker_init_fn`` that seeds numpy/random per worker
    make_generator  : seeded ``torch.Generator`` for DataLoader shuffling

Typical use (one call at the start of every run, before building loaders/models)::

    seed_everything(run_cfg.training.random_seed, deterministic=True)
    loader = DataLoader(ds, shuffle=True, generator=make_generator(seed),
                        worker_init_fn=seed_worker)
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

# cuBLAS needs a fixed workspace to be deterministic on CUDA >= 10.2. The
# variable must be set before the first cuBLAS handle is created, so it only
# takes effect when seed_everything() runs before any GPU matmul in the process
# (it is inherited by spawned DataLoader / ProcessPool workers).
_CUBLAS_WORKSPACE_CONFIG = ":4096:8"


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed every global RNG used during training.

    Seeds ``random``, ``numpy`` (legacy global RNG), ``torch`` (CPU) and all
    CUDA devices. With ``deterministic=True`` also disables cuDNN autotuning,
    forces deterministic cuDNN kernels and enables
    ``torch.use_deterministic_algorithms(True, warn_only=True)`` so that ops
    without a deterministic implementation emit a warning instead of failing.

    With ``deterministic=False`` the determinism flags are switched off and
    ``cudnn.benchmark`` is enabled (fastest, not bit-reproducible).

    Args:
        seed:          Non-negative integer seed.
        deterministic: Enforce deterministic GPU kernels (slower).
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", _CUBLAS_WORKSPACE_CONFIG)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.use_deterministic_algorithms(False)


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn``: derive numpy/random seeds from torch's.

    PyTorch already seeds each worker's torch RNG with ``base_seed + worker_id``
    (``base_seed`` drawn from the loader's ``generator``); this propagates that
    seed to ``numpy`` and ``random`` so any augmentation using them is
    reproducible too.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    """Return a CPU ``torch.Generator`` seeded with ``seed``."""
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g
