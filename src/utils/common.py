"""Common utilities."""

from __future__ import annotations

import torch


class ConfigurationError(Exception):
    """Raised when configuration is invalid."""

    pass


def get_device(device_id: int = 0) -> torch.device:
    """Return a safe device handle. Pass device_id < 0 to force CPU."""
    if device_id < 0 or not torch.cuda.is_available():
        return torch.device("cpu")
    if device_id >= torch.cuda.device_count():
        raise ValueError(
            f"GPU {device_id} not available. Available GPUs: {torch.cuda.device_count()}"
        )
    return torch.device(f"cuda:{device_id}")


def loader_dataset_size(loader) -> int:
    """Return the number of samples in a DataLoader's underlying dataset."""
    dataset = getattr(loader, "dataset", None)
    if dataset is not None and hasattr(dataset, "__len__"):
        return int(len(dataset))
    return 0
