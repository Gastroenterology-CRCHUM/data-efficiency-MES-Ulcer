"""
PyTorch Dataset for colonoscopy ulcer detection.


Exports
-------
    UlcerDataset
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.manifest_utils import normalize_manifest


class UlcerDataset(Dataset):
    """
    Frame classification dataset (binary ulcer or multiclass MES).

    ``__getitem__`` returns ``(image, label, clip_id, frame_id)`` where
    ``frame_id`` is the POSIX ``relative_path`` of the frame.

    Args:
        df:        Manifest rows (``filepath``, ``label``, ``video_id``,
                   ``patient_id``), normalised with ``normalize_manifest``.
        data_dir:  Root directory from which `relative_path` is resolved.
        transform: Optional torchvision transform pipeline.
        cache_dir: Frame cache (src.data.frame_cache); frames are read from it instead.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        data_dir: Path,
        transform=None,
        label_col: str = "label",
        cache_dir: Path | None = None,
    ) -> None:
        self.df = normalize_manifest(df).reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.label_col = label_col
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        rel_path = str(row["relative_path"])
        image_path = (
            self.data_dir / rel_path if self.cache_dir is None else self.cache_dir / f"{rel_path}.png"
        )
        try:
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            with Image.open(image_path) as im:
                image = im.convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to load image {rel_path}: {type(e).__name__}: {e}") from e

        label = torch.tensor(int(row[self.label_col]), dtype=torch.long)
        clip_id = str(row["video_id"])
        id_frame = rel_path

        if self.transform:
            image = self.transform(image)

        return image, label, clip_id, id_frame

