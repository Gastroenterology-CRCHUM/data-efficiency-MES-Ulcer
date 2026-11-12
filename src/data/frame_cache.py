"""Lossless cache of the deterministic preprocessing (resize-with-pad, then CLAHE).

The random training augmentations run after these two steps, so loading a cached
frame gives exactly the tensors of the full pipeline at a fraction of the decode
cost. Cache layout: ``<data_dir>/../cache_<img_size>[_clahe]/<filepath>.png``.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from src.data.manifest_utils import load_manifest
from src.data.transforms import deterministic_steps


def cache_dir_for(data_dir: Path | str, img_size: int, equalize: bool) -> Path:
    """Cache folder next to ``data_dir`` for one input size / CLAHE setting."""
    return Path(data_dir).parent / f"cache_{img_size}{'_clahe' if equalize else ''}"


def cached_path(cache_dir: Path, relative_path: str) -> Path:
    return cache_dir / f"{relative_path}.png"


def cache_is_complete(cache_dir: Path, relative_paths) -> bool:
    """True when every frame has a cached file."""
    return cache_dir.is_dir() and all(cached_path(cache_dir, p).is_file() for p in relative_paths)


def build_frame_cache(
    manifest_path: Path | str, data_dir: Path | str, img_size: int, equalize: bool, workers: int = 8
) -> tuple[Path, int]:
    """Write the cached PNG of every manifest frame that has none. Returns (cache_dir, n_written)."""
    data_dir = Path(data_dir)
    cache_dir = cache_dir_for(data_dir, img_size, equalize)
    steps = deterministic_steps(img_size, equalize)
    todo = [p for p in load_manifest(manifest_path)["relative_path"]
            if not cached_path(cache_dir, p).is_file()]

    def write(rel: str) -> None:
        with Image.open(data_dir / rel) as im:
            img = im.convert("RGB")
        for step in steps:
            img = step(img)
        out = cached_path(cache_dir, rel)
        out.parent.mkdir(parents=True, exist_ok=True)
        img.save(out, format="PNG", compress_level=1)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(write, todo))
    return cache_dir, len(todo)
