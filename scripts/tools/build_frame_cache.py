"""Build the 256-px frame cache (resize-with-pad + CLAHE) for one dataset.

    python -m scripts.tools.build_frame_cache --task mes      # also: ulcer, limuc

Rerun after adding frames: only missing cache files are written. The loaders use
the cache automatically once it covers every manifest frame.
"""

from __future__ import annotations

import argparse
import os
import time

from src.config.models import ModelRegistryEntry
from src.data.frame_cache import build_frame_cache

TASKS = {
    "mes": ("data/mes/splits/dataset_manifest.csv", "data/mes/processed"),
    "ulcer": ("data/ulcer/splits/dataset_manifest.csv", "data/ulcer/processed"),
    "limuc": ("data_LIMUC/mes/splits/dataset_manifest.csv", "data_LIMUC/mes/processed"),
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--img-size", type=int,
                        default=ModelRegistryEntry.__dataclass_fields__["img_size"].default)
    parser.add_argument("--no-equalize", action="store_true", help="Cache without CLAHE")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 8)
    args = parser.parse_args(argv)

    manifest, data_dir = TASKS[args.task]
    start = time.time()
    cache_dir, n = build_frame_cache(manifest, data_dir, args.img_size, not args.no_equalize, args.workers)
    print(f"{cache_dir}: {n} frame(s) written in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
