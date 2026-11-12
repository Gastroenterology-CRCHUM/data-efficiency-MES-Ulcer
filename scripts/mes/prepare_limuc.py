"""Convert the LIMUC release (Labeled Images for Ulcerative Colitis, CC BY 4.0) into the
processed-frames tree read by ``create_manifest`` and ``run_data_efficiency``.

Input  (``--input-dir``: the unpacked ``patient_based_classified_images`` folder)
    {patient_id}/Mayo {0-3}/*.bmp          352 x 288, 24-bit BMP
Output (``--output-dir``)
    score_{0-3}/{patient_id}/{stem}.png    lossless PNG, every frame is its own clip
    preprocess_stats.json

The source resolution is kept (``--size W H`` resizes). Every source file is checked before
anything is written: empty or truncated BMPs abort the run; ``--check-only`` writes nothing.

    python -m scripts.mes.prepare_limuc --check-only
    python -m scripts.mes.prepare_limuc --input-dir data_LIMUC/patient_based_classified_images --output-dir data_LIMUC/mes/processed
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from src.data.manifest_utils import IMAGE_EXTENSIONS

MAYO_DIR = re.compile(r"^mayo\s*(\d+)$", re.IGNORECASE)
def _display_path(path: Path) -> str:
    """POSIX path relative to the working directory when possible (no local prefixes)."""
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def plan_conversion(src_root: Path) -> tuple[list[dict], dict]:
    """Map every LIMUC source image to its processed path.

    Regular files are ``{patient}/Mayo {X}/<file>`` -> ``score_{X}/{patient}/<stem>.png``.
    Returns:
        (items, counts): items are dicts with ``src`` (Path), ``dst`` (POSIX path
        relative to the output root), ``label`` and ``patient_id``; counts has
        ``skipped_other`` (non-image or unexpected files).
    """
    src_root = Path(src_root)
    items: list[dict] = []
    counts = Counter(skipped_other=0)
    unexpected: list[str] = []
    for path in sorted(src_root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(src_root)
        parts = rel.parts
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            counts["skipped_other"] += 1
            continue

        if len(parts) == 3 and MAYO_DIR.match(parts[1]):
            patient_id, class_dir = parts[0], parts[1]
        else:
            unexpected.append(rel.as_posix())
            counts["skipped_other"] += 1
            continue
        label = int(MAYO_DIR.match(class_dir).group(1))
        items.append(
            {
                "src": path,
                "dst": f"score_{label}/{patient_id}/{path.stem}.png",
                "label": label,
                "patient_id": patient_id,
            }
        )
    if unexpected:
        print(
            f"  WARNING: {len(unexpected)} image(s) outside {{patient}}/Mayo X/, e.g. {unexpected[:3]}"
        )

    dst_counts = Counter(it["dst"] for it in items)
    clashes = [d for d, n in dst_counts.items() if n > 1]
    if clashes:
        raise ValueError(
            f"{len(clashes)} output path(s) would be written twice, e.g. {clashes[:3]}"
        )
    return items, dict(counts)


def check_sources(items: list[dict]) -> list[tuple[Path, str]]:
    """Return (path, problem) for empty files and BMPs shorter than their header size."""
    problems: list[tuple[Path, str]] = []
    for it in items:
        src = it["src"]
        size = src.stat().st_size
        if size == 0:
            problems.append((src, "empty"))
            continue
        if src.suffix.lower() == ".bmp":
            with open(src, "rb") as fh:
                head = fh.read(6)
            if head[:2] != b"BM":
                problems.append((src, "no BMP signature"))
                continue
            declared = int.from_bytes(head[2:6], "little")
            if size < declared:
                problems.append((src, f"truncated ({size} of {declared} bytes)"))
    return problems


def convert_one(src: Path, dst: Path, size: tuple[int, int] | None) -> bool:
    """Decode ``src``, resize to ``size`` (W, H) with INTER_LINEAR, write PNG ``dst``."""
    # imdecode/imencode (not imread/imwrite) so non-ASCII paths work on Windows;
    # same codecs and defaults, so the pixels and bytes are identical.
    frame = cv2.imdecode(np.fromfile(str(src), dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return False
    if size is not None:
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)
    ok, buf = cv2.imencode(".png", frame)
    if not ok:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    buf.tofile(str(dst))
    return True


def main(args: argparse.Namespace) -> None:
    src_root = Path(args.input_dir)
    dst_root = Path(args.output_dir)
    size = None if args.size is None else (int(args.size[0]), int(args.size[1]))
    if not src_root.is_dir():
        raise FileNotFoundError(f"LIMUC source folder not found: {src_root}")

    t0 = time.perf_counter()
    items, counts = plan_conversion(src_root)
    if not items:
        raise RuntimeError(f"No images found under {src_root}; expected {{patient}}/Mayo X/*.bmp")
    labels = Counter(it["label"] for it in items)
    n_patients = len({it["patient_id"] for it in items})
    print(f"Source : {_display_path(src_root)}")
    print(
        f"Plan   : {len(items):,} images, {n_patients} patients, "
        f"per class {dict(sorted(labels.items()))}, skipped other={counts['skipped_other']}"
    )

    problems = check_sources(items)
    print(
        f"Check  : {len(problems):,} unreadable source file(s)  ({time.perf_counter() - t0:.1f}s)"
    )
    if problems:
        kinds = Counter(p.split(" (")[0] for _, p in problems)
        for path, problem in problems[:5]:
            print(f"  {path.relative_to(src_root).as_posix()}: {problem}")
        raise SystemExit(
            f"{len(problems):,} of {len(items):,} source files are empty or truncated "
            f"({dict(kinds)}); re-copy or re-extract the LIMUC archive. Nothing was written."
        )
    if args.check_only:
        return

    if dst_root.exists() and any(dst_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{dst_root} is not empty; pass --overwrite to write into it")

    t1 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        results = list(
            pool.map(lambda it: convert_one(it["src"], dst_root / it["dst"], size), items)
        )
    failed = [it["src"] for it, ok in zip(items, results) if not ok]
    elapsed = time.perf_counter() - t1

    stats = {
        "input_dir": _display_path(src_root),
        "output_dir": _display_path(dst_root),
        "total_input_frames": len(items),
        "preprocessed_frames": len(items) - len(failed),
        "failed_frames": len(failed),
        "n_patients": n_patients,
        "label_distribution": {str(k): int(v) for k, v in sorted(labels.items())},
        "transform": {
            "resize_wh": list(size) if size else None,
            "interpolation": "INTER_LINEAR" if size else None,
            "format": "png",
            "crop": None,
        },
        "opencv_version": cv2.__version__,
    }
    dst_root.mkdir(parents=True, exist_ok=True)
    with open(dst_root / "preprocess_stats.json", "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)

    print(
        f"Wrote  : {stats['preprocessed_frames']:,} PNG to {_display_path(dst_root)} ({elapsed:.0f}s)"
    )
    if failed:
        print(f"  {len(failed)} image(s) could not be decoded, e.g. {[p.name for p in failed[:3]]}")
        raise SystemExit(1)
    print(
        "Next   : python -m scripts.mes.create_manifest "
        f"--input-dir {_display_path(dst_root)} --output-dir <splits dir> "
        "--split-strategy stratified"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert the LIMUC release to processed PNG frames"
    )
    parser.add_argument(
        "--input-dir",
        default="data_LIMUC/patient_based_classified_images",
        help="LIMUC 'patient_based_classified_images' folder",
    )
    parser.add_argument(
        "--output-dir",
        default="data_LIMUC/mes/processed",
        help="Output frames root (score_X/{patient}/*.png)",
    )
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        metavar=("W", "H"),
        default=None,
        help="Optional output width and height; native source resolution is kept by default",
    )
    parser.add_argument(
        "--workers", type=int, default=min(8, os.cpu_count() or 1), help="Conversion threads"
    )
    parser.add_argument(
        "--check-only", action="store_true", help="Plan and integrity check only; write nothing"
    )
    parser.add_argument("--overwrite", action="store_true", help="Write into a non-empty --output-dir")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
