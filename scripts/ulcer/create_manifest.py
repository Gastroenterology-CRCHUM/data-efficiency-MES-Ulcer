"""
scripts/ulcer/create_manifest.py
=================================
Scan the ulcer raw directory and generate a patient-level
train / val / test manifest CSV.

Data structure
--------------
data/ulcer/raw/
├── 0/                    <- label 0 (no ulcer)
│   └── vid_03_XXXX/      <- patient video  (= patient_id)
│       └── normal_N/     <- annotated clip (= clip_id)
│           └── *.jpg
└── 1/                    <- label 1 (ulcer)
    └── vid_03_XXXX/
        └── ulcer_N/
            └── *.jpg

Output: data/ulcer/splits/dataset_manifest.csv

Usage
-----
    python -m scripts.ulcer.create_manifest
    python -m scripts.ulcer.create_manifest --val-ratio 0.15 --test-ratio 0.15 --seed 42
    python -m scripts.ulcer.create_manifest --raw-dir data/ulcer/raw --out-dir data/ulcer/splits
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.data.manifest_utils import collect_frames, patient_level_split, print_split_stats


def main(args: argparse.Namespace) -> None:
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

    print(f"Scanning: {raw_dir}")
    rows = collect_frames(raw_dir)
    print(f"Found {len(rows):,} frames")

    labels = sorted({r["label"] for r in rows})
    print(f"Classes: {labels}  (0=NoUlcer, 1=Ulcer)")

    rows = patient_level_split(rows, args.val_ratio, args.test_ratio, args.seed)
    print_split_stats(rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dataset_manifest.csv"
    fieldnames = ["filepath", "label", "split", "patient_id", "video_id"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()
        csv.DictWriter(f, fieldnames=fieldnames).writerows(rows)
    print(f"Manifest saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create ulcer detection manifest")
    parser.add_argument("--raw-dir",    default="data/ulcer/processed", help="Frames directory")
    parser.add_argument("--out-dir",    default="data/ulcer/splits", help="Output directory")
    parser.add_argument("--val-ratio",  type=float, default=0.15,    help="Validation fraction")
    parser.add_argument("--test-ratio", type=float, default=0.15,    help="Test fraction")
    parser.add_argument("--seed",       type=int,   default=42,      help="Random seed")
    args = parser.parse_args()
    main(args)
