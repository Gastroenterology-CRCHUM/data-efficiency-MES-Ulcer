"""
scripts/mes/create_manifest.py
================================
Scan the MES raw directory and generate a patient-level
train / val / test manifest CSV.

Data structure
--------------
data/mes/raw/
├── 0/                    <- Mayo score 0
│   └── vid_03_XXXX/      <- patient video  (= patient_id)
│       └── mayo_NNN/     <- annotated clip (= clip_id)
│           └── *.jpg
├── 1/                    <- Mayo score 1
├── 2/                    <- Mayo score 2
└── 3/                    <- Mayo score 3

Output: data/mes/splits/dataset_manifest.csv

Usage
-----
    python -m scripts.mes.create_manifest
    python -m scripts.mes.create_manifest --val-ratio 0.15 --test-ratio 0.15 --seed 42
    python -m scripts.mes.create_manifest --raw-dir data/mes/raw --out-dir data/mes/splits
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

    if not rows:
        raise RuntimeError("No frames found. Check the raw directory structure.")

    labels = sorted({r["label"] for r in rows})
    print(f"Classes: {labels}  (Mayo scores)")

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
    parser = argparse.ArgumentParser(description="Create MES scoring manifest")
    parser.add_argument("--raw-dir",    default="data/mes/processed", help="Frames directory")
    parser.add_argument("--out-dir",    default="data/mes/splits", help="Output directory")
    parser.add_argument("--val-ratio",  type=float, default=0.15,  help="Validation fraction")
    parser.add_argument("--test-ratio", type=float, default=0.15,  help="Test fraction")
    parser.add_argument("--seed",       type=int,   default=42,    help="Random seed")
    args = parser.parse_args()
    main(args)
