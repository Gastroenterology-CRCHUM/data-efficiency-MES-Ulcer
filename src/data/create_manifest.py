"""Shared implementation for the MES and ulcer manifest entry points."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from src.data.manifest_utils import collect_frames, link_patients, load_video_data
from src.data.splits import SPLIT_STRATEGIES, assign_splits

FIELDNAMES = ["filepath", "label", "split", "patient_id", "video_id"]
LINKED_FIELDNAMES = FIELDNAMES + ["procedure_id"]


def video_data_path(args: argparse.Namespace) -> Path | None:
    """``--video-data``; default ``video_data.csv`` two levels above ``--input-dir``."""
    if args.video_data == "none":
        return None
    if args.video_data:
        return Path(args.video_data)
    return Path(args.input_dir).parent.parent / "video_data.csv"


def create_manifest(
    args: argparse.Namespace,
    *,
    class_description: str,
    dataset: str,
) -> None:
    """Scan frames, link patients (``video_data.csv``), split by patient, write the manifest.

    ``dataset`` is the ``video_data.csv`` value of this task (``mes`` or ``ulcer``).
    Without the file, ``patient_id`` is the procedure folder.
    """
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    out_path = output_dir / "dataset_manifest.csv"

    if not input_dir.exists():
        raise FileNotFoundError(f"Frames directory not found: {input_dir}")
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"{out_path} exists; pass --overwrite to replace it")

    print(f"Scanning: {input_dir}  (layout=auto-detected)")
    rows = collect_frames(
        input_dir,
        check_duplicates=not args.skip_duplicate_check,
        drop_cross_clip_duplicates=args.drop_cross_clip_duplicates,
    )
    print(f"Found {len(rows):,} frames")

    if not rows:
        raise RuntimeError(
            f"No frames found under {input_dir}; expected {{label}}/{{patient}}/{{clip}}/<image> "
            "or {label}/{patient}/<image>"
        )

    link_path = video_data_path(args)
    video_data = load_video_data(link_path) if link_path else None
    if video_data is None:
        print("  [video_data] not used: patient_id = procedure folder")
    else:
        rows = link_patients(rows, video_data, dataset)
        print(f"  [video_data] {len({r['procedure_id'] for r in rows})} procedures -> "
              f"{len({r['patient_id'] for r in rows})} patients")

    labels = sorted({r["label"] for r in rows})
    print(f"Classes: {labels}  ({class_description})")

    rows, split_info = assign_splits(
        rows,
        args.val_ratio,
        args.test_ratio,
        args.seed,
        strategy=args.split_strategy,
        test_cutoff=args.test_cutoff,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = FIELDNAMES if video_data is None else LINKED_FIELDNAMES
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    split_info["source_dir"] = input_dir.as_posix()
    if video_data is not None:
        split_info["video_data_sha256"] = hashlib.sha256(link_path.read_bytes()).hexdigest()
        split_info["n_procedures"] = len({r["procedure_id"] for r in rows})
    with open(output_dir / "split_info.json", "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)
    print(f"Manifest saved: {out_path}  (+ split_info.json)")


def build_parser(
    *,
    description: str,
    input_default: str,
    output_default: str,
) -> argparse.ArgumentParser:
    """Build the CLI shared by both manifest wrappers."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input-dir", dest="input_dir", default=input_default,
                        help=f"Processed-frames directory (default: {input_default})")
    parser.add_argument("--output-dir", dest="output_dir", default=output_default,
                        help="Output manifest directory")
    parser.add_argument("--split-strategy", choices=SPLIT_STRATEGIES, required=True,
                        help="Patient-level split: stratified, temporal (needs --test-cutoff) or coverage")
    parser.add_argument("--test-cutoff", default=None,
                        help="temporal: test = patients first seen on/after this date (YYYY-MM-DD)")
    parser.add_argument("--val-ratio", type=float, default=0.15,
                        help="Validation fraction (temporal: of development patients)")
    parser.add_argument("--test-ratio", type=float, default=0.15,
                        help="Test fraction (ignored by temporal)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--video-data", default=None,
                        help="Patient linkage CSV (default: <input-dir>/../../video_data.csv "
                             "if present; 'none' to disable)")
    parser.add_argument("--skip-duplicate-check", action="store_true",
                        help="Do not md5-hash frames to report byte-identical copies")
    parser.add_argument("--drop-cross-clip-duplicates", action="store_true",
                        help="Drop copies of an image that already appears in an earlier clip")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace an existing dataset_manifest.csv")
    return parser
