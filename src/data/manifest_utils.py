"""
Manifest creation utilities (collect frames, patient-level split, stats).

Expected raw structure (both tasks)
------------------------------------
raw_dir/
├── {label}/              <- integer label (0, 1, 2, 3, ...)
│   └── {video_id}/       <- patient video folder
│       └── {clip_id}/    <- annotated clip folder
│           └── *.jpg
└── ...
"""

from __future__ import annotations

import random
from pathlib import Path


def collect_frames(raw_dir: Path) -> list[dict]:
    """Scan raw_dir and return one row per frame.

    Top-level subdirectory names must be integers (the class label).
    """
    rows = []
    for label_dir in sorted(raw_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        try:
            label = int(label_dir.name)
        except ValueError:
            continue  # skip non-numeric folders (e.g. .gitkeep)

        for patient_dir in sorted(label_dir.iterdir()):
            if not patient_dir.is_dir():
                continue
            patient_id = patient_dir.name

            for clip_dir in sorted(patient_dir.iterdir()):
                if not clip_dir.is_dir():
                    continue
                clip_id = f"{patient_id}__{clip_dir.name}"

                for img in sorted(clip_dir.glob("*.jpg")):
                    filepath = img.relative_to(raw_dir)
                    rows.append({
                        "filepath": str(filepath),
                        "label": label,
                        "patient_id": patient_id,
                        "video_id": clip_id,
                    })
    return rows


def patient_level_split(
    rows: list[dict],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> list[dict]:
    """Stratified patient-level split (greedy, rarest-class-first).

    Each patient is assigned to exactly one split (train / val / test).
    Classes are processed from rarest to most common. For each class,
    unassigned patients are first used to guarantee coverage of test,
    then train, then val — in that priority order. Remaining patients
    are distributed by ratio.

    Note: with only 1 or 2 patients in a class it is impossible to cover
    all three splits; a warning is printed when coverage is incomplete.
    """
    from collections import defaultdict

    # Frames per (patient, label)
    patient_class_counts: dict = defaultdict(lambda: defaultdict(int))
    for row in rows:
        patient_class_counts[row["patient_id"]][row["label"]] += 1

    all_classes = sorted({row["label"] for row in rows})
    all_patients = sorted(patient_class_counts)

    # All patients that have at least one frame of each class
    class_patients: dict = defaultdict(set)
    for pid, counts in patient_class_counts.items():
        for lbl in counts:
            class_patients[lbl].add(pid)

    patient_to_split: dict = {}

    # Process rarest classes first so they get first pick of unassigned patients
    for label in sorted(all_classes, key=lambda c: len(class_patients[c])):
        unassigned = sorted(p for p in class_patients[label] if p not in patient_to_split)

        def count_in(split: str) -> int:
            return sum(1 for p in class_patients[label] if patient_to_split.get(p) == split)

        rng = random.Random(seed ^ (label * 0x9E3779B9))
        rng.shuffle(unassigned)
        idx = 0

        # Priority: test → train → val
        for target in ("test", "train", "val"):
            if count_in(target) == 0 and idx < len(unassigned):
                patient_to_split[unassigned[idx]] = target
                idx += 1

        # Distribute remaining unassigned patients by ratio
        remaining = unassigned[idx:]
        n = len(remaining)
        n_test = round(n * test_ratio)
        n_val = round(n * val_ratio)
        for pid in remaining[:n_test]:
            patient_to_split[pid] = "test"
        for pid in remaining[n_test:n_test + n_val]:
            patient_to_split[pid] = "val"
        for pid in remaining[n_test + n_val:]:
            patient_to_split[pid] = "train"

    # Assign all remaining patients (those whose classes were already covered)
    unassigned_rest = [p for p in all_patients if p not in patient_to_split]
    rng_main = random.Random(seed)
    rng_main.shuffle(unassigned_rest)
    n = len(unassigned_rest)
    n_test = round(n * test_ratio)
    n_val = round(n * val_ratio)
    for pid in unassigned_rest[:n_test]:
        patient_to_split[pid] = "test"
    for pid in unassigned_rest[n_test:n_test + n_val]:
        patient_to_split[pid] = "val"
    for pid in unassigned_rest[n_test + n_val:]:
        patient_to_split[pid] = "train"

    for row in rows:
        row["split"] = patient_to_split[row["patient_id"]]

    # Warn if any class is absent from a split
    for label in all_classes:
        for split in ("train", "val", "test"):
            has = any(r["label"] == label and r["split"] == split for r in rows)
            if not has:
                n_patients = len(class_patients[label])
                print(
                    f"  WARNING: class {label} absent from '{split}' "
                    f"(only {n_patients} patient(s) total — impossible to cover all 3 splits)"
                )

    return rows


def print_split_stats(rows: list[dict]) -> None:
    """Print frame counts, class distribution and patient counts per split."""
    labels = sorted({r["label"] for r in rows})
    for split in ("train", "val", "test"):
        subset = [r for r in rows if r["split"] == split]
        n_patients = len({r["patient_id"] for r in subset})
        class_counts = {lbl: sum(1 for r in subset if r["label"] == lbl) for lbl in labels}
        dist = "  ".join(f"[{k}]={v:,}" for k, v in class_counts.items() if v > 0)
        print(f"  {split:5s}: {len(subset):6,} frames | {n_patients:2d} patients | {dist}")
