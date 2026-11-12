"""
Manifest creation utilities (collect frames, duplicate check, patient-level split, stats).

Expected frames structure (both tasks, e.g. ``data/{task}/processed/``)
----------------------------------------------------------------------
frames_dir/
├── {label}/              <- integer label (0, 1, 2, 3, ...) or ``score_{label}``
│   └── {video_id}/       <- patient video folder
│       └── {clip_id}/    <- annotated clip folder
│           └── *.jpg | *.jpeg | *.png | *.bmp
└── ...

A flat 3-level layout ``{label}/{patient_id}/<image>`` (no clip folders, e.g.
LIMUC) is also accepted: each image then forms its own single-frame clip
``{patient_id}__{image stem}``. Hidden entries (names starting with ``.``,
e.g. ``.ipynb_checkpoints``) are skipped at every level.

Manifest columns
----------------
filepath      frame path relative to ``frames_dir``, always POSIX ("/") separators
label         integer class label
split         train | val | test
patient_id    patient pseudonym from ``video_data.csv``, else the procedure folder
video_id      clip key ``{procedure}__{clip_id}``
procedure_id  procedure (video) folder; written only with ``video_data.csv``
"""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd

LINK_REQUIRED = ("video_id", "patient_id", "dataset")
"""Required ``video_data.csv`` columns (optional: date_of_procedure, age, gender)."""
_RAW_HOSPITAL_ID = r"\d{5,7}"

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp"})
"""Frame file extensions collected into manifests (matched case-insensitively)."""

SPLITS = ("train", "val", "test")
MIN_TRAIN_PATIENTS = 2
"""Per-class train-patient floor: the quota split never drafts a patient into
val/test beyond coverage if that leaves one of its classes with fewer train
patients, and split_summary warns when a class has fewer."""
MIN_EVAL_PATIENTS = 2
"""split_summary warns when a class has fewer val / test patients than this."""
# ---------------------------------------------------------------------------
# Manifest normalisation (shared by the dataloader and the dataset)
# ---------------------------------------------------------------------------


def normalize_manifest(df: pd.DataFrame) -> pd.DataFrame:
    """Copy with ``relative_path`` (from ``filepath``, ``/`` separators) and ``clip_key``.

    Idempotent; used by the dataloader and the dataset.
    """
    df = df.copy()
    if "relative_path" not in df.columns and "filepath" in df.columns:
        df = df.rename(columns={"filepath": "relative_path"})
    if "relative_path" in df.columns:
        df["relative_path"] = df["relative_path"].astype(str).str.replace("\\", "/", regex=False)
    if "clip_key" not in df.columns and "video_id" in df.columns:
        df["clip_key"] = df["video_id"]
    return df


def load_manifest(manifest_path: Path | str) -> pd.DataFrame:
    """Read a manifest CSV and return it normalised (see normalize_manifest)."""
    return normalize_manifest(pd.read_csv(manifest_path))


def _as_frame(df_or_rows) -> pd.DataFrame:
    return df_or_rows if isinstance(df_or_rows, pd.DataFrame) else pd.DataFrame(list(df_or_rows))


# ---------------------------------------------------------------------------
# Frame collection
# ---------------------------------------------------------------------------


def _parse_label_dir(name: str) -> int | None:
    """Label from a class folder name: ``"2"`` or ``"score_2"``; None otherwise."""
    if name.startswith("score_"):
        name = name[len("score_") :]
    try:
        return int(name)
    except ValueError:
        return None


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def _is_hidden(path: Path) -> bool:
    """Dot-entries (``.ipynb_checkpoints``, ``.DS_Store``, ``._x.jpg``) are never frames."""
    return path.name.startswith(".")


def _frame_row(img: Path, raw_dir: Path, label: int, patient_id: str, clip_id: str) -> dict:
    return {
        "filepath": img.relative_to(raw_dir).as_posix(),
        "label": label,
        "patient_id": patient_id,
        "video_id": clip_id,
    }


def collect_frames(
    raw_dir: Path,
    *,
    check_duplicates: bool = True,
    drop_cross_clip_duplicates: bool = False,
) -> list[dict]:
    """Scan ``raw_dir`` (the processed frames folder) and return one row per frame.

    Top-level subdirectory names must be integers or ``score_{int}`` (the class
    label); other entries are skipped. Only image files (``IMAGE_EXTENSIONS``,
    case-insensitive) are collected, so the same folder yields the same
    manifest on every OS. ``filepath`` is written with POSIX separators.
    Hidden entries (``.ipynb_checkpoints``, ``.DS_Store``, ...) are skipped.

    Args:
        raw_dir:                    Frames root (e.g. ``data/mes/processed``).
        check_duplicates:           Hash every frame (md5) and warn about
                                    byte-identical copies (see find_duplicate_frames).
        drop_cross_clip_duplicates: Drop copies of an image that already appears
                                    in an earlier clip (manifest order). Within-clip
                                    repeats are always kept. Off by default.
    Returns:
        list of dicts with keys filepath, label, patient_id, video_id.
    """
    raw_dir = Path(raw_dir)
    rows: list[dict] = []
    n_hidden = 0
    for label_dir in sorted(raw_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        label = _parse_label_dir(label_dir.name)
        if label is None:
            continue  # skip non-label folders (e.g. .gitkeep)

        for patient_dir in sorted(label_dir.iterdir()):
            if not patient_dir.is_dir():
                continue
            if _is_hidden(patient_dir):
                n_hidden += 1
                continue
            patient_id = patient_dir.name

            for entry in sorted(patient_dir.iterdir()):
                if _is_hidden(entry):
                    n_hidden += 1
                elif entry.is_dir():
                    clip_id = f"{patient_id}__{entry.name}"
                    for img in sorted(entry.iterdir()):
                        if _is_hidden(img):
                            n_hidden += 1
                        elif _is_image(img):
                            rows.append(_frame_row(img, raw_dir, label, patient_id, clip_id))
                elif _is_image(entry):
                    # Flat layout: the image is its own single-frame clip.
                    clip_id = f"{patient_id}__{entry.stem}"
                    rows.append(_frame_row(entry, raw_dir, label, patient_id, clip_id))

    if n_hidden:
        print(
            f"  Skipped {n_hidden} hidden entr{'y' if n_hidden == 1 else 'ies'} (names starting with '.')"
        )

    if not rows:
        print(f"  WARNING: no image frames found under '{raw_dir.name}'")
        return rows

    # A clip key shared by several labels means two different frames collapse
    # into one clip (e.g. same file stem under two class folders).
    clip_labels: dict = defaultdict(set)
    for row in rows:
        clip_labels[row["video_id"]].add(row["label"])
    n_conflicts = sum(1 for labels in clip_labels.values() if len(labels) > 1)
    if n_conflicts:
        print(f"  WARNING: {n_conflicts} clip key(s) contain frames of more than one label")

    if check_duplicates or drop_cross_clip_duplicates:
        dups = find_duplicate_frames(pd.DataFrame(rows), raw_dir)
        report_duplicate_frames(dups)
        if drop_cross_clip_duplicates and not dups.empty:
            drop_idx = set(dups.index[dups["is_cross_clip_copy"]])
            rows = [r for i, r in enumerate(rows) if i not in drop_idx]
            print(f"  Dropped {len(drop_idx)} cross-clip duplicate frame(s)")
    return rows


# ---------------------------------------------------------------------------
# Patient linkage (video_data.csv)
# ---------------------------------------------------------------------------


def _few(items, n: int = 5) -> str:
    items = sorted(items)
    return ", ".join(items[:n]) + (" ..." if len(items) > n else "")


def load_video_data(path: Path | str) -> pd.DataFrame | None:
    """Procedure-to-patient table indexed by ``video_id``; None if ``path`` is absent.

    Exact duplicate rows are dropped. Missing required columns or values,
    conflicting rows for one procedure and raw hospital numbers raise ValueError.
    """
    path = Path(path)
    if not path.is_file():
        return None
    df = pd.read_csv(path, dtype=str)
    missing = [c for c in LINK_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing required column(s) {missing}")
    if df[list(LINK_REQUIRED)].isna().any().any():
        raise ValueError(f"{path.name}: empty values in {list(LINK_REQUIRED)}")
    n_dup = int(df.duplicated().sum())
    if n_dup:
        print(f"  [video_data] dropped {n_dup} exact duplicate row(s)")
        df = df.drop_duplicates()
    conflicts = df.loc[df["video_id"].duplicated(keep=False), "video_id"]
    if len(conflicts):
        raise ValueError(f"{path.name}: conflicting rows for {_few(set(conflicts))}")
    if df["patient_id"].str.fullmatch(_RAW_HOSPITAL_ID).any():
        raise ValueError(f"{path.name}: patient_id looks like raw hospital numbers")
    return df.set_index("video_id")


def link_patients(rows: list[dict], video_data: pd.DataFrame, dataset: str) -> list[dict]:
    """Set ``patient_id`` from ``video_data`` (in place); the folder moves to ``procedure_id``.

    Copies ``date_of_procedure`` when present. Raises ValueError for a procedure
    without a row or whose ``dataset`` is not ``dataset``.
    """
    procedures = {r["patient_id"] for r in rows}
    missing = procedures - set(video_data.index)
    if missing:
        raise ValueError(f"no video_data.csv row for {len(missing)} procedure(s): {_few(missing)}")
    wrong = {p for p in procedures if video_data.at[p, "dataset"] != dataset}
    if wrong:
        raise ValueError(f"{len(wrong)} procedure(s) are not '{dataset}' in video_data.csv: {_few(wrong)}")
    has_date = "date_of_procedure" in video_data.columns
    for r in rows:
        procedure = r["patient_id"]
        r["procedure_id"] = procedure
        r["patient_id"] = video_data.at[procedure, "patient_id"]
        if has_date:
            r["date_of_procedure"] = video_data.at[procedure, "date_of_procedure"]
    return rows


# ---------------------------------------------------------------------------
# Duplicate frames
# ---------------------------------------------------------------------------


def _md5_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def find_duplicate_frames(df, root: Path) -> pd.DataFrame:
    """Return the manifest rows whose image bytes are identical to another row.

    Args:
        df:   Manifest DataFrame (or list of row dicts) with ``filepath`` or
              ``relative_path``, ``label``, ``patient_id`` and ``video_id``/``clip_key``.
        root: Directory the frame paths are relative to.

    Returns:
        DataFrame (original index kept, normalised columns) restricted to rows
        in a duplicate group, sorted by hash, with extra columns:
            md5, dup_group_size, dup_n_clips, dup_n_patients, dup_n_labels,
            dup_n_splits (if a ``split`` column exists) and
            is_cross_clip_copy — True for rows whose image already appears in
            an earlier clip of the manifest (the copies a de-duplication drops).
        Files missing on disk are skipped (with a warning).
    """
    frame = normalize_manifest(_as_frame(df))
    root = Path(root)
    hashes: list[str | None] = []
    n_missing = 0
    for rel in frame["relative_path"]:
        try:
            hashes.append(_md5_file(root / rel))
        except FileNotFoundError:
            hashes.append(None)
            n_missing += 1
    if n_missing:
        print(f"  WARNING: find_duplicate_frames: {n_missing} file(s) missing - not hashed")

    frame["md5"] = hashes
    frame = frame.dropna(subset=["md5"])
    dups = frame[frame.groupby("md5")["md5"].transform("size") > 1].copy()
    extra = ["dup_group_size", "dup_n_clips", "dup_n_patients", "dup_n_labels"]
    if "split" in dups.columns:
        extra.append("dup_n_splits")
    if dups.empty:
        for col in extra:
            dups[col] = pd.Series(dtype=int)
        dups["is_cross_clip_copy"] = pd.Series(dtype=bool)
        return dups

    g = dups.groupby("md5", sort=False)
    dups["dup_group_size"] = g["md5"].transform("size")
    dups["dup_n_clips"] = g["clip_key"].transform("nunique")
    dups["dup_n_patients"] = g["patient_id"].transform("nunique")
    dups["dup_n_labels"] = g["label"].transform("nunique")
    if "split" in dups.columns:
        dups["dup_n_splits"] = g["split"].transform("nunique")
    dups["is_cross_clip_copy"] = dups["clip_key"] != g["clip_key"].transform("first")
    return dups.sort_values("md5", kind="stable")


def report_duplicate_frames(dups: pd.DataFrame, *, max_pairs: int = 10) -> dict:
    """Print a summary of find_duplicate_frames() output and return the counts."""
    if dups.empty:
        stats = {"n_duplicate_rows": 0, "n_distinct_images": 0, "n_excess_rows": 0}
        print("  Duplicate check: no byte-identical frames")
        return stats

    groups = dups.groupby("md5")
    stats = {
        "n_duplicate_rows": int(len(dups)),
        "n_distinct_images": int(groups.ngroups),
        "n_excess_rows": int(len(dups) - groups.ngroups),
        "n_cross_clip_copies": int(dups["is_cross_clip_copy"].sum()),
        "n_groups_cross_clip": int((groups["dup_n_clips"].first() > 1).sum()),
        "n_groups_cross_patient": int((groups["dup_n_patients"].first() > 1).sum()),
        "n_groups_cross_label": int((groups["dup_n_labels"].first() > 1).sum()),
    }
    if "dup_n_splits" in dups.columns:
        stats["n_groups_cross_split"] = int((groups["dup_n_splits"].first() > 1).sum())

    print(
        f"  WARNING: {stats['n_duplicate_rows']} frame(s) are byte-identical copies "
        f"({stats['n_distinct_images']} distinct images, {stats['n_excess_rows']} excess rows; "
        f"{stats['n_cross_clip_copies']} copies in a different clip)"
    )
    if stats["n_groups_cross_patient"]:
        print(
            f"  WARNING: {stats['n_groups_cross_patient']} duplicate image(s) shared by "
            "different patients - possible split leakage"
        )
    if stats["n_groups_cross_label"]:
        print(
            f"  WARNING: {stats['n_groups_cross_label']} duplicate image(s) carry "
            "conflicting labels"
        )
    if stats.get("n_groups_cross_split"):
        print(f"  WARNING: {stats['n_groups_cross_split']} duplicate image(s) span splits")

    cross = dups[dups["dup_n_clips"] > 1]
    if not cross.empty:
        pairs = (
            cross.groupby("md5")["clip_key"]
            .agg(lambda s: tuple(dict.fromkeys(s)))
            .value_counts()
            .head(max_pairs)
        )
        print("  Clips sharing identical frames (clips: n shared images):")
        for clips, n in pairs.items():
            print(f"    {' <-> '.join(clips)}: {n}")
    return stats


# ---------------------------------------------------------------------------
# Patient-level split
# ---------------------------------------------------------------------------


def patient_level_split(
    rows: list[dict],
    val_ratio: float,
    test_ratio: float,
    seed: int,
    *,
    verbose: bool = True,
) -> list[dict]:
    """Stratified patient-level split (greedy, rarest-class-first).

    Each patient is assigned to exactly one split (train / val / test).

    The quota algorithm fixes global patient targets first:
        ``n_test = round(N * test_ratio)``
        and ``n_val = round(N * val_ratio)`` (N = number of patients). Classes
        are processed from rarest to most common; for each class, unassigned
        patients first guarantee coverage of test, then train, then val (a
        split with ratio 0 gets no coverage patient and stays empty). Coverage
        assignments count toward the global targets (coverage wins if the two
        conflict). Each class then receives up to ``round(n_class * ratio)``
        test/val patients while the global targets allow. Remaining patients
        fill the remaining test/val targets (test first); the rest go to train.

        Train floor: beyond coverage, a patient is never moved to val/test if
        that would leave one of its classes with fewer than
        ``MIN_TRAIN_PATIENTS`` train patients (the floor wins over the global
        targets, which may then be missed). Val/test coverage patients are
        chosen to break the floor of as few classes as possible, so the
        leftover of a rarer class is not pulled out of train to cover a more
        common one. Rare-class leftovers therefore stay in train (a class of 4
        patients ends up 2 / 1 / 1). Among admissible val/test candidates, the
        one bringing the most classes still short of
        ``max(round(n_class * ratio), MIN_EVAL_PATIENTS)`` patients in that
        split is preferred; remaining ties follow a seeded shuffle.

    With ``verbose`` the achieved patient / frame split per class is printed
    (see split_summary) with warnings when a class is absent from a split, has
    fewer than ``MIN_TRAIN_PATIENTS`` patients in train or fewer than
    ``MIN_EVAL_PATIENTS`` patients in val or test.

    Returns:
        ``rows`` with a ``split`` key added to every row (modified in place).
    """
    if not (0 <= val_ratio < 1 and 0 <= test_ratio < 1 and val_ratio + test_ratio < 1):
        raise ValueError(
            f"need 0 <= val_ratio, test_ratio and val_ratio + test_ratio < 1, "
            f"got val={val_ratio}, test={test_ratio}"
        )

    rows = _patient_level_split_quota(rows, val_ratio, test_ratio, seed)

    if verbose:
        print_split_summary(split_summary(rows), val_ratio=val_ratio, test_ratio=test_ratio)
    return rows


def _class_index(rows: list[dict]) -> tuple[list, list, dict]:
    patient_class_counts: dict = defaultdict(lambda: defaultdict(int))
    for row in rows:
        patient_class_counts[row["patient_id"]][row["label"]] += 1
    all_classes = sorted({row["label"] for row in rows})
    all_patients = sorted(patient_class_counts)
    class_patients: dict = defaultdict(set)
    for pid, counts in patient_class_counts.items():
        for lbl in counts:
            class_patients[lbl].add(pid)
    return all_classes, all_patients, class_patients


def _patient_level_split_quota(
    rows: list[dict], val_ratio: float, test_ratio: float, seed: int
) -> list[dict]:
    """Quota split with a per-class train floor (see patient_level_split)."""
    all_classes, all_patients, class_patients = _class_index(rows)
    ratio = {"test": test_ratio, "val": val_ratio}
    target = {s: int(round(len(all_patients) * r)) for s, r in ratio.items()}
    coverage_order = [s for s in ("test", "train", "val") if s == "train" or ratio[s] > 0]

    patient_classes: dict = defaultdict(set)
    for lbl, pids in class_patients.items():
        for pid in pids:
            patient_classes[pid].add(lbl)

    patient_to_split: dict = {}
    n_assigned: dict = defaultdict(int)  # split -> patients
    count_in: dict = {s: defaultdict(int) for s in SPLITS}  # split -> class -> patients
    # Patients per class that are in train or still unassigned (= future train).
    train_pool = {lbl: len(pids) for lbl, pids in class_patients.items()}

    def assign(pid, split: str) -> None:
        patient_to_split[pid] = split
        n_assigned[split] += 1
        for lbl in patient_classes[pid]:
            count_in[split][lbl] += 1
            if split != "train":
                train_pool[lbl] -= 1

    # Val/test patients wanted per class: its ratio share, but at least
    # MIN_EVAL_PATIENTS (only used to rank candidates, never forces a draft).
    need = {
        split: {
            lbl: max(int(round(len(class_patients[lbl]) * ratio[split])), MIN_EVAL_PATIENTS)
            for lbl in all_classes
        }
        for split in ("test", "val")
    }

    def n_floor_breaks(pid) -> int:
        # Classes of pid left below the train floor if pid leaves train.
        return sum(1 for lbl in patient_classes[pid] if train_pool[lbl] - 1 < MIN_TRAIN_PATIENTS)

    def pick(candidates: list, split: str):
        # Fewest train-floor breaks, then most classes still short of patients
        # in `split`; min() keeps the first minimum, i.e. the earliest in the
        # seeded shuffle order, on ties.
        return min(
            candidates,
            key=lambda p: (
                n_floor_breaks(p),
                -sum(1 for lbl in patient_classes[p] if count_in[split][lbl] < need[split][lbl]),
            ),
        )

    # Process rarest classes first so they get first pick of unassigned patients
    for label in sorted(all_classes, key=lambda c: len(class_patients[c])):
        pool = sorted(p for p in class_patients[label] if p not in patient_to_split)
        rng = random.Random(seed ^ (label * 0x9E3779B9))
        rng.shuffle(pool)

        # 1. Coverage (test → train → val); counts toward the global targets
        #    and wins over the train floor. Train coverage takes the next
        #    patient; val/test coverage is ranked by pick(), so a leftover of
        #    an already-covered rarer class is not pulled out of train to
        #    cover this one when another patient can.
        for split in coverage_order:
            if count_in[split][label] > 0 or not pool:
                continue
            pid = pool[0] if split == "train" else pick(pool, split)
            assign(pid, split)
            pool.remove(pid)

        # 2. Per-class share of test / val, capped by the global targets and
        #    by the train floor (patients that would break it are skipped).
        n_class = len(class_patients[label])
        for split in ("test", "val"):
            if ratio[split] == 0:
                continue
            wanted = int(round(n_class * ratio[split]))
            while count_in[split][label] < wanted and n_assigned[split] < target[split]:
                candidates = [p for p in pool if n_floor_breaks(p) == 0]
                if not candidates:
                    break
                pid = pick(candidates, split)
                assign(pid, split)
                pool.remove(pid)
        # Remaining patients of this class go to the global pool below.

    # 3. Fill the remaining global targets (test first). Only patients that
    #    keep the train floor are drafted, so rare-class leftovers stay in
    #    train; the floor wins over the targets. Everything else is train.
    rest = [p for p in all_patients if p not in patient_to_split]
    random.Random(seed).shuffle(rest)
    for split in ("test", "val"):
        while n_assigned[split] < target[split]:
            candidates = [p for p in rest if n_floor_breaks(p) == 0]
            if not candidates:
                break
            pid = pick(candidates, split)
            assign(pid, split)
            rest.remove(pid)
    for pid in rest:
        assign(pid, "train")

    for row in rows:
        row["split"] = patient_to_split[row["patient_id"]]
    return rows


def _label_key(label) -> str:
    try:
        return str(int(label))
    except (TypeError, ValueError):
        return str(label)


def _n_patients_text(n: int, one: str = "1") -> str:
    """'1 patient' (or '<one> patient') for n == 1, else 'n patients'."""
    return f"{one} patient" if n == 1 else f"{n} patients"


def split_summary(df, label_col: str = "label") -> dict:
    """Patients / clips / frames per split and per class (JSON-serialisable).

    Args:
        df: Manifest DataFrame (or list of row dicts) with ``split``,
            ``patient_id``, ``label_col`` and ``video_id``/``clip_key``.

    Returns:
        dict with keys
            n_patients, n_clips, n_frames        {split: int}
            patient_fraction, frame_fraction     {split: float}
            per_class   {label: {split: {"patients", "clips", "frames"}}}
            n_patients_in_multiple_splits        int (should be 0)
            warnings    list[str] — class absent from a split, fewer than
                        MIN_TRAIN_PATIENTS patients of a class in train or
                        MIN_EVAL_PATIENTS in val/test, patient overlap.
        The three standard splits are always present (0 when empty).
    """
    frame = normalize_manifest(_as_frame(df))
    present = set(frame["split"].astype(str))
    splits = list(SPLITS) + sorted(present - set(SPLITS))
    n_patients_total = max(int(frame["patient_id"].nunique()), 1)
    n_frames_total = max(len(frame), 1)

    summary: dict = {
        "n_patients": {},
        "n_clips": {},
        "n_frames": {},
        "patient_fraction": {},
        "frame_fraction": {},
        "per_class": {},
        "n_patients_in_multiple_splits": 0,
        "warnings": [],
    }
    for split in splits:
        sub = frame[frame["split"] == split]
        n_pat = int(sub["patient_id"].nunique())
        summary["n_patients"][split] = n_pat
        summary["n_clips"][split] = int(sub["clip_key"].nunique())
        summary["n_frames"][split] = int(len(sub))
        summary["patient_fraction"][split] = n_pat / n_patients_total
        summary["frame_fraction"][split] = len(sub) / n_frames_total

    for label in sorted(frame[label_col].unique()):
        key = _label_key(label)
        per: dict = {}
        for split in splits:
            sub = frame[(frame["split"] == split) & (frame[label_col] == label)]
            per[split] = {
                "patients": int(sub["patient_id"].nunique()),
                "clips": int(sub["clip_key"].nunique()),
                "frames": int(len(sub)),
            }
        summary["per_class"][key] = per

        for split in SPLITS:
            if split not in present:
                continue  # split not requested (e.g. val_ratio == 0)
            n_pat = per[split]["patients"]
            if n_pat == 0:
                summary["warnings"].append(f"class {key} absent from '{split}'")
            elif split == "train" and n_pat < MIN_TRAIN_PATIENTS:
                summary["warnings"].append(
                    f"class {key} has only {_n_patients_text(n_pat)} in 'train': "
                    f"the model learns this class from {_n_patients_text(n_pat, 'a single')}"
                )
            elif split in ("val", "test") and n_pat < MIN_EVAL_PATIENTS:
                summary["warnings"].append(
                    f"class {key} has only {_n_patients_text(n_pat)} in '{split}': "
                    f"its per-class metrics rest on {_n_patients_text(n_pat, 'a single')}"
                )

    n_multi = int((frame.groupby("patient_id")["split"].nunique() > 1).sum())
    summary["n_patients_in_multiple_splits"] = n_multi
    if n_multi:
        summary["warnings"].append(f"{n_multi} patient(s) appear in more than one split")
    return summary


def print_split_summary(
    summary: dict, *, val_ratio: float | None = None, test_ratio: float | None = None
) -> None:
    """Pretty-print split_summary(): achieved vs requested split, per class table."""
    splits = [s for s in summary["n_frames"] if s in SPLITS or summary["n_frames"][s]]
    if val_ratio is not None and test_ratio is not None:
        req = {"train": 1 - val_ratio - test_ratio, "val": val_ratio, "test": test_ratio}
        print("  Requested patients train/val/test: " + " / ".join(f"{req[s]:.0%}" for s in SPLITS))
    print(f"  {'split':<6}{'patients':>14}{'frames':>18}{'clips':>8}")
    for split in splits:
        print(
            f"  {split:<6}"
            f"{summary['n_patients'][split]:>6} ({summary['patient_fraction'][split]:>5.1%})"
            f"{summary['n_frames'][split]:>9,} ({summary['frame_fraction'][split]:>5.1%})"
            f"{summary['n_clips'][split]:>8}"
        )
    print("  Per class: patients / frames:")
    print("  " + f"{'class':<7}" + "".join(f"{s:>16}" for s in splits))
    for key, per in summary["per_class"].items():
        cells = "".join(
            f"{per[s]['patients']:>5} / {per[s]['frames']:<8,}".rjust(16) for s in splits
        )
        print(f"  {key:<7}{cells}")
    for msg in summary["warnings"]:
        print(f"  WARNING: {msg}")
