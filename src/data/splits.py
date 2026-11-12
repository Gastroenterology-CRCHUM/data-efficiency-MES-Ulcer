"""
Patient-level train / val / test split strategies used by ``create_manifest``.

``temporal``
    Test = patients whose first procedure is on or after ``test_cutoff``
    (``date_of_procedure`` from ``video_data.csv``); val = ``val_ratio`` of the
    other patients, drawn at random within strata of each patient's highest grade.
``coverage``
    Greedy rarest-class-first assignment (manifest_utils.patient_level_split);
    every class appears in train and test.
``stratified`` (used for the published splits)
    Stratified on each patient's modal label with two chained
    ``train_test_split`` calls; rare strata assigned by hand. Kept verbatim:
    changing it changes the published LIMUC split.

All return the rows with a ``split`` key and a JSON-able ``split_info`` dict.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.manifest_utils import (
    SPLITS,
    _as_frame,
    patient_level_split,
    print_split_summary,
    split_summary,
)

SPLIT_STRATEGIES = ("temporal", "coverage", "stratified")
"""Patient-level split strategies selectable in ``create_manifest --split-strategy``."""


# ---------------------------------------------------------------------------
# Temporal split
# ---------------------------------------------------------------------------


def temporal_patient_split(
    rows, test_cutoff: str, val_ratio: float, seed: int, *, verbose: bool = True
) -> tuple[list[dict], dict]:
    """Hold out patients first seen on or after ``test_cutoff``; stratified random val.

    Args:
        rows:        Frame rows with ``patient_id``, ``label`` and ``date_of_procedure``.
        test_cutoff: ISO date. A patient with any earlier procedure stays in development.
        val_ratio:   Fraction of development patients in val, per highest-grade
                     stratum (at least one when the stratum has two or more patients).
        seed:        Seed of the val draw.
    """
    rows = _as_rows(rows)
    if not 0 <= val_ratio < 1:
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")
    if any(not r.get("date_of_procedure") for r in rows):
        raise ValueError("temporal split needs date_of_procedure for every frame (video_data.csv)")
    cutoff = pd.Timestamp(test_cutoff)
    first: dict[str, pd.Timestamp] = {}
    top: dict[str, int] = {}
    for r in rows:
        pid, day = str(r["patient_id"]), pd.Timestamp(r["date_of_procedure"])
        first[pid] = min(first.get(pid, day), day)
        top[pid] = max(top.get(pid, -1), int(r["label"]))
    test = {p for p, day in first.items() if day >= cutoff}
    development = sorted(set(first) - test)
    if not test or not development:
        raise ValueError(f"test_cutoff {test_cutoff} leaves an empty test or development set")

    rng = np.random.default_rng(seed)
    val: set[str] = set()
    for grade in sorted({top[p] for p in development}):
        stratum = rng.permutation([p for p in development if top[p] == grade]).tolist()
        n_val = max(1, round(val_ratio * len(stratum))) if len(stratum) >= 2 and val_ratio else 0
        val.update(stratum[:n_val])
    for r in rows:
        pid = str(r["patient_id"])
        r["split"] = "test" if pid in test else "val" if pid in val else "train"

    n = len(first)
    info = build_split_info(
        rows,
        strategy="temporal",
        train_ratio=(n - len(test) - len(val)) / n,
        val_ratio=len(val) / n,
        test_ratio=len(test) / n,
        seed=seed,
    )
    info.update(split_strategy="temporal", test_cutoff=cutoff.date().isoformat(),
                val_ratio_of_development=val_ratio)
    if verbose:
        print(f"  Temporal split: test = first procedure on/after {info['test_cutoff']}")
        print_split_summary(split_summary(rows))
    return rows, info


# ---------------------------------------------------------------------------
# Stratified split (port)
# ---------------------------------------------------------------------------


def modal_patient_labels(
    rows: list[dict], patient_col: str = "patient_id", label_col: str = "label"
) -> dict[str, str]:
    """Most frequent frame label of every patient, as a string (ties -> smallest label).

    Same result as the original per-patient ``pandas.Series.mode().iloc[0]``.
    """
    counts: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        counts[str(row[patient_col])][int(row[label_col])] += 1
    modal: dict[str, str] = {}
    for pid, cnt in counts.items():
        top = max(cnt.values())
        modal[pid] = str(min(lbl for lbl, n in cnt.items() if n == top))
    return modal


def split_with_rare_strata(
    ids: list,
    strat_labels: list[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_seed: int,
    rare_threshold: int = 3,
) -> tuple[list, list, list, str, list]:
    """Stratified 3-way split of ``ids`` with manual assignment of rare strata.

    A stratum is rare when it has fewer than
    ``max(rare_threshold, ceil(1 / test_ratio) + 1)`` members. Rare strata are
    shuffled with ``np.random.default_rng(random_seed)`` and assigned test
    first, then train, val gets the rest. Common strata go through two
    stratified ``train_test_split`` calls (unstratified fallback on
    ``ValueError``). ``train_ratio`` only matters for rare strata.

    Returns:
        (train_ids, val_ids, test_ids, strategy, rare_ids); strategy is
        ``"stratified"``, ``"partial"`` (some strata assigned by hand) or
        ``"random"`` (stratification failed).
    """
    # NOTE: verbatim port — the published LIMUC split depends on every call below.
    effective_threshold = max(rare_threshold, math.ceil(1.0 / test_ratio) + 1)
    bin_counts = Counter(strat_labels)
    common_ids, common_labels, rare_ids, rare_labels = [], [], [], []
    for id_, label in zip(ids, strat_labels):
        if bin_counts[label] >= effective_threshold:
            common_ids.append(id_)
            common_labels.append(label)
        else:
            rare_ids.append(id_)
            rare_labels.append(label)

    train_out: list = []
    val_out: list = []
    test_out: list = []

    if rare_ids:
        rng = np.random.default_rng(random_seed)
        rare_by_label: dict[str, list] = {}
        for id_, label in zip(rare_ids, rare_labels):
            rare_by_label.setdefault(label, []).append(id_)
        for label in sorted(rare_by_label):
            group = rng.permutation(rare_by_label[label]).tolist()
            n = len(group)
            if n >= 3:
                # At least 1 in each split; train gets the remainder.
                n_test = max(1, round(n * test_ratio))
                n_val = max(1, round(n * val_ratio))
                n_train = n - n_test - n_val
                if n_train < 1:
                    n_val = max(0, n_val - (1 - n_train))
                    n_train = n - n_test - n_val
            elif n == 2:
                n_test, n_val, n_train = 1, 0, 1
            else:
                n_test, n_val, n_train = 1, 0, 0
            test_out += group[:n_test]
            train_out += group[n_test : n_test + n_train]
            val_out += group[n_test + n_train :]

    strategy = "partial" if rare_ids else "stratified"
    if common_ids:
        try:
            tr_ids, vt_ids, _, vt_labels = train_test_split(
                common_ids,
                common_labels,
                test_size=(val_ratio + test_ratio),
                random_state=random_seed,
                stratify=common_labels,
            )
            relative_test = test_ratio / (val_ratio + test_ratio)
            try:
                v_ids, t_ids = train_test_split(
                    vt_ids,
                    test_size=relative_test,
                    random_state=random_seed,
                    stratify=vt_labels,
                )
            except ValueError:
                v_ids, t_ids = train_test_split(
                    vt_ids, test_size=relative_test, random_state=random_seed
                )
        except ValueError:
            strategy = "random"
            tr_ids, vt_ids = train_test_split(
                common_ids, test_size=(val_ratio + test_ratio), random_state=random_seed
            )
            relative_test = test_ratio / (val_ratio + test_ratio)
            v_ids, t_ids = train_test_split(
                vt_ids, test_size=relative_test, random_state=random_seed
            )
        train_out += list(tr_ids)
        val_out += list(v_ids)
        test_out += list(t_ids)

    return train_out, val_out, test_out, strategy, rare_ids


def stratified_patient_split(
    rows,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    *,
    train_ratio: float | None = None,
    rare_threshold: int = 3,
    patient_col: str = "patient_id",
    label_col: str = "label",
    verbose: bool = True,
) -> tuple[list[dict], dict]:
    """Patient-level split stratified on the modal patient label (LIMUC strategy).

    Args:
        rows:           Frame rows (list of dicts or DataFrame) with ``patient_col``
                        and an integer ``label_col``.
        val_ratio:      Fraction of patients for val (> 0).
        test_ratio:     Fraction of patients for test (> 0).
        seed:           ``random_state`` of the split (the LIMUC run used 42).
        train_ratio:    Recorded in split_info and used for rare strata only;
                        defaults to ``1 - val_ratio - test_ratio``.
        rare_threshold: Minimum stratum size before manual assignment
                        (see split_with_rare_strata).
        verbose:        Print the achieved split (split_summary).

    Returns:
        (rows, split_info): ``rows`` as a list of dicts with a ``split`` key
        added (dict rows are modified in place), and the split_info dict.
    """
    if not (0 < val_ratio < 1 and 0 < test_ratio < 1 and val_ratio + test_ratio < 1):
        raise ValueError(
            "the stratified split needs 0 < val_ratio, test_ratio and "
            f"val_ratio + test_ratio < 1, got val={val_ratio}, test={test_ratio}"
        )
    if train_ratio is None:
        train_ratio = 1.0 - val_ratio - test_ratio
    rows = _as_rows(rows)
    if not rows:
        raise ValueError("stratified_patient_split: no rows to split")

    modal = modal_patient_labels(rows, patient_col, label_col)
    patients = sorted(modal)
    strat_labels = [modal[pid] for pid in patients]
    train_ids, val_ids, test_ids, strategy, rare_ids = split_with_rare_strata(
        patients, strat_labels, train_ratio, val_ratio, test_ratio, seed, rare_threshold
    )

    split_map: dict[str, str] = {str(pid): "train" for pid in train_ids}
    split_map.update({str(pid): "val" for pid in val_ids})
    split_map.update({str(pid): "test" for pid in test_ids})
    for row in rows:
        row["split"] = split_map[str(row[patient_col])]

    info = build_split_info(
        rows,
        strategy=strategy,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        patient_col=patient_col,
        label_col=label_col,
    )
    info["split_strategy"] = "stratified"
    info["n_rare_patients"] = len(rare_ids)
    if verbose:
        print(f"  Stratified split: strategy={strategy}, rare-stratum patients={len(rare_ids)}")
        print_split_summary(split_summary(rows), val_ratio=val_ratio, test_ratio=test_ratio)
    return rows, info


# ---------------------------------------------------------------------------
# Dispatcher and split_info
# ---------------------------------------------------------------------------


def _as_rows(rows) -> list[dict]:
    if isinstance(rows, list):
        return rows
    return _as_frame(rows).to_dict("records")


def build_split_info(
    rows,
    *,
    strategy: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    patient_col: str = "patient_id",
    label_col: str = "label",
) -> dict:
    """``split_info.json`` content: strategy, ratios, seed and per-split counts.

    Adds ``total_frames``, ``n_patients`` and ``label_distribution`` over all rows.
    Label keys are strings (JSON object keys).
    """
    rows = _as_rows(rows)
    splits: dict = {}
    for split in SPLITS:
        sub = [r for r in rows if r.get("split") == split]
        counts = Counter(int(r[label_col]) for r in sub)
        splits[split] = {
            "n_patients": len({str(r[patient_col]) for r in sub}),
            "n_frames": len(sub),
            "label_counts": {str(k): int(counts[k]) for k in sorted(counts)},
        }
    total = Counter(int(r[label_col]) for r in rows)
    return {
        "strategy": strategy,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "random_seed": seed,
        "splits": splits,
        "total_frames": len(rows),
        "n_patients": len({str(r[patient_col]) for r in rows}),
        "label_distribution": {str(k): int(total[k]) for k in sorted(total)},
    }


def assign_splits(
    rows: list[dict],
    val_ratio: float,
    test_ratio: float,
    seed: int,
    *,
    strategy: str = "stratified",
    test_cutoff: str | None = None,
    verbose: bool = True,
) -> tuple[list[dict], dict]:
    """Assign train / val / test by patient with the selected strategy.

    ``test_ratio`` is ignored by ``temporal``, which needs ``test_cutoff`` and
    reads ``val_ratio`` as a fraction of development patients.
    Returns (rows with a ``split`` key, split_info dict).
    """
    if strategy not in SPLIT_STRATEGIES:
        raise ValueError(f"strategy must be one of {SPLIT_STRATEGIES}, got {strategy!r}")
    if strategy == "temporal":
        if not test_cutoff:
            raise ValueError("the temporal split needs test_cutoff (YYYY-MM-DD)")
        return temporal_patient_split(rows, test_cutoff, val_ratio, seed, verbose=verbose)
    if strategy == "stratified":
        return stratified_patient_split(rows, val_ratio, test_ratio, seed, verbose=verbose)

    rows = patient_level_split(rows, val_ratio, test_ratio, seed, verbose=verbose)
    info = build_split_info(
        rows,
        strategy="quota",
        train_ratio=1.0 - val_ratio - test_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    info["split_strategy"] = "coverage"
    return rows, info
