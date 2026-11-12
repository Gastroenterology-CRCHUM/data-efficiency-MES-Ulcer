"""Patient-level train / val / test splits (manifest_utils.patient_level_split, src.data.splits)."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict

import pandas as pd
import pytest

from src.data.manifest_utils import (
    MIN_TRAIN_PATIENTS,
    SPLITS,
    patient_level_split,
    split_summary,
)
from src.data.splits import (
    SPLIT_STRATEGIES,
    assign_splits,
    build_split_info,
    modal_patient_labels,
    split_with_rare_strata,
    stratified_patient_split,
)
from tests.conftest import make_rows, rows_with_labels


def _patient_splits(rows: list[dict]) -> dict[str, set[str]]:
    splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        splits[row["patient_id"]].add(row["split"])
    return splits


def _patients_in(rows: list[dict], split: str) -> set[str]:
    return {r["patient_id"] for r in rows if r["split"] == split}


# ---------------------------------------------------------------------------
# Coverage split (patient_level_split)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 7, 42])
def test_no_patient_overlap(seed):
    rows = make_rows(40, 4, seed=seed)
    out = patient_level_split(rows, 0.15, 0.15, seed, verbose=False)

    assert out is rows  # modified in place, order kept
    assert all(r["split"] in SPLITS for r in rows)
    assert all(len(s) == 1 for s in _patient_splits(rows).values())
    assert split_summary(rows)["n_patients_in_multiple_splits"] == 0


def test_quota_hits_requested_patient_counts_exactly():
    # 60 single-clip patients, classes of 30 / 20 / 10 patients: every quota is reachable.
    rows = rows_with_labels({0: 30, 1: 20, 2: 10})
    patient_level_split(rows, 0.15, 0.15, seed=42, verbose=False)

    counts = {s: len(_patients_in(rows, s)) for s in SPLITS}
    assert counts == {"train": 42, "val": 9, "test": 9}


@pytest.mark.parametrize("seed", [1, 2, 3, 42])
@pytest.mark.parametrize(("val_ratio", "test_ratio"), [(0.15, 0.15), (0.1, 0.2), (0.2, 0.1)])
def test_quota_patient_ratios_within_tolerance(seed, val_ratio, test_ratio):
    rows = make_rows(80, 4, seed=seed)
    n_patients = len({r["patient_id"] for r in rows})
    patient_level_split(rows, val_ratio, test_ratio, seed, verbose=False)

    summary = split_summary(rows)
    # Coverage assignments can exceed a target by at most one patient per class
    # (4 classes), and the target itself is round(N * ratio).
    tol = (4 + 0.5) / n_patients
    assert abs(summary["patient_fraction"]["val"] - val_ratio) <= tol
    assert abs(summary["patient_fraction"]["test"] - test_ratio) <= tol
    assert sum(summary["n_patients"].values()) == n_patients


def test_every_class_covered_in_every_split():
    # >= 3 patients per class -> each class reaches train, val and test.
    rows = rows_with_labels({0: 12, 1: 3, 2: 5, 3: 20})
    patient_level_split(rows, 0.15, 0.15, seed=11, verbose=False)

    summary = split_summary(rows)
    for label, per_split in summary["per_class"].items():
        for split in SPLITS:
            assert per_split[split]["patients"] >= 1, (label, split)
    assert not [w for w in summary["warnings"] if "absent" in w]


def test_zero_val_ratio_leaves_val_empty():
    rows = make_rows(30, 3, seed=4)
    patient_level_split(rows, 0.0, 0.2, seed=4, verbose=False)

    assert not _patients_in(rows, "val")
    summary = split_summary(rows)
    for per_split in summary["per_class"].values():
        assert per_split["train"]["patients"] >= 1
        assert per_split["test"]["patients"] >= 1
    # 'val' was not requested, so its emptiness is not a warning.
    assert not [w for w in summary["warnings"] if "'val'" in w]


def test_deterministic_and_row_order_independent():
    rows = make_rows(50, 4, seed=9)
    first = patient_level_split([dict(r) for r in rows], 0.15, 0.15, seed=3, verbose=False)
    again = patient_level_split([dict(r) for r in rows], 0.15, 0.15, seed=3, verbose=False)
    shuffled = [dict(r) for r in rows]
    random.Random(0).shuffle(shuffled)
    shuffled = patient_level_split(shuffled, 0.15, 0.15, seed=3, verbose=False)

    def mapping(out):
        return {r["patient_id"]: r["split"] for r in out}

    assert mapping(first) == mapping(again) == mapping(shuffled)
    other = patient_level_split([dict(r) for r in rows], 0.15, 0.15, seed=4, verbose=False)
    assert mapping(other) != mapping(first)


def test_warns_when_a_class_has_fewer_than_two_patients(capsys):
    # Class 3 has only 2 patients: one goes to test, one to train, none to val.
    rows = rows_with_labels({0: 20, 1: 10, 2: 6, 3: 2})
    patient_level_split(rows, 0.15, 0.15, seed=0, verbose=True)
    printed = capsys.readouterr().out

    summary = split_summary(rows)
    per3 = summary["per_class"]["3"]
    assert per3["test"]["patients"] == 1 and per3["train"]["patients"] == 1
    assert per3["val"]["patients"] == 0
    assert "class 3 absent from 'val'" in summary["warnings"]
    assert any(w.startswith("class 3 has only 1 patient in 'test'") for w in summary["warnings"])
    assert any(w.startswith("class 3 has only 1 patient in 'train'") for w in summary["warnings"])
    assert "WARNING: class 3 absent from 'val'" in printed


def test_single_patient_class_goes_to_test():
    rows = rows_with_labels({0: 20, 1: 1})
    patient_level_split(rows, 0.15, 0.15, seed=0, verbose=False)
    assert split_summary(rows)["per_class"]["1"]["test"]["patients"] == 1


# Class sets of 21 synthetic patients shaped like the in-house MES data: Mayo 2
# and 3 are held by only 5 and 4 patients, mostly next to other grades.
_MES_LIKE_PATIENTS = [(0, 2, 3), (0, 2), (0, 1), (0, 1), (0, 3), (0, 1, 2), (1, 3), (0, 1)]
_MES_LIKE_PATIENTS += [(0, 1), (2, 3), (0, 2), (0, 1), (1,)] + [(0,)] * 8


def _rows_from_label_sets(label_sets: list[tuple[int, ...]], frames: int = 3) -> list[dict]:
    """One clip per (patient, label): ``label_sets[i]`` are the classes of patient i."""
    rows: list[dict] = []
    for i, labels in enumerate(label_sets):
        pid = f"p{i:03d}"
        for j, label in enumerate(labels):
            for k in range(frames):
                rows.append(
                    {
                        "filepath": f"{label}/{pid}/c{j}/f{k:03d}.png",
                        "label": label,
                        "patient_id": pid,
                        "video_id": f"{pid}__c{j}",
                    }
                )
    return rows


@pytest.mark.parametrize("seed", range(30))
def test_quota_keeps_the_train_floor_for_rare_multi_class_patients(seed):
    # The leftover 4th patient of the rarest class must stay in train: it may
    # neither fill a global val/test target nor cover another class in val/test.
    rows = _rows_from_label_sets(_MES_LIKE_PATIENTS)
    patient_level_split(rows, 0.15, 0.15, seed, verbose=False)

    summary = split_summary(rows)
    for label, per_split in summary["per_class"].items():
        assert per_split["train"]["patients"] >= MIN_TRAIN_PATIENTS, (label, per_split)
        assert per_split["val"]["patients"] >= 1 and per_split["test"]["patients"] >= 1
    assert not [w for w in summary["warnings"] if "'train'" in w]


def test_train_floor_wins_over_the_global_targets():
    # 4 + 4 single-class patients at 40 / 40 %: targets of 3 val and 3 test
    # patients. Coverage puts one patient per class in each split; a third
    # val / test patient would leave its class a single train patient.
    rows = rows_with_labels({0: 4, 1: 4})
    patient_level_split(rows, 0.4, 0.4, seed=0, verbose=False)

    assert {s: len(_patients_in(rows, s)) for s in SPLITS} == {"train": 4, "val": 2, "test": 2}
    summary = split_summary(rows)
    assert all(per["train"]["patients"] == 2 for per in summary["per_class"].values())


def test_split_summary_counts_and_json():
    rows = make_rows(30, 3, seed=2)
    patient_level_split(rows, 0.2, 0.2, seed=2, verbose=False)
    summary = split_summary(pd.DataFrame(rows))

    json.dumps(summary)  # JSON-serialisable
    assert sum(summary["n_frames"].values()) == len(rows)
    assert sum(summary["n_clips"].values()) == len({r["video_id"] for r in rows})
    assert abs(sum(summary["frame_fraction"].values()) - 1.0) < 1e-12
    for label, per_split in summary["per_class"].items():
        n_label = sum(1 for r in rows if str(r["label"]) == label)
        assert sum(v["frames"] for v in per_split.values()) == n_label


def test_split_summary_flags_patient_overlap():
    rows = rows_with_labels({0: 4, 1: 4})
    for r in rows:
        r["split"] = "train"
    rows[0]["split"] = "test"  # p000 now in train and test
    summary = split_summary(rows)
    assert summary["n_patients_in_multiple_splits"] == 1
    assert any("more than one split" in w for w in summary["warnings"])


@pytest.mark.parametrize(
    ("val_ratio", "test_ratio"), [(0.5, 0.5), (-0.1, 0.2), (0.2, 1.0), (0.7, 0.4)]
)
def test_invalid_ratios_raise(val_ratio, test_ratio):
    with pytest.raises(ValueError):
        patient_level_split(make_rows(10, 2), val_ratio, test_ratio, 0, verbose=False)


# ---------------------------------------------------------------------------
# Stratified split (LIMUC port)
# ---------------------------------------------------------------------------


def test_modal_patient_labels_ties_go_to_the_smallest_label():
    rows = [
        {"patient_id": "a", "label": 3},
        {"patient_id": "a", "label": 1},
        {"patient_id": "a", "label": 3},
        {"patient_id": "a", "label": 1},
        {"patient_id": "b", "label": 2},
        {"patient_id": "b", "label": 0},
        {"patient_id": "b", "label": 2},
    ]
    assert modal_patient_labels(rows) == {"a": "1", "b": "2"}


@pytest.mark.parametrize("seed", [0, 42])
def test_stratified_no_overlap_and_deterministic(seed):
    rows = make_rows(60, 4, class_probs=[0.4, 0.3, 0.2, 0.1], seed=seed)
    out, info = stratified_patient_split([dict(r) for r in rows], 0.15, 0.15, seed, verbose=False)
    again, info_again = stratified_patient_split(
        [dict(r) for r in rows], 0.15, 0.15, seed, verbose=False
    )

    assert all(len(s) == 1 for s in _patient_splits(out).values())
    assert [r["split"] for r in out] == [r["split"] for r in again]
    assert info == info_again
    n = len({r["patient_id"] for r in rows})
    n_val_test = len(_patients_in(out, "val")) + len(_patients_in(out, "test"))
    assert abs(n_val_test / n - 0.30) <= 0.1


def test_stratified_row_order_does_not_matter():
    rows = make_rows(40, 3, seed=5)
    a, _ = stratified_patient_split([dict(r) for r in rows], 0.15, 0.15, 42, verbose=False)
    shuffled = [dict(r) for r in rows]
    random.Random(1).shuffle(shuffled)
    b, _ = stratified_patient_split(shuffled, 0.15, 0.15, 42, verbose=False)
    assert {r["patient_id"]: r["split"] for r in a} == {r["patient_id"]: r["split"] for r in b}


def test_stratified_rare_strata_are_assigned_test_first():
    # With test_ratio 0.15 a stratum needs >= ceil(1/0.15)+1 = 8 patients to be
    # split by train_test_split; smaller strata are assigned by hand.
    rows = rows_with_labels({0: 20, 1: 12, 2: 5, 3: 1})
    out, info = stratified_patient_split(rows, 0.15, 0.15, 42, verbose=False)

    per_class = split_summary(out)["per_class"]
    assert per_class["3"]["test"]["patients"] == 1  # a single-patient stratum -> test
    assert per_class["2"]["test"]["patients"] == 1  # 5 patients: 1 test, 1 val, 3 train
    assert per_class["2"]["val"]["patients"] == 1
    assert per_class["2"]["train"]["patients"] == 3
    assert info["strategy"] == "partial"
    assert info["n_rare_patients"] == 6
    assert info["split_strategy"] == "stratified"


def test_split_with_rare_strata_two_member_stratum():
    ids = [f"x{i}" for i in range(22)]
    labels = ["0"] * 20 + ["1"] * 2
    train, val, test, strategy, rare = split_with_rare_strata(ids, labels, 0.7, 0.15, 0.15, 0)
    assert sorted(rare) == ["x20", "x21"]
    assert len({"x20", "x21"} & set(test)) == 1
    assert len({"x20", "x21"} & set(train)) == 1
    assert sorted(train + val + test) == sorted(ids)
    assert strategy == "partial"


def test_stratified_split_info_schema():
    rows = make_rows(40, 3, seed=8)
    out, info = stratified_patient_split(pd.DataFrame(rows), 0.15, 0.15, 42, verbose=False)

    json.dumps(info)
    assert isinstance(out, list) and all("split" in r for r in out)
    assert set(info["splits"]) == set(SPLITS)
    assert sum(s["n_frames"] for s in info["splits"].values()) == info["total_frames"] == len(rows)
    assert sum(s["n_patients"] for s in info["splits"].values()) == info["n_patients"]
    for split, s in info["splits"].items():
        assert sum(s["label_counts"].values()) == s["n_frames"], split
    expected = Counter(str(r["label"]) for r in rows)
    assert info["label_distribution"] == {k: expected[k] for k in sorted(expected)}
    assert info["random_seed"] == 42
    assert info["train_ratio"] == pytest.approx(0.70)


@pytest.mark.parametrize(("val_ratio", "test_ratio"), [(0.0, 0.15), (0.15, 0.0), (0.5, 0.5)])
def test_stratified_requires_positive_ratios(val_ratio, test_ratio):
    with pytest.raises(ValueError):
        stratified_patient_split(make_rows(20, 2), val_ratio, test_ratio, 0, verbose=False)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def test_assign_splits_coverage_matches_patient_level_split():
    rows = make_rows(40, 4, seed=6)
    direct = patient_level_split([dict(r) for r in rows], 0.15, 0.15, 42, verbose=False)
    via, info = assign_splits(
        [dict(r) for r in rows], 0.15, 0.15, 42, strategy="coverage", verbose=False
    )

    assert [r["split"] for r in via] == [r["split"] for r in direct]
    assert info["split_strategy"] == "coverage"
    assert info["strategy"] == "quota"
    assert info == build_split_info(
        via,
        strategy="quota",
        train_ratio=1.0 - 0.15 - 0.15,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    ) | {"split_strategy": "coverage"}


def test_assign_splits_defaults_to_stratified():
    rows = make_rows(40, 3, seed=6)
    via, info = assign_splits([dict(r) for r in rows], 0.15, 0.15, 42, verbose=False)
    direct, _ = stratified_patient_split([dict(r) for r in rows], 0.15, 0.15, 42, verbose=False)
    assert [r["split"] for r in via] == [r["split"] for r in direct]
    assert info["split_strategy"] == "stratified"


def test_assign_splits_stratified_dispatch():
    rows = make_rows(40, 3, seed=6)
    via, info = assign_splits(
        [dict(r) for r in rows], 0.15, 0.15, 42, strategy="stratified", verbose=False
    )
    direct, _ = stratified_patient_split([dict(r) for r in rows], 0.15, 0.15, 42, verbose=False)
    assert [r["split"] for r in via] == [r["split"] for r in direct]
    assert info["split_strategy"] == "stratified"


def test_assign_splits_rejects_unknown_options():
    assert SPLIT_STRATEGIES == ("temporal", "coverage", "stratified")
    with pytest.raises(ValueError, match="strategy"):
        assign_splits(make_rows(10, 2), 0.15, 0.15, 0, strategy="random", verbose=False)
