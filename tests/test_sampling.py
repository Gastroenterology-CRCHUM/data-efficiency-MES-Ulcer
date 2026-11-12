"""Training-subset sampling (src.data.dataloader._sampling_train, describe_subset)."""

from __future__ import annotations

import pytest

from src.data.dataloader import _sampling_train, describe_subset
from tests.conftest import frames_from_clip_sizes, make_train_frames

RATIOS = (0.05, 0.1, 0.25, 0.5, 0.75, 1.0)


def _clips(df) -> set[str]:
    return set(df["clip_key"].unique())


def _frames(df) -> set[str]:
    return set(df["relative_path"])


def _sample(df, ratio, seed=42):
    return _sampling_train(df, ratio, "label", seed, verbose=False)


@pytest.fixture(scope="module")
def train_df():
    # Imbalanced 4-class train set: 40 / 25 / 8 / 3 clips of 5-60 frames.
    return make_train_frames({0: 40, 1: 25, 2: 8, 3: 3}, seed=1)


# ---------------------------------------------------------------------------
# Frame budget
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 42, 12345])
@pytest.mark.parametrize("ratio", [0.001, 0.1, 0.25, 0.5, 0.75])
def test_every_class_holds_exactly_its_budget(train_df, seed, ratio):
    sub = _sample(train_df, ratio, seed)
    for label, full_c in train_df.groupby("label"):
        assert (sub["label"] == label).sum() == max(1, round(ratio * len(full_c))), label


@pytest.mark.parametrize("seed", [0, 42, 12345])
@pytest.mark.parametrize("ratio", [0.1, 0.25, 0.5, 0.75])
def test_realized_fraction_equals_ratio(train_df, seed, ratio):
    info = describe_subset(_sample(train_df, ratio, seed), train_df)
    # Each class is rounded to the nearest frame.
    assert abs(info["realized_fraction"] - ratio) <= train_df["label"].nunique() / (2 * len(train_df))
    assert info["class_frac"] == pytest.approx(info["class_frac_full"], abs=0.01)


@pytest.mark.parametrize("seed", [0, 7, 42, 2**40 + 3])
def test_subsets_are_nested_across_ratios(train_df, seed):
    subsets = [_frames(_sample(train_df, r, seed)) for r in RATIOS]
    for smaller, larger in zip(subsets, subsets[1:]):
        assert smaller <= larger
    assert subsets[-1] == _frames(train_df)


def test_deterministic_for_a_seed_and_seed_dependent(train_df):
    a = _sample(train_df, 0.5, seed=3)
    b = _sample(train_df, 0.5, seed=3)
    assert a.index.equals(b.index)
    assert _clips(_sample(train_df, 0.5, seed=4)) != _clips(a)


def test_every_class_is_kept_at_a_tiny_ratio(train_df):
    sub = _sample(train_df, 0.001, seed=42)
    assert set(sub["label"]) == set(train_df["label"])


@pytest.mark.parametrize("ratio", [0.1, 0.3, 0.6])
def test_at_most_one_partial_clip_per_class_cut_to_its_first_frames(train_df, ratio):
    sub = _sample(train_df, ratio, seed=42)
    full_paths = train_df.sort_values("relative_path").groupby("clip_key")["relative_path"].agg(list)
    sub_paths = sub.sort_values("relative_path").groupby("clip_key")["relative_path"].agg(list)
    partial = [key for key, paths in sub_paths.items() if len(paths) < len(full_paths[key])]
    assert train_df.groupby("clip_key")["label"].first()[partial].is_unique
    for key in partial:
        assert sub_paths[key] == full_paths[key][: len(sub_paths[key])]


def test_ratio_one_returns_everything(train_df):
    sub = _sample(train_df, 1.0, seed=42)
    assert len(sub) == len(train_df)
    assert sorted(sub.index) == sorted(train_df.index)


def test_row_order_does_not_change_the_subset(train_df):
    shuffled = train_df.sample(frac=1.0, random_state=0)
    assert _frames(_sample(shuffled, 0.3, seed=9)) == _frames(_sample(train_df, 0.3, seed=9))


def test_class_selection_is_independent_of_other_classes(train_df):
    # The clip order of a class depends on (seed, class) only.
    only_0_1 = train_df[train_df["label"].isin([0, 1])]
    full = _sample(train_df, 0.4, seed=5)
    part = _sample(only_0_1, 0.4, seed=5)
    for label in (0, 1):
        assert _frames(full[full["label"] == label]) == _frames(part[part["label"] == label])


def test_budget_rule_on_a_hand_case():
    # One class, 4 clips of 10 frames -> the first round(r * 40) frames of the clip order:
    # whole clips, then the next one cut.
    df = frames_from_clip_sizes({0: [10, 10, 10, 10]})
    expected = {0.1: [4], 0.25: [10], 0.3: [10, 2], 0.5: [10, 10], 0.9: [10, 10, 10, 6]}
    for ratio, sizes in expected.items():
        sub = _sample(df, ratio, seed=0)
        assert sorted(sub.groupby("clip_key").size(), reverse=True) == sizes, ratio


@pytest.mark.parametrize("ratio", [0.0, -0.1, 1.5])
def test_invalid_ratio_raises(train_df, ratio):
    with pytest.raises(ValueError, match="subset_ratio"):
        _sample(train_df, ratio)


def test_verbose_prints_the_realized_subset(train_df, capsys):
    _sampling_train(train_df, 0.5, "label", 42, verbose=True)
    out = capsys.readouterr().out
    assert "frame_budget" in out and "frames:" in out


# ---------------------------------------------------------------------------
# describe_subset
# ---------------------------------------------------------------------------


def test_describe_subset_hand_case():
    full = frames_from_clip_sizes({0: [1, 2, 3, 4, 5, 6], 1: [10, 10, 10]})
    sub = full[full["clip_key"].isin(["a0", "b0"])]
    info = describe_subset(sub, full)

    assert info["n_train"] == 11 and info["n_train_full"] == 51
    assert info["n_clips"] == 2 and info["n_clips_full"] == 9
    assert info["n_patients"] == 2 and info["n_patients_full"] == 9
    assert info["realized_fraction"] == pytest.approx(11 / 51)
    assert info["clip_fraction"] == pytest.approx(2 / 9)
    assert info["class_frac"] == pytest.approx({"0": 1 / 11, "1": 10 / 11})
    assert info["class_frac_full"] == pytest.approx({"0": 21 / 51, "1": 30 / 51})
    assert info["class_realized_fraction"] == pytest.approx({"0": 1 / 21, "1": 10 / 30})
