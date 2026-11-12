"""Training and evaluation pipelines (src.data.transforms.get_transforms)."""

from __future__ import annotations

import pytest
import torchvision.transforms as T

from src.data.transforms import get_transforms


def _types(pipeline: T.Compose) -> list[type]:
    return [type(t) for t in pipeline.transforms]


def test_training_geometry_is_one_affine_transform():
    pipeline = get_transforms(64, is_training=True, cached=True)
    affine = next(t for t in pipeline.transforms if isinstance(t, T.RandomAffine))
    assert affine.degrees == [-30.0, 30.0]
    assert affine.translate == (0.1, 0.1)
    assert affine.shear == [-5.0, 5.0, -5.0, 5.0]
    assert T.RandomRotation not in _types(pipeline)
    assert T.RandAugment not in _types(pipeline) and T.RandomErasing not in _types(pipeline)


@pytest.mark.parametrize("flag, step", [("use_randaugment", T.RandAugment),
                                        ("use_random_erasing", T.RandomErasing)])
def test_optional_augmentations(flag, step):
    assert step in _types(get_transforms(64, is_training=True, cached=True, **{flag: True}))


def test_evaluation_pipeline_has_no_random_step():
    assert _types(get_transforms(64, is_training=False, cached=True)) == [T.ToTensor, T.Normalize]
