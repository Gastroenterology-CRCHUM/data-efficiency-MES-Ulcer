"""Model registry (src.config.models): keys, normalisation stats, weight provenance."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

import src.config as config_pkg
from src.config.models import (
    DINO_HUB_COMMIT,
    DINO_HUB_REPO,
    GASTRONET_WEIGHTS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MODEL_REGISTRY,
    ModelConfig,
    get_img_size,
    get_model_entry,
    get_norm_stats,
)
from src.utils import ConfigurationError

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", sorted(MODEL_REGISTRY))
def test_every_registry_key_validates(name):
    cfg = ModelConfig(model=name)
    assert cfg.model == name
    for freeze in (-1, 0, 2):
        ModelConfig(model=name, freeze_layers=freeze, head_type="mlp1", num_classes=4)


@pytest.mark.parametrize("name", sorted(MODEL_REGISTRY))
def test_registry_entries_are_complete(name):
    entry = get_model_entry(name)
    assert entry.classifier
    assert entry.description and entry.architecture
    assert entry.pretrain_data and entry.pretrain_method
    assert isinstance(get_img_size(name), int) and get_img_size(name) > 0
    if entry.gastronet is not None:
        # External weights are file names relative to PathConfig.pretrained_dir.
        assert not Path(entry.gastronet).is_absolute()
        assert len(Path(entry.gastronet).parts) == 1


@pytest.mark.parametrize("name", sorted(MODEL_REGISTRY))
def test_get_norm_stats(name):
    mean, std = get_norm_stats(name)
    assert len(mean) == len(std) == 3
    assert all(0 < v < 1 for v in (*mean, *std))
    if name == "vits16_imagenet_hf":  # timm augreg_in1k normalisation
        assert mean == std == (0.5, 0.5, 0.5)
    else:
        assert mean == IMAGENET_MEAN and std == IMAGENET_STD


def test_vitb16_gastronet_was_removed():
    assert "vitb16_gastronet" not in MODEL_REGISTRY
    assert "vitb16" not in GASTRONET_WEIGHTS
    with pytest.raises(ConfigurationError):
        ModelConfig(model="vitb16_gastronet")


def test_vit_base_dinov2_pair_differs_only_in_pretraining_data():
    gastronet, lvd = MODEL_REGISTRY["vitb14_gastronet"], MODEL_REGISTRY["vitb14_dinov2"]
    assert gastronet.hf_model_id == lvd.hf_model_id and "reg4" in lvd.hf_model_id  # 4 registers
    assert (gastronet.architecture, gastronet.pretrain_method) == (lvd.architecture, lvd.pretrain_method)
    assert gastronet.gastronet == GASTRONET_WEIGHTS["vitb14"] and lvd.gastronet is None
    assert {gastronet.img_size, lvd.img_size, MODEL_REGISTRY["vitb16_dinov3"].img_size} == {336}


def test_unknown_model_and_invalid_fields_raise():
    with pytest.raises(ValueError, match="not found"):
        get_model_entry("no_such_model")
    with pytest.raises(ConfigurationError):
        ModelConfig(model="no_such_model")
    with pytest.raises(ConfigurationError):
        ModelConfig(head_type="transformer")
    with pytest.raises(ConfigurationError):
        ModelConfig(freeze_layers=-2)
    with pytest.raises(ConfigurationError):
        ModelConfig(threshold=1.0)
    with pytest.raises(ConfigurationError):
        ModelConfig(num_classes=0)


def test_dino_hub_is_pinned_to_a_commit():
    assert re.fullmatch(r"[0-9a-f]{40}", DINO_HUB_COMMIT)
    assert f"facebookresearch/dino:{DINO_HUB_COMMIT}" == DINO_HUB_REPO
    for entry in MODEL_REGISTRY.values():
        if entry.hub_model is not None:
            assert entry.weights == DINO_HUB_REPO


def test_package_exports():
    assert config_pkg.get_norm_stats is get_norm_stats
    assert config_pkg.MODEL_REGISTRY is MODEL_REGISTRY


@pytest.mark.parametrize(
    "plan", sorted((REPO_ROOT / "configs" / "experiments").glob("*.yaml")), ids=lambda p: p.name
)
def test_experiment_plans_only_use_registry_models(plan):
    spec = yaml.safe_load(plan.read_text(encoding="utf-8"))
    models = [run["model"] for run in spec["models"]]
    assert models
    assert set(models) <= set(MODEL_REGISTRY)
