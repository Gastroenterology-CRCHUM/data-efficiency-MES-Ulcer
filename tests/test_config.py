"""Configuration loading, overrides and validation (src.config)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from src.config import (
    Config,
    EvaluationConfig,
    PathConfig,
    TrainingConfig,
    apply_overrides,
    legacy_dict_to_config,
    load_config,
    validate_config,
)
from src.config.paths import MesPaths
from src.utils import ConfigurationError

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_base_yaml_lists_every_field():
    # The runners' default config may differ from the dataclass defaults, but lists every field.
    path = REPO_ROOT / "configs" / "base.yaml"
    load_config(path)  # validated
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    defaults = Config().to_dict()
    for section in ("model", "training", "evaluation"):
        assert set(data[section]) == set(defaults[section]), section
    assert load_config() == Config()


def test_defaults():
    training, evaluation = TrainingConfig(), EvaluationConfig()
    assert training.deterministic is True
    assert training.es_patience == 20
    assert evaluation.bootstrap_samples == 1000
    assert evaluation.confidence_interval == 95


def test_to_dict_is_json_with_posix_paths():
    data = Config().to_dict()
    text = json.dumps(data)
    assert "\\" not in text
    assert data["paths"]["pretrained_dir"] == "data/assets/pretrained"
    assert set(data) == {"model", "training", "evaluation", "paths"}


def test_load_config_json_and_yaml(tmp_path):
    spec = {"model": {"model": "efficientnetb0"}, "training": {"epochs": 7, "batch_size": 8}}
    (tmp_path / "c.json").write_text(json.dumps(spec), encoding="utf-8")
    (tmp_path / "c.YML").write_text(yaml.safe_dump(spec), encoding="utf-8")
    for name in ("c.json", "c.YML"):
        cfg = load_config(tmp_path / name)
        assert cfg.model.model == "efficientnetb0"
        assert (cfg.training.epochs, cfg.training.batch_size) == (7, 8)
        assert cfg.evaluation == EvaluationConfig()


@pytest.mark.parametrize(
    "spec",
    [{"modle": {}}, {"training": {"epoch": 3}}, {"paths": {"mes": {"rot": "x"}}}, ["a"]],
    ids=["section", "key", "nested-key", "not-a-mapping"],
)
def test_load_config_rejects_unknown_keys(tmp_path, spec):
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_config(path)


def test_load_config_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")
    (tmp_path / "c.toml").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported"):
        load_config(tmp_path / "c.toml")


def test_apply_overrides_routes_flat_keys():
    base = Config()
    cfg = apply_overrides(
        base,
        {
            "model": "resnet50_5M",
            "freeze_layers": -1,
            "learning_rate": 3e-5,
            "dropout_rate": 0.2,
            "device_id": 1,
            "bootstrap_samples": 500,
        },
    )
    assert cfg.model.model == "resnet50_5M" and cfg.model.freeze_layers == -1
    assert cfg.training.learning_rate == 3e-5
    assert cfg.model.dropout_rate == 0.2
    assert cfg.training.device_id == 1
    assert cfg.evaluation.bootstrap_samples == 500
    assert base == Config()  # the base is not modified


def test_apply_overrides_sections_and_revalidation():
    cfg = apply_overrides(Config(), {"training": {"epochs": 3}, "paths": {"mes": {"root": "x"}}})
    assert cfg.training.epochs == 3
    assert cfg.paths.mes.processed == Path("x") / "processed"
    with pytest.raises(ConfigurationError):
        apply_overrides(Config(), {"epochs": 0})  # __post_init__ runs again


@pytest.mark.parametrize("key, bad", [("layer_decay", 0.0), ("layer_decay", 1.5),
                                      ("drop_path_rate", -0.1), ("drop_path_rate", 1.0)])
def test_vit_regularisation_ranges(key, bad):
    cfg = apply_overrides(Config(), {"layer_decay": 0.75, "drop_path_rate": 0.1})
    assert (cfg.training.layer_decay, cfg.model.drop_path_rate) == (0.75, 0.1)
    with pytest.raises(ConfigurationError, match=key):
        apply_overrides(Config(), {key: bad})


def test_unknown_override_keys_warn_or_raise():
    cfg = apply_overrides(Config(), {"not_a_field": 1})
    assert cfg == Config()
    with pytest.raises(ConfigurationError, match="not_a_field"):
        apply_overrides(Config(), {"not_a_field": 1}, strict=True)


def test_legacy_dict_to_config_uses_base_then_dataclass_defaults():
    base = apply_overrides(Config(), {"epochs": 5})
    cfg = legacy_dict_to_config({"model": "efficientnetb0", "warmup_epochs": 2}, base=base)
    assert cfg.model.model == "efficientnetb0"
    assert cfg.training.epochs == 5 and cfg.training.warmup_epochs == 2
    plain = legacy_dict_to_config({"learning_rate": 1e-4})
    assert plain.model.model == "resnet50_5M" and plain.model.freeze_layers == 0


def test_path_config_coercion():
    paths = PathConfig(mes="elsewhere/mes", ulcer={"root": "u"}, pretrained_dir="w")
    assert isinstance(paths.mes, MesPaths)
    assert paths.mes.processed == Path("elsewhere/mes/processed")
    assert paths.ulcer_splits_dir == Path("u/splits")
    assert paths.pretrained_dir == Path("w")
    with pytest.raises(ConfigurationError):
        PathConfig(mes={"root": "x", "extra": 1})
    with pytest.raises(ConfigurationError):
        PathConfig(pretrained_dir=3)


def test_validate_config():
    cfg = Config()
    validate_config(cfg, img_size=256)
    with pytest.raises(ValueError, match="img_size|image size|256"):
        validate_config(cfg, img_size=224)
    cfg.training.epochs = 0  # changed after construction
    with pytest.raises(ValueError):
        validate_config(cfg)
    validate_config(None)  # skipped
    with pytest.raises(ConfigurationError, match="active_loss_weight"):
        TrainingConfig(active_loss_weight=-1.0)
