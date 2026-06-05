"""Configuration loading and management."""

from __future__ import annotations

import json
from pathlib import Path

from .models import ModelConfig
from .paths import PathConfig, get_default_paths
from .training import EvaluationConfig, TrainingConfig


class Config:
    """Global configuration container."""

    def __init__(
        self,
        model: ModelConfig | None = None,
        training: TrainingConfig | None = None,
        evaluation: EvaluationConfig | None = None,
        paths: PathConfig | None = None,
    ):
        self.model = model or ModelConfig()
        self.training = training or TrainingConfig()
        self.evaluation = evaluation or EvaluationConfig()
        self.paths = paths or get_default_paths()

    def __repr__(self) -> str:
        return (
            f"Config(\n"
            f"  model={self.model},\n"
            f"  training={self.training},\n"
            f"  evaluation={self.evaluation},\n"
            f"  paths={self.paths}\n"
            f")"
        )


def load_config(config_path: Path | None = None) -> Config:
    """Load configuration from file or use defaults.

    Args:
        config_path: Path to config JSON/YAML file (optional).

    Returns:
        Loaded Config instance.
    """
    if config_path is None:
        cfg = Config()
        _validate_loaded_config(cfg)
        return cfg

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if config_path.suffix == ".json":
        cfg = _load_json_config(config_path)
        _validate_loaded_config(cfg)
        return cfg
    elif config_path.suffix in (".yaml", ".yml"):
        cfg = _load_yaml_config(config_path)
        _validate_loaded_config(cfg)
        return cfg
    else:
        raise ValueError(f"Unsupported config format: {config_path.suffix}")


def _validate_loaded_config(config: Config) -> None:
    from .validation import validate_config

    validate_config(config)


def _build_config_from_dict(data: dict) -> Config:
    return Config(
        model=ModelConfig(**data.get("model", {})),
        training=TrainingConfig(**data.get("training", {})),
        evaluation=EvaluationConfig(**data.get("evaluation", {})),
        paths=PathConfig(**data.get("paths", {})),
    )


def _load_json_config(path: Path) -> Config:
    with open(path) as f:
        return _build_config_from_dict(json.load(f))


def _load_yaml_config(path: Path) -> Config:
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML is required for YAML config files. Install with: pip install pyyaml"
        ) from None
    with open(path) as f:
        return _build_config_from_dict(yaml.safe_load(f) or {})


def legacy_dict_to_config(config_dict: dict) -> Config:
    """Convert legacy dict-based config to Config object."""
    model_config = ModelConfig(
        model=config_dict.get("model", "vitb16_imagenet"),
        num_classes=config_dict.get("num_classes", 1),
        freeze_layers=config_dict.get("freeze_layers", -1),
        threshold=config_dict.get("threshold", 0.5),
        dropout_rate=config_dict.get("dropout_rate", 0.5),
        head_type=config_dict.get("head_type", "linear"),
    )

    training_config = TrainingConfig(
        batch_size=config_dict.get("batch_size", 64),
        epochs=config_dict.get("epochs", 100),
        learning_rate=config_dict.get("learning_rate", 5e-5),
        optimizer=config_dict.get("optimizer", "AdamW"),
        weight_decay=config_dict.get("weight_decay", 1e-2),
        label_smoothing=config_dict.get("label_smoothing", 0.0),
        class_weights=config_dict.get("class_weights"),
        dropout_rate=config_dict.get("dropout_rate", 0.5),
        lr_patience=config_dict.get("lr_patience", 10),
        lr_factor=config_dict.get("lr_factor", 0.5),
        es_patience=config_dict.get("es_patience", 20),
        equalize=config_dict.get("equalize", True),
        num_workers=config_dict.get("num_workers", 8),
        device_id=config_dict.get("device", 0),
    )

    return Config(model=model_config, training=training_config)
