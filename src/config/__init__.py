"""Configuration module for data-efficiency experiments."""

from .loader import Config, apply_overrides, legacy_dict_to_config, load_config
from .models import MODEL_REGISTRY, ModelConfig, get_img_size, get_model_entry, get_norm_stats
from .paths import PathConfig
from .training import EvaluationConfig, TrainingConfig
from .validation import validate_config

__all__ = [
    "Config",
    "ModelConfig",
    "TrainingConfig",
    "EvaluationConfig",
    "PathConfig",
    "MODEL_REGISTRY",
    "apply_overrides",
    "get_img_size",
    "get_model_entry",
    "get_norm_stats",
    "load_config",
    "legacy_dict_to_config",
    "validate_config",
]
