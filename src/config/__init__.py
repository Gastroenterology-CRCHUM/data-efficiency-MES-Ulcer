"""Configuration module for data-efficiency experiments."""

from .loader import Config, legacy_dict_to_config, load_config
from .models import MODEL_REGISTRY, ModelConfig, get_img_size, get_model_entry
from .paths import PathConfig
from .training import EvaluationConfig, TrainingConfig

__all__ = [
    "Config",
    "ModelConfig",
    "TrainingConfig",
    "EvaluationConfig",
    "PathConfig",
    "MODEL_REGISTRY",
    "get_img_size",
    "get_model_entry",
    "load_config",
    "legacy_dict_to_config",
]
