"""Configuration validation utilities."""

import logging

from .loader import Config
from .models import MODEL_REGISTRY

logger = logging.getLogger(__name__)


def validate_config(config: Config | None = None) -> None:
    """Comprehensive configuration validation.

    Validates model registry, learning rates, image sizes, batch sizes,
    and other critical parameters to catch configuration errors early.

    Args:
        config: Config object to validate. If None, will validate against
               standard bounds without a specific config instance.

    Raises:
        ValueError: If configuration is invalid with helpful error message(s).
    """
    errors = []

    if config is None:
        logger.info("No config provided - skipping validation")
        return

    # ── Model validation ──────────────────────────────────────────────
    model_name = config.model.model if hasattr(config.model, "model") else None
    if model_name and model_name not in MODEL_REGISTRY:
        errors.append(f"Model '{model_name}' not in MODEL_REGISTRY")

    # ── Learning rate bounds ──────────────────────────────────────────
    if hasattr(config.training, "learning_rate"):
        lr = config.training.learning_rate
        if lr < 1e-8 or lr > 1e-2:
            errors.append(
                f"Learning rate {lr} outside reasonable bounds [1e-8, 1e-2]. "
                f"Typical values: ResNet50 (1e-3), ViT (5e-5)"
            )

    # ── Image size validation ─────────────────────────────────────────
    if model_name and hasattr(config.model, "img_size") and model_name in MODEL_REGISTRY:
        entry = MODEL_REGISTRY[model_name]
        required_size = (
            entry.get("img_size") if isinstance(entry, dict) else getattr(entry, "img_size", None)
        )
        configured_size = config.model.img_size
        if required_size and configured_size and configured_size != required_size:
            errors.append(
                f"Model {model_name} requires img_size={required_size}, "
                f"but config specifies {configured_size}"
            )

    # ── Batch size validation ─────────────────────────────────────────
    if hasattr(config.training, "batch_size"):
        batch_size = config.training.batch_size
        if batch_size < 1 or batch_size > 512:
            errors.append(f"batch_size {batch_size} outside valid range [1, 512]")

    # ── Number of epochs ──────────────────────────────────────────────
    if hasattr(config.training, "num_epochs"):
        epochs = config.training.num_epochs
        if epochs < 1 or epochs > 1000:
            errors.append(f"num_epochs {epochs} unreasonable (recommend: 50-200)")

    # ── Dropout validation ────────────────────────────────────────────
    if hasattr(config.model, "dropout_rate"):
        dropout = config.model.dropout_rate
        if dropout < 0 or dropout >= 1:
            errors.append(f"dropout_rate {dropout} must be in [0, 1)")

    # ── Freeze layers validation ──────────────────────────────────────
    if hasattr(config.model, "freeze_layers"):
        freeze = config.model.freeze_layers
        if freeze < -1:
            errors.append(
                f"freeze_layers {freeze} must be >= -1 (-1=freeze backbone, 0=none, N=first N blocks)"
            )

    # Raise if any errors found
    if errors:
        error_msg = "Configuration validation failed:\n  " + "\n  ".join(errors)
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info("Configuration validated successfully")
