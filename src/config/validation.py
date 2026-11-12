"""Configuration validation utilities."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .loader import Config

logger = logging.getLogger(__name__)


def validate_config(config: Config | None = None, *, img_size: int | None = None) -> None:
    """Comprehensive configuration validation.

    Validates the model registry key, learning rate, image size, batch size,
    epochs, dropout and freezing, and re-runs every section's dataclass
    ``__post_init__`` checks, so fields changed after construction (e.g. a
    per-run config whose ``epochs`` / ``learning_rate`` were overridden by the
    runner) are validated too. Call it on each per-run config after its last
    change.

    Args:
        config:   Config object to validate. If None, validation is skipped.
        img_size: Input size the loaders will use; must equal the model's
                  ``MODEL_REGISTRY[...].img_size`` when given.

    Raises:
        ValueError: If configuration is invalid with helpful error message(s).
    """
    # Imported lazily: src.config.models pulls in torchvision / model builders.
    from src.utils import ConfigurationError

    from .models import MODEL_REGISTRY

    errors = []

    if config is None:
        logger.info("No config provided - skipping validation")
        return

    # ── Dataclass checks (re-run to catch post-construction changes) ──
    for section in (config.model, config.training, config.evaluation, config.paths):
        post_init = getattr(section, "__post_init__", None)
        if post_init is None:
            continue
        try:
            post_init()
        except (ConfigurationError, ValueError, TypeError) as e:
            errors.append(str(e))

    # ── Model validation ──────────────────────────────────────────────
    model_name = getattr(config.model, "model", None)
    if model_name and model_name not in MODEL_REGISTRY:
        errors.append(f"Model '{model_name}' not in MODEL_REGISTRY")

    # ── Learning rate bounds ──────────────────────────────────────────
    lr = getattr(config.training, "learning_rate", None)
    if lr is not None and (lr < 1e-8 or lr > 1e-2):
        errors.append(
            f"Learning rate {lr} outside reasonable bounds [1e-8, 1e-2]. "
            f"Typical values: ResNet50 (1e-3), ViT (5e-5)"
        )

    # ── Image size validation ─────────────────────────────────────────
    configured_size = img_size if img_size is not None else getattr(config.model, "img_size", None)
    if model_name in MODEL_REGISTRY and configured_size is not None:
        required_size = MODEL_REGISTRY[model_name].img_size
        if configured_size != required_size:
            errors.append(
                f"Model {model_name} requires img_size={required_size}, "
                f"but {configured_size} was configured"
            )

    # ── Batch size validation ─────────────────────────────────────────
    batch_size = getattr(config.training, "batch_size", None)
    if batch_size is not None and (batch_size < 1 or batch_size > 512):
        errors.append(f"batch_size {batch_size} outside valid range [1, 512]")

    # ── Number of epochs ──────────────────────────────────────────────
    epochs = getattr(config.training, "epochs", None)
    if epochs is not None and (epochs < 1 or epochs > 1000):
        errors.append(f"epochs {epochs} outside valid range [1, 1000] (typical: 50-200)")

    # ── Dropout validation ────────────────────────────────────────────
    dropout = getattr(config.model, "dropout_rate", None)
    if dropout is not None and (dropout < 0 or dropout >= 1):
        errors.append(f"dropout_rate {dropout} must be in [0, 1)")

    # ── Freeze layers validation ──────────────────────────────────────
    freeze = getattr(config.model, "freeze_layers", None)
    if freeze is not None and freeze < -1:
        errors.append(
            f"freeze_layers {freeze} must be >= -1 (-1=freeze backbone, 0=none, N=first N blocks)"
        )

    # Raise if any errors found (dedupe: __post_init__ and explicit checks overlap)
    errors = list(dict.fromkeys(errors))
    if errors:
        error_msg = "Configuration validation failed:\n  " + "\n  ".join(errors)
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info("Configuration validated successfully")
