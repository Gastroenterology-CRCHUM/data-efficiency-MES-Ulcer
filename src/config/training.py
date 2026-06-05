"""Training configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    subset_ratio: float = 1.0
    """Subset ratio of the total training dataset available (mainly for data efficiency)."""

    random_seed: int = 42
    """Random seed for reproducibility."""

    batch_size: int = 64
    """Number of samples per batch."""

    epochs: int = 100
    """Maximum number of training epochs."""

    learning_rate: float = 5e-5
    """Initial learning rate."""

    optimizer: str = "AdamW"
    """Optimizer type: Adam or AdamW."""

    weight_decay: float = 1e-2
    """L2 regularization coefficient."""

    label_smoothing: float = 0.0
    """Label smoothing (0.0 = no smoothing). Useful for multi-class."""

    # Loss and class balance
    class_weights: Union[float, list[float]] | None = None
    """
    Class weights:
    - None: auto-compute from manifest
    - float: manual weight for positive class (binary only)
    - list: weights per class (multi-class)
    """

    # Regularization
    dropout_rate: float = 0.5
    """Dropout probability before classification head."""

    # Learning rate scheduling
    lr_patience: int = 10
    """Patience for ReduceLROnPlateau (epochs without improvement)."""

    lr_factor: float = 0.5
    """LR reduction factor (multiplied by current LR on plateau)."""

    # Early stopping
    es_patience: int = 10
    """Early stopping patience (epochs without AUROC improvement)."""

    # Data
    equalize: bool = True
    """Apply CLAHE histogram equalization to images."""

    # Output mode
    aggregate_by_clip: bool = True
    """Aggregate predictions at clip level when evaluating (binary only)."""

    run_train: bool = True
    """Whether to execute training stage."""

    run_test: bool = True
    """Whether to execute test stage."""

    run_threshold_tuning: bool = False
    """Whether to execute threshold tuning stage (binary only)."""

    # Hardware
    num_workers: int = 8
    """Number of data loading workers."""

    device_id: int = 0
    """GPU device ID (CPU if < 0)."""

    use_amp: bool = True
    """Use mixed precision (AMP) training."""

    # LR scheduling — warmup + cosine annealing
    warmup_epochs: int = 5
    """Linear warmup epochs before cosine decay begins."""

    min_lr: float = 1e-7
    """Minimum learning rate at the end of cosine annealing."""

    # Augmentation — RandAugment
    use_randaugment: bool = False
    """Apply RandAugment to training images."""

    randaugment_m: int = 9
    """RandAugment magnitude (0–30). 9 = medium strength."""

    # Augmentation — Random Erasing
    use_random_erasing: bool = False
    """Apply RandomErasing after normalisation."""

    random_erasing_p: float = 0.25
    """RandomErasing probability."""

    def __post_init__(self):
        """Validate configuration."""
        from src.utils import ConfigurationError

        errors = []

        if not (0 < self.subset_ratio <= 1):
            errors.append(f"subset_ratio must be in (0, 1], got {self.subset_ratio}")

        if self.batch_size < 1:
            errors.append(f"batch_size must be >= 1, got {self.batch_size}")

        if self.epochs < 1:
            errors.append(f"epochs must be >= 1, got {self.epochs}")

        if self.learning_rate <= 0:
            errors.append(f"learning_rate must be > 0, got {self.learning_rate}")

        if self.optimizer not in ("Adam", "AdamW"):
            errors.append(f"optimizer must be Adam or AdamW, got {self.optimizer}")

        if not (0 <= self.label_smoothing < 1):
            errors.append(f"label_smoothing must be in [0, 1), got {self.label_smoothing}")

        if not (0 <= self.dropout_rate < 1):
            errors.append(f"dropout_rate must be in [0, 1), got {self.dropout_rate}")

        if self.lr_factor <= 0 or self.lr_factor >= 1:
            errors.append(
                f"lr_factor must be in (0, 1), got {self.lr_factor}. "
                f"Typical: 0.1 (aggressive) to 0.5 (gentle)."
            )

        if self.num_workers < 0:
            errors.append(f"num_workers must be >= 0, got {self.num_workers}")

        if not isinstance(self.class_weights, (float, list)) and self.class_weights:
            errors.append(
                f"class_weights must be float, list or None, must coincide with num_classes,"
                f"got {self.class_weights}."
            )

        if self.warmup_epochs < 0:
            errors.append(f"warmup_epochs must be >= 0, got {self.warmup_epochs}")

        if self.min_lr <= 0:
            errors.append(f"min_lr must be > 0, got {self.min_lr}")

        if not (0 <= self.randaugment_m <= 30):
            errors.append(f"randaugment_m must be in [0, 30], got {self.randaugment_m}")

        if not (0 <= self.random_erasing_p <= 1):
            errors.append(f"random_erasing_p must be in [0, 1], got {self.random_erasing_p}")

        if errors:
            raise ConfigurationError("Training config errors:\n  • " + "\n  • ".join(errors))

    @classmethod
    def default(cls) -> TrainingConfig:
        """Get default training configuration."""
        return cls()


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""

    bootstrap_samples: int = 10_000
    """Number of bootstrap samples for CI computation."""

    confidence_interval: int = 95
    """Confidence interval percentage (e.g., 95 = 95% CI)."""

    def __post_init__(self):
        """Validate configuration."""
        if self.bootstrap_samples < 100:
            raise ValueError(f"bootstrap_samples must be >= 100, got {self.bootstrap_samples}")

        if not (50 <= self.confidence_interval < 100):
            raise ValueError(
                f"confidence_interval must be in [50, 100), got {self.confidence_interval}"
            )
