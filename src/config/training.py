"""Training configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union


@dataclass
class TrainingConfig:
    """Training hyperparameters: the GastroNet-5M fine-tuning protocol, with AMP on.

    Fallbacks for keys missing from ``configs/base.yaml``, the runners' default config.
    """

    subset_ratio: float = 1.0
    """Subset ratio of the total training dataset available (mainly for data efficiency)."""

    random_seed: int = 42
    """Seed of the run: training subset, global RNGs (``seed_everything``) and loader shuffling."""

    deterministic: bool = True
    """Deterministic cuDNN / torch algorithms (``src.utils.seed_everything``)."""

    batch_size: int = 32
    """Number of samples per batch."""

    epochs: int = 100
    """Maximum number of training epochs."""

    learning_rate: float = 1.0e-5
    """Peak learning rate (same for every backbone)."""

    layer_decay: float = 1.0
    """ViT backbones only: layer-wise LR decay. The last block trains at learning_rate *
    layer_decay, each earlier block one factor lower, the patch / position embeddings lowest;
    the final norm and the head at learning_rate. 1 = off."""

    optimizer: str = "Adam"
    """Optimizer type: Adam or AdamW."""

    weight_decay: float = 1e-3
    """Weight decay (L2 for Adam, decoupled for AdamW)."""

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

    active_loss_weight: float = 1.0
    """MES only: weight of the added active (Mayo 2-3) vs remission CE on p2 + p3; 0 = off."""

    # Early stopping
    es_patience: int = 20
    """Early stopping patience (epochs without val-AUROC improvement)."""

    es_start_epoch: int = 5
    """Patience counts only the epochs after this one (0 = from the first epoch)."""

    grad_clip_norm: float = 10.0
    """Max global gradient norm per optimizer step."""

    # Data
    equalize: bool = True
    """Apply CLAHE histogram equalization to images."""

    # Hardware
    num_workers: int = 8
    """Number of data loading workers."""

    device_id: int = 0
    """GPU device ID (CPU if < 0)."""

    use_amp: bool = True
    """Use mixed precision (AMP) training."""

    # LR scheduling — warmup + cosine annealing
    warmup_epochs: int = 10
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

        if self.random_seed < 0:
            errors.append(f"random_seed must be >= 0, got {self.random_seed}")

        if self.batch_size < 1:
            errors.append(f"batch_size must be >= 1, got {self.batch_size}")

        if self.epochs < 1:
            errors.append(f"epochs must be >= 1, got {self.epochs}")

        if self.learning_rate <= 0:
            errors.append(f"learning_rate must be > 0, got {self.learning_rate}")

        if not (0 < self.layer_decay <= 1):
            errors.append(f"layer_decay must be in (0, 1], got {self.layer_decay}")

        if self.optimizer not in ("Adam", "AdamW"):
            errors.append(f"optimizer must be Adam or AdamW, got {self.optimizer}")

        if not (0 <= self.label_smoothing < 1):
            errors.append(f"label_smoothing must be in [0, 1), got {self.label_smoothing}")

        if self.active_loss_weight < 0:
            errors.append(f"active_loss_weight must be >= 0, got {self.active_loss_weight}")

        if self.es_patience < 1:
            errors.append(f"es_patience must be >= 1, got {self.es_patience}")

        if self.es_start_epoch < 0:
            errors.append(f"es_start_epoch must be >= 0, got {self.es_start_epoch}")

        if self.grad_clip_norm <= 0:
            errors.append(f"grad_clip_norm must be > 0, got {self.grad_clip_norm}")

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


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""

    bootstrap_samples: int = 1000
    """Number of bootstrap resamples for CIs (1000 in the published runs, =
    ``compute_metrics_with_ci``'s default)."""

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
