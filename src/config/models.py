"""Model registry and configuration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import torch
from torchvision import models
from torchvision.models import (
    EfficientNet_B0_Weights,
    EfficientNet_B1_Weights,
    EfficientNet_B4_Weights,
    ResNet18_Weights,
    ResNet50_Weights,
    ViT_B_16_Weights,
)


class HeadType(str, Enum):
    """Classification head types."""

    LINEAR = "linear"
    MLP1 = "mlp1"
    MLP2 = "mlp2"


@dataclass
class ModelRegistryEntry:
    """Individual model registry entry."""

    builder: Callable | None
    weights: str | object
    classifier: str
    hub_model: str | None = None
    """Second positional arg to torch.hub.load. None for torchvision models."""
    hf_model_id: str | None = None
    """HuggingFace model ID. When set, builder/hub_model are ignored."""
    gastronet: Path | None = None
    img_size: int = 224
    description: str = ""
    architecture: str = ""
    """Human-readable architecture name for display (e.g. 'ViT-Base/16')."""
    pretrain_data: str = ""
    """Pretraining dataset (e.g. 'ImageNet', 'GastroNet-5M')."""
    pretrain_method: str = ""
    """Pretraining methodology (e.g. 'Supervised', 'Self-sup. (DINOv1)')."""

    def __str__(self) -> str:
        return self.description


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    model: str = "resnet50_gastronet"
    """Model name from MODEL_REGISTRY."""

    num_classes: int = 1
    """1 for binary (BCE), 2+ for multi-class (CrossEntropy)."""

    freeze_layers: int = 0
    """
    Layer freezing strategy:
    - 0: full fine-tuning (no freezing)
    - -1: freeze backbone, train head only
    - N: freeze first N transformer blocks/layer groups
    """

    threshold: float = 0.5
    """Binary decision threshold. Ignored for multi-class."""

    dropout_rate: float = 0.5
    """Dropout probability before classification head."""

    head_type: str = HeadType.LINEAR.value
    """Classification head type: linear, mlp1, or mlp2."""

    def __post_init__(self):
        from src.utils import ConfigurationError

        if self.model not in MODEL_REGISTRY:
            raise ConfigurationError(
                f"Model '{self.model}' not in MODEL_REGISTRY.\n"
                f"Available: {list(MODEL_REGISTRY.keys())}"
            )

        if self.head_type not in [e.value for e in HeadType]:
            raise ConfigurationError(
                f"head_type must be one of {[e.value for e in HeadType]}, got '{self.head_type}'"
            )

        if self.num_classes < 1:
            raise ConfigurationError(f"num_classes must be >= 1, got {self.num_classes}")

        if self.freeze_layers < -1:
            raise ConfigurationError(f"freeze_layers must be >= -1, got {self.freeze_layers}")

        if not 0 < self.threshold < 1:
            raise ConfigurationError(f"threshold must be in (0, 1), got {self.threshold}")


# ════════════════════════════════════════════════════════════════════════════════
# Model Registry Definition
# ════════════════════════════════════════════════════════════════════════════════

GASTRONET_WEIGHTS = {
    "vits16": Path("VITS_GastroNet-5M_DINOv1.pth"),
    "resnet50": Path("RN50_GastroNet-5M_DINOv1.pth"),
    "resnet50_1M": Path("RN50_GastroNet-1M_DINOv1.pth"),
    "resnet50_5M": Path("RN50_GastroNet-5M_DINOv1.pth"),
    "resnet50_200K": Path("RN50_GastroNet-200K_DINOv1.pth"),
    "vitb16": Path("DINOv2.pth"),
}

MODEL_REGISTRY: dict[str, ModelRegistryEntry] = {
    # ════ ResNet-18 ════
    "resnet18": ModelRegistryEntry(
        builder=models.resnet18,
        weights=ResNet18_Weights.DEFAULT,
        classifier="fc",
        description="ResNet-18 — ImageNet",
        architecture="ResNet-18",
        pretrain_data="ImageNet",
        pretrain_method="Supervised",
    ),
    # ════ ResNet-50 ════
    "resnet50_imagenet_sup": ModelRegistryEntry(
        builder=models.resnet50,
        weights=ResNet50_Weights.DEFAULT,
        classifier="fc",
        description="ResNet-50 — ImageNet / Supervised",
        architecture="ResNet-50",
        pretrain_data="ImageNet",
        pretrain_method="Supervised",
    ),
    "resnet50_imagenet": ModelRegistryEntry(
        builder=torch.hub.load,
        weights="facebookresearch/dino:main",
        hub_model="dino_resnet50",
        classifier="fc",
        description="ResNet-50 — ImageNet / DINOv1",
        architecture="ResNet-50",
        pretrain_data="ImageNet",
        pretrain_method="Self-sup. (DINOv1)",
    ),
    "resnet50_gastronet": ModelRegistryEntry(
        builder=torch.hub.load,
        weights="facebookresearch/dino:main",
        hub_model="dino_resnet50",
        classifier="fc",
        gastronet=GASTRONET_WEIGHTS["resnet50"],
        description="ResNet-50 — GastroNet-5M / DINOv1",
        architecture="ResNet-50",
        pretrain_data="GastroNet-5M",
        pretrain_method="Self-sup. (DINOv1)",
    ),
    "resnet50_1M": ModelRegistryEntry(
        builder=torch.hub.load,
        weights="facebookresearch/dino:main",
        hub_model="dino_resnet50",
        classifier="fc",
        gastronet=GASTRONET_WEIGHTS["resnet50_1M"],
        description="ResNet-50 — GastroNet-1M",
        architecture="ResNet-50",
        pretrain_data="GastroNet-1M",
        pretrain_method="Self-sup. (DINOv1)",
    ),
    "resnet50_5M": ModelRegistryEntry(
        builder=torch.hub.load,
        weights="facebookresearch/dino:main",
        hub_model="dino_resnet50",
        classifier="fc",
        gastronet=GASTRONET_WEIGHTS["resnet50_5M"],
        description="ResNet-50 — GastroNet-5M",
        architecture="ResNet-50",
        pretrain_data="GastroNet-5M",
        pretrain_method="Self-sup. (DINOv1)",
    ),
    "resnet50_200K": ModelRegistryEntry(
        builder=torch.hub.load,
        weights="facebookresearch/dino:main",
        hub_model="dino_resnet50",
        classifier="fc",
        gastronet=GASTRONET_WEIGHTS["resnet50_200K"],
        description="ResNet-50 — GastroNet-200K",
        architecture="ResNet-50",
        pretrain_data="GastroNet-200K",
        pretrain_method="Self-sup. (DINOv1)",
    ),
    # ════ EfficientNet ════
    "efficientnetb0": ModelRegistryEntry(
        builder=models.efficientnet_b0,
        weights=EfficientNet_B0_Weights.DEFAULT,
        classifier="classifier",
        description="EfficientNet-B0 — ImageNet",
        architecture="EfficientNet-B0",
        pretrain_data="ImageNet",
        pretrain_method="Supervised",
    ),
    "efficientnetb1": ModelRegistryEntry(
        builder=models.efficientnet_b1,
        weights=EfficientNet_B1_Weights.DEFAULT,
        classifier="classifier",
        img_size=240,
        description="EfficientNet-B1 — ImageNet",
        architecture="EfficientNet-B1",
        pretrain_data="ImageNet",
        pretrain_method="Supervised",
    ),
    "efficientnetb4": ModelRegistryEntry(
        builder=models.efficientnet_b4,
        weights=EfficientNet_B4_Weights.DEFAULT,
        classifier="classifier",
        img_size=380,
        description="EfficientNet-B4 — ImageNet",
        architecture="EfficientNet-B4",
        pretrain_data="ImageNet",
        pretrain_method="Supervised",
    ),
    # ════ ViT-Base/16 ════
    "vitb16_imagenet_sup": ModelRegistryEntry(
        builder=models.vit_b_16,
        weights=ViT_B_16_Weights.IMAGENET1K_V1,
        classifier="heads.head",
        description="ViT-Base/16 — ImageNet / Supervised",
        architecture="ViT-Base/16",
        pretrain_data="ImageNet",
        pretrain_method="Supervised",
    ),
    "vitb16_imagenet": ModelRegistryEntry(
        builder=torch.hub.load,
        weights="facebookresearch/dino:main",
        hub_model="dino_vitb16",
        classifier="head",
        description="ViT-Base/16 — ImageNet / DINOv1",
        architecture="ViT-Base/16",
        pretrain_data="ImageNet",
        pretrain_method="Self-sup. (DINOv1)",
    ),
    "vitb16_gastronet": ModelRegistryEntry(
        builder=torch.hub.load,
        weights="facebookresearch/dino:main",
        hub_model="dino_vitb16",
        classifier="head",
        gastronet=GASTRONET_WEIGHTS["vitb16"],  # NOTE: DINOv2 GastroNet weights
        description="ViT-Base/16 — GastroNet-5M / DINOv2",
        architecture="ViT-Base/16",
        pretrain_data="GastroNet-5M",
        pretrain_method="Self-sup. (DINOv2)",
    ),
    # ════ ViT-Small/16 ════
    "vits16_imagenet_hf": ModelRegistryEntry(
        builder=None,
        weights="timm/vit_small_patch16_224.augreg_in1k",
        classifier="head",
        hf_model_id="timm/vit_small_patch16_224.augreg_in1k",
        description="ViT-Small/16 — ImageNet / Supervised (timm)",
        architecture="ViT-Small/16",
        pretrain_data="ImageNet",
        pretrain_method="Supervised",
    ),
    "vits16_imagenet": ModelRegistryEntry(
        builder=torch.hub.load,
        weights="facebookresearch/dino:main",
        hub_model="dino_vits16",
        classifier="head",
        description="ViT-Small/16 — ImageNet / DINOv1",
        architecture="ViT-Small/16",
        pretrain_data="ImageNet",
        pretrain_method="Self-sup. (DINOv1)",
    ),
    "vits16_gastronet": ModelRegistryEntry(
        builder=torch.hub.load,
        weights="facebookresearch/dino:main",
        hub_model="dino_vits16",
        classifier="head",
        gastronet=GASTRONET_WEIGHTS["vits16"],
        description="ViT-Small/16 — GastroNet-5M / DINOv1",
        architecture="ViT-Small/16",
        pretrain_data="GastroNet-5M",
        pretrain_method="Self-sup. (DINOv1)",
    ),
}


def get_model_entry(model_name: str) -> ModelRegistryEntry:
    """Get model registry entry by name."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Model '{model_name}' not found. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_name]


def get_img_size(model_name: str) -> int:
    """Get input image size for a model."""
    return get_model_entry(model_name).img_size
