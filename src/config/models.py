"""Model registry and configuration.

Backbone sources
----------------
- torchvision models are built with their weight enums (``weights=...``).
- DINO ResNet-50 is built locally as ``torchvision.models.resnet50(weights=None)``
  with ``fc = Identity`` and the official DINO ImageNet checkpoint, which is what
  ``torch.hub.load("facebookresearch/dino", "dino_resnet50")`` does, without running
  remote code or the deprecated ``pretrained=`` argument.
- DINO ViTs are fetched with ``torch.hub.load`` from the official
  ``facebookresearch/dino`` repository, pinned to commit ``DINO_HUB_COMMIT``
  (``trust_repo=True``; ``skip_validation=True`` because torch.hub only validates
  branch and tag names). The code is cached under
  ``<torch hub dir>/facebookresearch_dino_<commit>``.
- DINOv2 / DINOv3 ViTs are timm models from the HuggingFace hub (``hf_model_id``),
  built at ``img_size`` with ``timm_kwargs``. DINOv3 weights are under the DINOv3
  licence (timm copy not gated); DINOv2 weights are Apache-2.0.

GastroNet entries then overwrite every backbone tensor from the ``.pth`` file in
``PathConfig.pretrained_dir`` (see ``ClassifierModel._load_external_weights``). A timm
GastroNet entry is built without downloading the hub weights, and timm's checkpoint filter
converts the file (DINOv2 keys, register tokens, position grid).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

DINO_HUB_COMMIT = "7c446df5b9f45747937fb0d72314eb9f7b66930a"
"""HEAD of github.com/facebookresearch/dino (main) at the time of pinning."""
DINO_HUB_REPO = f"facebookresearch/dino:{DINO_HUB_COMMIT}"
"""torch.hub reference used for every DINO ViT backbone (pinned commit, not a branch)."""
DINO_RESNET50_URL = (
    "https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth"
)
"""Official DINO ImageNet ResNet-50 checkpoint (same file the hub entry downloads)."""


def dino_resnet50(weights: str | None = DINO_RESNET50_URL) -> nn.Module:
    """ResNet-50 with ``fc = Identity``, optionally initialised from a DINO checkpoint URL.

    Equivalent to ``torch.hub.load("facebookresearch/dino:main", "dino_resnet50")``
    but built from torchvision directly (``weights=None`` instead of the deprecated
    ``pretrained=False``) so no remote hub code is executed.

    Args:
        weights: URL of a DINO ResNet-50 state dict, or None for random
                 initialisation (GastroNet entries overwrite every tensor anyway).
    """
    model = models.resnet50(weights=None)
    model.fc = nn.Identity()
    if weights is not None:
        state_dict = torch.hub.load_state_dict_from_url(
            weights, map_location="cpu", weights_only=True
        )
        model.load_state_dict(state_dict, strict=True)
    return model


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
    """Second positional arg to torch.hub.load. None for torchvision / local builders."""
    hf_model_id: str | None = None
    """timm model on the HuggingFace hub (``timm/...``). When set, builder/hub_model are ignored."""
    timm_kwargs: dict = field(default_factory=dict)
    """Extra ``timm.create_model`` arguments (``img_size`` is always passed)."""
    gastronet: Path | None = None
    """External backbone weights, relative to ``PathConfig.pretrained_dir``."""
    img_size: int = 256
    """Input size (GastroNet-5M fine-tuning protocol)."""
    mean: tuple[float, float, float] = IMAGENET_MEAN
    """Per-channel input normalisation mean used during pretraining."""
    std: tuple[float, float, float] = IMAGENET_STD
    """Per-channel input normalisation std used during pretraining."""
    description: str = ""
    architecture: str = ""
    """Human-readable architecture name for display (e.g. 'ViT-Base/16')."""
    pretrain_data: str = ""
    """Pretraining dataset (e.g. 'ImageNet', 'GastroNet-5M')."""
    pretrain_method: str = ""
    """Pretraining methodology (e.g. 'Supervised', 'Self-sup. (DINOv1)')."""

    @property
    def is_vit(self) -> bool:
        """Vision Transformer backbone: ``drop_path_rate`` and ``layer_decay`` apply."""
        return self.architecture.startswith("ViT")

    def __str__(self) -> str:
        return self.description


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    model: str = "resnet50_5M"
    """Model name from MODEL_REGISTRY."""

    num_classes: int = 1
    """1 for binary (BCE), 2+ for multi-class (CrossEntropy)."""

    freeze_layers: int = 0
    """
    Layer freezing strategy:
    - 0: full fine-tuning (no freezing)
    - -1: linear probe — backbone frozen (requires_grad=False, kept in eval mode), head trained
    - N: freeze the first N layer groups (ResNet: stem, layer1..4; EfficientNet: features[i];
      ViT: patch/position embeddings + first N transformer blocks)
    """

    threshold: float = 0.5
    """Binary decision threshold. Ignored for multi-class."""

    dropout_rate: float = 0.5
    """Dropout probability before classification head."""

    drop_path_rate: float = 0.0
    """ViT backbones only: stochastic depth, rising linearly over the blocks to this rate. 0 = off."""

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

        if not 0 <= self.drop_path_rate < 1:
            raise ConfigurationError(f"drop_path_rate must be in [0, 1), got {self.drop_path_rate}")


# ════════════════════════════════════════════════════════════════════════════════
# Model Registry Definition
# ════════════════════════════════════════════════════════════════════════════════

GASTRONET_WEIGHTS = {
    "vits16": Path("VITS_GastroNet-5M_DINOv1.pth"),
    "resnet50_1M": Path("RN50_GastroNet-1M_DINOv1.pth"),
    "resnet50_5M": Path("RN50_GastroNet-5M_DINOv1.pth"),
    "resnet50_200K": Path("RN50_GastroNet-200K_DINOv1.pth"),
    "vitb14": Path("DINOv2.pth"),  # ViT-B/14 with 4 registers, pretrained at 336 px
}

MODEL_REGISTRY: dict[str, ModelRegistryEntry] = {
    # ════ ResNet-50 (DINOv1) ════
    "resnet50_imagenet": ModelRegistryEntry(
        builder=dino_resnet50,
        weights=DINO_RESNET50_URL,
        classifier="fc",
        description="ResNet-50 — ImageNet / DINOv1",
        architecture="ResNet-50",
        pretrain_data="ImageNet",
        pretrain_method="Self-sup. (DINOv1)",
    ),
    "resnet50_1M": ModelRegistryEntry(
        builder=dino_resnet50,
        weights=None,  # every tensor is overwritten by the GastroNet file
        classifier="fc",
        gastronet=GASTRONET_WEIGHTS["resnet50_1M"],
        description="ResNet-50 — GastroNet-1M",
        architecture="ResNet-50",
        pretrain_data="GastroNet-1M",
        pretrain_method="Self-sup. (DINOv1)",
    ),
    "resnet50_5M": ModelRegistryEntry(
        builder=dino_resnet50,
        weights=None,  # every tensor is overwritten by the GastroNet file
        classifier="fc",
        gastronet=GASTRONET_WEIGHTS["resnet50_5M"],
        description="ResNet-50 — GastroNet-5M",
        architecture="ResNet-50",
        pretrain_data="GastroNet-5M",
        pretrain_method="Self-sup. (DINOv1)",
    ),
    "resnet50_200K": ModelRegistryEntry(
        builder=dino_resnet50,
        weights=None,  # every tensor is overwritten by the GastroNet file
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
    # ════ ViT-Small/16 (DINOv1) ════
    "vits16_imagenet": ModelRegistryEntry(
        builder=torch.hub.load,
        weights=DINO_HUB_REPO,
        hub_model="dino_vits16",
        classifier="head",
        description="ViT-Small/16 — ImageNet / DINOv1",
        architecture="ViT-Small/16",
        pretrain_data="ImageNet",
        pretrain_method="Self-sup. (DINOv1)",
    ),
    "vits16_gastronet": ModelRegistryEntry(
        builder=torch.hub.load,
        weights=DINO_HUB_REPO,
        hub_model="dino_vits16",
        classifier="head",
        gastronet=GASTRONET_WEIGHTS["vits16"],
        description="ViT-Small/16 — GastroNet-5M / DINOv1",
        architecture="ViT-Small/16",
        pretrain_data="GastroNet-5M",
        pretrain_method="Self-sup. (DINOv1)",
    ),
    # ════ Generic DINOv2 / DINOv3 (timm) ════
    "vits14_dinov2": ModelRegistryEntry(
        builder=None,
        weights=None,
        classifier="head",
        hf_model_id="timm/vit_small_patch14_dinov2.lvd142m",
        description="ViT-Small/14 — LVD-142M / DINOv2",
        architecture="ViT-Small/14",
        pretrain_data="LVD-142M",
        pretrain_method="Self-sup. (DINOv2)",
    ),
    "vits16_dinov3": ModelRegistryEntry(
        builder=None,
        weights=None,
        classifier="head",
        hf_model_id="timm/vit_small_patch16_dinov3.lvd1689m",
        timm_kwargs={"global_pool": "token"},  # CLS token, as for DINOv1 / DINOv2
        description="ViT-Small/16 — LVD-1689M / DINOv3",
        architecture="ViT-Small/16",
        pretrain_data="LVD-1689M",
        pretrain_method="Self-sup. (DINOv3)",
    ),
    # ════ ViT-Base (exploratory): DINOv2 GastroNet-5M vs LVD-142M, generic DINOv3 ════
    # 336 px, the native grid of the GastroNet file, for all three.
    "vitb14_gastronet": ModelRegistryEntry(
        builder=None,
        weights=None,
        classifier="head",
        hf_model_id="timm/vit_base_patch14_reg4_dinov2.lvd142m",  # architecture only
        gastronet=GASTRONET_WEIGHTS["vitb14"],
        img_size=336,
        description="ViT-Base/14 — GastroNet-5M / DINOv2",
        architecture="ViT-Base/14",
        pretrain_data="GastroNet-5M",
        pretrain_method="Self-sup. (DINOv2)",
    ),
    "vitb14_dinov2": ModelRegistryEntry(
        builder=None,
        weights=None,
        classifier="head",
        hf_model_id="timm/vit_base_patch14_reg4_dinov2.lvd142m",
        img_size=336,
        description="ViT-Base/14 — LVD-142M / DINOv2",
        architecture="ViT-Base/14",
        pretrain_data="LVD-142M",
        pretrain_method="Self-sup. (DINOv2)",
    ),
    "vitb16_dinov3": ModelRegistryEntry(
        builder=None,
        weights=None,
        classifier="head",
        hf_model_id="timm/vit_base_patch16_dinov3.lvd1689m",
        timm_kwargs={"global_pool": "token"},  # CLS token, as for DINOv1 / DINOv2
        img_size=336,
        description="ViT-Base/16 — LVD-1689M / DINOv3",
        architecture="ViT-Base/16",
        pretrain_data="LVD-1689M",
        pretrain_method="Self-sup. (DINOv3)",
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


def get_norm_stats(
    model_name: str,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return ``(mean, std)`` input normalisation used when the backbone was pretrained.

    ImageNet statistics for every registered backbone.
    """
    entry = get_model_entry(model_name)
    return tuple(entry.mean), tuple(entry.std)  # type: ignore[return-value]
