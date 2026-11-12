"""
src/models/classifier.py
========================
ClassifierModel wraps any backbone declared in MODEL_REGISTRY and provides:
  - registry-driven construction (no hard-coded model names)
  - external (GastroNet) weight loading: unwraps teacher/student/model/state_dict
    wrappers, strips ``module.`` / ``backbone.`` prefixes, drops projection-head keys
    and refuses to continue when any backbone tensor is missing
  - layer freezing: full fine-tuning / linear probe (backbone frozen and kept in
    eval mode) / first N layer groups
  - per-epoch train / validate loops with separate backbone and head learning rates
    (ViTs: optional stochastic depth and layer-wise LR decay)
  - one checkpoint directory per run with best.pt, history.json and run_config.json
  - test evaluation with point estimates + bootstrap CIs and optional clip-level
    aggregation
"""

from __future__ import annotations

import importlib
import json
import math
import os
import re
import time
import warnings
from collections import defaultdict
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from src.config import PathConfig, get_model_entry
from src.config.models import HeadType
from src.evaluation.aggregation import aggregate_frame_to_clip
from src.evaluation.metrics import compute_metrics_with_ci
from src.utils import ConfigurationError
from src.utils.logging import setup_logging

logger = setup_logging(__name__)


# Wrapper keys unwrapped (in this order) from external checkpoints, e.g. DINO's
# {"teacher": {...}, "student": {...}}. The teacher is the network DINO evaluates.
_WRAPPER_KEYS = ("teacher", "state_dict", "model", "student")
# Prefixes stripped from every key, e.g. "module.backbone.conv1.weight" -> "conv1.weight".
_STRIP_PREFIXES = ("module.", "backbone.")
# Projection / classification heads that are never part of the backbone.
_HEAD_KEY_PREFIXES = ("head.", "dino_head.", "ibot_head.", "fc.")


class _TimmBackbone(nn.Module):
    """timm backbone (``timm/...`` on the HuggingFace hub) returning pooled features.

    ``head = Identity`` makes _replace_last_layer wrap it with _HeadedBackbone,
    the same path as the DINO ViTs. ``pretrained=False`` builds the architecture only,
    for external weights loaded afterwards (see :meth:`convert`).
    """

    def __init__(self, model_id: str, img_size: int, pretrained: bool = True, **kwargs) -> None:
        super().__init__()
        import timm  # lazy import

        self.vit = timm.create_model(
            f"hf_hub:{model_id}", pretrained=pretrained, num_classes=0, img_size=img_size, **kwargs
        )
        self.head = nn.Identity()  # sentinel for _replace_last_layer

    def convert(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """External backbone weights (e.g. original DINOv2 keys) in the timm format of ``vit``.

        Applies the ``checkpoint_filter_fn`` of the model's timm module, as timm does for its
        own pretrained checkpoints: DINOv2 ``register_tokens`` become ``reg_token``,
        ``mask_token`` is dropped and the position embedding is resampled to the model grid.
        """
        module = importlib.import_module(type(self.vit).__module__)
        return module.checkpoint_filter_fn(dict(state_dict), self.vit)

    @property
    def pretrained_norm(self) -> tuple[tuple, tuple] | None:
        """(mean, std) declared by the timm pretrained config, if available."""
        cfg = getattr(self.vit, "pretrained_cfg", None)
        if isinstance(cfg, Mapping) and cfg.get("mean") is not None and cfg.get("std") is not None:
            return tuple(cfg["mean"]), tuple(cfg["std"])
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit(x)


def _resample_dino_pos_embed(pos_embed: torch.Tensor, grid: int) -> torch.Tensor:
    """Bicubic resampling of a DINOv1 position embedding to a ``grid`` x ``grid`` patch grid.

    Same call as DINO's ``interpolate_pos_encoding`` (scale factor with the +0.1
    offset), so the result equals what the hub model computes on every forward.
    """
    class_pos, patch_pos = pos_embed[:, :1], pos_embed[:, 1:]
    n, dim = patch_pos.shape[1], patch_pos.shape[2]
    side = math.sqrt(n)
    patch_pos = nn.functional.interpolate(
        patch_pos.reshape(1, int(side), int(side), dim).permute(0, 3, 1, 2),
        scale_factor=((grid + 0.1) / side, (grid + 0.1) / side),
        mode="bicubic",
    )
    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, dim)
    return torch.cat((class_pos, patch_pos), dim=1)


def _fix_dino_pos_embed(vit: nn.Module, img_size: int) -> None:
    """Resample a DINOv1 hub ViT's position embedding once to the ``img_size`` grid.

    The hub model (pretrained at 224 px) otherwise re-interpolates it bicubically on
    every forward, whose CUDA backward has no deterministic kernel. Afterwards
    ``interpolate_pos_encoding`` returns the parameter unchanged. A load_state_dict
    pre-hook resamples incoming 224-px embeddings (GastroNet weights, checkpoints
    saved before this change) the same way.
    """
    grid = img_size // vit.patch_embed.patch_size
    if vit.pos_embed.shape[1] == grid * grid + 1:
        return
    with torch.no_grad():
        vit.pos_embed = nn.Parameter(_resample_dino_pos_embed(vit.pos_embed, grid))

    def resample_incoming(module, state_dict, prefix, *args) -> None:
        key = f"{prefix}pos_embed"
        if key in state_dict and state_dict[key].shape[1] != module.pos_embed.shape[1]:
            with torch.no_grad():
                state_dict[key] = _resample_dino_pos_embed(state_dict[key].float(), grid)

    vit.register_load_state_dict_pre_hook(resample_incoming)


class _HeadedBackbone(nn.Module):
    """Wraps a backbone whose forward() does not call its classification head.

    DINO ViT models return the raw CLS token (shape [B, embed_dim]) without
    passing through self.head.  This wrapper makes the head an explicit call.
    """

    def __init__(self, backbone: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


class _TrainingInterrupted(Exception):
    """Raised when a KeyboardInterrupt occurs inside fit().

    checkpoint_dir : set by fit() so callers can log best.pt as an artifact.
    result         : set by run_data_efficiency callers to propagate partial results.
    """

    def __init__(
        self,
        checkpoint_dir: Path | None = None,
        result: dict | None = None,
    ) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.result = result


class FitResult(tuple):
    """Return value of :meth:`ClassifierModel.fit`.

    Unpacks like the historical 3-tuple ``(val_probs, val_labels, checkpoint_dir)``
    and exposes the training outcome as attributes:

        best_val_auroc : val AUROC at the best epoch (NaN if never computable)
        best_val_f1    : val F1@monitor-threshold at the best epoch
        best_epoch     : 1-based epoch of the restored checkpoint (None if none saved)
        stopped_epoch  : last epoch that was evaluated
        stop_reason    : "max_epochs" | "early_stopping" | "nan_val"
                         (``ClassifierModel.stop_reason`` can also be "nan_train_loss"
                         or "interrupted"; fit() then raises instead of returning)
        history        : list of per-epoch records (same content as history.json)
    """

    def __new__(
        cls,
        val_probs: np.ndarray | None = None,
        val_labels: np.ndarray | None = None,
        checkpoint_dir: Path | None = None,
        *,
        best_val_auroc: float = float("nan"),
        best_val_f1: float = float("nan"),
        best_epoch: int | None = None,
        stopped_epoch: int | None = None,
        stop_reason: str | None = None,
        history: list[dict] | None = None,
    ) -> FitResult:
        obj = super().__new__(cls, (val_probs, val_labels, checkpoint_dir))
        obj.best_val_auroc = best_val_auroc
        obj.best_val_f1 = best_val_f1
        obj.best_epoch = best_epoch
        obj.stopped_epoch = stopped_epoch
        obj.stop_reason = stop_reason
        obj.history = history if history is not None else []
        return obj

    def __getnewargs__(self) -> tuple:
        # tuple's default passes the whole 3-tuple as one argument, which would put it
        # in val_probs on pickle/deepcopy. Attributes come back from __dict__.
        return tuple(self)

    @property
    def val_probs(self) -> np.ndarray | None:
        return self[0]

    @property
    def val_labels(self) -> np.ndarray | None:
        return self[1]

    @property
    def checkpoint_dir(self) -> Path | None:
        return self[2]


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _extract_backbone_state_dict(
    checkpoint: Any, drop_prefixes: tuple[str, ...] = ()
) -> dict[str, torch.Tensor]:
    """Return a flat backbone state dict from a raw external checkpoint.

    Unwraps (possibly nested) ``teacher`` / ``state_dict`` / ``model`` / ``student``
    wrappers, strips ``module.`` and ``backbone.`` prefixes (repeatedly, so
    ``module.backbone.x`` → ``x``), drops projection-head keys (``head.``,
    ``dino_head.``, ``ibot_head.``, ``fc.`` plus ``drop_prefixes``) and non-tensor
    entries.
    """
    head_prefixes = _HEAD_KEY_PREFIXES + tuple(drop_prefixes)
    state = checkpoint
    unwrapped = True
    while isinstance(state, Mapping) and unwrapped:
        unwrapped = False
        for key in _WRAPPER_KEYS:
            inner = state.get(key)
            if isinstance(inner, Mapping):
                state = inner
                unwrapped = True
                break
    if not isinstance(state, Mapping):
        raise TypeError(f"Expected a state dict, got {type(state).__name__}.")

    out: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            continue
        name = str(key)
        stripped = True
        while stripped:
            stripped = False
            for prefix in _STRIP_PREFIXES:
                if name.startswith(prefix):
                    name = name[len(prefix) :]
                    stripped = True
        if name.startswith(head_prefixes):
            continue
        out[name] = value
    return out


def _modules_outside(root: nn.Module, dotted_path: str) -> list[nn.Module]:
    """Children of ``root`` (recursively along ``dotted_path``) that do not contain it.

    E.g. for ``root=ResNet`` and ``"fc"`` → every top-level child except ``fc``;
    for ``root=VisionTransformer`` and ``"heads.head"`` → ``conv_proj``, ``encoder``
    and any sibling of ``head`` inside ``heads``.
    """
    outside: list[nn.Module] = []
    module = root
    for part in dotted_path.split("."):
        for name, child in module.named_children():
            if name != part:
                outside.append(child)
        module = getattr(module, part)
    return outside


def _vit_embedding_params(vit: nn.Module) -> list[nn.Parameter]:
    """Token and position embeddings of a DINO / timm ViT (the patch embedding excluded)."""
    return [
        p
        for attr in ("cls_token", "pos_embed", "reg_token", "dist_token")
        if isinstance(p := getattr(vit, attr, None), nn.Parameter)
    ]


def _vit_layer_ids(vit: nn.Module) -> dict[int, int]:
    """Layer of every ViT parameter, keyed by ``id(param)``: 0 = patch / token / position
    embeddings, i + 1 = block i, ``len(vit.blocks) + 1`` = everything after the blocks."""
    depth = len(vit.blocks)
    layers = {id(p): depth + 1 for p in vit.parameters()}
    for p in [*vit.patch_embed.parameters(), *_vit_embedding_params(vit)]:
        layers[id(p)] = 0
    for i, block in enumerate(vit.blocks):
        for p in block.parameters():
            layers[id(p)] = i + 1
    return layers


def _safe_run_name(run_name: str) -> str:
    """Filesystem-safe version of a run name (letters, digits, ``._-``)."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", run_name).strip("._") or "run"


def _finite_or_none(value: Any) -> float | None:
    """Float for JSON, with NaN / inf mapped to None (strict JSON has no NaN)."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _jsonable(obj: Any) -> Any:
    """Recursively convert ``obj`` into strict-JSON-serialisable values."""
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return _finite_or_none(obj)
    if isinstance(obj, Path):
        return obj.as_posix()
    if obj is None or isinstance(obj, str):
        return obj
    return str(obj)


def _describe_loader(loader: Any) -> dict:
    """Best-effort description (img_size, equalize, mean, std) of a loader's preprocessing.

    With a frame cache (``cache_<size>[_clahe]``) the resize and CLAHE steps are read
    from the cache folder name, since the transform then starts after them.
    """
    dataset = getattr(loader, "dataset", None)
    info: dict = {}
    steps = getattr(getattr(dataset, "transform", None), "transforms", None)
    if steps is None:
        return info
    info["equalize"] = False
    cache_dir = getattr(dataset, "cache_dir", None)
    if cache_dir is not None:
        m = re.fullmatch(r"cache_(\d+)(_clahe)?", Path(cache_dir).name)
        if m:
            info.update(img_size=int(m.group(1)), equalize=bool(m.group(2)), frame_cache=True)
    for step in steps:
        cls_name = type(step).__name__
        if cls_name == "ResizeWithPad" and hasattr(step, "target_size"):
            info["img_size"] = int(step.target_size)
        elif "CLAHE" in cls_name.upper():
            info["equalize"] = True
        elif cls_name == "Normalize":
            info["mean"] = [float(v) for v in step.mean]
            info["std"] = [float(v) for v in step.std]
    return info


ACTIVE_CUTOFF = 2
"""MES grades >= 2 are active disease (Mayo 2-3 vs remission 0-1)."""


class _ActiveDiseaseLoss(nn.Module):
    """Weighted CE + ``active_weight`` x CE of active disease (class >= ACTIVE_CUTOFF) on p2 + p3.

    The added term uses the CE's per-sample class weights and normalisation, so both terms
    have the same optimum.
    """

    def __init__(self, weight: torch.Tensor, label_smoothing: float, active_weight: float):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
        self.register_buffer("weight", weight)
        self.active_weight = float(active_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_p = logits.float().log_softmax(dim=1)
        log_group = torch.where(
            targets >= ACTIVE_CUTOFF,
            torch.logsumexp(log_p[:, ACTIVE_CUTOFF:], dim=1),
            torch.logsumexp(log_p[:, :ACTIVE_CUTOFF], dim=1),
        )
        w = self.weight[targets]
        return self.ce(logits, targets) - self.active_weight * (w * log_group).sum() / w.sum()


class ClassifierModel(nn.Module):
    """
    Unified classifier wrapping any backbone declared in MODEL_REGISTRY.

    Args:
        base_model:      Key in MODEL_REGISTRY (e.g. 'vits16_gastronet').
        num_classes:     1 = binary (BCEWithLogitsLoss), >=2 = CrossEntropyLoss.
        optimizer:       'Adam' | 'AdamW'.
        learning_rate:   Initial backbone LR (the head uses learning_rate * head_lr_scale).
        class_weights:   None/False → auto-computed balanced weights from the training
                         set (binary: pos_weight = n_neg / n_pos);
                         float → manual pos_weight (binary only);
                         list  → per-class weights (multiclass).
        threshold:       Decision threshold (binary only).
        dropout_rate:    Dropout probability in the classification head.
        drop_path_rate:  ViT backbones only (else 0): stochastic depth, rising linearly over
                         the blocks to this rate.
        num_epochs:      Maximum training epochs (display only; fit() takes ``epochs``).
        freeze_layers:   0  = full fine-tuning
                         -1 = linear probe: backbone frozen (requires_grad=False) and kept
                              in eval mode during training (BN statistics, dropout and
                              stochastic depth are fixed); only the head trains
                         N  = freeze the first N layer groups
                              (ResNet: stem, layer1..layer4; EfficientNet: features[i];
                              ViT: patch/position embeddings + first N blocks).
                              Frozen groups are also kept in eval mode.
        gastronet_path:  External backbone weights: filename relative to
                         ``PathConfig.pretrained_dir`` (absolute paths also accepted),
                         or None.
        es_patience:     Early-stopping patience (epochs without val-AUROC gain).
        es_start_epoch:  Patience counts only the epochs after this one (0 = from the first).
        grad_clip_norm:  Max global gradient norm per step.
        weight_decay:    Optimizer L2 regularisation.
        label_smoothing: Label smoothing for CrossEntropyLoss (multi-class only).
        active_loss_weight: > 2 classes only: adds this weight x CE of active disease
                         (classes >= 2 vs the rest, on p2 + p3); 0 = plain CE.
        label_col:       Label column of the training DataFrame used for class weights.
        head_type:       'linear' | 'mlp1' | 'mlp2'.
        random_seed:     Seed of the bootstrap resampling in test_evaluation (global RNG
                         seeding is the caller's job, see src.utils.seed).
        warmup_epochs:   Linear LR warmup epochs before cosine annealing.
        min_lr:          Cosine annealing floor (eta_min).
        n_bootstrap:     Bootstrap resamples for test-set CIs.
        ci:              CI level, fraction (0.95) or percentage (95).
        head_lr_scale:   Head LR = learning_rate * head_lr_scale (separate param group).
        layer_decay:     ViT backbones only (else 1): layer-wise LR decay, layer k of
                         ``_vit_layer_ids`` at learning_rate * layer_decay ** (depth + 1 - k).
        use_amp:         Default for fit(use_amp=None): mixed precision on CUDA.
        models_dir:      Default checkpoint root (fit(checkpoint_root=...) overrides it).
                         None → PathConfig().ulcer_models_dir.
    """

    def __init__(
        self,
        base_model: str,
        num_classes: int,
        optimizer: str,
        learning_rate: float,
        class_weights: float | list[float] | None = None,
        threshold: float = 0.5,
        dropout_rate: float = 0.5,
        drop_path_rate: float = 0.0,
        num_epochs: int = 100,
        freeze_layers: int = 0,
        gastronet_path: Path | None = None,
        es_patience: int = 20,
        es_start_epoch: int = 0,
        grad_clip_norm: float = 10.0,
        weight_decay: float = 0.01,
        label_smoothing: float = 0.0,
        active_loss_weight: float = 0.0,
        label_col: str = "label",
        head_type: str = "linear",  # "linear" | "mlp1" | "mlp2"
        random_seed: int = 42,
        warmup_epochs: int = 5,
        min_lr: float = 1e-7,
        n_bootstrap: int = 1000,
        ci: float = 0.95,
        head_lr_scale: float = 1.0,
        layer_decay: float = 1.0,
        use_amp: bool = True,
        models_dir: Path | str | None = None,
    ):
        super().__init__()

        paths = PathConfig()
        self.paths = paths
        self.random_seed = random_seed

        entry = get_model_entry(base_model)

        self.name = base_model
        self.number_epochs = num_epochs
        self.number_classes = num_classes
        self.class_weights = class_weights
        self.lr = learning_rate
        self.threshold = threshold if num_classes == 1 else None
        self.clip_threshold: float = threshold if num_classes == 1 else 0.5
        self.optimizer = optimizer
        self.es_patience = es_patience
        self.es_start_epoch = int(es_start_epoch)
        self.grad_clip_norm = float(grad_clip_norm)
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        if active_loss_weight < 0:
            raise ValueError(f"active_loss_weight must be >= 0, got {active_loss_weight}")
        self.active_loss_weight = float(active_loss_weight) if num_classes > ACTIVE_CUTOFF else 0.0
        self.label_col = label_col
        self.head_type = head_type
        self.freeze_layers = freeze_layers
        self.dropout_rate = dropout_rate
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.n_bootstrap = int(n_bootstrap)
        ci = float(ci)
        self.ci = ci / 100.0 if ci > 1.0 else ci
        if not 0.0 < self.ci < 1.0:
            raise ValueError(f"ci must be a fraction in (0, 1) or a percentage, got {ci}")
        self.head_lr_scale = float(head_lr_scale)
        if self.head_lr_scale <= 0:
            raise ValueError(f"head_lr_scale must be > 0, got {head_lr_scale}")
        if not 0 <= drop_path_rate < 1:
            raise ValueError(f"drop_path_rate must be in [0, 1), got {drop_path_rate}")
        if not 0 < layer_decay <= 1:
            raise ValueError(f"layer_decay must be in (0, 1], got {layer_decay}")
        # ViT-only settings, recorded as off for the other backbones.
        self.drop_path_rate = float(drop_path_rate) if entry.is_vit else 0.0
        self.layer_decay = float(layer_decay) if entry.is_vit else 1.0
        self.use_amp = bool(use_amp)
        self.models_dir = Path(models_dir) if models_dir is not None else paths.ulcer_models_dir
        self.img_size: int = entry.img_size
        self.norm_mean: tuple[float, float, float] = tuple(entry.mean)  # type: ignore[assignment]
        self.norm_std: tuple[float, float, float] = tuple(entry.std)  # type: ignore[assignment]
        self.gastronet_file: str | None = (
            Path(gastronet_path).name if gastronet_path is not None else None
        )

        # Training outcome (set by fit()).
        self.best_val_auroc: float = float("nan")
        self.best_val_f1: float = float("nan")
        self.best_epoch: int | None = None
        self.stopped_epoch: int | None = None
        self.stop_reason: str | None = None
        self.checkpoint_dir: Path | None = None
        self.history: list[dict] = []
        self.pos_weight: float | None = None
        self._frozen_modules: list[nn.Module] = []

        # ── 1. Build backbone ─────────────────────────────────────────
        self.base_model = self._build_backbone(entry)
        self._check_backbone_norm(entry)

        # ── 2. Load GastroNet (or any external) weights ───────────────
        if gastronet_path is not None:
            self._load_external_weights(gastronet_path, paths)
        elif entry.gastronet is not None:
            logger.warning(
                f"'{base_model}' declares external weights ({entry.gastronet}) but "
                "gastronet_path=None: the backbone keeps its builder initialisation. "
                "This is only correct when a full checkpoint is loaded afterwards."
            )

        # ── 3. Replace classifier head ────────────────────────────────
        self._replace_last_layer(num_classes, dropout_rate)

        # ── 4. Freeze layers ──────────────────────────────────────────
        self._freeze_layers(freeze_layers)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _build_backbone(self, entry) -> nn.Module:
        """Instantiate the backbone from a registry entry.

        Priority: hf_model_id > hub_model (torch.hub) > local / torchvision builder.
        Stochastic depth (``drop_path_rate``) is a constructor argument of the DINO and
        timm ViTs.
        """
        vit_kwargs = {"drop_path_rate": self.drop_path_rate} if self.drop_path_rate > 0 else {}
        if entry.hf_model_id is not None:
            # GastroNet entries: every tensor comes from the external file, so no hub weights.
            return _TimmBackbone(entry.hf_model_id, entry.img_size, pretrained=entry.gastronet is None,
                                 **entry.timm_kwargs, **vit_kwargs)
        if entry.hub_model is not None:
            # entry.weights is an official repo pinned to a commit SHA: trust it
            # explicitly, and skip the branch/tag lookup, which only knows branch heads.
            # verbose=False: the "Using cache found in <abs path>" line would put the
            # local home directory into notebook outputs and run logs.
            backbone = entry.builder(
                entry.weights,
                entry.hub_model,
                trust_repo=True,
                skip_validation=True,
                verbose=False,
                **vit_kwargs,
            )
            if hasattr(backbone, "interpolate_pos_encoding"):  # DINOv1 ViT
                _fix_dino_pos_embed(backbone, entry.img_size)
            return backbone
        if vit_kwargs:
            raise ConfigurationError(f"'{self.name}': drop_path_rate needs a DINO hub or timm ViT.")
        return entry.builder(weights=entry.weights)

    def _check_backbone_norm(self, entry) -> None:
        """Warn when the registry mean/std disagree with the timm pretrained config."""
        backbone = self.base_model
        declared = getattr(backbone, "pretrained_norm", None)
        if declared is None:
            return
        mean, std = declared
        if not (
            np.allclose(mean, entry.mean, atol=1e-4) and np.allclose(std, entry.std, atol=1e-4)
        ):
            logger.warning(
                f"'{self.name}': registry normalisation mean={entry.mean} std={entry.std} "
                f"differs from the pretrained config mean={mean} std={std}."
            )

    @staticmethod
    def _resolve_weight_path(gastronet_path: Path | str, paths: PathConfig) -> Path:
        """Filename relative to paths.pretrained_dir; absolute or existing paths used as-is."""
        path = Path(gastronet_path)
        if path.is_absolute():
            return path
        candidate = paths.pretrained_dir / path
        if not candidate.exists() and path.exists():
            return path
        return candidate

    def _load_external_weights(self, gastronet_path: Path | str, paths: PathConfig) -> None:
        """Load GastroNet (or any external) backbone weights into self.base_model.

        Raises:
            FileNotFoundError: weight file does not exist.
            RuntimeError:      a backbone tensor is missing from the file, or shapes differ.
        """
        full_path = self._resolve_weight_path(gastronet_path, paths)
        if not full_path.exists():
            raise FileNotFoundError(f"External weight file not found: {full_path}")
        raw = torch.load(full_path, map_location="cpu", weights_only=True)
        # The registry classifier (e.g. EfficientNet "classifier") is replaced later.
        classifier_prefix = (f"{get_model_entry(self.name).classifier}.",)
        head_prefixes = _HEAD_KEY_PREFIXES + classifier_prefix
        state_dict = _extract_backbone_state_dict(raw, drop_prefixes=classifier_prefix)
        target = self.base_model
        if isinstance(target, _TimmBackbone):  # original keys -> timm format of the inner model
            state_dict, target = target.convert(state_dict), target.vit

        backbone_keys = [k for k in target.state_dict() if not k.startswith(head_prefixes)]
        try:
            missing, unexpected = target.load_state_dict(state_dict, strict=False)
        except RuntimeError as exc:  # size mismatch
            raise RuntimeError(
                f"[GastroNet] {full_path.name} does not match the '{self.name}' backbone: {exc}"
            ) from exc

        missing_backbone = [k for k in missing if not k.startswith(head_prefixes)]
        if missing_backbone:
            raise RuntimeError(
                f"[GastroNet] {full_path.name}: {len(missing_backbone)}/{len(backbone_keys)} "
                f"backbone tensors of '{self.name}' are missing from the checkpoint "
                f"(e.g. {missing_backbone[:3]}). Refusing to continue with a partially "
                "initialised backbone."
            )
        if unexpected:
            logger.warning(
                f"[GastroNet] {len(unexpected)} key(s) in {full_path.name} do not belong to "
                f"the backbone and were ignored (e.g. {list(unexpected)[:3]})."
            )
        logger.info(
            f"[GastroNet] Loaded {len(backbone_keys)}/{len(backbone_keys)} backbone tensors "
            f"from {full_path.name}."
        )

    # ------------------------------------------------------------------
    # Architecture helpers
    # ------------------------------------------------------------------

    def _replace_last_layer(self, num_classes: int, dropout_rate: float):
        classifier_name = get_model_entry(self.name).classifier
        classif = self._get_layer(classifier_name)
        use_wrapper = False

        if isinstance(classif, nn.Linear):
            in_features = int(classif.in_features)
        elif isinstance(classif, nn.Sequential):
            last_layer = classif[-1]
            if not isinstance(last_layer, nn.Linear):
                raise TypeError(
                    f"Expected last layer of '{classifier_name}' to be Linear, got "
                    f"{type(last_layer).__name__}"
                )
            in_features = int(last_layer.in_features)
        elif isinstance(classif, nn.Identity):
            # DINO models / timm wrapper: forward() returns the pooled features without
            # a head. Infer the feature size with a dummy forward pass in eval mode, so
            # no BatchNorm running statistic is touched, then wrap with _HeadedBackbone.
            img_size = get_model_entry(self.name).img_size
            device = next(self.base_model.parameters()).device
            was_training = self.base_model.training
            self.base_model.eval()
            try:
                with torch.no_grad():
                    dummy = torch.zeros(1, 3, img_size, img_size, device=device)
                    in_features = int(self.base_model(dummy).shape[-1])
            finally:
                self.base_model.train(was_training)
            use_wrapper = True
        else:
            raise TypeError(
                f"Unsupported classifier layer type for '{classifier_name}': "
                f"{type(classif).__name__}"
            )

        if self.head_type == "linear":
            new_head = nn.Sequential(
                nn.Dropout(p=dropout_rate, inplace=False),
                nn.Linear(in_features, num_classes),
            )
        elif self.head_type == "mlp1":
            new_head = nn.Sequential(
                nn.Linear(in_features, 256),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(256, num_classes),
            )
        elif self.head_type == "mlp2":
            new_head = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(256, num_classes),
            )
        else:
            choices = [h.value for h in HeadType]
            raise ValueError(f"Unknown head_type '{self.head_type}'. Use one of {choices}.")

        if use_wrapper:
            self.base_model = _HeadedBackbone(self.base_model, new_head)
        else:
            parent = self.base_model
            *parents, last = classifier_name.split(".")
            for part in parents:
                parent = getattr(parent, part)
            setattr(parent, last, new_head)

    def _get_layer(self, dot_path: str) -> nn.Module:
        layer = self.base_model
        for attr in dot_path.split("."):
            layer = getattr(layer, attr)
        return layer

    @property
    def head_prefix(self) -> str:
        """Parameter-name prefix of the classification head inside ``base_model``."""
        if isinstance(self.base_model, _HeadedBackbone):
            return "head"
        return get_model_entry(self.name).classifier

    def _is_head_param(self, name: str) -> bool:
        prefix = self.head_prefix
        return name == prefix or name.startswith(prefix + ".")

    def _layer_groups(self) -> list[tuple[str, list[nn.Module], list[nn.Parameter]]]:
        """Ordered (label, modules, extra parameters) groups used by freeze_layers=N.

        Raises:
            ConfigurationError: partial freezing is not supported for this backbone.
        """
        arch = self.unwrapped_backbone

        # ResNet (torchvision or DINO): stem, layer1..layer4.
        if all(hasattr(arch, a) for a in ("conv1", "bn1", "layer1", "layer4")):
            groups: list[tuple[str, list[nn.Module], list[nn.Parameter]]] = [
                ("stem", [arch.conv1, arch.bn1], [])
            ]
            groups += [(f"layer{i}", [getattr(arch, f"layer{i}")], []) for i in range(1, 5)]
            return groups

        # EfficientNet: features[0] = stem conv, features[1..-2] = MBConv stages,
        # features[-1] = final 1x1 head conv.
        features = getattr(arch, "features", None)
        if isinstance(features, nn.Sequential):
            return [(f"features.{i}", [m], []) for i, m in enumerate(features)]

        vit = self._vit_module()
        if vit is not None:
            blocks = list(vit.blocks)
            groups = [(f"blocks.{i}", [m], []) for i, m in enumerate(blocks)]
            groups[0] = (
                "embeddings+blocks.0",
                [vit.patch_embed, blocks[0]],
                _vit_embedding_params(vit),
            )
            return groups

        # torchvision ViT: conv_proj + class_token + encoder.pos_embedding, encoder.layers.
        encoder = getattr(arch, "encoder", None)
        if hasattr(arch, "conv_proj") and encoder is not None and hasattr(encoder, "layers"):
            embed_params = [
                p
                for p in (
                    getattr(arch, "class_token", None),
                    getattr(encoder, "pos_embedding", None),
                )
                if isinstance(p, nn.Parameter)
            ]
            layers = list(encoder.layers)
            groups = [(f"encoder.layers.{i}", [m], []) for i, m in enumerate(layers)]
            groups[0] = (
                "embeddings+encoder.layers.0",
                [arch.conv_proj, layers[0]],
                embed_params,
            )
            return groups

        raise ConfigurationError(
            f"Partial freezing (freeze_layers=N>0) is not implemented for '{self.name}'. "
            "Use 0 (full fine-tuning) or -1 (linear probe)."
        )

    def _freeze_layers(self, num_layers: int):
        """
        Freeze backbone layers.

            0  → nothing frozen (full fine-tuning)
            -1 → linear probe: every backbone parameter frozen and the backbone kept
                 in eval mode during training; only the head trains
            N  → first N layer groups frozen (see ``_layer_groups``) and kept in eval
                 mode during training

        Raises:
            ConfigurationError: N < -1, N larger than the number of groups, or partial
                freezing unsupported for this backbone.
        """
        self._frozen_modules = []
        if num_layers == 0:
            logger.info("No layers frozen — full fine-tuning.")
            return
        if num_layers < -1:
            raise ConfigurationError(f"freeze_layers must be >= -1, got {num_layers}")

        if num_layers == -1:
            for name, param in self.base_model.named_parameters():
                if not self._is_head_param(name):
                    param.requires_grad = False
            self._frozen_modules = _modules_outside(self.base_model, self.head_prefix)
            n_trainable = sum(p.numel() for p in self.base_model.parameters() if p.requires_grad)
            logger.info(
                "Linear probe — backbone frozen and kept in eval mode; "
                f"{n_trainable:,} trainable head parameters."
            )
            return

        groups = self._layer_groups()
        if num_layers > len(groups):
            raise ConfigurationError(
                f"freeze_layers={num_layers} exceeds the {len(groups)} layer groups of "
                f"'{self.name}' ({[g[0] for g in groups]}). Use -1 to freeze the whole backbone."
            )
        for _label, modules, params in groups[:num_layers]:
            for module in modules:
                for p in module.parameters():
                    p.requires_grad = False
            for p in params:
                p.requires_grad = False
            self._frozen_modules.extend(modules)
        logger.info(
            f"Frozen {num_layers}/{len(groups)} layer groups of '{self.name}': "
            f"{[g[0] for g in groups[:num_layers]]}."
        )

    def _set_train_mode(self) -> None:
        """base_model.train(), with frozen modules kept in eval mode."""
        self.base_model.train()
        for module in self._frozen_modules:
            module.eval()

    def train(self, mode: bool = True) -> ClassifierModel:
        """``nn.Module.train``, keeping frozen backbone modules in eval mode."""
        super().train(mode)
        if mode:
            for module in self._frozen_modules:
                module.eval()
        return self

    @property
    def unwrapped_backbone(self) -> nn.Module:
        """Raw backbone without the _HeadedBackbone wrapper (if any)."""
        if isinstance(self.base_model, _HeadedBackbone):
            return self.base_model.backbone
        return self.base_model

    def _vit_module(self) -> nn.Module | None:
        """The DINO ViT (blocks on the backbone) or timm ViT (``_TimmBackbone.vit``), else None."""
        arch = self.unwrapped_backbone
        vit = arch if hasattr(arch, "blocks") else getattr(arch, "vit", None)
        if vit is not None and hasattr(vit, "blocks") and hasattr(vit, "patch_embed"):
            return vit
        return None

    # ------------------------------------------------------------------
    # Class-weight / criterion / optimizer setup
    # ------------------------------------------------------------------

    def _setup_class_weights(
        self,
        train_df: pd.DataFrame,
        label_col: str = "label",
    ) -> list[float]:
        """Compute per-class weights from training data.

        Binary  (num_classes=1): scalar class_weights → [1.0, w]; None/False → auto.
        Multiclass (num_classes>1): list of correct length → used as-is;
            None or scalar → auto-compute balanced weights from train counts.

        Auto weights are sklearn-style balanced: w_c = n / (n_classes * n_c).
        For binary BCE, ``_create_criterion`` turns them into
        pos_weight = w_1 / w_0 = n_neg / n_pos.

        ``train_df`` is the training subset actually used.
        """
        n_classes = max(self.number_classes, 2)
        cw = self.class_weights

        # User-provided list (multiclass explicit weights)
        if isinstance(cw, list):
            if len(cw) == n_classes:
                return [float(w) for w in cw]
            logger.warning(
                f"class_weights list length {len(cw)} != {n_classes} "
                "— falling back to auto-computed balanced weights."
            )

        # Binary with scalar positive-class weight (bool False/True means "auto")
        is_scalar = isinstance(cw, (float, int)) and not isinstance(cw, bool)
        if self.number_classes == 1 and is_scalar:
            return [1.0, float(cw)]  # type: ignore[arg-type]

        # Auto-compute balanced weights (multiclass default, or binary with None)
        if cw is None or isinstance(cw, (bool, list)) or self.number_classes > 1:
            counts = train_df[label_col].value_counts()
            n = len(train_df)
            weights = []
            for i in range(n_classes):
                cnt = int(counts.get(i, 0))
                if cnt == 0:
                    logger.warning(f"Class {i}: 0 training samples — assigning weight 1.0.")
                    weights.append(1.0)
                else:
                    w = n / (n_classes * cnt)
                    weights.append(w)
                logger.info(f"Class {i}: {cnt} frames, weight {weights[-1]:.4f}")
            return weights

        return [1.0] * n_classes

    def _create_criterion(self, weights: list[float], device: torch.device):
        if self.number_classes == 1:
            # Balanced BCE: relative positive weight w1 / w0 (= n_neg / n_pos for auto
            # weights, = w for a manual scalar since weights = [1.0, w]).
            self.pos_weight = float(weights[1]) / float(weights[0])
            logger.info(f"BCEWithLogitsLoss pos_weight = {self.pos_weight:.4f}")
            return nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(self.pos_weight, dtype=torch.float32, device=device)
            )
        self.pos_weight = None
        weight = torch.tensor(weights).float().to(device)
        if self.active_loss_weight > 0:
            return _ActiveDiseaseLoss(weight, self.label_smoothing, self.active_loss_weight)
        return nn.CrossEntropyLoss(weight=weight, label_smoothing=self.label_smoothing)

    def _param_groups(self) -> list[dict]:
        """Trainable parameters split into a backbone group (lr) and a head group
        (lr * head_lr_scale). With layer_decay < 1 (ViT) the backbone is split by layer
        (``_vit_layer_ids``): layer k at lr * layer_decay ** (depth + 1 - k), named
        ``backbone.<k>``; the layer after the blocks keeps lr and the name ``backbone``.
        Frozen parameters are excluded; empty groups dropped."""
        vit = self._vit_module() if self.layer_decay < 1 else None
        layers = _vit_layer_ids(vit) if vit is not None else {}
        top = len(vit.blocks) + 1 if vit is not None else 0
        backbone: dict[int, list[nn.Parameter]] = defaultdict(list)
        head = []
        for name, param in self.base_model.named_parameters():
            if not param.requires_grad:
                continue
            if self._is_head_param(name):
                head.append(param)
            else:
                backbone[layers.get(id(param), top)].append(param)
        groups = [
            {
                "params": params,
                "lr": self.lr * self.layer_decay ** (top - k),
                "name": "backbone" if k == top else f"backbone.{k}",
            }
            for k, params in sorted(backbone.items())
        ]
        if head:
            groups.append({"params": head, "lr": self.lr * self.head_lr_scale, "name": "head"})
        if not groups:
            raise RuntimeError(f"'{self.name}': no trainable parameters.")
        return groups

    def _create_optimizer(self):
        params = self._param_groups()
        if self.optimizer == "Adam":
            return torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        if self.optimizer == "AdamW":
            return torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        raise ValueError(f"Unknown optimizer '{self.optimizer}'. Use 'Adam' or 'AdamW'.")

    def _create_scheduler(self, optimizer, epochs: int) -> LambdaLR:
        """Linear warm-up over ``warmup_epochs`` (from 1e-3 of each group's LR), then cosine
        annealing until ``epochs`` down to min_lr / learning_rate of it; stepped once per epoch.
        Every group gets the same factor, so head and layer-wise LR ratios hold throughout."""
        warmup = self.warmup_epochs if 0 < self.warmup_epochs < epochs else 0
        span = max(1, epochs - warmup)
        floor = self.min_lr / self.lr

        def factor(epoch: int) -> float:
            if epoch < warmup:
                return 1e-3 + (1 - 1e-3) * epoch / warmup
            return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * (epoch - warmup) / span))

        return LambdaLR(optimizer, factor)

    @staticmethod
    def _group_lr(optimizer, name: str) -> float | None:
        for group in optimizer.param_groups:
            if group.get("name") == name:
                return float(group["lr"])
        return None

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _make_checkpoint_dir(
        self, checkpoint_root: Path | None, run_name: str | None = None
    ) -> tuple[Path, Path]:
        """
        Create and return ``(checkpoint_dir, checkpoint_path)`` for this run.

        The directory layout is:
            <root>/<model_name>/<freeze_layers>/<head_type>/<run_name>_<YYYYmmdd_HHMMSS>_<pid>/

        ``root`` is ``checkpoint_root`` or ``self.models_dir``; ``run_name`` defaults to
        ``"run"``. The leaf is created with ``exist_ok=False`` so two runs can never
        share a directory; if the same process creates the same run within one
        second, ``_2``, ``_3``, ... is appended.
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = Path(checkpoint_root) if checkpoint_root is not None else self.models_dir
        parent = root / self.name / str(self.freeze_layers) / self.head_type
        parent.mkdir(parents=True, exist_ok=True)
        stem = f"{_safe_run_name(run_name) if run_name else 'run'}_{ts}_{os.getpid()}"
        for attempt in range(1, 1000):
            checkpoint_dir = parent / (stem if attempt == 1 else f"{stem}_{attempt}")
            try:
                checkpoint_dir.mkdir(exist_ok=False)
            except FileExistsError:
                continue
            return checkpoint_dir, checkpoint_dir / "best.pt"
        raise FileExistsError(f"Could not create a unique checkpoint directory under {parent}")

    def _base_run_config(
        self,
        train_loader,
        val_loader,
        *,
        run_name: str | None,
        epochs: int,
        patience: int,
        min_delta: float,
        amp_enabled: bool,
        class_weights: list[float],
    ) -> dict:
        """Everything needed to rebuild / re-evaluate this run from its checkpoint."""
        entry = get_model_entry(self.name)
        val_tf = _describe_loader(val_loader)
        train_tf = _describe_loader(train_loader)
        loader_norm = val_tf.get("mean"), val_tf.get("std")
        if loader_norm[0] is not None and not (
            np.allclose(loader_norm[0], self.norm_mean, atol=1e-4)
            and np.allclose(loader_norm[1], self.norm_std, atol=1e-4)
        ):
            logger.warning(
                f"'{self.name}' was pretrained with mean={self.norm_mean} std={self.norm_std} "
                f"but the loaders normalise with mean={loader_norm[0]} std={loader_norm[1]}. "
                "Pass get_norm_stats(model) to the loader builders."
            )
        return {
            "run_name": run_name,
            "model": self.name,
            "architecture": entry.architecture,
            "pretrain_data": entry.pretrain_data,
            "pretrain_method": entry.pretrain_method,
            "external_weights": self.gastronet_file,
            "num_classes": self.number_classes,
            "label_col": self.label_col,
            "img_size": val_tf.get("img_size", self.img_size),
            "equalize": val_tf.get("equalize"),
            "mean": list(self.norm_mean),
            "std": list(self.norm_std),
            "loader_transform": {"train": train_tf, "val": val_tf},
            "head_type": self.head_type,
            "freeze_layers": self.freeze_layers,
            "dropout_rate": self.dropout_rate,
            "drop_path_rate": self.drop_path_rate,
            "threshold": self.threshold,
            "optimizer": self.optimizer,
            "learning_rate": self.lr,
            "head_lr_scale": self.head_lr_scale,
            "layer_decay": self.layer_decay,
            "weight_decay": self.weight_decay,
            "label_smoothing": self.label_smoothing,
            "active_loss_weight": self.active_loss_weight,
            "warmup_epochs": self.warmup_epochs,
            "min_lr": self.min_lr,
            "epochs": epochs,
            "es_patience": patience,
            "es_start_epoch": self.es_start_epoch,
            "grad_clip_norm": self.grad_clip_norm,
            "min_delta": min_delta,
            "use_amp": amp_enabled,
            "random_seed": self.random_seed,
            "class_weights": class_weights,
            "pos_weight": self.pos_weight,
            "n_train": len(train_loader.dataset) if hasattr(train_loader, "dataset") else None,
            "n_val": len(val_loader.dataset) if hasattr(val_loader, "dataset") else None,
            "checkpoint_file": "best.pt",
            "created": datetime.now().isoformat(timespec="seconds"),
            "torch_version": torch.__version__,
        }

    def update_run_config(self, updates: Mapping, checkpoint_dir: Path | None = None) -> Path:
        """Merge ``updates`` into ``run_config.json`` of a run and return its path.

        Runners use it to record values known only after fit(), e.g. the tuned
        validation threshold: ``model.update_run_config({"tuned_threshold": thr})``.

        Args:
            updates:        Keys to add / overwrite (converted to strict JSON).
            checkpoint_dir: Run directory; defaults to the last directory created by fit().
        """
        target = Path(checkpoint_dir) if checkpoint_dir is not None else self.checkpoint_dir
        if target is None:
            raise RuntimeError("No run directory yet: call fit() first or pass checkpoint_dir.")
        path = target / "run_config.json"
        current: dict = {}
        if path.exists():
            with open(path, encoding="utf-8") as fh:
                current = json.load(fh)
        current.update(_jsonable(dict(updates)))
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(current, fh, indent=2, allow_nan=False)
        os.replace(tmp, path)
        return path

    def _save_history(self, checkpoint_dir: Path) -> None:
        if not self.history:
            return
        with open(checkpoint_dir / "history.json", "w", encoding="utf-8") as fh:
            json.dump(_jsonable(self.history), fh, indent=2, allow_nan=False)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader,
        val_loader,
        epochs: int,
        device: torch.device,
        use_amp: bool | None = None,
        checkpoint_root: Path | None = None,
        es_patience: int | None = None,
        min_delta: float = 1e-4,
        run_name: str | None = None,
        run_metadata: Mapping | None = None,
    ) -> FitResult:
        """
        Full training loop with early stopping, LR scheduling and AMP.

        Criteria — three separate roles, no mixing:
            val_loss          → monitored only (logged in history.json)
            AUROC (or F1@0.5) → checkpoint selection + early stopping (threshold-free)
            val sweep (once)  → threshold locked after training by the caller, NOT here

        Scheduling: linear warmup over ``warmup_epochs`` then cosine annealing to
        ``min_lr``, stepped once per epoch (``_create_scheduler``). Backbone and head are
        separate parameter groups (head LR = LR * head_lr_scale; ViT blocks scaled by
        ``layer_decay``), and the schedule keeps their ratios.

        Files written to the run directory
        (``<root>/<model>/<freeze>/<head>/<run_name>_<YYYYmmdd_HHMMSS>_<pid>/``):
            best.pt          state dict of ``base_model`` at the best epoch
            history.json     one record per evaluated epoch, including the epoch on
                             which early stopping or the NaN guard fired
            run_config.json  model key, img_size, equalize, mean/std, head_type,
                             freeze_layers, threshold, hyper-parameters, run_metadata,
                             and (after training) best_epoch, best_val_auroc,
                             stopped_epoch, stop_reason

        Args:
            use_amp:      Mixed precision on CUDA; None → ``self.use_amp``.
            checkpoint_root: Root directory; None → ``self.models_dir``.
            es_patience:  Override self.es_patience (used by run_data_efficiency
                        to scale patience with subset_ratio).
            min_delta:    Minimum improvement in the checkpoint criterion to count
                        as progress for early stopping. Prevents stopping on noise.
            run_name:     Prefix of the run directory (e.g. the runner's run name).
            run_metadata: Extra JSON-serialisable fields for run_config.json (seed,
                        subset ratio, equalize, img_size, ...); they override the
                        values inferred from the model / loaders.

        Returns:
            FitResult, which unpacks as ``(best_probs, best_labels, checkpoint_dir)``:
            val probabilities and labels at the best epoch and the run directory,
            plus attributes best_val_auroc, best_val_f1, best_epoch, stopped_epoch,
            stop_reason and history. The same values are stored on ``self``.
        """
        model = self.base_model.to(device)

        weights = self._setup_class_weights(train_loader.dataset.df, label_col=self.label_col)  # type: ignore[union-attr]
        criterion = self._create_criterion(weights, device)
        optimizer = self._create_optimizer()
        scheduler = self._create_scheduler(optimizer, epochs)

        amp_requested = self.use_amp if use_amp is None else bool(use_amp)
        amp_enabled = amp_requested and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        if amp_enabled:
            logger.info("Mixed precision (AMP) enabled.")

        patience = es_patience if es_patience is not None else self.es_patience
        best_checkpoint_score = -1.0  # AUROC when valid, F1@0.5 as fallback — for comparison only
        epochs_no_improve = 0
        best_probs: np.ndarray | None = None
        best_labels: np.ndarray | None = None
        _interrupted = False
        epoch = 0
        stop_reason = "max_epochs"
        # Fixed monitoring threshold — used only for per-epoch F1 display/logging.
        # The real threshold is locked ONCE after training (see below).
        monitor_threshold = self.threshold if self.number_classes == 1 else 0.5

        # Reset the outcome of any previous fit() call.
        self.best_val_auroc = float("nan")
        self.best_val_f1 = float("nan")
        self.best_epoch = None
        self.stopped_epoch = None
        self.stop_reason = None
        self.history = []

        checkpoint_dir, checkpoint_path = self._make_checkpoint_dir(checkpoint_root, run_name)
        self.checkpoint_dir = checkpoint_dir
        run_config = self._base_run_config(
            train_loader,
            val_loader,
            run_name=run_name,
            epochs=epochs,
            patience=patience,
            min_delta=min_delta,
            amp_enabled=amp_enabled,
            class_weights=weights,
        )
        run_config.update(dict(run_metadata or {}))
        self.update_run_config(run_config, checkpoint_dir)

        try:
            for epoch in range(1, epochs + 1):
                # LRs used during this epoch (read before scheduler.step()).
                head_lr = self._group_lr(optimizer, "head")
                backbone_lr = self._group_lr(optimizer, "backbone")
                current_lr = backbone_lr if backbone_lr is not None else head_lr

                train_loss, train_acc = self._train_one_epoch(
                    train_loader,
                    optimizer,
                    criterion,
                    device,
                    epoch,
                    scaler,
                    amp_enabled,
                )
                val_loss, val_acc, precision, recall, f1, auroc, probs, labels = self.validate(
                    val_loader, criterion, device, monitor_threshold
                )

                # One history record per evaluated epoch, including the epoch on
                # which early stopping or a NaN guard fires (is_best set below).
                record = {
                    "epoch": epoch,
                    "train_loss": _finite_or_none(round(float(train_loss), 6)),
                    "train_acc": _finite_or_none(round(float(train_acc), 6)),
                    "val_loss": _finite_or_none(round(float(val_loss), 6)),
                    "val_acc": _finite_or_none(round(float(val_acc), 6)),
                    "val_f1": _finite_or_none(round(float(f1), 6)),
                    "val_auroc": _finite_or_none(round(float(auroc), 6)),
                    "lr": current_lr,
                    "head_lr": head_lr,
                    "is_best": False,
                }

                # Check for NaN in metrics (the epoch is still recorded in history.json)
                if np.isnan(train_loss):
                    self.history.append(record)
                    stop_reason = "nan_train_loss"
                    raise ValueError(f"Training diverged: NaN loss at epoch {epoch}")
                if np.isnan(val_loss) or np.isnan(f1):
                    logger.warning(
                        f"NaN in validation metrics at epoch {epoch}: "
                        f"val_loss={val_loss}, f1={f1}, auroc={auroc} — stopping."
                    )
                    self.history.append(record)
                    stop_reason = "nan_val"
                    break

                # ── Checkpoint criterion: AUROC when valid, F1@0.5 as fallback ─
                # AUROC is threshold-independent — model selection stays orthogonal
                # to threshold selection (which happens once after training ends).
                checkpoint_score = auroc if not np.isnan(auroc) else f1
                criterion_label = "AUROC" if not np.isnan(auroc) else "F1@0.5"

                head_str = (
                    f" | head LR {head_lr:.2e}" if self.head_lr_scale != 1.0 and head_lr else ""
                )
                logger.info(
                    f"Epoch {epoch:>3}/{epochs} | "
                    f"Train {train_loss:.4f} | Val {val_loss:.4f} | "
                    f"Acc {val_acc:.4f} | P {precision:.4f} | R {recall:.4f} | "
                    f"F1@0.5 {f1:.4f} | AUROC {auroc:.4f} | LR {current_lr:.2e}{head_str}"
                )

                # ── Scheduler step (cosine annealing, once per epoch) ─────────────
                scheduler.step()

                # ── Checkpoint + Early stopping on the checkpoint criterion ───────
                is_best = checkpoint_score > best_checkpoint_score + min_delta
                record["is_best"] = bool(is_best)
                self.history.append(record)
                if is_best:
                    best_checkpoint_score = checkpoint_score
                    self.best_val_f1 = float(f1)
                    self.best_val_auroc = float(auroc)
                    self.best_epoch = epoch
                    best_probs = probs
                    best_labels = labels
                    torch.save(model.state_dict(), checkpoint_path)
                    extra = (
                        f"F1@0.5={f1:.4f}" if criterion_label == "AUROC" else f"AUROC={auroc:.4f}"
                    )
                    logger.info(
                        f"  -> Best {criterion_label}={checkpoint_score:.4f} "
                        f"({extra}) — checkpoint saved."
                    )
                    epochs_no_improve = 0
                else:
                    if epoch > self.es_start_epoch:
                        epochs_no_improve += 1
                    logger.info(
                        f"  No {criterion_label} improvement for "
                        f"{epochs_no_improve}/{patience} epoch(s). "
                        f"(best {criterion_label}={best_checkpoint_score:.4f})"
                    )
                    if epochs_no_improve >= patience:
                        logger.info(f"  Early stopping at epoch {epoch}.")
                        stop_reason = "early_stopping"
                        break

        except KeyboardInterrupt:
            logger.warning(f"Training interrupted at epoch {epoch} — restoring best checkpoint.")
            _interrupted = True
            stop_reason = "interrupted"

        finally:
            # ── Save training history + outcome (also when an exception escapes) ─
            self.stopped_epoch = epoch
            self.stop_reason = stop_reason
            self._save_history(checkpoint_dir)
            self.update_run_config(
                {
                    "best_epoch": self.best_epoch,
                    "best_val_auroc": self.best_val_auroc,
                    "best_val_f1": self.best_val_f1,
                    "stopped_epoch": epoch,
                    "stop_reason": stop_reason,
                    "epochs_completed": len(self.history),
                    "has_checkpoint": checkpoint_path.exists(),
                },
                checkpoint_dir,
            )

        # ── Restore best checkpoint ───────────────────────────────────────
        self._restore_best_checkpoint(
            checkpoint_path, device, float(self.best_val_f1), float(self.best_val_auroc)
        )

        if _interrupted:
            raise _TrainingInterrupted(checkpoint_dir) from None

        return FitResult(
            best_probs,
            best_labels,
            checkpoint_dir,
            best_val_auroc=self.best_val_auroc,
            best_val_f1=self.best_val_f1,
            best_epoch=self.best_epoch,
            stopped_epoch=self.stopped_epoch,
            stop_reason=self.stop_reason,
            history=self.history,
        )

    def _train_one_epoch(
        self,
        dataloader,
        optimizer,
        criterion,
        device: torch.device,
        epoch: int,
        scaler: torch.amp.GradScaler,
        amp_enabled: bool,
    ) -> tuple[float, float]:
        """Single training epoch (used by ``fit``).

        Frozen modules (linear probe / partial freezing) stay in eval mode.
        """
        self._set_train_mode()
        total_loss, correct, total = 0.0, 0, 0

        for images, labels, *_ in tqdm(
            dataloader, desc=f"Epoch [{epoch}/{self.number_epochs}]", leave=False
        ):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = self.base_model(images)
                outputs = logits.reshape(-1) if self.number_classes == 1 else logits
                targets = labels.float().reshape(-1) if self.number_classes == 1 else labels
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.base_model.parameters(), max_norm=self.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            with torch.no_grad():
                probs = (
                    torch.sigmoid(logits.squeeze(1))
                    if self.number_classes == 1
                    else torch.softmax(logits, dim=1)
                )
                threshold = self.threshold if self.threshold is not None else 0.5
                preds = (
                    (probs >= threshold).long() if self.number_classes == 1 else probs.argmax(dim=1)
                )
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        return total_loss / len(dataloader), correct / total if total > 0 else 0.0


    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def validate(self, dataloader, criterion, device, threshold):
        self.base_model.eval()
        running_loss = 0.0
        all_probs, all_labels = [], []

        with torch.no_grad():
            for images, labels, *_ in dataloader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                logits = self.base_model(images)

                probs = (
                    torch.sigmoid(logits.squeeze(1))
                    if self.number_classes == 1
                    else torch.softmax(logits, dim=1)
                )
                outputs = logits.reshape(-1) if self.number_classes == 1 else logits
                targets = labels.float().reshape(-1) if self.number_classes == 1 else labels

                running_loss += criterion(outputs, targets).item()
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(targets.cpu().numpy())

        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        preds = (
            (all_probs >= threshold).astype(int)
            if self.number_classes == 1
            else all_probs.argmax(axis=1)
        )

        avg = "binary" if self.number_classes == 1 else "macro"
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, preds, average=avg, zero_division=0
        )
        # AUROC is undefined when the val set lacks a class → NaN (F1 fallback in fit()).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            try:
                if self.number_classes > 2:
                    auroc = roc_auc_score(
                        all_labels,
                        all_probs,
                        multi_class="ovr",
                        labels=list(range(self.number_classes)),
                    )
                elif self.number_classes == 2:
                    auroc = roc_auc_score(all_labels, all_probs[:, 1])
                else:
                    auroc = roc_auc_score(all_labels, all_probs)
            except ValueError:
                auroc = float("nan")
        return (
            running_loss / len(dataloader),
            accuracy_score(all_labels, preds),
            precision,
            recall,
            f1,
            auroc,
            all_probs,
            all_labels,
        )

    def _restore_best_checkpoint(
        self,
        checkpoint_path: Path,
        device: torch.device,
        best_val_f1: float,
        best_val_auroc: float,
    ) -> None:
        """Load best weights from *checkpoint_path* into self.base_model and log the result."""
        if checkpoint_path.exists():
            self.base_model.load_state_dict(
                torch.load(checkpoint_path, map_location=device, weights_only=True)
            )
            logger.info(
                f"Best weights restored (epoch {self.best_epoch}) — "
                f"val F1: {best_val_f1:.4f} val auroc: {best_val_auroc:.4f}"
            )
        else:
            logger.warning("No checkpoint found — returning last-epoch weights.")

    def _collect_predictions(self, loader, device: torch.device) -> dict:
        """Run inference on ``loader`` (eval mode, no grad).

        Returns a dict with ``labels`` (N,), ``probs`` ((N,) sigmoid for 1-logit
        binary, (N, C) softmax otherwise), ``clip_ids``, ``frame_ids`` (lists, empty
        when the loader yields none) and ``inference_times`` (s / image per batch).
        """
        self.base_model.to(device)
        self.base_model.eval()
        y_true, y_prob, clip_ids, frame_ids, inference_times = [], [], [], [], []

        def _iter_loader(loader_):
            try:
                yield from loader_
            except RuntimeError as exc:
                if "worker" in str(exc).lower() or "exited unexpectedly" in str(exc):
                    logger.warning(
                        "DataLoader workers crashed — retrying with num_workers=0. "
                        f"Original error: {exc}"
                    )
                    from torch.utils.data import DataLoader

                    # Restart from the beginning; drop the partial results.
                    for buf in (y_true, y_prob, clip_ids, frame_ids, inference_times):
                        buf.clear()
                    safe_loader = DataLoader(
                        loader_.dataset,
                        batch_size=loader_.batch_size,
                        shuffle=False,
                        num_workers=0,
                        collate_fn=loader_.collate_fn,
                        pin_memory=False,
                    )
                    yield from safe_loader
                else:
                    raise

        with torch.no_grad():
            for batch in tqdm(
                _iter_loader(loader), desc="Inference", leave=False, total=len(loader)
            ):
                images, labels = batch[0], batch[1]
                clip_id = batch[2] if len(batch) > 2 else None
                frame_id = batch[3] if len(batch) > 3 else None

                images = images.to(device, non_blocking=True)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                logits = self.base_model(images)
                probs = (
                    torch.sigmoid(logits.squeeze(1))
                    if self.number_classes == 1
                    else torch.softmax(logits, dim=1)
                )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                inference_times.append((time.perf_counter() - t0) / images.size(0))

                y_true.extend(labels.cpu().numpy())
                y_prob.extend(probs.float().cpu().numpy())
                if clip_id is not None:
                    clip_ids.extend(clip_id)
                if frame_id is not None:
                    frame_ids.extend(frame_id)

        return {
            "labels": np.asarray(y_true),
            "probs": np.asarray(y_prob),
            "clip_ids": clip_ids,
            "frame_ids": frame_ids,
            "inference_times": inference_times,
        }

    def get_labels_and_probs(self, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(labels, probs)`` for any loader, in loader order.

        ``probs`` is (N,) sigmoid probabilities for 1-logit binary models and (N, C)
        softmax probabilities otherwise — the same arrays test_evaluation() stores
        under "labels" / "probabilities". Use it e.g. to save val probabilities.
        """
        out = self._collect_predictions(loader, device)
        return out["labels"], out["probs"]

    def test_evaluation(
        self, test_loader, device, threshold, aggregate_by_clip=True, cluster_by_patient=True
    ):
        """Evaluate on ``test_loader``.

        Metrics are full-sample point estimates with percentile bootstrap CIs
        (``self.n_bootstrap`` resamples at level ``self.ci``). With
        ``cluster_by_patient`` the bootstrap resamples whole patients (the
        dataset's ``patient_id``), else whole clips when the loader yields clip ids.

        ``threshold`` applies to binary (``num_classes == 1``) models only; None falls
        back to ``self.threshold``. Multiclass models ignore it and use argmax.
        """
        if threshold is None and self.number_classes == 1:
            threshold = self.threshold if self.threshold is not None else 0.5
        if device.type == "cuda":
            torch.cuda.empty_cache()
        pred = self._collect_predictions(test_loader, device)
        y_true, y_prob = pred["labels"], pred["probs"]
        clip_ids, frame_ids = pred["clip_ids"], pred["frame_ids"]
        inference_times = pred["inference_times"]

        y_pred = (
            (y_prob >= threshold).astype(int) if self.number_classes == 1 else y_prob.argmax(axis=1)
        )

        is_binary = self.number_classes <= 2
        n_label_classes = max(2, self.number_classes)
        cm_labels = list(range(n_label_classes))

        # Binary → 1D prob vector; multiclass → full (N, C) matrix for OVR AUROC
        if self.number_classes == 2:
            y_prob_for_metrics = y_prob[:, 1]  # positive-class prob for binary AUROC
        elif self.number_classes == 1:
            y_prob_for_metrics = y_prob  # already 1D
        else:
            y_prob_for_metrics = y_prob  # (N, C) matrix — OVR/macro AUROC in metrics.py

        groups = None
        df = getattr(test_loader.dataset, "df", None)
        if cluster_by_patient and frame_ids and df is not None and "patient_id" in df:
            patient_of = dict(zip(df["relative_path"].astype(str), df["patient_id"].astype(str)))
            groups = np.asarray([patient_of[str(f)] for f in frame_ids])
        elif clip_ids:
            groups = np.asarray([str(c) for c in clip_ids])
        if groups is not None and len(groups) != len(y_true):
            logger.warning("cluster ids length mismatch — frame-level (i.i.d.) bootstrap used.")
            groups = None

        metrics = compute_metrics_with_ci(
            y_true,
            y_pred,
            y_prob_for_metrics,
            n_bootstrap=self.n_bootstrap,
            seed=self.random_seed,
            ci=self.ci,
            groups=groups,
            n_classes=n_label_classes,
        )

        threshold_str = f"{threshold:.3f}" if threshold is not None else "argmax"
        logger.info(
            f"Test set evaluation at threshold {threshold_str} : "
            f"Accuracy={metrics.get('_Accuracy_mean', float('nan')):.4f} | "
            f"Precision={metrics.get('_Precision_mean', float('nan')):.4f} | "
            f"Recall={metrics.get('_Sensitivity_mean', float('nan')):.4f} | "
            f"F1={metrics.get('_F1_mean', float('nan')):.4f} | "
            f"AUROC={metrics.get('_AUROC_mean', float('nan')):.4f}"
        )

        results = {
            "model_name": self.name,
            "threshold": threshold,
            "frame_level": {
                "accuracy": metrics["_Accuracy_mean"],
                "precision": metrics.get("_Precision_mean", float("nan")),
                "recall": metrics["_Sensitivity_mean"],
                "f1": metrics["_F1_mean"],
                "roc_auc": metrics["_AUROC_mean"],
                "confusion_matrix": confusion_matrix(y_true, y_pred, labels=cm_labels),
            },
            "inference_time_ms": (
                float(np.mean(inference_times)) * 1000 if inference_times else float("nan")
            ),
            "predictions": y_pred,
            "labels": y_true,
            "probabilities": y_prob,
            "probabilities_1d": y_prob_for_metrics,
            "clip_ids": clip_ids if clip_ids else None,
            "frame_ids": frame_ids if frame_ids else None,
            "metrics_with_ci": metrics,
            "num_classes": self.number_classes,
        }

        if aggregate_by_clip and clip_ids and is_binary:
            if len(clip_ids) != len(y_true):
                logger.warning("clip_ids length mismatch — skipping clip aggregation.")
            else:
                results["clip_threshold"] = self.clip_threshold
                results = self._add_clip_level_results(
                    results,
                    y_pred,
                    y_true,
                    y_prob_for_metrics,
                    clip_ids,
                    self.clip_threshold,
                )
        elif aggregate_by_clip and clip_ids and not is_binary:
            if len(clip_ids) != len(y_true):
                logger.warning("clip_ids length mismatch — skipping clip aggregation.")
            else:
                # Multiclass clip aggregation: average per-class probs across frames,
                # then take argmax as clip prediction.
                from scipy import stats as _stats

                clip_data: dict = defaultdict(lambda: {"probs": [], "labels": []})
                for i, cid in enumerate(clip_ids):
                    clip_data[cid]["probs"].append(y_prob[i])  # shape (n_classes,)
                    clip_data[cid]["labels"].append(int(y_true[i]))

                clip_ids_agg, clip_y_true, clip_y_pred, clip_y_prob = [], [], [], []
                for cid, data in clip_data.items():
                    mean_prob = np.mean(data["probs"], axis=0)  # (n_classes,)
                    clip_y_pred.append(int(np.argmax(mean_prob)))
                    clip_y_prob.append(mean_prob)
                    # Modal label for the clip
                    clip_y_true.append(int(_stats.mode(data["labels"], keepdims=True).mode[0]))
                    clip_ids_agg.append(cid)

                clip_y_true = np.array(clip_y_true)
                clip_y_pred = np.array(clip_y_pred)
                clip_y_prob = np.array(clip_y_prob)  # (n_clips, n_classes)

                clip_metrics = compute_metrics_with_ci(
                    clip_y_true,
                    clip_y_pred,
                    clip_y_prob,
                    n_bootstrap=self.n_bootstrap,
                    seed=self.random_seed,
                    ci=self.ci,
                    n_classes=n_label_classes,
                )
                results["clip_level"] = {
                    "method": "mean_prob_argmax",
                    "accuracy": clip_metrics["_Accuracy_mean"],
                    "precision": clip_metrics.get("_Precision_mean", float("nan")),
                    "recall": clip_metrics["_Sensitivity_mean"],
                    "f1": clip_metrics["_F1_mean"],
                    "roc_auc": clip_metrics["_AUROC_mean"],
                    "confusion_matrix": confusion_matrix(
                        clip_y_true, clip_y_pred, labels=cm_labels
                    ),
                    "n_clips": len(clip_ids_agg),
                    "clip_ids": clip_ids_agg,
                    "clip_predictions": clip_y_pred,
                    "clip_labels": clip_y_true,
                    "clip_probs": clip_y_prob,
                    "metrics_with_ci": clip_metrics,
                }

        return results

    def _add_clip_level_results(self, results, y_pred, y_true, y_prob, clip_ids, threshold):
        best_method = f"mean_prob-{threshold}"

        clip_res = aggregate_frame_to_clip(y_prob, y_pred, y_true, clip_ids, best_method)
        results["clip_level"] = {
            "method": best_method,
            "accuracy": clip_res["accuracy"],
            "precision": clip_res["precision"],
            "recall": clip_res["recall"],
            "f1": clip_res["f1"],
            "roc_auc": clip_res["roc_auc"],
            "confusion_matrix": clip_res["confusion_matrix"],
            "n_clips": clip_res["n_clips"],
            "clip_ids": clip_res["clip_ids"],
            "clip_predictions": clip_res["y_pred"],
            "clip_labels": clip_res["y_true"],
            "clip_probs": clip_res["y_prob_clip"],
        }

        return results

