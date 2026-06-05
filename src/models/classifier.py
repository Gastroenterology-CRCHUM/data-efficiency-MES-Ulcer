"""
src/models/classifier.py
========================
ClassifierModel wraps any backbone declared in MODEL_REGISTRY and provides:
  - registry-driven construction (no hard-coded model names)
  - GastroNet / custom weight loading with wrapper-key handling
  - layer freezing (full / backbone-only / N blocks)
  - per-epoch train / validate loops
  - test evaluation with bootstrap CIs and optional clip-level aggregation
"""

from __future__ import annotations

import time
import warnings
from datetime import datetime
from pathlib import Path


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
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from src.config import PathConfig, get_model_entry
from src.evaluation.aggregation import aggregate_frame_to_clip, compare_aggregation_methods
from src.evaluation.metrics import compute_metrics_with_ci
from src.utils.logging import setup_logging

logger = setup_logging(__name__)

HEAD_TYPES = ("linear", "mlp1", "mlp2")


class _HFViTBackbone(nn.Module):
    """Thin wrapper around a HuggingFace or timm ViT that returns the CLS pooled output.

    Exposes a `.head = nn.Identity()` sentinel so that _replace_last_layer detects
    it as an Identity-headed model and wraps it with _HeadedBackbone — identical
    code path as DINO ViT models.

    model_id starting with "timm/" → loaded via timm (hf_hub prefix added automatically).
    Otherwise → loaded via transformers.ViTModel.
    """

    def __init__(self, model_id: str) -> None:
        super().__init__()
        self._is_timm = model_id.startswith("timm/")
        if self._is_timm:
            import timm as _timm  # lazy import

            self.vit = _timm.create_model(f"hf_hub:{model_id}", pretrained=True, num_classes=0)
        else:
            from transformers import ViTModel  # lazy import

            self.vit = ViTModel.from_pretrained(model_id)
        self.head = nn.Identity()  # sentinel for _replace_last_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._is_timm:
            return self.vit(x)
        return self.vit(pixel_values=x).pooler_output


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


class ClassifierModel(nn.Module):
    """
    Unified classifier wrapping any backbone declared in MODEL_REGISTRY.

    Args:
        base_model:      Key in MODEL_REGISTRY (e.g. 'vits16_gastronet').
        num_classes:     1 = binary (BCEWithLogitsLoss), 2 = CrossEntropyLoss.
        class_weights:   None/False → auto-compute from manifest;
                         float → manual positive-class weight (binary only).
        optimizer:       'Adam' | 'AdamW'.
        learning_rate:   Initial LR.
        threshold:       Decision threshold (binary only).
        dropout_rate:    Dropout probability before the final linear layer.
        num_epochs:      Maximum training epochs.
        freeze_layers:   0  = full fine-tuning
                         -1 = freeze backbone, train head only
                         N  = freeze first N blocks / layer-groups
        gastronet_path:  Full Path to a GastroNet .pth file, or None.
        es_patience:     Early-stopping patience (epochs without val-F1 gain).
        lr_patience:     ReduceLROnPlateau patience.
        lr_factor:       ReduceLROnPlateau reduction factor.
        weight_decay:    Optimizer L2 regularisation.
        label_smoothing: Label smoothing for CrossEntropyLoss (multi-class only).
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
        num_epochs: int = 100,
        freeze_layers: int = 0,
        gastronet_path: Path | None = None,
        es_patience: int = 10,
        lr_patience: int = 5,
        lr_factor: float = 0.2,
        weight_decay: float = 0.01,
        label_smoothing: float = 0.0,
        label_col: str = "label",  # ← "ulcer_size" for Pipeline C
        head_type: str = "linear",  # "linear" | "mlp1" | "mlp2"
        random_seed: int = 42,
        warmup_epochs: int = 5,
        min_lr: float = 1e-7,
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
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.label_col = label_col
        self.head_type = head_type
        self.freeze_layers = freeze_layers
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr

        # ── 1. Build backbone ─────────────────────────────────────────
        self.base_model = self._build_backbone(entry)

        # ── 2. Load GastroNet (or any external) weights ───────────────
        if gastronet_path is not None:
            self._load_external_weights(gastronet_path, paths)

        # ── 3. Replace classifier head ────────────────────────────────
        self._replace_last_layer(num_classes, dropout_rate)

        # ── 4. Freeze layers ──────────────────────────────────────────
        self._freeze_layers(freeze_layers)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _build_backbone(self, entry) -> nn.Module:
        """Instantiate the backbone from a registry entry.

        Priority: hf_model_id > hub_model > torchvision builder.
        """
        if entry.hf_model_id is not None:
            return _HFViTBackbone(entry.hf_model_id)
        if entry.hub_model is not None:
            return entry.builder(entry.weights, entry.hub_model)
        return entry.builder(weights=entry.weights)

    def _load_external_weights(self, gastronet_path: Path, paths: PathConfig) -> None:
        """Load GastroNet (or any external) weights into self.base_model."""
        full_path = paths.pretrained_dir / Path(gastronet_path)
        if not full_path.exists():
            raise FileNotFoundError(f"GastroNet weight file not found: {full_path}")
        state_dict = torch.load(full_path, map_location="cpu", weights_only=True)
        if isinstance(state_dict, dict):
            state_dict = state_dict.get("model") or state_dict.get("state_dict") or state_dict
        missing, unexpected = self.base_model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(
                f"[GastroNet] {len(missing)} missing key(s) "
                "(classifier head will be randomly initialised)."
            )
        if unexpected:
            logger.info(f"[GastroNet] {len(unexpected)} unexpected key(s) ignored.")
        logger.info(f"[GastroNet] Loaded: {full_path.name}")

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
            # DINO ViT: forward() returns the raw CLS token without calling self.head.
            # Infer embed_dim via a dummy forward pass, then wrap with _HeadedBackbone.
            img_size = get_model_entry(self.name).img_size
            device = next(self.base_model.parameters()).device
            with torch.no_grad():
                dummy = torch.zeros(1, 3, img_size, img_size, device=device)
                in_features = int(self.base_model(dummy).shape[-1])
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
            raise ValueError(f"Unknown head_type '{self.head_type}'. Use one of {HEAD_TYPES}.")

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

    def _freeze_layers(self, num_layers: int):
        """
        Freeze backbone layers.

            0  → nothing frozen (full fine-tuning)
            -1 → entire backbone frozen, only head trainable
            N  → first N blocks / layer-groups frozen
        """
        if num_layers == 0:
            logger.info("No layers frozen — full fine-tuning.")
            return

        is_headed = isinstance(self.base_model, _HeadedBackbone)
        # For _HeadedBackbone, parameters are named "backbone.*" and "head.*".
        # For regular models, the head lives at classifier_name (e.g. "fc", "head").
        head_prefix = "head" if is_headed else get_model_entry(self.name).classifier
        arch = self.base_model.backbone if is_headed else self.base_model

        if num_layers == -1:
            for name, param in self.base_model.named_parameters():
                if not name.startswith(head_prefix):
                    param.requires_grad = False
            logger.info("Backbone fully frozen — only classification head is trainable.")
            return

        child_map = dict(arch.named_children())
        classifier_name = get_model_entry(self.name).classifier

        if classifier_name == "fc" and "fc" in child_map:
            groups = [m for name, m in arch.named_children() if name != "fc"]
            n = min(num_layers, len(groups))
            for m in groups[:n]:
                for p in m.parameters():
                    p.requires_grad = False
            logger.info(f"ResNet: frozen {n}/{len(groups)} layer groups.")

        elif "features" in child_map:
            feat_blocks = list(child_map["features"].children())
            n = min(num_layers, len(feat_blocks))
            for m in feat_blocks[:n]:
                for p in m.parameters():
                    p.requires_grad = False
            logger.info(f"EfficientNet: frozen {n}/{len(feat_blocks)} feature blocks.")

        elif "blocks" in child_map:
            vit_blocks = list(child_map["blocks"].children())
            n = min(num_layers, len(vit_blocks))
            for m in vit_blocks[:n]:
                for p in m.parameters():
                    p.requires_grad = False
            logger.info(f"ViT: frozen {n}/{len(vit_blocks)} transformer blocks.")

        else:
            logger.warning(f"Partial freezing not implemented for '{self.name}' — skipped.")

    @property
    def unwrapped_backbone(self) -> nn.Module:
        """Raw backbone without the _HeadedBackbone wrapper (if any)."""
        if isinstance(self.base_model, _HeadedBackbone):
            return self.base_model.backbone
        return self.base_model

    # ------------------------------------------------------------------
    # Class-weight / criterion / optimizer setup
    # ------------------------------------------------------------------

    def _setup_class_weights(
        self,
        manifest_path_or_df: Path | pd.DataFrame,
        label_col: str = "label",  # ← "ulcer_size" for Pipeline C
    ) -> list[float]:
        """Compute per-class weights from training data.

        Binary  (num_classes=1): scalar class_weights → [1.0, w]; None → auto.
        Multiclass (num_classes>1): list of correct length → used as-is;
            None or scalar → auto-compute balanced weights from train counts.

        Args:
            manifest_path_or_df: Either a Path to the manifest CSV (uses
                split=="train" rows) or a DataFrame of the actual training
                set (e.g. the fold training data in CV mode).
        """
        n_classes = max(self.number_classes, 2)

        # User-provided list (multiclass explicit weights)
        if isinstance(self.class_weights, list):
            if len(self.class_weights) == n_classes:
                return [float(w) for w in self.class_weights]
            logger.warning(
                f"class_weights list length {len(self.class_weights)} != {n_classes} "
                "— falling back to auto-computed balanced weights."
            )

        # Binary with scalar positive-class weight
        if self.number_classes == 1 and isinstance(self.class_weights, (float, int)):
            return [1.0, float(self.class_weights)]

        # Auto-compute balanced weights (multiclass default, or binary with None)
        if self.class_weights is None or self.number_classes > 1:
            if isinstance(manifest_path_or_df, pd.DataFrame):
                train_df = manifest_path_or_df
            else:
                df = pd.read_csv(manifest_path_or_df)
                train_df = df[df["split"] == "train"]
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
            return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weights[1]).float().to(device))
        return nn.CrossEntropyLoss(
            weight=torch.tensor(weights).float().to(device),
            label_smoothing=self.label_smoothing,
        )

    def _create_optimizer(self):
        params = self.base_model.parameters()
        if self.optimizer == "Adam":
            return torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        if self.optimizer == "AdamW":
            return torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        raise ValueError(f"Unknown optimizer '{self.optimizer}'. Use 'Adam' or 'AdamW'.")

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _make_checkpoint_dir(self, checkpoint_root: Path | None) -> tuple[Path, Path]:
        """
        Create and return ``(checkpoint_dir, checkpoint_path)`` for this run.

        The directory layout is:
            <checkpoint_root>/<model_name>/<freeze_layers>/<head_type>/<timestamp>/
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = checkpoint_root if checkpoint_root is not None else self.paths.ulcer_models_dir
        checkpoint_dir = root / self.name / str(self.freeze_layers) / self.head_type / ts
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return checkpoint_dir, checkpoint_dir / "best.pt"

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader,
        val_loader,
        epochs: int,
        device: torch.device,
        use_amp: bool = True,
        checkpoint_root: Path | None = None,
        es_patience: int | None = None,
        min_delta: float = 1e-4,
    ) -> tuple[np.ndarray | None, np.ndarray | None, Path]:
        """
        Full training loop with early stopping, LR scheduling and AMP.

        Criteria — three separate roles, no mixing:
            val_loss          → scheduler only (smooth, differentiable)
            AUROC (or F1@0.5) → checkpoint selection + early stopping (threshold-free)
            val sweep (once)  → threshold locked after training, NOT during

        Threshold methodology:
            The decision threshold is determined exactly once, after training ends,
            by sweeping on the val set at the best-checkpoint model state.  It is
            never updated during the training loop — AUROC-based checkpointing is
            deliberately threshold-independent so that model selection and
            operating-point selection remain orthogonal.

        Args:
            es_patience:  Override self.es_patience (used by run_data_efficiency
                        to scale patience with subset_ratio).
            min_delta:    Minimum improvement in the checkpoint criterion to count
                        as progress for early stopping. Prevents stopping on noise.

        Returns:
            best_probs     : Val probabilities at the best-checkpoint epoch.
            best_labels    : Val labels at the best-checkpoint epoch.
            checkpoint_dir : Directory containing best.pt.
        """
        model = self.base_model.to(device)

        weights = self._setup_class_weights(train_loader.dataset.df, label_col=self.label_col)  # type: ignore[union-attr]
        criterion = self._create_criterion(weights, device)
        optimizer = self._create_optimizer()
        if self.warmup_epochs > 0 and self.warmup_epochs < epochs:
            _warmup = LinearLR(
                optimizer, start_factor=1e-3, end_factor=1.0, total_iters=self.warmup_epochs
            )
            _cosine = CosineAnnealingLR(
                optimizer, T_max=max(1, epochs - self.warmup_epochs), eta_min=self.min_lr
            )
            scheduler = SequentialLR(
                optimizer, schedulers=[_warmup, _cosine], milestones=[self.warmup_epochs]
            )
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=self.min_lr)

        amp_enabled = use_amp and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        if amp_enabled:
            logger.info("Mixed precision (AMP) enabled.")

        patience = es_patience if es_patience is not None else self.es_patience
        best_val_f1 = -1.0
        best_val_auroc = float("nan")
        best_checkpoint_score = -1.0  # AUROC when valid, F1@0.5 as fallback — for comparison only
        epochs_no_improve = 0
        best_probs: np.ndarray | None = None
        best_labels: np.ndarray | None = None
        _interrupted = False
        epoch = 0
        history: list[dict] = []
        # Fixed monitoring threshold — used only for per-epoch F1 display/logging.
        # The real threshold is locked ONCE after training (see below).
        monitor_threshold = self.threshold if self.number_classes == 1 else 0.5

        checkpoint_dir, checkpoint_path = self._make_checkpoint_dir(checkpoint_root)

        try:
            for epoch in range(1, epochs + 1):
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

                # Check for NaN in metrics
                if np.isnan(train_loss):
                    raise ValueError(f"Training diverged: NaN loss at epoch {epoch}")
                if np.isnan(val_loss) or np.isnan(f1):
                    logger.warning(
                        f"NaN in validation metrics at epoch {epoch}: "
                        f"val_loss={val_loss}, f1={f1}, auroc={auroc}"
                    )
                    break

                # ── Checkpoint criterion: AUROC when valid, F1@0.5 as fallback ─
                # AUROC is threshold-independent — model selection stays orthogonal
                # to threshold selection (which happens once after training ends).
                checkpoint_score = auroc if not np.isnan(auroc) else f1
                criterion_label = "AUROC" if not np.isnan(auroc) else "F1@0.5"

                current_lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    f"Epoch {epoch:>3}/{epochs} | "
                    f"Train {train_loss:.4f} | Val {val_loss:.4f} | "
                    f"Acc {val_acc:.4f} | P {precision:.4f} | R {recall:.4f} | "
                    f"F1@0.5 {f1:.4f} | AUROC {auroc:.4f} | LR {current_lr:.2e}"
                )
                self._log_epoch_metrics(
                    epoch,
                    train_loss,
                    train_acc,
                    val_loss,
                    val_acc,
                    precision,
                    recall,
                    f1,
                    auroc,
                    current_lr,
                )

                # ── Scheduler step (cosine annealing, once per epoch) ─────────────
                scheduler.step()

                # ── Checkpoint + Early stopping on clinical criterion ──────────────
                is_best = checkpoint_score > best_checkpoint_score + min_delta
                if is_best:
                    best_checkpoint_score = checkpoint_score
                    best_val_f1 = f1
                    best_val_auroc = auroc
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
                    epochs_no_improve += 1
                    logger.info(
                        f"  No {criterion_label} improvement for "
                        f"{epochs_no_improve}/{patience} epoch(s). "
                        f"(best {criterion_label}={best_checkpoint_score:.4f})"
                    )
                    if epochs_no_improve >= patience:
                        logger.info(f"  Early stopping at epoch {epoch}.")
                        break

                history.append({
                    "epoch": epoch,
                    "train_loss": round(float(train_loss), 6),
                    "train_acc":  round(float(train_acc),  6),
                    "val_loss":   round(float(val_loss),   6),
                    "val_acc":    round(float(val_acc),    6),
                    "val_f1":     round(float(f1),         6),
                    "val_auroc":  round(float(auroc),      6),
                    "lr":         float(current_lr),
                    "is_best":    bool(is_best),
                })

        except KeyboardInterrupt:
            logger.warning(f"Training interrupted at epoch {epoch} — restoring best checkpoint.")
            _interrupted = True

        # ── Save training history ─────────────────────────────────────────
        if history:
            import json as _json
            with open(checkpoint_dir / "history.json", "w", encoding="utf-8") as _fh:
                _json.dump(history, _fh, indent=2)

        # ── Restore best checkpoint ───────────────────────────────────────
        self._restore_best_checkpoint(
            checkpoint_path, device, float(best_val_f1), float(best_val_auroc)
        )

        if _interrupted:
            raise _TrainingInterrupted(checkpoint_dir) from None

        return best_probs, best_labels, checkpoint_dir

    def _train_one_epoch(
        self,
        dataloader,
        optimizer,
        criterion,
        device: torch.device,
        epoch: int,
        scaler: torch.cuda.amp.GradScaler,
        amp_enabled: bool,
    ) -> tuple[float, float]:
        """Single training epoch — used by both ``train_one_epoch`` and ``fit``."""
        self.base_model.train()
        total_loss, correct, total = 0.0, 0, 0

        for images, labels, *_ in tqdm(
            dataloader, desc=f"Epoch [{epoch}/{self.number_epochs}]", leave=False
        ):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = self.base_model(images)
                if hasattr(logits, "logits"):
                    logits = logits.logits
                outputs = logits.reshape(-1) if self.number_classes == 1 else logits
                targets = labels.float().reshape(-1) if self.number_classes == 1 else labels
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.base_model.parameters(), max_norm=1.0)
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

    def train_one_epoch(
        self, dataloader, optimizer, criterion, device, epoch
    ) -> tuple[float, float]:
        """
        Single training epoch (public API).

        Delegates to ``_train_one_epoch`` with AMP disabled so callers that
        manage their own scaler (e.g. hyperparameter_search.py) share the
        same implementation.
        """
        dummy_scaler = torch.cuda.amp.GradScaler(enabled=False)
        return self._train_one_epoch(
            dataloader,
            optimizer,
            criterion,
            device,
            epoch,
            scaler=dummy_scaler,
            amp_enabled=False,
        )

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
                if hasattr(logits, "logits"):
                    logits = logits.logits

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
        # For multiclass (>2 classes), must specify multi_class parameter
        if self.number_classes > 2:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                try:
                    auroc = roc_auc_score(
                        all_labels,
                        all_probs,
                        multi_class="ovr",
                        labels=list(range(self.number_classes)),
                    )
                except ValueError:
                    auroc = float("nan")
        elif self.number_classes == 2:
            auroc = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            auroc = roc_auc_score(all_labels, all_probs)
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
                f"Best weights restored — val F1: {best_val_f1:.4f} val auroc:{best_val_auroc:.4f}"
            )
        else:
            logger.warning("No checkpoint found — returning last-epoch weights.")

    def test_evaluation(self, test_loader, device, threshold, aggregate_by_clip=True):
        self.base_model.to(device)
        self.base_model.eval()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        y_true, y_prob, clip_ids, frame_ids, inference_times = [], [], [], [], []

        def _iter_loader(loader):
            try:
                yield from loader
            except RuntimeError as exc:
                if "worker" in str(exc).lower() or "exited unexpectedly" in str(exc):
                    logger.warning(
                        "DataLoader workers crashed — retrying with num_workers=0. "
                        f"Original error: {exc}"
                    )
                    from torch.utils.data import DataLoader

                    safe_loader = DataLoader(
                        loader.dataset,
                        batch_size=loader.batch_size,
                        shuffle=False,
                        num_workers=0,
                        collate_fn=loader.collate_fn,
                        pin_memory=False,
                    )
                    yield from safe_loader
                else:
                    raise

        with torch.no_grad():
            for batch in tqdm(
                _iter_loader(test_loader), desc="Testing", leave=False, total=len(test_loader)
            ):
                images, labels = batch[0], batch[1]
                clip_id = batch[2] if len(batch) > 2 else None
                frame_id = batch[3] if len(batch) > 3 else None

                images = images.to(device, non_blocking=True)
                t0 = time.time()
                logits = self.base_model(images)
                if hasattr(logits, "logits"):
                    logits = logits.logits
                probs = (
                    torch.sigmoid(logits.squeeze(1))
                    if self.number_classes == 1
                    else torch.softmax(logits, dim=1)
                )
                inference_times.append((time.time() - t0) / images.size(0))

                y_true.extend(labels.numpy())
                y_prob.extend(probs.cpu().numpy())
                if clip_id is not None:
                    clip_ids.extend(clip_id)
                if frame_id is not None:
                    frame_ids.extend(frame_id)

        y_true = np.array(y_true)
        y_prob = np.array(y_prob)
        y_pred = (
            (y_prob >= threshold).astype(int) if self.number_classes == 1 else y_prob.argmax(axis=1)
        )

        is_binary = self.number_classes <= 2

        # Binary → 1D prob vector; multiclass → full (N, C) matrix for OVR AUROC
        if self.number_classes == 2:
            y_prob_for_metrics = y_prob[:, 1]  # positive-class prob for binary AUROC
        elif self.number_classes == 1:
            y_prob_for_metrics = y_prob  # already 1D
        else:
            y_prob_for_metrics = y_prob  # (N, C) matrix — OVR/macro AUROC in metrics.py

        metrics = compute_metrics_with_ci(y_true, y_pred, y_prob_for_metrics, seed=self.random_seed)

        logger.info(
            f"Test set evaluation at threshold {threshold:.3f} : "
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
                "confusion_matrix": confusion_matrix(y_true, y_pred),
            },
            "inference_time_ms": np.mean(inference_times) * 1000,
            "predictions": y_pred,
            "labels": y_true,
            "probabilities": y_prob,
            "probabilities_1d": y_prob_for_metrics,
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
                    best_method=f"mean_prob-{self.clip_threshold}",
                )
        elif aggregate_by_clip and clip_ids and not is_binary:
            if len(clip_ids) != len(y_true):
                logger.warning("clip_ids length mismatch — skipping clip aggregation.")
            else:
                # Multiclass clip aggregation: average per-class probs across frames,
                # then take argmax as clip prediction.
                from collections import defaultdict as _dd

                clip_data: dict = _dd(lambda: {"probs": [], "labels": []})
                for i, cid in enumerate(clip_ids):
                    clip_data[cid]["probs"].append(y_prob[i])  # shape (n_classes,)
                    clip_data[cid]["labels"].append(int(y_true[i]))

                clip_ids_agg, clip_y_true, clip_y_pred, clip_y_prob = [], [], [], []
                for cid, data in clip_data.items():
                    mean_prob = np.mean(data["probs"], axis=0)  # (n_classes,)
                    clip_y_pred.append(int(np.argmax(mean_prob)))
                    clip_y_prob.append(mean_prob)
                    # Modal label for the clip
                    from scipy import stats as _stats

                    clip_y_true.append(int(_stats.mode(data["labels"], keepdims=True).mode[0]))
                    clip_ids_agg.append(cid)

                clip_y_true = np.array(clip_y_true)
                clip_y_pred = np.array(clip_y_pred)
                clip_y_prob = np.array(clip_y_prob)  # (n_clips, n_classes)

                clip_metrics = compute_metrics_with_ci(
                    clip_y_true, clip_y_pred, clip_y_prob, seed=self.random_seed
                )
                results["clip_level"] = {
                    "method": "mean_prob_argmax",
                    "accuracy": clip_metrics["_Accuracy_mean"],
                    "precision": clip_metrics.get("_Precision_mean", float("nan")),
                    "recall": clip_metrics["_Sensitivity_mean"],
                    "f1": clip_metrics["_F1_mean"],
                    "roc_auc": clip_metrics["_AUROC_mean"],
                    "confusion_matrix": confusion_matrix(clip_y_true, clip_y_pred),
                    "n_clips": len(clip_ids_agg),
                    "clip_ids": clip_ids_agg,
                    "clip_predictions": clip_y_pred,
                    "clip_labels": clip_y_true,
                    "clip_probs": clip_y_prob,
                }

        return results

    def _add_clip_level_results(
        self, results, y_pred, y_true, y_prob, clip_ids, compare=False, best_method=None
    ):
        if compare:
            comparison_df = compare_aggregation_methods(
                probabilities=y_prob,
                predictions=y_pred,
                labels=y_true,
                clip_ids=clip_ids,
            )
            logger.info("\n" + "=" * 80)
            logger.info("CLIP-LEVEL AGGREGATION COMPARISON")
            logger.info("=" * 80)
            logger.info("\n" + comparison_df.to_string(index=False))

            best_method = comparison_df.iloc[0]["method"]
            logger.info(f"Best method: {best_method} (F1={comparison_df.iloc[0]['f1']:.4f})")

        if not best_method:
            best_method = f"mean_prob-{self.threshold}"  # default method if not comparing

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

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _log_epoch_metrics(
        self,
        epoch,
        train_loss,
        train_acc,
        val_loss,
        val_acc,
        precision,
        recall,
        f1,
        auroc,
        lr,
    ):
        # No-op: epoch metrics are captured in the history dict returned by fit()
        pass
