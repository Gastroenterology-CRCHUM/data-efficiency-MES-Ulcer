"""
Saliency visualisation — GradCAM for CNNs, attention map for ViTs.

For ResNet / EfficientNet (CNN):
    Standard GradCAM — hooks the last conv block, pools gradients spatially,
    produces a 7×14 heatmap and upsamples to image size.

For ViT (DINO / timm):
    CLS-token attention map — temporarily patches the last self-attention
    module to capture the softmax attention weights, then averages over heads.
    The result is the attention the CLS token pays to each image patch,
    reshaped to a 14×14 grid.  No gradient computation needed.

Public API
----------
    get_target_layer(model) -> nn.Module | None   (CNN only; None for ViT)
    visualize_predictions(model, dataloader, label_map, device, ...) -> None
"""

from __future__ import annotations

import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from src.models.classifier import ClassifierModel

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------


def _to_uint8(tensor: torch.Tensor) -> np.ndarray:
    """CHW float tensor (ImageNet-normalised) → HWC uint8 RGB."""
    img = tensor.cpu().float().numpy()
    if img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
    img = img * _IMAGENET_STD + _IMAGENET_MEAN
    img = np.clip(img, 0.0, 1.0)
    return (img * 255).astype(np.uint8)


def _overlay(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a [0, 1] float heatmap onto an HWC uint8 RGB image."""
    h, w = image.shape[:2]
    heat_u8 = (cv2.resize(heatmap, (w, h)) * 255).astype(np.uint8)
    colormap = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    colormap_rgb = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)
    blended = alpha * colormap_rgb.astype(np.float32) + (1 - alpha) * image.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Architecture detection
# ---------------------------------------------------------------------------


def _last_child(module: nn.Module) -> nn.Module:
    """Return the last child submodule (works for Sequential and ModuleList)."""
    children = list(module.children())
    return children[-1] if children else module


def _vit_module(model: ClassifierModel) -> nn.Module | None:
    """Return the ViT submodule that owns .blocks, or None for CNN models.

    Handles:
      - DINO ViT (unwrapped_backbone has .blocks directly)
      - timm ViT wrapped in _HFViTBackbone (.vit.blocks)
    """
    backbone = model.unwrapped_backbone
    if hasattr(backbone, "blocks"):
        return backbone
    # _HFViTBackbone with a timm inner model
    if hasattr(backbone, "vit") and hasattr(backbone.vit, "blocks"):
        return backbone.vit  # type: ignore[attr-defined]
    return None


def get_target_layer(model: ClassifierModel) -> nn.Module | None:
    """Return the conv layer to hook for GradCAM (CNN models only).

    - ResNet       → last block of ``layer4``
    - EfficientNet → last block of ``features``
    - ViT          → None  (use attention map instead)
    - Others       → None  (unsupported)
    """
    if _vit_module(model) is not None:
        return None  # ViT: do not use GradCAM
    backbone = model.unwrapped_backbone
    if hasattr(backbone, "layer4"):
        return _last_child(backbone.layer4)  # type: ignore[arg-type]
    if hasattr(backbone, "features"):
        return _last_child(backbone.features)  # type: ignore[arg-type]
    return None


# ---------------------------------------------------------------------------
# ViT — CLS-token attention map
# ---------------------------------------------------------------------------


def _extract_vit_attention(vit: nn.Module, image: torch.Tensor) -> np.ndarray | None:
    """
    CLS-token attention map from the last transformer block.

    Temporarily patches the forward method of the last block's attention
    module to capture the softmax weights before they are discarded.
    Works with DINO-style attention modules that expose .qkv and .num_heads.

    Returns a [0, 1] float32 array shaped (grid, grid) — typically 14×14
    for a 224 px image with patch_size=16.  Returns None if the module is
    not compatible (e.g. HuggingFace attention without .qkv).
    """
    blocks = list(getattr(vit, "blocks", []))
    if not blocks:
        return None
    last_attn = blocks[-1].attn
    if not (hasattr(last_attn, "qkv") and hasattr(last_attn, "num_heads")):
        return None

    captured: list[torch.Tensor] = []

    def hooked_forward(x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        nh = last_attn.num_heads
        head_dim = C // nh
        qkv = last_attn.qkv(x).reshape(B, N, 3, nh, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * last_attn.scale
        attn = attn.softmax(dim=-1)
        captured.append(attn.detach())          # (B, n_heads, N, N)
        if hasattr(last_attn, "attn_drop"):
            attn = last_attn.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = last_attn.proj(out)
        if hasattr(last_attn, "proj_drop"):
            out = last_attn.proj_drop(out)
        return out

    # Swap forward on the instance (not the class) — safe to restore
    original_fwd = last_attn.forward
    last_attn.forward = hooked_forward
    try:
        with torch.no_grad():
            vit(image)
    finally:
        last_attn.forward = original_fwd

    if not captured:
        return None

    # (n_heads, N, N) — N = n_patches + 1, CLS token is index 0
    attn = captured[0][0]       # drop batch dim
    cls_attn = attn[:, 0, 1:]   # (n_heads, n_patches) — CLS → each patch
    avg = cls_attn.mean(dim=0)  # average over heads → (n_patches,)

    grid = int(math.isqrt(avg.shape[0]))
    heatmap = avg.reshape(grid, grid).cpu().numpy()
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return heatmap.astype(np.float32)


# ---------------------------------------------------------------------------
# CNN — GradCAM
# ---------------------------------------------------------------------------


class _GradCAM:
    """Forward/backward hooks on a conv layer → spatial activation map."""

    def __init__(self, target_layer: nn.Module) -> None:
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._hooks = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, _module, _input, output):
        self._activations = output.detach().clone()

    def _save_gradient(self, _module, _grad_in, grad_out):
        self._gradients = grad_out[0].detach().clone()

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()

    def generate(self, logits: torch.Tensor, class_idx: int) -> np.ndarray:
        """Backprop through ``class_idx`` and return a [0, 1] heatmap."""
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        if logits.shape[1] == 1:
            logits[:, 0].backward(retain_graph=True)
        else:
            logits[:, class_idx].backward(retain_graph=True)

        acts = self._activations   # (1, C, H, W)
        grads = self._gradients    # (1, C, H, W)
        if acts is None or grads is None:
            raise RuntimeError("GradCAM hooks did not capture data.")

        weights = grads.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * acts).sum(dim=1).squeeze(0)    # (H, W)
        cam = torch.relu(cam).cpu().numpy()
        if cam.max() > 1e-8:
            cam /= cam.max()
        return cam.astype(np.float32)


def _compute_gradcam(
    model_forward: nn.Module,
    image: torch.Tensor,
    target_layer: nn.Module,
    class_idx: int,
) -> np.ndarray:
    """One-shot GradCAM for a single image (batch size 1)."""
    gc = _GradCAM(target_layer)
    model_forward.zero_grad()
    image = image.detach().requires_grad_(True)
    logits = model_forward(image)
    if hasattr(logits, "logits"):
        logits = logits.logits
    heatmap = gc.generate(logits, class_idx)
    gc.remove()
    return heatmap


# ---------------------------------------------------------------------------
# Main visualisation function
# ---------------------------------------------------------------------------


def visualize_predictions(
    model: ClassifierModel,
    dataloader,
    label_map: dict[int, str],
    device: torch.device,
    n_samples: int = 8,
    alpha: float = 0.45,
    title: str = "Saliency",
    ncols: int = 4,
    class_idx: int | None = None,
) -> None:
    """
    Display saliency overlays for ``n_samples`` images from ``dataloader``.

    CNN backbones use GradCAM (gradient-weighted class activation map).
    ViT backbones use the CLS-token attention map from the last block.

    Args:
        model:      A loaded ``ClassifierModel`` (weights already restored).
        dataloader: Any DataLoader whose batches start with (image_tensor, label, ...).
        label_map:  {int → str} class name mapping.
        device:     Torch device.
        n_samples:  Number of images to show.
        alpha:      Heatmap opacity (0 = original only, 1 = heatmap only).
        title:      Figure suptitle.
        ncols:      Columns in the grid.
        class_idx:  Target class for GradCAM.  None = use predicted class.
                    Ignored for ViT (attention map is class-agnostic).
    """
    vit = _vit_module(model)
    is_vit = vit is not None
    method_label = "Attention" if is_vit else "GradCAM"

    if not is_vit:
        target_layer = get_target_layer(model)
        if target_layer is None:
            print(
                f"[explainability] '{model.name}': no supported layer found "
                "(expected layer4 / features for CNN, or blocks for ViT). Skipping."
            )
            return

    model.base_model.to(device)
    model.base_model.eval()

    # ── Collect samples ───────────────────────────────────────────────────────
    images_collected: list[torch.Tensor] = []
    labels_collected: list[int] = []
    for batch in dataloader:
        imgs, lbls = batch[0], batch[1]
        for i in range(len(imgs)):
            images_collected.append(imgs[i].unsqueeze(0))
            labels_collected.append(int(lbls[i]))
            if len(images_collected) >= n_samples:
                break
        if len(images_collected) >= n_samples:
            break

    n = len(images_collected)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    axes = np.array(axes).reshape(-1)
    fig.suptitle(f"{title}  [{method_label}]", fontsize=14, fontweight="bold")

    for idx, (img_t, true_label) in enumerate(zip(images_collected, labels_collected)):
        ax = axes[idx]
        img_t = img_t.to(device)

        # ── Predict ────────────────────────────────────────────────────────────
        with torch.no_grad():
            logits = model.base_model(img_t)
            if hasattr(logits, "logits"):
                logits = logits.logits
            logits_np = logits.cpu().squeeze()
            if model.number_classes == 1:
                prob_pos = float(torch.sigmoid(logits_np))
                pred_idx = int(prob_pos >= (model.threshold or 0.5))
                prob_str = f"{prob_pos:.2f}"
            else:
                probs = torch.softmax(logits_np, dim=-1).numpy()
                pred_idx = int(np.argmax(probs))
                prob_str = f"{probs[pred_idx]:.2f}"

        # ── Saliency ───────────────────────────────────────────────────────────
        heatmap: np.ndarray | None = None
        err_msg = ""
        try:
            if is_vit:
                # ViT: CLS-token attention map — no gradient needed
                assert vit is not None
                heatmap = _extract_vit_attention(vit, img_t)
            else:
                # CNN: GradCAM
                target_cls = class_idx if class_idx is not None else pred_idx
                with torch.enable_grad():
                    heatmap = _compute_gradcam(
                        model.base_model, img_t, target_layer, target_cls  # type: ignore[arg-type]
                    )
        except Exception as exc:  # noqa: BLE001
            err_msg = str(exc)

        if heatmap is None:
            heatmap = np.zeros((14, 14), dtype=np.float32)
            if err_msg:
                ax.set_xlabel(f"[{method_label} failed: {err_msg}]", fontsize=6, color="red")

        # ── Display ────────────────────────────────────────────────────────────
        orig_rgb = _to_uint8(img_t.squeeze(0).cpu())
        blended = _overlay(orig_rgb, heatmap, alpha=alpha)
        ax.imshow(blended)

        pred_name = label_map.get(pred_idx, str(pred_idx))
        true_name = label_map.get(true_label, str(true_label))
        color = "green" if pred_idx == true_label else "red"
        ax.set_title(
            f"Pred: {pred_name} ({prob_str})\nTrue: {true_name}",
            fontsize=9,
            color=color,
            fontweight="bold" if pred_idx != true_label else "normal",
        )
        ax.axis("off")

    for ax in axes[n:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.show()
