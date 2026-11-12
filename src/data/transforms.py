"""
Image transformation pipeline for colonoscopy ulcer detection.

Exports
-------
    ResizeWithPad  : aspect-ratio-preserving resize + zero-padding to square
    CLAHE_Y        : CLAHE on luminance channel (Y in YCbCr)
    get_transforms : build the full torchvision pipeline
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
import torchvision.transforms as T
from PIL import Image

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)
"""Default normalisation (same values as ``src.config.models.IMAGENET_MEAN/STD``)."""


class ResizeWithPad:
    """Resize to `target_size` while preserving aspect ratio, then pad to square."""

    def __init__(self, target_size: int, fill: int = 0):
        self.target_size = target_size
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        scale = self.target_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = T.functional.resize(img, (new_h, new_w))

        pad_w = self.target_size - new_w
        pad_h = self.target_size - new_h
        padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
        return T.functional.pad(img, padding, fill=self.fill)


class CLAHE_Y:
    """
    Contrast Limited Adaptive Histogram Equalization applied only to the
    luminance channel (Y in YCbCr) to enhance tissue details without
    altering colour information.
    """

    def __call__(self, img: Image.Image) -> Image.Image:
        img_np = np.array(img)
        ycrcb = cv2.cvtColor(img_np, cv2.COLOR_RGB2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        y_eq = clahe.apply(y)
        img_eq = cv2.cvtColor(cv2.merge((y_eq, cr, cb)), cv2.COLOR_YCrCb2RGB)
        return Image.fromarray(img_eq)


def deterministic_steps(img_size: int, equalize: bool) -> list:
    """The fixed preprocessing steps (cached by ``src.data.frame_cache``)."""
    return [ResizeWithPad(img_size, fill=0)] + ([CLAHE_Y()] if equalize else [])


def get_transforms(
    img_size: int,
    is_training: bool = True,
    equalize: bool = True,
    cached: bool = False,
    rotation_degrees: int = 30,
    translate_fraction: float = 0.1,
    shear_degrees: float = 5.0,
    horizontal_flip_prob: float = 0.5,
    brightness_factor: float = 0.15,
    contrast_factor: float = 0.25,
    saturation_factor: float = 0.05,
    use_randaugment: bool = False,
    randaugment_n: int = 2,
    randaugment_m: int = 9,
    use_random_erasing: bool = False,
    random_erasing_p: float = 0.25,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
) -> T.Compose:
    """
    Build the torchvision transform pipeline.

    Args:
        img_size:             Target square resolution in pixels.
        is_training:          Whether to include random augmentations.
        equalize:             Whether to apply CLAHE on the Y channel.
        cached:               Inputs are already resized / equalised (frame cache):
                              skip ``deterministic_steps``.
        rotation_degrees:     Max random rotation angle (degrees).
        translate_fraction:   Max random shift, fraction of the width / height.
        shear_degrees:        Max random shear along each axis (degrees).
        horizontal_flip_prob: Probability for RandomHorizontalFlip.
        brightness_factor:    ColorJitter brightness range.
        contrast_factor:      ColorJitter contrast range.
        saturation_factor:    ColorJitter saturation range.
        use_randaugment:      Apply RandAugment after standard augmentations.
        randaugment_n:        Number of RandAugment operations per image.
        randaugment_m:        RandAugment magnitude (0–30).
        use_random_erasing:   Apply RandomErasing after normalisation.
        random_erasing_p:     RandomErasing probability.
        mean:                 Per-channel normalisation mean of the backbone's
                              pretraining (default: ImageNet). Use
                              ``src.config.models.get_norm_stats(model)``.
        std:                  Per-channel normalisation std (default: ImageNet).

    Returns:
        torchvision.transforms.Compose pipeline.

    Notes:
        - Rotation, shift and shear are one RandomAffine (a single resampling).
        - Hue shift is disabled to preserve tissue colour fidelity.
        - RandomErasing uses a small scale (0.02–0.10) to avoid erasing
          clinically relevant tissue regions.
    """
    pipeline: list = [] if cached else deterministic_steps(img_size, equalize)

    if is_training:
        pipeline += [
            T.RandomHorizontalFlip(p=horizontal_flip_prob),
            T.RandomAffine(
                degrees=rotation_degrees,
                translate=(translate_fraction, translate_fraction),
                shear=(-shear_degrees, shear_degrees, -shear_degrees, shear_degrees),
            ),
            T.ColorJitter(
                brightness=brightness_factor,
                contrast=contrast_factor,
                saturation=saturation_factor,
                hue=0.0,
            ),
        ]
        if use_randaugment:
            pipeline.append(T.RandAugment(num_ops=randaugment_n, magnitude=randaugment_m))

    pipeline += [
        T.ToTensor(),
        T.Normalize(
            mean=tuple(IMAGENET_MEAN if mean is None else mean),
            std=tuple(IMAGENET_STD if std is None else std),
        ),
    ]

    if is_training and use_random_erasing:
        pipeline.append(T.RandomErasing(p=random_erasing_p, scale=(0.02, 0.10), ratio=(0.3, 3.3)))

    return T.Compose(pipeline)
