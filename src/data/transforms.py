"""
Image transformation pipeline for colonoscopy ulcer detection.

Exports
-------
    ResizeWithPad  : aspect-ratio-preserving resize + zero-padding to square
    CLAHE_Y        : CLAHE on luminance channel (Y in YCbCr)
    get_transforms : build the full torchvision pipeline
"""

from __future__ import annotations

import cv2
import numpy as np
import torchvision.transforms as T
from PIL import Image


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


def get_transforms(
    img_size: int,
    is_training: bool = True,
    equalize: bool = True,
    rotation_degrees: int = 30,
    horizontal_flip_prob: float = 0.5,
    brightness_factor: float = 0.15,
    contrast_factor: float = 0.25,
    saturation_factor: float = 0.05,
    use_randaugment: bool = False,
    randaugment_n: int = 2,
    randaugment_m: int = 9,
    use_random_erasing: bool = False,
    random_erasing_p: float = 0.25,
) -> T.Compose:
    """
    Build the torchvision transform pipeline.

    Args:
        img_size:             Target square resolution in pixels.
        is_training:          Whether to include random augmentations.
        equalize:             Whether to apply CLAHE on the Y channel.
        rotation_degrees:     Max random rotation angle (degrees).
        horizontal_flip_prob: Probability for RandomHorizontalFlip.
        brightness_factor:    ColorJitter brightness range.
        contrast_factor:      ColorJitter contrast range.
        saturation_factor:    ColorJitter saturation range.
        use_randaugment:      Apply RandAugment after standard augmentations.
        randaugment_n:        Number of RandAugment operations per image.
        randaugment_m:        RandAugment magnitude (0–30).
        use_random_erasing:   Apply RandomErasing after normalisation.
        random_erasing_p:     RandomErasing probability.

    Returns:
        torchvision.transforms.Compose pipeline.

    Notes:
        - Hue shift is disabled to preserve tissue colour fidelity.
        - ImageNet statistics are used for transfer-learning compatibility.
        - RandomErasing uses a small scale (0.02–0.10) to avoid erasing
          clinically relevant tissue regions.
    """
    pipeline: list = [ResizeWithPad(img_size, fill=0)]

    if equalize:
        pipeline.append(CLAHE_Y())

    if is_training:
        pipeline += [
            T.RandomHorizontalFlip(p=horizontal_flip_prob),
            T.RandomRotation(degrees=rotation_degrees),
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
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]

    if is_training and use_random_erasing:
        pipeline.append(T.RandomErasing(p=random_erasing_p, scale=(0.02, 0.10), ratio=(0.3, 3.3)))

    return T.Compose(pipeline)
