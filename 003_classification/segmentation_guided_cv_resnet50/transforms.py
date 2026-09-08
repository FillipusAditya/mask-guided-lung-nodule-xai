"""Paired CT/probability-map transforms for the guided classifier."""

from __future__ import annotations

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2


def build_train_transform(
    height: int,
    width: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    seed: int,
) -> A.Compose:
    """Match the baseline augmentation while preserving soft mask values."""

    return A.Compose(
        [
            A.Resize(
                height=height,
                width=width,
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_LINEAR,
            ),
            A.HorizontalFlip(p=0.5),
            A.Rotate(
                limit=15,
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0.0,
                fill_mask=0.0,
                p=0.5,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.10,
                contrast_limit=0.10,
                p=0.3,
            ),
            A.GaussNoise(std_range=(0.01, 0.03), p=0.2),
            A.Normalize(mean=mean, std=std, max_pixel_value=1.0),
            ToTensorV2(),
        ],
        seed=seed,
        is_check_shapes=False,
    )


def build_val_transform(
    height: int,
    width: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    seed: int,
) -> A.Compose:
    """Build deterministic paired validation/test preprocessing."""

    return A.Compose(
        [
            A.Resize(
                height=height,
                width=width,
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_LINEAR,
            ),
            A.Normalize(mean=mean, std=std, max_pixel_value=1.0),
            ToTensorV2(),
        ],
        seed=seed,
        is_check_shapes=False,
    )
