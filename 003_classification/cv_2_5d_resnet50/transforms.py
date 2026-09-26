"""Albumentations pipelines shared by 2.5D training and testing."""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2


def build_train_transform(config: dict) -> A.Compose:
    data = config["data"]
    aug = config["augmentation"]
    return A.Compose(
        [
            A.Resize(int(data["input_height"]), int(data["input_width"])),
            A.HorizontalFlip(p=float(aug["horizontal_flip_probability"])),
            A.Rotate(limit=int(aug["rotation_limit"]), border_mode=0, p=float(aug["rotation_probability"])),
            A.RandomBrightnessContrast(
                brightness_limit=float(aug["brightness_limit"]),
                contrast_limit=float(aug["contrast_limit"]),
                p=float(aug["brightness_contrast_probability"]),
            ),
            A.GaussNoise(
                std_range=tuple(float(v) for v in aug["noise_std_range"]),
                p=float(aug["noise_probability"]),
            ),
            A.Normalize(
                mean=tuple(data["normalization_mean"]),
                std=tuple(data["normalization_std"]),
                max_pixel_value=1.0,
            ),
            ToTensorV2(),
        ],
        seed=int(config["training"]["transform_seed"]),
    )


def build_eval_transform(config: dict) -> A.Compose:
    data = config["data"]
    return A.Compose(
        [
            A.Resize(int(data["input_height"]), int(data["input_width"])),
            A.Normalize(
                mean=tuple(data["normalization_mean"]),
                std=tuple(data["normalization_std"]),
                max_pixel_value=1.0,
            ),
            ToTensorV2(),
        ],
        seed=int(config["training"]["transform_seed"]),
    )

