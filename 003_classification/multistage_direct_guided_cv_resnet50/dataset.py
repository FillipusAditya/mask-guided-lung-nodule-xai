"""Paired CT and probability-map dataset for direct-guided classification."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

try:
    from ..utils.dataset import LungClassificationDataset
except ImportError:
    from utils.dataset import LungClassificationDataset


class DirectGuidedClassificationDataset(LungClassificationDataset):
    """Load a CT image, probability map, and classification label."""

    def __init__(self, *args, probability_root: str | Path, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.probability_root = Path(probability_root)

        if not self.probability_root.is_dir():
            raise FileNotFoundError(
                f"Probability-map directory not found: {self.probability_root}"
            )

        missing_probability_maps = [
            str(filename)
            for filename in self.metadata["filename"]
            if not self.get_probability_path_from_filename(filename).is_file()
        ]
        if missing_probability_maps:
            examples = ", ".join(missing_probability_maps[:5])
            raise FileNotFoundError(
                f"Missing {len(missing_probability_maps)} probability maps; "
                f"examples: {examples}"
            )

    def get_probability_path_from_filename(self, filename: str) -> Path:
        """Return the probability-map path for one metadata filename."""

        return self.probability_root / Path(str(filename)).name

    def get_probability_path(self, index: int) -> Path:
        """Return the probability-map path for one dataset index."""

        filename = str(self.metadata.iloc[index]["filename"])
        return self.get_probability_path_from_filename(filename)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        row = self.metadata.iloc[index]
        filename = str(row["filename"])

        ct_image = np.load(
            self.get_ct_path(index),
            allow_pickle=False,
        ).astype(np.float32, copy=False)
        probability_map = np.load(
            self.get_probability_path(index),
            allow_pickle=False,
        ).astype(np.float32, copy=False)

        if ct_image.ndim != 2 or probability_map.ndim != 2:
            raise ValueError(
                f"Expected paired 2D arrays for {filename}; received "
                f"CT {ct_image.shape}, probability {probability_map.shape}."
            )
        if not np.isfinite(ct_image).all():
            raise ValueError(f"CT image contains non-finite values: {filename}")
        if not np.isfinite(probability_map).all():
            raise ValueError(
                f"Probability map contains non-finite values: {filename}"
            )
        if probability_map.min() < 0.0 or probability_map.max() > 1.0:
            raise ValueError(f"Probability map is outside [0, 1]: {filename}")

        ct_image = np.repeat(ct_image[..., None], repeats=3, axis=2)

        if self.transform is not None:
            transformed = self.transform(
                image=ct_image,
                mask=probability_map,
            )
            ct_tensor = transformed["image"].float()
            probability_tensor = transformed["mask"].float()
        else:
            ct_tensor = torch.from_numpy(
                ct_image.transpose(2, 0, 1)
            ).float()
            probability_tensor = torch.from_numpy(probability_map).float()

        if probability_tensor.ndim == 2:
            probability_tensor = probability_tensor.unsqueeze(0)
        probability_tensor = probability_tensor.clamp_(0.0, 1.0)

        if ct_tensor.ndim != 3 or ct_tensor.shape[0] != 3:
            raise ValueError(f"Invalid transformed CT shape for {filename}.")
        if probability_tensor.shape != (1, *ct_tensor.shape[-2:]):
            raise ValueError(
                f"Invalid transformed probability-map shape for {filename}."
            )

        target = torch.tensor(
            self.class_to_idx[row["label"]],
            dtype=torch.long,
        )
        return ct_tensor, probability_tensor, target


def create_direct_guided_dataloader(
    root_dir: str | Path,
    split: str,
    batch_size: int,
    probability_root: str | Path,
    transform=None,
    class_to_idx: dict[str, int] | None = None,
    ct_path_column: str = "ct_windowed_path",
    shuffle: bool = False,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
    metadata_path: str | Path | None = None,
    cv_fold: int | None = None,
) -> DataLoader:
    """Build a DataLoader that returns CT, probability, and label tensors."""

    dataset = DirectGuidedClassificationDataset(
        root_dir=root_dir,
        split=split,
        probability_root=probability_root,
        transform=transform,
        class_to_idx=class_to_idx,
        ct_path_column=ct_path_column,
        metadata_path=metadata_path,
        cv_fold=cv_fold,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
