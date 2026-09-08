"""Paired CT and offline U-Net probability-map classification dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from ..utils.dataset import LungClassificationDataset


class ProbabilityGuidedClassificationDataset(LungClassificationDataset):
    """Load CT and probability arrays with synchronized spatial transforms."""

    def __init__(self, *args, probability_root: str | Path, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.probability_root = Path(probability_root)
        if not self.probability_root.is_dir():
            raise FileNotFoundError(
                f"Probability-map directory not found: {self.probability_root}"
            )

        missing = [
            str(filename)
            for filename in self.metadata["filename"]
            if not (self.probability_root / Path(str(filename)).name).is_file()
        ]
        if missing:
            preview = ", ".join(missing[:5])
            raise FileNotFoundError(
                f"Missing {len(missing)} probability maps; examples: {preview}"
            )

    def get_probability_path(self, index: int) -> Path:
        filename = Path(str(self.metadata.iloc[index]["filename"])).name
        return self.probability_root / filename

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        row = self.metadata.iloc[index]
        filename = str(row["filename"])
        ct = np.load(self.get_ct_path(index), allow_pickle=False).astype(
            np.float32, copy=False
        )
        probability = np.load(
            self.get_probability_path(index), allow_pickle=False
        ).astype(np.float32, copy=False)

        if ct.ndim != 2 or probability.ndim != 2:
            raise ValueError(
                f"Expected paired 2D arrays for {filename}; received "
                f"CT {ct.shape}, probability {probability.shape}."
            )
        if not np.isfinite(ct).all() or not np.isfinite(probability).all():
            raise ValueError(f"Non-finite CT or probability values: {filename}")
        if probability.min() < 0.0 or probability.max() > 1.0:
            raise ValueError(f"Probability map is outside [0, 1]: {filename}")

        ct_rgb = np.repeat(ct[..., None], repeats=3, axis=2)
        if self.transform is not None:
            # CT and probability maps may have different source resolutions.
            # The first paired transform resizes both to the same target size;
            # subsequent random spatial transforms then share their parameters.
            transformed = self.transform(image=ct_rgb, mask=probability)
            ct_tensor = transformed["image"]
            probability_tensor = transformed["mask"]
        else:
            ct_tensor = torch.from_numpy(ct_rgb.transpose(2, 0, 1)).float()
            probability_tensor = torch.from_numpy(probability).float()

        if probability_tensor.ndim == 2:
            probability_tensor = probability_tensor.unsqueeze(0)
        probability_tensor = probability_tensor.float().clamp_(0.0, 1.0)
        if ct_tensor.ndim != 3 or ct_tensor.shape[0] != 3:
            raise ValueError(f"Invalid transformed CT shape for {filename}.")
        if probability_tensor.shape != (1, *ct_tensor.shape[-2:]):
            raise ValueError(f"Invalid transformed probability shape for {filename}.")

        inputs = torch.cat((ct_tensor.float(), probability_tensor), dim=0)
        target = torch.tensor(self.class_to_idx[row["label"]], dtype=torch.long)
        return inputs, target


def create_probability_dataloader(
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
    dataset = ProbabilityGuidedClassificationDataset(
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
