"""Dataset implementation for 2D lung nodule segmentation samples."""

from pathlib import Path

import numpy as np
import pandas as pd

from torch import Tensor
from torch.utils.data import Dataset


class LungDataset(Dataset):
    """
    PyTorch Dataset for lung nodule segmentation.
    """

    def __init__(
        self,
        root_dir: str | Path,
        split: str,
        transform=None,
    ) -> None:
        """
        Initialize the dataset.

        Parameters
        ----------
        root_dir : str or Path
            Root directory of the segmentation dataset.
        split : str
            Dataset split. Must be one of
            {"train", "val", "test"}.
        transform : callable, optional
            Albumentations transform pipeline.
        """

        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform

        if self.split not in {"train", "val", "test"}:
            raise ValueError(
                "split must be one of {'train', 'val', 'test'}"
            )

        # Dataset directories.
        self.ct_dir = self.root_dir / "ct"
        self.mask_dir = self.root_dir / "mask"

        # Load metadata and keep only the requested split.
        self.metadata = pd.read_csv(
            self.root_dir / "split_metadata.csv"
        )

        self.metadata = (
            self.metadata[self.metadata["split"] == self.split]
            .reset_index(drop=True)
        )

    def __len__(self) -> int:
        """
        Return the number of samples.

        Returns
        -------
        int
            Number of samples.
        """

        return len(self.metadata)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[Tensor, Tensor]:
        """
        Load one CT image and its corresponding segmentation mask.

        Parameters
        ----------
        index : int
            Sample index.

        Returns
        -------
        tuple[Tensor, Tensor]
            CT image and segmentation mask.
        """

        # Retrieve sample metadata.
        row = self.metadata.iloc[index]

        filename = row["filename"]

        ct_path = self.ct_dir / filename
        mask_path = self.mask_dir / filename

        # Verify that both files exist.
        if not ct_path.exists():
            raise FileNotFoundError(
                f"CT file not found: {ct_path}"
            )

        if not mask_path.exists():
            raise FileNotFoundError(
                f"Mask file not found: {mask_path}"
            )

        # Load CT image and segmentation mask.
        ct = np.load(ct_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.float32)

        # Validate the loaded sample before applying transformations.
        if ct.shape != mask.shape:
            raise ValueError(
                "Image and mask shapes do not match.\n"
                f"File      : {filename}\n"
                f"CT shape  : {ct.shape}\n"
                f"Mask shape: {mask.shape}"
            )

        unique_values = np.unique(mask)

        if not np.all(np.isin(unique_values, [0.0, 1.0])):
            raise ValueError(
                "Mask contains non-binary values.\n"
                f"File         : {filename}\n"
                f"Unique values: {unique_values}"
            )

        if ct.ndim != 2:
            raise ValueError(
                f"Expected 2D CT image, got shape {ct.shape}"
            )

        if mask.ndim != 2:
            raise ValueError(
                f"Expected 2D mask, got shape {mask.shape}"
            )

        # Apply Albumentations transforms.
        if self.transform is not None:
            transformed = self.transform(
                image=ct,
                mask=mask,
            )

            ct = transformed["image"]
            mask = transformed["mask"].unsqueeze(0)

        return ct, mask
