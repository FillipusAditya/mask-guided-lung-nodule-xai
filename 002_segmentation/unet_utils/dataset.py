"""Dataset for paired 2D CT images and lung-nodule masks."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .tiles import split_into_tiles


class LungDataset(Dataset):
    """
    PyTorch Dataset for 2D lung nodule segmentation.

    The dataset loads paired CT images and segmentation masks from NumPy
    files. Samples can be selected using either a fixed holdout split or a
    Group K-Fold cross-validation split.
    """

    def __init__(
        self,
        root_dir: str | Path,
        split: str,
        split_method: str = "holdout_split",
        metadata_filename: str | Path = "001_holdout_split_lidc_lndb.csv",
        fold: int | None = None,
        image_path_column: str = "ct_parenchyma_path",
        tile_grid_size: int = 4,
        transform=None,
    ) -> None:
        """
        Initialize the lung nodule segmentation dataset.

        Parameters
        ----------
        root_dir : str or Path
            Root directory containing the metadata, CT images, and masks.
        split : str
            Dataset subset. Expected values are ``"train"``, ``"val"``,
            or ``"test"``.
        split_method : str, default="holdout_split"
            Dataset splitting method. Use ``"holdout_split"`` for fixed
            splits or ``"group_kfold"`` for Group K-Fold cross-validation.
        metadata_filename : str or Path
            Metadata CSV filename or path.
        fold : int, optional
            Validation fold used by Group K-Fold cross-validation.
        image_path_column : str, default="ct_parenchyma_path"
            Metadata column containing relative CT image paths.
        tile_grid_size : int, default=4
            Number of tile rows and columns used to divide each sample.
        transform : callable, optional
            Albumentations transform applied jointly to the CT and mask.
        """

        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_path_column = image_path_column
        self.tile_grid_size = tile_grid_size

        # Resolve and load the metadata CSV.
        metadata_path = Path(metadata_filename)
        if not metadata_path.is_absolute():
            metadata_path = self.root_dir / metadata_path

        metadata = pd.read_csv(metadata_path)

        # Select samples using fixed holdout assignments.
        if split_method == "holdout_split":
            metadata = metadata[metadata["split"] == split]

        # Select development folds while preserving a fixed holdout test set.
        else:
            development = metadata["cv_role"] == "development"
            holdout_test = metadata["cv_role"] == "holdout_test"

            if split == "train":
                metadata = metadata[development & (metadata["cv_fold"] != fold)]
            elif split == "val":
                metadata = metadata[development & (metadata["cv_fold"] == fold)]
            else:
                metadata = metadata[holdout_test & (metadata["cv_fold"] == -1)]

        self.metadata = metadata.reset_index(drop=True)

    def __len__(self) -> int:
        """
        Return the number of selected samples.

        Returns
        -------
        int
            Number of samples in the selected subset.
        """

        return len(self.metadata)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """
        Load one CT image and its corresponding segmentation mask.

        Parameters
        ----------
        index : int
            Sample index.

        Returns
        -------
        tuple[Tensor, Tensor]
            CT image tiles and binary mask tiles as contiguous ``float32``
            tensors with shape ``[T, 1, H_tile, W_tile]``.
        """

        # Retrieve the sample paths from metadata.
        row = self.metadata.iloc[index]
        image_path = self.root_dir / row[self.image_path_column]
        mask_path = self.root_dir / row["mask_path"]

        # Load the CT image and segmentation mask.
        image = np.load(image_path, allow_pickle=False).astype(np.float32, copy=False)
        mask = np.load(mask_path, allow_pickle=False).astype(np.float32, copy=False)

        # Apply paired transforms to keep image and mask spatially aligned.
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        # Convert the transformed arrays to float32 tensors.
        image = torch.as_tensor(image, dtype=torch.float32)
        mask = torch.as_tensor(mask, dtype=torch.float32)

        # Add the channel dimension to grayscale tensors when necessary.
        if image.ndim == 2:
            image = image.unsqueeze(0)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        # Divide the image and mask into spatially aligned tiles.
        image = split_into_tiles(image, grid_size=self.tile_grid_size)
        mask = split_into_tiles(mask, grid_size=self.tile_grid_size)

        return image, mask
