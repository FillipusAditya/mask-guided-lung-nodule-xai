"""Dataset implementation for 2D lung nodule classification samples."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torch import Tensor
from torch.utils.data import Dataset


class LungClassificationDataset(Dataset):
    """Load CT slices and binary labels defined in split metadata."""

    VALID_SPLITS = {"train", "val", "test"}
    DEFAULT_CLASS_TO_IDX = {
        "benign": 0,
        "malignant": 1,
    }
    DEFAULT_CT_PATH_COLUMN = "ct_windowed_path"
    REQUIRED_COLUMNS = {"filename", "split", "label"}

    def __init__(
        self,
        root_dir: str | Path,
        split: str,
        transform=None,
        class_to_idx: dict[str, int] | None = None,
        ct_path_column: str = DEFAULT_CT_PATH_COLUMN,
    ) -> None:
        """Initialize a CSV-backed classification dataset split."""

        self.root_dir = Path(root_dir)
        self.metadata_path = self.root_dir / "split_metadata.csv"
        self.split = split.strip().lower()
        self.ct_path_column = ct_path_column.strip()
        self.transform = transform
        self.class_to_idx = dict(
            class_to_idx or self.DEFAULT_CLASS_TO_IDX
        )

        if self.split not in self.VALID_SPLITS:
            raise ValueError(
                "split must be one of {'train', 'val', 'test'}"
            )

        if not self.ct_path_column:
            raise ValueError("ct_path_column must not be empty.")

        if not self.metadata_path.is_file():
            raise FileNotFoundError(
                f"Split metadata not found: {self.metadata_path}"
            )

        if set(self.class_to_idx.values()) != set(
            range(len(self.class_to_idx))
        ):
            raise ValueError(
                "class_to_idx values must be contiguous indices starting at 0."
            )

        metadata = pd.read_csv(self.metadata_path)
        required_columns = self.REQUIRED_COLUMNS | {self.ct_path_column}
        missing_columns = required_columns - set(metadata.columns)

        if missing_columns:
            raise ValueError(
                "Split metadata is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        if metadata[list(required_columns)].isna().any().any():
            raise ValueError(
                f"Required columns must not contain missing values: "
                f"{sorted(required_columns)}"
            )

        metadata = metadata.copy()
        metadata["filename"] = metadata["filename"].astype(str).str.strip()
        metadata["split"] = metadata["split"].astype(str).str.strip().str.lower()
        metadata["label"] = metadata["label"].astype(str).str.strip().str.lower()
        metadata[self.ct_path_column] = (
            metadata[self.ct_path_column].astype(str).str.strip()
        )

        invalid_splits = set(metadata["split"].unique()) - self.VALID_SPLITS
        if invalid_splits:
            raise ValueError(
                f"Unsupported split values: {sorted(invalid_splits)}"
            )

        invalid_classes = (
            set(metadata["label"].unique()) - set(self.class_to_idx)
        )
        if invalid_classes:
            raise ValueError(
                f"Unsupported class values: {sorted(invalid_classes)}. "
                f"Expected: {sorted(self.class_to_idx)}"
            )

        self.metadata = (
            metadata[metadata["split"] == self.split]
            .reset_index(drop=True)
        )

        if self.metadata.empty:
            raise ValueError(
                f"No samples found for split '{self.split}' in "
                f"{self.metadata_path}."
            )

        self.classes = [
            class_name
            for class_name, _ in sorted(
                self.class_to_idx.items(),
                key=lambda item: item[1],
            )
        ]
        self.targets = (
            self.metadata["label"]
            .map(self.class_to_idx)
            .astype(int)
            .tolist()
        )

    def __len__(self) -> int:
        """Return the number of samples in this split."""

        return len(self.metadata)

    def get_ct_path(self, index: int) -> Path:
        """Return the CT path recorded by metadata for one sample."""

        relative_path = Path(
            str(self.metadata.iloc[index][self.ct_path_column])
        )
        if relative_path.is_absolute():
            return relative_path
        return self.root_dir / relative_path

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """Load one CT slice and its classification target."""

        row = self.metadata.iloc[index]
        filename = row["filename"]
        ct_path = self.get_ct_path(index)

        if not ct_path.is_file():
            raise FileNotFoundError(f"CT file not found: {ct_path}")

        ct = np.load(
            ct_path,
            allow_pickle=False,
        ).astype(np.float32, copy=False)

        if ct.ndim != 2:
            raise ValueError(
                f"Expected a 2D CT image, got shape {ct.shape} for {filename}."
            )

        if not np.isfinite(ct).all():
            raise ValueError(
                f"CT image contains non-finite values: {filename}"
            )

        # ResNet-50 pretrained on ImageNet expects three input channels.
        ct = np.repeat(ct[..., np.newaxis], repeats=3, axis=2)

        if self.transform is not None:
            ct = self.transform(image=ct)["image"]
        else:
            ct = torch.from_numpy(ct.transpose(2, 0, 1)).float()

        if not isinstance(ct, Tensor) or ct.ndim != 3 or ct.shape[0] != 3:
            raise ValueError(
                "Transform must return a tensor with shape [3, H, W], "
                f"but received {getattr(ct, 'shape', None)} for {filename}."
            )

        target = torch.tensor(
            self.class_to_idx[row["label"]],
            dtype=torch.long,
        )

        return ct, target
