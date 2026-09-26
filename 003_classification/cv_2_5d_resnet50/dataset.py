"""Dataset that stacks adjacent axial CT slices as pseudo-RGB input."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset


SLICE_PATTERN = re.compile(r"_slice_(?P<index>\d+)\.npy$")


def extract_slice_index(filename: str) -> int:
    """Extract the axial index encoded in a classification filename."""

    match = SLICE_PATTERN.search(Path(filename).name)
    if match is None:
        raise ValueError(f"Cannot extract slice index from filename: {filename}")
    return int(match.group("index"))


class TwoPointFiveDLungDataset(Dataset):
    """Return ``[z-1, z, z+1]`` as channels for every centre slice."""

    VALID_SPLITS = {"train", "val", "test"}

    def __init__(
        self,
        root_dir: str | Path,
        metadata_path: str | Path,
        split: str,
        transform: Any = None,
        class_to_idx: dict[str, int] | None = None,
        ct_path_column: str = "ct_parenchyma_path",
        nodule_column: str = "cv_nodule_id",
        group_column: str = "cv_group_id",
        role_column: str = "cv_role",
        fold_column: str = "cv_fold",
        development_role: str = "development",
        holdout_role: str = "holdout_test",
        cv_fold: int | None = None,
        slice_offsets: tuple[int, ...] = (-1, 0, 1),
        boundary_mode: str = "replicate_center",
    ) -> None:
        self.root_dir = Path(root_dir)
        self.metadata_path = Path(metadata_path)
        if not self.metadata_path.is_absolute():
            self.metadata_path = self.root_dir / self.metadata_path
        self.split = split.strip().lower()
        self.transform = transform
        self.ct_path_column = ct_path_column
        self.nodule_column = nodule_column
        self.group_column = group_column
        self.role_column = role_column
        self.fold_column = fold_column
        self.slice_offsets = tuple(int(value) for value in slice_offsets)
        self.boundary_mode = boundary_mode
        self.class_to_idx = dict(
            class_to_idx or {"benign": 0, "malignant": 1}
        )

        if self.split not in self.VALID_SPLITS:
            raise ValueError(f"Unsupported split: {split}")
        if self.slice_offsets != (-1, 0, 1):
            raise ValueError("This ResNet-50 implementation requires offsets [-1, 0, 1].")
        if boundary_mode not in {"replicate_center", "valid_only"}:
            raise ValueError("boundary_mode must be replicate_center or valid_only.")
        if set(self.class_to_idx.values()) != {0, 1}:
            raise ValueError("Binary class indices must be exactly {0, 1}.")
        if not self.metadata_path.is_file():
            raise FileNotFoundError(f"Metadata not found: {self.metadata_path}")

        frame = pd.read_csv(self.metadata_path)
        required = {
            "dataset", "patient_id", "filename", "label", self.ct_path_column,
            self.nodule_column, self.group_column, self.role_column, self.fold_column,
        }
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Metadata columns are missing: {sorted(missing)}")
        if frame[list(required)].isna().any().any():
            raise ValueError("Required metadata columns contain missing values.")

        frame = frame.copy()
        frame["label"] = frame["label"].astype(str).str.strip().str.lower()
        frame[self.role_column] = (
            frame[self.role_column].astype(str).str.strip().str.lower()
        )
        frame[self.fold_column] = pd.to_numeric(
            frame[self.fold_column], errors="raise"
        ).astype(int)
        frame["slice_index"] = frame["filename"].map(extract_slice_index)
        invalid_labels = set(frame["label"]) - set(self.class_to_idx)
        if invalid_labels:
            raise ValueError(f"Unsupported labels: {sorted(invalid_labels)}")

        if cv_fold is None:
            if "split" not in frame:
                raise ValueError("Metadata requires split when cv_fold is omitted.")
            partition = frame["split"].astype(str).str.lower().eq(self.split)
        elif self.split == "train":
            partition = frame[self.role_column].eq(development_role) & frame[
                self.fold_column
            ].ne(cv_fold)
        elif self.split == "val":
            partition = frame[self.role_column].eq(development_role) & frame[
                self.fold_column
            ].eq(cv_fold)
        else:
            partition = frame[self.role_column].eq(holdout_role)

        frame = frame.loc[partition].copy()
        if frame.empty:
            raise ValueError(f"No rows found for split={self.split}, fold={cv_fold}.")
        if frame.duplicated([self.nodule_column, "slice_index"]).any():
            raise ValueError("Duplicate slice indices exist inside a nodule.")
        if frame.groupby(self.nodule_column)["label"].nunique().gt(1).any():
            raise ValueError("A nodule has inconsistent labels.")
        if frame.groupby(self.nodule_column)[self.group_column].nunique().gt(1).any():
            raise ValueError("A nodule belongs to more than one patient group.")

        frame = frame.sort_values(
            [self.nodule_column, "slice_index"], kind="stable"
        ).reset_index(drop=True)
        lookup = {
            (str(row[self.nodule_column]), int(row["slice_index"])): index
            for index, row in frame.iterrows()
        }
        window_rows: list[dict[str, Any]] = []
        kept_indices: list[int] = []
        replicated = 0
        for center_index, row in frame.iterrows():
            nodule = str(row[self.nodule_column])
            center_slice = int(row["slice_index"])
            members: list[int] = []
            used_replication = False
            for offset in self.slice_offsets:
                member = lookup.get((nodule, center_slice + offset))
                if member is None:
                    if self.boundary_mode == "valid_only":
                        members = []
                        break
                    member = center_index
                    used_replication = True
                members.append(member)
            if not members:
                continue
            kept_indices.append(center_index)
            replicated += int(used_replication)
            member_rows = frame.iloc[members]
            window_rows.append(
                {
                    "input_slice_indices": tuple(
                        int(value) for value in member_rows["slice_index"]
                    ),
                    "input_filenames": tuple(member_rows["filename"].astype(str)),
                    "input_row_indices": tuple(members),
                    "boundary_replicated": used_replication,
                }
            )

        self.source_metadata = frame
        self.metadata = frame.iloc[kept_indices].reset_index(drop=True)
        self.windows = window_rows
        if self.metadata.empty:
            raise ValueError(
                "No 2.5D windows remain after applying the boundary policy."
            )
        self.boundary_replication_count = replicated
        self.classes = [
            name for name, _ in sorted(self.class_to_idx.items(), key=lambda item: item[1])
        ]
        self.targets = self.metadata["label"].map(self.class_to_idx).astype(int).tolist()

    def __len__(self) -> int:
        return len(self.metadata)

    def resolve_path(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.root_dir / path

    def get_input_paths(self, index: int) -> tuple[Path, Path, Path]:
        rows = self.source_metadata.iloc[list(self.windows[index]["input_row_indices"])]
        return tuple(self.resolve_path(value) for value in rows[self.ct_path_column])  # type: ignore[return-value]

    def load_window(self, index: int) -> np.ndarray:
        arrays = []
        shapes = set()
        for path in self.get_input_paths(index):
            if not path.is_file():
                raise FileNotFoundError(f"CT slice not found: {path}")
            array = np.load(path, allow_pickle=False).astype(np.float32, copy=False)
            if array.ndim != 2 or not np.isfinite(array).all():
                raise ValueError(f"Invalid 2D CT slice: {path} ({array.shape})")
            arrays.append(array)
            shapes.add(array.shape)
        if len(shapes) != 1:
            raise ValueError(f"Window slices have different shapes: {sorted(shapes)}")
        return np.stack(arrays, axis=-1)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        image = self.load_window(index)
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        if not isinstance(image, Tensor) or image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(f"Expected transformed shape [3,H,W], got {image.shape}.")
        target = torch.tensor(self.targets[index], dtype=torch.long)
        return image, target, torch.tensor(index, dtype=torch.long)
