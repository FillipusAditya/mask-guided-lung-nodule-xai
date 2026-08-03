from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class LungDataset(Dataset):
    """
    PyTorch Dataset for lung nodule segmentation.
    """

    def __init__(self, root_dir, split, transform=None):
        """
        Initialize the dataset.
        """

        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform

        if self.split not in {"train", "val", "test"}:
            raise ValueError(
                "split must be one of {'train', 'val', 'test'}"
            )

        self.ct_dir = self.root_dir / "ct"
        self.mask_dir = self.root_dir / "mask"

        self.metadata = pd.read_csv(
            self.root_dir / "split_metadata.csv"
        )

        self.metadata = (
            self.metadata[self.metadata["split"] == self.split]
            .reset_index(drop=True)
        )

    def __len__(self):
        """
        Return the number of samples.
        """
        return len(self.metadata)

    def __getitem__(self, index):
        """
        Load one CT image and its corresponding segmentation mask.
        """

        row = self.metadata.iloc[index]

        filename = row["filename"]

        ct_path = self.ct_dir / filename
        mask_path = self.mask_dir / filename

        if not ct_path.exists():
            raise FileNotFoundError(
                f"CT file not found: {ct_path}"
            )

        if not mask_path.exists():
            raise FileNotFoundError(
                f"Mask file not found: {mask_path}"
            )

        ct = np.load(ct_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.float32)

        if self.transform is not None:
            transformed = self.transform(
                image=ct,
                mask=mask,
            )

            ct = transformed["image"]
            mask = transformed["mask"]

        return ct, mask


if __name__ == "__main__":

    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    transform = A.Compose(
        [
            A.Resize(
                height=160,
                width=240,
            ),
            ToTensorV2(),
        ]
    )

    dataset = LungDataset(
        root_dir=Path(
            r"D:\xai_lung_nodules_research\dataset\_segmentation_dataset"
        ),
        split="train",
        transform=transform,
    )

    print(f"Dataset size : {len(dataset)}")

    ct, mask = dataset[0]

    print("\nFirst sample")

    print(f"CT type       : {type(ct)}")
    print(f"CT shape      : {ct.shape}")
    print(f"CT dtype      : {ct.dtype}")
    print(f"CT min        : {ct.min():.4f}")
    print(f"CT max        : {ct.max():.4f}")

    print()

    print(f"Mask type     : {type(mask)}")
    print(f"Mask shape    : {mask.shape}")
    print(f"Mask dtype    : {mask.dtype}")
    print(f"Mask min      : {mask.min():.4f}")
    print(f"Mask max      : {mask.max():.4f}")

    print()

    print(f"Unique labels : {mask.unique()}")