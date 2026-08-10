"""DataLoader factory for the lung nodule classification dataset."""

from pathlib import Path

from torch.utils.data import DataLoader

from .dataset import LungClassificationDataset


def create_dataloader(
    root_dir: str | Path,
    split: str,
    batch_size: int,
    transform=None,
    class_to_idx: dict[str, int] | None = None,
    shuffle: bool = False,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
) -> DataLoader:
    """Create a configured DataLoader for one classification split."""

    dataset = LungClassificationDataset(
        root_dir=root_dir,
        split=split,
        transform=transform,
        class_to_idx=class_to_idx,
    )

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
