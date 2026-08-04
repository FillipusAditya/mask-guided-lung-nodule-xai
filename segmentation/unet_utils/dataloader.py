from pathlib import Path

from torch.utils.data import DataLoader

from .dataset import LungDataset


def create_dataloader(
    root_dir: str | Path,
    split: str,
    batch_size: int,
    transform=None,
    shuffle: bool = False,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """
    Create a PyTorch DataLoader for a dataset split.
    """

    dataset = LungDataset(
        root_dir=root_dir,
        split=split,
        transform=transform,
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    return dataloader