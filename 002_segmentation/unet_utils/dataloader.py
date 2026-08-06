"""DataLoader factory for the lung nodule segmentation dataset."""

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
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
) -> DataLoader:
    """
    Create a PyTorch DataLoader for a dataset split.

    Parameters
    ----------
    root_dir : str or Path
        Root directory of the segmentation dataset.
    split : str
        Dataset split passed to :class:`LungDataset`.
    batch_size : int
        Number of samples in each batch.
    transform : callable, optional
        Transform applied to each dataset sample.
    shuffle : bool, default=False
        Whether to shuffle samples each epoch.
    num_workers : int, default=4
        Number of worker processes used to load data.
    pin_memory : bool, default=True
        Whether to pin loaded tensors in memory.
    drop_last : bool, default=False
        Whether to discard the final incomplete batch.
    persistent_workers : bool, default=True
        Whether worker processes persist between epochs when workers are used.
    prefetch_factor : int, default=2
        Number of batches each worker preloads when workers are used.

    Returns
    -------
    DataLoader
        Configured data loader for the requested dataset split.
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
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )

    return dataloader
