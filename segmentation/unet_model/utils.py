import random
from pathlib import Path

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

from dataset import LungDataset


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


def set_seed(seed: int) -> None:
    """
    Set random seeds to improve experiment reproducibility.
    """

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_history(
    history: dict,
    output_path: str | Path,
) -> None:
    """
    Save the training history to a CSV file.
    """

    history_df = pd.DataFrame(history)

    history_df.to_csv(
        output_path,
        index=False,
    )