"""Reproducibility configuration for Python, NumPy, and PyTorch."""

import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set the random seed for reproducible experiments.

    This function configures the random number generators for Python, NumPy,
    and PyTorch. It also configures cuDNN for deterministic or performance
    mode.

    Parameters
    ----------
    seed : int
        Random seed value.
    deterministic : bool, default=True
        If true, configure cuDNN for deterministic, reproducible results. If
        false, enable cuDNN benchmarking for better performance.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
