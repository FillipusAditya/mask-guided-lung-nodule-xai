import random

import numpy as np
import torch


#---------------------------------
# RANDOM SEED
#---------------------------------
def set_seed(
    seed: int,
    deterministic: bool = True,
) -> None:
    """
    Set the random seed for reproducible experiments.

    This function configures the random number generators for
    Python, NumPy, and PyTorch. It also configures cuDNN to
    operate in deterministic or performance mode.

    Args:
        seed (int):
            Random seed value.

        deterministic (bool):
            If True, configure cuDNN for deterministic and
            reproducible results. If False, enable cuDNN
            benchmarking for maximum performance.
    """

    # Python random module
    random.seed(seed)

    # NumPy random module
    np.random.seed(seed)

    # PyTorch CPU random generator
    torch.manual_seed(seed)

    # PyTorch CUDA random generator
    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Configure cuDNN behavior
    if deterministic:

        # Enable deterministic algorithms for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    else:

        # Enable cuDNN benchmarking for better performance
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True