import os

import numpy as np
import torch
from torch.utils.data import Dataset


class LNDbDataset(Dataset):
    """
    Dataset class for loading LNDb images and
    corresponding segmentation masks.
    """

    def __init__(self, image_dir, mask_dir, transform=None):
        # Store dataset directories and transformations
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # Collect and sort image filenames
        self.images = sorted(os.listdir(image_dir))

    def __len__(self):
        """
        Return the total number of samples.
        """
        return len(self.images)

    def __getitem__(self, index):
        """
        Load an image-mask pair and apply
        optional preprocessing or augmentation.
        """

        # Retrieve the filename for the current sample
        filename = self.images[index]

        # Construct image and mask file paths
        image_path = os.path.join(self.image_dir, filename)
        mask_path = os.path.join(self.mask_dir, filename)

        # Load image and mask from NumPy files
        image = np.load(image_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.float32)

        # Convert the mask to a binary representation
        mask = (mask > 0).astype(np.float32)

        # Apply data augmentation if provided
        if self.transform is not None:
            augmentations = self.transform(
                image=image,
                mask=mask,
            )

            image = augmentations["image"]
            mask = augmentations["mask"]

        # Convert NumPy arrays to PyTorch tensors
        else:
            image = torch.from_numpy(image).unsqueeze(0)
            mask = torch.from_numpy(mask)

        return image, mask