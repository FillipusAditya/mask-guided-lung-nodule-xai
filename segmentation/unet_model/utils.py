import torch
import torchvision
from torch.utils.data import DataLoader

from dataset import LungDataset


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    """
    Save the model checkpoint to disk.
    """
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    """
    Load model weights from a checkpoint.
    """
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def create_dataloader(
    root_dir,
    split,
    batch_size,
    transform=None,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    drop_last=False,
):
    """
    Create a PyTorch DataLoader for a specific dataset split.
    """

    dataset = LungDataset(
        root_dir=root_dir,
        split=split,
        transform=transform,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    return dataloader


def check_accuracy(loader, model, device="cuda"):
    """
    Evaluate the model using Dice Score, IoU,
    Precision, Sensitivity, and Specificity.
    """

    eps = 1e-8

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    # Switch the model to evaluation mode
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            # Move inputs and masks to the selected device
            x = x.to(device)
            y = y.unsqueeze(1).to(device)

            # Generate binary predictions
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

            # Accumulate confusion matrix values
            tp += ((preds == 1) & (y == 1)).sum().item()
            fp += ((preds == 1) & (y == 0)).sum().item()
            tn += ((preds == 0) & (y == 0)).sum().item()
            fn += ((preds == 0) & (y == 1)).sum().item()

    # Compute evaluation metrics
    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    sensitivity = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)

    print(f"Dice Score : {dice:.4f}")
    print(f"IoU        : {iou:.4f}")
    print(f"Precision  : {precision:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")

    # Switch back to training mode
    model.train()

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def save_predictions_as_imgs(
    loader,
    model,
    folder="saved_images/",
    device="cuda",
):
    """
    Save predicted segmentation masks and
    ground-truth masks as image files.
    """

    # Switch the model to evaluation mode
    model.eval()

    for idx, (x, y) in enumerate(loader):
        # Move input images to the selected device
        x = x.to(device)

        # Generate binary predictions
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.3).float()

        # Save predicted masks
        torchvision.utils.save_image(
            preds,
            f"{folder}/pred_{idx}.png",
        )

        # Save ground-truth masks
        torchvision.utils.save_image(
            y.unsqueeze(1),
            f"{folder}/{idx}.png",
        )

    # Restore training mode
    model.train()