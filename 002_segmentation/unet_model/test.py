from pathlib import Path
import json

import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from segmentation.unet_model.model import UNET
from segmentation.unet_utils import (
    BCEDiceLoss,
    create_dataloader,
    update_confusion_matrix,
    compute_segmentation_metrics,
)

# --------------------------------------------------
# PROJECT
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULT_DIR = (
    PROJECT_ROOT
    / "segmentation"
    / "unet_model"
    / "result_xxxxx"
)

BEST_MODEL_PATH = (
    RESULT_DIR
    / "best_model.pth"
)

DATASET_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "_segmentation_dataset"
)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

BATCH_SIZE = 2
NUM_WORKERS = 4
PIN_MEMORY = torch.cuda.is_available()

POS_WEIGHT = 20.0
BCE_WEIGHT = 0.5
DICE_WEIGHT = 0.5

PRED_THRESHOLD = 0.5

# --------------------------------------------------
# TEST
# --------------------------------------------------

def evaluate():

    transforms = A.Compose([
        A.Resize(512, 512),
        ToTensorV2(),
    ])

    test_loader = create_dataloader(
        root_dir=DATASET_ROOT,
        split="test",
        transform=transforms,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    model = UNET(
        in_channels=1,
        out_channels=1,
    ).to(DEVICE)

    model.load_state_dict(
        torch.load(
            BEST_MODEL_PATH,
            map_location=DEVICE,
        )
    )

    model.eval()

    pos_weight = torch.tensor(
        [POS_WEIGHT],
        device=DEVICE,
    )

    criterion = BCEDiceLoss(
        pos_weight=pos_weight,
        bce_weight=BCE_WEIGHT,
        dice_weight=DICE_WEIGHT,
    )

    running_loss = 0.0
    total_samples = 0

    tp = fp = tn = fn = 0

    with torch.no_grad():

        for images, masks in tqdm(
            test_loader,
            desc="Testing",
        ):

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            logits = model(images)

            loss = criterion(
                logits,
                masks,
            )

            running_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

            predictions = (
                torch.sigmoid(logits)
                > PRED_THRESHOLD
            ).float()

            _tp, _fp, _tn, _fn = update_confusion_matrix(
                predictions,
                masks,
            )

            tp += _tp
            fp += _fp
            tn += _tn
            fn += _fn

    metrics = compute_segmentation_metrics(
        true_positive=tp,
        false_positive=fp,
        true_negative=tn,
        false_negative=fn,
    )

    metrics["loss"] = (
        running_loss / total_samples
    )

    print()
    print("=" * 60)
    print("Test Results")
    print("=" * 60)
    print(f"Loss         : {metrics['loss']:.4f}")
    print(f"Dice Score   : {metrics['dice']:.4f}")
    print(f"IoU          : {metrics['iou']:.4f}")
    print(f"Precision    : {metrics['precision']:.4f}")
    print(f"Sensitivity  : {metrics['sensitivity']:.4f}")
    print(f"Specificity  : {metrics['specificity']:.4f}")

    with open(
        RESULT_DIR / "test_results.json",
        "w",
    ) as f:
        json.dump(
            metrics,
            f,
            indent=4,
        )


if __name__ == "__main__":
    evaluate()