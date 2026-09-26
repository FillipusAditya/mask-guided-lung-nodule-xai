"""Checkpoint compatibility tests for overlapping tile experiments."""

import sys
import tempfile
import unittest
from pathlib import Path

import torch

SEGMENTATION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SEGMENTATION_ROOT))

from unet_utils.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402


class CheckpointTilingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = torch.nn.Conv2d(1, 1, kernel_size=1)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        self.scaler = torch.amp.GradScaler("cuda", enabled=False)
        self.loss_config = {"name": "BCEDiceLoss", "smooth": 1e-6}
        self.scheduler_config = {"enabled": False}
        self.tiling_config = {
            "input_height": 512,
            "input_width": 512,
            "grid_size": 5,
            "overlap": 32,
            "blend_mode": "hann",
        }

    def test_tiling_configuration_is_saved_and_restored(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "checkpoint.pth"
            save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scaler=self.scaler,
                scheduler=None,
                loss_config=self.loss_config,
                scheduler_config=self.scheduler_config,
                tiling_config=self.tiling_config,
                epoch=3,
                best_val_loss=0.4,
                best_loss_epoch=3,
                best_val_dice=0.7,
                best_dice_epoch=2,
                epochs_without_improvement=0,
                save_path=checkpoint_path,
            )

            progress = load_checkpoint(
                checkpoint_path=checkpoint_path,
                model=self.model,
                optimizer=self.optimizer,
                scaler=self.scaler,
                scheduler=None,
                expected_loss_config=self.loss_config,
                expected_scheduler_config=self.scheduler_config,
                expected_tiling_config=self.tiling_config,
            )

            self.assertEqual(progress, (3, 0.4, 3, 0.7, 2, 0))

    def test_resume_rejects_a_different_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "checkpoint.pth"
            save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scaler=self.scaler,
                scheduler=None,
                loss_config=self.loss_config,
                scheduler_config=self.scheduler_config,
                tiling_config=self.tiling_config,
                epoch=1,
                best_val_loss=0.5,
                best_loss_epoch=1,
                best_val_dice=0.6,
                best_dice_epoch=1,
                epochs_without_improvement=0,
                save_path=checkpoint_path,
            )
            incompatible = {**self.tiling_config, "overlap": 0}

            with self.assertRaisesRegex(ValueError, "Tiling configuration mismatch"):
                load_checkpoint(
                    checkpoint_path=checkpoint_path,
                    model=self.model,
                    optimizer=self.optimizer,
                    scaler=self.scaler,
                    scheduler=None,
                    expected_loss_config=self.loss_config,
                    expected_scheduler_config=self.scheduler_config,
                    expected_tiling_config=incompatible,
                )


if __name__ == "__main__":
    unittest.main()
