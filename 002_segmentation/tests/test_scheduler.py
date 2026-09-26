"""Tests for the discrete validation-loss learning-rate scheduler."""

import sys
import unittest
from pathlib import Path

import torch

SEGMENTATION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SEGMENTATION_ROOT))

from unet_utils.scheduler import DiscreteReduceLROnPlateau  # noqa: E402


class DiscreteSchedulerTests(unittest.TestCase):
    def test_scheduler_only_uses_configured_levels(self) -> None:
        parameter = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.AdamW([parameter], lr=1e-3)
        levels = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6]
        scheduler = DiscreteReduceLROnPlateau(
            optimizer,
            learning_rates=levels,
            patience=0,
        )

        observed = [optimizer.param_groups[0]["lr"]]
        scheduler.step(1.0)
        for metric in range(2, 10):
            scheduler.step(float(metric))
            observed.append(optimizer.param_groups[0]["lr"])

        self.assertEqual(observed[:7], levels)
        self.assertTrue(all(rate == levels[-1] for rate in observed[6:]))

    def test_initial_optimizer_rate_must_match_first_level(self) -> None:
        parameter = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.AdamW([parameter], lr=5e-4)

        with self.assertRaisesRegex(ValueError, "optimizer learning rate"):
            DiscreteReduceLROnPlateau(
                optimizer,
                learning_rates=[1e-3, 5e-4, 1e-4],
            )

    def test_levels_must_be_strictly_decreasing(self) -> None:
        parameter = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.AdamW([parameter], lr=1e-3)

        with self.assertRaisesRegex(ValueError, "strictly decreasing"):
            DiscreteReduceLROnPlateau(
                optimizer,
                learning_rates=[1e-3, 5e-4, 5e-4],
            )


if __name__ == "__main__":
    unittest.main()
