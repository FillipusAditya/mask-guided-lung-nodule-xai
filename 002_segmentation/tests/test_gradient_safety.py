"""Tests for pre-optimizer gradient validation and clipping."""

import sys
import unittest
from pathlib import Path

import torch

SEGMENTATION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SEGMENTATION_ROOT))

from unet_utils.epoch import _clip_and_validate_gradients  # noqa: E402


class GradientSafetyTests(unittest.TestCase):
    def test_finite_gradients_are_clipped_by_global_l2_norm(self) -> None:
        model = torch.nn.Linear(2, 1, bias=False)
        model.weight.grad = torch.tensor([[3.0, 4.0]])

        original_norm = _clip_and_validate_gradients(
            model,
            max_norm=1.0,
            context="unit test",
        )

        self.assertAlmostEqual(float(original_norm), 5.0, places=6)
        self.assertAlmostEqual(float(model.weight.grad.norm()), 1.0, places=6)

    def test_nonfinite_gradient_is_rejected_before_optimizer_step(self) -> None:
        model = torch.nn.Linear(2, 1, bias=False)
        model.weight.grad = torch.tensor([[float("inf"), 1.0]])

        with self.assertRaisesRegex(FloatingPointError, "Non-finite gradient norm"):
            _clip_and_validate_gradients(
                model,
                max_norm=1.0,
                context="unit test",
            )

    def test_none_checks_gradients_without_clipping(self) -> None:
        model = torch.nn.Linear(2, 1, bias=False)
        model.weight.grad = torch.tensor([[3.0, 4.0]])

        gradient_norm = _clip_and_validate_gradients(
            model,
            max_norm=None,
            context="unit test",
        )

        self.assertAlmostEqual(float(gradient_norm), 5.0, places=6)
        torch.testing.assert_close(model.weight.grad, torch.tensor([[3.0, 4.0]]))


if __name__ == "__main__":
    unittest.main()
