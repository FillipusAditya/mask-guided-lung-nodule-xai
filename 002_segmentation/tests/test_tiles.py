"""Tests for complete and differentiable overlapping tile reconstruction."""

import sys
import unittest
from pathlib import Path

import torch

SEGMENTATION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SEGMENTATION_ROOT))

from unet_utils.tiles import (  # noqa: E402
    compute_tile_layout,
    merge_tiles,
    split_into_tiles,
)


class TileUtilitiesTests(unittest.TestCase):
    def test_legacy_non_overlapping_round_trip(self) -> None:
        image = torch.arange(512 * 512, dtype=torch.float32).reshape(1, 512, 512)

        tiles = split_into_tiles(image, grid_size=4, overlap=0)
        reconstructed = merge_tiles(
            tiles.unsqueeze(0),
            grid_size=4,
            overlap=0,
            output_size=(512, 512),
        )

        self.assertEqual(tuple(tiles.shape), (16, 1, 128, 128))
        torch.testing.assert_close(reconstructed.squeeze(0), image)

    def test_recommended_overlap_has_exact_coverage(self) -> None:
        layout = compute_tile_layout((512, 512), grid_size=5, overlap=32)

        self.assertEqual((layout.tile_height, layout.tile_width), (128, 128))
        self.assertEqual((layout.stride_height, layout.stride_width), (96, 96))
        self.assertEqual((layout.covered_height, layout.covered_width), (512, 512))
        self.assertEqual(layout.padding, (0, 0))

    def test_overlapping_round_trip_for_both_blend_modes(self) -> None:
        image = torch.rand(1, 512, 512)
        tiles = split_into_tiles(image, grid_size=5, overlap=32).unsqueeze(0)

        self.assertEqual(tuple(tiles.shape), (1, 25, 1, 128, 128))
        for blend_mode in ("uniform", "hann"):
            with self.subTest(blend_mode=blend_mode):
                reconstructed = merge_tiles(
                    tiles,
                    grid_size=5,
                    overlap=32,
                    output_size=(512, 512),
                    blend_mode=blend_mode,
                )
                torch.testing.assert_close(reconstructed.squeeze(0), image)

    def test_padding_is_removed_without_losing_source_pixels(self) -> None:
        image = torch.rand(1, 511, 509)
        tiles = split_into_tiles(image, grid_size=5, overlap=32).unsqueeze(0)
        reconstructed = merge_tiles(
            tiles,
            grid_size=5,
            overlap=32,
            output_size=(511, 509),
            blend_mode="hann",
        )

        self.assertEqual(tuple(reconstructed.shape), (1, 1, 511, 509))
        torch.testing.assert_close(reconstructed.squeeze(0), image)

    def test_overlap_merge_preserves_gradients(self) -> None:
        tiles = torch.rand(2, 25, 1, 128, 128, requires_grad=True)
        reconstructed = merge_tiles(
            tiles,
            grid_size=5,
            overlap=32,
            output_size=(512, 512),
            blend_mode="hann",
        )

        reconstructed.mean().backward()

        self.assertIsNotNone(tiles.grad)
        self.assertTrue(torch.isfinite(tiles.grad).all().item())
        self.assertTrue((tiles.grad > 0).all().item())

    def test_unknown_blend_mode_is_rejected(self) -> None:
        tiles = torch.ones(1, 25, 1, 128, 128)
        with self.assertRaisesRegex(ValueError, "Unsupported tile blend mode"):
            merge_tiles(
                tiles,
                grid_size=5,
                overlap=32,
                output_size=(512, 512),
                blend_mode="invalid",
            )


if __name__ == "__main__":
    unittest.main()
