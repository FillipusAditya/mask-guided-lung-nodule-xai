"""Tests for normalized CT-volume PNG export."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from export_png import export_volume_file, normalized_to_uint8


class ExportPngTests(unittest.TestCase):
    def test_normalized_values_map_to_uint8(self) -> None:
        volume = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)

        result = normalized_to_uint8(volume)

        np.testing.assert_array_equal(
            result,
            np.array([[[0, 128, 255]]], dtype=np.uint8),
        )

    def test_every_axial_slice_is_exported(self) -> None:
        volume = np.linspace(0.0, 1.0, 3 * 4 * 5, dtype=np.float32).reshape(
            3,
            4,
            5,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "scan.npy"
            output_dir = root / "png" / "scan"
            np.save(input_path, volume)

            result = export_volume_file(input_path, output_dir)
            output_paths = sorted(output_dir.glob("slice_*.png"))

            self.assertEqual(result.status, "written")
            self.assertEqual(result.slice_count, 3)
            self.assertEqual(
                [path.name for path in output_paths],
                ["slice_0000.png", "slice_0001.png", "slice_0002.png"],
            )
            with Image.open(output_paths[0]) as image:
                self.assertEqual(image.mode, "L")
                self.assertEqual(image.size, (5, 4))

    def test_invalid_range_is_rejected(self) -> None:
        volume = np.array([[[0.0, 1.1]]], dtype=np.float32)

        with self.assertRaises(ValueError):
            normalized_to_uint8(volume)

    def test_complete_output_is_skipped_and_partial_output_is_rejected(self) -> None:
        volume = np.zeros((2, 3, 4), dtype=np.float32)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "scan.npy"
            output_dir = root / "png"
            np.save(input_path, volume)
            export_volume_file(input_path, output_dir)

            skipped = export_volume_file(input_path, output_dir)
            self.assertEqual(skipped.status, "skipped")

            (output_dir / "slice_0001.png").unlink()
            with self.assertRaises(FileExistsError):
                export_volume_file(input_path, output_dir)


if __name__ == "__main__":
    unittest.main()
