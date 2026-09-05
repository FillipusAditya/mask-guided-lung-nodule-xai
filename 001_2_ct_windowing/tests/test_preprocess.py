"""Tests for median filtering and CT lung-window preprocessing."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from preprocess import (
    median_filter_ct,
    preprocess_ct,
    preprocess_ct_file,
    validate_median_filter_size,
    window_and_normalize_ct,
)


class PreprocessTests(unittest.TestCase):
    def test_window_matches_parenchyma_parameters(self) -> None:
        source = np.array([[[-2000, -1400, -600, 200, 500]]], dtype=np.int16)
        actual = window_and_normalize_ct(source, -600.0, 1600.0)
        expected = np.array([[[0.0, 0.0, 0.5, 1.0, 1.0]]], dtype=np.float32)
        np.testing.assert_allclose(actual, expected)
        self.assertEqual(actual.dtype, np.float32)

    def test_median_filter_is_slice_wise(self) -> None:
        source = np.zeros((2, 3, 3), dtype=np.int16)
        source[0, 1, 1] = 1000
        source[1, :, :] = 100
        actual = median_filter_ct(source, (1, 3, 3))
        self.assertTrue(np.all(actual[0] == 0))
        self.assertTrue(np.all(actual[1] == 100))

    def test_preprocess_filters_before_windowing(self) -> None:
        source = np.full((1, 3, 3), -600, dtype=np.int16)
        source[0, 1, 1] = 200
        actual = preprocess_ct(
            source,
            window_level=-600.0,
            window_width=1600.0,
            median_filter_size=(1, 3, 3),
        )
        np.testing.assert_allclose(actual, np.full(source.shape, 0.5))

    def test_file_processing_writes_metadata_and_skips_pair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.npy"
            output_path = root / "output" / "scan.npy"
            np.save(input_path, np.full((2, 3, 4), -600, dtype=np.int16))
            input_path.with_suffix(".json").write_text(
                json.dumps({"source_id": "scan", "spacing_xyz_mm": [1, 1, 2]})
            )

            first = preprocess_ct_file(
                input_path,
                output_path,
                window_level=-600.0,
                window_width=1600.0,
                median_filter_size=(1, 3, 3),
            )
            second = preprocess_ct_file(
                input_path,
                output_path,
                window_level=-600.0,
                window_width=1600.0,
                median_filter_size=(1, 3, 3),
            )
            output = np.load(output_path, allow_pickle=False)
            metadata = json.loads(output_path.with_suffix(".json").read_text())

            self.assertEqual(first.status, "written")
            self.assertEqual(second.status, "skipped")
            self.assertEqual(output.dtype, np.float32)
            self.assertEqual(metadata["spacing_xyz_mm"], [1, 1, 2])
            self.assertEqual(metadata["preprocessing"]["window_bounds_hu"], [-1400.0, 200.0])

    def test_cross_slice_kernel_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_median_filter_size((3, 3, 3))


if __name__ == "__main__":
    unittest.main()
