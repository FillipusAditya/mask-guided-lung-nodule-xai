"""Tests for validated CT conversion without touching project datasets."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from convert import (
    convert_lidc_scan,
    convert_lndb_file,
    save_volume_with_metadata,
    validate_volume,
)


class ConvertTests(unittest.TestCase):
    def test_lidc_transposes_and_writes_geometry(self) -> None:
        source = np.arange(3 * 4 * 2, dtype=np.int16).reshape(3, 4, 2)
        scan = SimpleNamespace(
            patient_id="LIDC-IDRI-0001",
            study_instance_uid="study12345",
            series_instance_uid="series67890",
            pixel_spacing=0.7,
            slice_spacing=2.5,
            slice_thickness=2.5,
            slice_zvals=np.array([10.0, 12.5]),
            to_volume=lambda: source,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "scan.npy"
            result = convert_lidc_scan(scan, output)
            actual = np.load(output, allow_pickle=False)
            metadata = json.loads(output.with_suffix(".json").read_text())

            np.testing.assert_array_equal(actual, np.transpose(source, (2, 0, 1)))
            self.assertEqual(actual.dtype, np.int16)
            self.assertEqual(metadata["shape_zyx"], [2, 3, 4])
            self.assertEqual(metadata["spacing_xyz_mm"], [0.7, 0.7, 2.5])
            self.assertEqual(result.status, "written")

    def test_complete_existing_pair_is_skipped(self) -> None:
        volume = np.zeros((2, 3, 4), dtype=np.int16)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "scan.npy"
            metadata = {"dataset": "test", "source_id": "scan"}
            save_volume_with_metadata(volume, output, metadata)
            result = save_volume_with_metadata(volume + 1, output, metadata)

            self.assertEqual(result.status, "skipped")
            self.assertTrue(np.all(np.load(output) == 0))

    def test_lndb_preserves_simpleitk_geometry(self) -> None:
        import SimpleITK as sitk

        source = np.arange(2 * 3 * 4, dtype=np.int16).reshape(2, 3, 4)
        image = sitk.GetImageFromArray(source)
        image.SetSpacing((0.6, 0.7, 1.5))
        image.SetOrigin((-10.0, 20.0, 30.0))

        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "LNDb-test.mhd"
            output_path = Path(directory) / "output" / "LNDb-test.npy"
            sitk.WriteImage(image, str(input_path))
            convert_lndb_file(input_path, output_path)

            actual = np.load(output_path, allow_pickle=False)
            metadata = json.loads(output_path.with_suffix(".json").read_text())
            np.testing.assert_array_equal(actual, source)
            self.assertEqual(metadata["spacing_xyz_mm"], [0.6, 0.7, 1.5])
            self.assertEqual(metadata["origin_xyz_mm"], [-10.0, 20.0, 30.0])

    def test_partial_existing_pair_is_rejected(self) -> None:
        volume = np.zeros((1, 2, 2), dtype=np.int16)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "scan.npy"
            np.save(output, volume)
            with self.assertRaises(FileExistsError):
                save_volume_with_metadata(
                    volume, output, {"dataset": "test", "source_id": "scan"}
                )

    def test_invalid_volume_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_volume(np.zeros((2, 2), dtype=np.int16))
        with self.assertRaises(ValueError):
            validate_volume(np.array([[[40_000]]], dtype=np.int32))


if __name__ == "__main__":
    unittest.main()
