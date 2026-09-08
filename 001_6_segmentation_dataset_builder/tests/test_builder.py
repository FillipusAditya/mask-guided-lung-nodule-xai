"""Tests for the shared prepared-segmentation-dataset builder."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from segmentation_dataset import (  # noqa: E402
    DatasetSpec,
    build_dataset,
    extract_nodule_id,
    extract_slice_index,
)
from segmentation_dataset.builder import create_build_plan  # noqa: E402


class BuilderTests(unittest.TestCase):
    """Exercise naming, preflight, paired export, and metadata generation."""

    def make_spec(self, root: Path) -> DatasetSpec:
        """Create a small LNDb-like dataset specification."""
        return DatasetSpec(
            slug="test",
            display_name="Synthetic",
            ct_volume_dirs={
                "ct_windowed": root / "source" / "windowed",
                "ct_parenchyma": root / "source" / "parenchyma",
            },
            consensus_mask_dir=root / "source" / "masks",
            output_dir=root / "output",
            nodule_prefix="finding",
            identifier_column="finding_id",
            study_column="patient_id",
        )

    def create_sources(self, spec: DatasetSpec) -> None:
        """Write one study, two CT representations, and one mask."""
        study_name = "LNDb-0001"
        for index, directory in enumerate(spec.ct_volume_dirs.values()):
            directory.mkdir(parents=True, exist_ok=True)
            volume = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
            np.save(directory / f"{study_name}.npy", volume + index)

        mask_dir = (
            spec.consensus_mask_dir
            / study_name
            / "finding_2"
        )
        mask_dir.mkdir(parents=True, exist_ok=True)
        mask = np.zeros((4, 5), dtype=np.uint8)
        mask[1:3, 2:4] = 1
        np.save(mask_dir / "slice_1.npy", mask)

    def test_build_dataset_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            spec = self.make_spec(Path(directory))
            self.create_sources(spec)

            summary = build_dataset(spec)

            self.assertEqual(summary.studies, 1)
            self.assertEqual(summary.samples, 1)
            self.assertEqual(summary.created, 1)
            row = summary.metadata.iloc[0]
            self.assertEqual(row["patient_id"], "LNDb-0001")
            self.assertEqual(row["finding_id"], 2)
            self.assertEqual(row["slice_index"], 1)
            self.assertEqual(row["mask_pixels"], 4)
            self.assertEqual(row["mask_path"], row["mask_path"].replace("\\", "/"))

            windowed = np.load(spec.output_dir / row["ct_windowed_path"])
            mask = np.load(spec.output_dir / row["mask_path"])
            self.assertEqual(windowed.shape, mask.shape)
            self.assertEqual(mask.dtype, np.bool_)

            with self.assertRaises(FileExistsError):
                build_dataset(spec)

            overwritten = build_dataset(spec, overwrite=True)
            self.assertEqual(overwritten.created, 1)

    def test_preflight_detects_missing_volume_before_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            spec = self.make_spec(Path(directory))
            self.create_sources(spec)
            missing = (
                spec.ct_volume_dirs["ct_parenchyma"]
                / "LNDb-0001.npy"
            )
            missing.unlink()

            with self.assertRaisesRegex(FileNotFoundError, "Missing CT volumes"):
                create_build_plan(spec)
            self.assertFalse(spec.output_dir.exists())

    def test_strict_name_parsing(self) -> None:
        self.assertEqual(extract_slice_index("slice_42.npy"), 42)
        self.assertEqual(extract_nodule_id("finding_7", "finding"), 7)
        with self.assertRaises(ValueError):
            extract_slice_index("mask_42.npy")
        with self.assertRaises(ValueError):
            extract_nodule_id("cluster_7", "finding")


if __name__ == "__main__":
    unittest.main()
