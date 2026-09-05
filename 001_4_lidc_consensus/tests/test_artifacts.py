"""Tests for LIDC segmented PNG and quality-control artifacts."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
import numpy as np
from PIL import Image


matplotlib.use("Agg")

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from lidc_consensus.artifacts import (  # noqa: E402
    prepare_cluster_artifact,
    preprocess_ct_for_display,
    save_consensus_quality_control,
    save_segmented_nodule_png,
)


class LIDCArtifactTests(unittest.TestCase):
    """Test preprocessing, pylidc alignment, and image exports."""

    def test_preprocessing_returns_normalized_float32(self) -> None:
        volume = np.array([[[-1400, -600, 200]]], dtype=np.int16)

        result = preprocess_ct_for_display(
            volume,
            median_filter_size=(1, 1, 1),
        )

        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_allclose(result, [[[0.0, 0.5, 1.0]]])

    @patch("lidc_consensus.artifacts.enable_pylidc_numpy_compatibility")
    @patch("lidc_consensus.artifacts.consensus")
    def test_prepare_cluster_converts_pylidc_axis_order(
        self,
        mock_consensus,
        _mock_compatibility,
    ) -> None:
        reader_mask = np.zeros((2, 3, 2), dtype=bool)
        reader_mask[0, 1, 0] = True
        mock_consensus.return_value = (
            reader_mask.copy(),
            (slice(1, 3), slice(2, 5), slice(4, 6)),
            [reader_mask, reader_mask],
        )
        normalized_ct = np.zeros((8, 7, 9), dtype=np.float32)

        artifact = prepare_cluster_artifact([object(), object()], normalized_ct)

        self.assertEqual(artifact["ct_crop"].shape, (2, 2, 3))
        self.assertEqual(artifact["consensus_mask"].shape, (2, 2, 3))
        self.assertEqual(artifact["slice_offset"], 4)
        self.assertEqual(int(artifact["agreement_map"].max()), 2)

    def test_segmented_png_and_quality_control_are_saved(self) -> None:
        mask = np.zeros((2, 3, 4), dtype=bool)
        mask[0, 1, 2] = True
        artifact = {
            "ct_crop": np.full((2, 3, 4), 0.5, dtype=np.float32),
            "consensus_mask": mask,
            "agreement_map": mask.astype(np.uint8) * 2,
            "slice_offset": 10,
            "annotation_count": 2,
        }

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            png_result = save_segmented_nodule_png(
                artifact,
                output_dir / "segmented",
            )
            qc_result = save_consensus_quality_control(
                artifact,
                "LIDC-IDRI-0001 | Cluster 0",
                output_dir / "qc",
                cluster_id=0,
            )

            self.assertEqual(png_result.output_paths[0].name, "slice_10.png")
            image = np.asarray(Image.open(png_result.output_paths[0]))
            self.assertEqual(image.shape, (3, 4))
            self.assertEqual(int(np.count_nonzero(image)), 1)
            self.assertTrue(all(path.is_file() for path in qc_result.output_paths))


if __name__ == "__main__":
    unittest.main()
