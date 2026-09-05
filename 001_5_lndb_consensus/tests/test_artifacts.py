"""Tests for LNDb segmented PNG and quality-control artifacts."""

import sys
import tempfile
import unittest
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image


matplotlib.use("Agg")

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from lndb_consensus.artifacts import (  # noqa: E402
    preprocess_ct_for_display,
    save_consensus_quality_control,
    save_segmented_nodule_png,
)
from lndb_consensus.visualize import _resolve_scan_title  # noqa: E402


class LNDbArtifactTests(unittest.TestCase):
    """Test preprocessing and the two optional image exports."""

    def test_preprocessing_returns_normalized_float32(self) -> None:
        volume = np.array([[[-1400, -600, 200]]], dtype=np.int16)

        result = preprocess_ct_for_display(
            volume,
            median_filter_size=(1, 1, 1),
        )

        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_allclose(result, [[[0.0, 0.5, 1.0]]])

    def test_visualization_title_is_derived_from_scan(self) -> None:
        title = _resolve_scan_title(
            {"lndb_id": 7, "finding_id": 3},
            title=None,
        )

        self.assertEqual(title, "LNDb-0007 | Finding 3")

    def test_segmented_png_and_quality_control_are_saved(self) -> None:
        ct_volume = np.zeros((2, 5, 6), dtype=np.int16)
        normalized_ct = np.full(ct_volume.shape, 0.5, dtype=np.float32)
        mask = np.zeros((2, 3, 4), dtype=bool)
        mask[0, 1, 2] = True
        scan = {
            "lndb_id": 1,
            "finding_id": 2,
            "ct_volume": ct_volume,
            "ct_crop": np.zeros((2, 3, 4), dtype=np.int16),
            "consensus_bbox": {
                "xmin": 1,
                "xmax": 4,
                "ymin": 1,
                "ymax": 3,
                "zmin": 0,
                "zmax": 1,
            },
            "crop_origin": {"x": 1, "y": 1, "z": 0},
            "consensus_mask": mask,
            "agreement_map": mask.astype(np.uint8) * 2,
            "radiologists": [{"radid": 1}, {"radid": 2}],
        }

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            png_result = save_segmented_nodule_png(
                scan,
                normalized_ct,
                output_dir / "segmented",
            )
            qc_result = save_consensus_quality_control(
                scan,
                normalized_ct,
                output_dir / "qc",
            )

            self.assertEqual(png_result.output_paths[0].name, "slice_0.png")
            image = np.asarray(Image.open(png_result.output_paths[0]))
            self.assertEqual(image.shape, (3, 4))
            self.assertEqual(int(np.count_nonzero(image)), 1)
            self.assertTrue(all(path.is_file() for path in qc_result.output_paths))


if __name__ == "__main__":
    unittest.main()
