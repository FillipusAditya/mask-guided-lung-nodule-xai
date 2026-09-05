"""Unit tests for the modular LIDC-IDRI consensus pipeline."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from generate_consensus_masks import validate_input_metadata  # noqa: E402
from lidc_consensus import (  # noqa: E402
    cluster_uid,
    restore_consensus_slices,
    save_consensus_slices,
    scan_directory_name,
)


class FakeScan:
    """Minimal scan object exposing identifiers used for output naming."""

    patient_id = "LIDC-IDRI-0001"
    study_instance_uid = "1.2.3.30178"
    series_instance_uid = "1.2.3.03192"


class LIDCConsensusTests(unittest.TestCase):
    """Test metadata contracts, mask restoration, and safe export."""

    def test_identifiers_match_dataset_convention(self) -> None:
        self.assertEqual(
            cluster_uid("LIDC-IDRI-0001", 3),
            "LIDC-IDRI-0001_cluster_3",
        )
        self.assertEqual(
            scan_directory_name(FakeScan()),
            "LIDC-IDRI-0001_30178_03192",
        )

    def test_restore_omits_empty_slices(self) -> None:
        cropped = np.zeros((2, 3, 3), dtype=bool)
        cropped[:, :, 0] = True
        cropped[0, 1, 2] = True

        restored = restore_consensus_slices(
            cropped_mask=cropped,
            bounding_box=(slice(1, 3), slice(2, 5), slice(7, 10)),
            image_shape=(6, 8),
        )

        self.assertEqual(sorted(restored), [7, 9])
        self.assertEqual(restored[7].shape, (6, 8))
        self.assertEqual(restored[7].dtype, np.bool_)
        self.assertEqual(int(restored[7].sum()), 6)
        self.assertEqual(int(restored[9].sum()), 1)

    def test_export_protects_existing_slices(self) -> None:
        masks = {
            7: np.array([[0, 1], [1, 0]], dtype=np.uint8),
            9: np.ones((2, 2), dtype=bool),
        }

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            paths = save_consensus_slices(masks, output_dir)

            self.assertEqual(
                [path.name for path in paths],
                ["slice_7.npy", "slice_9.npy"],
            )
            self.assertEqual(np.load(paths[0]).dtype, np.bool_)

            with self.assertRaises(FileExistsError):
                save_consensus_slices(masks, output_dir)

            save_consensus_slices(masks, output_dir, overwrite=True)

    def test_metadata_validation_rejects_inconsistent_uid(self) -> None:
        metadata = pd.DataFrame(
            {
                "patient_id": ["LIDC-IDRI-0001"],
                "study_instance_uid": ["study"],
                "series_instance_uid": ["series"],
                "cluster_id": [2],
                "cluster_uid": ["LIDC-IDRI-0001_cluster_9"],
            }
        )

        with self.assertRaisesRegex(ValueError, "cluster_uid"):
            validate_input_metadata(metadata)


if __name__ == "__main__":
    unittest.main()
