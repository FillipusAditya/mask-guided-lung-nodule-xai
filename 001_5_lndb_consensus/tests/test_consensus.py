"""Unit tests for the LNDb consensus pipeline."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from generate_consensus_metadata import (  # noqa: E402
    consensus_metadata,
    validate_input_metadata,
)
from lndb_consensus.bbox import (  # noqa: E402
    compute_bounding_boxes,
    compute_consensus_bounding_box,
)
from lndb_consensus.consensus import (  # noqa: E402
    create_consensus_mask,
    restore_consensus_mask,
)
from lndb_consensus.export import save_consensus_slices  # noqa: E402


class ConsensusTests(unittest.TestCase):
    """Test deterministic array operations and output safeguards."""

    def test_consensus_threshold_uses_ceiling(self) -> None:
        agreement = np.array([[[1, 2, 3]]])
        scan = {
            "agreement_map": agreement,
            "radiologists": [{}, {}, {}],
        }

        result = create_consensus_mask(scan, clevel=0.5)

        self.assertEqual(result["consensus_threshold"], 2)
        np.testing.assert_array_equal(
            result["consensus_mask"],
            np.array([[[False, True, True]]]),
        )

    def test_consensus_bounding_box_encloses_all_masks(self) -> None:
        first = np.zeros((4, 6, 8), dtype=bool)
        second = np.zeros_like(first)
        first[1:3, 2:5, 3:6] = True
        second[0:2, 1:3, 5:8] = True
        scan = {
            "radiologists": [
                {"radid": 1, "mask_nodule": first},
                {"radid": 2, "mask_nodule": second},
            ]
        }

        result = compute_bounding_boxes(scan)
        result = compute_consensus_bounding_box(result)

        self.assertEqual(
            result["consensus_bbox"],
            {"xmin": 3, "xmax": 7, "ymin": 1, "ymax": 4, "zmin": 0, "zmax": 2},
        )

    def test_restore_and_export_consensus_slices_once(self) -> None:
        scan = {
            "ct_volume": np.zeros((4, 5, 6), dtype=np.int16),
            "consensus_bbox": {
                "xmin": 2,
                "xmax": 3,
                "ymin": 1,
                "ymax": 2,
                "zmin": 1,
                "zmax": 2,
            },
            "consensus_mask": np.ones((2, 2, 2), dtype=bool),
        }
        scan = restore_consensus_mask(scan)

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            save_consensus_slices(scan, output_dir)
            outputs = sorted(output_dir.glob("slice_*.npy"))

            self.assertEqual(
                [path.name for path in outputs],
                ["slice_1.npy", "slice_2.npy"],
            )
            np.testing.assert_array_equal(
                np.load(outputs[0]),
                scan["consensus_mask_full"][1],
            )

            with self.assertRaises(FileExistsError):
                save_consensus_slices(scan, output_dir)

            save_consensus_slices(scan, output_dir, overwrite=True)

    def test_metadata_contract_and_serialization(self) -> None:
        metadata = pd.DataFrame(
            {
                "lndbid": [1],
                "findingid": [2],
                "radid": ["1,2,3"],
                "radfindingid": ["1,1,1"],
                "label": ["Benign"],
            }
        )
        validate_input_metadata(metadata)

        record = consensus_metadata(
            {
                "consensus_bbox": {
                    "xmin": 10,
                    "xmax": 12,
                    "ymin": 20,
                    "ymax": 23,
                    "zmin": 4,
                    "zmax": 5,
                },
                "consensus_slices": np.array([4, 5]),
            }
        )

        self.assertEqual(record["bbox_width"], 3)
        self.assertEqual(record["bbox_height"], 4)
        self.assertEqual(record["bbox_depth"], 2)
        self.assertEqual(record["consensus_bbox_volume"], 24)
        self.assertEqual(record["consensus_slice_list"], "4,5")


if __name__ == "__main__":
    unittest.main()
