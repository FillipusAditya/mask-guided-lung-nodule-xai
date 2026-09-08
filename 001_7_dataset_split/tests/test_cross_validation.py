"""Tests for classification cross-validation helpers."""

from pathlib import Path
import sys
import unittest

import pandas as pd


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from dataset_split.cross_validation import (  # noqa: E402
    HOLDOUT_FOLD,
    HOLDOUT_ROLE,
    add_identifiers,
    add_slice_fold_assignments,
    build_nodule_metadata,
)


class CrossValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.metadata = pd.DataFrame(
            [
                {
                    "dataset": "LNDb",
                    "patient_id": "LNDb-0001",
                    "filename": "LNDb-0001_finding_1_slice_10.npy",
                    "label": "benign",
                    "split": "train",
                },
                {
                    "dataset": "LNDb",
                    "patient_id": "LNDb-0001",
                    "filename": "LNDb-0001_finding_1_slice_11.npy",
                    "label": "benign",
                    "split": "train",
                },
                {
                    "dataset": "LNDb",
                    "patient_id": "LNDb-0002",
                    "filename": "LNDb-0002_finding_2_slice_12.npy",
                    "label": "malignant",
                    "split": "test",
                },
            ]
        )

    def test_nodule_rows_are_collapsed_before_fold_assignment(self) -> None:
        identified = add_identifiers(self.metadata)
        nodules = build_nodule_metadata(identified)
        self.assertEqual(len(nodules), 1)
        self.assertEqual(nodules.iloc[0]["nodule_id"], "LNDb-0001_finding_1")

    def test_fold_is_mapped_to_slices_and_test_stays_holdout(self) -> None:
        identified = add_identifiers(self.metadata)
        nodules = build_nodule_metadata(identified)
        nodules["cv_fold"] = 2
        output = add_slice_fold_assignments(identified, nodules)
        development = output[output["cv_role"] == "development"]
        holdout = output[output["cv_role"] == HOLDOUT_ROLE]
        self.assertEqual(set(development["cv_fold"]), {2})
        self.assertEqual(set(holdout["cv_fold"]), {HOLDOUT_FOLD})


if __name__ == "__main__":
    unittest.main()
