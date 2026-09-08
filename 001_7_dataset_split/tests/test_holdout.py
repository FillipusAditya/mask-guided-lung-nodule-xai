"""Tests for patient-level holdout helpers."""

from pathlib import Path
import sys
import unittest


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from dataset_split.holdout import (  # noqa: E402
    LIDC_DATASET_NAME,
    LNDB_DATASET_NAME,
    parse_sample_filename,
    split_patients,
)
from dataset_split.pipeline import normalize_dataset, output_paths  # noqa: E402


class HoldoutTests(unittest.TestCase):
    def test_parse_lndb_filename(self) -> None:
        sample = parse_sample_filename(
            "LNDb-0042_finding_3_slice_17.npy"
        )
        self.assertEqual(sample["dataset"], LNDB_DATASET_NAME)
        self.assertEqual(sample["patient_id"], "LNDb-0042")
        self.assertEqual(sample["label_key"], (42, 3, 17))

    def test_parse_lidc_filename(self) -> None:
        sample = parse_sample_filename(
            "LIDC-IDRI-0042_12345_67890_cluster_3_slice_17.npy"
        )
        self.assertEqual(sample["dataset"], LIDC_DATASET_NAME)
        self.assertEqual(
            sample["label_key"],
            ("LIDC-IDRI-0042", "12345", "67890", 3, 17),
        )

    def test_split_patients_is_reproducible_and_disjoint(self) -> None:
        patient_ids = [f"patient-{index:02d}" for index in range(20)]
        first = split_patients(patient_ids)
        second = split_patients(patient_ids)
        self.assertEqual(first, second)
        train_ids, val_ids, test_ids = first
        self.assertFalse(train_ids & val_ids)
        self.assertFalse(train_ids & test_ids)
        self.assertFalse(val_ids & test_ids)
        self.assertEqual(
            train_ids | val_ids | test_ids,
            set(patient_ids),
        )

    def test_dataset_aliases_and_numbered_outputs(self) -> None:
        self.assertEqual(normalize_dataset("LIDC"), LIDC_DATASET_NAME)
        self.assertEqual(normalize_dataset("lndb"), LNDB_DATASET_NAME)
        lidc_paths = output_paths("lidc", "/tmp/output")
        lndb_paths = output_paths("lndb", "/tmp/output")
        self.assertEqual(lidc_paths[1].name, "002_holdout_split_lidc.csv")
        self.assertEqual(lndb_paths[1].name, "003_holdout_split_lndb.csv")


if __name__ == "__main__":
    unittest.main()
