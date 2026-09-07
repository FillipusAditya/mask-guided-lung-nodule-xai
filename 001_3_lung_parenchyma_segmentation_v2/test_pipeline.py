"""Fast synthetic tests for the V2 preprocessing pipeline."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from step_2_segmentation import (
    clean_reference,
    finalize_mask_volume,
    has_bilateral_lung_components,
    match_candidate_components,
    propagate_bidirectionally,
    remove_wide_flat_table,
    remove_trachea,
    repair_boundary,
    select_reference_index,
)
from step_3_median_filter import filter_volume
from step_4_normalize_and_mask import filter_normalize_and_mask, normalize_and_mask
from quality_control import calculate_inclusion, save_study_overlay


class PipelineTests(unittest.TestCase):
    def test_wide_flat_lower_table_is_removed_without_removing_lungs(self):
        mask = np.zeros((512, 512), dtype=bool)
        mask[140:350, 100:220] = True
        mask[140:350, 290:410] = True
        mask[380:430, 40:470] = True
        cleaned = remove_wide_flat_table(mask)
        self.assertTrue(cleaned[200, 150])
        self.assertTrue(cleaned[200, 350])
        self.assertFalse(cleaned[400, 250])

    def test_reference_selection_prefers_bilateral_lungs(self):
        candidates = np.zeros((3, 128, 128), dtype=bool)
        candidates[1, 30:100, 12:48] = True
        candidates[1, 30:100, 80:116] = True
        reference_index = select_reference_index(candidates)
        reference = clean_reference(candidates[reference_index])
        self.assertEqual(reference_index, 1)
        self.assertTrue(has_bilateral_lung_components(reference))

    def test_component_matching_rejects_disconnected_artifact(self):
        candidate = np.zeros((64, 64), dtype=bool)
        candidate[20:45, 10:30] = True
        candidate[3:10, 52:60] = True
        support = np.zeros_like(candidate)
        support[18:47, 8:32] = True
        matched, seed = match_candidate_components(candidate, support)
        self.assertTrue(seed.any())
        self.assertTrue(matched[30, 20])
        self.assertFalse(matched[5, 55])

    def test_off_center_small_lung_is_not_removed_as_trachea(self):
        mask = np.zeros((512, 512), dtype=bool)
        mask[230:260, 315:345] = True
        cleaned = remove_trachea(mask)
        self.assertEqual(int(cleaned.sum()), int(mask.sum()))

    def test_boundary_repair_is_not_smaller(self):
        mask = np.zeros((96, 96), dtype=bool)
        mask[20:75, 20:75] = True
        mask[40:55, 60:75] = False
        repaired = repair_boundary(mask)
        self.assertGreaterEqual(int(repaired.sum()), int(mask.sum()))

    def test_propagation_recovers_after_an_empty_slice(self):
        candidates = np.zeros((5, 64, 64), dtype=np.uint8)
        lungs = np.zeros((64, 64), dtype=bool)
        lungs[18:48, 8:26] = True
        lungs[18:48, 38:56] = True
        candidates[2] = lungs
        candidates[0] = lungs
        protected, failed = propagate_bidirectionally(
            candidates,
            reference_index=2,
            reference_mask=lungs,
            in_place=True,
        )
        self.assertIs(protected, candidates)
        self.assertTrue(protected[0].any())
        self.assertFalse(protected[1].any())
        self.assertEqual(failed, [])

    def test_finalization_can_reuse_the_mask_buffer(self):
        protected = np.zeros((2, 64, 64), dtype=np.uint8)
        protected[:, 10:54, 8:56] = 1
        final = finalize_mask_volume(protected, in_place=True)
        self.assertIs(final, protected)
        self.assertEqual(final.dtype, np.uint8)

    def test_median_normalize_and_mask_contract(self):
        volume = np.full((3, 32, 32), -600, dtype=np.int16)
        volume[:, 10, 10] = 200
        mask = np.zeros_like(volume, dtype=np.uint8)
        mask[:, 4:28, 4:28] = 1
        result = normalize_and_mask(filter_volume(volume), mask)
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(np.allclose(result[:, 10, 10], 0.5))
        self.assertGreaterEqual(float(result.min()), 0.0)
        self.assertLessEqual(float(result.max()), 1.0)
        self.assertTrue(np.all(result[mask == 0] == 0.0))

    def test_slice_wise_preprocessing_matches_volume_preprocessing(self):
        rng = np.random.default_rng(7)
        volume = rng.integers(-1400, 300, size=(4, 32, 32), dtype=np.int16)
        mask = np.zeros_like(volume, dtype=np.uint8)
        mask[:, 4:28, 5:27] = 1
        expected = normalize_and_mask(filter_volume(volume), mask)
        actual = filter_normalize_and_mask(volume, mask)
        np.testing.assert_allclose(actual, expected)

    def test_nodule_qc_inclusion_and_single_png(self):
        lung = np.zeros((2, 32, 32), dtype=np.uint8)
        lung[:, 4:28, 4:28] = 1
        parenchyma = lung.astype(np.float32) * 0.5
        nodule = np.zeros((32, 32), dtype=bool)
        nodule[20:30, 20:30] = True
        findings = [{
            "finding_id": 7,
            "display_id": "Nodule 7",
            "class": "Malignant",
            "masks": {1: nodule},
        }]
        overall, per_slice = calculate_inclusion(lung, {1: nodule})
        self.assertEqual(overall, 64.0)
        self.assertEqual(per_slice[1], 64.0)
        with TemporaryDirectory() as directory:
            output = Path(directory) / "one_study.png"
            statistics = save_study_overlay(
                "Synthetic-Study", lung, parenchyma, findings, output
            )
            self.assertTrue(output.is_file())
            self.assertEqual(statistics[0]["class"], "Malignant")

    def test_nodule_qc_does_not_write_an_empty_png(self):
        lung = np.zeros((2, 32, 32), dtype=np.uint8)
        parenchyma = np.zeros_like(lung, dtype=np.float32)
        with TemporaryDirectory() as directory:
            output = Path(directory) / "must_not_exist.png"
            with self.assertRaisesRegex(ValueError, "consensus nodule"):
                save_study_overlay(
                    "No-Nodule-Study", lung, parenchyma, [], output
                )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
