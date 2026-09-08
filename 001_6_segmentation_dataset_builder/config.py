"""Source and output configuration for prepared segmentation datasets."""

from pathlib import Path

from segmentation_dataset import DatasetSpec


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "000_dataset"

LNDB_ROOT = DATASET_ROOT / "_lndb"
LIDC_ROOT = DATASET_ROOT / "_lidc"

LNDB_SPEC = DatasetSpec(
    slug="lndb",
    display_name="LNDb",
    ct_volume_dirs={
        "ct_windowed": LNDB_ROOT / "002_windowed_npy",
        "ct_parenchyma": LNDB_ROOT / "004_volume_parenchyma_npy",
    },
    consensus_mask_dir=LNDB_ROOT / "005_mask_consensus_npy",
    output_dir=LNDB_ROOT / "007_segmentation_dataset_npy",
    nodule_prefix="finding",
    identifier_column="finding_id",
    study_column="patient_id",
)

LIDC_SPEC = DatasetSpec(
    slug="lidc",
    display_name="LIDC-IDRI",
    ct_volume_dirs={
        "ct_windowed": LIDC_ROOT / "002_windowed_npy",
        # The source directory currently uses this legacy spelling on disk.
        "ct_parenchyma": LIDC_ROOT / "004_volume_parencyma_npy",
    },
    consensus_mask_dir=LIDC_ROOT / "005_mask_consensus_npy",
    output_dir=LIDC_ROOT / "007_segmentation_dataset_npy",
    nodule_prefix="cluster",
    identifier_column="cluster_id",
    study_column="scan_id",
)

DATASET_SPECS = {
    LNDB_SPEC.slug: LNDB_SPEC,
    LIDC_SPEC.slug: LIDC_SPEC,
}
