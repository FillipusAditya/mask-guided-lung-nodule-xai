from pathlib import Path

import numpy as np
import pandas as pd
import pylidc as pl
from pylidc.utils import consensus
from tqdm import tqdm


# Root directory of the project.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Metadata containing the selected nodule clusters
METADATA_CSV_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "_lidc"
    / "000_metadata"
    / "002_cluster_metadata_cleaned_v2.csv"
)

# Directory where consensus masks will be stored
OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "dataset"
    / "_lidc"
    / "007_consensus_nodules_npy_v2"
)

# Metadata with an additional column containing the consensus mask path
OUTPUT_METADATA_CSV_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "_lidc"
    / "000_metadata"
    / "003_cluster_metadata_cleaned_path_v2.csv"
)


def generate_consensus_masks(
    metadata_csv_path: Path,
    output_directory: Path,
    output_metadata_csv_path: Path,
) -> None:
    """
    Generate 50% consensus segmentation masks for the selected LIDC-IDRI
    nodule clusters.

    Workflow
    --------
    1. Load the cleaned cluster metadata.
    2. Iterate through every LIDC scan.
    3. Identify clusters listed in the metadata.
    4. Compute the 50% consensus mask using pylidc.
    5. Restore each mask to the original CT slice dimensions.
    6. Save every consensus slice as an individual NumPy file.
    7. Save updated metadata containing the output directory of each cluster.
    """

    # Load cluster metadata
    metadata_df = pd.read_csv(metadata_csv_path)

    # Use a set for fast membership lookup
    selected_cluster_uids = set(metadata_df["cluster_uid"])

    # Create a copy so the original metadata remains unchanged
    output_metadata_df = metadata_df.copy()

    # Column storing the directory containing the generated masks
    output_metadata_df["consensus_mask_path"] = None

    # Load every scan available in the LIDC database
    scans = pl.query(pl.Scan).all()

    for scan in tqdm(scans, desc="Generating consensus masks", unit="scan"):

        # Basic scan identifiers
        patient_id = scan.patient_id
        study_uid = scan.study_instance_uid[-5:]
        series_uid = scan.series_instance_uid[-5:]

        # Load the CT volume to obtain the original slice dimensions
        ct_volume = scan.to_volume()

        # Retrieve clustered annotations for this scan
        annotation_clusters = scan.cluster_annotations()

        # Process every annotation cluster
        for cluster_index, annotation_cluster in enumerate(annotation_clusters):

            cluster_uid = f"{patient_id}_cluster_{cluster_index}"

            # Skip clusters that are not part of the selected metadata
            if cluster_uid not in selected_cluster_uids:
                continue

            # Create the output directory for this cluster
            cluster_output_dir = (
                output_directory
                / f"{patient_id}_{study_uid}_{series_uid}"
                / f"cluster_{cluster_index}"
            )

            cluster_output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            # Record the output directory in the metadata
            output_metadata_df.loc[
                output_metadata_df["cluster_uid"] == cluster_uid,
                "consensus_mask_path",
            ] = str(cluster_output_dir)

            # Compute the 50% consensus segmentation mask
            # consensus_mask : Boolean mask inside the bounding box
            # consensus_bbox : Bounding box coordinates (y, x, z)
            consensus_mask, consensus_bbox, _ = consensus(
                annotation_cluster,
                clevel=0.5,
            )

            y_bbox, x_bbox, z_bbox = consensus_bbox

            # Save only slices containing at least one consensus voxel.
            for local_slice_index, original_slice_index in enumerate(
                range(z_bbox.start, z_bbox.stop)
            ):
                # Skip slices without consensus voxels
                if not np.any(consensus_mask[:, :, local_slice_index]):
                    continue

                # Create an empty mask with the same height and width as the
                # original CT slice.
                full_slice_mask = np.zeros(
                    ct_volume.shape[:2],
                    dtype=np.uint8,
                )

                # Insert the consensus mask back into its original image
                # coordinates using the bounding box.
                full_slice_mask[
                    y_bbox,
                    x_bbox,
                ] = consensus_mask[:, :, local_slice_index]

                # Save the reconstructed full-size mask.
                np.save(
                    cluster_output_dir / f"slice_{original_slice_index}.npy",
                    full_slice_mask,
                )

    # Save the updated metadata
    output_metadata_df.to_csv(
        output_metadata_csv_path,
        index=False,
    )


if __name__ == "__main__":
    generate_consensus_masks(
        metadata_csv_path=METADATA_CSV_PATH,
        output_directory=OUTPUT_DIRECTORY,
        output_metadata_csv_path=OUTPUT_METADATA_CSV_PATH,
    )
