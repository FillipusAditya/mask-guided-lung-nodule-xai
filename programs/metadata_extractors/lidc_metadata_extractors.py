import argparse
from pathlib import Path

import pandas as pd
import pylidc as pl
from tqdm import tqdm


def extract_cluster_metadata():
    """
    Extract metadata from PyLIDC nodule clustering results.

    Returns
    -------
    study_df : pd.DataFrame
        Metadata at study level.

    cluster_df : pd.DataFrame
        Metadata at nodule-cluster level.
    """

    scans = pl.query(pl.Scan).all()

    study_records = []
    cluster_records = []

    for scan in tqdm(scans, desc="Extracting cluster metadata"):

        patient_id = scan.patient_id
        study_uid = scan.study_instance_uid

        clusters = scan.cluster_annotations()

        study_records.append(
            {
                "patient_id": patient_id,
                "study_instance_uid": study_uid,
                "num_clusters": len(clusters),
            }
        )

        for cluster_idx, cluster in enumerate(clusters, start=1):

            num_annotations = len(cluster)

            cluster_records.append(
                {
                    "patient_id": patient_id,
                    "study_instance_uid": study_uid,
                    "cluster_id": cluster_idx,
                    "annotation_ids": [ann.id for ann in cluster],
                    "num_annotations": num_annotations,
                    "avg_diameter_mm": sum(ann.diameter for ann in cluster) / num_annotations,
                    "avg_malignancy": sum(ann.malignancy for ann in cluster) / num_annotations,
                }
            )

    study_df = pd.DataFrame(study_records)
    cluster_df = pd.DataFrame(cluster_records)

    return study_df, cluster_df


def save_cluster_metadata(output_dir="metadata",):
    """
    Extract metadata and save as CSV.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    study_df, cluster_df = extract_cluster_metadata()

    study_df.to_csv(
        output_dir / "study_cluster_metadata.csv", 
        index=False,
    )

    cluster_df.to_csv(
        output_dir / "cluster_metadata.csv", 
        index=False,
    )

    return study_df, cluster_df


if __name__ == "__main__":
    save_cluster_metadata()
    