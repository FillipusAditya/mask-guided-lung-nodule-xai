from pathlib import Path

import pandas as pd
import pylidc as pl
from pylidc.utils import consensus
from tqdm import tqdm

from collections import defaultdict


def checker():
    scans = pl.query(pl.Scan).all()

    patient_num_studies = defaultdict(int)
    patient_num_cluster_studies = defaultdict(int)

    for scan in scans:
        patient_num_studies[scan.patient_id] += 1

        if len(scan.cluster_annotations()) > 0:
            patient_num_cluster_studies[scan.patient_id] += 1

    missing_patients = [
        patient
        for patient in patient_num_studies
        if patient_num_cluster_studies[patient] == 0
    ]

    print(f"Patients with no clustered nodules: {len(missing_patients)}")


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
    # counter = 0

    for scan in tqdm(scans, desc="Extracting cluster metadata"):
        # counter += 1
        # if counter == 5:
        #     break

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

            label = ""
            status = ""
            num_annotations = len(cluster)
            avg_diameter = sum(ann.diameter for ann in cluster) / num_annotations
            avg_malignancy = sum(ann.malignancy for ann in cluster) / num_annotations
            _, cbbox, _ = consensus(cluster, clevel=0.5)

            if 1.0 <= avg_malignancy <= 2.5:
                label = 0  # benign
            elif 3.5 <= avg_malignancy <= 5.0:
                label = 1  # malignant
            else:
                label = 2  # ambiguous

            if (
                num_annotations >= 3
                and 3 <= avg_diameter <= 30
                and label != 2
            ):
                status = 1  # included
            else:
                status = 0  # excluded


            cluster_records.append(
                {
                    "patient_id": patient_id,
                    "study_instance_uid": study_uid,
                    "cluster_id": cluster_idx,
                    "annotation_ids": [ann.id for ann in cluster],
                    "num_annotations": num_annotations,
                    "avg_diameter_mm": avg_diameter,
                    "avg_malignancy": avg_malignancy,
                    "consensus_slice": list(range(cbbox[2].start, cbbox[2].stop)),
                    "label": label, 
                    "status": status
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
    save_cluster_metadata("../../dataset/metadata/lidc_idri/")
    
