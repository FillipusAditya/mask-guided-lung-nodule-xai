"""Stable names for LIDC-IDRI scans and annotation clusters."""

from typing import Any


def cluster_uid(patient_id: str, cluster_index: int) -> str:
    """Return the metadata identifier for one annotation cluster."""
    if cluster_index < 0:
        raise ValueError("cluster_index must be non-negative.")
    return f"{patient_id}_cluster_{cluster_index}"


def scan_directory_name(scan: Any) -> str:
    """Return the output directory name used for a pylidc scan."""
    patient_id = str(scan.patient_id)
    study_suffix = str(scan.study_instance_uid)[-5:]
    series_suffix = str(scan.series_instance_uid)[-5:]
    return f"{patient_id}_{study_suffix}_{series_suffix}"

