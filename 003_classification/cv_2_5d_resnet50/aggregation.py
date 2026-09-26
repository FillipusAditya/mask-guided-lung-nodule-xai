"""Window-to-nodule probability aggregation."""

from __future__ import annotations

import pandas as pd


def aggregate_nodule_predictions(
    frame: pd.DataFrame,
    nodule_column: str,
    threshold: float,
) -> pd.DataFrame:
    """Average window probabilities and return one prediction per nodule."""

    required = {nodule_column, "label", "true_index", "probability_benign", "probability_malignant"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction columns are missing: {sorted(missing)}")
    records = []
    identity_columns = [
        name for name in ("dataset", "patient_id", "cv_group_id", nodule_column)
        if name in frame.columns
    ]
    for nodule_id, group in frame.groupby(nodule_column, sort=False):
        if group["label"].nunique() != 1 or group["true_index"].nunique() != 1:
            raise ValueError(f"Inconsistent target within nodule: {nodule_id}")
        record = {name: group.iloc[0][name] for name in identity_columns}
        p_malignant = float(group["probability_malignant"].mean())
        record.update(
            {
                "label": group.iloc[0]["label"],
                "true_index": int(group.iloc[0]["true_index"]),
                "num_windows": int(len(group)),
                "probability_benign": 1.0 - p_malignant,
                "probability_malignant": p_malignant,
                "predicted_index": int(p_malignant >= threshold),
                "predicted_class": "malignant" if p_malignant >= threshold else "benign",
            }
        )
        records.append(record)
    return pd.DataFrame(records)

