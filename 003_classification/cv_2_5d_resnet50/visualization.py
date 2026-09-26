"""Paged seven-panel XAI visualizations for 2.5D windows."""

from __future__ import annotations

from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .dataset import TwoPointFiveDLungDataset


def normalize_ct(array: np.ndarray) -> np.ndarray:
    finite = array[np.isfinite(array)]
    lower, upper = np.percentile(finite, (1.0, 99.0))
    if upper <= lower:
        return np.zeros_like(array, dtype=np.float32)
    return np.clip((array - lower) / (upper - lower), 0.0, 1.0)


def resize_2d(array: np.ndarray, shape: tuple[int, int], mode: str = "bilinear") -> np.ndarray:
    if array.shape == shape:
        return array
    options = {} if mode == "nearest" else {"align_corners": False}
    return F.interpolate(
        torch.from_numpy(array.astype(np.float32))[None, None], size=shape,
        mode=mode, **options,
    )[0, 0].numpy()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def save_visualizations(
    predictions: pd.DataFrame,
    dataset: TwoPointFiveDLungDataset,
    gradcam_dir: Path,
    lrp_dir: Path,
    output_dir: Path,
    dpi: int = 120,
    rows_per_page: int = 12,
) -> None:
    """Save one or more figures for every nodule."""

    output_dir.mkdir(parents=True, exist_ok=True)
    for nodule_id, group in predictions.groupby(dataset.nodule_column, sort=False):
        group = group.sort_values("slice_index")
        for page, start in enumerate(range(0, len(group), rows_per_page), start=1):
            chunk = group.iloc[start : start + rows_per_page]
            figure, axes = plt.subplots(
                len(chunk), 7, figsize=(21, max(3, 2.8 * len(chunk))), squeeze=False
            )
            for row_number, (_, row) in enumerate(chunk.iterrows()):
                sample_index = int(row["sample_index"])
                window = dataset.load_window(sample_index)
                filename = Path(str(row["filename"])).name
                gradcam = np.load(gradcam_dir / filename, allow_pickle=False)
                relevance = np.load(lrp_dir / filename, allow_pickle=False)
                shape = gradcam.shape
                displays = [resize_2d(normalize_ct(window[..., i]), shape) for i in range(3)]
                for channel in range(3):
                    axes[row_number, channel].imshow(displays[channel], cmap="gray")
                    axes[row_number, channel].set_title(
                        f"CT channel {channel}\nslice {dataset.windows[sample_index]['input_slice_indices'][channel]}"
                    )
                if "mask_path" in row and pd.notna(row["mask_path"]):
                    mask_path = dataset.resolve_path(str(row["mask_path"]))
                    if mask_path.is_file():
                        mask = np.load(mask_path, allow_pickle=False).astype(np.float32)
                        mask = resize_2d(mask, shape, mode="nearest")
                        if mask.max() > 0:
                            axes[row_number, 1].contour(mask > 0, levels=[0.5], colors="cyan", linewidths=1)
                axes[row_number, 3].imshow(displays[1], cmap="gray")
                axes[row_number, 3].imshow(gradcam, cmap="jet", alpha=0.45, vmin=0, vmax=1)
                axes[row_number, 3].set_title("Grad-CAM\ncentral slice")
                for channel in range(3):
                    axis = axes[row_number, 4 + channel]
                    axis.imshow(displays[channel], cmap="gray")
                    axis.imshow(relevance[channel], cmap="seismic", alpha=0.5, vmin=-1, vmax=1)
                    axis.set_title(f"LRP channel {channel}")
                for axis in axes[row_number]:
                    axis.axis("off")
                axes[row_number, 0].set_ylabel(
                    f"z={int(row['slice_index'])}\ntrue={row['label']}\npred={row['predicted_class']}",
                    fontsize=9,
                )
            figure.suptitle(f"2.5D ResNet-50 XAI — {nodule_id} — page {page}")
            figure.tight_layout(rect=(0, 0, 1, 0.98))
            figure.savefig(output_dir / f"{safe_name(str(nodule_id))}_page_{page:02d}.png", dpi=dpi)
            plt.close(figure)

