"""Generate reproducible CT thumbnails for the segmentation flow diagram."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image

from config import HU_THRESHOLD, WINDOW_LEVEL, WINDOW_WIDTH
from step_2_segmentation import (
    build_candidate_volume,
    clean_reference,
    finalize_mask_volume,
    propagate_bidirectionally,
    select_reference_index,
    threshold_lung,
)
from step_3_median_filter import filter_slice
from step_4_normalize_and_mask import filter_normalize_and_mask


MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent
DEFAULT_PATIENT_DIR = (
    PROJECT_ROOT / "000_dataset" / "lidc_idri" / "LIDC-IDRI-0001"
)
DEFAULT_OUTPUT_DIR = MODULE_DIR / "pipeline_visuals"


def _largest_dicom_series(patient_dir: Path) -> Path:
    series = []
    for directory in patient_dir.rglob("*"):
        if directory.is_dir():
            paths = list(directory.glob("*.dcm"))
            if paths:
                series.append((len(paths), directory))
    if not series:
        raise FileNotFoundError(f"No DICOM series found below {patient_dir}")
    return max(series, key=lambda item: item[0])[1]


def _load_dicom_volume(series_dir: Path) -> np.ndarray:
    slices = []
    for path in series_dir.glob("*.dcm"):
        image = pydicom.dcmread(path)
        z_position = float(image.ImagePositionPatient[-1])
        instance_number = float(getattr(image, "InstanceNumber", 0))
        scaled = (
            image.pixel_array.astype(np.float32) * float(image.RescaleSlope)
            + float(image.RescaleIntercept)
        )
        slices.append((z_position, instance_number, scaled.astype(np.int16)))
    if not slices:
        raise FileNotFoundError(f"No DICOM slices found in {series_dir}")
    slices.sort(key=lambda item: (item[0], item[1]))
    return np.stack([item[2] for item in slices])


def _window_to_uint8(image: np.ndarray) -> np.ndarray:
    lower = WINDOW_LEVEL - WINDOW_WIDTH / 2.0
    upper = WINDOW_LEVEL + WINDOW_WIDTH / 2.0
    normalized = np.clip(image.astype(np.float32), lower, upper)
    normalized = (normalized - lower) / (upper - lower)
    return np.round(normalized * 255.0).astype(np.uint8)


def _gray_rgb(image: np.ndarray) -> np.ndarray:
    grayscale = _window_to_uint8(image)
    return np.repeat(grayscale[..., None], 3, axis=2)


def _overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float = 0.48,
) -> np.ndarray:
    output = _gray_rgb(image).astype(np.float32)
    foreground = mask.astype(bool)
    output[foreground] = (
        (1.0 - alpha) * output[foreground]
        + alpha * np.asarray(color, dtype=np.float32)
    )
    return np.clip(output, 0, 255).astype(np.uint8)


def _save(image: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path, optimize=True)


def generate_visuals(patient_dir: Path, output_dir: Path) -> None:
    series_dir = _largest_dicom_series(patient_dir)
    volume = _load_dicom_volume(series_dir)

    candidates = build_candidate_volume(volume)
    reference_index = select_reference_index(candidates)
    reference = clean_reference(candidates[reference_index])
    protected, _ = propagate_bidirectionally(
        candidates,
        reference_index,
        reference,
        in_place=False,
    )
    final = finalize_mask_volume(protected, in_place=False).astype(np.uint8)
    parenchyma = filter_normalize_and_mask(volume, final)

    propagation_index = min(reference_index + 8, len(volume) - 1)
    if not protected[propagation_index].any():
        propagation_index = reference_index

    original = volume[reference_index]
    threshold = threshold_lung(original)
    candidate = candidates[reference_index].astype(bool)
    filtered = filter_slice(original)
    filtered_rgb = _gray_rgb(filtered)
    parenchyma_rgb = np.repeat(
        np.round(parenchyma[reference_index] * 255.0)
        .astype(np.uint8)[..., None],
        3,
        axis=2,
    )

    _save(_gray_rgb(original), output_dir / "original_ct.png")
    _save(
        _overlay(original, threshold, (23, 162, 184), alpha=0.42),
        output_dir / "threshold_overlay.png",
    )
    _save(
        _overlay(original, candidate, (14, 137, 121)),
        output_dir / "candidate_overlay.png",
    )
    _save(
        _overlay(original, reference, (111, 78, 186)),
        output_dir / "clean_reference_overlay.png",
    )
    _save(
        _overlay(
            volume[propagation_index],
            protected[propagation_index],
            (111, 78, 186),
        ),
        output_dir / "propagated_overlay.png",
    )
    _save(
        _overlay(original, final[reference_index], (224, 126, 31)),
        output_dir / "final_mask_overlay.png",
    )
    _save(filtered_rgb, output_dir / "filtered_windowed_ct.png")
    _save(parenchyma_rgb, output_dir / "segmented_parenchyma.png")

    metadata = (
        f"Patient: {patient_dir.name}\n"
        f"Series: {series_dir.relative_to(patient_dir)}\n"
        f"Slices: {len(volume)}\n"
        f"Reference index: {reference_index}\n"
        f"Propagation example index: {propagation_index}\n"
        f"Threshold: {HU_THRESHOLD} HU\n"
    )
    (output_dir / "visualization_metadata.txt").write_text(metadata)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient-dir", type=Path, default=DEFAULT_PATIENT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    generate_visuals(args.patient_dir, args.output_dir)


if __name__ == "__main__":
    main()
