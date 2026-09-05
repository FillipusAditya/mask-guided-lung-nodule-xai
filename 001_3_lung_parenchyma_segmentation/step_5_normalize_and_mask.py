"""Step 5: apply lung windowing, normalization, and the final lung mask."""

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from config import WINDOW_LEVEL, WINDOW_WIDTH


def window_and_normalize(data: np.ndarray) -> np.ndarray:
    """Map the lung window to float32 values in [0, 1]."""
    lower_bound = WINDOW_LEVEL - (WINDOW_WIDTH / 2.0)
    upper_bound = WINDOW_LEVEL + (WINDOW_WIDTH / 2.0)

    # In-place arithmetic limits memory use for large 3D volumes.
    normalized = data.astype(np.float32, copy=True)
    np.clip(normalized, lower_bound, upper_bound, out=normalized)
    normalized -= lower_bound
    normalized /= upper_bound - lower_bound
    return normalized


def normalize_and_mask(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Normalize CT data and set every value outside the mask to zero."""
    if data.ndim not in (2, 3):
        raise ValueError(f"Expected a 2D or 3D CT array, received {data.shape}.")

    if data.shape != mask.shape:
        raise ValueError(
            f"CT and mask shapes must match: {data.shape} != {mask.shape}."
        )

    normalized = window_and_normalize(data)
    normalized[~mask.astype(bool)] = np.float32(0.0)
    return normalized


def process_file(
    input_path: str | Path,
    mask_path: str | Path,
    output_path: str | Path,
) -> None:
    """Normalize and mask one NumPy CT file."""
    data = np.load(input_path, allow_pickle=False)
    mask = np.load(mask_path, allow_pickle=False)
    result = normalize_and_mask(data, mask)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, result)


def process_batch(
    input_dir: str | Path,
    mask_dir: str | Path,
    output_dir: str | Path,
) -> None:
    """Normalize and mask every NumPy CT file in a directory."""
    input_dir = Path(input_dir)
    mask_dir = Path(mask_dir)
    output_dir = Path(output_dir)
    files = sorted(input_dir.glob("*.npy"))

    if not files:
        raise FileNotFoundError(f"No .npy files found in {input_dir}.")

    for input_path in tqdm(
        files,
        desc="Normalizing lung parenchyma",
        unit="volume",
        dynamic_ncols=True,
    ):
        mask_path = mask_dir / input_path.name
        if not mask_path.is_file():
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        process_file(input_path, mask_path, output_dir / input_path.name)
    print(f"Normalize and mask | Total: {len(files)} | Written: {len(files)}")


def main() -> None:
    """Run the final step for one NumPy file or a directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    single = subparsers.add_parser("single")
    single.add_argument("input_file", type=Path)
    single.add_argument("mask_file", type=Path)
    single.add_argument("output_file", type=Path)

    batch = subparsers.add_parser("batch")
    batch.add_argument("input_dir", type=Path)
    batch.add_argument("mask_dir", type=Path)
    batch.add_argument("output_dir", type=Path)

    args = parser.parse_args()

    if args.command == "single":
        process_file(args.input_file, args.mask_file, args.output_file)
    else:
        process_batch(args.input_dir, args.mask_dir, args.output_dir)


if __name__ == "__main__":
    main()
