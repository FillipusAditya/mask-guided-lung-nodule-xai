"""Step 3: erode the outer boundary of a binary lung mask."""

import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion
from tqdm import tqdm

from config import EROSION_ITERATIONS


def erode_image(mask: np.ndarray) -> np.ndarray:
    """Erode one 2D binary mask."""
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D mask, received {mask.shape}.")

    return binary_erosion(
        mask.astype(bool),
        iterations=EROSION_ITERATIONS,
    ).astype(np.uint8)


def erode_volume(mask: np.ndarray) -> np.ndarray:
    """Erode every axial mask independently and preserve shape (N, H, W)."""
    if mask.ndim != 3:
        raise ValueError(f"Expected shape (N, H, W), received {mask.shape}.")

    eroded = np.empty(mask.shape, dtype=np.uint8)

    for slice_index, slice_mask in enumerate(mask):
        eroded[slice_index] = erode_image(slice_mask)

    return eroded


def process_file(input_path: str | Path, output_path: str | Path) -> None:
    """Erode one NumPy mask file."""
    mask = np.load(input_path, allow_pickle=False)

    if mask.ndim == 2:
        result = erode_image(mask)
    elif mask.ndim == 3:
        result = erode_volume(mask)
    else:
        raise ValueError(f"Expected a 2D or 3D mask, received {mask.shape}.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, result)


def process_batch(input_dir: str | Path, output_dir: str | Path) -> None:
    """Erode every NumPy mask in a directory."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    files = sorted(input_dir.glob("*.npy"))

    if not files:
        raise FileNotFoundError(f"No .npy files found in {input_dir}.")

    for input_path in tqdm(
        files,
        desc="Eroding lung masks",
        unit="volume",
        dynamic_ncols=True,
    ):
        process_file(input_path, output_dir / input_path.name)
    print(f"Mask erosion | Total: {len(files)} | Written: {len(files)}")


def main() -> None:
    """Run erosion for one NumPy file or a directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    single = subparsers.add_parser("single")
    single.add_argument("input_file", type=Path)
    single.add_argument("output_file", type=Path)

    batch = subparsers.add_parser("batch")
    batch.add_argument("input_dir", type=Path)
    batch.add_argument("output_dir", type=Path)

    args = parser.parse_args()

    if args.command == "single":
        process_file(args.input_file, args.output_file)
    else:
        process_batch(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
