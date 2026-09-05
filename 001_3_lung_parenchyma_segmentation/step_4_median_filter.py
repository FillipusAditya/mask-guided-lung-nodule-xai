"""Step 4: reduce CT noise with a 2D median filter per axial slice."""

import argparse
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from tqdm import tqdm

from config import MEDIAN_FILTER_SIZE


def filter_image(image: np.ndarray) -> np.ndarray:
    """Apply a 3 x 3 median filter to one 2D CT image."""
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D image, received {image.shape}.")

    return ndi.median_filter(image, size=MEDIAN_FILTER_SIZE[1:])


def filter_volume(volume: np.ndarray) -> np.ndarray:
    """Filter a volume without mixing information between axial slices."""
    if volume.ndim != 3:
        raise ValueError(f"Expected shape (N, H, W), received {volume.shape}.")

    return ndi.median_filter(volume, size=MEDIAN_FILTER_SIZE)


def process_file(input_path: str | Path, output_path: str | Path) -> None:
    """Median-filter one NumPy image or volume."""
    data = np.load(input_path, allow_pickle=False)

    if data.ndim == 2:
        result = filter_image(data)
    elif data.ndim == 3:
        result = filter_volume(data)
    else:
        raise ValueError(f"Expected a 2D or 3D array, received {data.shape}.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, result)


def process_batch(input_dir: str | Path, output_dir: str | Path) -> None:
    """Median-filter every NumPy file in a directory."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    files = sorted(input_dir.glob("*.npy"))

    if not files:
        raise FileNotFoundError(f"No .npy files found in {input_dir}.")

    for input_path in tqdm(
        files,
        desc="Filtering CT volumes",
        unit="volume",
        dynamic_ncols=True,
    ):
        process_file(input_path, output_dir / input_path.name)
    print(f"Median filter | Total: {len(files)} | Written: {len(files)}")


def main() -> None:
    """Run median filtering for one NumPy file or a directory."""
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
