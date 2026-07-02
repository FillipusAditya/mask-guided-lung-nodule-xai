import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm


class NPYVisualizer:

    def __init__(
        self,
        cols=8,
        max_slices_per_figure=64,
        figsize_scale=2,
        cmap="gray",
        dpi=300,
        interpolation=None,
    ):
        self.cols = cols
        self.max_slices_per_figure = max_slices_per_figure
        self.figsize_scale = figsize_scale
        self.cmap = cmap
        self.dpi = dpi
        self.interpolation = interpolation

    def load_volume(
        self,
        npy_path: Path
    ):
        # Load volume from .npy file
        return np.load(npy_path)

    def _save_chunk(
        self,
        volume,
        start_slice,
        end_slice,
        output_path,
    ):
        # Extract slice range
        chunk = volume[start_slice:end_slice]

        num_slices = len(chunk)

        rows = math.ceil(
            num_slices / self.cols
        )

        # Create grid figure
        fig, axes = plt.subplots(
            rows,
            self.cols,
            figsize=(
                self.cols * self.figsize_scale,
                rows * self.figsize_scale
            )
        )

        if rows == 1 and self.cols == 1:
            axes = [axes]
        else:
            axes = np.array(axes).flatten()

        # Plot each slice
        for local_idx in range(num_slices):

            global_idx = (
                start_slice + local_idx
            )

            axes[local_idx].imshow(
                chunk[local_idx],
                cmap=self.cmap,
                interpolation=self.interpolation,
            )

            axes[local_idx].set_title(
                f"Slice {global_idx}",
                fontsize=8,
            )

            axes[local_idx].axis("off")

        # Hide unused axes
        for i in range(
            num_slices,
            len(axes)
        ):
            axes[i].axis("off")

        plt.tight_layout()

        # Create output directory
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        # Save figure
        plt.savefig(
            output_path,
            dpi=self.dpi,
            bbox_inches="tight",
        )

        plt.close()

    def process_file(
        self,
        npy_path,
        output_dir,
    ):
        npy_path = Path(npy_path)
        output_dir = Path(output_dir)

        # Load volume
        volume = self.load_volume(
            npy_path
        )

        total_slices = volume.shape[0]

        # Calculate number of figures
        num_parts = math.ceil(
            total_slices /
            self.max_slices_per_figure
        )

        for part_idx in range(num_parts):

            start_slice = (
                part_idx *
                self.max_slices_per_figure
            )

            end_slice = min(
                start_slice +
                self.max_slices_per_figure,
                total_slices
            )

            output_path = (
                output_dir /
                f"{npy_path.stem}_part{part_idx + 1}.png"
            )

            self._save_chunk(
                volume,
                start_slice,
                end_slice,
                output_path,
            )

    def process_folder(
        self,
        input_dir,
        output_dir,
    ):
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        # Find all .npy files
        npy_files = sorted(
            input_dir.glob("*.npy")
        )

        # Process each volume
        for npy_file in tqdm(
            npy_files,
            desc="Processing volumes"
        ):
            self.process_file(
                npy_file,
                output_dir,
            )


visualizer = NPYVisualizer(
    cols=8,
    max_slices_per_figure=64,
)

# visualizer.process_file(
#     npy_path="volume/LNDb-0001.npy",
#     output_dir="visualization",
# )

visualizer.process_folder(
    input_dir="../../dataset/npy_files/lidc_idri_npy/volume",
    output_dir="../../dataset/scan_bulk_visualization/lidc_idri",
)