"""Utilities for splitting and blending overlapping segmentation tiles."""

from dataclasses import dataclass
from math import ceil

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class TileLayout:
    """Spatial layout shared by tile extraction and reconstruction."""

    image_height: int
    image_width: int
    grid_size: int
    overlap: int
    tile_height: int
    tile_width: int
    stride_height: int
    stride_width: int
    covered_height: int
    covered_width: int

    @property
    def padding(self) -> tuple[int, int]:
        """Return right and bottom padding required by the layout."""

        return (
            self.covered_width - self.image_width,
            self.covered_height - self.image_height,
        )


def compute_tile_layout(
    image_size: tuple[int, int],
    grid_size: int = 4,
    overlap: int = 0,
) -> TileLayout:
    """Calculate a regular grid that covers every input pixel.

    Tile dimensions are rounded up when an image cannot be covered exactly.
    Any resulting excess is represented by right/bottom padding and is cropped
    after reconstruction.
    """

    image_height, image_width = (int(value) for value in image_size)
    grid_size = int(grid_size)
    overlap = int(overlap)

    if image_height <= 0 or image_width <= 0:
        raise ValueError("image dimensions must be positive.")
    if grid_size <= 0:
        raise ValueError("grid_size must be a positive integer.")
    if overlap < 0:
        raise ValueError("overlap cannot be negative.")
    if overlap >= min(image_height, image_width):
        raise ValueError("overlap must be smaller than both image dimensions.")

    tile_height = ceil(
        (image_height + (grid_size - 1) * overlap) / grid_size
    )
    tile_width = ceil((image_width + (grid_size - 1) * overlap) / grid_size)
    stride_height = tile_height - overlap
    stride_width = tile_width - overlap

    if stride_height <= 0 or stride_width <= 0:
        raise ValueError("overlap must be smaller than the calculated tile size.")

    covered_height = tile_height + (grid_size - 1) * stride_height
    covered_width = tile_width + (grid_size - 1) * stride_width

    return TileLayout(
        image_height=image_height,
        image_width=image_width,
        grid_size=grid_size,
        overlap=overlap,
        tile_height=tile_height,
        tile_width=tile_width,
        stride_height=stride_height,
        stride_width=stride_width,
        covered_height=covered_height,
        covered_width=covered_width,
    )


def split_into_tiles(
    tensor: Tensor,
    grid_size: int = 4,
    overlap: int = 0,
) -> Tensor:
    """Split ``[C, H, W]`` data into a complete overlapping tile grid.

    Tiles are returned in row-major order. When rounding is needed, the input
    is padded on the right and bottom so no source pixel is omitted.
    """

    if tensor.ndim != 3:
        raise ValueError(
            "split_into_tiles expects a [C, H, W] tensor, "
            f"but received shape {tuple(tensor.shape)}."
        )

    _, height, width = tensor.shape
    layout = compute_tile_layout(
        image_size=(height, width),
        grid_size=grid_size,
        overlap=overlap,
    )
    padding_right, padding_bottom = layout.padding
    if padding_right or padding_bottom:
        tensor = F.pad(tensor, (0, padding_right, 0, padding_bottom))

    tiles = tensor.unfold(
        1, layout.tile_height, layout.stride_height
    ).unfold(2, layout.tile_width, layout.stride_width)
    tiles = tiles.permute(1, 2, 0, 3, 4)

    return tiles.reshape(
        grid_size**2,
        tensor.size(0),
        layout.tile_height,
        layout.tile_width,
    ).contiguous()


def _blend_weights(
    height: int,
    width: int,
    blend_mode: str,
    reference: Tensor,
) -> Tensor:
    """Create a strictly positive two-dimensional tile blending window."""

    if blend_mode == "uniform":
        return reference.new_ones((1, height, width))
    if blend_mode != "hann":
        raise ValueError(
            f"Unsupported tile blend mode: {blend_mode!r}. "
            "Supported modes are: uniform, hann."
        )

    weight_dtype = (
        reference.dtype if reference.is_floating_point() else torch.float32
    )
    vertical = torch.hann_window(
        height,
        periodic=False,
        dtype=weight_dtype,
        device=reference.device,
    )
    horizontal = torch.hann_window(
        width,
        periodic=False,
        dtype=weight_dtype,
        device=reference.device,
    )
    # A pure Hann window is zero at its edge. A small floor ensures that the
    # outer image border remains covered where no neighbouring tile exists.
    return torch.outer(vertical, horizontal).clamp_min(1e-3).unsqueeze(0)


def merge_tiles(
    tiles: Tensor,
    grid_size: int = 4,
    overlap: int = 0,
    output_size: tuple[int, int] | None = None,
    blend_mode: str = "uniform",
) -> Tensor:
    """Reconstruct full images and normalize all overlapping contributions.

    ``blend_mode="uniform"`` averages predictions equally, while ``"hann"``
    favours tile centres to reduce boundary seams. ``output_size`` should be
    supplied to remove any right/bottom padding introduced during splitting.
    """

    if tiles.ndim != 5:
        raise ValueError(
            "merge_tiles expects a [B, T, C, H, W] tensor, "
            f"but received shape {tuple(tiles.shape)}."
        )

    batch_size, tile_count, channels, tile_height, tile_width = tiles.shape
    if tile_count != grid_size**2:
        raise ValueError(
            f"Expected {grid_size**2} tiles for a {grid_size}x{grid_size} grid, "
            f"but received {tile_count}."
        )
    if overlap < 0 or overlap >= min(tile_height, tile_width):
        raise ValueError("overlap must be non-negative and smaller than each tile.")

    stride_height = tile_height - overlap
    stride_width = tile_width - overlap
    covered_height = tile_height + (grid_size - 1) * stride_height
    covered_width = tile_width + (grid_size - 1) * stride_width

    if output_size is None:
        output_height, output_width = covered_height, covered_width
    else:
        output_height, output_width = (int(value) for value in output_size)
        if (
            not 0 < output_height <= covered_height
            or not 0 < output_width <= covered_width
        ):
            raise ValueError(
                "output_size must be positive and no larger than the tiled coverage "
                f"({covered_height}, {covered_width})."
            )
        expected = compute_tile_layout(output_size, grid_size, overlap)
        if (tile_height, tile_width) != (
            expected.tile_height,
            expected.tile_width,
        ):
            raise ValueError(
                "Tile dimensions do not match output_size, grid_size, and overlap: "
                f"received {(tile_height, tile_width)}, expected "
                f"{(expected.tile_height, expected.tile_width)}."
            )

    weights = _blend_weights(
        tile_height,
        tile_width,
        blend_mode=blend_mode,
        reference=tiles,
    )
    weighted_tiles = tiles * weights.view(1, 1, 1, tile_height, tile_width)

    # Fold performs differentiable overlap-add reconstruction.
    tile_columns = weighted_tiles.permute(0, 2, 3, 4, 1).reshape(
        batch_size,
        channels * tile_height * tile_width,
        tile_count,
    )
    images = F.fold(
        tile_columns,
        output_size=(covered_height, covered_width),
        kernel_size=(tile_height, tile_width),
        stride=(stride_height, stride_width),
    )

    weight_columns = weights.reshape(1, tile_height * tile_width, 1).expand(
        1, tile_height * tile_width, tile_count
    )
    weight_sum = F.fold(
        weight_columns,
        output_size=(covered_height, covered_width),
        kernel_size=(tile_height, tile_width),
        stride=(stride_height, stride_width),
    )
    images = images / weight_sum.clamp_min(torch.finfo(images.dtype).eps)

    return images[:, :, :output_height, :output_width].contiguous()
