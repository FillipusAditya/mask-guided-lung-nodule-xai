"""Utilities for splitting and merging segmentation tiles."""

from torch import Tensor


def split_into_tiles(tensor: Tensor, grid_size: int = 4) -> Tensor:
    """
    Split an image tensor into non-overlapping tiles.

    Tiles are arranged in row-major order, starting from the top-left tile
    and ending at the bottom-right tile.

    Parameters
    ----------
    tensor : Tensor
        Image or mask tensor with shape ``[C, H, W]``.
    grid_size : int, default=4
        Number of tile rows and columns. The total number of tiles is
        ``grid_size**2``.

    Returns
    -------
    Tensor
        Contiguous tile tensor with shape
        ``[grid_size**2, C, H_tile, W_tile]``, where
        ``H_tile = H // grid_size`` and ``W_tile = W // grid_size``.

    Notes
    -----
    The image height and width must be divisible by ``grid_size`` so that
    every pixel belongs to exactly one tile.
    """

    channels, height, width = tensor.shape
    tile_height = height // grid_size
    tile_width = width // grid_size

    tiles = tensor.unfold(1, tile_height, tile_height)
    tiles = tiles.unfold(2, tile_width, tile_width)
    tiles = tiles.permute(1, 2, 0, 3, 4)

    return tiles.reshape(
        grid_size**2,
        channels,
        tile_height,
        tile_width,
    ).contiguous()


def merge_tiles(tiles: Tensor, grid_size: int = 4) -> Tensor:
    """
    Reconstruct full images from non-overlapping tiles.

    The function expects the same row-major tile order produced by
    :func:`split_into_tiles` and places each tile back into its original
    spatial position.

    Parameters
    ----------
    tiles : Tensor
        Batched tile tensor with shape ``[B, T, C, H_tile, W_tile]``, where
        ``T`` must equal ``grid_size**2``.
    grid_size : int, default=4
        Number of tile rows and columns used to reconstruct each image.

    Returns
    -------
    Tensor
        Contiguous image tensor with shape
        ``[B, C, grid_size * H_tile, grid_size * W_tile]``.
    """

    batch_size, _, channels, tile_height, tile_width = tiles.shape

    images = tiles.reshape(
        batch_size,
        grid_size,
        grid_size,
        channels,
        tile_height,
        tile_width,
    )
    images = images.permute(0, 3, 1, 4, 2, 5)

    return images.reshape(
        batch_size,
        channels,
        grid_size * tile_height,
        grid_size * tile_width,
    ).contiguous()
