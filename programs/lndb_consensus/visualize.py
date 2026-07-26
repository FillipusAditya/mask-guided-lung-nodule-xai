import math

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import BoundaryNorm
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

def show_ct_mask_overlay(
    ct_volume,
    mask_volume,
    slices,
    bbox=None,
    n_columns=5,
    figsize_per_subplot=3,
    mask_cmap="autumn",
    mask_alpha=0.7,
    vmin=0,
    vmax=None,
    save_path=None,
):
    """
    Display CT slices with a mask overlay and an optional bounding box.

    """

    # Exit if there are no slices to visualize.
    if len(slices) == 0:
        print("No slices to display.")
        return

    # Use the maximum mask value if not specified.
    if vmax is None:
        vmax = mask_volume.max()

    # Compute subplot layout.
    n_rows = math.ceil(len(slices) / n_columns)

    # Create figure.
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(
            n_columns * figsize_per_subplot,
            n_rows * figsize_per_subplot,
        ),
    )

    # Flatten axes for easier iteration.
    axes = np.atleast_1d(axes).flatten()

    # Check whether a valid bounding box is provided.
    draw_bbox = (
        bbox is not None
        and isinstance(bbox, dict)
        and all(
            key in bbox
            for key in (
                "xmin",
                "xmax",
                "ymin",
                "ymax",
                "zmin",
                "zmax",
            )
        )
    )

    # Display each slice.
    for ax, slice_idx in zip(axes, slices):

        ct_slice = ct_volume[slice_idx]
        mask_slice = mask_volume[slice_idx]

        # Display CT image.
        ax.imshow(
            ct_slice,
            cmap="gray",
        )

        # Overlay mask.
        ax.imshow(
            np.ma.masked_where(mask_slice == 0, mask_slice),
            cmap=mask_cmap,
            alpha=mask_alpha,
            interpolation="none",
            vmin=vmin,
            vmax=vmax,
        )

        # Draw bounding box if available.
        if draw_bbox and bbox["zmin"] <= slice_idx <= bbox["zmax"]:
            rect = Rectangle(
                (bbox["xmin"], bbox["ymin"]),
                bbox["xmax"] - bbox["xmin"],
                bbox["ymax"] - bbox["ymin"],
                linewidth=2,
                edgecolor="lime",
                facecolor="none",
            )
            ax.add_patch(rect)

        ax.set_title(f"Slice {slice_idx}")
        ax.axis("off")

    # Hide unused subplot cells.
    for ax in axes[len(slices):]:
        ax.axis("off")

    # Add colorbar for non-binary masks.
    if vmax > 1:
        cbar = fig.colorbar(
            axes[0].images[-1],
            ax=axes.tolist(),
            shrink=0.85,
            pad=0.02,
        )
        cbar.set_label("Agreement")

    # Adjust subplot spacing.
    plt.tight_layout(
        rect=[0, 0, 1, 0.95],
        w_pad=1.0,
        h_pad=1.5,
    )

    # Display figure.
    if save_path is None:
        plt.show()
    else:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


def show_consensus_bbox(scan, save_path=None):
    """
    Display one CT slice that is annotated by every radiologist together with
    each radiologist's bounding box and the consensus bounding box.
    """

    ct_volume = scan["ct_volume"]
    radiologists = scan["radiologists"]
    consensus_bbox = scan["consensus_bbox"]

    # Find slices annotated by every radiologist.
    common_slices = sorted(
        set(radiologists[0]["nodule_slices"]).intersection(
            *[set(rad["nodule_slices"]) for rad in radiologists[1:]]
        )
    )

    if len(common_slices) == 0:
        print("No common slice found.")
        return

    # Select the middle slice.
    slice_idx = common_slices[len(common_slices) // 2]

    colors = ["red", "lime", "cyan", "yellow", "magenta"]

    fig, ax = plt.subplots(figsize=(7, 7))

    # Display CT slice.
    ax.imshow(ct_volume[slice_idx], cmap="gray")

    legend_handles = []

    # Draw each radiologist's bounding box.
    for i, rad in enumerate(radiologists):

        bbox = rad.get("bbox")

        if bbox is None:
            continue

        rect = Rectangle(
            (bbox["xmin"], bbox["ymin"]),
            bbox["xmax"] - bbox["xmin"],
            bbox["ymax"] - bbox["ymin"],
            linewidth=2,
            edgecolor=colors[i],
            facecolor="none",
        )

        ax.add_patch(rect)

        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=colors[i],
                lw=2,
                label=f"Radiologist {rad['radid']}",
            )
        )

    # Draw consensus bounding box.
    if consensus_bbox is not None:

        rect = Rectangle(
            (consensus_bbox["xmin"], consensus_bbox["ymin"]),
            consensus_bbox["xmax"] - consensus_bbox["xmin"],
            consensus_bbox["ymax"] - consensus_bbox["ymin"],
            linewidth=3,
            edgecolor="white",
            linestyle="--",
            facecolor="none",
        )

        ax.add_patch(rect)

        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="white",
                lw=3,
                linestyle="--",
                label="Consensus",
            )
        )

    ax.set_title(f"Slice {slice_idx}")
    ax.axis("off")

    ax.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=10,
    )

    plt.tight_layout()
    
    # Display figure.
    if save_path is None:
        plt.show()
    else:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


def show_agreement_map(
    scan,
    slices=None,
    alpha=0.6,
    n_columns=5,
    figsize_per_subplot=3,
    save_path=None,
):
    """
    Display the cropped CT slices with the agreement map overlay.
    """

    ct_crop = scan["ct_crop"]
    agreement_map = scan["agreement_map"]

    num_radiologists = len(scan["radiologists"])

    # Original z-index of the cropped volume
    slice_offset = scan["crop_origin"]["z"]

    # Automatically select slices containing agreement voxels
    if slices is None:
        slices = np.where(
            np.any(agreement_map > 0, axis=(1, 2))
        )[0]

    if len(slices) == 0:
        print("No agreement voxels found.")
        return

    # Discrete colormap
    colors = [
        "#4DB6AC",  # Agreement = 1
        "#FFD54F",  # Agreement = 2
        "#D32F2F",  # Agreement = 3
        "#7B1FA2",  # Agreement = 4
    ]

    cmap = ListedColormap(colors[:num_radiologists])

    bounds = np.arange(0.5, num_radiologists + 1.5, 1)
    norm = BoundaryNorm(bounds, cmap.N)

    n_rows = math.ceil(len(slices) / n_columns)

    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(
            n_columns * figsize_per_subplot,
            n_rows * figsize_per_subplot,
        ),
    )

    axes = np.atleast_1d(axes).flatten()

    im = None

    for ax, z in zip(axes, slices):

        # CT crop
        ax.imshow(
            ct_crop[z],
            cmap="gray",
            interpolation="none",
        )

        # Agreement map
        im = ax.imshow(
            np.ma.masked_where(
                agreement_map[z] == 0,
                agreement_map[z],
            ),
            cmap=cmap,
            norm=norm,
            alpha=alpha,
            interpolation="none",
        )

        # Display original slice number
        ax.set_title(f"Slice {z + slice_offset}")
        ax.axis("off")

    # Hide unused axes
    for ax in axes[len(slices):]:
        ax.axis("off")

    fig.subplots_adjust(
        right=0.90,
        wspace=0.08,
        hspace=0.20,
    )

    cbar = fig.colorbar(
        im,
        ax=axes.tolist(),
        ticks=np.arange(1, num_radiologists + 1),
        fraction=0.035,
        pad=0.02,
    )

    cbar.set_label("Agreement Level")

    cbar.set_ticklabels(
        [
            f"{i} Radiologist" if i == 1 else f"{i} Radiologists"
            for i in range(1, num_radiologists + 1)
        ]
    )

    # Display figure.
    if save_path is None:
        plt.show()
    else:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


def show_restored_consensus_mask(
    scan,
    slices=None,
    alpha=0.5,
    n_columns=5,
    figsize_per_subplot=3,
    save_path=None
):
    """
    Display the restored consensus mask overlaid on the original CT volume.
    """

    ct_volume = scan["ct_volume"]
    consensus_mask = scan["consensus_mask_full"]

    # Automatically select slices containing consensus voxels
    if slices is None:
        slices = np.where(
            np.any(consensus_mask, axis=(1, 2))
        )[0]

    if len(slices) == 0:
        print("No consensus voxels found.")
        return

    n_rows = math.ceil(len(slices) / n_columns)

    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(
            n_columns * figsize_per_subplot,
            n_rows * figsize_per_subplot,
        ),
    )

    axes = np.atleast_1d(axes).flatten()

    for ax, z in zip(axes, slices):

        # Original CT
        ax.imshow(
            ct_volume[z],
            cmap="gray",
        )

        # Restored consensus mask
        ax.imshow(
            np.ma.masked_where(
                consensus_mask[z] == 0,
                consensus_mask[z],
            ),
            cmap="spring",
            alpha=alpha,
            interpolation="none",
            vmin=0,
            vmax=1,
        )

        ax.set_title(f"Slice {z}")
        ax.axis("off")

    # Hide unused axes
    for ax in axes[len(slices):]:
        ax.axis("off")

    fig.suptitle(
        "Restored Consensus Mask",
        fontsize=14,
    )

    fig.subplots_adjust(
        top=0.90,
        wspace=0.10,
        hspace=0.25,
    )

    # Display figure.
    if save_path is None:
        plt.show()
    else:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


def show_consensus_mask(
    scan,
    slices=None,
    alpha=0.5,
    n_columns=5,
    figsize_per_subplot=3,
    save_path=None
):
    """
    Display the cropped CT slices with the consensus mask overlay.
    """

    ct_crop = scan["ct_crop"]
    consensus_mask = scan["consensus_mask"]

    # Original z-index of the cropped volume
    slice_offset = scan["crop_origin"]["z"]

    # Automatically select slices containing consensus voxels
    if slices is None:
        slices = np.where(
            np.any(consensus_mask, axis=(1, 2))
        )[0]

    if len(slices) == 0:
        print("No consensus voxels found.")
        return

    n_rows = math.ceil(len(slices) / n_columns)

    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(
            n_columns * figsize_per_subplot,
            n_rows * figsize_per_subplot,
        ),
    )

    axes = np.atleast_1d(axes).flatten()

    for ax, z in zip(axes, slices):

        # CT crop
        ax.imshow(
            ct_crop[z],
            cmap="gray",
            interpolation="none",
        )

        # Consensus mask
        ax.imshow(
            np.ma.masked_where(
                consensus_mask[z] == 0,
                consensus_mask[z],
            ),
            cmap="spring",
            alpha=alpha,
            interpolation="none",
        )

        # Display original slice number
        ax.set_title(f"Slice {z + slice_offset}")
        ax.axis("off")

    # Hide unused axes
    for ax in axes[len(slices):]:
        ax.axis("off")

    fig.suptitle(
        "Consensus Mask",
        fontsize=14,
    )

    fig.subplots_adjust(
        top=0.90,
        wspace=0.10,
        hspace=0.25,
    )

    # Display figure.
    if save_path is None:
        plt.show()
    else:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


def show_restored_consensus_mask(
    scan,
    slices=None,
    alpha=0.5,
    n_columns=5,
    figsize_per_subplot=3,
    save_path=None
):
    """
    Display the restored consensus mask overlaid on the original CT volume.
    """

    ct_volume = scan["ct_volume"]
    consensus_mask = scan["consensus_mask_full"]

    # Automatically select slices containing consensus voxels
    if slices is None:
        slices = np.where(
            np.any(consensus_mask, axis=(1, 2))
        )[0]

    if len(slices) == 0:
        print("No consensus voxels found.")
        return

    n_rows = math.ceil(len(slices) / n_columns)

    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(
            n_columns * figsize_per_subplot,
            n_rows * figsize_per_subplot,
        ),
    )

    axes = np.atleast_1d(axes).flatten()

    for ax, z in zip(axes, slices):

        # Original CT
        ax.imshow(
            ct_volume[z],
            cmap="gray",
        )

        # Restored consensus mask
        ax.imshow(
            np.ma.masked_where(
                consensus_mask[z] == 0,
                consensus_mask[z],
            ),
            cmap="spring",
            alpha=alpha,
            interpolation="none",
            vmin=0,
            vmax=1,
        )

        ax.set_title(f"Slice {z}")
        ax.axis("off")

    # Hide unused axes
    for ax in axes[len(slices):]:
        ax.axis("off")

    fig.suptitle(
        "Restored Consensus Mask",
        fontsize=14,
    )

    fig.subplots_adjust(
        top=0.90,
        wspace=0.10,
        hspace=0.25,
    )

    # Display figure.
    if save_path is None:
        plt.show()
    else:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()