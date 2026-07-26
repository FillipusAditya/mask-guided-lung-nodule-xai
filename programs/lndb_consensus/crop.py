def crop_scan(scan):
    """
    Crop the CT volume and all binary masks using the consensus
    bounding box while preserving the crop origin.
    """

    bbox = scan["consensus_bbox"]

    zmin, zmax = bbox["zmin"], bbox["zmax"]
    ymin, ymax = bbox["ymin"], bbox["ymax"]
    xmin, xmax = bbox["xmin"], bbox["xmax"]

    # Crop CT volume
    ct_crop = scan["ct_volume"][
        zmin:zmax + 1,
        ymin:ymax + 1,
        xmin:xmax + 1,
    ]

    # Crop every radiologist mask
    radiologists = []

    for rad in scan["radiologists"]:

        mask_crop = rad["mask_nodule"][
            zmin:zmax + 1,
            ymin:ymax + 1,
            xmin:xmax + 1,
        ]

        radiologists.append({
            **rad,
            "mask_crop": mask_crop,
        })

    return {
        **scan,
        "ct_crop": ct_crop,

        # Save crop origin in the original CT coordinates
        "crop_origin": {
            "z": zmin,
            "y": ymin,
            "x": xmin,
        },

        "radiologists": radiologists,
    }