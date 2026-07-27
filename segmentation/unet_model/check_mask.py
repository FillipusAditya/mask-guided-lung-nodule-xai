from pathlib import Path
import numpy as np

# Path ke folder mask
MASK_DIR = Path("../../dataset/_lndb/008_consensus_v1_split/train/mask")

files = sorted(MASK_DIR.glob("*.npy"))

print(f"Number of mask files : {len(files)}")

total_pixels = 0
positive_pixels = 0

unique_shapes = set()
unique_dtypes = set()
unique_values = set()

empty_masks = []

for file in files:
    mask = np.load(file)

    unique_shapes.add(mask.shape)
    unique_dtypes.add(mask.dtype)

    values = np.unique(mask)
    unique_values.update(values.tolist())

    total_pixels += mask.size
    positive_pixels += (mask > 0).sum()

    if (mask > 0).sum() == 0:
        empty_masks.append(file.name)

print("\nDataset Summary")
print("------------------------------")
print(f"Unique shapes : {unique_shapes}")
print(f"Unique dtypes : {unique_dtypes}")
print(f"Unique values : {sorted(unique_values)}")

print("\nPixel Statistics")
print("------------------------------")
print(f"Total pixels      : {total_pixels:,}")
print(f"Positive pixels   : {positive_pixels:,}")
print(f"Negative pixels   : {total_pixels - positive_pixels:,}")

positive_ratio = positive_pixels / total_pixels

print(f"Positive ratio    : {positive_ratio:.8f}")
print(f"Positive percent  : {positive_ratio * 100:.6f}%")

print("\nMask Statistics")
print("------------------------------")
print(f"Empty masks : {len(empty_masks)}")

if len(empty_masks) > 0:
    print("\nFirst 10 empty masks")
    for name in empty_masks[:10]:
        print(name)