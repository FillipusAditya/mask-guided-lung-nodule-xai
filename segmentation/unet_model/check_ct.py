from pathlib import Path
import numpy as np

# Path ke folder CT
CT_DIR = Path("../../dataset/_lndb/008_consensus_v1_split/train/ct")

files = sorted(CT_DIR.glob("*.npy"))

print(f"Number of CT slices : {len(files)}")

shapes = []
dtypes = []
mins = []
maxs = []
means = []
stds = []

for file in files:
    image = np.load(file)

    shapes.append(image.shape)
    dtypes.append(image.dtype)

    mins.append(image.min())
    maxs.append(image.max())
    means.append(image.mean())
    stds.append(image.std())

print("\nDataset Summary")
print("------------------------------")
print(f"Unique shapes : {set(shapes)}")
print(f"Unique dtypes : {set(dtypes)}")

print("\nGlobal Statistics")
print("------------------------------")
print(f"Minimum value : {min(mins):.4f}")
print(f"Maximum value : {max(maxs):.4f}")
print(f"Mean value    : {np.mean(means):.4f}")
print(f"Std value     : {np.mean(stds):.4f}")

print("\nExample File")
print("------------------------------")
example = np.load(files[0])

print(f"Filename : {files[0].name}")
print(f"Shape    : {example.shape}")
print(f"Dtype    : {example.dtype}")
print(f"Min      : {example.min():.4f}")
print(f"Max      : {example.max():.4f}")
print(f"Mean     : {example.mean():.4f}")
print(f"Std      : {example.std():.4f}")