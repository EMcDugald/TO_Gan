import os
import csv
import numpy as np

data_root = "/home/u26/emcdugald/TO_Gan/TO_3D_data_scratch/data"

shapes = np.load(os.path.join(data_root, "shapes.npy"), allow_pickle=True)

target_shapes = [
    (32, 32, 32), (40, 40, 20), (60, 40, 20), (64, 32, 16),
    (80, 40, 15), (120, 20, 20), (120, 40, 10),
]

labeled_parts = set()
with open(os.path.join(data_root, "summary.csv"), newline="") as f:
    lines = (line for line in f if not line.lstrip().startswith("#"))
    reader = csv.DictReader(lines)
    for row in reader:
        raw = (row.get("deformation_p99") or "").strip()
        if raw == "" or raw.lower() == "nan":
            continue
        try:
            float(raw)
        except ValueError:
            continue
        labeled_parts.add(int(float(row["part"])))

print(f"Total labeled parts (valid deformation_p99): {len(labeled_parts)}\n")
print(f"{'shape':<16}{'total_in_universe':<20}{'labeled':<12}{'pct_labeled':<12}")
for shp in target_shapes:
    idxs = [i for i, s in enumerate(shapes) if tuple(np.asarray(s).tolist()) == shp]
    labeled_here = len(set(idxs) & labeled_parts)
    total_here = len(idxs)
    pct = 100.0 * labeled_here / total_here if total_here else 0.0
    print(f"{str(shp):<16}{total_here:<20}{labeled_here:<12}{pct:<12.1f}")
