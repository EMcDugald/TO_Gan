import os
import numpy as np

data_root = "/home/u26/emcdugald/TO_Gan/TO_3D_data_scratch/data"

topo = np.load(os.path.join(data_root, "topologies.npy"), allow_pickle=True)
shapes = np.load(os.path.join(data_root, "shapes.npy"), allow_pickle=True)
bcs = np.load(os.path.join(data_root, "boundary_conditions.npy"), allow_pickle=True)
loads = np.load(os.path.join(data_root, "loads.npy"), allow_pickle=True)

print("topo:", type(topo), topo.shape, topo.dtype)
print("shapes:", type(shapes), shapes.shape, shapes.dtype)
print("bcs:", type(bcs), bcs.shape, bcs.dtype)
print("loads:", type(loads), loads.shape, loads.dtype)

for name, arr in [("shapes", shapes), ("bcs", bcs), ("loads", loads), ("topo", topo)]:
    print("\n---", name, "---")
    for i in [0, 1, 2]:
        x = arr[i]
        print(f"{name}[{i}] type={type(x)}")
        try:
            xa = np.asarray(x)
            print("  asarray shape:", xa.shape, "dtype:", xa.dtype)
            if xa.ndim == 0:
                print("  value:", xa)
            elif xa.ndim == 1:
                print("  first few:", xa[:10])
            elif xa.ndim == 2:
                print("  first rows:\n", xa[:3])
            elif xa.ndim == 3:
                print("  first slice shape:", xa[0].shape)
        except Exception as e:
            print("  asarray failed:", repr(e))

idx32 = [i for i, shp in enumerate(shapes) if tuple(np.asarray(shp).tolist()) == (32, 32, 32)]
print("\nnum 32^3 entries:", len(idx32))
print("first 10 indices:", idx32[:10])

for i in idx32[:3]:
    print(f"\n===== sample {i} =====")
    print("shape:", np.asarray(shapes[i]))
    bc = np.asarray(bcs[i])
    ld = np.asarray(loads[i])
    tp = np.asarray(topo[i])
    print("bc shape:", bc.shape, "dtype:", bc.dtype)
    print("bc first rows:", bc[:5])
    print("load shape:", ld.shape, "dtype:", ld.dtype)
    print("load contents:", ld)
    print("topo shape before reshape:", tp.shape, "dtype:", tp.dtype)
    print("topo min/max:", tp.min(), tp.max())

bc_col_counts = {}
for i in idx32[:50]:
    bc = np.asarray(bcs[i])
    if bc.ndim == 2:
        bc_col_counts[bc.shape[1]] = bc_col_counts.get(bc.shape[1], 0) + 1
print("\nBC column counts among first 50 shape-matched samples:", bc_col_counts)

load_shapes = {}
for i in idx32[:50]:
    ld = np.asarray(loads[i])
    load_shapes[ld.shape] = load_shapes.get(ld.shape, 0) + 1
print("Load shapes among first 50 shape-matched samples:", load_shapes)

# ---- added: does topo contain ONLY {0,1}? build_condition_vector / the
# x0-diffusion target scheme assumes strictly binary occupancy. ----
print("\n--- topo binariness check (first 200 entries) ---")
bad = []
for i in range(min(200, len(topo))):
    tp = np.asarray(topo[i])
    uniq = np.unique(tp)
    if not np.all(np.isin(uniq, [0, 1])):
        bad.append((i, uniq[:10]))
print(f"non-binary entries among first 200: {len(bad)}")
if bad:
    print("examples:", bad[:5])

# ---- added: is len(shapes)/len(topo) large enough to cover every "part"
# index referenced in summary.csv? (data-gen assumes summary.csv's "part"
# column indexes directly into these arrays -- this at least checks the
# array is long enough; it does NOT prove the part<->index correspondence
# is semantically correct, just that it's not out-of-bounds.) ----
print("\n--- array length vs. summary.csv part-index coverage ---")
summary_csv = os.path.join(data_root, "summary.csv")
if os.path.exists(summary_csv):
    import csv as csv_mod
    max_part = -1
    n_rows = 0
    with open(summary_csv, newline="") as f:
        lines = (line for line in f if not line.lstrip().startswith("#"))
        reader = csv_mod.DictReader(lines)
        for row in reader:
            n_rows += 1
            try:
                max_part = max(max_part, int(float(row["part"])))
            except (KeyError, ValueError):
                pass
    print(f"summary.csv: {n_rows} rows, max part index = {max_part}")
    print(f"topo/shapes/bcs/loads array length = {len(shapes)}")
    if max_part >= len(shapes):
        print("WARNING: summary.csv references part indices >= array length "
              "-- part<->array-index assumption is WRONG, or these arrays "
              "are a subset/different ordering than what summary.csv indexes.")
    else:
        print("OK: all summary.csv part indices are within array bounds "
              "(does not by itself prove correct correspondence).")
else:
    print(f"summary.csv not found at {summary_csv} -- skipping this check")