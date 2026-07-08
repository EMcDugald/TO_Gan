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
print("num 32^3 entries:", len(idx32))
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

print("BC column counts among first 50 shape-matched samples:", bc_col_counts)


load_shapes = {}
for i in idx32[:50]:
    ld = np.asarray(loads[i])
    load_shapes[ld.shape] = load_shapes.get(ld.shape, 0) + 1

print("Load shapes among first 50 shape-matched samples:", load_shapes)