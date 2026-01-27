import os
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# -----------------------------
# Config
# -----------------------------

DATA_ROOT = "/home/u26/emcdugald/TO_Gan/TO_3D_data_scratch/data"
OUTPUT_ROOT = "/home/u26/emcdugald/TO_Gan/TO_3D_data_scratch/bc_load_catalog_323232"

TARGET_SHAPE = (32, 32, 32)
LOAD_ROUND_DECIMALS = 2      # how coarsely to bin loads

MAX_TOTAL_PLOTS = 100        # total images to generate
N_REP_PER_BIN = 1            # images per selected bin (1 so MAX_TOTAL_PLOTS == number of bins)

os.makedirs(OUTPUT_ROOT, exist_ok=True)

# -----------------------------
# Helper functions
# -----------------------------

def plot_voxel(binary_arr, save_path, title=""):
    """Simple 3D voxel plot."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(binary_arr, edgecolor='k', linewidth=0.2)
    ax.set_title(title)
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.close(fig)

def bc_to_key(bc_arr):
    """
    Convert a (N_bc, 6) BC array to an order-invariant hashable key.
    Sort rows, round, then turn each row into a tuple.
    """
    bc_np = np.asarray(bc_arr, dtype=float)
    # sort rows lexicographically
    bc_sorted = bc_np[np.lexsort(bc_np.T[::-1])]
    # round to reduce tiny float differences
    bc_rounded = np.round(bc_sorted, 4)
    return tuple(map(tuple, bc_rounded))

def load_to_key(load_arr, decimals=2):
    """
    Convert (1, 6) load array to a rounded tuple key.
    """
    load_np = np.asarray(load_arr, dtype=float).reshape(-1)
    load_rounded = np.round(load_np, decimals=decimals)
    return tuple(load_rounded)

# -----------------------------
# Main
# -----------------------------

def main():
    # Load arrays
    topo = np.load(os.path.join(DATA_ROOT, "topologies.npy"), allow_pickle=True)
    shapes = np.load(os.path.join(DATA_ROOT, "shapes.npy"), allow_pickle=True)
    bcs = np.load(os.path.join(DATA_ROOT, "boundary_conditions.npy"), allow_pickle=True)
    loads = np.load(os.path.join(DATA_ROOT, "loads.npy"), allow_pickle=True)

    print("Loaded:")
    print("  topo.shape:", topo.shape)
    print("  shapes.shape:", shapes.shape)
    print("  bcs.shape:", bcs.shape)
    print("  loads.shape:", loads.shape)

    # Restrict to 32x32x32
    indices_323232 = [
        i for i, shp in enumerate(shapes)
        if tuple(shp) == TARGET_SHAPE
    ]
    print(f"Found {len(indices_323232)} entries with shape {TARGET_SHAPE}")

    # Group indices by (BC key, load key)
    groups = {}
    for i in indices_323232:
        bc_key = bc_to_key(bcs[i])
        load_key = load_to_key(loads[i], decimals=LOAD_ROUND_DECIMALS)
        key = (bc_key, load_key)
        if key not in groups:
            groups[key] = []
        groups[key].append(i)

    print(f"Formed {len(groups)} (BC, load) bins")

    # -------------------------
    # Limit total number of plotted structures
    # -------------------------
    all_keys = list(groups.keys())
    if len(all_keys) <= MAX_TOTAL_PLOTS:
        selected_keys = all_keys
    else:
        selected_keys = random.sample(all_keys, MAX_TOTAL_PLOTS)

    print(f"Will plot at most {len(selected_keys) * N_REP_PER_BIN} structures "
          f"from {len(selected_keys)} randomly selected bins")

    # For each selected group, create directory and save voxel plots
    for (bc_key, load_key) in selected_keys:
        idx_list = groups[(bc_key, load_key)]

        # Compact folder naming: hash for BC, rounded numbers for load
        load_str = "_".join(f"{x:.{LOAD_ROUND_DECIMALS}f}" for x in load_key)
        bc_hash = hash(bc_key) & 0xffffffff  # 32-bit hash for shorter name

        group_dir = os.path.join(
            OUTPUT_ROOT,
            f"bc{bc_hash}_load_{load_str}"
        )
        os.makedirs(group_dir, exist_ok=True)

        # Meta file with human-readable BC and load
        meta_path = os.path.join(group_dir, "meta.txt")
        if not os.path.exists(meta_path):
            with open(meta_path, "w") as f:
                f.write("BC (sorted, rounded):\n")
                for row in bc_key:
                    f.write("  " + " ".join(f"{x:.4f}" for x in row) + "\n")
                f.write("\nLoad (rounded):\n")
                f.write("  " + " ".join(f"{x:.{LOAD_ROUND_DECIMALS}f}" for x in load_key) + "\n")
                f.write(f"\nTotal structures in this bin: {len(idx_list)}\n")

        # One representative index per bin
        rep_indices = idx_list[:N_REP_PER_BIN]
        for j, idx in enumerate(rep_indices):
            shape = tuple(shapes[idx])
            assert shape == TARGET_SHAPE, f"Unexpected shape {shape} at idx {idx}"
            voxel_arr = topo[idx].astype(int).reshape(shape)
            out_png = os.path.join(group_dir, f"voxel_{idx}_rep{j}.png")
            title = f"idx={idx}, shape={shape}"
            plot_voxel(voxel_arr, out_png, title=title)

    print(f"Catalog written under {OUTPUT_ROOT}")

if __name__ == "__main__":
    main()
