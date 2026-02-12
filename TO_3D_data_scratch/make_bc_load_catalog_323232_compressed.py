import os
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# -----------------------------
# Config
# -----------------------------

DATA_ROOT = "/home/u26/emcdugald/TO_Gan/TO_3D_data_scratch/data"
OUTPUT_ROOT = "/home/u26/emcdugald/TO_Gan/TO_3D_data_scratch/bc_load_catalog_323232_compressed"

TARGET_SHAPE = (32, 32, 32)

MAX_TOTAL_PLOTS = 500      # upper bound on total images
N_REP_PER_BIN = 5          # images per selected bin

N_LOAD_MAG_BINS = 50        # 5–10 is reasonable; change as desired

os.makedirs(OUTPUT_ROOT, exist_ok=True)

# -----------------------------
# Helper functions
# -----------------------------


def plot_voxel(binary_arr, save_path, title=""):
    """Simple 3D voxel plot."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.voxels(binary_arr, edgecolor="k", linewidth=0.2)
    ax.set_title(title)
    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def bc_num_int(bc_arr):
    """
    Exact number of BC points (e.g., 3, 4, 5).
    """
    return int(np.asarray(bc_arr).shape[0])


# def bc_pattern_full(bc_arr):
#     """
#     Full BC pattern as axes fixed anywhere:
#         one of: 'none','X','Y','Z','XY','XZ','YZ','XYZ'
#     """
#     bc = np.asarray(bc_arr, dtype=float)
#     dofs = bc[:, 3:]          # (N_bc, 3)
#     dof_fixed = (dofs > 0.5).astype(int)
#     fixed_counts = dof_fixed.sum(axis=0)  # (nx, ny, nz)
#     axes = ''.join(ax for ax, c in zip("XYZ", fixed_counts) if c > 0)
#     return axes if axes else "none"

def bc_pattern_full(bc_arr):
    """
    BC pattern as counts of fixed axes:
    e.g. '322' means:
      - 3 support points fix X
      - 2 support points fix Y
      - 2 support points fix Z
    """
    bc = np.asarray(bc_arr, dtype=float)
    dofs = bc[:, 3:]                  # (N_bc, 3)
    dof_fixed = (dofs > 0.5).astype(int)
    fixed_counts = dof_fixed.sum(axis=0)   # (nx, ny, nz)
    nx, ny, nz = map(int, fixed_counts)
    return f"{nx}{ny}{nz}"



def compute_load_magnitudes(loads, indices):
    mags = []
    for i in indices:
        load = np.asarray(loads[i], dtype=float).reshape(-1)
        fx, fy, fz = load[3:6]
        mags.append(np.linalg.norm([fx, fy, fz]))
    return np.array(mags, dtype=float)


def compute_mag_bin_edges(mags, n_bins):
    """
    Quantile-based bin edges for magnitudes.
    Returns array of length n_bins+1 (edges), including min and max.
    """
    # Use slightly extended range to avoid edge issues
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(mags, qs)
    return edges


def load_mag_bin_index(mag, edges):
    """
    Given a magnitude and bin edges, return integer bin index in [0, n_bins-1].
    """
    # np.searchsorted returns index in [1..len(edges)-1]; clamp to [0..n_bins-1]
    idx = int(np.searchsorted(edges, mag, side="right") - 1)
    n_bins = len(edges) - 1
    if idx < 0:
        idx = 0
    elif idx >= n_bins:
        idx = n_bins - 1
    return idx


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

    # Compute load magnitudes on 32^3 subset and define quantile bins
    mags = compute_load_magnitudes(loads, indices_323232)
    print("Load magnitude stats on 32x32x32 subset:")
    print("  min:", float(mags.min()), "max:", float(mags.max()))
    print("  mean:", float(mags.mean()))
    for q in [0.25, 0.5, 0.75]:
        print(f"  quantile {q}: {float(np.quantile(mags, q))}")

    mag_edges = compute_mag_bin_edges(mags, N_LOAD_MAG_BINS)
    print("Load magnitude bin edges:", mag_edges)

    # Group indices by compressed key
    # Key = (BCn_int, BCpattern_full, LoadMag_bin_index)
    groups = {}
    for i in indices_323232:
        n_int = bc_num_int(bcs[i])
        pat_full = bc_pattern_full(bcs[i])

        load = np.asarray(loads[i], dtype=float).reshape(-1)
        fx, fy, fz = load[3:6]
        mag = float(np.linalg.norm([fx, fy, fz]))
        mag_bin = load_mag_bin_index(mag, mag_edges)

        key = (n_int, pat_full, mag_bin)
        groups.setdefault(key, []).append(i)

    print(f"Formed {len(groups)} compressed (BC, load) bins")
    for key, idxs in groups.items():
        print(f"  Bin {key}: {len(idxs)} samples")

    # Limit number of bins / plots
    all_keys = list(groups.keys())
    if len(all_keys) <= MAX_TOTAL_PLOTS:
        selected_keys = all_keys
    else:
        selected_keys = random.sample(all_keys, MAX_TOTAL_PLOTS)

    print(
        f"Will plot at most {len(selected_keys) * N_REP_PER_BIN} structures "
        f"from {len(selected_keys)} randomly selected bins"
    )

    # For each selected group, create directory and save voxel plots
    for (n_int, pat_full, mag_bin) in selected_keys:
        idx_list = groups[(n_int, pat_full, mag_bin)]

        group_dir = os.path.join(
            OUTPUT_ROOT,
            f"BCn_{n_int}_BCpattern_{pat_full}_LoadMagBin_{mag_bin}"
        )
        os.makedirs(group_dir, exist_ok=True)

        # Meta file with basic info + indices
        meta_path = os.path.join(group_dir, "meta.txt")
        if not os.path.exists(meta_path):
            with open(meta_path, "w") as f:
                f.write("Compressed BC/Load features for this bin:\n")
                f.write(f"  BCn_int: {n_int}\n")
                f.write(f"  BCpattern_full: {pat_full}\n")
                f.write(f"  LoadMag_bin_index: {mag_bin}\n")
                f.write(f"  LoadMag_bin_edges: {mag_edges.tolist()}\n\n")

                f.write("All indices in this bin:\n")
                for idx in idx_list:
                    f.write(f"  {idx}\n")

                f.write("\nNote: raw BCs and loads are in boundary_conditions.npy and loads.npy.\n")

        # Representative voxel plots
        rep_indices = idx_list[:N_REP_PER_BIN]
        for j, idx in enumerate(rep_indices):
            shape = tuple(shapes[idx])
            assert shape == TARGET_SHAPE, f"Unexpected shape {shape} at idx {idx}"
            voxel_arr = topo[idx].astype(int).reshape(shape)
            out_png = os.path.join(group_dir, f"voxel_{idx}_rep{j}.png")
            title = f"idx={idx}, shape={shape}"
            plot_voxel(voxel_arr, out_png, title=title)

        # -----------------------------
        # Plot load‑magnitude distribution
        # -----------------------------
        plt.figure()
        plt.hist(mags, bins=50, density=False, alpha=0.7, edgecolor="k")
        plt.xlabel("Load magnitude")
        plt.ylabel("Count")
        plt.title("Distribution of load magnitudes (32×32×32 subset)")
        plt.tight_layout()

        mag_hist_path = os.path.join(OUTPUT_ROOT, "load_magnitude_histogram.png")
        plt.savefig(mag_hist_path, dpi=200)
        plt.close()
        print(f"Saved load‑magnitude histogram to {mag_hist_path}")

    print(f"Compressed catalog written under {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
