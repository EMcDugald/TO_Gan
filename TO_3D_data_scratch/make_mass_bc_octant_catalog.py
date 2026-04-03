import os
import numpy as np
import matplotlib.pyplot as plt

DATA_ROOT = "/home/u26/emcdugald/TO_Gan/TO_3D_data_scratch/data"
OUTPUT_ROOT = "/home/u26/emcdugald/TO_Gan/TO_3D_data_scratch/octant_mass_catalog_323232"
TARGET_SHAPE = (32, 32, 32)

N_MASS_BINS = 10
MAX_TOTAL_PLOTS = 500
N_REP_PER_BIN = 5

os.makedirs(OUTPUT_ROOT, exist_ok=True)


def plot_voxel(binary_arr, save_path, title=""):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.voxels(binary_arr, edgecolor="k", linewidth=0.2)
    ax.set_title(title)
    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def mass_fraction(voxel_arr):
    v = (voxel_arr > 0).astype(float)
    return float(v.mean())


def compute_bin_edges(values, n_bins):
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    return np.quantile(values, qs)


def bin_index(x, edges):
    idx = int(np.searchsorted(edges, x, side="right") - 1)
    n_bins = len(edges) - 1
    if idx < 0:
        idx = 0
    elif idx >= n_bins:
        idx = n_bins - 1
    return idx


def bc_octant_code(bc_arr):
    bc = np.asarray(bc_arr, dtype=float)

    if bc.ndim != 2 or bc.shape[1] < 3:
        raise ValueError(f"Unexpected BC array shape: {bc.shape}")

    pts = bc[:, :3]

    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    spans = np.where((maxs - mins) > 0, maxs - mins, 1.0)

    norm = (pts - mins) / spans

    octants = np.zeros(8, dtype=int)
    for x, y, z in norm:
        ix = 1 if x >= 0.5 else 0
        iy = 1 if y >= 0.5 else 0
        iz = 1 if z >= 0.5 else 0
        oct_idx = ix * 4 + iy * 2 + iz
        octants[oct_idx] = 1

    return "".join(str(v) for v in octants.tolist())


def main():
    topo = np.load(os.path.join(DATA_ROOT, "topologies.npy"), allow_pickle=True)
    shapes = np.load(os.path.join(DATA_ROOT, "shapes.npy"), allow_pickle=True)
    bcs = np.load(os.path.join(DATA_ROOT, "boundary_conditions.npy"), allow_pickle=True)

    print("Loaded:")
    print("  topo.shape:", topo.shape)
    print("  shapes.shape:", shapes.shape)
    print("  bcs.shape:", bcs.shape)

    indices = [i for i, shp in enumerate(shapes) if tuple(shp) == TARGET_SHAPE]
    print(f"Found {len(indices)} entries with shape {TARGET_SHAPE}")

    mass_vals = []
    oct_codes = []

    for i in indices:
        voxel = topo[i].astype(int).reshape(TARGET_SHAPE)
        mass_vals.append(mass_fraction(voxel))
        oct_codes.append(bc_octant_code(bcs[i]))

    mass_vals = np.asarray(mass_vals, dtype=float)
    mass_edges = compute_bin_edges(mass_vals, N_MASS_BINS)

    print("Mass bin edges:", mass_edges)

    groups = {}
    for local_idx, i in enumerate(indices):
        voxel = topo[i].astype(int).reshape(TARGET_SHAPE)
        m = mass_vals[local_idx]
        m_bin = bin_index(m, mass_edges)
        oct_code = oct_codes[local_idx]
        key = f"{oct_code}{m_bin}"
        groups.setdefault(key, []).append(i)

    print(f"Formed {len(groups)} bins")
    for key, idxs in groups.items():
        print(f"  Bin {key}: {len(idxs)} samples")

    all_keys = list(groups.keys())
    if len(all_keys) <= MAX_TOTAL_PLOTS:
        selected_keys = all_keys
    else:
        rng = np.random.default_rng(0)
        selected_keys = list(rng.choice(all_keys, size=MAX_TOTAL_PLOTS, replace=False))

    print(
        f"Will plot at most {len(selected_keys) * N_REP_PER_BIN} structures "
        f"from {len(selected_keys)} selected bins"
    )

    meta_path = os.path.join(OUTPUT_ROOT, "catalog_meta.txt")
    with open(meta_path, "w") as f:
        f.write(f"TARGET_SHAPE = {TARGET_SHAPE}\n")
        f.write(f"N_MASS_BINS = {N_MASS_BINS}\n")
        f.write(f"Mass bin edges = {mass_edges.tolist()}\n")
        f.write("Key format: 8 octant occupancy bits followed by 1 mass-bin digit\n")
    print(f"Saved catalog metadata to {meta_path}")

    for key in selected_keys:
        idx_list = groups[key]
        group_dir = os.path.join(OUTPUT_ROOT, f"key_{key}")
        os.makedirs(group_dir, exist_ok=True)

        meta_path = os.path.join(group_dir, "meta.txt")
        if not os.path.exists(meta_path):
            with open(meta_path, "w") as f:
                f.write("Octant occupancy + mass-bin catalog entry\n")
                f.write(f"Key: {key}\n")
                f.write(f"Mass bin edges: {mass_edges.tolist()}\n")
                f.write(f"Indices in this bin ({len(idx_list)}):\n")
                for idx in idx_list:
                    f.write(f"  {idx}\n")

        rep_indices = idx_list[:N_REP_PER_BIN]
        for j, idx in enumerate(rep_indices):
            shape = tuple(shapes[idx])
            assert shape == TARGET_SHAPE, f"Unexpected shape {shape} at idx {idx}"
            voxel_arr = topo[idx].astype(int).reshape(shape)
            out_png = os.path.join(group_dir, f"voxel_{idx}_rep{j}.png")
            plot_voxel(voxel_arr, out_png, title=f"idx={idx}, key={key}")

    hist_path = os.path.join(OUTPUT_ROOT, "mass_fraction_histogram.png")
    plt.figure()
    plt.hist(mass_vals, bins=50, edgecolor="k", alpha=0.7)
    plt.xlabel("Mass fraction")
    plt.ylabel("Count")
    plt.title("Mass fraction distribution (32x32x32 subset)")
    plt.tight_layout()
    plt.savefig(hist_path, dpi=200)
    plt.close()
    print(f"Saved mass histogram to {hist_path}")

    print(f"Catalog written under {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()