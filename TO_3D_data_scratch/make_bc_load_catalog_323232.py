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

MAX_TOTAL_PLOTS = 100      # total images to generate (upper bound)
N_REP_PER_BIN = 1          # images per selected bin

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


def bin_coord(v):
    """Bin a scalar in [0,1] into 'low', 'mid', 'high'."""
    if v < 1.0 / 3.0:
        return "low"
    elif v < 2.0 / 3.0:
        return "mid"
    else:
        return "high"


def bc_features(bc_arr, tol=1e-3):
    """
    Extract interpretable BC features from a (N_bc, 6) array.

    Returns:
        (faces_key, bc_center_bins, axes_fixed)
        - faces_key: tuple of faces like ('X0','Y1',...)
        - bc_center_bins: ('low'/'mid'/'high',)*3 for mean (x,y,z)
        - axes_fixed: e.g. 'X', 'YZ', 'XYZ', 'none'
    """
    bc = np.asarray(bc_arr, dtype=float)
    pts = bc[:, :3]
    dofs = bc[:, 3:]

    # Faces for each support point
    faces = []
    for x, y, z in pts:
        if abs(x) < tol:
            faces.append("X0")
        elif abs(x - 1.0) < tol:
            faces.append("X1")
        if abs(y) < tol:
            faces.append("Y0")
        elif abs(y - 1.0) < tol:
            faces.append("Y1")
        if abs(z) < tol:
            faces.append("Z0")
        elif abs(z - 1.0) < tol:
            faces.append("Z1")

    faces_key = tuple(sorted(set(faces)))

    # BC center (mean coord) binned
    mean_xyz = pts.mean(axis=0)
    bc_center_bins = tuple(bin_coord(v) for v in mean_xyz)

    # DOF pattern: which axes are fixed anywhere
    dof_fixed = (dofs > 0.5).astype(int)
    fixed_counts = dof_fixed.sum(axis=0)  # (nx, ny, nz)
    axes_fixed = "".join(ax for ax, c in zip("XYZ", fixed_counts) if c > 0)
    if not axes_fixed:
        axes_fixed = "none"

    return faces_key, bc_center_bins, axes_fixed


def load_features(load_arr, tol=1e-3):
    """
    Extract interpretable load features from a (1,6) array.

    Returns:
        (face, pos_bins, dir_bin, mag_bin)
        - face: 'X0','X1','Y0','Y1','Z0','Z1','interior'
        - pos_bins: (u_bin, v_bin) in 'low'/'mid'/'high'
        - dir_bin: e.g. 'posX','negY','none'
        - mag_bin: 'small','medium','large'
    """
    load = np.asarray(load_arr, dtype=float).reshape(-1)
    x, y, z, fx, fy, fz = load

    # Face by which coord is on boundary
    face = "interior"
    if abs(x) < tol:
        face = "X0"
    elif abs(x - 1.0) < tol:
        face = "X1"
    elif abs(y) < tol:
        face = "Y0"
    elif abs(y - 1.0) < tol:
        face = "Y1"
    elif abs(z) < tol:
        face = "Z0"
    elif abs(z - 1.0) < tol:
        face = "Z1"

    # Position on that face: choose two in-plane coords and bin
    if face in ["X0", "X1"]:
        u, v = y, z
    elif face in ["Y0", "Y1"]:
        u, v = x, z
    elif face in ["Z0", "Z1"]:
        u, v = x, y
    else:
        u, v = x, y  # fallback

    u_bin = bin_coord(u)
    v_bin = bin_coord(v)

    # Direction and magnitude
    F = np.array([fx, fy, fz], dtype=float)
    mag = float(np.linalg.norm(F))
    if mag < 1e-6:
        dir_bin = "none"
    else:
        k = int(np.argmax(np.abs(F)))
        axis = "XYZ"[k]
        sign = "pos" if F[k] > 0 else "neg"
        dir_bin = f"{sign}{axis}"

    # Simple magnitude bins; you can refine later
    if mag < 0.2:
        mag_bin = "small"
    elif mag < 0.6:
        mag_bin = "medium"
    else:
        mag_bin = "large"

    return face, (u_bin, v_bin), dir_bin, mag_bin


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

    # Group indices by interpretable (BC features, load features)
    groups = {}
    for i in indices_323232:
        bc_key = bc_features(bcs[i])
        load_key = load_features(loads[i])
        key = (bc_key, load_key)
        groups.setdefault(key, []).append(i)

    print(f"Formed {len(groups)} feature-based (BC, load) bins")

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
    for bc_key, load_key in selected_keys:
        idx_list = groups[(bc_key, load_key)]

        # Unpack feature keys
        faces_key, bc_center_bins, axes_fixed = bc_key
        face, (u_bin, v_bin), dir_bin, mag_bin = load_key

        faces_str = "-".join(faces_key) if faces_key else "none"
        bc_center_str = "-".join(bc_center_bins)

        group_dir = os.path.join(
            OUTPUT_ROOT,
            f"BCfaces_{faces_str}"
            f"_BCcenter_{bc_center_str}"
            f"_BCdofs_{axes_fixed}"
            f"_Lface_{face}"
            f"_Lpos_{u_bin}{v_bin}"
            f"_Ldir_{dir_bin}"
            f"_Lmag_{mag_bin}",
        )
        os.makedirs(group_dir, exist_ok=True)

        # Meta file with raw BC/load arrays and index list
        meta_path = os.path.join(group_dir, "meta.txt")
        if not os.path.exists(meta_path):
            with open(meta_path, "w") as f:
                f.write("BC features:\n")
                f.write(f"  faces_key: {faces_key}\n")
                f.write(f"  bc_center_bins: {bc_center_bins}\n")
                f.write(f"  axes_fixed: {axes_fixed}\n\n")

                f.write("Load features:\n")
                f.write(f"  face: {face}\n")
                f.write(f"  pos_bins: ({u_bin}, {v_bin})\n")
                f.write(f"  dir_bin: {dir_bin}\n")
                f.write(f"  mag_bin: {mag_bin}\n\n")

                f.write("All indices in this bin:\n")
                for idx in idx_list:
                    f.write(f"  {idx}\n")

        # Representative voxel plots
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
