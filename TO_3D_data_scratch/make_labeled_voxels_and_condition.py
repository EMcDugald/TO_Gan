import numpy as np
from scipy.ndimage import label
import os

# -----------------------------
# Geometry metrics
# -----------------------------

def compute_relative_internal_void_volume(voxel_arr):
    empty_voxels = np.logical_not(voxel_arr)
    labeled_voids, num_features = label(empty_voxels)

    border_labels = set()
    border_labels.update(np.unique(labeled_voids[0, :, :]))
    border_labels.update(np.unique(labeled_voids[-1, :, :]))
    border_labels.update(np.unique(labeled_voids[:, 0, :]))
    border_labels.update(np.unique(labeled_voids[:, -1, :]))
    border_labels.update(np.unique(labeled_voids[:, :, 0]))
    border_labels.update(np.unique(labeled_voids[:, :, -1]))
    border_labels.discard(0)

    internal_void_volume = 0
    for void_label in range(1, num_features + 1):
        if void_label not in border_labels:
            internal_void_volume += np.sum(labeled_voids == void_label)

    total_volume = np.prod(voxel_arr.shape)
    relative_void_volume = internal_void_volume / total_volume
    return relative_void_volume


def mass_fraction(voxel_arr):
    v = (voxel_arr > 0).astype(float)
    return v.mean()


def compactness(voxel_arr):
    v = (voxel_arr > 0)
    total = v.sum()
    if total == 0:
        return 0.0
    xs = np.where(v.any(axis=(1, 2)))[0]
    ys = np.where(v.any(axis=(0, 2)))[0]
    zs = np.where(v.any(axis=(0, 1)))[0]
    bbox_vol = ((xs[-1] - xs[0] + 1) *
                (ys[-1] - ys[0] + 1) *
                (zs[-1] - zs[0] + 1))
    return float(total) / float(bbox_vol)


# -----------------------------
# BC / load features
# -----------------------------

def bc_num_int(bc_arr):
    return int(np.asarray(bc_arr).shape[0])


def bc_pattern_full(bc_arr):
    """
    'xyz' counts string, e.g. '322' means:
      - 3 support points fix X
      - 2 support points fix Y
      - 2 support points fix Z
    """
    bc = np.asarray(bc_arr, dtype=float)
    dofs = bc[:, 3:]                  # (N_bc, 3)
    dof_fixed = (dofs > 0.5).astype(int)
    fixed_counts = dof_fixed.sum(axis=0)   # (nx, ny, nz)
    nx, ny, nz = map(int, fixed_counts)
    return nx, ny, nz


def load_magnitude(load_arr):
    load = np.asarray(load_arr, dtype=float).reshape(-1)
    fx, fy, fz = load[3:6]
    return float(np.linalg.norm([fx, fy, fz]))


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


def make_condition_vector(
    n_bc, nx, ny, nz,
    mag_bin, mass_bin, comp_bin,
    mass_frac_val, comp_val,
    max_bc=10, max_fix=10,
    n_mag_bins=3, n_mass_bins=3, n_comp_bins=3
):
    # Continuous (roughly [0,1])
    v_bc  = min(n_bc, max_bc) / max_bc
    v_nx  = min(nx, max_fix) / max_fix
    v_ny  = min(ny, max_fix) / max_fix
    v_nz  = min(nz, max_fix) / max_fix

    v_mass = float(mass_frac_val)
    v_comp = float(comp_val)

    def one_hot(k, K):
        z = np.zeros(K, dtype=np.float32)
        z[int(k)] = 1.0
        return z

    oh_mag  = one_hot(mag_bin,  n_mag_bins)
    oh_mbin = one_hot(mass_bin, n_mass_bins)
    oh_cbin = one_hot(comp_bin, n_comp_bins)

    vec = np.concatenate([
        np.array([v_bc, v_nx, v_ny, v_nz, v_mass, v_comp], dtype=np.float32),
        oh_mag, oh_mbin, oh_cbin
    ])
    return vec.astype(np.float32)


# -----------------------------
# Main generation
# -----------------------------

score_cutoff = .7

def generate_voxels_for_shape_conditional(
    shape_tuple,
    n_samples,
    topo,
    shapes,
    bcs,
    loads,
    outdir="./",
    n_mag_bins=3,
    n_mass_bins=3,
    n_comp_bins=3
):
    # indices with this shape
    indices = [i for i, shp in enumerate(shapes) if tuple(shp) == shape_tuple]
    if len(indices) < n_samples:
        print(
            f"Warning: Requested {n_samples} samples for shape {shape_tuple}, "
            f"but only {len(indices)} available. Using all available."
        )
        selected = indices
    else:
        selected = np.random.choice(indices, n_samples, replace=False)

    # First pass: compute scalar features for all selected indices
    mags        = []
    mass_fracs  = []
    comp_vals   = []
    bc_counts   = []  # (n_bc, nx, ny, nz)

    for idx in selected:
        voxel_arr = topo[idx].astype(int).reshape(shape_tuple)

        n_bc = bc_num_int(bcs[idx])
        nx, ny, nz = bc_pattern_full(bcs[idx])
        mag = load_magnitude(loads[idx])
        mfrac = mass_fraction(voxel_arr)
        comp  = compactness(voxel_arr)

        bc_counts.append((n_bc, nx, ny, nz))
        mags.append(mag)
        mass_fracs.append(mfrac)
        comp_vals.append(comp)

    mags        = np.asarray(mags, dtype=float)
    mass_fracs  = np.asarray(mass_fracs, dtype=float)
    comp_vals   = np.asarray(comp_vals, dtype=float)

    # Normalize mass and compactness to [0,1]
    eps = 1e-8
    m_min, m_max = mass_fracs.min(), mass_fracs.max()
    c_min, c_max = comp_vals.min(), comp_vals.max()

    m_norm = (mass_fracs - m_min) / (m_max - m_min + eps)

    # Flip compactness so 1 = worst (large / diffuse), 0 = best (small / compact)
    c_raw_norm = (comp_vals - c_min) / (c_max - c_min + eps)
    c_norm = 1.0 - c_raw_norm

    # Composite score and median cutoff
    score = 0.5 * m_norm + 0.5 * c_norm

    cutoff = np.quantile(score, score_cutoff)
    print("Composite score stats: min", float(score.min()),
      "max", float(score.max()), f"q={score_cutoff} cutoff", float(cutoff))

    # Bin edges for mag / mass / compactness (for condition bins)
    mag_edges  = compute_bin_edges(mags,       n_mag_bins)
    mfrac_edges = compute_bin_edges(mass_fracs, n_mass_bins)
    comp_edges  = compute_bin_edges(comp_vals,  n_comp_bins)

    print("Mag bin edges:", mag_edges)
    print("Mass fraction bin edges:", mfrac_edges)
    print("Compactness bin edges:", comp_edges)

    # Second pass: build (voxel, cond_vec, label)
    results = []
    labels_list = []

    for local_idx, idx in enumerate(selected):
        voxel_arr = topo[idx].astype(int).reshape(shape_tuple)

        n_bc, nx, ny, nz = bc_counts[local_idx]
        mag      = mags[local_idx]
        mfrac    = mass_fracs[local_idx]
        comp     = comp_vals[local_idx]
        s        = score[local_idx]

        mag_bin   = bin_index(mag,   mag_edges)
        mass_bin  = bin_index(mfrac, mfrac_edges)
        comp_bin  = bin_index(comp,  comp_edges)

        cond_vec = make_condition_vector(
            n_bc, nx, ny, nz,
            mag_bin, mass_bin, comp_bin,
            mfrac, comp,
            max_bc=10, max_fix=10,
            n_mag_bins=n_mag_bins,
            n_mass_bins=n_mass_bins,
            n_comp_bins=n_comp_bins
        )

        # NEW: human-readable condition string
        cond_str = (
            f"BCn_{n_bc}_BCpattern_{nx}{ny}{nz}_"
            f"LoadMagBin_{mag_bin}_MassBin_{mass_bin}_CompBin_{comp_bin}"
            )
        label_val = 1 if s < cutoff else 0

        labels_list.append(label_val)

        results.append((voxel_arr, cond_vec, label_val, cond_str))

    n_pos = int(sum(labels_list))
    n_neg = len(labels_list) - n_pos

    fname = (
        f"{len(results)}_labeled_voxels_"
        f"{shape_tuple[0]}x{shape_tuple[1]}x{shape_tuple[2]}_"
        f"score{score_cutoff}.npy"
    )
    data_path = os.path.join(outdir, fname)
    np.save(data_path, np.array(results, dtype=object))

    # NEW: save cutoff and score_cutoff in a sidecar .npz
    meta_fname = (
        f"{len(results)}_labeled_voxels_"
        f"{shape_tuple[0]}x{shape_tuple[1]}x{shape_tuple[2]}_"
        f"score{score_cutoff}_meta.npz"
    )
    meta_path = os.path.join(outdir, meta_fname)
    np.savez(
        meta_path,
        cutoff_train=float(cutoff),
        score_cutoff=float(score_cutoff),
        m_min=float(m_min),
        m_max=float(m_max),
        c_min=float(c_min),
        c_max=float(c_max),
    )

    print(
        f"Saved {len(results)} labeled voxel structures of shape {shape_tuple} "
        f"to {fname} (cutoff={cutoff:.4f}, pos={n_pos}, neg={n_neg})"
    )



if __name__ == "__main__":
    # Adjust these paths as needed
    data_root = "./data"
    topo   = np.load(os.path.join(data_root, "topologies.npy"), allow_pickle=True)
    shapes = np.load(os.path.join(data_root, "shapes.npy"), allow_pickle=True)
    bcs    = np.load(os.path.join(data_root, "boundary_conditions.npy"), allow_pickle=True)
    loads  = np.load(os.path.join(data_root, "loads.npy"), allow_pickle=True)

    unique_shapes = set(tuple(shape) for shape in shapes)
    print("Unique shapes detected:", unique_shapes)

    target_shape = (32, 32, 32)
    n_voxels_to_make = 10000

    # New output dir (cond_gan instead of gan)
    output_dir = "/xdisk/hdb/emcdugald/to_cond_gan/train_data/323232"
    os.makedirs(output_dir, exist_ok=True)

    if target_shape is None:
        for shape_tuple in unique_shapes:
            generate_voxels_for_shape_conditional(
                shape_tuple,
                n_voxels_to_make,
                topo,
                shapes,
                bcs,
                loads,
                outdir=output_dir,
            )
    else:
        if target_shape not in unique_shapes:
            print(
                f"Error: Target shape {target_shape} not found in your data. "
                f"Available shapes are: {unique_shapes}"
            )
        else:
            generate_voxels_for_shape_conditional(
                target_shape,
                n_voxels_to_make,
                topo,
                shapes,
                bcs,
                loads,
                outdir=output_dir,
            )
