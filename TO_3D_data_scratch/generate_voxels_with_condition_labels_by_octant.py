import numpy as np
import os


TARGET_SHAPE = (32, 32, 32)
N_MASS_BINS = 5
SCORE_CUTOFF = 0.7


def mass_fraction(voxel_arr):
    return float((voxel_arr > 0).astype(float).mean())


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
    """
    Return an 8-bit string indicating which global octants contain at least one BC point.
    Octant ordering: bit i corresponds to (ix, iy, iz) where i = ix*4 + iy*2 + iz.
    Assumes BC coordinates already live in the global unit cube [0,1]^3.
    """
    bc = np.asarray(bc_arr, dtype=float)
    if bc.ndim != 2 or bc.shape[1] < 3:
        raise ValueError(f"Unexpected BC array shape: {bc.shape}")

    pts = bc[:, :3]

    octants = np.zeros(8, dtype=int)
    for x, y, z in pts:
        ix = 1 if x >= 0.5 else 0
        iy = 1 if y >= 0.5 else 0
        iz = 1 if z >= 0.5 else 0
        octants[ix * 4 + iy * 2 + iz] = 1

    return "".join(str(v) for v in octants.tolist())


def make_condition_vector(octant_bits, mass_bin, n_mass_bins=5, mass_frac_val=0.0):
    """
    Condition vector:
      - 8 binary floats for octant occupancy
      - one-hot over mass bins (length n_mass_bins)
      - 1 continuous mass fraction value
    Total length = 8 + n_mass_bins + 1
    """
    oct_vec = np.array([float(b) for b in octant_bits], dtype=np.float32)
    oh_mass = np.zeros(n_mass_bins, dtype=np.float32)
    oh_mass[int(mass_bin)] = 1.0
    return np.concatenate([oct_vec, oh_mass, [float(mass_frac_val)]]).astype(np.float32)


def generate_voxels_octant_mass_matched(
    shape_tuple,
    n_samples,
    topo,
    shapes,
    bcs,
    outdir="./",
    n_mass_bins=5,
    score_cutoff=0.7,
    rng_seed=0,
):
    indices = [i for i, shp in enumerate(shapes) if tuple(shp) == shape_tuple]
    if len(indices) == 0:
        raise ValueError(f"No samples found with shape {shape_tuple}")

    print(f"Found {len(indices)} total entries with shape {shape_tuple}")

    all_mass_fracs = []
    all_comp_vals = []
    all_oct_codes = []

    for i in indices:
        voxel_arr = topo[i].astype(int).reshape(shape_tuple)
        all_mass_fracs.append(mass_fraction(voxel_arr))
        all_comp_vals.append(compactness(voxel_arr))
        all_oct_codes.append(bc_octant_code(bcs[i]))

    all_mass_fracs = np.asarray(all_mass_fracs, dtype=float)
    all_comp_vals = np.asarray(all_comp_vals, dtype=float)

    mass_edges = compute_bin_edges(all_mass_fracs, n_mass_bins)
    print("Mass bin edges (full population):", mass_edges)

    # Score stats from full population as well
    eps = 1e-8
    m_min, m_max = all_mass_fracs.min(), all_mass_fracs.max()
    c_min, c_max = all_comp_vals.min(), all_comp_vals.max()

    m_norm_all = (all_mass_fracs - m_min) / (m_max - m_min + eps)
    c_norm_all = 1.0 - (all_comp_vals - c_min) / (c_max - c_min + eps)
    all_scores = 0.5 * m_norm_all + 0.5 * c_norm_all
    cutoff = np.quantile(all_scores, score_cutoff)

    print(
        f"Full-population score stats: "
        f"min={all_scores.min():.4f} "
        f"max={all_scores.max():.4f} "
        f"q{score_cutoff}={cutoff:.4f}"
    )

    # -------------------------
    # Sample from the full population
    # -------------------------
    if len(indices) < n_samples:
        print(f"Warning: only {len(indices)} available, using all.")
        selected = indices
    else:
        rng = np.random.default_rng(rng_seed)
        selected = list(rng.choice(indices, size=n_samples, replace=False))

    selected_set = set(selected)

    # Build a lookup from global index -> precomputed descriptors
    global_desc = {}
    for local_idx, global_idx in enumerate(indices):
        if global_idx in selected_set:
            global_desc[global_idx] = {
                "mass_frac": all_mass_fracs[local_idx],
                "compactness": all_comp_vals[local_idx],
                "oct_code": all_oct_codes[local_idx],
                "score": all_scores[local_idx],
            }

    # -------------------------
    # Assemble training tuples
    # -------------------------
    results = []
    labels_list = []

    for idx in selected:
        voxel_arr = topo[idx].astype(int).reshape(shape_tuple)

        mfrac = global_desc[idx]["mass_frac"]
        m_bin = bin_index(mfrac, mass_edges)
        oct_code = global_desc[idx]["oct_code"]
        s = global_desc[idx]["score"]

        cond_vec = make_condition_vector(
            oct_code,
            m_bin,
            n_mass_bins=n_mass_bins,
            mass_frac_val=mfrac,
        )

        label_val = 1 if s < cutoff else 0
        cond_str = f"oct_{oct_code}_massBin_{m_bin}"

        labels_list.append(label_val)
        results.append((voxel_arr, cond_vec, label_val, cond_str))

    n_pos = sum(labels_list)
    n_neg = len(labels_list) - n_pos

    fname = (
        f"{len(results)}_labeled_voxels_"
        f"{shape_tuple[0]}x{shape_tuple[1]}x{shape_tuple[2]}_"
        f"octmass_score{score_cutoff}.npy"
    )
    data_path = os.path.join(outdir, fname)
    np.save(data_path, np.array(results, dtype=object))

    meta_fname = fname.replace(".npy", "_meta.npz")
    meta_path = os.path.join(outdir, meta_fname)
    np.savez(
        meta_path,
        cutoff_train=float(cutoff),
        score_cutoff=float(score_cutoff),
        m_min=float(m_min),
        m_max=float(m_max),
        c_min=float(c_min),
        c_max=float(c_max),
        mass_bin_edges=mass_edges,
        n_mass_bins=int(n_mass_bins),
        target_shape=np.array(shape_tuple, dtype=int),
        n_total_shape_matches=int(len(indices)),
        n_selected=int(len(selected)),
        rng_seed=int(rng_seed),
        bc_octant_mode="global_unit_cube",
        score_mode="0.5*mass_norm + 0.5*(1-compactness_norm)",
    )

    print(
        f"Saved {len(results)} samples to {fname} "
        f"(pos={n_pos}, neg={n_neg}, cutoff={cutoff:.4f})"
    )
    print(f"Data path: {data_path}")
    print(f"Meta path: {meta_path}")


if __name__ == "__main__":
    data_root = "./data"
    topo = np.load(os.path.join(data_root, "topologies.npy"), allow_pickle=True)
    shapes = np.load(os.path.join(data_root, "shapes.npy"), allow_pickle=True)
    bcs = np.load(os.path.join(data_root, "boundary_conditions.npy"), allow_pickle=True)
    loads = np.load(os.path.join(data_root, "loads.npy"), allow_pickle=True)

    output_dir = "/xdisk/hdb/emcdugald/to_cond_gan/train_data/323232/octant"
    os.makedirs(output_dir, exist_ok=True)

    generate_voxels_octant_mass_matched(
        shape_tuple=TARGET_SHAPE,
        n_samples=10,
        topo=topo,
        shapes=shapes,
        bcs=bcs,
        outdir=output_dir,
        n_mass_bins=N_MASS_BINS,
        score_cutoff=SCORE_CUTOFF,
        rng_seed=0,
    )