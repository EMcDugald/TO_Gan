import numpy as np
import os


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
    bbox_vol = ((xs[-1]-xs[0]+1) * (ys[-1]-ys[0]+1) * (zs[-1]-zs[0]+1))
    return float(total) / float(bbox_vol)


def compute_bin_edges(values, n_bins):
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    return np.quantile(values, qs)


def bin_index(x, edges):
    idx = int(np.searchsorted(edges, x, side="right") - 1)
    n_bins = len(edges) - 1
    if idx < 0: idx = 0
    elif idx >= n_bins: idx = n_bins - 1
    return idx


def bc_octant_code(bc_arr):
    """
    Return an 8-bit string indicating which octants contain at least one BC point.
    Octant ordering: bit i corresponds to (ix, iy, iz) where i = ix*4 + iy*2 + iz.
    Coordinates are normalized by the BC cloud's own bounding box.
    """
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
        octants[ix * 4 + iy * 2 + iz] = 1
    return "".join(str(v) for v in octants.tolist())


def make_condition_vector(octant_bits, mass_bin, n_mass_bins=3, mass_frac_val=0.0):
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


score_cutoff = 0.7


def generate_voxels_octant_mass(
    shape_tuple,
    n_samples,
    topo,
    shapes,
    bcs,
    outdir="./",
    n_mass_bins=3,
):
    indices = [i for i, shp in enumerate(shapes) if tuple(shp) == shape_tuple]
    if len(indices) < n_samples:
        print(f"Warning: only {len(indices)} available, using all.")
        selected = indices
    else:
        selected = list(np.random.choice(indices, n_samples, replace=False))

    # First pass: compute mass fracs and compactness for scoring + binning
    mass_fracs = []
    comp_vals = []
    oct_codes = []

    for idx in selected:
        voxel_arr = topo[idx].astype(int).reshape(shape_tuple)
        mass_fracs.append(mass_fraction(voxel_arr))
        comp_vals.append(compactness(voxel_arr))
        oct_codes.append(bc_octant_code(bcs[idx]))

    mass_fracs = np.asarray(mass_fracs, dtype=float)
    comp_vals  = np.asarray(comp_vals,  dtype=float)

    # Normalize for composite score (same logic as before)
    eps = 1e-8
    m_min, m_max = mass_fracs.min(), mass_fracs.max()
    c_min, c_max = comp_vals.min(),  comp_vals.max()

    m_norm   = (mass_fracs - m_min) / (m_max - m_min + eps)
    c_norm   = 1.0 - (comp_vals - c_min) / (c_max - c_min + eps)
    score    = 0.5 * m_norm + 0.5 * c_norm
    cutoff   = np.quantile(score, score_cutoff)

    print(f"Score stats: min={score.min():.4f} max={score.max():.4f} q{score_cutoff}={cutoff:.4f}")

    # Mass bin edges (for condition vector)
    mass_edges = compute_bin_edges(mass_fracs, n_mass_bins)
    print("Mass bin edges:", mass_edges)

    # Second pass: assemble (voxel, cond_vec, label, cond_str)
    results = []
    labels_list = []

    for local_idx, idx in enumerate(selected):
        voxel_arr  = topo[idx].astype(int).reshape(shape_tuple)
        mfrac      = mass_fracs[local_idx]
        m_bin      = bin_index(mfrac, mass_edges)
        oct_code   = oct_codes[local_idx]
        s          = score[local_idx]

        cond_vec   = make_condition_vector(
            oct_code, m_bin,
            n_mass_bins=n_mass_bins,
            mass_frac_val=mfrac,
        )
        label_val  = 1 if s < cutoff else 0
        cond_str   = f"oct_{oct_code}_massBin_{m_bin}"

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
    meta_path  = os.path.join(outdir, meta_fname)
    np.savez(
        meta_path,
        cutoff_train=float(cutoff),
        score_cutoff=float(score_cutoff),
        m_min=float(m_min),
        m_max=float(m_max),
        c_min=float(c_min),
        c_max=float(c_max),
        mass_bin_edges=mass_edges,
    )

    print(
        f"Saved {len(results)} samples to {fname} "
        f"(pos={n_pos}, neg={n_neg}, cutoff={cutoff:.4f})"
    )


if __name__ == "__main__":
    data_root = "./data"
    topo   = np.load(os.path.join(data_root, "topologies.npy"), allow_pickle=True)
    shapes = np.load(os.path.join(data_root, "shapes.npy"), allow_pickle=True)
    bcs    = np.load(os.path.join(data_root, "boundary_conditions.npy"), allow_pickle=True)
    loads  = np.load(os.path.join(data_root, "loads.npy"), allow_pickle=True)

    output_dir = "/xdisk/hdb/emcdugald/to_cond_gan/train_data/323232/octant"
    os.makedirs(output_dir, exist_ok=True)

    generate_voxels_octant_mass(
        shape_tuple=(32, 32, 32),
        n_samples=10000,
        topo=topo,
        shapes=shapes,
        bcs=bcs,
        outdir=output_dir,
        n_mass_bins=3,
    )