import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


TARGET_SHAPE = (32, 32, 32)
N_SPATIAL_BINS = 10


def mass_fraction(voxel_arr):
    return float((voxel_arr > 0).astype(np.float32).mean())


def compute_quantile_edges(values, n_bins):
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(values, qs)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-8
    return edges.astype(np.float64)


def bin_index(x, edges):
    idx = int(np.searchsorted(edges, x, side="right") - 1)
    n_bins = len(edges) - 1
    if idx < 0:
        idx = 0
    elif idx >= n_bins:
        idx = n_bins - 1
    return idx


def _as_float_array(arr):
    return np.asarray(arr, dtype=np.float64)


def parse_bc_points(bc_arr):
    bc = _as_float_array(bc_arr)
    if bc.ndim != 2 or bc.shape[1] < 3:
        raise ValueError(f"Unexpected BC array shape: {bc.shape}")
    return bc[:, :3]


def parse_load_point(load_arr):
    ld = _as_float_array(load_arr)
    if ld.ndim == 3:
        ld = ld[0]
    if ld.ndim != 2 or ld.shape[0] < 1 or ld.shape[1] < 3:
        raise ValueError(f"Unexpected load array shape: {ld.shape}")
    return ld[0, :3]


def parse_load_vector(load_arr):
    ld = _as_float_array(load_arr)
    if ld.ndim == 3:
        ld = ld[0]
    if ld.ndim != 2 or ld.shape[0] < 1 or ld.shape[1] < 6:
        raise ValueError(f"Unexpected load array shape: {ld.shape}")
    return ld[0, 3:6]


def summarize_range(vals):
    vals = np.asarray(vals, dtype=np.float64)
    return np.array([vals.min(), vals.max()], dtype=np.float32)


def pad_bc_points(bc_pts, max_bc_points):
    bc_pts = np.asarray(bc_pts, dtype=np.float32)
    if bc_pts.ndim != 2 or bc_pts.shape[1] != 3:
        raise ValueError(f"pad_bc_points expected shape (N,3), got {bc_pts.shape}")
    out = np.zeros((max_bc_points, 3), dtype=np.float32)
    mask = np.zeros((max_bc_points,), dtype=np.float32)
    n = min(len(bc_pts), max_bc_points)
    out[:n] = bc_pts[:n]
    mask[:n] = 1.0
    return out, mask


def make_fine_condition_vector(bc_arr, load_arr, max_bc_points):
    bc_pts = parse_bc_points(bc_arr)
    load_pt = parse_load_point(load_arr)
    load_vec = parse_load_vector(load_arr)
    bc_pad, bc_mask = pad_bc_points(bc_pts, max_bc_points)

    cond_parts = [
        bc_pad.reshape(-1).astype(np.float32),
        bc_mask.astype(np.float32),
        np.array([float(len(bc_pts))], dtype=np.float32),
        load_pt.astype(np.float32),
        load_vec.astype(np.float32),
    ]
    return np.concatenate(cond_parts).astype(np.float32)


def make_coarse_condition_vector(bc_arr, load_arr, spatial_edges, max_bc_points, include_load_vector=True):
    bc_pts = parse_bc_points(bc_arr)
    load_pt = parse_load_point(load_arr)
    load_vec = parse_load_vector(load_arr)

    x_edges, y_edges, z_edges = spatial_edges

    bc_x_rng = summarize_range(bc_pts[:, 0])
    bc_y_rng = summarize_range(bc_pts[:, 1])
    bc_z_rng = summarize_range(bc_pts[:, 2])

    bc_bin_features = np.zeros((max_bc_points, 3), dtype=np.float32)
    bc_mask = np.zeros((max_bc_points,), dtype=np.float32)
    n = min(len(bc_pts), max_bc_points)
    for j in range(n):
        coord = bc_pts[j]
        bc_bin_features[j, 0] = bin_index(coord[0], x_edges)
        bc_bin_features[j, 1] = bin_index(coord[1], y_edges)
        bc_bin_features[j, 2] = bin_index(coord[2], z_edges)
        bc_mask[j] = 1.0

    load_xyz_bins = np.array([
        bin_index(load_pt[0], x_edges),
        bin_index(load_pt[1], y_edges),
        bin_index(load_pt[2], z_edges),
    ], dtype=np.float32)

    cond_parts = [
        bc_x_rng,
        bc_y_rng,
        bc_z_rng,
        bc_bin_features.reshape(-1),
        bc_mask,
        np.array([float(len(bc_pts))], dtype=np.float32),
        load_xyz_bins,
    ]
    if include_load_vector:
        cond_parts.append(load_vec.astype(np.float32))
    return np.concatenate(cond_parts).astype(np.float32)


def save_mass_histogram_png(mass_values, cutoff, cutoff_method, png_path, hist_counts=None, hist_edges=None, smooth_counts=None):
    vals = np.asarray(mass_values, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8, 5))

    if hist_edges is not None:
        ax.hist(vals, bins=hist_edges, color='steelblue', alpha=0.65, edgecolor='black')
    else:
        ax.hist(vals, bins=128, color='steelblue', alpha=0.65, edgecolor='black')

    if hist_counts is not None and hist_edges is not None and smooth_counts is not None:
        centers = 0.5 * (hist_edges[:-1] + hist_edges[1:])
        ax.plot(centers, smooth_counts, color='darkorange', linewidth=2, label='Smoothed histogram')

    ax.axvline(cutoff, color='crimson', linestyle='--', linewidth=2, label=f'Threshold = {cutoff:.6f}')
    ax.set_title(f'Mass distribution ({cutoff_method})')
    ax.set_xlabel('Mass fraction')
    ax.set_ylabel('Count')
    ax.legend()
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(png_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def find_mass_threshold_valley(mass_values, n_bins=128, min_peak_frac=0.05):
    vals = np.asarray(mass_values, dtype=np.float64)
    hist, edges = np.histogram(vals, bins=n_bins)
    smooth = hist.astype(np.float64).copy()
    if len(smooth) >= 3:
        kernel = np.array([1, 2, 3, 2, 1], dtype=np.float64)
        kernel = kernel / kernel.sum()
        smooth = np.convolve(smooth, kernel, mode='same')

    peaks = []
    for i in range(1, len(smooth) - 1):
        if smooth[i] >= smooth[i - 1] and smooth[i] >= smooth[i + 1]:
            peaks.append(i)

    if len(peaks) < 2:
        thr = float(np.median(vals))
        return thr, 'median_fallback', hist, edges, smooth

    peak_heights = np.array([smooth[i] for i in peaks])
    max_h = peak_heights.max() if len(peak_heights) else 0.0
    valid_peaks = [p for p in peaks if smooth[p] >= min_peak_frac * max_h]
    if len(valid_peaks) < 2:
        valid_peaks = sorted(peaks, key=lambda i: smooth[i], reverse=True)[:2]
    else:
        valid_peaks = sorted(valid_peaks, key=lambda i: smooth[i], reverse=True)[:2]

    p1, p2 = sorted(valid_peaks[:2])
    if p2 <= p1 + 1:
        thr = float(np.median(vals))
        return thr, 'median_adjacent_peak_fallback', hist, edges, smooth

    valley_idx = p1 + int(np.argmin(smooth[p1:p2 + 1]))
    thr = float(0.5 * (edges[valley_idx] + edges[valley_idx + 1]))
    return thr, 'valley_between_two_main_modes', hist, edges, smooth


def generate_voxels_mass_conditioned(shape_tuple, n_samples, topo, shapes, bcs, loads, outdir='./', conditioning_mode='fine', n_spatial_bins=10, positive_if='low_mass', rng_seed=0, mass_quantile=None):
    indices = [i for i, shp in enumerate(shapes) if tuple(shp) == tuple(shape_tuple)]
    if not indices:
        raise ValueError(f'No samples found with shape {shape_tuple}')

    print(f'Found {len(indices)} total entries with shape {shape_tuple}')

    all_mass_fracs = []
    all_bc_pts = []
    all_load_pts = []
    bc_counts = []
    for i in indices:
        voxel_arr = np.asarray(topo[i]).astype(np.float32).reshape(shape_tuple)
        all_mass_fracs.append(mass_fraction(voxel_arr))
        bc_pts = parse_bc_points(bcs[i])
        all_bc_pts.append(bc_pts)
        all_load_pts.append(parse_load_point(loads[i]))
        bc_counts.append(bc_pts.shape[0])

    all_mass_fracs = np.asarray(all_mass_fracs, dtype=np.float64)
    bc_counts = np.asarray(bc_counts, dtype=np.int32)
    max_bc_points = int(bc_counts.max())
    min_bc_points = int(bc_counts.min())
    unique_bc_counts = sorted(set(int(x) for x in bc_counts.tolist()))
    print('BC point counts found:', unique_bc_counts)
    print('Using max_bc_points for padding:', max_bc_points)

    bc_all = np.concatenate(all_bc_pts, axis=0)
    load_all = np.stack(all_load_pts, axis=0)
    xyz_all = np.concatenate([bc_all, load_all], axis=0)
    spatial_edges = tuple(compute_quantile_edges(xyz_all[:, ax], n_spatial_bins) for ax in range(3))
    print('Spatial bin edges x/y/z:', spatial_edges)

    if mass_quantile is not None:
        if not (0.0 <= mass_quantile <= 1.0):
            raise ValueError(f'mass_quantile must be in [0, 1], got {mass_quantile}')
        cutoff = float(np.quantile(all_mass_fracs, mass_quantile))
        cutoff_method = f'user_quantile_{mass_quantile:.4f}'
        hist, hist_edges = np.histogram(all_mass_fracs, bins=128)
        smooth = hist.astype(np.float64).copy()
        if len(smooth) >= 3:
            kernel = np.array([1, 2, 3, 2, 1], dtype=np.float64)
            kernel = kernel / kernel.sum()
            smooth = np.convolve(smooth, kernel, mode='same')
    else:
        cutoff, cutoff_method, hist, hist_edges, smooth = find_mass_threshold_valley(all_mass_fracs)

    print(f'Mass threshold method: {cutoff_method}')
    print(f'Mass threshold value: {cutoff:.6f}')

    os.makedirs(outdir, exist_ok=True)

    hist_png_name = (
        f'mass_hist_'
        f'{shape_tuple[0]}x{shape_tuple[1]}x{shape_tuple[2]}_'
        f'{conditioning_mode}_{positive_if}_thr{cutoff:.6f}.png'
    )
    hist_png_path = os.path.join(outdir, hist_png_name)

    save_mass_histogram_png(
        mass_values=all_mass_fracs,
        cutoff=cutoff,
        cutoff_method=cutoff_method,
        png_path=hist_png_path,
        hist_counts=hist,
        hist_edges=hist_edges,
        smooth_counts=smooth,
    )

    print(f'Mass histogram PNG: {hist_png_path}')

    if len(indices) < n_samples:
        print(f'Warning: only {len(indices)} available, using all.')
        selected = indices
    else:
        rng = np.random.default_rng(rng_seed)
        selected = list(rng.choice(indices, size=n_samples, replace=False))

    results = []
    labels_list = []

    for idx in selected:
        voxel_arr = np.asarray(topo[idx]).astype(np.int8).reshape(shape_tuple)
        mfrac = mass_fraction(voxel_arr)
        bc_pts = parse_bc_points(bcs[idx])

        if conditioning_mode == 'fine':
            cond_vec = make_fine_condition_vector(bcs[idx], loads[idx], max_bc_points=max_bc_points)
            load_pt = parse_load_point(loads[idx])
            cond_str = (
                'fine_'
                + f'bcCount_{len(bc_pts)}_'
                + 'bc_' + '_'.join([f'({p[0]:.4f},{p[1]:.4f},{p[2]:.4f})' for p in bc_pts])
                + f'_load_({load_pt[0]:.4f},{load_pt[1]:.4f},{load_pt[2]:.4f})'
            )
        elif conditioning_mode == 'coarse':
            cond_vec = make_coarse_condition_vector(bcs[idx], loads[idx], spatial_edges, max_bc_points=max_bc_points, include_load_vector=True)
            load_pt = parse_load_point(loads[idx])
            cond_str = (
                f'coarse_bcCount_{len(bc_pts)}'
                f'_bcx_[{bc_pts[:,0].min():.4f},{bc_pts[:,0].max():.4f}]'
                f'_bcy_[{bc_pts[:,1].min():.4f},{bc_pts[:,1].max():.4f}]'
                f'_bcz_[{bc_pts[:,2].min():.4f},{bc_pts[:,2].max():.4f}]'
                f'_loadBin_({bin_index(load_pt[0], spatial_edges[0])},'
                f'{bin_index(load_pt[1], spatial_edges[1])},'
                f'{bin_index(load_pt[2], spatial_edges[2])})'
            )
        else:
            raise ValueError("conditioning_mode must be 'fine' or 'coarse'")

        if positive_if == 'low_mass':
            label_val = 1 if mfrac <= cutoff else 0
        elif positive_if == 'high_mass':
            label_val = 1 if mfrac >= cutoff else 0
        else:
            raise ValueError("positive_if must be 'low_mass' or 'high_mass'")

        labels_list.append(label_val)
        results.append((voxel_arr, cond_vec, label_val, cond_str))

    n_pos = int(sum(labels_list))
    n_neg = int(len(labels_list) - n_pos)

    fname = (
        f'{len(results)}_labeled_voxels_'
        f'{shape_tuple[0]}x{shape_tuple[1]}x{shape_tuple[2]}_'
        f'{conditioning_mode}_bcLoadOnly_massLabel_'
        f'{positive_if}_thr{cutoff:.6f}.npy'
    )
    data_path = os.path.join(outdir, fname)
    np.save(data_path, np.array(results, dtype=object))

    meta_fname = fname.replace('.npy', '_meta.npz')
    meta_path = os.path.join(outdir, meta_fname)
    np.savez(
        meta_path,
        mass_cutoff_value=float(cutoff),
        mass_cutoff_method=cutoff_method,
        positive_if=str(positive_if),
        conditioning_mode=str(conditioning_mode),
        spatial_bin_edges_x=np.asarray(spatial_edges[0], dtype=np.float64),
        spatial_bin_edges_y=np.asarray(spatial_edges[1], dtype=np.float64),
        spatial_bin_edges_z=np.asarray(spatial_edges[2], dtype=np.float64),
        n_spatial_bins=int(n_spatial_bins),
        target_shape=np.array(shape_tuple, dtype=int),
        n_total_shape_matches=int(len(indices)),
        n_selected=int(len(selected)),
        rng_seed=int(rng_seed),
        bc_shape_example=np.array(parse_bc_points(bcs[indices[0]]).shape, dtype=int),
        load_shape_example=np.array(np.asarray(loads[indices[0]]).shape, dtype=int),
        bc_count_min=int(min_bc_points),
        bc_count_max=int(max_bc_points),
        bc_count_unique=np.array(unique_bc_counts, dtype=int),
        cond_dim=int(len(results[0][1])) if len(results) > 0 else -1,
        mass_hist_counts=np.asarray(hist, dtype=np.int64),
        mass_hist_edges=np.asarray(hist_edges, dtype=np.float64),
        mass_hist_smooth=np.asarray(smooth, dtype=np.float64),
        mass_hist_png=str(hist_png_path),
        cond_description=(
            'fine: padded BC xyz coords + BC mask + BC count + load xyz + load dir; '
            'coarse: BC xyz min/max ranges + padded BC per-point xyz bins + BC mask + BC count + load xyz bins + load dir'
        ),
        label_mode='mass_only',
        mass_quantile=np.nan if mass_quantile is None else float(mass_quantile),
        mass_threshold_source='user_quantile' if mass_quantile is not None else 'auto_valley',
    )

    print(f'Saved {len(results)} samples to {fname} (pos={n_pos}, neg={n_neg}, cutoff={cutoff:.6f})')
    print(f'Data path: {data_path}')
    print(f'Meta path: {meta_path}')
    if len(results) > 0:
        print(f'Condition dim: {len(results[0][1])}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate mass-labeled voxel training data with BC/load-only conditioning and padded BC vectors.')
    parser.add_argument('--data-root', type=str, default='./data')
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--shape', type=int, nargs=3, default=list(TARGET_SHAPE))
    parser.add_argument('--n-samples', type=int, default=10000)
    parser.add_argument('--conditioning-mode', type=str, choices=['fine', 'coarse'], default='fine')
    parser.add_argument('--n-spatial-bins', type=int, default=N_SPATIAL_BINS)
    parser.add_argument('--positive-if', type=str, choices=['low_mass', 'high_mass'], default='low_mass')
    parser.add_argument('--rng-seed', type=int, default=0)
    parser.add_argument('--mass-quantile', type=float, default=None, help='If provided, use this quantile in [0,1] as the mass threshold instead of automatic valley detection.')
    args = parser.parse_args()

    topo = np.load(os.path.join(args.data_root, 'topologies.npy'), allow_pickle=True)
    shapes = np.load(os.path.join(args.data_root, 'shapes.npy'), allow_pickle=True)
    bcs = np.load(os.path.join(args.data_root, 'boundary_conditions.npy'), allow_pickle=True)
    loads = np.load(os.path.join(args.data_root, 'loads.npy'), allow_pickle=True)

    generate_voxels_mass_conditioned(
        shape_tuple=tuple(args.shape),
        n_samples=args.n_samples,
        topo=topo,
        shapes=shapes,
        bcs=bcs,
        loads=loads,
        outdir=args.outdir,
        conditioning_mode=args.conditioning_mode,
        n_spatial_bins=args.n_spatial_bins,
        positive_if=args.positive_if,
        rng_seed=args.rng_seed,
        mass_quantile=args.mass_quantile,
    )