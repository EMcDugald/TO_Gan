import os
import json
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
    return bc[:, :3].astype(np.float32)


def parse_bc_dofs(bc_arr):
    bc = _as_float_array(bc_arr)
    if bc.ndim != 2 or bc.shape[1] < 6:
        raise ValueError(f"Unexpected BC array shape for DOFs: {bc.shape}")
    return bc[:, 3:6].astype(np.float32)


def parse_load_point(load_arr):
    ld = _as_float_array(load_arr)
    if ld.ndim == 3:
        ld = ld[0]
    if ld.ndim != 2 or ld.shape[0] < 1 or ld.shape[1] < 3:
        raise ValueError(f"Unexpected load array shape: {ld.shape}")
    return ld[0, :3].astype(np.float32)


def parse_load_vector(load_arr):
    ld = _as_float_array(load_arr)
    if ld.ndim == 3:
        ld = ld[0]
    if ld.ndim != 2 or ld.shape[0] < 1 or ld.shape[1] < 6:
        raise ValueError(f"Unexpected load array shape: {ld.shape}")
    return ld[0, 3:6].astype(np.float32)


def summarize_range(vals):
    vals = np.asarray(vals, dtype=np.float64)
    return np.array([vals.min(), vals.max()], dtype=np.float32)


def pad_array_rows(arr, max_rows, n_cols):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != n_cols:
        raise ValueError(f"Expected shape (N,{n_cols}), got {arr.shape}")
    out = np.zeros((max_rows, n_cols), dtype=np.float32)
    mask = np.zeros((max_rows,), dtype=np.float32)
    n = min(len(arr), max_rows)
    out[:n] = arr[:n]
    mask[:n] = 1.0
    return out, mask, n


def build_condition_vector(
    bc_arr,
    load_arr,
    conditioning_spec,
    spatial_edges,
    max_bc_points,
):
    bc_pts = parse_bc_points(bc_arr)
    bc_dofs = parse_bc_dofs(bc_arr)
    load_pt = parse_load_point(load_arr)
    load_vec = parse_load_vector(load_arr)

    bc_pts_pad, bc_mask, bc_count = pad_array_rows(bc_pts, max_bc_points, 3)
    bc_dofs_pad, _, _ = pad_array_rows(bc_dofs, max_bc_points, 3)

    x_edges, y_edges, z_edges = spatial_edges

    parts = []
    slices = {}
    pos = 0

    def add_part(name, arr):
        nonlocal pos
        arr = np.asarray(arr, dtype=np.float32).reshape(-1)
        slices[name] = (pos, pos + len(arr))
        parts.append(arr)
        pos += len(arr)

    bc_loc_mode = conditioning_spec["bc_locations"]
    bc_dof_mode = conditioning_spec["bc_dofs"]
    load_loc_mode = conditioning_spec["load_location"]
    load_dir_mode = conditioning_spec["load_direction"]

    if bc_loc_mode == "fine":
        add_part("bc_points", bc_pts_pad.reshape(-1))
        add_part("bc_mask", bc_mask)
        add_part("bc_count", np.array([float(bc_count)], dtype=np.float32))
    elif bc_loc_mode == "coarse":
        bc_x_rng = summarize_range(bc_pts[:, 0])
        bc_y_rng = summarize_range(bc_pts[:, 1])
        bc_z_rng = summarize_range(bc_pts[:, 2])

        bc_bins = np.zeros((max_bc_points, 3), dtype=np.float32)
        for j in range(bc_count):
            coord = bc_pts[j]
            bc_bins[j, 0] = bin_index(coord[0], x_edges)
            bc_bins[j, 1] = bin_index(coord[1], y_edges)
            bc_bins[j, 2] = bin_index(coord[2], z_edges)

        add_part("bc_x_range", bc_x_rng)
        add_part("bc_y_range", bc_y_rng)
        add_part("bc_z_range", bc_z_rng)
        add_part("bc_bins", bc_bins.reshape(-1))
        add_part("bc_mask", bc_mask)
        add_part("bc_count", np.array([float(bc_count)], dtype=np.float32))
    elif bc_loc_mode != "omit":
        raise ValueError(f"Unknown bc_locations mode: {bc_loc_mode}")

    if bc_dof_mode == "fine":
        add_part("bc_dofs", bc_dofs_pad.reshape(-1))
    elif bc_dof_mode != "omit":
        raise ValueError(f"Unknown bc_dofs mode: {bc_dof_mode}")

    if load_loc_mode == "fine":
        add_part("load_point", load_pt)
    elif load_loc_mode == "coarse":
        load_xyz_bins = np.array([
            bin_index(load_pt[0], x_edges),
            bin_index(load_pt[1], y_edges),
            bin_index(load_pt[2], z_edges),
        ], dtype=np.float32)
        add_part("load_bins", load_xyz_bins)
    elif load_loc_mode != "omit":
        raise ValueError(f"Unknown load_location mode: {load_loc_mode}")

    if load_dir_mode == "fine":
        add_part("load_dir", load_vec)
    elif load_dir_mode != "omit":
        raise ValueError(f"Unknown load_direction mode: {load_dir_mode}")

    cond_vec = np.concatenate(parts).astype(np.float32) if parts else np.zeros((0,), dtype=np.float32)

    aux = {
        "bc_points": bc_pts,
        "bc_dofs": bc_dofs,
        "load_point": load_pt,
        "load_dir": load_vec,
        "bc_count": int(bc_count),
        "bc_mask": bc_mask,
        "slices": slices,
    }
    return cond_vec, aux


def make_condition_string(conditioning_spec, aux, spatial_edges):
    bc_pts = aux["bc_points"]
    bc_dofs = aux["bc_dofs"]
    load_pt = aux["load_point"]
    load_vec = aux["load_dir"]
    bc_count = aux["bc_count"]

    parts = ["cond"]

    parts.append(f"bcLoc_{conditioning_spec['bc_locations']}")
    parts.append(f"bcDofs_{conditioning_spec['bc_dofs']}")
    parts.append(f"loadLoc_{conditioning_spec['load_location']}")
    parts.append(f"loadDir_{conditioning_spec['load_direction']}")
    parts.append(f"bcCount_{bc_count}")

    if conditioning_spec["bc_locations"] == "fine":
        parts.append(
            "bc_" + "_".join(
                [f"({p[0]:.4f},{p[1]:.4f},{p[2]:.4f})" for p in bc_pts]
            )
        )
    elif conditioning_spec["bc_locations"] == "coarse":
        parts.append(f"bcx_[{bc_pts[:,0].min():.4f},{bc_pts[:,0].max():.4f}]")
        parts.append(f"bcy_[{bc_pts[:,1].min():.4f},{bc_pts[:,1].max():.4f}]")
        parts.append(f"bcz_[{bc_pts[:,2].min():.4f},{bc_pts[:,2].max():.4f}]")

    if conditioning_spec["bc_dofs"] == "fine":
        parts.append(
            "bcD_" + "_".join(
                [f"({int(d[0])},{int(d[1])},{int(d[2])})" for d in bc_dofs]
            )
        )

    if conditioning_spec["load_location"] == "fine":
        parts.append(f"load_({load_pt[0]:.4f},{load_pt[1]:.4f},{load_pt[2]:.4f})")
    elif conditioning_spec["load_location"] == "coarse":
        x_edges, y_edges, z_edges = spatial_edges
        parts.append(
            f"loadBin_({bin_index(load_pt[0], x_edges)},"
            f"{bin_index(load_pt[1], y_edges)},"
            f"{bin_index(load_pt[2], z_edges)})"
        )

    if conditioning_spec["load_direction"] == "fine":
        parts.append(f"loadDir_({load_vec[0]:.4f},{load_vec[1]:.4f},{load_vec[2]:.4f})")

    return "_".join(parts)


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


def parse_conditioning_spec(spec_str):
    tokens = [tok.strip() for tok in spec_str.split(",") if tok.strip()]
    spec = {
        "bc_locations": "coarse",
        "bc_dofs": "omit",
        "load_location": "coarse",
        "load_direction": "omit",
    }
    for tok in tokens:
        key, val = [x.strip() for x in tok.split("=", 1)]
        if key not in spec:
            raise ValueError(f"Unknown conditioning spec key: {key}")
        spec[key] = val
    return spec


def conditioning_spec_to_string(spec):
    return ",".join([
        f"bc_locations={spec['bc_locations']}",
        f"bc_dofs={spec['bc_dofs']}",
        f"load_location={spec['load_location']}",
        f"load_direction={spec['load_direction']}",
    ])


def generate_dataset_common(
    shape_tuple,
    n_samples,
    topo,
    shapes,
    bcs,
    loads,
    outdir,
    conditioning_spec,
    positive_if,
    rng_seed,
    mass_quantile,
    n_spatial_bins,
    label_mode,
):
    indices = [i for i, shp in enumerate(shapes) if tuple(np.asarray(shp).tolist()) == tuple(shape_tuple)]
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

    bc_all = np.concatenate(all_bc_pts, axis=0)
    load_all = np.stack(all_load_pts, axis=0)
    xyz_all = np.concatenate([bc_all, load_all], axis=0)
    spatial_edges = tuple(compute_quantile_edges(xyz_all[:, ax], n_spatial_bins) for ax in range(3))

    if mass_quantile is not None:
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

    os.makedirs(outdir, exist_ok=True)

    spec_str = conditioning_spec_to_string(conditioning_spec)
    spec_slug = (
        f"bcLoc-{conditioning_spec['bc_locations']}_"
        f"bcDofs-{conditioning_spec['bc_dofs']}_"
        f"loadLoc-{conditioning_spec['load_location']}_"
        f"loadDir-{conditioning_spec['load_direction']}"
    )

    hist_png_name = (
        f"mass_hist_{shape_tuple[0]}x{shape_tuple[1]}x{shape_tuple[2]}_"
        f"{spec_slug}_{positive_if}_thr{cutoff:.6f}.png"
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

    if len(indices) < n_samples:
        selected = indices
    else:
        rng = np.random.default_rng(rng_seed)
        selected = list(rng.choice(indices, size=n_samples, replace=False))

    results = []
    labels_list = []
    example_slices = None

    for idx in selected:
        voxel_arr = np.asarray(topo[idx]).astype(np.int8).reshape(shape_tuple)
        mfrac = mass_fraction(voxel_arr)

        if positive_if == 'low_mass':
            label_val = 1 if mfrac <= cutoff else 0
        elif positive_if == 'high_mass':
            label_val = 1 if mfrac >= cutoff else 0
        else:
            raise ValueError("positive_if must be 'low_mass' or 'high_mass'")

        cond_vec, aux = build_condition_vector(
            bc_arr=bcs[idx],
            load_arr=loads[idx],
            conditioning_spec=conditioning_spec,
            spatial_edges=spatial_edges,
            max_bc_points=max_bc_points,
        )

        if label_mode == "append_to_condition":
            cond_vec = np.concatenate([cond_vec, np.array([float(label_val)], dtype=np.float32)]).astype(np.float32)

        cond_str = make_condition_string(conditioning_spec, aux, spatial_edges)
        if label_mode == "append_to_condition":
            cond_str = f"label_{label_val}_" + cond_str

        sample_info = {
            "source_index": int(idx),
            "part_shape": np.asarray(shapes[idx], dtype=np.int32),
            "bc_count": int(aux["bc_count"]),
            "mass_fraction": float(mfrac),
        }

        results.append((voxel_arr, cond_vec, int(label_val), cond_str, sample_info))
        labels_list.append(label_val)

        if example_slices is None:
            example_slices = dict(aux["slices"])

    n_pos = int(sum(labels_list))
    n_neg = int(len(labels_list) - n_pos)

    return {
        "results": results,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "cutoff": float(cutoff),
        "cutoff_method": cutoff_method,
        "hist": hist,
        "hist_edges": hist_edges,
        "smooth": smooth,
        "hist_png_path": hist_png_path,
        "selected": selected,
        "indices": indices,
        "max_bc_points": max_bc_points,
        "min_bc_points": min_bc_points,
        "unique_bc_counts": unique_bc_counts,
        "spatial_edges": spatial_edges,
        "conditioning_spec": conditioning_spec,
        "conditioning_spec_str": spec_str,
        "conditioning_spec_slug": spec_slug,
        "example_slices": example_slices if example_slices is not None else {},
        "label_mode": label_mode,
    }


def save_dataset_and_meta(
    payload,
    outdir,
    shape_tuple,
    positive_if,
    rng_seed,
    mass_quantile,
    n_spatial_bins,
    file_prefix,
):
    results = payload["results"]
    cutoff = payload["cutoff"]
    spec_slug = payload["conditioning_spec_slug"]
    label_mode = payload["label_mode"]

    fname = (
        f"{len(results)}_{file_prefix}_"
        f"{shape_tuple[0]}x{shape_tuple[1]}x{shape_tuple[2]}_"
        f"{spec_slug}_"
        f"{positive_if}_thr{cutoff:.6f}.npy"
    )
    data_path = os.path.join(outdir, fname)
    np.save(data_path, np.array(results, dtype=object))

    meta_fname = fname.replace(".npy", "_meta.npz")
    meta_path = os.path.join(outdir, meta_fname)

    np.savez(
        meta_path,
        mass_cutoff_value=float(payload["cutoff"]),
        mass_cutoff_method=payload["cutoff_method"],
        positive_if=str(positive_if),
        conditioning_spec_json=json.dumps(payload["conditioning_spec"]),
        conditioning_spec_str=payload["conditioning_spec_str"],
        bc_locations_mode=str(payload["conditioning_spec"]["bc_locations"]),
        bc_dofs_mode=str(payload["conditioning_spec"]["bc_dofs"]),
        load_location_mode=str(payload["conditioning_spec"]["load_location"]),
        load_direction_mode=str(payload["conditioning_spec"]["load_direction"]),
        label_mode=str(label_mode),
        spatial_bin_edges_x=np.asarray(payload["spatial_edges"][0], dtype=np.float64),
        spatial_bin_edges_y=np.asarray(payload["spatial_edges"][1], dtype=np.float64),
        spatial_bin_edges_z=np.asarray(payload["spatial_edges"][2], dtype=np.float64),
        n_spatial_bins=int(n_spatial_bins),
        target_shape=np.array(shape_tuple, dtype=int),
        n_total_shape_matches=int(len(payload["indices"])),
        n_selected=int(len(payload["selected"])),
        rng_seed=int(rng_seed),
        bc_count_min=int(payload["min_bc_points"]),
        bc_count_max=int(payload["max_bc_points"]),
        bc_count_unique=np.array(payload["unique_bc_counts"], dtype=int),
        cond_dim=int(len(results[0][1])) if len(results) > 0 else -1,
        cond_slices_json=json.dumps(payload["example_slices"]),
        mass_hist_counts=np.asarray(payload["hist"], dtype=np.int64),
        mass_hist_edges=np.asarray(payload["hist_edges"], dtype=np.float64),
        mass_hist_smooth=np.asarray(payload["smooth"], dtype=np.float64),
        mass_hist_png=str(payload["hist_png_path"]),
        mass_quantile=np.nan if mass_quantile is None else float(mass_quantile),
        mass_threshold_source='user_quantile' if mass_quantile is not None else 'auto_valley',
    )

    print(f"Saved {len(results)} samples")
    print(f"Data path: {data_path}")
    print(f"Meta path: {meta_path}")
    if len(results) > 0:
        print(f"Condition dim: {len(results[0][1])}")

    return data_path, meta_path