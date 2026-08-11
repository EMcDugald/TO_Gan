import os
import json
import numpy as np
import matplotlib.pyplot as plt


# -------------------------------------------------------------------------
# Basic constants
# -------------------------------------------------------------------------

N_SPATIAL_BINS = 10


# -------------------------------------------------------------------------
# Helpers copied/adapted from existing utils
# -------------------------------------------------------------------------

def mass_fraction(voxel_arr):
    """Binary occupancy fraction (still useful for diagnostics)."""
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


def load_manufacturability_scalar(csv_path, scalar_col, part_col="part"):
    """
    Reads a summary CSV (e.g. summary.csv / summary_RI_HS.csv) that begins
    with '#'-prefixed comment lines, then a header row, then data rows.
    Returns {part_index (int): value (float)} for every row with a finite,
    parseable value in `scalar_col`. Rows with missing/NaN/unparseable
    values are skipped (left unlabeled), not zero-filled.
    """
    import csv as csv_mod
    values = {}
    with open(csv_path, newline="") as f:
        content_lines = (line for line in f if not line.lstrip().startswith("#"))
        reader = csv_mod.DictReader(content_lines)
        if reader.fieldnames is None or scalar_col not in reader.fieldnames:
            raise ValueError(
                f"Column {scalar_col!r} not found in {csv_path}; "
                f"available columns: {reader.fieldnames}"
            )
        for row in reader:
            raw = (row.get(scalar_col) or "").strip()
            if raw == "" or raw.lower() == "nan":
                continue
            try:
                val = float(raw)
            except ValueError:
                continue
            if not np.isfinite(val):
                continue
            try:
                part_idx = int(float(row[part_col]))
            except (TypeError, ValueError):
                continue
            values[part_idx] = val
    return values


def compute_manufacturability_stats(values_by_idx, percentile_threshold=80.0):
    """
    values_by_idx: {part_index: scalar_value}, the FULL labeled universe
    (i.e. computed before shape filtering / n_samples subsampling, so the
    split is a stable property of the label itself).

    Returns dict with:
      threshold_value    : the percentile_threshold-th percentile of the
                            scalar's distribution (values <= this are
                            "positive"/manufacturable; assumes lower =
                            better, e.g. lower deformation).
      percentile_rank_fn  : callable, raw scalar value -> empirical CDF
                            rank in (0, 1], relative to this population.
      n_labeled_total     : population size the stats were computed over.
    """
    idxs = sorted(values_by_idx.keys())
    if not idxs:
        raise ValueError("No labeled parts found -- cannot compute manufacturability stats")
    vals = np.array([values_by_idx[i] for i in idxs], dtype=np.float64)
    sorted_vals = np.sort(vals)
    threshold_value = float(np.percentile(sorted_vals, percentile_threshold))

    def percentile_rank(x):
        return float(np.searchsorted(sorted_vals, x, side="right")) / float(len(sorted_vals))

    return {
        "threshold_value": threshold_value,
        "percentile_rank_fn": percentile_rank,
        "n_labeled_total": len(idxs),
    }


def append_manufacturability_to_cond(cond_vec, slices, next_pos, manuf_mode,
                                     has_label, value_norm, label_pm1):
    """
    Appends 2 scalars to cond_vec: [manuf_value or manuf_label, has_label].
    has_label=0.0 + value/label=0.0 marks a genuinely unlabeled sample,
    distinguishable from a real value/label of exactly 0 via the mask bit.
    """
    if manuf_mode == "omit":
        return cond_vec, slices, next_pos
    if manuf_mode == "scalar":
        add_vals = np.array([value_norm, has_label], dtype=np.float32)
        names = ("manuf_value", "manuf_has_label")
    elif manuf_mode == "binary":
        add_vals = np.array([label_pm1, has_label], dtype=np.float32)
        names = ("manuf_label", "manuf_has_label")
    else:
        raise ValueError(f"Unknown manufacturability mode: {manuf_mode}")
    cond_vec = np.concatenate([cond_vec, add_vals]).astype(np.float32)
    slices[names[0]] = (next_pos, next_pos + 1)
    slices[names[1]] = (next_pos + 1, next_pos + 2)
    next_pos += 2
    return cond_vec, slices, next_pos


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


# -------------------------------------------------------------------------
# Conditioning spec parsing
# -------------------------------------------------------------------------

def parse_conditioning_spec(spec_str):
    tokens = [tok.strip() for tok in spec_str.split(",") if tok.strip()]
    spec = {
        "bc_locations": "coarse",
        "bc_dofs": "omit",
        "load_location": "coarse",
        "load_direction": "omit",
        "manufacturability": "omit",
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
        f"manufacturability={spec['manufacturability']}",
    ])


# -------------------------------------------------------------------------
# Condition vector construction (BC/load -> cond_vec)
# -------------------------------------------------------------------------

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

    # BC locations
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

    # BC DOFs
    if bc_dof_mode == "fine":
        add_part("bc_dofs", bc_dofs_pad.reshape(-1))
    elif bc_dof_mode != "omit":
        raise ValueError(f"Unknown bc_dofs mode: {bc_dof_mode}")

    # Load location
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

    # Load direction
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


# -------------------------------------------------------------------------
# Dataset generation for neural fields (mixed shapes)
# -------------------------------------------------------------------------

def generate_dataset_mixed_shapes(
    shapes_filter,
    n_samples,
    topo,
    shapes,
    bcs,
    loads,
    vfs,
    outdir,
    conditioning_spec,
    rng_seed,
    n_spatial_bins,
    vf_mode="append_to_condition",
    include_shape_in_cond=True,
    manufacturability_csv=None,
    manuf_scalar_column="deformation_p99",
    manuf_percentile_threshold=80.0,
    n_unlabeled_extra=0,
):
    """
    Build a dataset for neural-field training with mixed part shapes.
    Each entry: (voxel_arr, cond_vec, label_dummy, cond_str, sample_info).
    - voxel_arr: binary/int8 voxel grid with shape (Nx, Ny, Nz).
    - cond_vec: BC/load cond + optional VF + optional normalized shape tuple
      + optional manufacturability [value_or_label, has_label].
    - label_dummy: integer (0) placeholder (unrelated to manufacturability;
      still supervised point-wise, not by mass class).
    - cond_str: human-readable conditioning string.
    - sample_info: dict with part_shape, mass_fraction, volume_fraction,
      and (if manufacturability requested) manuf_has_label/manuf_raw_value/
      manuf_percentile_rank/manuf_label.

    manufacturability_csv/manuf_scalar_column/manuf_percentile_threshold are
    only used when conditioning_spec["manufacturability"] != "omit". The
    percentile threshold/rank are computed over the FULL labeled CSV
    population (before shape filtering or n_samples subsampling), so the
    positive/negative split is stable across runs regardless of which
    shapes or how many samples this particular call selects.

    n_unlabeled_extra: number of additional shape-filtered parts with NO
    manufacturability label to draw into the pool alongside the labeled
    universe (0 -> labeled-only; >0 -> labeled + unlabeled mix).
    """

    all_shapes = [tuple(np.asarray(shp).tolist()) for shp in shapes]

    # Select indices based on optional shapes_filter
    if shapes_filter is None:
        indices = list(range(len(shapes)))
    else:
        allowed = set(tuple(s) for s in shapes_filter)
        indices = [i for i, shp in enumerate(all_shapes) if shp in allowed]

    if not indices:
        raise ValueError("No samples found for the requested shapes")

    vfs = np.asarray(vfs, dtype=np.float64).reshape(-1)

    unique_shapes = sorted(set(all_shapes[i] for i in indices))
    print(f"Found {len(indices)} total entries with shapes {unique_shapes}")

    # Gather mass + BC/load info for spatial edges
    all_mass_fracs = []
    all_bc_pts = []
    all_load_pts = []
    bc_counts = []

    for i in indices:
        shape_tuple = tuple(all_shapes[i])
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

    os.makedirs(outdir, exist_ok=True)

    spec_str = conditioning_spec_to_string(conditioning_spec)
    spec_slug = (
        f"bcLoc-{conditioning_spec['bc_locations']}_"
        f"bcDofs-{conditioning_spec['bc_dofs']}_"
        f"loadLoc-{conditioning_spec['load_location']}_"
        f"loadDir-{conditioning_spec['load_direction']}"
    )

    # ---- manufacturability: load labels + compute global stats ----
    manuf_mode = conditioning_spec.get("manufacturability", "omit")
    labeled_map = {}
    manuf_stats = None
    if manuf_mode != "omit":
        if manufacturability_csv is None:
            raise ValueError(
                "manufacturability_csv is required when "
                "conditioning_spec['manufacturability'] != 'omit'"
            )
        labeled_map = load_manufacturability_scalar(manufacturability_csv, manuf_scalar_column)
        manuf_stats = compute_manufacturability_stats(labeled_map, manuf_percentile_threshold)
        print(f"Manufacturability: loaded {manuf_stats['n_labeled_total']} labeled parts "
              f"from {manufacturability_csv} (column={manuf_scalar_column}); "
              f"{manuf_percentile_threshold:.1f}th percentile threshold = "
              f"{manuf_stats['threshold_value']:.6g} (<=threshold -> positive)")
        spec_slug += f"_manuf-{manuf_mode}"

    # ---- build the sampling pool ----
    rng = np.random.default_rng(rng_seed)
    if manuf_mode == "omit":
        pool = list(indices)
    else:
        labeled_universe = sorted(set(labeled_map.keys()) & set(indices))
        if not labeled_universe:
            raise ValueError(
                "No labeled parts (from manufacturability_csv) found among "
                "the shape-filtered indices"
            )
        pool = list(labeled_universe)
        if n_unlabeled_extra > 0:
            unlabeled_candidates = sorted(set(indices) - set(labeled_map.keys()))
            n_draw = min(n_unlabeled_extra, len(unlabeled_candidates))
            if n_draw < n_unlabeled_extra:
                print(f"WARNING: requested {n_unlabeled_extra} unlabeled extras but only "
                      f"{len(unlabeled_candidates)} shape-filtered parts lack a "
                      f"manufacturability label; using {n_draw}.")
            drawn_unlabeled = (rng.choice(unlabeled_candidates, size=n_draw, replace=False).tolist()
                               if n_draw > 0 else [])
            pool = pool + drawn_unlabeled
        print(f"Manufacturability pool: {len(labeled_universe)} labeled + "
              f"{len(pool) - len(labeled_universe)} unlabeled = {len(pool)} total "
              f"(shape-filtered universe = {len(indices)})")

    if len(pool) <= n_samples:
        selected = pool
    else:
        selected = list(rng.choice(pool, size=n_samples, replace=False))

    results = []
    example_slices = None

    for idx in selected:
        shape_tuple = tuple(all_shapes[idx])
        voxel_arr = np.asarray(topo[idx]).astype(np.int8).reshape(shape_tuple)
        mfrac = mass_fraction(voxel_arr)
        vf_val = float(vfs[idx])

        cond_vec, aux = build_condition_vector(
            bc_arr=bcs[idx],
            load_arr=loads[idx],
            conditioning_spec=conditioning_spec,
            spatial_edges=spatial_edges,
            max_bc_points=max_bc_points,
        )

        slices = dict(aux["slices"])
        next_pos = len(cond_vec)

        # Append volume fraction as scalar conditioning
        if vf_mode == "append_to_condition":
            cond_vec = np.concatenate(
                [cond_vec, np.array([vf_val], dtype=np.float32)]
            ).astype(np.float32)
            slices["volume_fraction"] = (next_pos, next_pos + 1)
            next_pos += 1
        elif vf_mode != "omit":
            raise ValueError("vf_mode must be 'omit' or 'append_to_condition'")

        # Append normalized shape tuple (Nx, Ny, Nz) as conditioning
        if include_shape_in_cond:
            shape_arr = np.asarray(shape_tuple, dtype=np.float32)
            max_dim = float(shape_arr.max())
            shape_norm = shape_arr / max_dim if max_dim > 0 else shape_arr
            cond_vec = np.concatenate([cond_vec, shape_norm]).astype(np.float32)
            slices["shape_tuple"] = (next_pos, next_pos + 3)
            next_pos += 3

        # Manufacturability conditioning
        has_label = idx in labeled_map
        if has_label:
            raw_val = float(labeled_map[idx])
            value_norm = manuf_stats["percentile_rank_fn"](raw_val)
            label_pm1 = 1.0 if raw_val <= manuf_stats["threshold_value"] else -1.0
        else:
            raw_val = None
            value_norm = 0.0
            label_pm1 = 0.0

        cond_vec, slices, next_pos = append_manufacturability_to_cond(
            cond_vec, slices, next_pos, manuf_mode,
            has_label=float(has_label), value_norm=value_norm, label_pm1=label_pm1,
        )

        cond_str = make_condition_string(conditioning_spec, aux, spatial_edges)
        cond_str = f"{cond_str}_vf_{vf_val:.6f}"
        cond_str = f"{cond_str}_shape_{shape_tuple[0]}x{shape_tuple[1]}x{shape_tuple[2]}"
        if manuf_mode != "omit":
            if has_label:
                cond_str = (f"{cond_str}_manuf_{manuf_scalar_column}_{raw_val:.6f}"
                           f"_pr_{value_norm:.4f}_lbl_{int(label_pm1)}")
            else:
                cond_str = f"{cond_str}_manuf_unlabeled"

        sample_info = {
            "source_index": int(idx),
            "part_shape": np.asarray(shape_tuple, dtype=np.int32),
            "bc_count": int(aux["bc_count"]),
            "mass_fraction": float(mfrac),
            "volume_fraction": vf_val,
            "manuf_has_label": bool(has_label) if manuf_mode != "omit" else None,
            "manuf_raw_value": raw_val if manuf_mode != "omit" else None,
            "manuf_percentile_rank": value_norm if manuf_mode != "omit" else None,
            "manuf_label": label_pm1 if manuf_mode != "omit" else None,
        }

        # Dummy label = 0 (neural field is supervised point-wise, not via mass class)
        results.append((voxel_arr, cond_vec, int(0), cond_str, sample_info))

        if example_slices is None:
            example_slices = dict(slices)

    return {
        "results": results,
        "hist": None,
        "hist_edges": None,
        "smooth": None,
        "hist_png_path": None,
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
        "vf_mode": vf_mode,
        "manufacturability_mode": manuf_mode,
        "manuf_scalar_column": manuf_scalar_column if manuf_mode != "omit" else None,
        "manuf_percentile_threshold": manuf_percentile_threshold if manuf_mode != "omit" else None,
        "manuf_threshold_value": manuf_stats["threshold_value"] if manuf_stats is not None else None,
        "manuf_n_labeled_total": manuf_stats["n_labeled_total"] if manuf_stats is not None else 0,
        "manuf_n_labeled_used": sum(1 for i in selected if i in labeled_map),
        "manuf_n_unlabeled_used": sum(1 for i in selected if i not in labeled_map),
    }


# -------------------------------------------------------------------------
# Saving dataset + meta for neural fields
# -------------------------------------------------------------------------

def save_dataset_and_meta_neural(
    payload,
    outdir,
    rng_seed,
    n_spatial_bins,
    file_prefix,
):
    results = payload["results"]
    spec_slug = payload["conditioning_spec_slug"]
    vf_mode = payload["vf_mode"]

    # Summarize shapes for naming
    part_shapes = [tuple(r[4]["part_shape"].tolist()) for r in results]
    unique_shapes = sorted(set(part_shapes))
    shapes_str = "_".join(f"{sx}x{sy}x{sz}" for (sx, sy, sz) in unique_shapes)

    fname = (
        f"{len(results)}_{file_prefix}_"
        f"shapes-{shapes_str}_"
        f"{spec_slug}_"
        f"vfmode-{vf_mode}.npy"
    )
    data_path = os.path.join(outdir, fname)
    np.save(data_path, np.array(results, dtype=object))

    meta_fname = fname.replace(".npy", "_meta.npz")
    meta_path = os.path.join(outdir, meta_fname)

    np.savez(
        meta_path,
        conditioning_spec_json=json.dumps(payload["conditioning_spec"]),
        conditioning_spec_str=payload["conditioning_spec_str"],
        bc_locations_mode=str(payload["conditioning_spec"]["bc_locations"]),
        bc_dofs_mode=str(payload["conditioning_spec"]["bc_dofs"]),
        load_location_mode=str(payload["conditioning_spec"]["load_location"]),
        load_direction_mode=str(payload["conditioning_spec"]["load_direction"]),
        vf_mode=str(vf_mode),
        spatial_bin_edges_x=np.asarray(payload["spatial_edges"][0], dtype=np.float64),
        spatial_bin_edges_y=np.asarray(payload["spatial_edges"][1], dtype=np.float64),
        spatial_bin_edges_z=np.asarray(payload["spatial_edges"][2], dtype=np.float64),
        n_spatial_bins=int(n_spatial_bins),
        unique_shapes=np.asarray(unique_shapes, dtype=int),
        cond_slices_json=json.dumps(payload["example_slices"]),
        cond_dim=int(len(results[0][1])) if len(results) > 0 else -1,
        rng_seed=int(rng_seed),
        manufacturability_mode=str(payload.get("manufacturability_mode", "omit")),
        manuf_scalar_column=str(payload.get("manuf_scalar_column") or ""),
        manuf_percentile_threshold=float(payload.get("manuf_percentile_threshold")
                                         if payload.get("manuf_percentile_threshold") is not None else -1.0),
        manuf_threshold_value=float(payload.get("manuf_threshold_value")
                                    if payload.get("manuf_threshold_value") is not None else np.nan),
        manuf_n_labeled_total=int(payload.get("manuf_n_labeled_total") or 0),
        manuf_n_labeled_used=int(payload.get("manuf_n_labeled_used") or 0),
        manuf_n_unlabeled_used=int(payload.get("manuf_n_unlabeled_used") or 0),
    )

    print(f"Saved {len(results)} samples")
    print(f"Data path: {data_path}")
    print(f"Meta path: {meta_path}")
    if len(results) > 0:
        print(f"Condition dim: {len(results[0][1])}")
    if payload.get("manufacturability_mode", "omit") != "omit":
        print(f"Manufacturability mode: {payload['manufacturability_mode']} "
              f"(column={payload['manuf_scalar_column']}, "
              f"threshold={payload['manuf_threshold_value']:.6g} "
              f"@ p{payload['manuf_percentile_threshold']:.1f}); "
              f"{payload['manuf_n_labeled_used']} labeled / "
              f"{payload['manuf_n_unlabeled_used']} unlabeled in this dataset")

    return data_path, meta_path