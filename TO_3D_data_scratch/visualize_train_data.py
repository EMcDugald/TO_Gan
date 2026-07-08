import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt


def parse_bc_points(bc_arr):
    bc = np.asarray(bc_arr, dtype=np.float64)
    if bc.ndim != 2 or bc.shape[1] < 3:
        raise ValueError(f"Unexpected BC array shape: {bc.shape}")
    return bc[:, :3].astype(np.float32)


def parse_bc_dofs(bc_arr):
    bc = np.asarray(bc_arr, dtype=np.float64)
    if bc.ndim != 2 or bc.shape[1] < 6:
        raise ValueError(f"Unexpected BC array shape for DOFs: {bc.shape}")
    return bc[:, 3:6].astype(np.float32)


def parse_load_arr(load_arr):
    ld = np.asarray(load_arr, dtype=np.float64)
    if ld.ndim == 3:
        ld = ld[0]
    if ld.ndim != 2 or ld.shape[0] < 1 or ld.shape[1] < 6:
        raise ValueError(f"Unexpected load array shape: {ld.shape}")
    return ld[0, :3].astype(np.float32), ld[0, 3:6].astype(np.float32)


def bin_center_from_edges(edges, idx):
    idx = int(idx)
    if idx < 0 or idx >= len(edges) - 1:
        return None
    return float(0.5 * (edges[idx] + edges[idx + 1]))


def build_bc_bins_for_plot(cond_vec, cond_slices, bc_count):
    if "bc_bins" not in cond_slices:
        return None
    s, e = cond_slices["bc_bins"]
    flat = np.asarray(cond_vec[s:e], dtype=np.float64)
    max_bc_points = len(flat) // 3
    keep = min(int(bc_count), max_bc_points)
    out = []
    for i in range(keep):
        out.append((int(flat[3 * i + 0]), int(flat[3 * i + 1]), int(flat[3 * i + 2])))
    return out


def build_load_bins_for_plot(cond_vec, cond_slices):
    if "load_bins" not in cond_slices:
        return None
    s, e = cond_slices["load_bins"]
    vals = np.asarray(cond_vec[s:e], dtype=np.float64)
    if len(vals) < 3:
        return None
    return (int(vals[0]), int(vals[1]), int(vals[2]))


def draw_coarse_boxes(ax, bc_bins, spatial_edges, shape):
    if bc_bins is None or spatial_edges is None:
        return
    nx, ny, nz = shape
    x_edges, y_edges, z_edges = spatial_edges
    colors = ["red", "darkred", "firebrick", "indianred", "maroon", "salmon"]
    for i, (bx, by, bz) in enumerate(bc_bins):
        if bx < 0 or by < 0 or bz < 0:
            continue
        if bx >= len(x_edges) - 1 or by >= len(y_edges) - 1 or bz >= len(z_edges) - 1:
            continue
        x0, x1 = x_edges[bx] * (nx - 1), x_edges[bx + 1] * (nx - 1)
        y0, y1 = y_edges[by] * (ny - 1), y_edges[by + 1] * (ny - 1)
        z0, z1 = z_edges[bz] * (nz - 1), z_edges[bz + 1] * (nz - 1)
        c = colors[i % len(colors)]
        lines = [
            ([x0, x1], [y0, y0], [z0, z0]), ([x0, x1], [y1, y1], [z0, z0]),
            ([x0, x1], [y0, y0], [z1, z1]), ([x0, x1], [y1, y1], [z1, z1]),
            ([x0, x0], [y0, y1], [z0, z0]), ([x1, x1], [y0, y1], [z0, z0]),
            ([x0, x0], [y0, y1], [z1, z1]), ([x1, x1], [y0, y1], [z1, z1]),
            ([x0, x0], [y0, y0], [z0, z1]), ([x1, x1], [y0, y0], [z0, z1]),
            ([x0, x0], [y1, y1], [z0, z1]), ([x1, x1], [y1, y1], [z0, z1]),
        ]
        for xs, ys, zs in lines:
            ax.plot(xs, ys, zs, color=c, linestyle="--", linewidth=1.2, alpha=0.8)


def plot_voxel_with_overlays(
    voxel_arr,
    bc_pts,
    load_pt,
    load_vec,
    bc_bins=None,
    load_bins=None,
    spatial_edges=None,
    title="",
    save_path="plot.png",
):
    voxel_arr = np.asarray(voxel_arr)
    nx, ny, nz = voxel_arr.shape

    bc_plot = None
    if bc_pts is not None:
        bc_plot = np.column_stack([
            bc_pts[:, 0] * (nx - 1),
            bc_pts[:, 1] * (ny - 1),
            bc_pts[:, 2] * (nz - 1),
        ])

    load_plot = None
    if load_pt is not None:
        load_plot = np.array([
            load_pt[0] * (nx - 1),
            load_pt[1] * (ny - 1),
            load_pt[2] * (nz - 1),
        ], dtype=np.float64)

    fig = plt.figure(figsize=(18, 6))
    views = [(25, 35, "View 1"), (25, 125, "View 2"), (65, 35, "View 3")]

    for k, (elev, azim, subtitle) in enumerate(views, start=1):
        ax = fig.add_subplot(1, 3, k, projection="3d")
        ax.voxels(voxel_arr > 0, edgecolor="k", linewidth=0.15, alpha=0.72)
        ax.set_xlim(0, nx)
        ax.set_ylim(0, ny)
        ax.set_zlim(0, nz)

        if bc_plot is not None:
            ax.scatter(
                bc_plot[:, 0], bc_plot[:, 1], bc_plot[:, 2],
                c="red", s=36, marker="o", edgecolors="white", linewidths=0.6,
                depthshade=False, label="BC"
            )
            for i, p in enumerate(bc_plot):
                ax.text(p[0], p[1], p[2], f"BC{i}", color="red", fontsize=8)

        if bc_bins is not None and spatial_edges is not None:
            draw_coarse_boxes(ax, bc_bins, spatial_edges, (nx, ny, nz))

        if load_plot is not None:
            ax.scatter(
                [load_plot[0]], [load_plot[1]], [load_plot[2]],
                c="dodgerblue", s=64, marker="^", edgecolors="white", linewidths=0.7,
                depthshade=False, label="Load point"
            )
            if load_vec is not None:
                scale = max(nx, ny, nz) * 0.18
                ax.quiver(
                    load_plot[0], load_plot[1], load_plot[2],
                    load_vec[0], load_vec[1], load_vec[2],
                    color="dodgerblue", linewidth=2.0, length=scale, normalize=True,
                )
            ax.text(load_plot[0], load_plot[1], load_plot[2], "Load", color="dodgerblue", fontsize=8)

        if load_bins is not None and spatial_edges is not None:
            x_edges, y_edges, z_edges = spatial_edges
            cx = bin_center_from_edges(x_edges, load_bins[0])
            cy = bin_center_from_edges(y_edges, load_bins[1])
            cz = bin_center_from_edges(z_edges, load_bins[2])
            if cx is not None and cy is not None and cz is not None:
                ax.scatter(
                    [cx * (nx - 1)], [cy * (ny - 1)], [cz * (nz - 1)],
                    c="cyan", s=50, marker="x", linewidths=2.0,
                    depthshade=False, label="Load bin center"
                )

        ax.view_init(elev=elev, azim=azim)
        ax.set_title(subtitle)
        ax.set_box_aspect((nx, ny, nz))
        ax.set_axis_off()

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=220)
    plt.close(fig)


def decode_condition_fields(cond_vec, cond_slices):
    decoded = {}
    for key in [
        "bc_points", "bc_mask", "bc_count", "bc_dofs", "load_point", "load_bins",
        "load_dir", "bc_x_range", "bc_y_range", "bc_z_range", "bc_bins"
    ]:
        if key in cond_slices:
            s, e = cond_slices[key]
            vals = np.asarray(cond_vec[s:e])
            if key == "bc_count":
                decoded[f"cond_{key}"] = float(vals[0]) if len(vals) else None
            else:
                decoded[f"cond_{key}"] = vals.tolist()
    return decoded


def select_indices(n_total, sample_idx, n_random, rng_seed):
    if sample_idx is not None:
        if sample_idx < 0 or sample_idx >= n_total:
            raise ValueError(f"sample-idx {sample_idx} out of range for {n_total} samples")
        return [int(sample_idx)], {"selection_mode": "single", "rng_seed": None}
    if n_random is None or n_random <= 0:
        raise ValueError("When --sample-idx is not used, --n-random must be positive")
    n_pick = min(int(n_random), int(n_total))
    rng = np.random.default_rng(rng_seed)
    picks = sorted(rng.choice(n_total, size=n_pick, replace=False).tolist())
    return picks, {"selection_mode": "random", "rng_seed": int(rng_seed)}


def main():
    parser = argparse.ArgumentParser(
        description="Visualize GAN/diffusion train-data samples with BC/load overlays and JSON condition summaries."
    )
    parser.add_argument("--mode", type=str, choices=["gan", "diffusion"], required=True,
                        help="Which dataset type: gan (label separate) or diffusion (label appended).")
    parser.add_argument("--train-data-npy", type=str, required=True,
                        help="Path to the generated train-data .npy file.")
    parser.add_argument("--train-data-meta", type=str, required=True,
                        help="Path to the corresponding _meta.npz file.")
    parser.add_argument("--raw-data-root", type=str, required=True,
                        help="Root containing topologies.npy, shapes.npy, boundary_conditions.npy, loads.npy.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sample-idx", type=int, default=None,
                       help="Visualize exactly one prescribed sample index from the train-data array.")
    group.add_argument("--n-random", type=int, default=None,
                       help="Visualize N randomly chosen samples from the train-data array.")
    parser.add_argument("--rng-seed", type=int, default=0,
                        help="Random seed used when --n-random is provided.")
    parser.add_argument("--outdir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    x = np.load(args.train_data_npy, allow_pickle=True)
    m = np.load(args.train_data_meta, allow_pickle=True)

    bcs = np.load(os.path.join(args.raw_data_root, "boundary_conditions.npy"), allow_pickle=True)
    loads = np.load(os.path.join(args.raw_data_root, "loads.npy"), allow_pickle=True)

    conditioning_spec = json.loads(m["conditioning_spec_json"].item())
    cond_slices = json.loads(m["cond_slices_json"].item())

    bc_loc_mode = conditioning_spec["bc_locations"]
    load_loc_mode = conditioning_spec["load_location"]

    spatial_edges = None
    if bc_loc_mode == "coarse" or load_loc_mode == "coarse":
        spatial_edges = (
            np.asarray(m["spatial_bin_edges_x"], dtype=np.float64),
            np.asarray(m["spatial_bin_edges_y"], dtype=np.float64),
            np.asarray(m["spatial_bin_edges_z"], dtype=np.float64),
        )

    selected_indices, selection_meta = select_indices(len(x), args.sample_idx, args.n_random, args.rng_seed)

    run_summary = {
        "mode": args.mode,
        "train_data_npy": args.train_data_npy,
        "train_data_meta": args.train_data_meta,
        "raw_data_root": args.raw_data_root,
        "selection": {
            **selection_meta,
            "requested_sample_idx": args.sample_idx,
            "requested_n_random": args.n_random,
            "selected_sample_indices": [int(i) for i in selected_indices],
            "num_selected": len(selected_indices),
        },
        "dataset_metadata": {
            "label_mode": m["label_mode"].item(),
            "cond_dim": int(m["cond_dim"]),
            "conditioning_spec": conditioning_spec,
            "mass_cutoff_value": float(m["mass_cutoff_value"]),
            "mass_cutoff_method": m["mass_cutoff_method"].item(),
            "mass_threshold_source": m["mass_threshold_source"].item(),
            "mass_quantile": None if np.isnan(m["mass_quantile"]).item() else float(m["mass_quantile"]),
        },
        "parts": []
    }

    for sample_idx in selected_indices:
        sample = x[sample_idx]
        voxel_arr = sample[0]
        cond_vec = np.asarray(sample[1], dtype=np.float64)
        label_val = int(sample[2])
        cond_str = str(sample[3])
        sample_info = sample[4]

        src_idx = int(sample_info["source_index"])
        raw_bc = bcs[src_idx]
        raw_load = loads[src_idx]
        raw_bc_pts = parse_bc_points(raw_bc)
        raw_bc_dofs = parse_bc_dofs(raw_bc)
        raw_load_pt, raw_load_vec = parse_load_arr(raw_load)

        bc_count = int(sample_info["bc_count"])
        bc_bins_for_plot = None
        if bc_loc_mode == "coarse" and spatial_edges is not None:
            bc_bins_for_plot = build_bc_bins_for_plot(cond_vec, cond_slices, bc_count)

        load_bins_for_plot = None
        if load_loc_mode == "coarse" and spatial_edges is not None:
            load_bins_for_plot = build_load_bins_for_plot(cond_vec, cond_slices)

        base = f"sample_{args.mode}_idx{sample_idx:05d}"
        out_png = os.path.join(args.outdir, f"{base}.png")
        out_json = os.path.join(args.outdir, f"{base}.json")

        title = f"{args.mode} sample {sample_idx}, src={src_idx}"
        plot_voxel_with_overlays(
            voxel_arr=voxel_arr,
            bc_pts=raw_bc_pts,
            load_pt=raw_load_pt,
            load_vec=raw_load_vec,
            bc_bins=bc_bins_for_plot,
            load_bins=load_bins_for_plot,
            spatial_edges=spatial_edges,
            title=title,
            save_path=out_png,
        )

        part_record = {
            "sample_array_index": int(sample_idx),
            "source_index": src_idx,
            "part_shape": np.asarray(sample_info["part_shape"]).tolist(),
            "bc_count": bc_count,
            "mass_fraction": float(sample_info["mass_fraction"]),
            "conditioning_spec": conditioning_spec,
            "label_mode": m["label_mode"].item(),
            "label_value": label_val,
            "cond_dim": int(m["cond_dim"]),
            "cond_str": cond_str,
            "raw_bc_points": raw_bc_pts.tolist(),
            "raw_bc_dofs": raw_bc_dofs.tolist(),
            "raw_load_point": raw_load_pt.tolist(),
            "raw_load_dir": raw_load_vec.tolist(),
            "cond_slices": cond_slices,
            "cond_vector": cond_vec.tolist(),
            "plot_png": os.path.basename(out_png),
        }
        part_record.update(decode_condition_fields(cond_vec, cond_slices))

        if args.mode == "diffusion" and m["label_mode"].item() == "append_to_condition":
            part_record["cond_label_bit"] = float(cond_vec[-1])

        if bc_bins_for_plot is not None:
            part_record["plot_bc_bins"] = [list(t) for t in bc_bins_for_plot]
        if load_bins_for_plot is not None:
            part_record["plot_load_bins"] = list(load_bins_for_plot)
        if spatial_edges is not None:
            part_record["spatial_bin_edges"] = {
                "x": spatial_edges[0].tolist(),
                "y": spatial_edges[1].tolist(),
                "z": spatial_edges[2].tolist(),
            }

        with open(out_json, "w") as f:
            json.dump(part_record, f, indent=2)

        run_summary["parts"].append({
            "sample_array_index": int(sample_idx),
            "source_index": src_idx,
            "plot_png": os.path.basename(out_png),
            "part_json": os.path.basename(out_json),
            "label_value": label_val,
            "mass_fraction": float(sample_info["mass_fraction"]),
            "bc_count": bc_count,
            "cond_str": cond_str,
        })

        print(f"Saved plot to {out_png}")
        print(f"Saved JSON to {out_json}")

    summary_json = os.path.join(args.outdir, "run_summary.json")
    with open(summary_json, "w") as f:
        json.dump(run_summary, f, indent=2)

    print(f"Saved run summary to {summary_json}")


if __name__ == "__main__":
    main()