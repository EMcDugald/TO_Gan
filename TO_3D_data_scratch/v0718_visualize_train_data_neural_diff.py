import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt


def bin_center_from_edges(edges, idx):
    idx = int(idx)
    if idx < 0 or idx >= len(edges) - 1:
        return None
    return float(0.5 * (edges[idx] + edges[idx + 1]))


# def plot_voxel_with_overlays(
#     voxel_arr,
#     bc_pts,
#     load_pt,
#     load_vec,
#     title="",
#     save_path="plot.png",
# ):
#     voxel_arr = np.asarray(voxel_arr)
#     nx, ny, nz = voxel_arr.shape

#     bc_plot = None
#     if bc_pts is not None and len(bc_pts) > 0:
#         bc_pts = np.asarray(bc_pts, dtype=np.float64)
#         bc_plot = np.column_stack([
#             bc_pts[:, 0] * (nx - 1),
#             bc_pts[:, 1] * (ny - 1),
#             bc_pts[:, 2] * (nz - 1),
#         ])

#     load_plot = None
#     if load_pt is not None:
#         lp = np.asarray(load_pt, dtype=np.float64)
#         load_plot = np.array([
#             lp[0] * (nx - 1),
#             lp[1] * (ny - 1),
#             lp[2] * (nz - 1),
#         ], dtype=np.float64)

#     fig = plt.figure(figsize=(18, 6))
#     views = [(25, 35, "View 1"), (25, 125, "View 2"), (65, 35, "View 3")]

#     for k, (elev, azim, subtitle) in enumerate(views, start=1):
#         ax = fig.add_subplot(1, 3, k, projection="3d")
#         ax.voxels(voxel_arr > 0, edgecolor="k", linewidth=0.15, alpha=0.72)
#         ax.set_xlim(0, nx)
#         ax.set_ylim(0, ny)
#         ax.set_zlim(0, nz)

#         if bc_plot is not None:
#             ax.scatter(
#                 bc_plot[:, 0], bc_plot[:, 1], bc_plot[:, 2],
#                 c="red", s=36, marker="o", edgecolors="white", linewidths=0.6,
#                 depthshade=False, label="BC"
#             )
#             for i, p in enumerate(bc_plot):
#                 ax.text(p[0], p[1], p[2], f"BC{i}", color="red", fontsize=8)

#         if load_plot is not None:
#             ax.scatter(
#                 [load_plot[0]], [load_plot[1]], [load_plot[2]],
#                 c="dodgerblue", s=64, marker="^", edgecolors="white", linewidths=0.7,
#                 depthshade=False, label="Load point"
#             )
#             if load_vec is not None:
#                 lv = np.asarray(load_vec, dtype=np.float64)
#                 scale = max(nx, ny, nz) * 0.18
#                 ax.quiver(
#                     load_plot[0], load_plot[1], load_plot[2],
#                     lv[0], lv[1], lv[2],
#                     color="dodgerblue", linewidth=2.0, length=scale, normalize=True,
#                 )
#             ax.text(load_plot[0], load_plot[1], load_plot[2], "Load", color="dodgerblue", fontsize=8)

#         ax.view_init(elev=elev, azim=azim)
#         ax.set_title(subtitle)
#         ax.set_box_aspect((nx, ny, nz))
#         ax.set_axis_off()

#     fig.suptitle(title)
#     plt.tight_layout()
#     plt.savefig(save_path, bbox_inches="tight", dpi=220)
#     plt.close(fig)



def plot_voxel_with_overlays(
    voxel_arr,
    bc_pts,
    load_pt,
    load_vec,
    title="",
    save_path="plot.png",
):
    voxel_arr = np.asarray(voxel_arr)
    nx, ny, nz = voxel_arr.shape

    # NITO-style global length scale based on shape tuple
    c = float(max(nx, ny, nz))

    # --- BC points: NITO-normalized coords rescaled per axis ---
    bc_plot = None
    if bc_pts is not None and len(bc_pts) > 0:
        bc_pts = np.asarray(bc_pts, dtype=np.float64)
        bc_plot = np.column_stack([
            bc_pts[:, 0] * (c - 1),
            bc_pts[:, 1] * (c - 1),
            bc_pts[:, 2] * (c - 1),
        ])

    # --- Load point: same rescaling ---
    load_plot = None
    if load_pt is not None:
        lp = np.asarray(load_pt, dtype=np.float64)
        load_plot = np.array([
            # lp[0] * (nx / c) * (nx - 1),
            # lp[1] * (ny / c) * (ny - 1),
            # lp[2] * (nz / c) * (nz - 1),
            lp[0] * c,
            lp[1] * c,
            lp[2] * c,
        ], dtype=np.float64)

    fig = plt.figure(figsize=(18, 6))
    views = [(25, 35, "View 1"), (25, 125, "View 2"), (65, 35, "View 3")]

    for k, (elev, azim, subtitle) in enumerate(views, start=1):
        ax = fig.add_subplot(1, 3, k, projection="3d")
        ax.voxels(voxel_arr > 0, edgecolor="k", linewidth=0.15, alpha=0.72)
        ax.set_xlim(0, nx)
        ax.set_ylim(0, ny)
        ax.set_zlim(0, nz)

        # Plot BC points
        if bc_plot is not None:
            ax.scatter(
                bc_plot[:, 0], bc_plot[:, 1], bc_plot[:, 2],
                c="red", s=36, marker="o", edgecolors="white", linewidths=0.6,
                depthshade=False, label="BC"
            )
            for i, p in enumerate(bc_plot):
                ax.text(p[0], p[1], p[2], f"BC{i}", color="red", fontsize=8)

        # Plot load point + direction
        if load_plot is not None:
            ax.scatter(
                [load_plot[0]], [load_plot[1]], [load_plot[2]],
                c="dodgerblue", s=64, marker="^", edgecolors="white", linewidths=0.7,
                depthshade=False, label="Load point"
            )
            if load_vec is not None:
                lv = np.asarray(load_vec, dtype=np.float64)
                scale = max(nx, ny, nz) * 0.18
                ax.quiver(
                    load_plot[0], load_plot[1], load_plot[2],
                    lv[0], lv[1], lv[2],
                    color="dodgerblue", linewidth=2.0, length=scale, normalize=True,
                )
            ax.text(load_plot[0], load_plot[1], load_plot[2], "Load", color="dodgerblue", fontsize=8)

        ax.view_init(elev=elev, azim=azim)
        ax.set_title(subtitle)
        ax.set_box_aspect((nx, ny, nz))
        ax.set_axis_off()

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=220)
    plt.close(fig)


def decode_condition_fields(cond_vec, cond_slices):
    """
    Decode fine-mode fields from cond_vec using cond_slices:
    - bc_points
    - bc_mask
    - bc_count
    - bc_dofs
    - load_point
    - load_dir
    - volume_fraction
    - shape_tuple
    """
    decoded = {}

    # Helper to slice safely
    def get_slice(name):
        if name not in cond_slices:
            return None
        s, e = cond_slices[name]
        return np.asarray(cond_vec[s:e], dtype=np.float32)

    # BC points (fine)
    bc_points = get_slice("bc_points")
    bc_mask = get_slice("bc_mask")
    bc_count_arr = get_slice("bc_count")
    bc_dofs = get_slice("bc_dofs")

    if bc_points is not None and bc_mask is not None and bc_count_arr is not None:
        max_bc_points = len(bc_mask)
        bc_count = int(float(bc_count_arr[0]))
        bc_count = min(bc_count, max_bc_points)
        bc_points = bc_points.reshape(max_bc_points, 3)[:bc_count]
        decoded["bc_points"] = bc_points
        decoded["bc_mask"] = bc_mask
        decoded["bc_count"] = bc_count
    else:
        decoded["bc_points"] = None
        decoded["bc_mask"] = None
        decoded["bc_count"] = None

    if bc_dofs is not None:
        bc_dofs = bc_dofs.reshape(-1, 3)
        decoded["bc_dofs"] = bc_dofs
    else:
        decoded["bc_dofs"] = None

    # Load point / direction (fine)
    load_point = get_slice("load_point")
    if load_point is not None and len(load_point) >= 3:
        decoded["load_point"] = load_point[:3]
    else:
        decoded["load_point"] = None

    load_dir = get_slice("load_dir")
    if load_dir is not None and len(load_dir) >= 3:
        decoded["load_dir"] = load_dir[:3]
    else:
        decoded["load_dir"] = None

    # Volume fraction (scalar)
    vf = get_slice("volume_fraction")
    decoded["volume_fraction"] = float(vf[0]) if vf is not None and len(vf) > 0 else None

    # Shape tuple (normalized), plus raw shape from sample_info
    shape_norm = get_slice("shape_tuple")
    decoded["shape_norm"] = shape_norm if shape_norm is not None else None

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
        description="Visualize neural-field train-data samples with BC/load overlays decoded from cond_vec."
    )
    parser.add_argument("--train-data-npy", type=str, required=True,
                        help="Path to the neural-field train-data .npy file.")
    parser.add_argument("--train-data-meta", type=str, required=True,
                        help="Path to the corresponding _meta.npz file.")
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

    conditioning_spec = json.loads(m["conditioning_spec_json"].item())
    cond_slices = json.loads(m["cond_slices_json"].item())

    selected_indices, selection_meta = select_indices(len(x), args.sample_idx, args.n_random, args.rng_seed)

    run_summary = {
        "train_data_npy": args.train_data_npy,
        "train_data_meta": args.train_data_meta,
        "selection": {
            **selection_meta,
            "requested_sample_idx": args.sample_idx,
            "requested_n_random": args.n_random,
            "selected_sample_indices": [int(i) for i in selected_indices],
            "num_selected": len(selected_indices),
        },
        "dataset_metadata": {
            "cond_dim": int(m["cond_dim"]),
            "conditioning_spec": conditioning_spec,
            "vf_mode": m["vf_mode"].item(),
        },
        "parts": []
    }

    for sample_idx in selected_indices:
        sample = x[sample_idx]
        voxel_arr = sample[0]
        cond_vec = np.asarray(sample[1], dtype=np.float32)
        label_dummy = int(sample[2])
        cond_str = str(sample[3])
        sample_info = sample[4]

        decoded = decode_condition_fields(cond_vec, cond_slices)

        part_shape = np.asarray(sample_info["part_shape"]).tolist()
        bc_count = int(sample_info["bc_count"])
        mass_fraction = float(sample_info["mass_fraction"])
        volume_fraction = float(sample_info["volume_fraction"])

        # Plot overlays
        base = f"sample_neural_idx{sample_idx:05d}"
        out_png = os.path.join(args.outdir, f"{base}.png")
        out_json = os.path.join(args.outdir, f"{base}.json")

        title = (
            f"Neural-field train sample {sample_idx}, "
            f"shape={part_shape}, vf={volume_fraction:.4f}, bc_count={bc_count}"
        )

        plot_voxel_with_overlays(
            voxel_arr=voxel_arr,
            bc_pts=decoded["bc_points"],
            load_pt=decoded["load_point"],
            load_vec=decoded["load_dir"],
            title=title,
            save_path=out_png,
        )

        part_record = {
            "sample_array_index": int(sample_idx),
            "part_shape": part_shape,
            "bc_count": bc_count,
            "mass_fraction": mass_fraction,
            "volume_fraction": volume_fraction,
            "cond_dim": int(m["cond_dim"]),
            "cond_str": cond_str,
            "conditioning_spec": conditioning_spec,
            "cond_slices": cond_slices,
            "cond_vector": cond_vec.tolist(),
            "decoded_condition": {
                "bc_points": decoded["bc_points"].tolist() if decoded["bc_points"] is not None else None,
                "bc_dofs": decoded["bc_dofs"].tolist() if decoded["bc_dofs"] is not None else None,
                "load_point": decoded["load_point"].tolist() if decoded["load_point"] is not None else None,
                "load_dir": decoded["load_dir"].tolist() if decoded["load_dir"] is not None else None,
                "volume_fraction": decoded["volume_fraction"],
                "shape_norm": decoded["shape_norm"].tolist() if decoded["shape_norm"] is not None else None,
            },
            "plot_png": os.path.basename(out_png),
        }

        with open(out_json, "w") as f:
            json.dump(part_record, f, indent=2)

        run_summary["parts"].append({
            "sample_array_index": int(sample_idx),
            "part_shape": part_shape,
            "plot_png": os.path.basename(out_png),
            "part_json": os.path.basename(out_json),
            "volume_fraction": volume_fraction,
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