"""
Sanity-check visualizer for the manufacturability-conditioned train-data
files (v0804_make_train_data_manuf.sh outputs). Based on
v0718_visualize_train_data_neural_diff.py's BC/load overlay plotting
(unchanged -- that isotropic max(nx,ny,nz) scaling was already verified
correct against GT), extended to:

  1. decode the manufacturability slot(s) directly from cond_vec (using
     cond_slices from meta.npz), so you're checking what the MODEL actually
     sees, not just what was intended
  2. cross-check that decoded value against sample_info's recorded
     manuf_raw_value / manuf_percentile_rank / manuf_label / manuf_has_label
     (computed independently at data-gen time) -- any mismatch here would
     mean cond_vec got corrupted between generation and decoding
  3. print the dataset-level threshold info (from meta.npz) once, and put
     per-sample manufacturability status in every plot title
  4. optionally filter which samples to visualize by manufacturability
     status (labeled / unlabeled / positive / negative), so you can spot
     check both populations in a labeled+unlabeled dataset

Usage:
  python v0804_visualize_train_data_manuf.py \
      --train-data-npy <path>/N_neuralfield_..._manuf-binary_....npy \
      --train-data-meta <path>/..._meta.npz \
      --outdir viz_out --n-random 12 --filter all

  --filter {all, labeled, unlabeled, positive, negative}
    (positive/negative only meaningful for binary-mode datasets; for
    scalar-mode datasets they select has_label True with
    percentile_rank <=/> the dataset's recorded threshold-percentile)
"""

import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt


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

    # Isotropic length scale, matching the [0,1]-by-max(Nx,Ny,Nz) convention
    # used everywhere else in this codebase (confirmed correct vs GT).
    c = float(max(nx, ny, nz))

    bc_plot = None
    if bc_pts is not None and len(bc_pts) > 0:
        bc_pts = np.asarray(bc_pts, dtype=np.float64)
        bc_plot = np.column_stack([
            bc_pts[:, 0] * (c - 1),
            bc_pts[:, 1] * (c - 1),
            bc_pts[:, 2] * (c - 1),
        ])

    load_plot = None
    if load_pt is not None:
        lp = np.asarray(load_pt, dtype=np.float64)
        load_plot = np.array([lp[0] * c, lp[1] * c, lp[2] * c], dtype=np.float64)

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


def decode_condition_fields(cond_vec, cond_slices, manuf_mode):
    decoded = {}

    def get_slice(name):
        if name not in cond_slices:
            return None
        s, e = cond_slices[name]
        return np.asarray(cond_vec[s:e], dtype=np.float32)

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
        decoded["bc_count"] = bc_count
    else:
        decoded["bc_points"] = None
        decoded["bc_count"] = None

    decoded["bc_dofs"] = bc_dofs.reshape(-1, 3) if bc_dofs is not None else None

    load_point = get_slice("load_point")
    decoded["load_point"] = load_point[:3] if load_point is not None and len(load_point) >= 3 else None

    load_dir = get_slice("load_dir")
    decoded["load_dir"] = load_dir[:3] if load_dir is not None and len(load_dir) >= 3 else None

    vf = get_slice("volume_fraction")
    decoded["volume_fraction"] = float(vf[0]) if vf is not None and len(vf) > 0 else None

    # ---- manufacturability, decoded directly from cond_vec ----
    decoded["manuf_has_label_from_cond"] = None
    decoded["manuf_value_from_cond"] = None
    decoded["manuf_label_from_cond"] = None
    if manuf_mode != "omit":
        hl = get_slice("manuf_has_label")
        if hl is not None and len(hl) > 0:
            decoded["manuf_has_label_from_cond"] = bool(round(float(hl[0])))
        if manuf_mode == "scalar":
            v = get_slice("manuf_value")
            if v is not None and len(v) > 0:
                decoded["manuf_value_from_cond"] = float(v[0])
        elif manuf_mode == "binary":
            l = get_slice("manuf_label")
            if l is not None and len(l) > 0:
                decoded["manuf_label_from_cond"] = float(l[0])

    return decoded


def sample_passes_filter(sample_info, manuf_mode, filter_mode):
    if filter_mode == "all":
        return True
    has_label = sample_info.get("manuf_has_label")
    if filter_mode == "unlabeled":
        return has_label is False
    if filter_mode == "labeled":
        return has_label is True
    if filter_mode in ("positive", "negative"):
        if not has_label:
            return False
        if manuf_mode == "binary":
            label = sample_info.get("manuf_label")
            return (label is not None) and (
                (filter_mode == "positive" and label > 0) or
                (filter_mode == "negative" and label < 0)
            )
        elif manuf_mode == "scalar":
            # No stored +/- label in scalar mode -- fall back to whether the
            # raw value was <= the dataset's recorded threshold.
            raw = sample_info.get("manuf_raw_value")
            return raw is not None  # caller compares against threshold separately
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Visualize manufacturability train-data samples with BC/load overlays "
                    "and manufacturability status, cross-checked against sample_info."
    )
    parser.add_argument("--train-data-npy", type=str, required=True)
    parser.add_argument("--train-data-meta", type=str, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sample-idx", type=int, default=None)
    group.add_argument("--n-random", type=int, default=None)
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--filter", type=str, default="all",
                        choices=["all", "labeled", "unlabeled", "positive", "negative"])
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    x = np.load(args.train_data_npy, allow_pickle=True)
    m = np.load(args.train_data_meta, allow_pickle=True)

    conditioning_spec = json.loads(m["conditioning_spec_json"].item())
    cond_slices = json.loads(m["cond_slices_json"].item())
    manuf_mode = str(m["manufacturability_mode"]) if "manufacturability_mode" in m else "omit"

    print(f"Dataset: {args.train_data_npy}")
    print(f"  n_samples = {len(x)}, cond_dim = {int(m['cond_dim'])}")
    print(f"  manufacturability_mode = {manuf_mode}")
    if manuf_mode != "omit":
        print(f"  manuf_scalar_column = {m['manuf_scalar_column'].item()}")
        print(f"  manuf_percentile_threshold = {float(m['manuf_percentile_threshold']):.1f}")
        print(f"  manuf_threshold_value = {float(m['manuf_threshold_value']):.6g}")
        print(f"  manuf_n_labeled_used = {int(m['manuf_n_labeled_used'])}")
        print(f"  manuf_n_unlabeled_used = {int(m['manuf_n_unlabeled_used'])}")
    print()

    threshold_value = float(m["manuf_threshold_value"]) if manuf_mode != "omit" else None
    threshold_pctl = float(m["manuf_percentile_threshold"]) if manuf_mode != "omit" else None

    # ---- build eligible index list per --filter ----
    if args.sample_idx is not None:
        eligible = [args.sample_idx]
    else:
        eligible = []
        for i in range(len(x)):
            sample_info = x[i][4]
            if manuf_mode == "scalar" and args.filter in ("positive", "negative"):
                raw = sample_info.get("manuf_raw_value")
                if raw is None:
                    continue
                is_pos = raw <= threshold_value
                if (args.filter == "positive") == is_pos:
                    eligible.append(i)
            elif sample_passes_filter(sample_info, manuf_mode, args.filter):
                eligible.append(i)
        if not eligible:
            raise ValueError(f"No samples matched --filter={args.filter} in this dataset")
        rng = np.random.default_rng(args.rng_seed)
        n_pick = min(args.n_random, len(eligible))
        eligible = sorted(rng.choice(eligible, size=n_pick, replace=False).tolist())

    print(f"Visualizing {len(eligible)} sample(s), filter={args.filter}\n")

    n_mismatches = 0
    run_summary = {"train_data_npy": args.train_data_npy, "filter": args.filter, "parts": []}

    for sample_idx in eligible:
        sample = x[sample_idx]
        voxel_arr = sample[0]
        cond_vec = np.asarray(sample[1], dtype=np.float32)
        cond_str = str(sample[3])
        sample_info = sample[4]

        decoded = decode_condition_fields(cond_vec, cond_slices, manuf_mode)
        part_shape = np.asarray(sample_info["part_shape"]).tolist()
        vf = sample_info["volume_fraction"]

        # ---- manufacturability title text + cross-check ----
        manuf_title = ""
        if manuf_mode != "omit":
            has_label_recorded = sample_info.get("manuf_has_label")
            has_label_decoded = decoded["manuf_has_label_from_cond"]
            mismatch = (bool(has_label_recorded) != bool(has_label_decoded))

            if has_label_recorded:
                raw_val = sample_info.get("manuf_raw_value")
                pctile = sample_info.get("manuf_percentile_rank")
                lbl = sample_info.get("manuf_label")
                if manuf_mode == "scalar":
                    decoded_val = decoded["manuf_value_from_cond"]
                    if decoded_val is not None and pctile is not None:
                        mismatch = mismatch or (abs(decoded_val - pctile) > 1e-4)
                    manuf_title = (f"manuf: {sample_info.get('source_index')} labeled, "
                                   f"raw={raw_val:.4g}, pctile={pctile:.3f} "
                                   f"(thresh@p{threshold_pctl:.0f}={threshold_value:.4g})")
                elif manuf_mode == "binary":
                    decoded_lbl = decoded["manuf_label_from_cond"]
                    if decoded_lbl is not None and lbl is not None:
                        mismatch = mismatch or (abs(decoded_lbl - lbl) > 1e-4)
                    sign = "POSITIVE" if lbl and lbl > 0 else "NEGATIVE"
                    manuf_title = (f"manuf: labeled, {sign} (raw={raw_val:.4g}, "
                                   f"thresh@p{threshold_pctl:.0f}={threshold_value:.4g})")
            else:
                manuf_title = "manuf: UNLABELED"

            if mismatch:
                n_mismatches += 1
                print(f"  [MISMATCH] idx {sample_idx}: sample_info vs cond_vec disagree "
                      f"on manufacturability fields!")

        title = (f"idx {sample_idx} | shape={part_shape} | vf={vf:.4f} | {manuf_title}")

        base = f"sample_manuf_idx{sample_idx:05d}"
        out_png = os.path.join(args.outdir, f"{base}.png")
        out_json = os.path.join(args.outdir, f"{base}.json")

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
            "volume_fraction": vf,
            "cond_str": cond_str,
            "sample_info": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                            for k, v in sample_info.items()},
            "decoded_from_cond_vec": {
                "manuf_has_label": decoded["manuf_has_label_from_cond"],
                "manuf_value": decoded["manuf_value_from_cond"],
                "manuf_label": decoded["manuf_label_from_cond"],
            },
            "plot_png": os.path.basename(out_png),
        }
        with open(out_json, "w") as f:
            json.dump(part_record, f, indent=2)

        run_summary["parts"].append({
            "sample_array_index": int(sample_idx),
            "plot_png": os.path.basename(out_png),
            "manuf_title": manuf_title,
        })
        print(f"Saved {out_png}")

    with open(os.path.join(args.outdir, "run_summary.json"), "w") as f:
        json.dump(run_summary, f, indent=2)

    print(f"\nDone. {len(eligible)} plots saved to {args.outdir}")
    if manuf_mode != "omit":
        print(f"cond_vec vs sample_info mismatches: {n_mismatches} "
              f"({'OK -- data is consistent' if n_mismatches == 0 else 'INVESTIGATE'})")


if __name__ == "__main__":
    main()
