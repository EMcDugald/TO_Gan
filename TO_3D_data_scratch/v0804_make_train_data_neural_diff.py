import argparse
import os
import numpy as np

from v0804_cond_data_utils_neural_diff import (
    N_SPATIAL_BINS,
    parse_conditioning_spec,
    generate_dataset_mixed_shapes,
    save_dataset_and_meta_neural,
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate neural-field training data for mixed-shape BC/load-conditioned 3D parts."
    )
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--outdir", type=str, required=True)

    # Optional filter: list of shapes to include, e.g. "32x32x32,120x80x40"
    parser.add_argument(
        "--shapes-filter",
        type=str,
        default="",
        help="Comma-separated list of shapes to include, e.g. '32x32x32,120x80x40'. Empty = all shapes.",
    )

    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument(
        "--conditioning-spec", type=str,
        default="bc_locations=coarse,bc_dofs=omit,load_location=coarse,load_direction=omit",
    )
    parser.add_argument("--n-spatial-bins", type=int, default=N_SPATIAL_BINS)
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument(
        "--vf-mode",
        type=str,
        choices=["omit", "append_to_condition"],
        default="append_to_condition",
    )
    parser.add_argument(
        "--include-shape-in-cond",
        type=str,
        choices=["true", "false"],
        default="true",
        help="Whether to append normalized (Nx,Ny,Nz) shape to cond_vec.",
    )

    # ---- manufacturability conditioning ----
    parser.add_argument(
        "--manufacturability-mode", type=str,
        choices=["omit", "scalar", "binary"], default="omit",
        help="omit: no manufacturability conditioning. scalar: append "
             "[percentile_rank_in_[0,1], has_label]. binary: append "
             "[+1/-1 positive/negative label, has_label]. Overrides any "
             "manufacturability= token in --conditioning-spec.",
    )
    parser.add_argument(
        "--manufacturability-csv", type=str, default=None,
        help="Path to summary.csv (or similar) with a 'part' column and the "
             "scalar column named by --manuf-scalar-column. Required if "
             "--manufacturability-mode != omit.",
    )
    parser.add_argument(
        "--manuf-scalar-column", type=str, default="deformation_p99",
        help="Column in --manufacturability-csv to use as the manufacturability scalar.",
    )
    parser.add_argument(
        "--manuf-percentile-threshold", type=float, default=80.0,
        help="Percentile (0-100) of the scalar's distribution, computed over "
             "the FULL labeled CSV population, used as the positive/negative "
             "split for binary mode (value <= threshold -> positive). Also "
             "used as the reference point when reporting the threshold in "
             "meta.npz for scalar mode.",
    )
    parser.add_argument(
        "--n-unlabeled-extra", type=int, default=0,
        help="Number of additional shape-filtered parts with NO "
             "manufacturability label to mix into the pool alongside the "
             "labeled universe (0 = labeled-only; >0 = labeled + unlabeled).",
    )

    args = parser.parse_args()

    if args.manufacturability_mode != "omit" and not args.manufacturability_csv:
        parser.error("--manufacturability-csv is required when "
                     "--manufacturability-mode != omit")

    # Load base arrays
    topo = np.load(os.path.join(args.data_root, "topologies.npy"), allow_pickle=True)
    shapes = np.load(os.path.join(args.data_root, "shapes.npy"), allow_pickle=True)
    bcs = np.load(os.path.join(args.data_root, "boundary_conditions.npy"), allow_pickle=True)
    loads = np.load(os.path.join(args.data_root, "loads.npy"), allow_pickle=True)
    vfs = np.load(os.path.join(args.data_root, "vfs.npy"), allow_pickle=True).reshape(-1)

    conditioning_spec = parse_conditioning_spec(args.conditioning_spec)
    conditioning_spec["manufacturability"] = args.manufacturability_mode

    # Parse shapes filter
    if args.shapes_filter.strip():
        shapes_filter_strs = [tok.strip() for tok in args.shapes_filter.split(",") if tok.strip()]
        shapes_filter = []
        for s in shapes_filter_strs:
            parts = s.split("x")
            if len(parts) != 3:
                raise ValueError(f"Invalid shape filter entry: {s!r}")
            nx, ny, nz = map(int, parts)
            shapes_filter.append((nx, ny, nz))
    else:
        shapes_filter = None

    include_shape = args.include_shape_in_cond.lower() == "true"

    payload = generate_dataset_mixed_shapes(
        shapes_filter=shapes_filter,
        n_samples=args.n_samples,
        topo=topo,
        shapes=shapes,
        bcs=bcs,
        loads=loads,
        vfs=vfs,
        outdir=args.outdir,
        conditioning_spec=conditioning_spec,
        rng_seed=args.rng_seed,
        n_spatial_bins=args.n_spatial_bins,
        vf_mode=args.vf_mode,
        include_shape_in_cond=include_shape,
        manufacturability_csv=args.manufacturability_csv,
        manuf_scalar_column=args.manuf_scalar_column,
        manuf_percentile_threshold=args.manuf_percentile_threshold,
        n_unlabeled_extra=args.n_unlabeled_extra,
    )

    save_dataset_and_meta_neural(
        payload=payload,
        outdir=args.outdir,
        rng_seed=args.rng_seed,
        n_spatial_bins=args.n_spatial_bins,
        file_prefix="neuralfield_condVF_shape_voxels",
    )


if __name__ == "__main__":
    main()