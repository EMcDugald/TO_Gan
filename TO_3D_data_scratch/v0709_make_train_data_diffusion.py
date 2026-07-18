import argparse
import os
import numpy as np

from v0709_cond_data_utils import (
    TARGET_SHAPE,
    N_SPATIAL_BINS,
    parse_conditioning_spec,
    generate_dataset_common,
    save_dataset_and_meta,
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate diffusion training data with mass label appended to condition vector."
    )
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--shape", type=int, nargs=3, default=list(TARGET_SHAPE))
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--conditioning-spec", type=str,
                        default="bc_locations=coarse,bc_dofs=omit,load_location=coarse,load_direction=omit")
    parser.add_argument("--n-spatial-bins", type=int, default=N_SPATIAL_BINS)
    parser.add_argument("--positive-if", type=str, choices=["low_mass", "high_mass"], default="low_mass")
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument("--mass-quantile", type=float, default=None)
    parser.add_argument("--vf-mode",type=str,choices=["omit", "append_to_condition"],default="omit")
    args = parser.parse_args()

    topo = np.load(os.path.join(args.data_root, "topologies.npy"), allow_pickle=True)
    shapes = np.load(os.path.join(args.data_root, "shapes.npy"), allow_pickle=True)
    bcs = np.load(os.path.join(args.data_root, "boundary_conditions.npy"), allow_pickle=True)
    loads = np.load(os.path.join(args.data_root, "loads.npy"), allow_pickle=True)
    vfs = np.load(os.path.join(args.data_root, "vfs.npy"), allow_pickle=True).reshape(-1)

    conditioning_spec = parse_conditioning_spec(args.conditioning_spec)

    payload = generate_dataset_common(
        shape_tuple=tuple(args.shape),
        n_samples=args.n_samples,
        topo=topo,
        shapes=shapes,
        bcs=bcs,
        loads=loads,
        vfs=vfs,
        outdir=args.outdir,
        conditioning_spec=conditioning_spec,
        positive_if=args.positive_if,
        rng_seed=args.rng_seed,
        mass_quantile=args.mass_quantile,
        n_spatial_bins=args.n_spatial_bins,
        label_mode="append_to_condition",
        vf_mode=args.vf_mode,
    )

    save_dataset_and_meta(
        payload=payload,
        outdir=args.outdir,
        shape_tuple=tuple(args.shape),
        positive_if=args.positive_if,
        rng_seed=args.rng_seed,
        mass_quantile=args.mass_quantile,
        n_spatial_bins=args.n_spatial_bins,
        file_prefix="diffusion_condLabelVF_voxels" if args.vf_mode == "append_to_condition" else "diffusion_condLabel_voxels",
    )


if __name__ == "__main__":
    main()