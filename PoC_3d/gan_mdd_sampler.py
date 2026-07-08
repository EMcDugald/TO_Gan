import argparse
import os
import sys
import json

import numpy as np
import torch

# -------------------------------------------------------------------------
# Path setup: adjust these to your actual repo layout
# -------------------------------------------------------------------------

# Example: trainers live in PoC_3d/ alongside this sampler
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)

# If trainers live elsewhere, update these sys.path inserts.
sys.path.insert(0, THIS_DIR)      # e.g. v0627_sampler.py + trainers in same dir
sys.path.insert(0, REPO_ROOT)     # repo root if needed

# Trainer module names — change if your files differ
NONUNET_TRAINER_MODULE = "gan_mdd_trainer"
UNET_TRAINER_MODULE = "gan_mdd_unet_trainer"


# -------------------------------------------------------------------------
# Helper to import architecture-specific pieces
# -------------------------------------------------------------------------

def import_trainer(arch):
    """
    Import dataset, G, D, plotting helpers, and load_checkpoint
    from the appropriate trainer module.
    """
    if arch == "nonunet":
        mod = __import__(NONUNET_TRAINER_MODULE, fromlist=[
            "CondVoxelDataset",
            "CondGenerator3d",
            "CondDiscriminator3d",
            "decode_condition_overlay",
            "plot_voxel_grid_3d",
            "unwrap_module",
            "load_checkpoint",
        ])
        CondVoxelDataset = mod.CondVoxelDataset
        GenClass = mod.CondGenerator3d
        DiscClass = mod.CondDiscriminator3d
        decode_condition_overlay = mod.decode_condition_overlay
        plot_voxel_grid_3d = mod.plot_voxel_grid_3d
        unwrap_module = mod.unwrap_module
        load_checkpoint = mod.load_checkpoint
        gen_kind = "CondGenerator3d"
        disc_kind = "CondDiscriminator3d"

    elif arch == "unet":
        mod = __import__(UNET_TRAINER_MODULE, fromlist=[
            "CondVoxelDataset",
            "CondUNetGenerator3D",
            "CondDiscriminator3DImproved",
            "decode_condition_overlay",
            "plot_voxel_grid_3d",
            "unwrap_module",
            "load_checkpoint",
        ])
        CondVoxelDataset = mod.CondVoxelDataset
        GenClass = mod.CondUNetGenerator3D
        DiscClass = mod.CondDiscriminator3DImproved
        decode_condition_overlay = mod.decode_condition_overlay
        plot_voxel_grid_3d = mod.plot_voxel_grid_3d
        unwrap_module = mod.unwrap_module
        load_checkpoint = mod.load_checkpoint
        gen_kind = "CondUNetGenerator3D"
        disc_kind = "CondDiscriminator3DImproved"

    else:
        raise ValueError(f"Unsupported arch {arch!r}; expected 'nonunet' or 'unet'")

    return {
        "CondVoxelDataset": CondVoxelDataset,
        "GenClass": GenClass,
        "DiscClass": DiscClass,
        "decode_condition_overlay": decode_condition_overlay,
        "plot_voxel_grid_3d": plot_voxel_grid_3d,
        "unwrap_module": unwrap_module,
        "load_checkpoint": load_checkpoint,
        "gen_kind": gen_kind,
        "disc_kind": disc_kind,
    }


# -------------------------------------------------------------------------
# Manifest and selection utilities
# -------------------------------------------------------------------------

def resolve_meta_path(data_path, meta_path=None):
    if meta_path is not None:
        return meta_path
    if data_path.endswith(".npy"):
        return data_path[:-4] + "_meta.npz"
    raise ValueError("Could not infer meta path; please provide --meta-path")


def select_indices(n_total, sample_idx, n_random, rng_seed):
    """
    Single explicit index or N random indices (without replacement).
    """
    if sample_idx is not None:
        if sample_idx < 0 or sample_idx >= n_total:
            raise ValueError(f"sample_idx {sample_idx} out of range for {n_total} samples")
        return [int(sample_idx)], {"selection_mode": "single", "rng_seed": None}

    if n_random is None or n_random <= 0:
        raise ValueError("When --sample-idx is not used, --n-random must be positive")

    n_pick = min(int(n_random), int(n_total))
    rng = np.random.default_rng(rng_seed)
    picks = sorted(rng.choice(n_total, size=n_pick, replace=False).tolist())
    return picks, {"selection_mode": "random", "rng_seed": int(rng_seed)}


def get_mass_meta(meta):
    if "mass_cutoff_value" not in meta:
        return None

    if "conditioning_spec_json" in meta:
        conditioning_spec = json.loads(str(meta["conditioning_spec_json"]))
        conditioning_mode = str(conditioning_spec.get("bc_locations", "unknown"))
    elif "conditioning_spec_str" in meta:
        conditioning_mode = str(meta["conditioning_spec_str"])
    elif "conditioning_mode" in meta:
        conditioning_mode = str(meta["conditioning_mode"])
    else:
        conditioning_mode = "unknown"

    mass_quantile = None
    if "mass_quantile" in meta:
        try:
            mq = float(meta["mass_quantile"])
            if not np.isnan(mq):
                mass_quantile = mq
        except Exception:
            mass_quantile = None

    return {
        "mass_cutoff_value": float(meta["mass_cutoff_value"]),
        "mass_cutoff_method": str(meta["mass_cutoff_method"]) if "mass_cutoff_method" in meta else "unknown",
        "positive_if": str(meta["positive_if"]) if "positive_if" in meta else "unknown",
        "conditioning_mode": conditioning_mode,
        "label_mode": str(meta["label_mode"]) if "label_mode" in meta else "unknown",
        "mass_quantile": mass_quantile,
        "mass_threshold_source": str(meta["mass_threshold_source"]) if "mass_threshold_source" in meta else "unknown",
    }

def get_cond_slices(meta):
    if "cond_slices_json" in meta:
        return json.loads(str(meta["cond_slices_json"]))
    if "cond_slices" in meta:
        return meta["cond_slices"].item()
    return None


def decode_condition_fields(cond_vec, meta):
    cond_vec = np.asarray(cond_vec, dtype=np.float64)
    cond_slices = get_cond_slices(meta)
    if cond_slices is None:
        return {}

    decoded = {}
    for key in [
        "bc_points",
        "bc_mask",
        "bc_count",
        "bc_dofs",
        "load_point",
        "load_bins",
        "load_dir",
        "bc_x_range",
        "bc_y_range",
        "bc_z_range",
        "bc_bins",
    ]:
        if key not in cond_slices:
            continue

        s, e = cond_slices[key]
        vals = np.asarray(cond_vec[s:e])

        if key == "bc_count":
            decoded[f"cond_{key}"] = float(vals[0]) if len(vals) else None
        else:
            decoded[f"cond_{key}"] = vals.tolist()

    return decoded


def summarize_fake_mass_from_bin(batch_bin_np, meta):
    """
    batch_bin_np: numpy array of shape (B, D, H, W) with values 0/1
    Returns None if mass metadata is unavailable.
    """
    batch_bin_np = np.asarray(batch_bin_np, dtype=np.float64)
    if batch_bin_np.ndim != 4:
        raise ValueError(f"Expected batch_bin_np shape (B,D,H,W), got {batch_bin_np.shape}")

    mass_info = get_mass_meta(meta)
    if mass_info is None:
        return None

    mass_fractions = batch_bin_np.mean(axis=(1, 2, 3))
    cutoff = mass_info["mass_cutoff_value"]
    positive_if = mass_info["positive_if"]

    if positive_if in ("low_mass", "lowmass"):
        is_positive = mass_fractions <= cutoff
    elif positive_if in ("high_mass", "highmass"):
        is_positive = mass_fractions >= cutoff
    else:
        is_positive = None

    return {
        "mass_fractions": mass_fractions,
        "cutoff": cutoff,
        "positive_if": positive_if,
        "is_positive": None if is_positive is None else is_positive,
        "positive_rate_percent": None if is_positive is None else float(is_positive.mean() * 100.0),
        "mean_mass": float(mass_fractions.mean()),
        "min_mass": float(mass_fractions.min()),
        "max_mass": float(mass_fractions.max()),
    }


# -------------------------------------------------------------------------
# Core sampling routine
# -------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Sample v0627 3D GAN-MDD checkpoints (non-UNet or UNet) on BC/load-conditioned voxel data."
    )

    parser.add_argument("--arch", type=str, choices=["nonunet", "unet"], required=True,
                        help="Generator architecture: nonunet (CondGenerator3d) or unet (CondUNetGenerator3D).")

    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to the BC/load-conditioned train-data .npy file (same format as trainers).")
    parser.add_argument("--meta-path", type=str, default=None,
                        help="Optional explicit meta .npz path; if omitted, inferred as data_path[:-4] + '_meta.npz'.")

    parser.add_argument("--checkpoint-path", type=str, required=True,
                        help="Path to a GAN checkpoint .pt file (ckpt_best.pt, ckpt_step_*.pt, or ckpt_final.pt).")

    parser.add_argument("--outdir", type=str, required=True,
                        help="Output directory for PNGs + JSON manifest.")

    parser.add_argument("--nz", type=int, required=True,
                        help="Noise dimension (must match the trainer used for the checkpoint).")
    parser.add_argument("--ngf", type=int, required=True,
                        help="Base channel count in G (ngf at training time).")
    parser.add_argument("--ndf", type=int, required=True,
                        help="Base channel count in D (must match training checkpoint).")
    parser.add_argument("--n-per-cond", type=int, default=4,
                        help="Number of generated samples to draw per selected condition.")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sample-idx", type=int, default=None,
                       help="Visualize exactly one prescribed sample index from the train-data array.")
    group.add_argument("--n-random", type=int, default=None,
                       help="Visualize N randomly chosen samples from the train-data array.")
    parser.add_argument("--rng-seed", type=int, default=0,
                        help="Random seed used when --n-random is provided.")

    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use: 'cuda' or 'cpu'.")

    args = parser.parse_args()

    # Import trainer-specific pieces
    trainer = import_trainer(args.arch)
    CondVoxelDataset = trainer["CondVoxelDataset"]
    GenClass = trainer["GenClass"]
    DiscClass = trainer["DiscClass"]
    decode_condition_overlay = trainer["decode_condition_overlay"]
    plot_voxel_grid_3d = trainer["plot_voxel_grid_3d"]
    unwrap_module = trainer["unwrap_module"]
    load_checkpoint = trainer["load_checkpoint"]
    gen_kind = trainer["gen_kind"]
    disc_kind = trainer["disc_kind"]

    os.makedirs(args.outdir, exist_ok=True)

    # Device
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[sampler] device: {device}")

    # Load meta
    meta_path = resolve_meta_path(args.data_path, args.meta_path)
    print(f"[sampler] meta path: {meta_path}")
    meta = np.load(meta_path, allow_pickle=True)
    mass_info = get_mass_meta(meta)
    cond_slices = get_cond_slices(meta)

    # Load dataset
    dataset = CondVoxelDataset(args.data_path)
    X = dataset.X          # (N, 1, D, H, W)
    C = dataset.C          # (N, cond_dim)
    cond_strs = dataset.cond_strs
    sample_infos = getattr(dataset, "sample_infos", [{} for _ in range(X.shape[0])])
    n_total = X.shape[0]
    print(f"[sampler] total dataset samples: {n_total}")

    shape3d = X.shape[2:]
    cond_dim = C.shape[1]
    print(f"[sampler] shape3d={shape3d}, cond_dim={cond_dim}")

    # Build generator
    if args.arch == "nonunet":
        netG = GenClass(args.nz, args.ngf, shape3d, cond_dim)
    else:  # unet
        netG = GenClass(args.nz, args.ngf, shape3d, cond_dim, cond_channels=8)

    # Build discriminator that matches the selected checkpoint architecture.
    if args.arch == "nonunet":
        netD = DiscClass(args.ndf, shape3d, cond_dim, nc=3)
    else:  # unet
        netD = DiscClass(args.ndf, shape3d, cond_dim, cond_channels=8, nc=3)

    # Optimizers are needed because load_checkpoint() also restores optimizer state.
    D_opt = torch.optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.999))
    G_opt = torch.optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.999))

    # Move to device and maybe DataParallel if user wants to reuse multi-GPU;
    # for simple sampling, single device is fine.
    netG = netG.to(device)
    netD = netD.to(device)
    # Load checkpoint
    print(f"[sampler] loading checkpoint: {args.checkpoint_path}")
    # netG, _, G_opt, _, step = load_checkpoint(
    #     args.checkpoint_path, netG, netD, G_opt, D_opt, device
    # )
    netG, netD, G_opt, D_opt, step = load_checkpoint(
        args.checkpoint_path, netG, netD, G_opt, D_opt, device
        )
    print(f"[sampler] checkpoint step: {step}")

    # Prepare index selection
    selected_indices, selection_meta = select_indices(
        n_total, args.sample_idx, args.n_random, args.rng_seed
    )
    print(f"[sampler] selected sample indices: {selected_indices}")

    # Try to get conditioning_spec for mode tags
    if "conditioning_spec_json" in meta:
        conditioning_spec = json.loads(str(meta["conditioning_spec_json"]))
    elif "conditioning_spec" in meta:
        conditioning_spec = meta["conditioning_spec"].item()
    else:
        conditioning_spec = {
            "bc_locations": str(meta["bc_locations_mode"]) if "bc_locations_mode" in meta else "unknown",
            "bc_dofs": str(meta["bc_dofs_mode"]) if "bc_dofs_mode" in meta else "unknown",
            "load_location": str(meta["load_location_mode"]) if "load_location_mode" in meta else "unknown",
            "load_direction": str(meta["load_direction_mode"]) if "load_direction_mode" in meta else "unknown",
        }

    # Run summary
    run_summary = {
        "arch": args.arch,
        "gen_kind": gen_kind,
        "data_path": args.data_path,
        "meta_path": meta_path,
        "checkpoint_path": args.checkpoint_path,
        "n_per_condition": int(args.n_per_cond),
        "selection": {
            **selection_meta,
            "requested_sample_idx": args.sample_idx,
            "requested_n_random": args.n_random,
            "selected_sample_indices": [int(i) for i in selected_indices],
            "num_selected": len(selected_indices),
        },
        # "dataset_metadata": {
        #     "cond_dim": int(cond_dim),
        #     "conditioning_spec": conditioning_spec,
        #     "shape3d": list(shape3d),
        #     "mass_info": mass_info,
        # },

        "dataset_metadata": {
            "cond_dim": int(cond_dim),
            "conditioning_spec": conditioning_spec,
            "cond_slices": cond_slices,
            "shape3d": list(shape3d),
            "mass_info": mass_info,
        },
        "parts": [],
    }

    # Switch generator to eval
    G_raw = unwrap_module(netG)
    G_raw.eval()

    with torch.no_grad():
        for i_idx, sample_idx in enumerate(selected_indices):
            # Fetch train sample
            voxel_arr = X[sample_idx].cpu().numpy()[0]  # (D,H,W)

            ### Possible code change ###
            cond_vec = C[sample_idx].cpu().numpy()
            # cond_t = C[sample_idx].to(device)          # (cond_dim,)
            # cond_vec = cond_t.detach().cpu().numpy()   # for JSON / decode / logging
            
            decoded_cond_fields = decode_condition_fields(cond_vec, meta)
            cond_str = cond_strs[int(sample_idx)]

            sample_info = sample_infos[int(sample_idx)] if sample_infos is not None else {}
            if sample_info is None:
                sample_info = {}

            source_index = sample_info.get("source_index", None) if isinstance(sample_info, dict) else None
            train_mass_fraction = sample_info.get("mass_fraction", None) if isinstance(sample_info, dict) else None
            volume_fraction = sample_info.get("volume_fraction", None) if isinstance(sample_info, dict) else None
            bc_count_meta = sample_info.get("bc_count", None) if isinstance(sample_info, dict) else None

            # Decode BC/load overlay from condition vector
            overlay = decode_condition_overlay(cond_vec, meta)
            bc_points = overlay["bc_points"]
            load_point = overlay["load_point"]
            load_dir = overlay["load_dir"]
            spec = overlay["conditioning_spec"]
            mode_tag = (
                f"bcLoc={spec.get('bc_locations')} "
                f"loadLoc={spec.get('load_location')} "
                f"loadDir={spec.get('load_direction')}"
            )

            # Generate ensemble for this condition
            n_fake = int(args.n_per_cond)
            z = torch.randn(n_fake, args.nz, device=device)

            ### Possible code change ###
            c_vis = (
                torch.tensor(cond_vec, dtype=torch.float32, device=device)
                .unsqueeze(0)
                .expand(n_fake, -1)
            )
            #c_vis = cond_t.unsqueeze(0).expand(n_fake, -1)

            fake = G_raw(z, c_vis).cpu().numpy()   # (n_fake, 1, D, H, W)

            fake_raw = fake[:, 0]                          # (n_fake, D, H, W)
            fake_bin = (fake_raw > 0.0).astype(np.float32)
            fake_mass_summary = summarize_fake_mass_from_bin(fake_bin, meta)

            base_prefix = f"sample_{args.arch}_idx{sample_idx:05d}"
            fake_raw_npy = os.path.join(args.outdir, f"{base_prefix}_fake_raw.npy")
            fake_bin_npy = os.path.join(args.outdir, f"{base_prefix}_fake_bin.npy")
            np.save(fake_raw_npy, fake_raw)
            np.save(fake_bin_npy, fake_bin)
            # Plot train voxel
            train_png = os.path.join(args.outdir, f"{base_prefix}_train.png")
            plot_voxel_grid_3d(
                voxel_arr,
                title=f"TRAIN sample idx={sample_idx} | {mode_tag}",
                save_path=train_png,
                bc_points=bc_points,
                load_point=load_point,
                load_vec=load_dir,
            )

            train_bin_npy = os.path.join(args.outdir, f"{base_prefix}_train_bin.npy")
            np.save(train_bin_npy, voxel_arr)

            fake_pngs = []
            for k in range(n_fake):
                fake_png = os.path.join(args.outdir, f"{base_prefix}_fake{k:02d}.png")

                mass_tag = ""
                if fake_mass_summary is not None:
                    mf = float(fake_mass_summary["mass_fractions"][k])
                    mass_tag = f" | mass={mf:.4f}"

                plot_voxel_grid_3d(
                    fake_raw[k],
                    title=f"FAKE k={k} | idx={sample_idx} | {mode_tag}{mass_tag}",
                    save_path=fake_png,
                    bc_points=bc_points,
                    load_point=load_point,
                    load_vec=load_dir,
                )
                fake_pngs.append(os.path.basename(fake_png))

            part_json = os.path.join(args.outdir, f"{base_prefix}.json")

            part_record = {
                "arch": args.arch,
                "gen_kind": gen_kind,
                "disc_kind": disc_kind,
                "checkpoint_step": int(step),
                "sample_array_index": int(sample_idx),
                "source_index": source_index,
                "train_mass_fraction": train_mass_fraction,
                "volume_fraction": volume_fraction,
                "bc_count_from_sample_info": bc_count_meta,
                "cond_dim": int(cond_dim),
                "cond_str": cond_str,
                "cond_vector": cond_vec.tolist(),
                "conditioning_spec": spec,
                "mode_tag": mode_tag,
                "train_plot_png": os.path.basename(train_png),
                "n_per_condition": int(n_fake),
                "fake_plot_pngs": fake_pngs,
                "fake_raw_npy": os.path.basename(fake_raw_npy),
                "fake_bin_npy": os.path.basename(fake_bin_npy),
                "bc_points": bc_points.tolist() if bc_points is not None else None,
                "load_point": load_point.tolist() if load_point is not None else None,
                "load_dir": load_dir.tolist() if load_dir is not None else None,
                "mass_summary": None,
            }

            if fake_mass_summary is not None:
                part_record["mass_summary"] = {
                    "mass_cutoff_value": float(fake_mass_summary["cutoff"]),
                    "positive_if": fake_mass_summary["positive_if"],
                    "mean_mass": float(fake_mass_summary["mean_mass"]),
                    "min_mass": float(fake_mass_summary["min_mass"]),
                    "max_mass": float(fake_mass_summary["max_mass"]),
                    "positive_rate_percent": (
                        None if fake_mass_summary["positive_rate_percent"] is None
                        else float(fake_mass_summary["positive_rate_percent"])
                    ),
                    "mass_fractions": fake_mass_summary["mass_fractions"].tolist(),
                    "is_positive": (
                        None if fake_mass_summary["is_positive"] is None
                        else fake_mass_summary["is_positive"].tolist()
                    ),
                }
            
            # 1) Merge decoded condition fields into the record
            part_record.update(decoded_cond_fields)
            part_record["part_json"] = os.path.basename(part_json)

            with open(part_json, "w") as f:
                json.dump(part_record, f, indent=2)

            run_summary["parts"].append(part_record)

    summary_json = os.path.join(args.outdir, "sampler_run_summary.json")
    with open(summary_json, "w") as f:
        json.dump(run_summary, f, indent=2)

    print(f"[sampler] saved run summary to {summary_json}")


if __name__ == "__main__":
    main()