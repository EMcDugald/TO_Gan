import argparse
import os
import sys
import json
from pathlib import Path
import functools

import numpy as np
import torch

# -------------------------------------------------------------------------
# Path + trainer module setup
# -------------------------------------------------------------------------

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)

sys.path.insert(0, THIS_DIR)
sys.path.insert(0, REPO_ROOT)

DIFFUSION_TRAINER_MODULE = "diffusion_trainer"


def import_diffusion_trainer():
    """
    Import dataset, model, meta helpers, and sampler utilities
    from the diffusion trainer module.
    """
    mod = __import__(
        DIFFUSION_TRAINER_MODULE,
        fromlist=[
            "CondVoxelDataset3D",
            "ScoreNet3DVoxelCond",
            "marginal_prob_mean",
            "marginal_prob_std",
            "drift_coeff",
            "load_meta_for_data",
            "decode_condition_overlay",
            "plot_voxel_with_conditions",
            "ode_sampler_voxel_cond",
        ],
    )
    return {
        "CondVoxelDataset3D": mod.CondVoxelDataset3D,
        "ScoreNet3DVoxelCond": mod.ScoreNet3DVoxelCond,
        "marginal_prob_mean": mod.marginal_prob_mean,
        "marginal_prob_std": mod.marginal_prob_std,
        "drift_coeff": mod.drift_coeff,
        "load_meta_for_data": mod.load_meta_for_data,
        "decode_condition_overlay": mod.decode_condition_overlay,
        "plot_voxel_with_conditions": mod.plot_voxel_with_conditions,
        "ode_sampler_voxel_cond": mod.ode_sampler_voxel_cond,
    }


# -------------------------------------------------------------------------
# Meta + mass utilities (mirroring GAN sampler)
# -------------------------------------------------------------------------

def resolve_meta_path(data_path, meta_path=None):
    if meta_path is not None:
        return meta_path
    if data_path.endswith(".npy"):
        return data_path[:-4] + "_meta.npz"
    raise ValueError("Could not infer meta path; please provide --meta-path")


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
# Selection utilities (aligned with GAN, but positive-only by default)
# -------------------------------------------------------------------------

def select_indices_by_subset(y, sample_idx, n_random, rng_seed, subset="positive_only"):
    """
    Select reference indices from either the positive-only subset (default)
    or the full dataset, mirroring GAN sampler behavior.

    subset: "positive_only" or "all"
    """
    y = np.asarray(y, dtype=np.int64)

    if subset == "positive_only":
        allowed = np.where(y == 1)[0]
        single_mode = "single_positive"
        random_mode = "random_positive"
    elif subset == "all":
        allowed = np.arange(len(y))
        single_mode = "single"
        random_mode = "random"
    else:
        raise ValueError(f"Unsupported subset {subset!r}; expected 'positive_only' or 'all'")

    if len(allowed) == 0:
        raise ValueError(f"No samples available for subset={subset}")

    allowed_set = set(allowed.tolist())

    if sample_idx is not None:
        if sample_idx < 0 or sample_idx >= len(y):
            raise ValueError(f"sample_idx {sample_idx} out of range for {len(y)} samples")
        if sample_idx not in allowed_set:
            raise ValueError(f"sample_idx {sample_idx} is not in subset={subset}")
        selected = [int(sample_idx)]
        meta = {"selection_mode": single_mode, "rng_seed": None}
        return selected, meta

    if n_random is None or n_random <= 0:
        raise ValueError("When --sample-idx is not used, --n-random must be positive")

    n_pick = min(int(n_random), len(allowed))
    rng = np.random.default_rng(rng_seed)
    picks = sorted(rng.choice(allowed, size=n_pick, replace=False).tolist())
    meta = {
        "selection_mode": random_mode,
        "rng_seed": int(rng_seed),
    }
    return picks, meta


# -------------------------------------------------------------------------
# Core sampling routine
# -------------------------------------------------------------------------

def run_diffusion_sampler(
    data_path,
    meta_path,
    checkpoint_path,
    outdir,
    img_size,
    unet_ch1_dim,
    unet_ch2_dim,
    unet_ch3_dim,
    t_embed_dim,
    cond_embed_dim,
    cond_ch,
    mode,
    sample_atol,
    sample_rtol,
    sample_eps,
    sample_idx,
    n_random,
    n_per_cond,
    rng_seed,
    subset,
    device_str,
):
    trainer = import_diffusion_trainer()
    CondVoxelDataset3D = trainer["CondVoxelDataset3D"]
    ScoreNet3DVoxelCond = trainer["ScoreNet3DVoxelCond"]
    marginal_prob_mean_fn = trainer["marginal_prob_mean"]
    marginal_prob_std_fn = trainer["marginal_prob_std"]
    drift_coeff_fn = trainer["drift_coeff"]
    load_meta_for_data = trainer["load_meta_for_data"]
    decode_condition_overlay = trainer["decode_condition_overlay"]
    plot_voxel_with_conditions = trainer["plot_voxel_with_conditions"]
    ode_sampler_voxel_cond = trainer["ode_sampler_voxel_cond"]

    os.makedirs(outdir, exist_ok=True)

    # Device
    if device_str == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[diffusion-sampler] device: {device}")

    # Load meta + dataset
    meta, meta_path = load_meta_for_data(data_path, meta_path)
    dataset = CondVoxelDataset3D(data_path, img_size=img_size)
    X = dataset.X          # (N, 1, D, H, W) in [-1,1]
    C = dataset.C          # (N, cond_dim)
    y = dataset.y          # (N,)
    cond_strs = dataset.cond_strs
    n_total = X.shape[0]
    sample_infos = getattr(dataset, "sample_infos", [{} for _ in range(n_total)])
    cond_dim = C.shape[1]
    print(f"[diffusion-sampler] total samples: {n_total}, cond_dim={cond_dim}")

    # Build marginal/std/drift partials using same bmin/bmax as trainer
    mp_mean = functools.partial(marginal_prob_mean_fn, bmin=0.1, bmax=20.0)
    mp_std = functools.partial(marginal_prob_std_fn, bmin=0.1, bmax=20.0)
    drift = functools.partial(drift_coeff_fn, bmin=0.1, bmax=20.0)

    # Construct score model (must match trainer hparams)
    score_model = ScoreNet3DVoxelCond(
        in_ch=1,
        cond_dim=cond_dim,
        cond_embed_dim=cond_embed_dim,
        cond_ch=cond_ch,
        c1=unet_ch1_dim,
        c2=unet_ch2_dim,
        c3=unet_ch3_dim,
        ed=t_embed_dim,
        mode=mode,
        marginal_prob_std=mp_std,
    ).to(device)

    # Load checkpoint (state_dict, as in trainer)
    print(f"[diffusion-sampler] loading checkpoint: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=device)
    score_model.load_state_dict(state)
    score_model.eval()

    # Selection (positive-only by default, but can be 'all')
    selected_indices, selection_meta = select_indices_by_subset(
        y, sample_idx, n_random, rng_seed, subset=subset)
    print(f"[diffusion-sampler] selected indices (subset={subset}): {selected_indices}")

    # Conditioning spec + cond_slices + label_mode (same logic as trainer)
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

    if "cond_slices_json" in meta:
        cond_slices = json.loads(str(meta["cond_slices_json"]))
    elif "cond_slices" in meta:
        cond_slices = meta["cond_slices"].item()
    else:
        cond_slices = None

    label_mode = str(meta["label_mode"]) if "label_mode" in meta else "unknown"
    mass_info = get_mass_meta(meta)

    # Run summary (aligned with GAN sampler schema)
    run_summary = {
        "arch": "diffusion_vpsde",
        "data_path": data_path,
        "meta_path": meta_path,
        "checkpoint_path": checkpoint_path,
        "selection": {
            **selection_meta,
            "requested_sample_idx": sample_idx,
            "requested_n_random": n_random,
            "selected_sample_indices": [int(i) for i in selected_indices],
            "num_selected": len(selected_indices),
            "subset_type": subset,
        },
        "dataset_metadata": {
            "cond_dim": int(cond_dim),
            "conditioning_spec": conditioning_spec,
            "cond_slices": cond_slices,
            "shape3d": [int(img_size), int(img_size), int(img_size)],
            "label_mode": label_mode,
            "mass_info": mass_info,
        },
        "sampler_config": {
            "mode": mode,
            "n_per_condition": int(n_per_cond),
            "atol": float(sample_atol),
            "rtol": float(sample_rtol),
            "eps": float(sample_eps),
            "img_size": int(img_size),
            "unet_ch1_dim": int(unet_ch1_dim),
            "unet_ch2_dim": int(unet_ch2_dim),
            "unet_ch3_dim": int(unet_ch3_dim),
            "t_embed_dim": int(t_embed_dim),
            "cond_embed_dim": int(cond_embed_dim),
            "cond_ch": int(cond_ch),
        },
        "parts": [],
    }

    # Diffusion sampling per condition
    with torch.no_grad():
        for sample_idx in selected_indices:
            voxel_arr = X[sample_idx]         # (1,D,H,W) in [-1,1]
            cond_vec = C[sample_idx]          # (cond_dim,)
            label_val = int(y[sample_idx])
            cond_str = cond_strs[int(sample_idx)]
            sample_info = sample_infos[int(sample_idx)] if sample_infos is not None else {}
            if sample_info is None:
                sample_info = {}

            source_index = sample_info.get("source_index", None) if isinstance(sample_info, dict) else None
            train_mass_fraction = sample_info.get("mass_fraction", None) if isinstance(sample_info, dict) else None
            volume_fraction = sample_info.get("volume_fraction", None) if isinstance(sample_info, dict) else None

            cond_label_bit = None
            if label_mode == "append_to_condition" and cond_dim > 0:
                cond_label_bit = float(cond_vec[-1])

            # Decode BC/load overlay
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

            base_prefix = f"sample_diffusion_idx{sample_idx:05d}"

            # Save reference train voxel (binary mask from [-1,1] field)
            voxel_np = voxel_arr[0]           # (D,H,W), numpy
            train_bin = (voxel_np > 0.0).astype(np.float32)
            train_png = os.path.join(outdir, f"{base_prefix}_train.png")
            plot_voxel_with_conditions(
                train_bin,
                save_path=train_png,
                title=f"TRAIN sample idx={sample_idx} | {mode_tag}",
                bc_points=bc_points,
                load_point=load_point,
                load_vec=load_dir,
            )
            train_bin_npy = os.path.join(outdir, f"{base_prefix}_train_bin.npy")
            np.save(train_bin_npy, train_bin)

            # Build batch condition: repeat same cond_vec n_per_cond times
            n_fake = int(n_per_cond)
            cond_batch = (
                torch.from_numpy(cond_vec.astype(np.float32))
                .to(device)
                .unsqueeze(0)
                .expand(n_fake, -1)
            )

            # ODE sampler call
            x_shape = torch.Size([n_fake, 1, img_size, img_size, img_size])
            x_traj, x_final = ode_sampler_voxel_cond(
                score_model,
                x_shape,
                cond_batch,
                mp_mean,
                mp_std,
                drift,
                mode=mode,
                atol=sample_atol,
                rtol=sample_rtol,
                device=device,
                eps=sample_eps,
            )

            # Continuous + binary fields
            x_samples = x_final.cpu().numpy()          # (n_fake,1,D,H,W)
            x_bin = (x_samples > 0.0).astype(np.float32)

            fake_raw = x_samples[:, 0]                 # (n_fake,D,H,W)
            fake_bin = x_bin[:, 0]                     # (n_fake,D,H,W)

            # Mass summary (GAN-style)
            fake_mass_summary = summarize_fake_mass_from_bin(fake_bin, meta)

            # Save arrays
            fake_raw_npy = os.path.join(outdir, f"{base_prefix}_fake_raw.npy")
            fake_bin_npy = os.path.join(outdir, f"{base_prefix}_fake_bin.npy")
            traj_npy = os.path.join(outdir, f"{base_prefix}_traj.npy")
            np.save(fake_raw_npy, fake_raw)
            np.save(fake_bin_npy, fake_bin)
            np.save(traj_npy, x_traj.cpu().numpy())

            # Plot ensemble
            fake_pngs = []
            for k in range(n_fake):
                fake_png = os.path.join(outdir, f"{base_prefix}_fake{k:02d}.png")

                mass_tag = ""
                if fake_mass_summary is not None:
                    mf = float(fake_mass_summary["mass_fractions"][k])
                    mass_tag = f" | mass={mf:.4f}"

                plot_voxel_with_conditions(
                    fake_bin[k],
                    save_path=fake_png,
                    title=f"DIFFUSION FAKE k={k} | idx={sample_idx} | {mode_tag}{mass_tag}",
                    bc_points=bc_points,
                    load_point=load_point,
                    load_vec=load_dir,
                )
                fake_pngs.append(os.path.basename(fake_png))

            # Part record (aligned with GAN schema, plus diffusion extras)
            part_record = {
                "sample_array_index": int(sample_idx),
                "source_index": source_index,
                "train_mass_fraction": train_mass_fraction,
                "volume_fraction": volume_fraction,
                "label_value": label_val,
                "cond_dim": int(cond_dim),
                "cond_label_bit": cond_label_bit,
                "cond_str": cond_str,
                "cond_vector": cond_vec.tolist(),
                "conditioning_spec": spec,
                "cond_slices": cond_slices,
                "mode_tag": mode_tag,
                "train_plot_png": os.path.basename(train_png),
                "train_bin_npy": os.path.basename(train_bin_npy),
                "n_per_condition": int(n_fake),
                "fake_plot_pngs": fake_pngs,
                "fake_raw_npy": os.path.basename(fake_raw_npy),
                "fake_bin_npy": os.path.basename(fake_bin_npy),
                "traj_npy": os.path.basename(traj_npy),
                "bc_points": bc_points.tolist() if bc_points is not None else None,
                "load_point": load_point.tolist() if load_point is not None else None,
                "load_dir": load_dir.tolist() if load_dir is not None else None,
                "mass_summary": None,
            }

            if cond_slices is not None:
                part_record.update(decode_condition_fields(cond_vec, cond_slices))

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

            run_summary["parts"].append(part_record)

    summary_json = os.path.join(outdir, "diffusion_sampler_run_summary.json")
    with open(summary_json, "w") as f:
        json.dump(run_summary, f, indent=2)

    print(f"[diffusion-sampler] saved run summary to {summary_json}")


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Sample VPSDE diffusion checkpoints on BC/load-conditioned voxel data."
    )

    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to the diffusion train-data .npy file (same format as diffusion trainer).")
    parser.add_argument("--meta-path", type=str, default=None,
                        help="Optional explicit meta .npz path; if omitted, inferred as data_path[:-4] + '_meta.npz'.")
    parser.add_argument("--checkpoint-path", type=str, required=True,
                        help="Path to a diffusion checkpoint .pth file (e.g., ckpt_best.pth).")
    parser.add_argument("--outdir", type=str, required=True,
                        help="Output directory for PNGs + arrays + JSON summary.")

    # Model arch args (must match trainer used for checkpoint)
    parser.add_argument("--img-size", type=int, default=32,
                        help="Cubic voxel size (D=H=W), must match training.")
    parser.add_argument("--unet-ch1-dim", type=int, default=32)
    parser.add_argument("--unet-ch2-dim", type=int, default=64)
    parser.add_argument("--unet-ch3-dim", type=int, default=128)
    parser.add_argument("--t-embed-dim", type=int, default=128)
    parser.add_argument("--cond-embed-dim", type=int, default=128)
    parser.add_argument("--cond-ch", type=int, default=8)
    parser.add_argument("--mode", type=str, default="X0",
                        help="Diffusion mode: 'X0' (only X0 supported in sampler).")

    # Sampler ODE settings (mirroring trainer defaults)
    parser.add_argument("--sample-atol", type=float, default=1e-4)
    parser.add_argument("--sample-rtol", type=float, default=1e-4)
    parser.add_argument("--sample-eps", type=float, default=1e-3)

    # Condition selection + ensemble size
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sample-idx", type=int, default=None,
                       help="Sample exactly one condition index.")
    group.add_argument("--n-random", type=int, default=None,
                       help="Sample N randomly chosen conditions.")
    parser.add_argument("--n-per-cond", type=int, default=4,
                        help="Number of generated samples per selected condition.")
    parser.add_argument("--rng-seed", type=int, default=0,
                        help="Random seed used when --n-random is provided.")

    parser.add_argument("--subset", type=str, choices=["positive_only", "all"], default="positive_only",
                        help="Pool used to choose reference conditions (default: positive_only).")

    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use: 'cuda' or 'cpu'.")

    args = parser.parse_args()

    run_diffusion_sampler(
        data_path=args.data_path,
        meta_path=args.meta_path,
        checkpoint_path=args.checkpoint_path,
        outdir=args.outdir,
        img_size=args.img_size,
        unet_ch1_dim=args.unet_ch1_dim,
        unet_ch2_dim=args.unet_ch2_dim,
        unet_ch3_dim=args.unet_ch3_dim,
        t_embed_dim=args.t_embed_dim,
        cond_embed_dim=args.cond_embed_dim,
        cond_ch=args.cond_ch,
        mode=args.mode,
        sample_atol=args.sample_atol,
        sample_rtol=args.sample_rtol,
        sample_eps=args.sample_eps,
        sample_idx=args.sample_idx,
        n_random=args.n_random,
        n_per_cond=args.n_per_cond,
        rng_seed=args.rng_seed,
        subset=args.subset,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()