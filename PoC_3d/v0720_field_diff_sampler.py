"""
Standalone ensemble sampler for the Perceiver field-diffusion model.

Loads a checkpoint + its hparams.json, picks conditions from the dataset,
and generates an ensemble of samples per condition.

Extras enabled by the field formulation:
  --upsample F        evaluate the learned field on an F-times finer grid
                      than the part's native resolution (same physical
                      domain, same isotropic frame).
  --guidance_scale w  classifier-free guidance (works because the trainer
                      drops the condition with prob cond_drop_prob):
                        y0hat = (1+w)*y0hat_cond - w*y0hat_uncond
                      w=0 -> plain conditional sampling (max diversity),
                      w in [1,4] -> tighter adherence to the condition.

Outputs per condition: GT png, member pngs, and an .npz with the
continuous fields + binarized fields + metadata.

Requires v0720_neural_diff_trainer_fixed.py and
v0720_field_diff_perceiver_trainer.py in the same directory.
"""

import argparse
import functools
import json
from pathlib import Path

import numpy as np
import torch
from scipy import integrate

from v0720_neural_diff_trainer_fixed import (
    NeuralFieldDataset3D,
    marginal_prob_mean,
    marginal_prob_std,
    drift_coeff,
    plot_voxel_with_overlays_implicit,
    decode_condition_overlay,
)
from v0720_field_diff_perceiver_trainer import PerceiverScoreField


def parse_args():
    p = argparse.ArgumentParser(description="Sample ensembles from a trained field diffusion model")
    p.add_argument("--ckpt", type=str, required=True,
                   help="Path to ckpt_best.pth (or any ckpt_*.pth)")
    p.add_argument("--hparams", type=str, default=None,
                   help="Path to hparams.json; default: <ckpt_dir>/../hparams.json")
    p.add_argument("--data_file", type=str, required=True)
    p.add_argument("--meta_path", type=str, default=None)
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--device", default="cuda", type=str)
    p.add_argument("--seed", default=0, type=int)

    # which conditions
    p.add_argument("--indices", type=str, default="",
                   help="Comma-separated dataset indices, e.g. '3,17,102'. "
                        "Empty -> draw --n_random random conditions.")
    p.add_argument("--n_random", default=4, type=int)

    # ensemble / sampling
    p.add_argument("--ensemble_size", default=8, type=int)
    p.add_argument("--upsample", default=1.0, type=float)
    p.add_argument("--guidance_scale", default=0.0, type=float)
    p.add_argument("--sample_context", default=4096, type=int)
    p.add_argument("--eval_chunk", default=16384, type=int)
    p.add_argument("--sample_atol", default=1e-4, type=float)
    p.add_argument("--sample_rtol", default=1e-4, type=float)
    p.add_argument("--sample_eps", default=1e-3, type=float)
    return p.parse_args()


def make_coords(shape, upsample=1.0):
    """
    Isotropic coordinates for (possibly upsampled) grid over the SAME
    physical domain as the native grid: index i on an axis of native size N
    lives at u = i/(max_dim-1) in [0,1]; upsampled grid spans the same
    interval with round(N*upsample) points. Returned in the SIREN frame
    [-1,1]. Matches NeuralFieldDataset3D exactly at upsample=1.
    """
    Nx, Ny, Nz = shape
    max_dim = float(max(Nx, Ny, Nz))
    denom = max(max_dim - 1.0, 1.0)
    sizes = [max(2, int(round(n * upsample))) for n in (Nx, Ny, Nz)]
    axes = [
        torch.linspace(0.0, (n - 1) / denom, m)
        for n, m in zip((Nx, Ny, Nz), sizes)
    ]
    xx, yy, zz = torch.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    coords01 = torch.stack([xx, yy, zz], dim=-1).view(-1, 3)
    return 2.0 * coords01 - 1.0, tuple(sizes)


def ode_sampler_field_cfg(model, coords, cond, mean_fn, std_fn, drift_fn,
                          guidance_scale=0.0, n_context=4096, eval_chunk=16384,
                          atol=1e-4, rtol=1e-4, eps=1e-3, device="cuda",
                          generator=None):
    """ODE sampler with optional classifier-free guidance."""
    N_pts = coords.shape[0]
    coords_dev = coords.to(device)
    cond_b = cond.unsqueeze(0).to(device)
    uncond_b = torch.zeros_like(cond_b)

    n_ctx = min(n_context, N_pts)
    perm = torch.randperm(N_pts, generator=generator)
    ctx_idx = perm[:n_ctx].to(device)
    coords_c = coords_dev[ctx_idx].unsqueeze(0)

    init_y = torch.randn(N_pts, 1, generator=generator).to(device)

    use_cfg = abs(guidance_scale) > 1e-8

    def y0hat_full(y, t):
        y_c = y[ctx_idx].unsqueeze(0)
        lat_c = model.compute_latents(coords_c, y_c, t, cond_b)
        lat_u = model.compute_latents(coords_c, y_c, t, uncond_b) if use_cfg else None
        outs = []
        for s in range(0, N_pts, eval_chunk):
            e = min(s + eval_chunk, N_pts)
            cq = coords_dev[s:e].unsqueeze(0)
            yq = y[s:e].unsqueeze(0)
            pred_c = model.decode_queries(lat_c, cq, yq, t, cond_b)
            if use_cfg:
                pred_u = model.decode_queries(lat_u, cq, yq, t, uncond_b)
                pred = (1.0 + guidance_scale) * pred_c - guidance_scale * pred_u
            else:
                pred = pred_c
            outs.append(pred.squeeze(0))
        return torch.cat(outs, dim=0)

    def score_eval(y_flat, t_scalar):
        y = torch.tensor(y_flat, device=device, dtype=torch.float32).view(N_pts, 1)
        t = torch.tensor([t_scalar], device=device, dtype=torch.float32)
        with torch.no_grad():
            mean = mean_fn(t)[:, None]
            std = std_fn(t)[:, None]
            y0hat = y0hat_full(y, t)
            score = -(y - mean * y0hat) / (std ** 2)
        return score.cpu().numpy().reshape(-1).astype(np.float64)

    def ode_func(t_scalar, y_flat):
        t_torch = torch.tensor(t_scalar, device=device, dtype=torch.float32)
        drift = drift_fn(t_torch).cpu().numpy()
        return drift * (y_flat + score_eval(y_flat, t_scalar))

    res = integrate.solve_ivp(
        ode_func, (1.0, eps), init_y.cpu().numpy().reshape(-1),
        rtol=rtol, atol=atol, method="RK45",
    )
    return torch.tensor(res.y[:, -1], dtype=torch.float32).view(N_pts, 1)


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ckpt_path = Path(args.ckpt)
    hparams_path = Path(args.hparams) if args.hparams else ckpt_path.parent.parent / "hparams.json"
    with open(hparams_path) as f:
        hp = json.load(f)
    print(f"Loaded hparams from {hparams_path}")

    meta_path = args.meta_path or args.data_file[:-4] + "_meta.npz"
    meta = np.load(meta_path, allow_pickle=True)

    dataset = NeuralFieldDataset3D(args.data_file)

    model = PerceiverScoreField(
        cond_dim=dataset.cond_dim,
        d_model=hp["d_model"],
        n_latents=hp["n_latents"],
        n_heads=hp["n_heads"],
        n_self_layers=hp["n_self_layers"],
        n_dec_layers=hp["n_dec_layers"],
        n_freq=hp["coord_fourier_freqs"],
        freq_scale=hp["coord_fourier_scale"],
        t_embed_dim=hp["t_embed_dim"],
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f"Loaded checkpoint {ckpt_path}")

    if args.guidance_scale != 0.0 and hp.get("cond_drop_prob", 0.0) <= 0.0:
        print("WARNING: guidance_scale != 0 but model was trained without "
              "condition dropout; unconditional branch is untrained.")

    if args.indices.strip():
        chosen = [int(s) for s in args.indices.split(",") if s.strip()]
    else:
        rng = np.random.default_rng(args.seed)
        chosen = list(rng.choice(len(dataset), size=args.n_random, replace=False))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    mean_fn = functools.partial(marginal_prob_mean, bmin=0.1, bmax=20.0)
    std_fn = functools.partial(marginal_prob_std, bmin=0.1, bmax=20.0)
    drift_fn = functools.partial(drift_coeff, bmin=0.1, bmax=20.0)

    for idx in chosen:
        idx = int(idx)
        x_vox = dataset.voxels[idx]
        native_shape = x_vox.shape
        cond = torch.from_numpy(dataset.conds[idx])
        overlay = decode_condition_overlay(dataset.conds[idx], meta)

        coords, grid_shape = make_coords(native_shape, upsample=args.upsample)
        print(f"[cond {idx}] native {native_shape} -> sampling grid {grid_shape} "
              f"({coords.shape[0]} pts), ensemble={args.ensemble_size}, "
              f"cfg w={args.guidance_scale}")

        cond_dir = outdir / f"cond_{idx:06d}"
        cond_dir.mkdir(parents=True, exist_ok=True)

        plot_voxel_with_overlays_implicit(
            x_vox, overlay["bc_points"], overlay["load_point"], overlay["load_dir"],
            title=f"GT idx={idx} | shape={native_shape}",
            save_path=cond_dir / "gt.png",
        )

        fields, bins = [], []
        for e in range(args.ensemble_size):
            gen = torch.Generator().manual_seed(args.seed * 100003 + idx * 1009 + e)
            with torch.no_grad():
                y_final = ode_sampler_field_cfg(
                    model, coords, cond, mean_fn, std_fn, drift_fn,
                    guidance_scale=args.guidance_scale,
                    n_context=args.sample_context,
                    eval_chunk=args.eval_chunk,
                    atol=args.sample_atol, rtol=args.sample_rtol,
                    eps=args.sample_eps, device=device, generator=gen,
                )
            field = y_final.cpu().numpy().reshape(grid_shape)
            x_bin = (field > 0.0).astype(np.uint8)
            fields.append(field.astype(np.float32))
            bins.append(x_bin)

            plot_voxel_with_overlays_implicit(
                x_bin.astype(np.float32),
                overlay["bc_points"], overlay["load_point"], overlay["load_dir"],
                title=f"idx={idx} member {e} | grid={grid_shape} | w={args.guidance_scale}",
                save_path=cond_dir / f"member_{e:02d}.png",
            )
            mfrac = float(x_bin.mean())
            print(f"  member {e}: mass fraction {mfrac:.4f}")

        np.savez_compressed(
            cond_dir / "ensemble.npz",
            fields=np.stack(fields),
            binaries=np.stack(bins),
            native_shape=np.asarray(native_shape, dtype=np.int32),
            grid_shape=np.asarray(grid_shape, dtype=np.int32),
            cond_vec=dataset.conds[idx],
            cond_str=str(dataset.cond_strs[idx]),
            source_index=idx,
            guidance_scale=float(args.guidance_scale),
            upsample=float(args.upsample),
        )
        print(f"  saved {cond_dir}/ensemble.npz")

    print(f"Done. Results in {outdir}")


if __name__ == "__main__":
    main()