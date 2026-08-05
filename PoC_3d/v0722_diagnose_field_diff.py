#!/usr/bin/env python
"""
v0722_diagnose_field_diff.py

Figure out WHY the perceiver field-diffusion samples are bad, before spending
another 24h of GPU on it. Four tests, cheapest first:

  1. ctx-ablation : does the model actually USE the context set, or has it
                    collapsed to per-point regression?  (the big one)
  2. ctx-sweep    : val loss vs. n_context at eval time. Tells you whether
                    more context helps at all, and how big the train(1024)
                    vs. sample(4096) mismatch is.
  3. sample-stats : run the sampler, then measure occupancy histogram, mass
                    fraction vs. conditioned VF, ensemble diversity (IoU),
                    and a speckle score. Distinguishes "wrong topology" from
                    "conditional mean + white noise".
  4. overfit      : train a FRESH model on K parts with FULL-grid context for
                    N steps. If it cannot memorize 16 parts, you have a bug,
                    not a data/compute problem. Run this one first if you
                    only have time for one.

Drop in ~/TO_Gan/PoC_3d/ next to the other v0720_* files.

Examples
--------
python v0722_diagnose_field_diff.py overfit \
    --data_file $DATA_FILE --nparts 16 --steps 4000

python v0722_diagnose_field_diff.py ctx-ablation \
    --run_dir /xdisk/hdb/emcdugald/checkpoints/field_diff/perceiverFieldDiff_...

python v0722_diagnose_field_diff.py ctx-sweep --run_dir <run_dir>

python v0722_diagnose_field_diff.py sample-stats --run_dir <run_dir> \
    --n_conditions 4 --ensemble_size 4
"""

import argparse
import functools
import json
import math
from pathlib import Path

import numpy as np
import torch

from v0720_field_diff_perceiver_trainer import (
    NeuralFieldDataset3D,
    make_loaders_implicit,
    marginal_prob_mean,
    marginal_prob_std,
    drift_coeff,
)
from v0720_field_diff_perceiver_trainer import PerceiverScoreField

BMIN, BMAX = 0.1, 20.0


# ---------------------------------------------------------------- utilities

def load_run(run_dir, ckpt_name="ckpt_best.pth", device="cuda",
             data_file=None, nsamples=None):
    """Rebuild dataset + model from a run directory's hparams.json."""
    run_dir = Path(run_dir)
    hp = json.loads((run_dir / "hparams.json").read_text())
    if data_file is not None:
        hp["data_file"] = data_file
    if nsamples is not None:
        hp["nsamples"] = nsamples

    dataset, _, train_idx, val_idx = make_loaders_implicit(
        npy_path=hp["data_file"], batchsize=1, n_points=hp["n_points"],
        nsamples=hp["nsamples"], val_frac=hp["val_frac"],
        seed=hp["seed"], num_workers=0,
    )
    model = PerceiverScoreField(
        cond_dim=dataset.cond_dim,
        d_model=hp["d_model"], n_latents=hp["n_latents"],
        n_heads=hp["n_heads"], n_self_layers=hp["n_self_layers"],
        n_dec_layers=hp["n_dec_layers"],
        n_freq=hp["coord_fourier_freqs"], freq_scale=hp["coord_fourier_scale"],
        t_embed_dim=hp["t_embed_dim"],
    ).to(device)
    sd = torch.load(run_dir / "checkpoints" / ckpt_name, map_location=device)
    model.load_state_dict(sd)
    model.eval()
    print(f"loaded {run_dir.name} / {ckpt_name}")
    return hp, dataset, train_idx, val_idx, model


def batch_from_indices(dataset, idxs, n_points, device, generator=None):
    """Stack n_points random points from each part -> [B,M,3],[B,M,1],[B,C]."""
    cs, ys, cd = [], [], []
    for i in idxs:
        c, y, k = dataset.sample_points(int(i), n_points, generator=generator)
        cs.append(c); ys.append(y); cd.append(k)
    return (torch.stack(cs).to(device),
            torch.stack(ys).to(device),
            torch.stack(cd).to(device))


def full_grid_batch(dataset, idx, device):
    coords, y0, cond = dataset.get_full_grid(int(idx), device=device)
    return coords, y0, cond


def t_buckets(n=8):
    edges = np.linspace(0.0, 1.0, n + 1)
    return list(zip(edges[:-1], edges[1:]))


# ------------------------------------------------------ test 1: ctx ablation

@torch.no_grad()
def test_ctx_ablation(args):
    """
    For each noise level bucket, compute x0-MSE under three context regimes:
      real   : the true noisy values at the context coordinates
      zeros  : context coordinates kept, values replaced by 0
      shuf   : context values taken from a DIFFERENT part in the batch

    Interpretation
    --------------
    real << zeros  ->  the latents carry real information. Good.
    real ~= zeros  ->  the model ignores the context and is doing per-point
                       regression y0hat = f(coord, y_q, t, cond). This is the
                       collapse mode; samples will be conditional-mean + noise
                       no matter how long you train.
    real ~= shuf   ->  same conclusion, and stronger evidence (a wrong context
                       should actively hurt if the model were using it).
    """
    device = args.device
    hp, dataset, _, val_idx, model = load_run(
        args.run_dir, args.ckpt, device, args.data_file, args.nsamples)
    mean_fn = functools.partial(marginal_prob_mean, bmin=BMIN, bmax=BMAX)
    std_fn = functools.partial(marginal_prob_std, bmin=BMIN, bmax=BMAX)

    M, Mc = hp["n_points"], hp["n_context"]
    B = args.batchsize
    g = torch.Generator().manual_seed(0)
    torch.manual_seed(0)

    rows = []
    for lo, hi in t_buckets(args.n_t_buckets):
        acc = {"real": 0.0, "zeros": 0.0, "shuf": 0.0}
        n = 0
        for rep in range(args.reps):
            sel = np.random.default_rng(1000 + rep).choice(
                val_idx, size=min(B, len(val_idx)), replace=False)
            coords, y0, cond = batch_from_indices(dataset, sel, M, device, g)
            t = (torch.rand(len(sel), device=device) * (hi - lo) + lo).clamp(1e-5, 1.0)
            z = torch.randn_like(y0)
            ms, st = mean_fn(t)[:, None, None], std_fn(t)[:, None, None]
            y_t = y0 * ms + z * st

            cc, yc = coords[:, :Mc], y_t[:, :Mc]
            variants = {
                "real": yc,
                "zeros": torch.zeros_like(yc),
                "shuf": yc[torch.randperm(yc.shape[0], device=device)],
            }
            for k, ycv in variants.items():
                yhat = model(coords, y_t, t, cond, coords_c=cc, y_c=ycv)
                acc[k] += torch.mean((yhat - y0) ** 2).item()
            n += 1
        rows.append((0.5 * (lo + hi),
                     acc["real"] / n, acc["zeros"] / n, acc["shuf"] / n))

    print("\n  t      MSE(real ctx)   MSE(zero ctx)   MSE(wrong ctx)   gain%")
    print("  " + "-" * 62)
    for tmid, r, z_, s in rows:
        gain = 100.0 * (z_ - r) / max(z_, 1e-12)
        print(f"  {tmid:4.2f}   {r:13.5f}   {z_:13.5f}   {s:14.5f}   {gain:5.1f}")
    print("\n  gain% near 0 across all t  ->  latents unused, pointwise collapse.")
    print("  gain% large at mid/low t   ->  the mechanism works; you are")
    print("                                undertrained or the sampler mismatches.\n")


# --------------------------------------------------------- test 2: ctx sweep

@torch.no_grad()
def test_ctx_sweep(args):
    """
    Val x0-MSE as a function of the number of context points used at eval time.
    Training used n_context = hp['n_context']; sampling used sample_context.
    A curve that keeps dropping past the training value means the model
    generalizes to bigger context and you should just train with more.
    A curve that is FLAT means extra context buys nothing (collapse).
    A curve that gets WORSE past the training value means train/test density
    mismatch is actively hurting your samples.
    """
    device = args.device
    hp, dataset, _, val_idx, model = load_run(
        args.run_dir, args.ckpt, device, args.data_file, args.nsamples)
    mean_fn = functools.partial(marginal_prob_mean, bmin=BMIN, bmax=BMAX)
    std_fn = functools.partial(marginal_prob_std, bmin=BMIN, bmax=BMAX)

    sizes = [int(s) for s in args.ctx_sizes.split(",")]
    Mq = args.n_query
    torch.manual_seed(0)
    sel = np.random.default_rng(0).choice(
        val_idx, size=min(args.n_parts, len(val_idx)), replace=False)

    print(f"\n  train n_context = {hp['n_context']}, "
          f"sample_context = {hp.get('sample_context')}")
    print(f"\n  n_ctx     MSE(t~U[0,1])   MSE(t<0.3)")
    print("  " + "-" * 40)

    for Mc in sizes:
        tot_all, tot_lo, n_all, n_lo = 0.0, 0.0, 0, 0
        for idx in sel:
            coords, y0, cond = full_grid_batch(dataset, idx, device)
            N = coords.shape[0]
            if Mc > N:
                continue
            for rep in range(args.reps):
                gg = torch.Generator(device="cpu").manual_seed(rep)
                perm = torch.randperm(N, generator=gg).to(device)
                ci, qi = perm[:Mc], perm[Mc:Mc + Mq]
                if qi.numel() < Mq:
                    qi = perm[:Mq]
                for band, t_val in (("all", 0.05 + 0.9 * rep / max(args.reps - 1, 1)),
                                    ("lo", 0.05 + 0.25 * rep / max(args.reps - 1, 1))):
                    t = torch.tensor([t_val], device=device)
                    zz = torch.randn(N, 1, device=device,
                                     generator=torch.Generator(device=device).manual_seed(rep))
                    y_t = y0 * mean_fn(t) + zz * std_fn(t)
                    yhat = model(coords[qi][None], y_t[qi][None], t, cond[None],
                                 coords_c=coords[ci][None], y_c=y_t[ci][None])
                    mse = torch.mean((yhat - y0[qi][None]) ** 2).item()
                    if band == "all":
                        tot_all += mse; n_all += 1
                    else:
                        tot_lo += mse; n_lo += 1
        print(f"  {Mc:6d}   {tot_all/max(n_all,1):13.5f}   {tot_lo/max(n_lo,1):10.5f}")
    print()


# ------------------------------------------------------ test 3: sample stats

def speckle_fraction(binvox):
    """Fraction of occupied voxels with <2 occupied 6-neighbours.
    Real TO parts are ~0.00-0.02. Score-collapse speckle is >0.3."""
    b = binvox.astype(np.uint8)
    p = np.pad(b, 1)
    nb = (p[2:, 1:-1, 1:-1] + p[:-2, 1:-1, 1:-1] +
          p[1:-1, 2:, 1:-1] + p[1:-1, :-2, 1:-1] +
          p[1:-1, 1:-1, 2:] + p[1:-1, 1:-1, :-2])
    occ = b > 0
    if occ.sum() == 0:
        return float("nan")
    return float((nb[occ] < 2).mean())


def find_vf(cond_vec, meta):
    try:
        cs = json.loads(str(meta["cond_slices_json"]))
    except Exception:
        return None
    for key in ("vf", "volume_fraction", "VF", "vfrac", "vol_frac"):
        if key in cs:
            s0, s1 = cs[key]
            return float(np.asarray(cond_vec)[s0:s1].ravel()[0])
    return None


@torch.no_grad()
def test_sample_stats(args):
    from v0720_field_diff_train_fast import heun_sampler_field_batched

    device = args.device
    hp, dataset, _, val_idx, model = load_run(
        args.run_dir, args.ckpt, device, args.data_file, args.nsamples)
    meta = np.load(hp["meta_path"], allow_pickle=True)
    mean_fn = functools.partial(marginal_prob_mean, bmin=BMIN, bmax=BMAX)
    std_fn = functools.partial(marginal_prob_std, bmin=BMIN, bmax=BMAX)
    drift_fn = functools.partial(drift_coeff, bmin=BMIN, bmax=BMAX)

    sel = np.random.default_rng(0).choice(
        val_idx, size=min(args.n_conditions, len(val_idx)), replace=False)

    for k, idx in enumerate(sel):
        coords, y0, cond = dataset.get_full_grid(int(idx), device="cpu")
        vox = dataset.voxels[int(idx)]
        Nx, Ny, Nz = vox.shape
        vf_cond = find_vf(dataset.conds[int(idx)], meta)

        y_all = heun_sampler_field_batched(
            model, coords, cond, mean_fn, std_fn, drift_fn,
            ensemble_size=args.ensemble_size, n_steps=args.n_steps,
            n_context=args.sample_context or hp["sample_context"],
            eval_chunk=hp["eval_chunk"], device=device,
            base_seed=1000 * int(idx), guidance_scale=args.guidance_scale,
        )                                    # [E, N, 1]

        f = y_all[..., 0].numpy()            # [E, N]
        bins = (f > 0.0).reshape(args.ensemble_size, Nx, Ny, Nz)
        gt_mf = float(vox.mean())

        print(f"\n=== condition {k} (idx {idx}) shape=({Nx},{Ny},{Nz}) ===")
        print(f"  GT mass fraction      : {gt_mf:.3f}"
              + (f"   cond VF: {vf_cond:.3f}" if vf_cond is not None else ""))
        print(f"  GT speckle fraction   : {speckle_fraction(vox > 0.5):.3f}")
        print(f"  field |mean| / std    : {np.abs(f).mean():.3f} / {f.std():.3f}")
        q = np.quantile(f, [0.01, 0.25, 0.5, 0.75, 0.99])
        print(f"  field quantiles       : " + " ".join(f"{v:+.2f}" for v in q))
        print(f"  frac |field|>0.5      : {(np.abs(f) > 0.5).mean():.3f}"
              "   (near 1.0 = saturated/decided, near 0 = mushy)")
        for e in range(args.ensemble_size):
            print(f"   member {e}: mass={bins[e].mean():.3f}  "
                  f"speckle={speckle_fraction(bins[e]):.3f}")
        ious = []
        for a in range(args.ensemble_size):
            for b in range(a + 1, args.ensemble_size):
                i = np.logical_and(bins[a], bins[b]).sum()
                u = np.logical_or(bins[a], bins[b]).sum()
                ious.append(i / max(u, 1))
        if ious:
            print(f"  pairwise member IoU   : mean {np.mean(ious):.3f} "
                  f"(1.0 = identical, ~mass_frac = independent coin flips)")
        np.savez(Path(args.out_dir) / f"diag_cond{k:02d}.npz",
                 field=f, binary=bins, gt=vox)

    print(f"\nsaved raw fields to {args.out_dir}\n")
    print("Read the numbers like this:")
    print("  speckle high + IoU ~ mass_frac   -> per-point independence:")
    print("     the score has factorized. More epochs will NOT fix it.")
    print("  speckle low + IoU ~ 1.0          -> mode collapse, no diversity.")
    print("  speckle low + IoU 0.5-0.8        -> working as intended.")
    print("  |field| tiny everywhere          -> model is outputting E[y0|cond];")
    print("     the binarization threshold is then just amplifying noise.\n")


# ---------------------------------------------------------- test 4: overfit

def test_overfit(args):
    """
    Train a FRESH model on a handful of parts, with FULL-grid context, and see
    if it can memorize them. Separates 'structural problem' from 'undertrained'.

    IMPORTANT: run this with --zero_cond.

    With the real conditioning vector, the model can memorize cond -> part and
    ignore the context set entirely, which is exactly the failure mode we are
    trying to detect. Zeroing cond makes the noisy context the ONLY way to tell
    the parts apart, so the loss becomes a direct measurement of the latent
    pathway.

    Loss is bucketed by t, because a single number is meaningless when t is
    redrawn every step (t~0 is trivial, t~1 is near-hopeless by construction).
    """
    device = args.device
    dataset, _, train_idx, _ = make_loaders_implicit(
        npy_path=args.data_file, batchsize=1, n_points=args.n_points,
        nsamples=args.nsamples or 512, val_frac=0.05, seed=0, num_workers=0,
    )
    sizes = [(int(i), int(np.prod(dataset.voxels[int(i)].shape))) for i in train_idx]
    sizes.sort(key=lambda p: p[1])
    parts = [i for i, _ in sizes[:args.nparts]]
    grids = [int(np.prod(dataset.voxels[p].shape)) for p in parts]
    print(f"overfitting {len(parts)} parts, grid sizes {grids}")
    print(f"conditioning: {'ZEROED (context-only test)' if args.zero_cond else 'REAL (weak test!)'}")

    # no-information baselines, in the same units as the loss
    mfs = [float(dataset.voxels[p].mean()) for p in parts]
    const_mse = float(np.mean([1.0 - (2 * m - 1) ** 2 for m in mfs]))
    print(f"baseline MSE, predict 0 everywhere        : 1.000")
    print(f"baseline MSE, predict per-part mean       : {const_mse:.3f}")
    print(f"  -> anything meaningfully below {const_mse:.3f} at high t is real "
          f"structure.\n")

    mean_fn = functools.partial(marginal_prob_mean, bmin=BMIN, bmax=BMAX)
    std_fn = functools.partial(marginal_prob_std, bmin=BMIN, bmax=BMAX)

    model = PerceiverScoreField(
        cond_dim=dataset.cond_dim, d_model=args.d_model,
        n_latents=args.n_latents, n_heads=8, n_self_layers=4, n_dec_layers=2,
        n_freq=64, freq_scale=args.freq_scale, t_embed_dim=128,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"params {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    n_bands = 5
    hist = {b: [] for b in range(n_bands)}
    log_every = max(args.steps // 20, 1)

    for step in range(args.steps):
        idx = parts[step % len(parts)]
        coords, y0, cond = dataset.get_full_grid(idx, device=device)
        if args.zero_cond:
            cond = torch.zeros_like(cond)
        N = coords.shape[0]
        t = torch.rand(1, device=device) * (1 - 1e-5) + 1e-5
        z = torch.randn_like(y0)
        y_t = y0 * mean_fn(t) + z * std_fn(t)

        Mc = min(args.ctx, N)
        perm = torch.randperm(N, device=device)
        ci, qi = perm[:Mc], perm[:args.n_points]
        yhat = model(coords[qi][None], y_t[qi][None], t, cond[None],
                     coords_c=coords[ci][None], y_c=y_t[ci][None])
        loss = torch.mean((yhat - y0[qi][None]) ** 2)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        hist[min(int(t.item() * n_bands), n_bands - 1)].append(loss.item())
        if step % log_every == 0 and step > 0:
            cells = []
            for b in range(n_bands):
                v = hist[b][-40:]
                lab = f"t{b/n_bands:.1f}-{(b+1)/n_bands:.1f}"
                cells.append(f"{lab}:{np.mean(v):.3f}" if v else f"{lab}:  -  ")
            print(f"  step {step:6d}  " + "  ".join(cells))

    print("\nHow to read the final row:")
    print("  low-t bands (0.0-0.4) near 0        : expected, this is mostly")
    print("      copying y_q. Not evidence of anything.")
    print("  mid/high-t bands (0.4-1.0) still at")
    print(f"      ~{const_mse:.2f} or above          : the context is being ignored.")
    print("      Structural problem -- more epochs will not fix it.")
    print("  mid/high-t bands well below")
    print(f"      {const_mse:.2f} (say <0.4)             : the latent pathway works.")
    print("      Your 300-epoch run was undertrained and/or the sampler")
    print("      operates at a context density the model never trained at.\n")


# ------------------------------------------------------------------- driver

def main():
    p = argparse.ArgumentParser()
    p.add_argument("test", choices=["ctx-ablation", "ctx-sweep",
                                    "sample-stats", "overfit"])
    p.add_argument("--run_dir", type=str, default=None)
    p.add_argument("--ckpt", type=str, default="ckpt_best.pth")
    p.add_argument("--data_file", type=str, default=None)
    p.add_argument("--nsamples", type=int, default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out_dir", type=str, default=".")

    p.add_argument("--batchsize", type=int, default=16)
    p.add_argument("--reps", type=int, default=8)
    p.add_argument("--n_t_buckets", type=int, default=8)

    p.add_argument("--ctx_sizes", type=str,
                   default="256,1024,2048,4096,8192,16384")
    p.add_argument("--n_query", type=int, default=2048)
    p.add_argument("--n_parts", type=int, default=8)

    p.add_argument("--n_conditions", type=int, default=4)
    p.add_argument("--ensemble_size", type=int, default=4)
    p.add_argument("--n_steps", type=int, default=120)
    p.add_argument("--sample_context", type=int, default=None)
    p.add_argument("--guidance_scale", type=float, default=0.0)

    p.add_argument("--nparts", type=int, default=16)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--n_points", type=int, default=2048)
    p.add_argument("--ctx", type=int, default=100000)   # full grid by default
    p.add_argument("--zero_cond", action="store_true",
                   help="zero the conditioning vector during overfit, so the "
                        "context set is the only way to distinguish parts")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_latents", type=int, default=256)
    p.add_argument("--freq_scale", type=float, default=8.0)

    args = p.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    {"ctx-ablation": test_ctx_ablation,
     "ctx-sweep": test_ctx_sweep,
     "sample-stats": test_sample_stats,
     "overfit": test_overfit}[args.test](args)


if __name__ == "__main__":
    main()