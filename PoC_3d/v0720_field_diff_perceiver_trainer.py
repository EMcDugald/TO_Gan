"""
Field diffusion with cross-point attention (Diffusion-Probabilistic-Fields /
PerceiverIO style) for BC/load-conditioned NITO parts with mixed shapes.

Why this exists: a strictly pointwise score net factorizes the reverse
diffusion into independent per-point processes, which collapses to per-point
regression (one averaged part + uncorrelated speckle; no topological
diversity). Here the score at every query point depends on a CONTEXT SET of
(coordinate, noisy value) pairs from the same part, routed through a fixed
set of latent tokens:

    context (coord, y_t) --cross-attn--> latents --self-attn--> latents
    query   (coord, y_t) --cross-attn--> latents --> y0hat(query)

Global structure is computed in the latent self-attention, so correlated
noise modes can be amplified into distinct coherent topologies -- the same
mechanism that gives your voxel U-Net its sample diversity -- while the
coordinate-based interface handles arbitrary part shapes.

Key differences from the pointwise trainer:
  * t is SHARED across all points of a part (required: the model couples
    points, and the sampler denoises the whole field at one t).
  * Training draws a random point set per part; a random subset of it is
    the context, all of it is the query set.
  * Sampling fixes a random context-index subset of the full grid once per
    part (so the ODE stays deterministic), re-encodes latents at every
    solver step from the current noisy values at those indices, and decodes
    all grid points in chunks.
  * Optional condition dropout (--cond_drop_prob) enables classifier-free
    guidance at sampling time later if you want to trade diversity/fidelity.

Requires v0720_neural_diff_trainer_fixed.py in the same directory.
"""

import os
from pathlib import Path
import argparse
import functools
import json
from datetime import datetime
import math

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch_ema import ExponentialMovingAverage
from scipy import integrate

from v0720_neural_diff_trainer_fixed import (
    NeuralFieldDataset3D,
    make_loaders_implicit,
    marginal_prob_mean,
    marginal_prob_std,
    drift_coeff,
    GaussianFourierProjection,
    plot_voxel_with_overlays_implicit,
    decode_condition_overlay,
)


# --------- args ---------

def parse_args():
    p = argparse.ArgumentParser(description="Perceiver field diffusion (DPF-style)")
    p.add_argument("--device", default="cuda", type=str)
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--batchsize", default=8, type=int)       # parts per step
    p.add_argument("--n_points", default=2048, type=int)     # query points per part
    p.add_argument("--n_context", default=1024, type=int)    # context points per part
    p.add_argument("--nepochs", default=500, type=int)
    p.add_argument("--lr", default=1e-4, type=float)
    p.add_argument("--cond_drop_prob", default=0.1, type=float)

    # model
    p.add_argument("--d_model", default=256, type=int)
    p.add_argument("--n_latents", default=256, type=int)
    p.add_argument("--n_heads", default=8, type=int)
    p.add_argument("--n_self_layers", default=4, type=int)
    p.add_argument("--n_dec_layers", default=2, type=int)
    p.add_argument("--coord_fourier_freqs", default=64, type=int)
    p.add_argument("--coord_fourier_scale", default=8.0, type=float)
    p.add_argument("--t_embed_dim", default=128, type=int)
    p.add_argument("--loss_weighting", default="Simple", type=str)

    # data
    p.add_argument("--data_file", type=str, required=True)
    p.add_argument("--meta_path", type=str, default=None)
    p.add_argument("--nsamples", default=0, type=int)
    p.add_argument("--val_frac", default=0.1, type=float)

    # logging / ckpts
    p.add_argument("--log_root", type=str, required=True)
    p.add_argument("--save_every_n_epochs", default=25, type=int)
    p.add_argument("--sample_every_n_epochs", default=25, type=int)
    p.add_argument("--sample_num", default=4, type=int)
    p.add_argument("--ensemble_size", default=4, type=int)   # samples per condition
    p.add_argument("--ema_decay", default=0.9999, type=float)
    p.add_argument("--load_ckpt", type=str, default=None)
    p.add_argument("--num_workers", default=0, type=int)
    p.add_argument("--tag", default="", type=str)

    # sampling
    p.add_argument("--sample_atol", default=1e-4, type=float)
    p.add_argument("--sample_rtol", default=1e-4, type=float)
    p.add_argument("--sample_eps", default=1e-3, type=float)
    p.add_argument("--sample_context", default=4096, type=int)  # ctx pts at sampling
    p.add_argument("--eval_chunk", default=16384, type=int)     # query chunk at sampling
    return vars(p.parse_args())


# --------- coordinate features ---------

class FourierFeatures(nn.Module):
    """Random Fourier features for coordinates in [-1,1]^3."""
    def __init__(self, in_dim=3, n_freq=64, scale=8.0):
        super().__init__()
        B = torch.randn(in_dim, n_freq) * scale
        self.register_buffer("B", B)

    @property
    def out_dim(self):
        return 2 * self.B.shape[1]

    def forward(self, x):
        proj = 2.0 * math.pi * (x @ self.B)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


# --------- transformer blocks (pre-LN) ---------

class CrossAttnBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, q, kv):
        h, _ = self.attn(self.ln_q(q), self.ln_kv(kv), self.ln_kv(kv),
                         need_weights=False)
        q = q + h
        q = q + self.mlp(self.ln2(q))
        return q


class SelfAttnBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        h, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x),
                         need_weights=False)
        x = x + h
        x = x + self.mlp(self.ln2(x))
        return x


# --------- Perceiver score field ---------

class PerceiverScoreField(nn.Module):
    """
    X0-prediction score field with global coupling via latent tokens.

    forward(coords_q, y_q, t, cond, coords_c=None, y_c=None):
      coords_q: [B, Mq, 3]   query coordinates (in [-1,1] isotropic frame)
      y_q:      [B, Mq, 1]   noisy values at queries
      t:        [B]
      cond:     [B, cond_dim]
      coords_c / y_c: context set; if None, the query set is the context.
    returns y0hat at queries: [B, Mq, 1]
    """
    def __init__(self, cond_dim, d_model=256, n_latents=256, n_heads=8,
                 n_self_layers=4, n_dec_layers=2,
                 n_freq=64, freq_scale=8.0, t_embed_dim=128):
        super().__init__()
        self.ff = FourierFeatures(3, n_freq, freq_scale)
        tok_in = self.ff.out_dim + 1  # fourier(coord) + noisy value

        self.token_mlp = nn.Sequential(
            nn.Linear(tok_in, d_model), nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.t_embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=t_embed_dim),
            nn.Linear(t_embed_dim, d_model), nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.cond_embed = nn.Sequential(
            nn.Linear(cond_dim, d_model), nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.latents = nn.Parameter(torch.randn(n_latents, d_model) * 0.02)
        self.encode = CrossAttnBlock(d_model, n_heads)
        self.self_blocks = nn.ModuleList(
            [SelfAttnBlock(d_model, n_heads) for _ in range(n_self_layers)]
        )
        self.decode = nn.ModuleList(
            [CrossAttnBlock(d_model, n_heads) for _ in range(n_dec_layers)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def compute_latents(self, coords_c, y_c, t, cond):
        """Encode context set -> latent tokens. Reusable at sampling time."""
        B = coords_c.shape[0]
        te = self.t_embed(t)[:, None, :]            # [B,1,d]
        ce = self.cond_embed(cond)[:, None, :]      # [B,1,d]

        ctx = self.token_mlp(
            torch.cat([self.ff(coords_c), y_c], dim=-1)
        ) + te + ce                                 # [B,Mc,d]

        lat = self.latents[None].expand(B, -1, -1) + te + ce
        lat = self.encode(lat, ctx)
        for blk in self.self_blocks:
            lat = blk(lat)
        return lat

    def decode_queries(self, lat, coords_q, y_q, t, cond):
        te = self.t_embed(t)[:, None, :]
        ce = self.cond_embed(cond)[:, None, :]
        q = self.token_mlp(
            torch.cat([self.ff(coords_q), y_q], dim=-1)
        ) + te + ce
        for blk in self.decode:
            q = blk(q, lat)
        return self.head(q)                         # [B,Mq,1]

    def forward(self, coords_q, y_q, t, cond, coords_c=None, y_c=None):
        if coords_c is None:
            coords_c, y_c = coords_q, y_q
        lat = self.compute_latents(coords_c, y_c, t, cond)
        return self.decode_queries(lat, coords_q, y_q, t, cond)


# --------- loss (shared t per part; context subset of query set) ---------

def field_diffusion_loss(model, y0, coords, cond, n_context,
                         mean_fn, std_fn, eps=1e-5,
                         loss_weighting="Simple", cond_drop_prob=0.0):
    B, M, _ = y0.shape
    device = y0.device

    t = torch.rand(B, device=device) * (1. - eps) + eps      # one t per part
    z = torch.randn_like(y0)
    ms = mean_fn(t)[:, None, None]
    st = std_fn(t)[:, None, None]
    y_t = y0 * ms + z * st

    if cond_drop_prob > 0.0:
        drop = (torch.rand(B, device=device) < cond_drop_prob).float()[:, None]
        cond = cond * (1.0 - drop)

    n_ctx = min(n_context, M)
    # points are already a random subset of the grid; take the first n_ctx
    coords_c, y_c = coords[:, :n_ctx], y_t[:, :n_ctx]

    y0hat = model(coords, y_t, t, cond, coords_c=coords_c, y_c=y_c)

    if loss_weighting == "Simple":
        return torch.mean((y0hat - y0) ** 2)
    elif loss_weighting == "Analytical":
        return torch.mean(((ms / st) * (y0hat - y0)) ** 2)
    raise ValueError("Invalid loss_weighting")


# --------- sampler ---------

def ode_sampler_field(model, coords, cond, mean_fn, std_fn, drift_fn,
                      n_context=4096, eval_chunk=16384,
                      atol=1e-4, rtol=1e-4, eps=1e-3, device="cuda",
                      generator=None):
    """
    Sample the full field. Context indices are drawn ONCE (fixed for the whole
    trajectory so the ODE is deterministic given the initial noise); latents
    are re-encoded from the current noisy values at those indices at every
    solver evaluation.
    """
    N_pts = coords.shape[0]
    coords_dev = coords.to(device)
    cond_b = cond.unsqueeze(0).to(device)

    n_ctx = min(n_context, N_pts)
    if generator is not None:
        ctx_idx = torch.randperm(N_pts, generator=generator)[:n_ctx]
    else:
        ctx_idx = torch.randperm(N_pts)[:n_ctx]
    ctx_idx = ctx_idx.to(device)
    coords_c = coords_dev[ctx_idx].unsqueeze(0)               # [1,Mc,3]

    init_y = torch.randn(N_pts, 1, generator=generator).to(device)

    def y0hat_full(y, t):
        y_c = y[ctx_idx].unsqueeze(0)                         # [1,Mc,1]
        lat = model.compute_latents(coords_c, y_c, t, cond_b)
        outs = []
        for s in range(0, N_pts, eval_chunk):
            e = min(s + eval_chunk, N_pts)
            outs.append(model.decode_queries(
                lat, coords_dev[s:e].unsqueeze(0),
                y[s:e].unsqueeze(0), t, cond_b
            ).squeeze(0))
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


def save_ensembles(model, dataset, indices_pool, model_dir, epoch, config,
                   device, mean_fn, std_fn, meta):
    """For each chosen condition, draw ensemble_size samples so you can
    inspect topological diversity directly."""
    samples_dir = model_dir / f"samples_epoch_{epoch:04d}"
    samples_dir.mkdir(parents=True, exist_ok=True)

    n = min(config["sample_num"], len(indices_pool))
    chosen = np.random.choice(indices_pool, size=n, replace=False)
    drift_fn = functools.partial(drift_coeff, bmin=0.1, bmax=20.0)

    for i, idx in enumerate(chosen):
        coords, _, cond = dataset.get_full_grid(int(idx), device="cpu")
        x_vox = dataset.voxels[int(idx)]
        Nx, Ny, Nz = x_vox.shape
        overlay = decode_condition_overlay(dataset.conds[int(idx)], meta)

        plot_voxel_with_overlays_implicit(
            x_vox, overlay["bc_points"], overlay["load_point"],
            overlay["load_dir"],
            title=f"GT cond {i} | shape=({Nx},{Ny},{Nz})",
            save_path=samples_dir / f"cond{i:02d}_gt.png",
        )

        for e in range(config["ensemble_size"]):
            gen = torch.Generator().manual_seed(1000 * int(idx) + e)
            y_final = ode_sampler_field(
                model, coords, cond, mean_fn, std_fn, drift_fn,
                n_context=config["sample_context"],
                eval_chunk=config["eval_chunk"],
                atol=config["sample_atol"], rtol=config["sample_rtol"],
                eps=config["sample_eps"], device=device, generator=gen,
            )
            x_bin = (y_final.cpu().numpy().reshape(Nx, Ny, Nz) > 0.0)
            plot_voxel_with_overlays_implicit(
                x_bin.astype(np.float32),
                overlay["bc_points"], overlay["load_point"], overlay["load_dir"],
                title=f"Epoch {epoch} cond {i} member {e}",
                save_path=samples_dir / f"cond{i:02d}_member{e}.png",
            )

    print(f"Saved ensembles to {samples_dir}")


# --------- validation ---------

def validate(model, dataset, val_indices, mean_fn, std_fn, device,
             eps, loss_weighting, n_points, n_context, n_repeats=2):
    model.eval()
    total, n_items = 0.0, 0
    with torch.no_grad():
        for idx in val_indices:
            for _ in range(n_repeats):
                coords, y0, cond = dataset.sample_points(int(idx), n_points)
                loss = field_diffusion_loss(
                    model,
                    y0.unsqueeze(0).to(device),
                    coords.unsqueeze(0).to(device),
                    cond.unsqueeze(0).to(device),
                    n_context, mean_fn, std_fn,
                    eps=eps, loss_weighting=loss_weighting,
                    cond_drop_prob=0.0,
                )
                total += loss.item()
                n_items += 1
    model.train()
    return total / max(1, n_items)


# --------- main ---------

def main():
    config = parse_args()
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    log_root = Path(config["log_root"])
    log_root.mkdir(parents=True, exist_ok=True)

    meta_path = config["meta_path"] or config["data_file"][:-4] + "_meta.npz"
    meta = np.load(meta_path, allow_pickle=True)
    if "conditioning_spec_json" in meta:
        conditioning_spec = json.loads(str(meta["conditioning_spec_json"]))
    else:
        conditioning_spec = {}

    extra_tag = f"_{config['tag']}" if config["tag"] else ""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = (
        f"perceiverFieldDiff_bs-{config['batchsize']}_pts-{config['n_points']}_"
        f"ctx-{config['n_context']}_d-{config['d_model']}_L-{config['n_latents']}"
        f"{extra_tag}_{timestamp}"
    )
    model_save_dir = log_root / run_name
    ckpt_dir = model_save_dir / "checkpoints"
    model_save_dir.mkdir(parents=True, exist_ok=False)
    ckpt_dir.mkdir()
    with open(model_save_dir / "hparams.json", "w") as f:
        json.dump({**config, "conditioning_spec": conditioning_spec,
                   "meta_path": str(meta_path)}, f, indent=2)

    dataset, train_loader, train_indices, val_indices = make_loaders_implicit(
        npy_path=config["data_file"],
        batchsize=config["batchsize"],
        n_points=config["n_points"],
        nsamples=config["nsamples"],
        val_frac=config["val_frac"],
        seed=config["seed"],
        num_workers=config["num_workers"],
    )

    mean_fn = functools.partial(marginal_prob_mean, bmin=0.1, bmax=20.0)
    std_fn = functools.partial(marginal_prob_std, bmin=0.1, bmax=20.0)

    model = PerceiverScoreField(
        cond_dim=dataset.cond_dim,
        d_model=config["d_model"],
        n_latents=config["n_latents"],
        n_heads=config["n_heads"],
        n_self_layers=config["n_self_layers"],
        n_dec_layers=config["n_dec_layers"],
        n_freq=config["coord_fourier_freqs"],
        freq_scale=config["coord_fourier_scale"],
        t_embed_dim=config["t_embed_dim"],
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    if config["load_ckpt"] is not None:
        model.load_state_dict(torch.load(config["load_ckpt"], map_location=device))
        print(f"Loaded checkpoint {config['load_ckpt']}")

    optimizer = Adam(model.parameters(), lr=config["lr"])
    ema = ExponentialMovingAverage(model.parameters(), decay=config["ema_decay"])
    writer = SummaryWriter(log_dir=str(model_save_dir))

    eps = 1e-5
    best_val = float("inf")
    keep_last_n = 3

    for epoch in range(config["nepochs"]):
        model.train()
        avg, n_items = 0.0, 0
        for coords_b, y0_b, cond_b, _ in train_loader:
            loss = field_diffusion_loss(
                model,
                y0_b.to(device, non_blocking=True),
                coords_b.to(device, non_blocking=True),
                cond_b.to(device, non_blocking=True),
                config["n_context"], mean_fn, std_fn,
                eps=eps, loss_weighting=config["loss_weighting"],
                cond_drop_prob=config["cond_drop_prob"],
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ema.update()
            avg += loss.item()
            n_items += 1

        train_avg = avg / max(1, n_items)
        writer.add_scalar("Loss/train", train_avg, epoch)
        print(f"Epoch {epoch} train loss: {train_avg:.6e}")

        val_loss = validate(
            model, dataset, val_indices, mean_fn, std_fn, device,
            eps, config["loss_weighting"],
            config["n_points"], config["n_context"],
        )
        writer.add_scalar("Loss/val", val_loss, epoch)
        print(f"Epoch {epoch} val loss: {val_loss:.6e}")

        if epoch % config["save_every_n_epochs"] == 0:
            with ema.average_parameters():
                torch.save(model.state_dict(), ckpt_dir / f"ckpt_{epoch}.pth")
        if val_loss < best_val:
            best_val = val_loss
            with ema.average_parameters():
                torch.save(model.state_dict(), ckpt_dir / "ckpt_best.pth")
            print(f"Saved new best checkpoint at epoch {epoch}")

        ckpts = sorted(ckpt_dir.glob("ckpt_*.pth"), key=os.path.getmtime)
        for p in ckpts[:-keep_last_n]:
            if p.name != "ckpt_best.pth":
                p.unlink()

        se = config["sample_every_n_epochs"]
        if se > 0 and (epoch % se == 0 or epoch == config["nepochs"] - 1):
            model.eval()
            with ema.average_parameters():
                save_ensembles(model, dataset, val_indices, model_save_dir,
                               epoch, config, device, mean_fn, std_fn, meta)
            model.train()

    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()