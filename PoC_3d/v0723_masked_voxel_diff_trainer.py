"""
Route C: mixed-shape voxel diffusion with the PROVEN 3D U-Net.

Strategy (see field_diffusion_context_v0722.md, "Route C fallback"):
  * Batches are bucketed by part shape (one bucket per (Nx,Ny,Nz)), so every
    batch is a homogeneous stack of voxel cubes -- no ragged collate.
  * Each part is padded (at the high end of each axis) to dims divisible by 8,
    which the U-Net's three stride-2 stages require for its skip connections.
    Padded region of x0 is filled with -1 (void).
  * A DOMAIN MASK channel (1 inside the part's true extent, 0 in padding) is
    concatenated to the network input, so the model always knows the domain.
  * The loss is MASKED: mean squared error over domain voxels only (per-part
    mean, then batch mean), so val loss is comparable across shapes.
  * At sampling time the model's x0-prediction in the padded region is pinned
    to -1 every solver step, keeping the pad state on a clean deterministic
    trajectory; the final field is cropped to the true shape and binarized
    at field > 0.

Why this cannot reproduce the perceiver failure: the U-Net sees its entire
state through convolution at every step -- there is no pointwise shortcut to
collapse onto. This same architecture (ScoreNet3DVoxelCond in
vF_diffusion_trainer.py) already shows genuine topological diversity at 32^3;
this file only adds the mask channel and bucketed/padded plumbing.

Data: the same mixed-shape npy consumed by the field trainers
(entries (voxel[Nx,Ny,Nz] {0,1}, cond_vec, 0, cond_str, sample_info)).
Loading reuses NeuralFieldDataset3D from v0720_neural_diff_trainer_fixed.py.

Sampling: batched fixed-step Heun (quadratic time spacing, as validated in
v0720_fast_patch.py style) -- one solve per condition generates the whole
ensemble. Optional classifier-free guidance via --cond_drop_prob at train
time and --guidance_scale in the sampler.

Requires v0720_neural_diff_trainer_fixed.py in the same directory.
"""

import os
from pathlib import Path
import argparse
import functools
import json
import math
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch_ema import ExponentialMovingAverage

from v0720_neural_diff_trainer_fixed import (
    NeuralFieldDataset3D,
    marginal_prob_mean,
    marginal_prob_std,
    drift_coeff,
    GaussianFourierProjection,
    plot_voxel_with_overlays_implicit,
    decode_condition_overlay,
)


# --------------------------------------------------------------------------
# args
# --------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Mixed-shape masked voxel diffusion (Route C)")
    p.add_argument("--device", default="cuda", type=str)
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--batchsize", default=8, type=int)
    p.add_argument("--nepochs", default=300, type=int)
    p.add_argument("--lr", default=1e-4, type=float)
    p.add_argument("--grad_clip", default=1.0, type=float)

    # U-Net (defaults = the proven 32^3 config)
    p.add_argument("--unet_ch1_dim", default=32, type=int)
    p.add_argument("--unet_ch2_dim", default=64, type=int)
    p.add_argument("--unet_ch3_dim", default=128, type=int)
    p.add_argument("--t_embed_dim", default=128, type=int)
    p.add_argument("--cond_embed_dim", default=128, type=int)
    p.add_argument("--cond_ch", default=8, type=int)
    p.add_argument("--cond_drop_prob", default=0.1, type=float)
    p.add_argument("--loss_weighting", default="Simple", type=str,
                   choices=["Simple", "Analytical"])

    # data
    p.add_argument("--data_file", type=str, required=True)
    p.add_argument("--meta_path", type=str, default=None)
    p.add_argument("--nsamples", default=0, type=int)
    p.add_argument("--val_frac", default=0.05, type=float)
    p.add_argument("--val_max_parts", default=256, type=int,
                   help="Cap on val parts per epoch (speed).")

    # logging / checkpoints / in-training sampling
    p.add_argument("--log_root", type=str, required=True)
    p.add_argument("--save_every_n_epochs", default=25, type=int)
    p.add_argument("--sample_every_n_epochs", default=25, type=int)
    p.add_argument("--sample_num", default=4, type=int,
                   help="Conditions per sampling event.")
    p.add_argument("--ensemble_size", default=4, type=int,
                   help="Members per condition (one batched Heun solve).")
    p.add_argument("--sample_n_steps", default=150, type=int)
    p.add_argument("--sample_eps", default=1e-3, type=float)
    p.add_argument("--ema_decay", default=0.9999, type=float)
    p.add_argument("--load_ckpt", type=str, default=None)
    p.add_argument("--num_workers", default=4, type=int)
    p.add_argument("--tag", default="", type=str)
    return vars(p.parse_args())


# --------------------------------------------------------------------------
# padding / dataset / bucketed batching
# --------------------------------------------------------------------------

def pad_to_multiple(n, m=8):
    return int(math.ceil(n / float(m)) * m)


class PaddedVoxelDataset(Dataset):
    """
    Wraps NeuralFieldDataset3D. Each item:
      x0   [1, Px, Py, Pz]  in {-1,+1}, padding filled with -1 (void)
      mask [1, Px, Py, Pz]  1 inside the true (Nx,Ny,Nz) extent, 0 in padding
      cond [cond_dim]
      idx  original dataset index
    (Px,Py,Pz) = each dim padded up to a multiple of 8; padding sits at the
    high end of each axis, so voxel (i,j,k) keeps its physical location.
    """

    def __init__(self, base: NeuralFieldDataset3D, indices):
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)
        self.true_shapes = [tuple(base.voxels[int(i)].shape) for i in self.indices]
        self.padded_shapes = [tuple(pad_to_multiple(s) for s in sh)
                              for sh in self.true_shapes]
        self.cond_dim = base.cond_dim

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        vox = self.base.voxels[idx]                      # float32 {0,1}
        Nx, Ny, Nz = vox.shape
        Px, Py, Pz = self.padded_shapes[i]

        x0 = np.full((1, Px, Py, Pz), -1.0, dtype=np.float32)
        x0[0, :Nx, :Ny, :Nz] = vox * 2.0 - 1.0
        mask = np.zeros((1, Px, Py, Pz), dtype=np.float32)
        mask[0, :Nx, :Ny, :Nz] = 1.0

        cond = self.base.conds[idx]
        return (torch.from_numpy(x0), torch.from_numpy(mask),
                torch.from_numpy(cond), idx)


class ShapeBucketBatchSampler(Sampler):
    """
    Yields batches of dataset POSITIONS whose parts all share one padded
    shape, so default collate can stack them. Within-bucket order and the
    order of batches are reshuffled every epoch (when shuffle=True).
    Remainder batches are kept (may be smaller than batchsize).
    """

    def __init__(self, padded_shapes, batchsize, seed=0, shuffle=True):
        self.batchsize = int(batchsize)
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.buckets = defaultdict(list)
        for pos, sh in enumerate(padded_shapes):
            self.buckets[sh].append(pos)
        self.n_batches = sum(
            math.ceil(len(v) / self.batchsize) for v in self.buckets.values())
        sizes = {sh: len(v) for sh, v in self.buckets.items()}
        print(f"Shape buckets (padded): {sizes}")

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        batches = []
        for positions in self.buckets.values():
            pos = np.asarray(positions)
            if self.shuffle:
                pos = self.rng.permutation(pos)
            for s in range(0, len(pos), self.batchsize):
                batches.append(pos[s:s + self.batchsize].tolist())
        if self.shuffle:
            order = self.rng.permutation(len(batches))
            batches = [batches[k] for k in order]
        return iter(batches)


def make_loaders_masked(npy_path, batchsize, nsamples, val_frac=0.05,
                        seed=42, num_workers=0):
    base = NeuralFieldDataset3D(npy_path)

    N_total = len(base)
    if nsamples is None or nsamples <= 0:
        nsamples = N_total
    nsamples = min(nsamples, N_total)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(N_total)[:nsamples]
    n_val = max(1, int(val_frac * nsamples))
    val_indices = perm[:n_val]
    train_indices = perm[n_val:]

    train_set = PaddedVoxelDataset(base, train_indices)
    val_set = PaddedVoxelDataset(base, val_indices)

    train_loader = DataLoader(
        train_set,
        batch_sampler=ShapeBucketBatchSampler(
            train_set.padded_shapes, batchsize, seed=seed, shuffle=True),
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_sampler=ShapeBucketBatchSampler(
            val_set.padded_shapes, batchsize, seed=seed, shuffle=False),
        num_workers=num_workers, pin_memory=True,
    )
    print(f"Train parts: {len(train_indices)}  Val parts: {len(val_indices)}")
    return base, train_set, val_set, train_loader, val_loader, train_indices, val_indices


# --------------------------------------------------------------------------
# model: the proven U-Net + a mask input channel
# --------------------------------------------------------------------------

class Dense3D(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[..., None, None, None]


class CondProjectTo3D(nn.Module):
    def __init__(self, cond_dim, cond_embed_dim, cond_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, cond_embed_dim),
            nn.SiLU(),
            nn.Linear(cond_embed_dim, cond_ch),
        )

    def forward(self, cond, spatial_shape):
        B = cond.shape[0]
        D, H, W = spatial_shape
        x = self.net(cond).view(B, -1, 1, 1, 1)
        return x.expand(B, x.shape[1], D, H, W)


class MaskedScoreNet3D(nn.Module):
    """
    ScoreNet3DVoxelCond (vF_diffusion_trainer.py) with one extra input
    channel for the domain mask. Fully convolutional: works on any spatial
    dims divisible by 8. X0-prediction mode.
      forward(x [B,1,D,H,W], mask [B,1,D,H,W], t [B], cond [B,cond_dim])
      -> x0hat [B,1,D,H,W]
    """

    def __init__(self, cond_dim, cond_embed_dim, cond_ch, c1, c2, c3, ed):
        super().__init__()
        self.cond_proj = (CondProjectTo3D(cond_dim, cond_embed_dim, cond_ch)
                          if cond_dim > 0 and cond_ch > 0 else None)
        self.cond_ch = cond_ch if self.cond_proj is not None else 0

        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=ed),
            nn.Linear(ed, ed),
        )
        self.act = lambda x: x + torch.sin(x) ** 2   # X0-mode activation

        in_ch = 1 + 1 + self.cond_ch                 # x + mask + cond channels
        self.conv1 = nn.Conv3d(in_ch, c1, 3, stride=1, padding=1, bias=False)
        self.dense1 = Dense3D(ed, c1)
        self.gnorm1 = nn.GroupNorm(max(1, c1 // 8), c1)

        self.conv2 = nn.Conv3d(c1, c2, 3, stride=2, padding=1, bias=False)
        self.dense2 = Dense3D(ed, c2)
        self.gnorm2 = nn.GroupNorm(max(1, c2 // 8), c2)

        self.conv3 = nn.Conv3d(c2, c3, 3, stride=2, padding=1, bias=False)
        self.dense3 = Dense3D(ed, c3)
        self.gnorm3 = nn.GroupNorm(max(1, c3 // 8), c3)

        self.conv4 = nn.Conv3d(c3, c3, 3, stride=2, padding=1, bias=False)
        self.dense4 = Dense3D(ed, c3)
        self.gnorm4 = nn.GroupNorm(max(1, c3 // 8), c3)

        self.tconv4 = nn.ConvTranspose3d(c3, c3, 3, stride=2, padding=1,
                                         output_padding=1, bias=False)
        self.dense5 = Dense3D(ed, c3)
        self.tgnorm4 = nn.GroupNorm(max(1, c3 // 8), c3)

        self.tconv3 = nn.ConvTranspose3d(c3 + c3, c2, 3, stride=2, padding=1,
                                         output_padding=1, bias=False)
        self.dense6 = Dense3D(ed, c2)
        self.tgnorm3 = nn.GroupNorm(max(1, c2 // 8), c2)

        self.tconv2 = nn.ConvTranspose3d(c2 + c2, c1, 3, stride=2, padding=1,
                                         output_padding=1, bias=False)
        self.dense7 = Dense3D(ed, c1)
        self.tgnorm2 = nn.GroupNorm(max(1, c1 // 8), c1)

        self.tconv1 = nn.ConvTranspose3d(c1 + c1, 1, 3, stride=1, padding=1)

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"MaskedScoreNet3D has {n_params/1e6:.2f}M trainable parameters")

    def forward(self, x, mask, t, cond=None):
        embed = self.act(self.embed(t))

        h = torch.cat([x, mask], dim=1)
        if self.cond_proj is not None and cond is not None:
            h = torch.cat([h, self.cond_proj(cond, x.shape[-3:])], dim=1)

        h1 = self.act(self.gnorm1(self.conv1(h) + self.dense1(embed)))
        h2 = self.act(self.gnorm2(self.conv2(h1) + self.dense2(embed)))
        h3 = self.act(self.gnorm3(self.conv3(h2) + self.dense3(embed)))
        h4 = self.act(self.gnorm4(self.conv4(h3) + self.dense4(embed)))

        h = self.act(self.tgnorm4(self.tconv4(h4) + self.dense5(embed)))
        h = self.act(self.tgnorm3(self.tconv3(torch.cat([h, h3], dim=1))
                                  + self.dense6(embed)))
        h = self.act(self.tgnorm2(self.tconv2(torch.cat([h, h2], dim=1))
                                  + self.dense7(embed)))
        return self.tconv1(torch.cat([h, h1], dim=1))


# --------------------------------------------------------------------------
# loss (masked to domain voxels)
# --------------------------------------------------------------------------

def masked_vpsde_loss_x0(model, x0, mask, cond, mean_fn, std_fn,
                         eps=1e-5, loss_weighting="Simple",
                         cond_drop_prob=0.0):
    B = x0.shape[0]
    device = x0.device
    t = torch.rand(B, device=device) * (1. - eps) + eps
    z = torch.randn_like(x0)
    ms = mean_fn(t)[:, None, None, None, None]
    st = std_fn(t)[:, None, None, None, None]
    x_t = x0 * ms + z * st

    if cond is not None and cond_drop_prob > 0.0:
        keep = (torch.rand(B, device=device) > cond_drop_prob).float()[:, None]
        cond = cond * keep

    xhat = model(x_t, mask, t, cond)

    se = (xhat - x0) ** 2
    if loss_weighting == "Analytical":
        se = se * (ms / st) ** 2
    elif loss_weighting != "Simple":
        raise ValueError("Invalid loss_weighting")

    per_part = (se * mask).sum(dim=(1, 2, 3, 4)) / mask.sum(dim=(1, 2, 3, 4))
    return per_part.mean()


# --------------------------------------------------------------------------
# batched Heun sampler (quadratic time spacing, x0hat pinned in padding)
# --------------------------------------------------------------------------

def heun_sampler_masked(model, mask, cond, mean_fn, std_fn, drift_fn,
                        n_steps=150, eps=1e-3, device="cuda",
                        guidance_scale=0.0, generator=None):
    """
    mask: [B,1,D,H,W]; cond: [B,cond_dim].
    Returns the final continuous field [B,1,D,H,W] (cpu).
    guidance_scale w > 0 applies CFG: (1+w)*cond - w*uncond
    (valid only if the model was trained with cond_drop_prob > 0).
    """
    model.eval()
    B = mask.shape[0]
    mask = mask.to(device)
    cond = cond.to(device)
    x = torch.randn(mask.shape, device=device, generator=generator)

    use_cfg = guidance_scale > 0.0
    if use_cfg:
        cond_full = torch.cat([cond, torch.zeros_like(cond)], dim=0)
        mask_full = torch.cat([mask, mask], dim=0)
    else:
        cond_full, mask_full = cond, mask

    @torch.no_grad()
    def x0hat_eval(x_state, t_scalar):
        xb = torch.cat([x_state, x_state], dim=0) if use_cfg else x_state
        t = torch.full((xb.shape[0],), float(t_scalar), device=device)
        xhat = model(xb, mask_full, t, cond_full)
        if use_cfg:
            c, u = xhat[:B], xhat[B:]
            xhat = (1.0 + guidance_scale) * c - guidance_scale * u
        # pin padded region to void -> clean deterministic pad trajectory
        return xhat * mask + (-1.0) * (1.0 - mask)

    def dxdt(x_state, t_scalar):
        t1 = torch.tensor([float(t_scalar)], device=device)
        mean = mean_fn(t1).view(1, 1, 1, 1, 1)
        std = std_fn(t1).view(1, 1, 1, 1, 1)
        x0hat = x0hat_eval(x_state, t_scalar)
        score = -(x_state - mean * x0hat) / (std ** 2)
        drift = drift_fn(t1).view(1, 1, 1, 1, 1)
        return drift * (x_state + score)

    s = torch.linspace(0.0, 1.0, n_steps + 1)
    ts = (eps + (1.0 - eps) * (1.0 - s) ** 2).tolist()   # 1.0 -> eps
    for i in range(n_steps):
        t0, t1 = ts[i], ts[i + 1]
        dt = t1 - t0                                     # negative
        d0 = dxdt(x, t0)
        if i == n_steps - 1:
            x = x + dt * d0
        else:
            x_pred = x + dt * d0
            d1 = dxdt(x_pred, t1)
            x = x + 0.5 * dt * (d0 + d1)
    return x.cpu()


# --------------------------------------------------------------------------
# sample statistics (same metrics as v0722 diagnostics `sample-stats`)
# --------------------------------------------------------------------------

def speckle_fraction(binary):
    """Fraction of voxels that disagree with the MAJORITY of their in-domain
    6-neighbors (strict: #agreeing < #neighbors/2). Clean structures score
    ~0.00; iid coin-flip noise scores ~0.30+, the same order as the failed
    perceiver samples in the v0722 diagnostics. If exact comparability with
    those numbers matters, swap in the speckle function from
    v0722_diagnose_field_diff.py."""
    v = np.asarray(binary).astype(np.int8)
    agree = np.zeros(v.shape, dtype=np.int16)
    total = np.zeros(v.shape, dtype=np.int16)
    for ax in range(3):
        sl_a = [slice(None)] * 3
        sl_b = [slice(None)] * 3
        sl_a[ax] = slice(0, -1)
        sl_b[ax] = slice(1, None)
        a, b = tuple(sl_a), tuple(sl_b)
        eq = (v[a] == v[b]).astype(np.int16)
        agree[a] += eq
        agree[b] += eq
        total[a] += 1
        total[b] += 1
    return float((agree * 2 < total).mean())


def pairwise_iou(binaries):
    """Mean IoU over all member pairs. binaries: list of {0,1} arrays."""
    E = len(binaries)
    if E < 2:
        return float("nan")
    vals = []
    for i in range(E):
        for j in range(i + 1, E):
            a, b = binaries[i] > 0, binaries[j] > 0
            union = np.logical_or(a, b).sum()
            inter = np.logical_and(a, b).sum()
            vals.append(inter / union if union > 0 else 1.0)
    return float(np.mean(vals))


# --------------------------------------------------------------------------
# in-training ensemble sampling
# --------------------------------------------------------------------------

def save_sample_batch_masked(model, base, val_indices, model_dir, epoch,
                             config, device, mean_fn, std_fn, drift_fn, meta):
    samples_dir = model_dir / f"samples_epoch_{epoch:04d}"
    samples_dir.mkdir(parents=True, exist_ok=True)

    # SEEDED so the same conditions are sampled every epoch -> comparable
    n = min(config["sample_num"], len(val_indices))
    chosen = np.random.default_rng(12345).choice(
        np.asarray(val_indices), size=n, replace=False)
    E = config["ensemble_size"]

    manifest = samples_dir / "sample_manifest.txt"
    with open(manifest, "w") as f:
        f.write("# idx shape gt_mass member_masses speckles pairwise_iou cond_str\n")

        for ci, idx in enumerate(chosen):
            idx = int(idx)
            vox = base.voxels[idx]
            Nx, Ny, Nz = vox.shape
            Px, Py, Pz = (pad_to_multiple(Nx), pad_to_multiple(Ny),
                          pad_to_multiple(Nz))
            mask = torch.zeros(E, 1, Px, Py, Pz)
            mask[:, :, :Nx, :Ny, :Nz] = 1.0
            cond_vec = base.conds[idx]
            cond = torch.from_numpy(cond_vec).float()[None].expand(E, -1)

            gen = torch.Generator(device=device)
            gen.manual_seed(1000 * epoch + ci)   # varies across epochs on purpose
            x_final = heun_sampler_masked(
                model, mask, cond, mean_fn, std_fn, drift_fn,
                n_steps=config["sample_n_steps"], eps=config["sample_eps"],
                device=device, generator=gen,
            ).numpy()[:, 0, :Nx, :Ny, :Nz]                     # crop to true shape

            bins = [(x_final[e] > 0.0).astype(np.float32) for e in range(E)]
            masses = [float(b.mean()) for b in bins]
            speckles = [speckle_fraction(b) for b in bins]
            iou = pairwise_iou(bins)
            gt_mass = float(vox.mean())

            overlay = decode_condition_overlay(cond_vec, meta)
            for e in range(E):
                plot_voxel_with_overlays_implicit(
                    bins[e], overlay["bc_points"], overlay["load_point"],
                    overlay["load_dir"],
                    title=(f"Ep {epoch} cond {ci} member {e} | "
                           f"({Nx},{Ny},{Nz}) | mass {masses[e]:.3f} "
                           f"| speckle {speckles[e]:.3f}"),
                    save_path=samples_dir / f"cond{ci:02d}_member{e}.png",
                )
            plot_voxel_with_overlays_implicit(
                vox, overlay["bc_points"], overlay["load_point"],
                overlay["load_dir"],
                title=f"GT cond {ci} | ({Nx},{Ny},{Nz}) | mass {gt_mass:.3f}",
                save_path=samples_dir / f"cond{ci:02d}_gt.png",
            )
            np.savez_compressed(
                samples_dir / f"cond{ci:02d}_ensemble.npz",
                fields=x_final, binaries=np.stack(bins),
                cond=cond_vec, gt=vox, dataset_index=idx,
            )

            line = (f"{idx} ({Nx},{Ny},{Nz}) {gt_mass:.3f} "
                    f"{[round(m,3) for m in masses]} "
                    f"{[round(s,3) for s in speckles]} "
                    f"{iou:.3f} {base.cond_strs[idx]}")
            f.write(line + "\n")
            print("  [sample] " + line)

    print(f"Saved ensembles to {samples_dir}")


# --------------------------------------------------------------------------
# deterministic validation
# --------------------------------------------------------------------------

def compute_validation_loss(model, val_loader, mean_fn, std_fn, device,
                            eps, loss_weighting, max_parts=256):
    """Fixed RNG for t / noise draws -> smooth val curve, stable 'best'."""
    model.eval()
    total, n_items = 0.0, 0
    devices = [device] if device.type == "cuda" else []
    with torch.no_grad(), torch.random.fork_rng(devices=devices):
        torch.manual_seed(1234)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(1234)
        for x0, mask, cond, _ in val_loader:
            if n_items >= max_parts:
                break
            loss = masked_vpsde_loss_x0(
                model,
                x0.to(device, non_blocking=True),
                mask.to(device, non_blocking=True),
                cond.to(device, non_blocking=True),
                mean_fn, std_fn, eps=eps, loss_weighting=loss_weighting,
                cond_drop_prob=0.0,
            )
            total += loss.item() * x0.shape[0]
            n_items += x0.shape[0]
    model.train()
    return total / max(1, n_items)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    config = parse_args()
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    log_root = Path(config["log_root"])
    log_root.mkdir(parents=True, exist_ok=True)

    def resolve_meta_path(data_path, meta_path=None):
        if meta_path is not None:
            return meta_path
        if data_path.endswith(".npy"):
            return data_path[:-4] + "_meta.npz"
        raise ValueError("Provide --meta_path")

    meta_path = resolve_meta_path(config["data_file"], config["meta_path"])
    meta = np.load(meta_path, allow_pickle=True)
    if "conditioning_spec_json" in meta:
        conditioning_spec = json.loads(str(meta["conditioning_spec_json"]))
    elif "conditioning_spec" in meta:
        conditioning_spec = meta["conditioning_spec"].item()
    else:
        conditioning_spec = {}

    extra_tag = f"_{config['tag']}" if config["tag"] else ""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = (
        f"maskedVoxelDiff_bs-{config['batchsize']}_"
        f"c1-{config['unet_ch1_dim']}_c2-{config['unet_ch2_dim']}_"
        f"c3-{config['unet_ch3_dim']}_drop-{config['cond_drop_prob']}"
        f"{extra_tag}_{timestamp}"
    )
    model_save_dir = log_root / run_name
    ckpt_dir = model_save_dir / "checkpoints"
    model_save_dir.mkdir(parents=True, exist_ok=False)
    ckpt_dir.mkdir()

    with open(model_save_dir / "hparams.json", "w") as f:
        json.dump({**config, "conditioning_spec": conditioning_spec,
                   "meta_path": str(meta_path)}, f, indent=2)

    (base, train_set, val_set, train_loader, val_loader,
     train_indices, val_indices) = make_loaders_masked(
        npy_path=config["data_file"],
        batchsize=config["batchsize"],
        nsamples=config["nsamples"],
        val_frac=config["val_frac"],
        seed=config["seed"],
        num_workers=config["num_workers"],
    )

    mean_fn = functools.partial(marginal_prob_mean, bmin=0.1, bmax=20.0)
    std_fn = functools.partial(marginal_prob_std, bmin=0.1, bmax=20.0)
    drift_fn = functools.partial(drift_coeff, bmin=0.1, bmax=20.0)

    model = MaskedScoreNet3D(
        cond_dim=base.cond_dim,
        cond_embed_dim=config["cond_embed_dim"],
        cond_ch=config["cond_ch"],
        c1=config["unet_ch1_dim"],
        c2=config["unet_ch2_dim"],
        c3=config["unet_ch3_dim"],
        ed=config["t_embed_dim"],
    ).to(device)

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
        for x0, mask, cond, _ in train_loader:
            loss = masked_vpsde_loss_x0(
                model,
                x0.to(device, non_blocking=True),
                mask.to(device, non_blocking=True),
                cond.to(device, non_blocking=True),
                mean_fn, std_fn, eps=eps,
                loss_weighting=config["loss_weighting"],
                cond_drop_prob=config["cond_drop_prob"],
            )
            optimizer.zero_grad()
            loss.backward()
            if config["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               config["grad_clip"])
            optimizer.step()
            ema.update()
            avg += loss.item() * x0.shape[0]
            n_items += x0.shape[0]

        train_avg = avg / max(1, n_items)
        writer.add_scalar("Loss/train", train_avg, epoch)
        print(f"Epoch {epoch} train loss: {train_avg:.6e}")

        val_loss = compute_validation_loss(
            model, val_loader, mean_fn, std_fn, device, eps,
            config["loss_weighting"], max_parts=config["val_max_parts"])
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
            with ema.average_parameters():
                save_sample_batch_masked(
                    model, base, val_indices, model_save_dir, epoch,
                    config, device, mean_fn, std_fn, drift_fn, meta)
            model.train()

    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
