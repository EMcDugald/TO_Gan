"""
Implicit 3D conditional VPSDE diffusion for BC/load-conditioned NITO fields.

Fixed version of v0718_neural_diff_trainer.py. Changes vs. the original:

  (1) TARGET SCALING: occupancies are scaled from {0,1} to {-1,+1}
      (matching vF_diffusion_trainer and the N(0,1) diffusion prior).
      Sampling threshold `y > 0` is now correct.

  (2) COORDINATE AXIS ORDER: voxel arrays are (Nx, Ny, Nz); the coord
      grid now maps axis 0 -> x, axis 1 -> y, axis 2 -> z, so query
      coordinates live in the same (x, y, z) frame as the BC/load
      coordinates in the condition vector.

  (3) ISOTROPIC COORDINATES: all axes are normalized by the part's max
      dimension (same convention as the BC/load normalization), so a
      voxel step means the same thing in every part regardless of
      aspect ratio. Coordinates are then mapped to [-1, 1] for SIREN.
      The same affine map is exposed as `norm01_to_siren` so you can
      (optionally) remap cond BC/load coords into the identical frame.

  (4) POINT-SAMPLED BATCHING: each training step draws n_points random
      points from each of B parts and stacks them into [B, M, .].
      This (a) decorrelates gradients vs. contiguous chunking,
      (b) makes batchsize > 1 work with mixed shapes (default collate
      would previously have crashed), (c) gives per-part random t.

  (5) LOSS = MEAN over points (chunk-size independent, train/val
      comparable). "Analytical" weighting kept as an option.

  Validation and sampling still evaluate the FULL grid (in forward-only
  chunks to bound memory).
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
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch_ema import ExponentialMovingAverage
from scipy import integrate
import matplotlib.pyplot as plt


# --------- arg parsing ---------

def parse_args():
    p = argparse.ArgumentParser(
        description="Implicit 3D conditional VPSDE diffusion (fixed)"
    )
    p.add_argument("--device", default="cuda", type=str)
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--batchsize", default=8, type=int)        # parts per step
    p.add_argument("--n_points", default=4096, type=int)      # random pts per part per step
    p.add_argument("--steps_per_epoch", default=0, type=int)  # 0 = one pass over parts
    p.add_argument("--nepochs", default=500, type=int)
    p.add_argument("--lr", default=1e-4, type=float)

    # SIREN-based implicit score model
    p.add_argument("--width", default=256, type=int)
    p.add_argument("--depth", default=5, type=int)
    p.add_argument("--omega", default=30.0, type=float)
    p.add_argument("--t_embed_dim", default=128, type=int)

    p.add_argument("--mode", default="X0", type=str)
    p.add_argument("--loss_weighting", default="Simple", type=str)

    # data
    p.add_argument("--data_file", type=str, required=True)
    p.add_argument("--meta_path", type=str, default=None)
    p.add_argument("--nsamples", default=0, type=int)
    p.add_argument("--val_frac", default=0.1, type=float)

    # logging / checkpoints
    p.add_argument("--log_root", type=str, required=True)
    p.add_argument("--save_every_n_epochs", default=50, type=int)
    p.add_argument("--sample_every_n_epochs", default=50, type=int)
    p.add_argument("--sample_num", default=4, type=int)
    p.add_argument("--ema_decay", default=0.9999, type=float)
    p.add_argument("--load_ckpt", type=str, default=None)
    p.add_argument("--num_workers", default=0, type=int)
    p.add_argument("--tag", default="", type=str)

    # diffusion sampling hyperparams
    p.add_argument("--sample_atol", default=1e-4, type=float)
    p.add_argument("--sample_rtol", default=1e-4, type=float)
    p.add_argument("--sample_eps", default=1e-3, type=float)

    # forward-only chunk size for full-grid eval (val / sampling)
    p.add_argument("--eval_chunk", default=65536, type=int)
    return vars(p.parse_args())


# --------- dataset ---------

class NeuralFieldDataset3D(Dataset):
    """
    Each npy entry: (voxel_arr, cond_vec, label_dummy, cond_str, sample_info)
    voxel_arr is (Nx, Ny, Nz), varying across samples.

    Targets are stored scaled to {-1, +1}.                           # FIX (1)
    Coordinates: axis order (x, y, z) matching the voxel array,      # FIX (2)
    isotropically normalized by max(Nx,Ny,Nz)-1 into [0,1], then     # FIX (3)
    affinely mapped to [-1,1] for the SIREN.
    """

    def __init__(self, npy_path):
        npy_path = Path(npy_path)
        if not npy_path.exists():
            raise FileNotFoundError(f"Data file not found: {npy_path}")
        data = np.load(npy_path, allow_pickle=True)

        self.voxels = [x[0].astype(np.float32) for x in data]   # each (Nx,Ny,Nz), {0,1}
        self.conds = [x[1].astype(np.float32) for x in data]
        self.cond_strs = [x[3] for x in data]
        self.sample_infos = [x[4] if len(x) > 4 and x[4] is not None else {} for x in data]

        self.N = len(self.voxels)
        self.cond_dim = self.conds[0].shape[0]

        self._coord_cache = {}  # shape tuple -> coords [N_pts,3] (cpu)

        print(f"Loaded {self.N} neural-field samples (mixed shapes)")
        print(f"Condition dim: {self.cond_dim}")
        all_vals = np.concatenate([v.reshape(-1) for v in self.voxels[:min(self.N, 200)]])
        print(f"Raw occupancy range (subset): [{all_vals.min():.3f}, {all_vals.max():.3f}]")

    def __len__(self):
        return self.N

    @staticmethod
    def norm01_to_siren(u):
        """Map max-dim-normalized [0,1] coords to the SIREN's [-1,1] frame."""
        return 2.0 * u - 1.0

    def _coords_for_shape(self, shape):
        shape = tuple(int(s) for s in shape)
        if shape in self._coord_cache:
            return self._coord_cache[shape]
        Nx, Ny, Nz = shape
        max_dim = float(max(Nx, Ny, Nz))
        denom = max(max_dim - 1.0, 1.0)
        ax = torch.arange(Nx, dtype=torch.float32) / denom   # axis 0 = physical x  # FIX (2)
        ay = torch.arange(Ny, dtype=torch.float32) / denom
        az = torch.arange(Nz, dtype=torch.float32) / denom
        xx, yy, zz = torch.meshgrid(ax, ay, az, indexing="ij")
        coords01 = torch.stack([xx, yy, zz], dim=-1).view(-1, 3)   # [N_pts,3] in [0,1]
        coords = self.norm01_to_siren(coords01)                    # FIX (3)
        self._coord_cache[shape] = coords
        return coords

    def get_full_grid(self, idx, device="cpu"):
        x = self.voxels[idx]                               # (Nx,Ny,Nz) in {0,1}
        coords = self._coords_for_shape(x.shape).to(device)
        targets = torch.from_numpy(x).view(-1, 1).to(device) * 2.0 - 1.0   # FIX (1)
        cond = torch.from_numpy(self.conds[idx]).to(device)
        return coords, targets, cond

    def sample_points(self, idx, n_points, generator=None):
        """Random point subset for training. Returns cpu tensors [M,3], [M,1], [cond_dim]."""
        x = self.voxels[idx]
        coords = self._coords_for_shape(x.shape)           # [N_pts,3]
        targets = torch.from_numpy(x).view(-1, 1) * 2.0 - 1.0
        N_pts = coords.shape[0]
        if n_points >= N_pts:
            sel = torch.arange(N_pts)
        else:
            sel = torch.randint(0, N_pts, (n_points,), generator=generator)
        cond = torch.from_numpy(self.conds[idx])
        return coords[sel], targets[sel], cond


class PointSampleDataset(Dataset):
    """
    Wraps NeuralFieldDataset3D for DataLoader batching across mixed shapes:
    every __getitem__ returns a fixed-size random point set, so default
    collate can stack [B, M, .].                                     # FIX (4)
    """

    def __init__(self, base, indices, n_points):
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)
        self.n_points = n_points

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        coords, y0, cond = self.base.sample_points(idx, self.n_points)
        return coords, y0, cond, idx


def make_loaders_implicit(npy_path, batchsize, n_points, nsamples,
                          val_frac=0.1, seed=42, num_workers=0):
    dataset = NeuralFieldDataset3D(npy_path)

    N_total = len(dataset)
    if nsamples is None or nsamples <= 0:
        nsamples = N_total
    nsamples = min(nsamples, N_total)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(N_total)[:nsamples]
    n_val = max(1, int(val_frac * nsamples))
    val_indices = perm[:n_val]
    train_indices = perm[n_val:]

    train_set = PointSampleDataset(dataset, train_indices, n_points)

    train_loader = DataLoader(
        train_set, batch_size=batchsize, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )
    print(f"Train parts: {len(train_indices)}  Val parts: {len(val_indices)}")
    return dataset, train_loader, train_indices, val_indices


# --------- VPSDE marginal stats ---------

def marginal_prob_mean(t, bmin, bmax):
    log_coeff = 0.5 * (bmax - bmin) * t**2 + bmin * t
    return torch.exp(-0.5 * log_coeff)


def marginal_prob_std(t, bmin, bmax):
    log_coeff = 0.5 * (bmax - bmin) * t**2 + bmin * t
    return torch.sqrt(1. - torch.exp(-log_coeff))


def drift_coeff(t, bmin, bmax):
    betas = bmin + (bmax - bmin) * t
    return -0.5 * betas


# --------- time embedding ---------

class GaussianFourierProjection(nn.Module):
    def __init__(self, embed_dim, scale=30.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale,
                              requires_grad=False)

    def forward(self, t):
        x_proj = t[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


# --------- SIREN blocks ---------

class SineLayer(nn.Module):
    def __init__(self, cin, cout, omega, first=False):
        super().__init__()
        self.omega = omega
        self.lin = nn.Linear(cin, cout)
        with torch.no_grad():
            if first:
                self.lin.weight.uniform_(-1.0 / cin, 1.0 / cin)
            else:
                b = math.sqrt(6.0 / cin) / omega
                self.lin.weight.uniform_(-b, b)

    def forward(self, x):
        return torch.sin(self.omega * self.lin(x))


class ImplicitScoreNet(nn.Module):
    """
    Pointwise X0-prediction model.
      coords: [B, M, 3]  in [-1,1] (isotropic frame)
      y_t:    [B, M, 1]
      t:      [B]
      cond:   [B, cond_dim]
    -> y0hat: [B, M, 1]
    """
    def __init__(self, cond_dim, t_embed_dim=128,
                 width=256, depth=5, omega=30.0, mode="X0"):
        super().__init__()
        self.mode = mode

        self.t_embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=t_embed_dim),
            nn.Linear(t_embed_dim, t_embed_dim),
            nn.SiLU(),
        )
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, width),
            nn.SiLU(),
        )

        in_dim = 3 + 1 + t_embed_dim + width
        layers = [SineLayer(in_dim, width, omega, first=True)]
        for _ in range(depth - 1):
            layers.append(SineLayer(width, width, omega))
        self.body = nn.Sequential(*layers)

        self.head = nn.Linear(width, 1)
        with torch.no_grad():
            b = math.sqrt(6.0 / width) / omega
            self.head.weight.uniform_(-b, b)
            self.head.bias.zero_()

    def forward(self, coords, y_t, t, cond):
        B, M, _ = coords.shape
        t_emb = self.t_embed(t).unsqueeze(1).expand(B, M, -1)
        cond_emb = self.cond_proj(cond).unsqueeze(1).expand(B, M, -1)
        x = torch.cat([coords, y_t, t_emb, cond_emb], dim=-1)
        return self.head(self.body(x))


# --------- loss ---------

def vpsde_loss_implicit_x(model, y0, coords, cond,
                          marginal_prob_mean, marginal_prob_std,
                          eps=1e-5, loss_weighting="Simple"):
    B, M, _ = y0.shape
    device = y0.device
    random_t = torch.rand(B, device=device) * (1. - eps) + eps
    z = torch.randn_like(y0)

    ms = marginal_prob_mean(random_t)[:, None, None]
    st = marginal_prob_std(random_t)[:, None, None]

    y_t = y0 * ms + z * st
    y0hat = model(coords, y_t, random_t, cond)

    if loss_weighting == "Simple":
        loss = torch.mean((y0hat - y0) ** 2)                       # FIX (5): mean
    elif loss_weighting == "Analytical":
        loss = torch.mean(((ms / st) * (y0hat - y0)) ** 2)
    else:
        raise ValueError("Invalid loss_weighting for X0")
    return loss


# --------- plotting (unchanged behavior, plus GT side-by-side) ---------

def plot_voxel_with_overlays_implicit(voxel_arr, bc_pts, load_pt, load_vec,
                                      title="", save_path="plot.png"):
    voxel_arr = np.asarray(voxel_arr)
    nx, ny, nz = voxel_arr.shape
    c = float(max(nx, ny, nz))

    bc_plot = None
    if bc_pts is not None and len(bc_pts) > 0:
        bc_pts = np.asarray(bc_pts, dtype=np.float64)
        bc_plot = bc_pts * (c - 1.0)

    load_plot = None
    if load_pt is not None:
        load_plot = np.asarray(load_pt, dtype=np.float64) * (c - 1.0)

    fig = plt.figure(figsize=(18, 6))
    views = [(25, 35, "View 1"), (25, 125, "View 2"), (65, 35, "View 3")]

    for k, (elev, azim, subtitle) in enumerate(views, start=1):
        ax = fig.add_subplot(1, 3, k, projection="3d")
        ax.voxels(voxel_arr > 0, edgecolor="k", linewidth=0.15, alpha=0.72)
        ax.set_xlim(0, nx); ax.set_ylim(0, ny); ax.set_zlim(0, nz)

        if bc_plot is not None:
            ax.scatter(bc_plot[:, 0], bc_plot[:, 1], bc_plot[:, 2],
                       c="red", s=36, marker="o", edgecolors="white",
                       linewidths=0.6, depthshade=False, label="BC")
        if load_plot is not None:
            ax.scatter([load_plot[0]], [load_plot[1]], [load_plot[2]],
                       c="dodgerblue", s=64, marker="^", edgecolors="white",
                       linewidths=0.7, depthshade=False, label="Load")
            if load_vec is not None:
                lv = np.asarray(load_vec, dtype=np.float64)
                ax.quiver(load_plot[0], load_plot[1], load_plot[2],
                          lv[0], lv[1], lv[2], color="dodgerblue",
                          linewidth=2.0, length=c * 0.18, normalize=True)

        ax.view_init(elev=elev, azim=azim)
        ax.set_title(subtitle)
        ax.set_box_aspect((nx, ny, nz))
        ax.set_axis_off()

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=220)
    plt.close(fig)


# --------- BC/load decoding (same as original) ---------

def get_bin_centers(edges):
    edges = np.asarray(edges, dtype=np.float64)
    return 0.5 * (edges[:-1] + edges[1:])


def decode_condition_overlay(cond_vec, meta):
    cond_vec = np.asarray(cond_vec, dtype=np.float32)

    if "cond_slices" in meta:
        cond_slices = meta["cond_slices"].item()
    elif "cond_slices_json" in meta:
        cond_slices = json.loads(str(meta["cond_slices_json"]))
    else:
        cond_slices = None

    if "conditioning_spec" in meta:
        conditioning_spec = meta["conditioning_spec"].item()
    elif "conditioning_spec_json" in meta:
        conditioning_spec = json.loads(str(meta["conditioning_spec_json"]))
    else:
        conditioning_spec = {}

    bc_points = np.empty((0, 3), dtype=np.float32)
    load_point = None
    load_dir = None

    if cond_slices is not None and "load_dir" in cond_slices:
        s0, s1 = cond_slices["load_dir"]
        load_dir = np.asarray(cond_vec[s0:s1], dtype=np.float32)

    if cond_slices is not None and conditioning_spec.get("bc_locations") == "fine" \
            and "bc_points" in cond_slices:
        s0, s1 = cond_slices["bc_points"]
        bc_flat = cond_vec[s0:s1]
        bc_arr = bc_flat.reshape(-1, 3)
        if "bc_mask" in cond_slices:
            m0, m1 = cond_slices["bc_mask"]
            bc_mask = cond_vec[m0:m1] > 0.5
            bc_arr = bc_arr[bc_mask]
        if "bc_count" in cond_slices:
            c0, c1 = cond_slices["bc_count"]
            bc_count = int(round(float(cond_vec[c0:c1][0])))
            bc_arr = bc_arr[:bc_count]
        bc_points = np.asarray(bc_arr, dtype=np.float32)

    if cond_slices is not None and conditioning_spec.get("load_location") == "fine" \
            and "load_point" in cond_slices:
        s0, s1 = cond_slices["load_point"]
        load_point = np.asarray(cond_vec[s0:s1], dtype=np.float32)

    return {
        "bc_points": bc_points,
        "load_point": load_point,
        "load_dir": load_dir,
        "conditioning_spec": conditioning_spec,
    }


# --------- ODE sampler (chunked model eval) ---------

def ode_sampler_implicit(score_model, coords, cond,
                         marginal_prob_mean, marginal_prob_std, drift_coeff,
                         mode, atol=1e-4, rtol=1e-4, device="cuda",
                         eps=1e-3, eval_chunk=65536):
    N_pts = coords.shape[0]
    t0 = 1.0
    init_y = torch.randn(N_pts, 1, device=device)

    coords_dev = coords.to(device)
    cond_b = cond.unsqueeze(0).to(device)

    def x0hat_full(y, t):
        """Chunked forward pass over all points. y: [N_pts,1], t: [1]."""
        outs = []
        for s in range(0, N_pts, eval_chunk):
            e = min(s + eval_chunk, N_pts)
            outs.append(score_model(
                coords_dev[s:e].unsqueeze(0), y[s:e].unsqueeze(0), t, cond_b
            ).squeeze(0))
        return torch.cat(outs, dim=0)

    def score_eval_wrapper(y_flat, time_scalar):
        y = torch.tensor(y_flat, device=device, dtype=torch.float32).view(N_pts, 1)
        t = torch.tensor([time_scalar], device=device, dtype=torch.float32)
        with torch.no_grad():
            if mode == "X0":
                std = marginal_prob_std(t)[:, None]
                mean = marginal_prob_mean(t)[:, None]
                y0hat = x0hat_full(y, t)
                score = -(y - mean * y0hat) / (std ** 2)
            else:
                raise ValueError("Only X0 mode implemented")
        return score.cpu().numpy().reshape(-1).astype(np.float64)

    def ode_func(t_scalar, y_flat):
        t_torch = torch.tensor(t_scalar, device=device, dtype=torch.float32)
        drift = drift_coeff(t_torch).cpu().numpy()
        return drift * (y_flat + score_eval_wrapper(y_flat, t_scalar))

    res = integrate.solve_ivp(
        ode_func, (t0, eps), init_y.cpu().numpy().reshape(-1),
        rtol=rtol, atol=atol, method="RK45",
    )
    y_final = torch.tensor(res.y[:, -1], dtype=torch.float32).view(N_pts, 1)
    return y_final


def save_sample_batch_implicit(model, dataset, indices_pool, model_dir, epoch,
                               config, device, marginal_prob_mean_fn,
                               marginal_prob_std_fn, meta):
    samples_dir = model_dir / f"samples_epoch_{epoch:04d}"
    samples_dir.mkdir(parents=True, exist_ok=True)

    n = min(config["sample_num"], len(indices_pool))
    chosen = np.random.choice(indices_pool, size=n, replace=False)

    manifest_path = samples_dir / "sample_manifest.txt"
    with open(manifest_path, "w") as f:
        f.write("# idx  png  part_shape  bc_count  cond_str\n")

        for i, idx in enumerate(chosen):
            coords, targets, cond = dataset.get_full_grid(int(idx), device=device)
            y_final = ode_sampler_implicit(
                model, coords, cond,
                marginal_prob_mean_fn, marginal_prob_std_fn,
                functools.partial(drift_coeff, bmin=0.1, bmax=20.0),
                mode=config["mode"], atol=config["sample_atol"],
                rtol=config["sample_rtol"], device=device,
                eps=config["sample_eps"], eval_chunk=config["eval_chunk"],
            )

            x_vox = dataset.voxels[int(idx)]
            Nx, Ny, Nz = x_vox.shape
            y_np = y_final.cpu().numpy().reshape(Nx, Ny, Nz)
            x_bin = (y_np > 0.0).astype(np.float32)      # correct with {-1,+1} targets

            cond_vec = dataset.conds[int(idx)]
            overlay = decode_condition_overlay(cond_vec, meta)

            png_name = f"sample_{i:02d}.png"
            gt_name = f"sample_{i:02d}_gt.png"
            plot_voxel_with_overlays_implicit(
                x_bin, overlay["bc_points"], overlay["load_point"],
                overlay["load_dir"],
                title=f"Epoch {epoch} sample {i} | shape=({Nx},{Ny},{Nz})",
                save_path=samples_dir / png_name,
            )
            plot_voxel_with_overlays_implicit(
                x_vox, overlay["bc_points"], overlay["load_point"],
                overlay["load_dir"],
                title=f"GT {i} | shape=({Nx},{Ny},{Nz})",
                save_path=samples_dir / gt_name,
            )

            bc_count = len(overlay["bc_points"]) if overlay["bc_points"] is not None else 0
            f.write(f"{idx}\t{png_name}\t({Nx},{Ny},{Nz})\t{bc_count}\t"
                    f"{dataset.cond_strs[int(idx)]}\n")

    print(f"Saved implicit samples (+GT) to {samples_dir}")


# --------- validation (full grid, forward-only chunks) ---------

def compute_validation_loss_implicit(model, dataset, val_indices,
                                     marginal_prob_mean_fn, marginal_prob_std_fn,
                                     device, eps, loss_weighting, eval_chunk):
    model.eval()
    total, n_items = 0.0, 0
    with torch.no_grad():
        for idx in val_indices:
            coords, y0, cond = dataset.get_full_grid(int(idx), device=device)
            N_pts = coords.shape[0]
            part_loss, n_chunks = 0.0, 0
            for s in range(0, N_pts, eval_chunk):
                e = min(s + eval_chunk, N_pts)
                loss = vpsde_loss_implicit_x(
                    model,
                    y0[s:e].unsqueeze(0),
                    coords[s:e].unsqueeze(0),
                    cond.unsqueeze(0),
                    marginal_prob_mean_fn, marginal_prob_std_fn,
                    eps=eps, loss_weighting=loss_weighting,
                )
                part_loss += loss.item()
                n_chunks += 1
            total += part_loss / max(1, n_chunks)
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

    def resolve_meta_path(data_path, meta_path=None):
        if meta_path is not None:
            return meta_path
        if data_path.endswith(".npy"):
            return data_path[:-4] + "_meta.npz"
        raise ValueError("Could not infer meta path; please provide --meta_path")

    meta_path = resolve_meta_path(config["data_file"], config["meta_path"])
    meta = np.load(meta_path, allow_pickle=True)

    if "conditioning_spec_json" in meta:
        conditioning_spec = json.loads(str(meta["conditioning_spec_json"]))
    elif "conditioning_spec" in meta:
        conditioning_spec = meta["conditioning_spec"].item()
    else:
        conditioning_spec = {}

    cond_tag = (
        f"bcLoc-{conditioning_spec.get('bc_locations','unknown')}_"
        f"bcDofs-{conditioning_spec.get('bc_dofs','unknown')}_"
        f"loadLoc-{conditioning_spec.get('load_location','unknown')}_"
        f"loadDir-{conditioning_spec.get('load_direction','unknown')}"
    )
    extra_tag = f"_{config['tag']}" if config["tag"] else ""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = (
        f"{cond_tag}_implicitDiffFixed_mode-{config['mode']}_"
        f"epochs-{config['nepochs']}_bs-{config['batchsize']}_"
        f"pts-{config['n_points']}_width-{config['width']}_depth-{config['depth']}"
        f"{extra_tag}_{timestamp}"
    )

    model_save_dir = log_root / run_name
    ckpt_loc_dir = model_save_dir / "checkpoints"
    model_save_dir.mkdir(parents=True, exist_ok=False)
    ckpt_loc_dir.mkdir()

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

    marginal_prob_mean_fn = functools.partial(marginal_prob_mean, bmin=0.1, bmax=20.0)
    marginal_prob_std_fn = functools.partial(marginal_prob_std, bmin=0.1, bmax=20.0)

    model = ImplicitScoreNet(
        cond_dim=dataset.cond_dim,
        t_embed_dim=config["t_embed_dim"],
        width=config["width"],
        depth=config["depth"],
        omega=config["omega"],
        mode=config["mode"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params/1e6:.2f}M")

    if config["load_ckpt"] is not None:
        model.load_state_dict(torch.load(config["load_ckpt"], map_location=device))
        print(f"Loaded checkpoint from {config['load_ckpt']}")

    optimizer = Adam(model.parameters(), lr=config["lr"])
    ema = ExponentialMovingAverage(model.parameters(), decay=config["ema_decay"])
    writer = SummaryWriter(log_dir=str(model_save_dir))

    eps = 1e-5
    best_val_loss = float("inf")
    keep_last_n = 3

    for epoch in range(config["nepochs"]):
        model.train()
        avg_loss, num_items = 0.0, 0

        for coords_b, y0_b, cond_b, _ in train_loader:
            coords_b = coords_b.to(device, non_blocking=True)   # [B,M,3]
            y0_b = y0_b.to(device, non_blocking=True)           # [B,M,1]
            cond_b = cond_b.to(device, non_blocking=True)       # [B,cond_dim]

            loss = vpsde_loss_implicit_x(
                model, y0_b, coords_b, cond_b,
                marginal_prob_mean_fn, marginal_prob_std_fn,
                eps=eps, loss_weighting=config["loss_weighting"],
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.update()

            avg_loss += loss.item()
            num_items += 1

        train_avg = avg_loss / max(1, num_items)
        writer.add_scalar("Loss/train", train_avg, epoch)
        print(f"Epoch {epoch} train loss: {train_avg:.6e}")

        val_loss = compute_validation_loss_implicit(
            model, dataset, val_indices,
            marginal_prob_mean_fn, marginal_prob_std_fn,
            device, eps, config["loss_weighting"], config["eval_chunk"],
        )
        writer.add_scalar("Loss/val", val_loss, epoch)
        print(f"Epoch {epoch} val loss: {val_loss:.6e}")

        if epoch % config["save_every_n_epochs"] == 0:
            with ema.average_parameters():
                torch.save(model.state_dict(), ckpt_loc_dir / f"ckpt_{epoch}.pth")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            with ema.average_parameters():
                torch.save(model.state_dict(), ckpt_loc_dir / "ckpt_best.pth")
            print(f"Saved new best checkpoint at epoch {epoch}")

        ckpts = sorted(ckpt_loc_dir.glob("ckpt_*.pth"), key=os.path.getmtime)
        for p in ckpts[:-keep_last_n]:
            if p.name != "ckpt_best.pth":
                p.unlink()

        se = config["sample_every_n_epochs"]
        if se > 0 and (epoch % se == 0 or epoch == config["nepochs"] - 1):
            model.eval()
            with ema.average_parameters():
                save_sample_batch_implicit(
                    model, dataset, val_indices, model_save_dir, epoch,
                    config, device, marginal_prob_mean_fn,
                    marginal_prob_std_fn, meta,
                )
            model.train()

    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()