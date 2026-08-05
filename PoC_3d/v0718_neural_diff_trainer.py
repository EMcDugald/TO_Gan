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
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch_ema import ExponentialMovingAverage
from scipy import integrate
import matplotlib.pyplot as plt


# --------- arg parsing ---------

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    p = argparse.ArgumentParser(
        description="Implicit 3D conditional VPSDE diffusion for BC/load-conditioned NITO fields"
    )
    p.add_argument("--device", default="cuda", type=str)
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--batchsize", default=2, type=int)          # number of parts per batch
    p.add_argument("--nepochs", default=500, type=int)
    p.add_argument("--lr", default=1e-4, type=float)

    # SIREN-based implicit score model
    p.add_argument("--width", default=256, type=int)
    p.add_argument("--depth", default=4, type=int)
    p.add_argument("--omega", default=30.0, type=float)
    p.add_argument("--t_embed_dim", default=128, type=int)

    p.add_argument("--mode", default="X0", type=str)            # "X0" mode implemented
    p.add_argument("--loss_weighting", default="Simple", type=str)

    # data
    p.add_argument("--data_file", type=str, required=True)      # neural-field train-data .npy
    p.add_argument("--meta_path", type=str, default=None)
    p.add_argument("--nsamples", default=0, type=int)           # limit number of parts (optional)
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

    p.add_argument("--chunk_size", default=8192, type=int)
    return vars(p.parse_args())


# --------- dataset: neural field over NITO voxels ---------

class NeuralFieldDataset3D(Dataset):
    """
    Dataset for BC/load-conditioned implicit fields with mixed shapes.
    Each npy entry: (voxel_arr, cond_vec, label_dummy, cond_str, sample_info)
    voxel_arr can be D×H×W with varying D,H,W across samples.
    """

    def __init__(self, npy_path):
        npy_path = Path(npy_path)
        if not npy_path.exists():
            raise FileNotFoundError(f"Data file not found: {npy_path}")
        data = np.load(npy_path, allow_pickle=True)

        # Keep voxels as a list, not stacked
        self.voxels = [x[0].astype(np.float32) for x in data]      # each: [D,H,W]
        self.conds = [x[1].astype(np.float32) for x in data]       # each: [cond_dim]
        self.cond_strs = [x[3] for x in data]
        self.sample_infos = [x[4] if len(x) > 4 and x[4] is not None else {} for x in data]

        # For convenience, record cond_dim and count
        self.N = len(self.voxels)
        self.cond_dim = self.conds[0].shape[0]

        print(f"Loaded {self.N} neural-field samples with mixed voxel shapes")
        print(f"Condition dim: {self.cond_dim}")
        # You can compute global min/max if you like:
        all_vals = np.concatenate([v.reshape(-1) for v in self.voxels])
        print(f"Target value range: [{all_vals.min():.3f}, {all_vals.max():.3f}]")

    def __len__(self):
        return self.N

    def _make_coord_grid(self, D, H, W, device):
        z = torch.linspace(-1.0, 1.0, D, device=device)
        y = torch.linspace(-1.0, 1.0, H, device=device)
        x = torch.linspace(-1.0, 1.0, W, device=device)
        zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
        coords = torch.stack([xx, yy, zz], dim=-1).view(-1, 3)   # [N_pts,3]
        return coords

    def get_full_grid(self, idx, device="cpu"):
        """
        For sample idx:
          coords: [N_pts,3] in [-1,1]^3
          targets: [N_pts,1] binary occupancies
          cond: [cond_dim]
        """
        x = torch.from_numpy(self.voxels[idx]).to(device)  # [D,H,W]
        D, H, W = x.shape
        coords = self._make_coord_grid(D, H, W, device)
        targets = x.view(-1, 1)
        cond = torch.from_numpy(self.conds[idx]).to(device)
        return coords, targets, cond

    def __getitem__(self, idx):
        """
        Return per-sample voxel (for logging shape), condition, and index.
        The voxel field is kept as [D,H,W]; we don't stack into uniform shape.
        """
        x = torch.from_numpy(self.voxels[idx])    # [D,H,W]
        c = torch.from_numpy(self.conds[idx])     # [cond_dim]
        return x, c, idx


def make_loaders_implicit(
    npy_path,
    batchsize,
    nsamples,
    val_frac=0.1,
    seed=42,
    num_workers=0,
):
    dataset = NeuralFieldDataset3D(npy_path)

    N_total = len(dataset)
    if nsamples is None or nsamples <= 0:
        nsamples = N_total
    nsamples = min(nsamples, N_total)

    indices = np.arange(nsamples)

    class Subset(Dataset):
        def __init__(self, base_dataset, indices):
            self.base = base_dataset
            self.indices = indices
            self.N = len(indices)
            self.cond_dim = base_dataset.cond_dim
            self.cond_strs = [base_dataset.cond_strs[int(i)] for i in indices]
            self.sample_infos = [base_dataset.sample_infos[int(i)] for i in indices]

        def __len__(self):
            return self.N

        def __getitem__(self, idx):
            base_idx = int(self.indices[idx])
            x, c, _ = self.base[base_idx]
            return x, c, base_idx  # keep original index for get_full_grid

    subset = Subset(dataset, indices)

    torch.manual_seed(seed)
    n_val = max(1, int(val_frac * nsamples))
    n_train = nsamples - n_val
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(subset, [n_train, n_val], generator=generator)

    train_loader = DataLoader(
        train_set,
        batch_size=batchsize,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batchsize,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return dataset, train_loader, val_loader


# --------- VPSDE marginal stats ---------

def marginal_prob_mean(t, bmin, bmax):
    log_coeff = 0.5 * (bmax - bmin) * t**2 + bmin * t
    return torch.exp(-0.5 * log_coeff)


def marginal_prob_std(t, bmin, bmax):
    log_coeff = 0.5 * (bmax - bmin) * t**2 + bmin * t
    return torch.sqrt(1. - torch.exp(-log_coeff))


# --------- time embedding (Gaussian Fourier) ---------

class GaussianFourierProjection(nn.Module):
    def __init__(self, embed_dim, scale=30.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale,
                              requires_grad=False)

    def forward(self, t):
        x_proj = t[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


# --------- SIREN building blocks ---------

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


# --------- implicit score model ---------

class ImplicitScoreNet(nn.Module):
    """
    Pointwise diffusion model for implicit NITO fields.
    Inputs per batch:
      coords: [B, N_pts, 3] coordinates in [-1,1]^3
      y_t:    [B, N_pts, 1] noisy field values at time t
      t:      [B]
      cond:   [B, cond_dim]
    Outputs:
      [B, N_pts, 1]: y0hat (X0 mode)
    """
    def __init__(self, cond_dim, t_embed_dim=128,
                 width=256, depth=4, omega=30.0, mode="X0"):
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

        in_dim = 3 + 1 + t_embed_dim + width  # coords(3) + y_t(1) + t_embed + cond_emb(width)

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
        B, N_pts, _ = coords.shape

        t_emb = self.t_embed(t)                # [B, t_embed_dim]
        t_emb = t_emb.unsqueeze(1).expand(B, N_pts, -1)  # [B,N_pts,t_embed_dim]

        cond_emb = self.cond_proj(cond)        # [B,width]
        cond_emb = cond_emb.unsqueeze(1).expand(B, N_pts, -1)  # [B,N_pts,width]

        x = torch.cat([coords, y_t, t_emb, cond_emb], dim=-1)  # [B,N,in_dim]
        h = self.body(x)
        out = self.head(h)                     # [B,N_pts,1]
        return out


# --------- VPSDE loss on implicit fields ---------

def vpsde_loss_implicit_x(model, y0, coords, cond,
                          marginal_prob_mean, marginal_prob_std,
                          eps=1e-5, loss_weighting="Simple"):
    B, N_pts, _ = y0.shape
    device = y0.device
    random_t = torch.rand(B, device=device) * (1. - eps) + eps
    z = torch.randn_like(y0)

    mean_scale = marginal_prob_mean(random_t)    # [B]
    std = marginal_prob_std(random_t)           # [B]
    ms = mean_scale[:, None, None]              # [B,1,1]
    st = std[:, None, None]                     # [B,1,1]

    y_t = y0 * ms + z * st                      # forward diffusion on values

    y0hat = model(coords, y_t, random_t, cond)  # [B,N_pts,1]

    if loss_weighting == "Simple":
        loss = torch.mean(torch.sum((y0hat - y0) ** 2, dim=(1, 2)))
    elif loss_weighting == "Analytical":
        loss = torch.mean(torch.sum(((ms / st) * (y0hat - y0)) ** 2, dim=(1, 2)))
    else:
        raise ValueError("Invalid loss_weighting for X0")
    return loss


# --------- plotting helpers (NITO-style scaling) ---------

def plot_voxel_with_overlays_implicit(
    voxel_arr,
    bc_pts,
    load_pt,
    load_vec,
    title="",
    save_path="plot.png",
):
    voxel_arr = np.asarray(voxel_arr)
    nx, ny, nz = voxel_arr.shape

    c = float(max(nx, ny, nz))

    bc_plot = None
    if bc_pts is not None and len(bc_pts) > 0:
        bc_pts = np.asarray(bc_pts, dtype=np.float64)
        bc_plot = np.column_stack([
            bc_pts[:, 0] * (c - 1.0),
            bc_pts[:, 1] * (c - 1.0),
            bc_pts[:, 2] * (c - 1.0),
        ])

    load_plot = None
    if load_pt is not None:
        lp = np.asarray(load_pt, dtype=np.float64)
        load_plot = np.array([
            lp[0] * c,
            lp[1] * c,
            lp[2] * c,
        ], dtype=np.float64)

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


# --------- BC/load decoding from cond_vec ---------

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

    bc_count_max = int(meta["bc_count_max"]) if "bc_count_max" in meta else None

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

    if cond_slices is not None and conditioning_spec.get("bc_locations") == "fine" and "bc_points" in cond_slices:
        s0, s1 = cond_slices["bc_points"]
        bc_flat = cond_vec[s0:s1]

        if "bc_mask" in cond_slices:
            m0, m1 = cond_slices["bc_mask"]
            bc_mask = cond_vec[m0:m1] > 0.5
        else:
            bc_mask = None

        if "bc_count" in cond_slices:
            c0, c1 = cond_slices["bc_count"]
            bc_count = int(round(float(cond_vec[c0:c1][0])))
        else:
            bc_count = bc_count_max if bc_count_max is not None else len(bc_flat) // 3

        bc_arr = bc_flat.reshape(-1, 3)
        if bc_mask is not None:
            bc_points = bc_arr[bc_mask][:bc_count]
        else:
            bc_points = bc_arr[:bc_count]

    elif cond_slices is not None and conditioning_spec.get("bc_locations") == "coarse" and "bc_bins" in cond_slices:
        x_edges = np.asarray(meta["spatial_bin_edges_x"], dtype=np.float64)
        y_edges = np.asarray(meta["spatial_bin_edges_y"], dtype=np.float64)
        z_edges = np.asarray(meta["spatial_bin_edges_z"], dtype=np.float64)
        x_centers = get_bin_centers(x_edges)
        y_centers = get_bin_centers(y_edges)
        z_centers = get_bin_centers(z_edges)

        s0, s1 = cond_slices["bc_bins"]
        bc_bins_flat = cond_vec[s0:s1]
        bc_arr = bc_bins_flat.reshape(-1, 3)

        if "bc_mask" in cond_slices:
            m0, m1 = cond_slices["bc_mask"]
            bc_mask = cond_vec[m0:m1] > 0.5
            bc_arr = bc_arr[bc_mask]

        if "bc_count" in cond_slices:
            c0, c1 = cond_slices["bc_count"]
            bc_count = int(round(float(cond_vec[c0:c1][0])))
            bc_arr = bc_arr[:bc_count]

        pts = []
        for bx, by, bz in bc_arr:
            bx = int(np.clip(round(float(bx)), 0, len(x_centers) - 1))
            by = int(np.clip(round(float(by)), 0, len(y_centers) - 1))
            bz = int(np.clip(round(float(bz)), 0, len(z_centers) - 1))
            pts.append([x_centers[bx], y_centers[by], z_centers[bz]])
        bc_points = np.asarray(pts, dtype=np.float32)

    if cond_slices is not None and conditioning_spec.get("load_location") == "fine" and "load_point" in cond_slices:
        s0, s1 = cond_slices["load_point"]
        load_point = np.asarray(cond_vec[s0:s1], dtype=np.float32)

    elif cond_slices is not None and conditioning_spec.get("load_location") == "coarse" and "load_bins" in cond_slices:
        x_edges = np.asarray(meta["spatial_bin_edges_x"], dtype=np.float64)
        y_edges = np.asarray(meta["spatial_bin_edges_y"], dtype=np.float64)
        z_edges = np.asarray(meta["spatial_bin_edges_z"], dtype=np.float64)
        x_centers = get_bin_centers(x_edges)
        y_centers = get_bin_centers(y_edges)
        z_centers = get_bin_centers(z_edges)

        s0, s1 = cond_slices["load_bins"]
        load_bins = cond_vec[s0:s1]
        lbx = int(np.clip(round(float(load_bins[0])), 0, len(x_centers) - 1))
        lby = int(np.clip(round(float(load_bins[1])), 0, len(y_centers) - 1))
        lbz = int(np.clip(round(float(load_bins[2])), 0, len(z_centers) - 1))
        load_point = np.array([x_centers[lbx], y_centers[lby], z_centers[lbz]], dtype=np.float32)

    return {
        "bc_points": bc_points,
        "load_point": load_point,
        "load_dir": load_dir,
        "conditioning_spec": conditioning_spec,
    }


# --------- ODE sampler for implicit field ---------

def drift_coeff(t, bmin, bmax):
    betas = bmin + (bmax - bmin) * t
    return -0.5 * betas


def ode_sampler_implicit(score_model, coords, cond,
                         marginal_prob_mean, marginal_prob_std, drift_coeff,
                         mode, atol=1e-4, rtol=1e-4, device="cuda", eps=1e-3):
    N_pts = coords.shape[0]
    t0 = 1.0
    init_y = torch.randn(1, N_pts, 1, device=device)  # [1,N_pts,1]

    def score_eval_wrapper(y_flat, time_scalar):
        y = torch.tensor(y_flat, device=device, dtype=torch.float32).view(1, N_pts, 1)
        t = torch.tensor([time_scalar], device=device, dtype=torch.float32)
        coords_b = coords.unsqueeze(0)               # [1,N_pts,3]
        cond_b = cond.unsqueeze(0)                   # [1,cond_dim]
        with torch.no_grad():
            if mode == "X0":
                std = marginal_prob_std(t)           # [1]
                mean = marginal_prob_mean(t)         # [1]
                ms = mean[:, None, None]            # [1,1,1]
                st = std[:, None, None]             # [1,1,1]
                y0hat = score_model(coords_b, y, t, cond_b)
                score = -(y - ms * y0hat) / (st ** 2)
            else:
                raise ValueError("Only X0 mode implemented in implicit sampler")
        return score.cpu().numpy().reshape(-1).astype(np.float64)

    def ode_func(t_scalar, y_flat):
        t_torch = torch.tensor(t_scalar, device=device, dtype=torch.float32)
        drift = drift_coeff(t_torch).cpu().numpy()
        score_flat = score_eval_wrapper(y_flat, t_scalar)
        return drift * (y_flat + score_flat)

    res = integrate.solve_ivp(
        ode_func,
        (t0, eps),
        init_y.cpu().numpy().reshape(-1),
        rtol=rtol,
        atol=atol,
        method="RK45",
    )

    nsamples = res.y.shape[1]
    y_traj = torch.tensor(res.y, device=device, dtype=torch.float32)  # [N_pts*1, nsamples]
    y_traj = y_traj.view(1, N_pts, 1, nsamples)
    y_final = y_traj[..., -1]  # [1,N_pts,1]
    return y_traj, y_final


def save_sample_batch_implicit(model, dataset, model_dir, epoch, config, device,
                               marginal_prob_mean_fn, marginal_prob_std_fn, meta):
    samples_dir = model_dir / f"samples_epoch_{epoch:04d}"
    samples_dir.mkdir(parents=True, exist_ok=True)

    batch_size = min(config["sample_num"], len(dataset))
    chosen = np.random.choice(len(dataset), size=batch_size, replace=False)

    manifest_path = samples_dir / "sample_manifest.txt"
    with open(manifest_path, "w") as f:
        f.write("# idx  png_filename  part_shape  bc_count  cond_str\n")

        for i, idx in enumerate(chosen):
            coords, targets, cond = dataset.get_full_grid(int(idx), device=device)
            with torch.no_grad():
                y_traj, y_final = ode_sampler_implicit(
                    model,
                    coords,
                    cond,
                    marginal_prob_mean_fn,
                    marginal_prob_std_fn,
                    functools.partial(drift_coeff, bmin=0.1, bmax=20.0),
                    mode=config["mode"],
                    atol=config["sample_atol"],
                    rtol=config["sample_rtol"],
                    device=device,
                    eps=config["sample_eps"],
                )

            # Get per-sample voxel to recover shape
            x_vox = dataset.voxels[int(idx)]   # numpy [D,H,W]
            D, H, W = x_vox.shape

            y_samples = y_final.cpu().numpy().reshape(D, H, W)
            x_bin = (y_samples > 0.0).astype(np.float32)

            # Get per-sample condition vector
            cond_vec = dataset.conds[int(idx)]
            overlay = decode_condition_overlay(cond_vec, meta)
            bc_points = overlay["bc_points"]
            load_point = overlay["load_point"]
            load_dir = overlay["load_dir"]

            part_shape = (D, H, W)
            bc_count = len(bc_points) if bc_points is not None else 0
            cond_str = dataset.cond_strs[int(idx)] if hasattr(dataset, "cond_strs") else ""

            png_name = f"sample_{i:02d}.png"
            png_path = samples_dir / png_name

            title = f"Epoch {epoch} sample {i} | shape={part_shape} | bc_count={bc_count}"

            plot_voxel_with_overlays_implicit(
                voxel_arr=x_bin,
                bc_pts=bc_points,
                load_pt=load_point,
                load_vec=load_dir,
                title=title,
                save_path=png_path,
            )

            f.write(
                f"{i}\t{png_name}\t{part_shape}\t{bc_count}\t{cond_str}\n"
            )

    print(f"Saved implicit samples with BC/load overlays to {samples_dir}")


# --------- validation loss ---------

def compute_validation_loss_implicit(
    model, dataset, val_loader,
    marginal_prob_mean_fn, marginal_prob_std_fn,
    device, eps, loss_weighting
):
    model.eval()
    val_loss = 0.0
    num_items = 0

    with torch.no_grad():
        for X_batch, C_batch, idx_batch in val_loader:
            B_parts = X_batch.shape[0]
            for b in range(B_parts):
                idx = int(idx_batch[b].item())
                coords, targets, cond = dataset.get_full_grid(idx, device=device)
                coords = coords.unsqueeze(0)      # [1,N_pts,3]
                y0 = targets.unsqueeze(0)         # [1,N_pts,1]
                cond = cond.unsqueeze(0)          # [1,cond_dim]

                loss = vpsde_loss_implicit_x(
                    model,
                    y0,
                    coords,
                    cond,
                    marginal_prob_mean_fn,
                    marginal_prob_std_fn,
                    eps=eps,
                    loss_weighting=loss_weighting,
                )

                val_loss += loss.item()
                num_items += 1

    model.train()
    return val_loss / max(1, num_items)


# --------- main training loop ---------

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

    meta_info = {
        "conditioning_spec": conditioning_spec,
    }

    cond_tag = (
        f"bcLoc-{conditioning_spec.get('bc_locations','unknown')}_"
        f"bcDofs-{conditioning_spec.get('bc_dofs','unknown')}_"
        f"loadLoc-{conditioning_spec.get('load_location','unknown')}_"
        f"loadDir-{conditioning_spec.get('load_direction','unknown')}"
    )
    extra_tag = f"_{config['tag']}" if config["tag"] else ""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    run_name = (
        f"{cond_tag}_implicitDiff_mode-{config['mode']}_"
        f"epochs-{config['nepochs']}_bs-{config['batchsize']}_"
        f"width-{config['width']}_depth-{config['depth']}"
        f"{extra_tag}_{timestamp}"
    )

    model_save_dir = log_root / run_name
    ckpt_loc_dir = model_save_dir / "checkpoints"
    model_save_dir.mkdir(parents=True, exist_ok=False)
    ckpt_loc_dir.mkdir()

    with open(model_save_dir / "hparams.json", "w") as f:
        json.dump({**config, **meta_info, "meta_path": str(meta_path)}, f, indent=2)

    dataset, train_loader, val_loader = make_loaders_implicit(
        npy_path=config["data_file"],
        batchsize=config["batchsize"],
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

    if config["load_ckpt"] is not None:
        ckpt_loc = Path(config["load_ckpt"])
        model.load_state_dict(torch.load(ckpt_loc, map_location=device))
        print(f"Loaded previous checkpoint from {ckpt_loc}")

    optimizer = Adam(model.parameters(), lr=config["lr"])
    ema = ExponentialMovingAverage(model.parameters(), decay=config["ema_decay"])
    writer = SummaryWriter(log_dir=str(model_save_dir))

    eps = 1e-5
    best_val_loss = float("inf")
    keep_last_n = 3
    sample_every = config["sample_every_n_epochs"]

    # def train_one_epoch(epoch):
    #     model.train()
    #     avg_loss = 0.0
    #     num_items = 0

    #     for X_batch, C_batch, idx_batch in train_loader:
    #         B_parts = X_batch.shape[0]
    #         for b in range(B_parts):
    #             idx = int(idx_batch[b].item())
    #             coords, targets, cond = dataset.get_full_grid(idx, device=device)
    #             coords = coords.unsqueeze(0)   # [1,N_pts,3]
    #             y0 = targets.unsqueeze(0)      # [1,N_pts,1]
    #             cond = cond.unsqueeze(0)       # [1,cond_dim]

    #             loss = vpsde_loss_implicit_x(
    #                 model,
    #                 y0,
    #                 coords,
    #                 cond,
    #                 marginal_prob_mean_fn,
    #                 marginal_prob_std_fn,
    #                 eps=eps,
    #                 loss_weighting=config["loss_weighting"],
    #             )

    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()
    #             ema.update()

    #             avg_loss += loss.item()
    #             num_items += 1

    #     train_avg_loss = avg_loss / max(1, num_items)
    #     writer.add_scalar("Loss/train", train_avg_loss, epoch)
    #     print(f"Epoch {epoch} train loss (per-part): {train_avg_loss:.6e}")
    #     return train_avg_loss


    def train_one_epoch(epoch):
        model.train()
        avg_loss = 0.0
        num_items = 0

        chunk_size = config["chunk_size"]

        for X_batch, C_batch, idx_batch in train_loader:
            B_parts = X_batch.shape[0]
            for b in range(B_parts):
                idx = int(idx_batch[b].item())
                coords, targets, cond = dataset.get_full_grid(idx, device=device)
                N_pts = coords.shape[0]

                # Expand cond once per part
                cond_b = cond.unsqueeze(0)  # [1, cond_dim]

                for start in range(0, N_pts, chunk_size):
                    end = min(start + chunk_size, N_pts)
                    coords_chunk = coords[start:end].unsqueeze(0)   # [1,M,3]
                    y0_chunk = targets[start:end].unsqueeze(0)      # [1,M,1]

                    loss = vpsde_loss_implicit_x(
                        model,
                        y0_chunk,
                        coords_chunk,
                        cond_b,
                        marginal_prob_mean_fn,
                        marginal_prob_std_fn,
                        eps=eps,
                        loss_weighting=config["loss_weighting"],
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    ema.update()

                    avg_loss += loss.item()
                    num_items += 1

        train_avg_loss = avg_loss / max(1, num_items)
        writer.add_scalar("Loss/train", train_avg_loss, epoch)
        print(f"Epoch {epoch} train loss (per-part-chunk): {train_avg_loss:.6e}")
        return train_avg_loss

    def compute_validation_loss(epoch):
        val_loss = compute_validation_loss_implicit(
            model,
            dataset,
            val_loader,
            marginal_prob_mean_fn,
            marginal_prob_std_fn,
            device,
            eps,
            config["loss_weighting"],
        )
        writer.add_scalar("Loss/val", val_loss, epoch)
        print(f"Epoch {epoch} val loss (per-part): {val_loss:.6e}")
        return val_loss

    nepochs = config["nepochs"]
    save_every = config["save_every_n_epochs"]

    for epoch in range(nepochs):
        train_loss = train_one_epoch(epoch)
        val_loss = compute_validation_loss(epoch)

        if epoch % save_every == 0:
            with ema.average_parameters():
                ckpt_path = ckpt_loc_dir / f"ckpt_{epoch}.pth"
                torch.save(model.state_dict(), ckpt_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            with ema.average_parameters():
                best_ckpt_path = ckpt_loc_dir / "ckpt_best.pth"
                torch.save(model.state_dict(), best_ckpt_path)
            print(f"Saved new best checkpoint at epoch {epoch}")

        ckpts = sorted(ckpt_loc_dir.glob("ckpt_*.pth"), key=os.path.getmtime)
        ckpts_to_remove = ckpts[:-keep_last_n]
        for p in ckpts_to_remove:
            if p.name != "ckpt_best.pth":
                p.unlink()

        if sample_every > 0 and (epoch % sample_every == 0 or epoch == nepochs - 1):
            model.eval()
            with ema.average_parameters():
                save_sample_batch_implicit(
                    model,
                    dataset,
                    model_save_dir,
                    epoch,
                    config,
                    device,
                    marginal_prob_mean_fn,
                    marginal_prob_std_fn,
                    meta,
                )
            model.train()

    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()