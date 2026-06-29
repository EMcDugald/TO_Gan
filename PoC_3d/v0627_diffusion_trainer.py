import os
from pathlib import Path
import argparse
import functools
import json
from datetime import datetime
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch_ema import ExponentialMovingAverage
from scipy import integrate
import matplotlib.pyplot as plt


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="3D conditional VPSDE diffusion for BC/load-conditioned voxel structures"
    )
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--batchsize", default=4, type=int)
    parser.add_argument("--nepochs", default=500, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--unet_ch1_dim", default=32, type=int)
    parser.add_argument("--unet_ch2_dim", default=64, type=int)
    parser.add_argument("--unet_ch3_dim", default=128, type=int)
    parser.add_argument("--t_embed_dim", default=128, type=int)
    parser.add_argument("--cond_embed_dim", default=128, type=int)
    parser.add_argument("--cond_ch", default=8, type=int)
    parser.add_argument("--cond_drop_prob", default=0.0, type=float)
    parser.add_argument("--mode", default="X0", type=str)
    parser.add_argument("--loss_weighting", default="Simple", type=str)
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--meta_path", type=str, default=None)
    parser.add_argument("--img_size", default=32, type=int)
    parser.add_argument("--nsamples", default=0, type=int)
    parser.add_argument("--val_frac", default=0.1, type=float)
    parser.add_argument("--log_root", type=str, required=True)
    parser.add_argument("--save_every_n_epochs", default=50, type=int)
    parser.add_argument("--sample_every_n_epochs", default=50, type=int)
    parser.add_argument("--sample_num", default=4, type=int)
    parser.add_argument("--ema_decay", default=0.9999, type=float)
    parser.add_argument("--load_version", default=None, type=int)
    parser.add_argument("--sample_atol", default=1e-4, type=float)
    parser.add_argument("--sample_rtol", default=1e-4, type=float)
    parser.add_argument("--sample_eps", default=1e-3, type=float)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--tag", default="", type=str)
    return vars(parser.parse_args())


class CondVoxelDataset3D(Dataset):
    """
    Dataset for BC/load-conditioned voxel structures:
      npy entries: (voxel, cond_vec, label, cond_str)
      voxel in [0, 1] → rescaled to [-1, 1]
    """
    def __init__(self, npy_path, img_size=32):
        npy_path = Path(npy_path)
        if not npy_path.exists():
            raise FileNotFoundError(f"Data file not found: {npy_path}")
        data = np.load(npy_path, allow_pickle=True)

        voxels = [x[0] for x in data]
        conds = [x[1] for x in data]
        labels = [x[2] for x in data]
        cond_strs = [x[3] for x in data]

        X = np.stack(voxels).astype(np.float32)
        C = np.stack(conds).astype(np.float32)
        y = np.asarray(labels).astype(np.int64)

        if X.ndim == 4:
            X = X[:, None, :, :, :]
        # Scale to [-1, 1] like the GAN trainers
        X = X * 2.0 - 1.0

        self.X = X
        self.C = C
        self.y = y
        self.cond_strs = cond_strs
        self.N, self.Cx, self.D, self.H, self.W = self.X.shape
        self.cond_dim = self.C.shape[1]

        assert self.D == img_size and self.H == img_size and self.W == img_size, (
            f"Expected cubic {img_size}^3 data, got {self.X.shape}"
        )

        self.mean = float(self.X.mean())
        self.std = float(self.X.std()) if float(self.X.std()) > 0 else 1.0
        print(f"Loaded {self.N} samples with voxel shape {self.X.shape[1:]}")
        print(f"Condition dim: {self.cond_dim}")
        print(f"Value range: [{self.X.min():.3f}, {self.X.max():.3f}]")

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X[idx]),
            torch.from_numpy(self.C[idx]),
            torch.tensor(self.y[idx], dtype=torch.long),
            idx,
        )


def make_loaders_positive_only(
    npy_path,
    batchsize,
    nsamples,
    img_size=32,
    val_frac=0.1,
    seed=42,
    num_workers=0,
):
    """
    Make train/val loaders from positive-only subset (y == 1),
    for direct comparison to the GAN's positive distribution.
    """
    dataset = CondVoxelDataset3D(npy_path, img_size=img_size)

    # Positive mask
    pos_indices = np.where(dataset.y == 1)[0]
    if len(pos_indices) == 0:
        raise ValueError("No positive (label==1) samples found in dataset")

    # Restrict to nsamples positives if requested
    if nsamples is None or nsamples <= 0:
        nsamples = len(pos_indices)
    nsamples = min(nsamples, len(pos_indices))

    # Build positive-only sub-dataset
    pos_indices = pos_indices[:nsamples]
    pos_X = dataset.X[pos_indices]
    pos_C = dataset.C[pos_indices]
    pos_y = dataset.y[pos_indices]
    pos_cond_strs = [dataset.cond_strs[int(i)] for i in pos_indices]

    class PositiveSubset(Dataset):
        def __init__(self, X, C, y, cond_strs):
            self.X = X
            self.C = C
            self.y = y
            self.cond_strs = cond_strs
            self.N = X.shape[0]
            self.cond_dim = C.shape[1]

        def __len__(self):
            return self.N

        def __getitem__(self, idx):
            return (
                torch.from_numpy(self.X[idx]),
                torch.from_numpy(self.C[idx]),
                torch.tensor(self.y[idx], dtype=torch.long),
                idx,
            )

    pos_dataset = PositiveSubset(pos_X, pos_C, pos_y, pos_cond_strs)

    # Split positive-only subset into train/val
    torch.manual_seed(seed)
    n_val = max(1, int(val_frac * nsamples))
    n_train = nsamples - n_val
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(pos_dataset, [n_train, n_val], generator=generator)

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
    return pos_dataset, train_loader, val_loader


def marginal_prob_mean(t, bmin, bmax):
    log_coeff = 0.5 * (bmax - bmin) * t**2 + bmin * t
    return torch.exp(-0.5 * log_coeff)


def marginal_prob_std(t, bmin, bmax):
    log_coeff = 0.5 * (bmax - bmin) * t**2 + bmin * t
    return torch.sqrt(1. - torch.exp(-log_coeff))


class GaussianFourierProjection(nn.Module):
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, t):
        x_proj = t[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


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


class ScoreNet3DVoxelCond(nn.Module):
    def __init__(self, in_ch, cond_dim, cond_embed_dim, cond_ch, c1, c2, c3, ed, mode, marginal_prob_std):
        super().__init__()
        self.marginal_prob_std = marginal_prob_std
        self.in_ch = in_ch
        self.cond_ch = cond_ch
        self.cond_proj = CondProjectTo3D(cond_dim, cond_embed_dim, cond_ch) if cond_dim > 0 and cond_ch > 0 else None
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.ed = ed
        self.mode = mode

        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=ed),
            nn.Linear(ed, ed),
        )

        act_x0 = lambda x: x + torch.sin(x) ** 2
        act_noise = lambda x: x * torch.sigmoid(x)
        self.act = act_x0 if mode == "X0" else act_noise

        self.conv1 = nn.Conv3d(in_ch + cond_ch, c1, 3, stride=1, padding=1, bias=False)
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

        self.tconv4 = nn.ConvTranspose3d(c3, c3, 3, stride=2, padding=1, output_padding=1, bias=False)
        self.dense5 = Dense3D(ed, c3)
        self.tgnorm4 = nn.GroupNorm(max(1, c3 // 8), c3)

        self.tconv3 = nn.ConvTranspose3d(c3 + c3, c2, 3, stride=2, padding=1, output_padding=1, bias=False)
        self.dense6 = Dense3D(ed, c2)
        self.tgnorm3 = nn.GroupNorm(max(1, c2 // 8), c2)

        self.tconv2 = nn.ConvTranspose3d(c2 + c2, c1, 3, stride=2, padding=1, output_padding=1, bias=False)
        self.dense7 = Dense3D(ed, c1)
        self.tgnorm2 = nn.GroupNorm(max(1, c1 // 8), c1)

        self.tconv1 = nn.ConvTranspose3d(c1 + c1, in_ch, 3, stride=1, padding=1)

        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        num_params = sum(np.prod(p.size()) for p in model_parameters)
        print(f"ScoreNet3DVoxelCond has {num_params} trainable parameters")

    def forward(self, x, t, cond=None):
        embed = self.act(self.embed(t))

        if self.cond_proj is not None and cond is not None:
            cond_3d = self.cond_proj(cond, x.shape[-3:])
            x = torch.cat([x, cond_3d], dim=1)

        h1 = self.conv1(x)
        h1 = h1 + self.dense1(embed)
        h1 = self.gnorm1(h1)
        h1 = self.act(h1)

        h2 = self.conv2(h1)
        h2 = h2 + self.dense2(embed)
        h2 = self.gnorm2(h2)
        h2 = self.act(h2)

        h3 = self.conv3(h2)
        h3 = h3 + self.dense3(embed)
        h3 = self.gnorm3(h3)
        h3 = self.act(h3)

        h4 = self.conv4(h3)
        h4 = h4 + self.dense4(embed)
        h4 = self.gnorm4(h4)
        h4 = self.act(h4)

        h = self.tconv4(h4)
        h = h + self.dense5(embed)
        h = self.tgnorm4(h)
        h = self.act(h)

        h = self.tconv3(torch.cat([h, h3], dim=1))
        h = h + self.dense6(embed)
        h = self.tgnorm3(h)
        h = self.act(h)

        h = self.tconv2(torch.cat([h, h2], dim=1))
        h = h + self.dense7(embed)
        h = self.tgnorm2(h)
        h = self.act(h)

        h = self.tconv1(torch.cat([h, h1], dim=1))
        return h


def maybe_drop_condition(cond, drop_prob):
    if cond is None or drop_prob <= 0.0:
        return cond
    keep = (torch.rand(cond.shape[0], device=cond.device) > drop_prob).float().unsqueeze(1)
    return cond * keep


def vpsde_loss_fn_x(model, x, marginal_prob_mean, marginal_prob_std, cond, eps=1e-5, loss_weighting="Simple"):
    B = x.shape[0]
    device = x.device
    random_t = torch.rand(B, device=device) * (1. - eps) + eps
    z = torch.randn_like(x)

    mean_scale = marginal_prob_mean(random_t)
    std = marginal_prob_std(random_t)
    ms = mean_scale[:, None, None, None, None]
    st = std[:, None, None, None, None]

    perturbed_x = x * ms + z * st
    xhat = model(perturbed_x, random_t, cond)

    if loss_weighting == "Simple":
        loss = torch.mean(torch.sum((xhat - x) ** 2, dim=(1, 2, 3, 4)))
    elif loss_weighting == "Analytical":
        loss = torch.mean(torch.sum(((ms / st) * (xhat - x)) ** 2, dim=(1, 2, 3, 4)))
    else:
        raise ValueError("Invalid loss_weighting for X0")
    return loss, xhat


def vpsde_loss_fn_noise(model, x, marginal_prob_mean, marginal_prob_std, cond, eps=1e-5, loss_weighting="Analytical"):
    B = x.shape[0]
    device = x.device
    random_t = torch.rand(B, device=device) * (1. - eps) + eps
    z = torch.randn_like(x)

    mean_scale = marginal_prob_mean(random_t)
    std = marginal_prob_std(random_t)
    ms = mean_scale[:, None, None, None, None]
    st = std[:, None, None, None, None]

    perturbed_x = x * ms + z * st
    noisehat = model(perturbed_x, random_t, cond)

    if loss_weighting == "Analytical":
        loss = torch.mean(torch.sum((noisehat - z) ** 2, dim=(1, 2, 3, 4)))
    elif loss_weighting == "x0_simple":
        loss = torch.mean(torch.sum(((st / ms) * (noisehat - z)) ** 2, dim=(1, 2, 3, 4)))
    else:
        raise ValueError("Invalid loss_weighting for Noise")
    return loss


def vpsde_loss_fn(model, x, marginal_prob_mean, marginal_prob_std, cond, eps=1e-5, mode="X0", loss_weighting="Simple"):
    if mode == "X0":
        return vpsde_loss_fn_x(model, x, marginal_prob_mean, marginal_prob_std, cond, eps, loss_weighting)
    elif mode == "Noise":
        loss = vpsde_loss_fn_noise(model, x, marginal_prob_mean, marginal_prob_std, cond, eps, loss_weighting)
        return loss, None
    else:
        raise ValueError("Mode Score not implemented")


def drift_coeff(t, bmin, bmax):
    betas = bmin + (bmax - bmin) * t
    return -0.5 * betas


# ---- helpers for decoding and plotting BC/load overlays ----

def get_bin_centers(edges):
    edges = np.asarray(edges, dtype=np.float64)
    return 0.5 * (edges[:-1] + edges[1:])


def resolve_meta_path(data_path, meta_path=None):
    if meta_path is not None:
        return meta_path
    if data_path.endswith('.npy'):
        return data_path[:-4] + '_meta.npz'
    raise ValueError('Could not infer meta path; please provide --meta_path')


def load_meta_for_data(data_path, meta_path=None):
    meta_path = resolve_meta_path(data_path, meta_path)
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Meta file not found: {meta_path}")
    return np.load(meta_path, allow_pickle=True), meta_path


def decode_condition_overlay(cond_vec, meta):
    """
    Decode BC points and load point/dir from a condition vector that uses
    cond_slices + conditioning_spec, as in the new GAN trainers.
    """
    cond_vec = np.asarray(cond_vec, dtype=np.float32)

    # ---- load cond_slices ----
    if 'cond_slices' in meta:
        cond_slices = meta['cond_slices'].item()
    elif 'cond_slices_json' in meta:
        cond_slices = json.loads(str(meta['cond_slices_json']))
    else:
        cond_slices = None

    bc_count_max = int(meta['bc_count_max']) if 'bc_count_max' in meta else None

    # ---- load conditioning_spec ----
    if 'conditioning_spec' in meta:
        conditioning_spec = meta['conditioning_spec'].item()
    elif 'conditioning_spec_json' in meta:
        conditioning_spec = json.loads(str(meta['conditioning_spec_json']))
    else:
        # fall back to separate mode fields if present
        conditioning_spec = {
            'bc_locations': str(meta['bc_locations_mode']) if 'bc_locations_mode' in meta else 'unknown',
            'bc_dofs': str(meta['bc_dofs_mode']) if 'bc_dofs_mode' in meta else 'unknown',
            'load_location': str(meta['load_location_mode']) if 'load_location_mode' in meta else 'unknown',
            'load_direction': str(meta['load_direction_mode']) if 'load_direction_mode' in meta else 'unknown',
        }

    bc_points = np.empty((0, 3), dtype=np.float32)
    load_point = None
    load_dir = None

    # Load dir from slices if available
    if cond_slices is not None and 'load_dir' in cond_slices:
        s0, s1 = cond_slices['load_dir']
        load_dir = np.asarray(cond_vec[s0:s1], dtype=np.float32)

    # Fine BC locations
    if cond_slices is not None and conditioning_spec.get('bc_locations') == 'fine' and 'bc_points' in cond_slices:
        s0, s1 = cond_slices['bc_points']
        bc_flat = cond_vec[s0:s1]

        if 'bc_mask' in cond_slices:
            m0, m1 = cond_slices['bc_mask']
            bc_mask = cond_vec[m0:m1] > 0.5
        else:
            bc_mask = None

        if 'bc_count' in cond_slices:
            c0, c1 = cond_slices['bc_count']
            bc_count = int(round(float(cond_vec[c0:c1][0])))
        else:
            bc_count = bc_count_max if bc_count_max is not None else len(bc_flat) // 3

        bc_arr = bc_flat.reshape(-1, 3)
        if bc_mask is not None:
            bc_points = bc_arr[bc_mask][:bc_count]
        else:
            bc_points = bc_arr[:bc_count]

    # Coarse BC locations
    elif cond_slices is not None and conditioning_spec.get('bc_locations') == 'coarse' and 'bc_bins' in cond_slices:
        x_edges = np.asarray(meta['spatial_bin_edges_x'], dtype=np.float64)
        y_edges = np.asarray(meta['spatial_bin_edges_y'], dtype=np.float64)
        z_edges = np.asarray(meta['spatial_bin_edges_z'], dtype=np.float64)
        x_centers = get_bin_centers(x_edges)
        y_centers = get_bin_centers(y_edges)
        z_centers = get_bin_centers(z_edges)

        s0, s1 = cond_slices['bc_bins']
        bc_bins_flat = cond_vec[s0:s1]
        bc_arr = bc_bins_flat.reshape(-1, 3)

        if 'bc_mask' in cond_slices:
            m0, m1 = cond_slices['bc_mask']
            bc_mask = cond_vec[m0:m1] > 0.5
            bc_arr = bc_arr[bc_mask]

        if 'bc_count' in cond_slices:
            c0, c1 = cond_slices['bc_count']
            bc_count = int(round(float(cond_vec[c0:c1][0])))
            bc_arr = bc_arr[:bc_count]

        pts = []
        for bx, by, bz in bc_arr:
            bx = int(np.clip(round(float(bx)), 0, len(x_centers) - 1))
            by = int(np.clip(round(float(by)), 0, len(y_centers) - 1))
            bz = int(np.clip(round(float(bz)), 0, len(z_centers) - 1))
            pts.append([x_centers[bx], y_centers[by], z_centers[bz]])
        bc_points = np.asarray(pts, dtype=np.float32)

    # Fine load location
    if cond_slices is not None and conditioning_spec.get('load_location') == 'fine' and 'load_point' in cond_slices:
        s0, s1 = cond_slices['load_point']
        load_point = np.asarray(cond_vec[s0:s1], dtype=np.float32)

    # Coarse load location
    elif cond_slices is not None and conditioning_spec.get('load_location') == 'coarse' and 'load_bins' in cond_slices:
        x_edges = np.asarray(meta['spatial_bin_edges_x'], dtype=np.float64)
        y_edges = np.asarray(meta['spatial_bin_edges_y'], dtype=np.float64)
        z_edges = np.asarray(meta['spatial_bin_edges_z'], dtype=np.float64)
        x_centers = get_bin_centers(x_edges)
        y_centers = get_bin_centers(y_edges)
        z_centers = get_bin_centers(z_edges)

        s0, s1 = cond_slices['load_bins']
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


def plot_voxel_with_conditions(binary_arr, save_path, title='', bc_points=None, load_point=None, load_vec=None):
    binary_arr = np.asarray(binary_arr)
    nx, ny, nz = binary_arr.shape

    bc_plot = None
    if bc_points is not None and len(bc_points) > 0:
        bc_points = np.asarray(bc_points, dtype=np.float64)
        bc_plot = np.column_stack([
            bc_points[:, 0] * (nx - 1),
            bc_points[:, 1] * (ny - 1),
            bc_points[:, 2] * (nz - 1),
        ])

    load_plot = None
    if load_point is not None:
        load_point = np.asarray(load_point, dtype=np.float64)
        load_plot = np.array([
            load_point[0] * (nx - 1),
            load_point[1] * (ny - 1),
            load_point[2] * (nz - 1),
        ], dtype=np.float64)

    fig = plt.figure(figsize=(18, 6))
    views = [(25, 35, 'View 1'), (25, 125, 'View 2'), (65, 35, 'View 3')]

    for k, (elev, azim, subtitle) in enumerate(views, start=1):
        ax = fig.add_subplot(1, 3, k, projection='3d')
        ax.voxels(binary_arr > 0, edgecolor='k', linewidth=0.15, alpha=0.72)
        ax.set_xlim(0, nx)
        ax.set_ylim(0, ny)
        ax.set_zlim(0, nz)

        if bc_plot is not None:
            ax.scatter(
                bc_plot[:, 0], bc_plot[:, 1], bc_plot[:, 2],
                c='red', s=36, marker='o', edgecolors='white', linewidths=0.6,
                depthshade=False, label='BC'
            )
            for i, p in enumerate(bc_plot):
                ax.text(p[0], p[1], p[2], f'BC{i}', color='red', fontsize=8)

        if load_plot is not None:
            ax.scatter(
                [load_plot[0]], [load_plot[1]], [load_plot[2]],
                c='dodgerblue', s=64, marker='^', edgecolors='white', linewidths=0.7,
                depthshade=False, label='Load point'
            )
            if load_vec is not None and np.linalg.norm(load_vec) > 0:
                load_vec = np.asarray(load_vec, dtype=np.float64)
                scale = max(nx, ny, nz) * 0.18
                ax.quiver(
                    load_plot[0], load_plot[1], load_plot[2],
                    load_vec[0], load_vec[1], load_vec[2],
                    color='dodgerblue', linewidth=2.0, length=scale, normalize=True
                )
            ax.text(load_plot[0], load_plot[1], load_plot[2], 'Load', color='dodgerblue', fontsize=8)

        ax.view_init(elev=elev, azim=azim)
        ax.set_title(subtitle)
        ax.set_box_aspect((nx, ny, nz))
        ax.set_axis_off()

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=220)
    plt.close(fig)


# ---- ODE sampler ----

def ode_sampler_voxel_cond(score_model, x_shape, cond, marginal_prob_mean, marginal_prob_std, drift_coeff,
                           mode, atol=1e-4, rtol=1e-4, device="cuda", eps=1e-3):
    t0 = 1.0
    init_x = torch.randn(x_shape, device=device)

    def score_eval_wrapper(sample_flat, time_steps):
        sample = torch.tensor(sample_flat, device=device, dtype=torch.float32).reshape(x_shape)
        time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0],))
        with torch.no_grad():
            score = score_model(sample, time_steps, cond=cond)
        return score.cpu().numpy().reshape(-1).astype(np.float64)

    def ode_func(t_scalar, x_flat):
        time_steps = np.ones((x_shape[0],), dtype=np.float32) * t_scalar
        t_torch = torch.tensor(t_scalar, device=device, dtype=torch.float32)
        drift = drift_coeff(t_torch).cpu().numpy()
        std = marginal_prob_std(t_torch).cpu().numpy()
        mean_scale = marginal_prob_mean(t_torch).cpu().numpy()

        if mode == "X0":
            x0hat_flat = score_eval_wrapper(x_flat, time_steps)
            score = -(x_flat - mean_scale * x0hat_flat) / (std ** 2)
        else:
            raise ValueError("Only X0 mode implemented in voxel sampler")

        return drift * (x_flat + score)

    res = integrate.solve_ivp(
        ode_func,
        (t0, eps),
        init_x.cpu().numpy().reshape(-1),
        rtol=rtol,
        atol=atol,
        method="RK45",
    )

    nsamples = res.y.shape[1]
    x_traj = torch.tensor(res.y, device=device, dtype=torch.float32)
    x_traj = x_traj.view(*x_shape, nsamples)
    x_final = x_traj[..., -1]
    return x_traj, x_final


def save_sample_batch(model, dataset, model_dir, epoch, config, device,
                      marginal_prob_mean_fn, marginal_prob_std_fn, drift_coeff_fn, meta):
    samples_dir = model_dir / f"samples_epoch_{epoch:04d}"
    samples_dir.mkdir(parents=True, exist_ok=True)

    batch_size = config["sample_num"]
    img_size = config["img_size"]
    x_shape = torch.Size([batch_size, 1, img_size, img_size, img_size])

    # Sample conditions from positive-only dataset
    chosen = np.random.choice(len(dataset), size=batch_size, replace=(len(dataset) < batch_size))
    cond_np = dataset.C[chosen]
    cond = torch.from_numpy(cond_np).float().to(device)

    with torch.no_grad():
        x_traj, x_final = ode_sampler_voxel_cond(
            model,
            x_shape,
            cond,
            marginal_prob_mean_fn,
            marginal_prob_std_fn,
            drift_coeff_fn,
            mode=config["mode"],
            atol=config["sample_atol"],
            rtol=config["sample_rtol"],
            device=device,
            eps=config["sample_eps"],
        )

    x_samples = x_final.cpu().numpy()
    x_bin = (x_samples > 0.0).astype(np.float32)

    np.save(samples_dir / "traj.npy", x_traj.cpu().numpy())
    np.save(samples_dir / "fields_continuous.npy", x_samples)
    np.save(samples_dir / "fields_binary.npy", x_bin)
    np.save(samples_dir / "sample_conditions.npy", cond_np)

    manifest_path = samples_dir / "sample_manifest.txt"
    with open(manifest_path, "w") as f:
        f.write("# idx  png_filename  bc_mode  load_loc_mode  load_dir_mode\n")
        for i in range(min(batch_size, 16)):
            overlay = decode_condition_overlay(cond_np[i], meta)
            bc_points = overlay["bc_points"]
            load_point = overlay["load_point"]
            load_dir = overlay["load_dir"]
            spec = overlay["conditioning_spec"]

            png_name = f"sample_{i:02d}.png"
            png_path = samples_dir / png_name

            title = (
                f"Epoch {epoch} sample {i} | "
                f"bc={spec.get('bc_locations')} | "
                f"load_loc={spec.get('load_location')} | "
                f"load_dir={spec.get('load_direction')}"
            )
            plot_voxel_with_conditions(
                x_bin[i, 0],
                save_path=png_path,
                title=title,
                bc_points=bc_points,
                load_point=load_point,
                load_vec=load_dir,
            )

            f.write(
                f"{i}\t{png_name}\t"
                f"{spec.get('bc_locations')}\t"
                f"{spec.get('load_location')}\t"
                f"{spec.get('load_direction')}\n"
            )

    print(f"Saved samples with BC/load overlays to {samples_dir}")


def main():
    config = parse_args()
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    log_root = Path(config["log_root"])
    log_root.mkdir(parents=True, exist_ok=True)

    # Load meta and log some key mass/conditioning info
    meta, meta_path = load_meta_for_data(config["data_file"], config["meta_path"])

    if 'conditioning_spec_json' in meta:
        conditioning_spec = json.loads(str(meta['conditioning_spec_json']))
    elif 'conditioning_spec' in meta:
        conditioning_spec = meta['conditioning_spec'].item()
    else:
        conditioning_spec = {
            'bc_locations': str(meta['bc_locations_mode']) if 'bc_locations_mode' in meta else 'unknown',
            'bc_dofs': str(meta['bc_dofs_mode']) if 'bc_dofs_mode' in meta else 'unknown',
            'load_location': str(meta['load_location_mode']) if 'load_location_mode' in meta else 'unknown',
            'load_direction': str(meta['load_direction_mode']) if 'load_direction_mode' in meta else 'unknown',
        }

    meta_info = {
        'mass_cutoff_value': float(meta['mass_cutoff_value']) if 'mass_cutoff_value' in meta else None,
        'mass_cutoff_method': str(meta['mass_cutoff_method']) if 'mass_cutoff_method' in meta else 'unknown',
        'positive_if': str(meta['positive_if']) if 'positive_if' in meta else 'unknown',
        'label_mode': str(meta['label_mode']) if 'label_mode' in meta else 'unknown',
        'conditioning_spec': conditioning_spec,
    }

    # Build a descriptive run name similar to GAN trainers
    cond_tag = (
        f"bcLoc-{meta_info['conditioning_spec'].get('bc_locations','unknown')}_"
        f"bcDofs-{meta_info['conditioning_spec'].get('bc_dofs','unknown')}_"
        f"loadLoc-{meta_info['conditioning_spec'].get('load_location','unknown')}_"
        f"loadDir-{meta_info['conditioning_spec'].get('load_direction','unknown')}"
    )
    mass_tag = f"massLabel-{meta_info['positive_if']}"
    extra_tag = f"_{config['tag']}" if config["tag"] else ""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    run_name = (
        f"{cond_tag}_{mass_tag}_"
        f"mode-{config['mode']}_epochs-{config['nepochs']}_bs-{config['batchsize']}_"
        f"c1-{config['unet_ch1_dim']}_c2-{config['unet_ch2_dim']}_c3-{config['unet_ch3_dim']}"
        f"{extra_tag}_{timestamp}"
    )

    model_save_dir = log_root / run_name
    ckpt_loc_dir = model_save_dir / "checkpoints"
    model_save_dir.mkdir(parents=True, exist_ok=False)
    ckpt_loc_dir.mkdir()

    with open(model_save_dir / "hparams.yml", "w") as f:
        yaml.dump({**config, **meta_info, 'meta_path': str(meta_path)}, f)

    dataset, train_loader, val_loader = make_loaders_positive_only(
        npy_path=config["data_file"],
        batchsize=config["batchsize"],
        nsamples=config["nsamples"],
        img_size=config["img_size"],
        val_frac=config["val_frac"],
        seed=config["seed"],
        num_workers=config["num_workers"],
    )

    marginal_prob_mean_fn = functools.partial(marginal_prob_mean, bmin=0.1, bmax=20.0)
    marginal_prob_std_fn = functools.partial(marginal_prob_std, bmin=0.1, bmax=20.0)
    drift_coeff_fn = functools.partial(drift_coeff, bmin=0.1, bmax=20.0)

    score_model = ScoreNet3DVoxelCond(
        in_ch=1,
        cond_dim=dataset.cond_dim,
        cond_embed_dim=config["cond_embed_dim"],
        cond_ch=config["cond_ch"],
        c1=config["unet_ch1_dim"],
        c2=config["unet_ch2_dim"],
        c3=config["unet_ch3_dim"],
        ed=config["t_embed_dim"],
        mode=config["mode"],
        marginal_prob_std=marginal_prob_std_fn,
    ).to(device)

    if config["load_version"] is not None:
        prev_dir = log_root / f"version_{config['load_version']}" / "checkpoints"
        ckpt_loc = prev_dir / "ckpt_best.pth"
        score_model.load_state_dict(torch.load(ckpt_loc, map_location=device))
        print(f"Loaded previous best checkpoint from {ckpt_loc}")

    optimizer = Adam(score_model.parameters(), lr=config["lr"])
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config["ema_decay"])
    writer = SummaryWriter(log_dir=str(model_save_dir))

    eps = 1e-5
    best_val_loss = float("inf")
    keep_last_n = 3

    def compute_validation_loss():
        score_model.eval()
        val_loss = 0.0
        num_items = 0
        with torch.no_grad():
            for x_batch, c_batch, _, _ in val_loader:
                x = x_batch.to(device).float()
                cond = c_batch.to(device).float()
                loss, _ = vpsde_loss_fn(
                    score_model,
                    x,
                    marginal_prob_mean_fn,
                    marginal_prob_std_fn,
                    cond,
                    eps,
                    config["mode"],
                    config["loss_weighting"],
                )
                val_loss += loss.item() * x.shape[0]
                num_items += x.shape[0]
        score_model.train()
        return val_loss / max(1, num_items)

    nepochs = config["nepochs"]
    save_every = config["save_every_n_epochs"]
    sample_every = config["sample_every_n_epochs"]

    for epoch in range(nepochs):
        avg_loss = 0.0
        num_items = 0
        score_model.train()

        for x_batch, c_batch, _, _ in train_loader:
            x = x_batch.to(device).float()
            cond = c_batch.to(device).float()
            cond = maybe_drop_condition(cond, config["cond_drop_prob"])

            loss, _ = vpsde_loss_fn(
                score_model,
                x,
                marginal_prob_mean_fn,
                marginal_prob_std_fn,
                cond,
                eps,
                config["mode"],
                config["loss_weighting"],
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.update()

            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]

        train_avg_loss = avg_loss / max(1, num_items)
        writer.add_scalar("Loss/train", train_avg_loss, epoch)
        print(f"Epoch {epoch} train loss: {train_avg_loss:.6e}")

        val_loss = compute_validation_loss()
        writer.add_scalar("Loss/val", val_loss, epoch)
        print(f"Epoch {epoch} val loss: {val_loss:.6e}")

        if epoch % save_every == 0:
            with ema.average_parameters():
                ckpt_path = ckpt_loc_dir / f"ckpt_{epoch}.pth"
                torch.save(score_model.state_dict(), ckpt_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            with ema.average_parameters():
                best_ckpt_path = ckpt_loc_dir / "ckpt_best.pth"
                torch.save(score_model.state_dict(), best_ckpt_path)
            print(f"Saved new best checkpoint at epoch {epoch}")

        ckpts = sorted(ckpt_loc_dir.glob("ckpt_*.pth"), key=os.path.getmtime)
        ckpts_to_remove = ckpts[:-keep_last_n]
        for p in ckpts_to_remove:
            if p.name != "ckpt_best.pth":
                p.unlink()

        if sample_every > 0 and (epoch % sample_every == 0 or epoch == nepochs - 1):
            score_model.eval()
            with ema.average_parameters():
                save_sample_batch(
                    score_model,
                    dataset,
                    model_save_dir,
                    epoch,
                    config,
                    device,
                    marginal_prob_mean_fn,
                    marginal_prob_std_fn,
                    drift_coeff_fn,
                    meta,
                )
            score_model.train()

    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()