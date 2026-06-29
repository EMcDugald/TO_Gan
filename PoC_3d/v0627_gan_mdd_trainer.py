import argparse
import csv
import os
from datetime import datetime
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from tqdm import trange


class DataNotFoundError(Exception):
    pass


class CondVoxelDataset(Dataset):
    def __init__(self, filepath):
        if not os.path.exists(filepath):
            raise DataNotFoundError(f"File not found: {filepath}")
        data = np.load(filepath, allow_pickle=True)
        voxels = [x[0] for x in data]
        conds = [x[1] for x in data]
        labels = [x[2] for x in data]
        cond_strs = [x[3] for x in data]
        X = torch.tensor(np.stack(voxels), dtype=torch.float32)
        C = torch.tensor(np.stack(conds), dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.long)
        self.cond_strs = cond_strs
        if X.ndim == 4:
            X = X.unsqueeze(1)
        X = X * 2 - 1
        self.X = X
        self.C = C
        self.y = y
        print("Total samples:", X.shape[0])
        print("Condition dim:", C.shape[1])
        print("Positives:", int((y == 1).sum().item()), "Negatives:", int((y == 0).sum().item()))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.C[idx], self.y[idx]


class ReusableDataLoader:
    def __init__(self, X, C, batch_size, shuffle=True):
        self.X = X
        self.C = C
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(X.shape[0]))
        self.previous_indices = []

    def _shuffle_indices(self):
        self.indices = torch.randperm(len(self.indices)).tolist()

    def get_batch(self):
        queued = self.previous_indices
        while len(queued) < self.batch_size:
            if self.shuffle:
                self._shuffle_indices()
            queued.extend(self.indices)
        self.previous_indices = queued[self.batch_size:]
        batch_indices = queued[:self.batch_size]
        x_batch = torch.stack([self.X[i] for i in batch_indices])
        c_batch = torch.stack([self.C[i] for i in batch_indices])
        return x_batch, c_batch


def mass_fraction_batch(batch_np):
    v = (batch_np > 0).astype(np.float64)
    return v.mean(axis=(1, 2, 3))


def diversity_loss(x):
    r = torch.sum(x ** 2, dim=1, keepdim=True)
    D = r - 2 * torch.matmul(x, x.T) + r.T
    S = torch.exp(-0.5 * D ** 2)
    try:
        eig_val = torch.linalg.eigvalsh(S)
    except Exception:
        eig_val = torch.ones(x.size(0), device=x.device)
    loss = -torch.mean(torch.log(torch.clamp(eig_val, min=1e-7)))
    return loss


def eval_dpp_div_from_voxels(batch_np, device):
    x = torch.tensor(batch_np.reshape(batch_np.shape[0], -1), dtype=torch.float32, device=device)
    return float(diversity_loss(x).item())


def multiclass_ce_with_optional_label_smoothing(logits, target, smoothing=0.0):
    """
    Standard multiclass cross-entropy with optional label smoothing.

    logits: (B, C)
    target: (B,) integer class labels
    smoothing: epsilon in [0, 1)

    PyTorch-consistent smoothing:
      target_dist = (1-eps) * one_hot + eps / num_classes
    """
    if smoothing <= 0.0:
        return nn.functional.cross_entropy(logits, target)

    num_classes = logits.size(1)
    log_probs = torch.log_softmax(logits, dim=1)

    with torch.no_grad():
        true_dist = torch.full_like(logits, smoothing / num_classes)
        true_dist.scatter_(1, target.unsqueeze(1), (1.0 - smoothing) + smoothing / num_classes)

    return torch.sum(-true_dist * log_probs, dim=1).mean()


class CondGenerator3d(nn.Module):
    def __init__(self, nz, ngf, output_shape, cond_dim, cond_embed_dim=32):
        super().__init__()
        D, H, W = output_shape
        self.cond_fc = nn.Linear(cond_dim, cond_embed_dim)
        self.init_shape = (ngf * 8, D // 16, H // 16, W // 16)
        self.fc = nn.Linear(nz + cond_embed_dim, int(np.prod(self.init_shape)))
        self.main = nn.Sequential(
            nn.BatchNorm3d(ngf * 8), nn.ReLU(True),
            nn.ConvTranspose3d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ngf * 4), nn.ReLU(True),
            nn.ConvTranspose3d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ngf * 2), nn.ReLU(True),
            nn.ConvTranspose3d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ngf), nn.ReLU(True),
            nn.ConvTranspose3d(ngf, 1, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z, c):
        c_emb = torch.relu(self.cond_fc(c))
        x = self.fc(torch.cat([z, c_emb], dim=1))
        return self.main(x.view(x.size(0), *self.init_shape))


class CondDiscriminator3d(nn.Module):
    def __init__(self, ndf, input_shape, cond_dim, cond_embed_dim=32, nc=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(1, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ndf * 2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ndf * 4), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ndf * 8), nn.LeakyReLU(0.2, inplace=True),
        )
        D, H, W = input_shape
        feat_size = (ndf * 8) * (D // 16) * (H // 16) * (W // 16)
        self.cond_fc = nn.Linear(cond_dim, cond_embed_dim)
        self.fc = nn.Linear(feat_size + cond_embed_dim, nc)

    def forward(self, x, c):
        h = self.conv(x).view(x.size(0), -1)
        c_emb = torch.relu(self.cond_fc(c))
        return self.fc(torch.cat([h, c_emb], dim=1))


def unwrap_module(m):
    return m.module if isinstance(m, nn.DataParallel) else m


def save_checkpoint(step, netG, netD, G_opt, D_opt, path):
    G_raw = unwrap_module(netG)
    D_raw = unwrap_module(netD)
    torch.save({
        'step': step,
        'netG_state': G_raw.state_dict(),
        'netD_state': D_raw.state_dict(),
        'G_opt_state': G_opt.state_dict(),
        'D_opt_state': D_opt.state_dict(),
    }, path)


def load_checkpoint(path, netG, netD, G_opt, D_opt, device):
    if not os.path.exists(path):
        raise FileNotFoundError(f'No checkpoint found at {path}')
    ckpt = torch.load(path, map_location=device)
    netG.load_state_dict(ckpt['netG_state'])
    netD.load_state_dict(ckpt['netD_state'])
    G_opt.load_state_dict(ckpt['G_opt_state'])
    D_opt.load_state_dict(ckpt['D_opt_state'])
    for opt in (G_opt, D_opt):
        for state in opt.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
    return netG, netD, G_opt, D_opt, ckpt.get('step', 0)


def get_bin_centers(edges):
    edges = np.asarray(edges, dtype=np.float64)
    return 0.5 * (edges[:-1] + edges[1:])


def decode_condition_overlay(cond_vec, meta):
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


def plot_voxel_grid_3d(voxel_grid, title='', save_path=None, bc_points=None, load_point=None, load_vec=None):
    voxel_grid = np.asarray(voxel_grid)
    nx, ny, nz = voxel_grid.shape

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

    fig = plt.figure(figsize=(10, 4))
    views = [(25, 35, 'View 1'), (25, 125, 'View 2')]

    for k, (elev, azim, subtitle) in enumerate(views, start=1):
        ax = fig.add_subplot(1, 2, k, projection='3d')
        ax.voxels(voxel_grid > 0.1, edgecolor='k', linewidth=0.2, alpha=0.72)
        ax.set_xlim(0, nx)
        ax.set_ylim(0, ny)
        ax.set_zlim(0, nz)

        if bc_plot is not None:
            ax.scatter(
                bc_plot[:, 0], bc_plot[:, 1], bc_plot[:, 2],
                c='red', s=36, marker='o', edgecolors='white', linewidths=0.6,
                depthshade=False, label='BC'
            )

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

        ax.view_init(elev=elev, azim=azim)
        ax.set_title(subtitle)
        ax.set_box_aspect((nx, ny, nz))
        ax.set_axis_off()

    fig.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=220)
    plt.close(fig)


def GAN_step_MDD_3d_cond_3class(
    D, G, A, D_opt, G_opt, A_opt,
    P_batch, N_batch, c_batch, noise_batch,
    batch_size, device,
    diversity_weight=0.0,
    smooth_real=0.1,
    d_update=True, use_label_smoothing=True,
    use_diversity_loss=False,
):
    criterion = nn.CrossEntropyLoss()

    if d_update:
        D.zero_grad()
        y_pos = torch.full((batch_size,), 1, dtype=torch.long, device=device)
        y_neg = torch.full((batch_size,), 2, dtype=torch.long, device=device)
        y_fake = torch.full((batch_size,), 0, dtype=torch.long, device=device)

        out_real_pos = D(P_batch, c_batch)
        prob_real_pos = torch.softmax(out_real_pos, dim=1)
        L_D_real = multiclass_ce_with_optional_label_smoothing(
            out_real_pos,
            y_pos,
            smoothing=(smooth_real if use_label_smoothing else 0.0),
        )

        out_real_neg = D(N_batch, c_batch)
        prob_real_neg = torch.softmax(out_real_neg, dim=1)
        L_D_neg = criterion(out_real_neg, y_neg)

        fake_data_for_D = G(noise_batch, c_batch)
        out_fake = D(fake_data_for_D.detach(), c_batch)
        prob_fake_for_D = torch.softmax(out_fake, dim=1)
        L_D_fake = criterion(out_fake, y_fake)

        L_D_tot = L_D_real + L_D_neg + L_D_fake
        L_D_tot.backward()

        D_grad_norm = 0.0
        for p in D.parameters():
            if p.grad is not None:
                D_grad_norm += p.grad.detach().pow(2).sum().item()
        D_grad_norm = D_grad_norm ** 0.5
        D_opt.step()
    else:
        L_D_real = torch.tensor(0.0, device=device)
        L_D_neg = torch.tensor(0.0, device=device)
        L_D_fake = torch.tensor(0.0, device=device)
        D_grad_norm = 0.0

        out_real_pos = D(P_batch, c_batch)
        prob_real_pos = torch.softmax(out_real_pos, dim=1)

        out_real_neg = D(N_batch, c_batch)
        prob_real_neg = torch.softmax(out_real_neg, dim=1)

        fake_data_for_D = G(noise_batch, c_batch)
        out_fake = D(fake_data_for_D.detach(), c_batch)
        prob_fake_for_D = torch.softmax(out_fake, dim=1)

    G.zero_grad()
    fake_data = G(noise_batch, c_batch)
    out_fake_for_G = D(fake_data, c_batch)
    prob_fake_for_G = torch.softmax(out_fake_for_G, dim=1)

    y_pos = torch.full((batch_size,), 1, dtype=torch.long, device=device)
    L_G = multiclass_ce_with_optional_label_smoothing(
        out_fake_for_G,
        y_pos,
        smoothing=(smooth_real if use_label_smoothing else 0.0),
    )

    if use_diversity_loss and diversity_weight > 0:
        feat = fake_data.view(fake_data.size(0), -1)
        L_div = diversity_loss(feat)
        L_G_tot = L_G + diversity_weight * L_div
    else:
        L_div = None
        L_G_tot = L_G

    L_G_tot.backward()
    G_grad_norm = 0.0
    for p in G.parameters():
        if p.grad is not None:
            G_grad_norm += p.grad.detach().pow(2).sum().item()
    G_grad_norm = G_grad_norm ** 0.5
    G_opt.step()

    report = {
        'L_D_real': float(L_D_real.item()),
        'L_D_neg': float(L_D_neg.item()),
        'L_D_fake': float(L_D_fake.item()),
        'L_G': float(L_G.item()),
        'D_grad_norm': float(D_grad_norm),
        'G_grad_norm': float(G_grad_norm),
        'P_pos_real_pos': float(prob_real_pos[:, 1].mean().item()),
        'P_neg_real_neg': float(prob_real_neg[:, 2].mean().item()),
        'P_fake_fake_D': float(prob_fake_for_D[:, 0].mean().item()),
        'P_pos_fake_D': float(prob_fake_for_D[:, 1].mean().item()),
        'P_pos_fake_G': float(prob_fake_for_G[:, 1].mean().item()),
    }
    if L_div is not None:
        report['L_div'] = float(L_div.item())
    return report


def train_3d_cond(D, G, A, D_opt, G_opt, A_opt,
                  P_loader, N_loader,
                  num_steps, batch_size, noise_dim,
                  train_step_fn, device,
                  diversity_weight=0.0,
                  checkpoint_dir=None, ckpt_interval=None,
                  smooth_real=0.1,
                  C_P_full=None, cond_strs=None,
                  eval_every_steps=None, nz=None,
                  start_step=0, d_every=3,
                  use_label_smoothing=True,
                  use_diversity_loss=False,
                  n_vis_samples=10,
                  meta=None):
    best_G_loss = float('inf')
    steps_per_epoch = max(1, len(P_loader.X) // batch_size)
    metrics_file = None
    metrics_writer = None
    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        metrics_path = os.path.join(checkpoint_dir, 'metrics.csv')
        mode = 'a' if start_step > 0 and os.path.exists(metrics_path) else 'w'
        metrics_file = open(metrics_path, mode, newline='')
        metrics_writer = csv.writer(metrics_file)
        if mode == 'w':
            metrics_writer.writerow([
                'step', 'epoch',
                'L_D_real', 'L_D_neg', 'L_D_fake', 'L_G',
                'D_grad_norm', 'G_grad_norm', 'L_div',
                'P_pos_real_pos', 'P_neg_real_neg',
                'P_fake_fake_D', 'P_pos_fake_D', 'P_pos_fake_G'
            ])

    steps_range = trange(start_step, num_steps, position=0, leave=True)
    for step in steps_range:
        P_batch, cP = P_loader.get_batch()
        N_batch, _ = N_loader.get_batch()
        c_batch = cP.to(device)
        P_batch = P_batch.to(device)
        N_batch = N_batch.to(device)
        noise_batch = torch.randn(batch_size, noise_dim, device=device)
        d_update = ((step + 1) % d_every == 0)

        report = train_step_fn(
            D, G, A, D_opt, G_opt, A_opt,
            P_batch, N_batch, c_batch, noise_batch,
            batch_size, device,
            diversity_weight=diversity_weight,
            smooth_real=smooth_real,
            d_update=d_update,
            use_label_smoothing=use_label_smoothing,
            use_diversity_loss=use_diversity_loss,
        )

        steps_range.set_postfix({k: f"{v:.4f}" for k, v in report.items() if isinstance(v, float)})

        if metrics_writer is not None:
            epoch = step // steps_per_epoch
            metrics_writer.writerow([
                step + 1, epoch,
                report.get('L_D_real', float('nan')),
                report.get('L_D_neg', float('nan')),
                report.get('L_D_fake', float('nan')),
                report.get('L_G', float('nan')),
                report.get('D_grad_norm', float('nan')),
                report.get('G_grad_norm', float('nan')),
                report.get('L_div', float('nan')),
                report.get('P_pos_real_pos', float('nan')),
                report.get('P_neg_real_neg', float('nan')),
                report.get('P_fake_fake_D', float('nan')),
                report.get('P_pos_fake_D', float('nan')),
                report.get('P_pos_fake_G', float('nan')),
            ])

        current_G_loss = report.get('L_G', None)
        if checkpoint_dir is not None and current_G_loss is not None and current_G_loss < best_G_loss:
            best_G_loss = current_G_loss
            save_checkpoint(step + 1, G, D, G_opt, D_opt, os.path.join(checkpoint_dir, 'ckpt_best.pt'))

        if checkpoint_dir is not None and ckpt_interval is not None and (step + 1) % ckpt_interval == 0:
            save_checkpoint(step + 1, G, D, G_opt, D_opt, os.path.join(checkpoint_dir, f'ckpt_step_{step+1}.pt'))

        if (
            checkpoint_dir is not None and
            eval_every_steps is not None and
            C_P_full is not None and
            cond_strs is not None and
            nz is not None and
            meta is not None and
            (step + 1) % eval_every_steps == 0
        ):
            current_epoch = (step + 1) // steps_per_epoch
            samples_subdir = os.path.join(checkpoint_dir, f'epoch_{current_epoch:04d}')
            os.makedirs(samples_subdir, exist_ok=True)

            G_raw = unwrap_module(G)
            G_raw.eval()
            with torch.no_grad():
                z = torch.randn(n_vis_samples, nz, device=device)
                idx_vis = torch.randint(low=0, high=C_P_full.shape[0], size=(n_vis_samples,))
                c_vis = C_P_full[idx_vis].to(device)
                fake = G_raw(z, c_vis).cpu().numpy()
                cond_np = c_vis.cpu().numpy()
                cond_strs_vis = [cond_strs[int(i)] for i in idx_vis.cpu().numpy()]

            np.save(os.path.join(samples_subdir, 'fake_conditions.npy'), cond_np)
            np.save(os.path.join(samples_subdir, 'fake_indices.npy'), idx_vis.cpu().numpy())

            txt_path = os.path.join(samples_subdir, 'fake_voxel_conditions.txt')
            with open(txt_path, 'w') as f_txt:
                f_txt.write('# idx  filename  condition_vector  condition_string\n')
                for i in range(n_vis_samples):
                    overlay = decode_condition_overlay(cond_np[i], meta)
                    bc_points = overlay["bc_points"]
                    load_point = overlay["load_point"]
                    load_dir = overlay["load_dir"]
                    spec = overlay["conditioning_spec"]
                    mode_tag = (
                        f"bcLoc={spec.get('bc_locations')} "
                        f"loadLoc={spec.get('load_location')} "
                        f"loadDir={spec.get('load_direction')}"
                    )

                    filename = f'fake_voxel_{i}.png'
                    fig_path = os.path.join(samples_subdir, filename)
                    title = f'Fake sample {i} (epoch {current_epoch}) | mode={mode_tag}'
                    plot_voxel_grid_3d(
                        fake[i, 0],
                        title=title,
                        save_path=fig_path,
                        bc_points=bc_points,
                        load_point=load_point,
                        load_vec=load_dir,
                    )
                    cond_str_num = ' '.join(f'{v:.6f}' for v in cond_np[i])
                    f_txt.write(f'{i:03d}  {filename}  {cond_str_num}  {cond_strs_vis[i]}\n')
            G_raw.train()

    if checkpoint_dir is not None:
        save_checkpoint(num_steps, G, D, G_opt, D_opt, os.path.join(checkpoint_dir, 'ckpt_final.pt'))
    if metrics_file is not None:
        metrics_file.close()
    return D, G, A


def resolve_meta_path(data_path, meta_path=None):
    if meta_path is not None:
        return meta_path
    if data_path.endswith('.npy'):
        return data_path[:-4] + '_meta.npz'
    raise ValueError('Could not infer meta path; please provide --meta-path')


def get_mass_meta(meta):
    if 'mass_cutoff_value' not in meta:
        raise KeyError('Expected mass_cutoff_value in meta file for new BC/load-only dataset')

    # Prefer newer JSON/string-backed metadata, then fall back to older keys
    if 'conditioning_spec_json' in meta:
        conditioning_spec = json.loads(str(meta['conditioning_spec_json']))
        conditioning_mode = str(conditioning_spec.get('bc_locations', 'unknown'))
    elif 'conditioning_spec_str' in meta:
        # purely for logging if json is unavailable
        conditioning_mode = str(meta['conditioning_spec_str'])
    elif 'conditioning_mode' in meta:
        conditioning_mode = str(meta['conditioning_mode'])
    else:
        conditioning_mode = 'unknown'

    return {
        'mass_cutoff_value': float(meta['mass_cutoff_value']),
        'mass_cutoff_method': str(meta['mass_cutoff_method']) if 'mass_cutoff_method' in meta else 'unknown',
        'positive_if': str(meta['positive_if']) if 'positive_if' in meta else 'unknown',
        'conditioning_mode': conditioning_mode,
        'label_mode': str(meta['label_mode']) if 'label_mode' in meta else 'unknown',
        'mass_quantile': (
            None if 'mass_quantile' not in meta or np.isnan(float(meta['mass_quantile']))
            else float(meta['mass_quantile'])
        ),
        'mass_threshold_source': str(meta['mass_threshold_source']) if 'mass_threshold_source' in meta else 'unknown',
    }


def main():
    parser = argparse.ArgumentParser(description='Train v0627 non-UNet 3D GAN-MDD on BC/load-conditioned, mass-labeled voxel data.')
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--meta-path', type=str, default=None)
    parser.add_argument('--checkpoint-root', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--nz', type=int, default=300)
    parser.add_argument('--ngf', type=int, default=256)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--num-epochs', type=int, default=1000)
    parser.add_argument('--lr-d', type=float, default=2e-4)
    parser.add_argument('--lr-g', type=float, default=2e-4)
    parser.add_argument('--smooth-real', type=float, default=0.1)
    parser.add_argument('--d-every', type=int, default=3)
    parser.add_argument('--use-label-smoothing', action='store_true')
    parser.add_argument('--use-diversity-loss', action='store_true')
    parser.add_argument('--diversity-weight', type=float, default=1.0)
    parser.add_argument('--n-vis-samples', type=int, default=5)
    parser.add_argument('--ckpt-every-epochs', type=int, default=20)
    parser.add_argument('--eval-every-epochs', type=int, default=20)
    parser.add_argument('--resume-path', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--tag', type=str, default='')
    args = parser.parse_args()

    print('PyTorch version:', torch.__version__)
    print('CUDA available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('GPU device count (initial):', torch.cuda.device_count())
        print('GPU device name[0]:', torch.cuda.get_device_name(0))

    meta_path = resolve_meta_path(args.data_path, args.meta_path)
    meta = np.load(meta_path, allow_pickle=True)
    meta_info = get_mass_meta(meta)

    dataset = CondVoxelDataset(args.data_path)
    pos_mask = (dataset.y == 1)
    neg_mask = (dataset.y == 0)
    P = dataset.X[pos_mask]
    C_P = dataset.C[pos_mask]
    N = dataset.X[neg_mask]
    C_N = dataset.C[neg_mask]
    n_samples = P.shape[0] + N.shape[0]

    device = torch.device(args.device if (args.device != 'cuda' or torch.cuda.is_available()) else 'cpu')
    print('device:', device)
    print('CUDA device count (inside main):', torch.cuda.device_count())

    shape3d = P.shape[2:]
    cond_dim = C_P.shape[1]

    netG = CondGenerator3d(args.nz, args.ngf, shape3d, cond_dim)
    netD = CondDiscriminator3d(args.ndf, shape3d, cond_dim, nc=3)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('Generator params:', count_parameters(unwrap_module(netG)))
    print('Discriminator params:', count_parameters(unwrap_module(netD)))

    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        print(f'Using DataParallel on {torch.cuda.device_count()} GPUs')
        netG = nn.DataParallel(netG)
        netD = nn.DataParallel(netD)

    netG = netG.to(device)
    netD = netD.to(device)

    P_loader = ReusableDataLoader(P, C_P, args.batch_size)
    N_loader = ReusableDataLoader(N, C_N, args.batch_size)
    num_steps = args.num_epochs * len(P) // args.batch_size

    D_opt = optim.Adam(netD.parameters(), lr=args.lr_d, betas=(0.5, 0.999))
    G_opt = optim.Adam(netG.parameters(), lr=args.lr_g, betas=(0.5, 0.999))

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

    cond_tag = (
        f'bcLoc-{conditioning_spec.get("bc_locations", "unknown")}_'
        f'bcDofs-{conditioning_spec.get("bc_dofs", "unknown")}_'
        f'loadLoc-{conditioning_spec.get("load_location", "unknown")}_'
        f'loadDir-{conditioning_spec.get("load_direction", "unknown")}'
    )

    label_tag = meta_info['positive_if']
    extra_tag = f'_{args.tag}' if args.tag else ''
    hp_name = (
        f'{cond_tag}_massLabel_{label_tag}_epochs{args.num_epochs}_bs{args.batch_size}_'
        f'nz{args.nz}_ngf{args.ngf}_ndf{args.ndf}_nsamp{n_samples}_'
        f'lrD{args.lr_d}_lrG{args.lr_g}_smoothR{args.smooth_real}_'
        f'dEvery{args.d_every}_div{int(args.use_diversity_loss)}{extra_tag}'
    )
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    checkpoint_dir = os.path.join(args.checkpoint_root, f'{hp_name}_{timestamp}')
    os.makedirs(checkpoint_dir, exist_ok=True)

    start_step = 0
    if args.resume_path is not None:
        print(f'Resuming from checkpoint: {args.resume_path}')
        netG, netD, G_opt, D_opt, start_step = load_checkpoint(args.resume_path, netG, netD, G_opt, D_opt, device)
    else:
        print('Starting from scratch')

    with open(os.path.join(checkpoint_dir, 'hparams.txt'), 'w') as f_hp:
        f_hp.write(f'lr_D = {args.lr_d}\n')
        f_hp.write(f'lr_G = {args.lr_g}\n')
        f_hp.write(f'smooth_real = {args.smooth_real}\n')
        f_hp.write(f'd_every = {args.d_every}\n')
        f_hp.write(f'use_label_smoothing = {args.use_label_smoothing}\n')
        f_hp.write(f'use_diversity_loss = {args.use_diversity_loss}\n')
        f_hp.write(f'diversity_weight = {args.diversity_weight}\n')
        f_hp.write(f'batch_size = {args.batch_size}\n')
        f_hp.write(f'num_epochs = {args.num_epochs}\n')
        f_hp.write(f'nz = {args.nz}, ngf = {args.ngf}, ndf = {args.ndf}\n')
        f_hp.write(f'n_samples = {n_samples}\n')
        f_hp.write(f'cond_dim = {cond_dim}\n')
        f_hp.write(f'data_path = {args.data_path}\n')
        f_hp.write(f'meta_path = {meta_path}\n')
        f_hp.write(f'mass_cutoff_value = {meta_info["mass_cutoff_value"]:.6f}\n')
        f_hp.write(f'mass_cutoff_method = {meta_info["mass_cutoff_method"]}\n')
        f_hp.write(f'mass_threshold_source = {meta_info["mass_threshold_source"]}\n')
        f_hp.write(f'mass_quantile = {meta_info["mass_quantile"]}\n')
        f_hp.write(f'positive_if = {meta_info["positive_if"]}\n')
        f_hp.write(f'conditioning_spec = {conditioning_spec}\n')
        f_hp.write(f'label_mode = {meta_info["label_mode"]}\n')

    steps_per_epoch = max(1, len(P) // args.batch_size)
    ckpt_interval = args.ckpt_every_epochs * steps_per_epoch
    eval_every_steps = args.eval_every_epochs * steps_per_epoch

    pos_indices = torch.where(pos_mask)[0].cpu().numpy()
    cond_strs_P = [dataset.cond_strs[int(i)] for i in pos_indices]

    netD, netG, _ = train_3d_cond(
        netD, netG, None,
        D_opt, G_opt, None,
        P_loader, N_loader,
        num_steps, args.batch_size, args.nz,
        GAN_step_MDD_3d_cond_3class,
        device,
        diversity_weight=args.diversity_weight,
        checkpoint_dir=checkpoint_dir,
        ckpt_interval=ckpt_interval,
        smooth_real=args.smooth_real,
        C_P_full=C_P,
        cond_strs=cond_strs_P,
        eval_every_steps=eval_every_steps,
        nz=args.nz,
        start_step=start_step,
        d_every=args.d_every,
        use_label_smoothing=args.use_label_smoothing,
        use_diversity_loss=args.use_diversity_loss,
        n_vis_samples=args.n_vis_samples,
        meta=meta,
    )

    netG.eval()
    with torch.no_grad():
        z = torch.randn(args.batch_size, args.nz, device=device)
        idx_vis = torch.randint(low=0, high=C_P.shape[0], size=(args.batch_size,))
        c_vis = C_P[idx_vis].to(device)
        fake = netG(z, c_vis).cpu().numpy()
        cond_strs_vis = [cond_strs_P[int(i)] for i in idx_vis.cpu().numpy()]
        cond_np = c_vis.cpu().numpy()
        np.save(os.path.join(checkpoint_dir, 'fake_conditions_vis.npy'), cond_np)
        np.save(os.path.join(checkpoint_dir, 'fake_indices_vis.npy'), idx_vis.cpu().numpy())
        txt_path = os.path.join(checkpoint_dir, 'fake_voxel_conditions_final.txt')
        with open(txt_path, 'w') as f_txt:
            f_txt.write('# idx  filename  condition_vector  condition_string\n')
            for idx in range(args.batch_size):
                overlay = decode_condition_overlay(cond_np[idx], meta)
                bc_points = overlay["bc_points"]
                load_point = overlay["load_point"]
                load_dir = overlay["load_dir"]
                spec = overlay["conditioning_spec"]
                mode_tag = (
                    f"bcLoc={spec.get('bc_locations')} "
                    f"loadLoc={spec.get('load_location')} "
                    f"loadDir={spec.get('load_direction')}"
                )

                filename = f'fake_voxel_{idx}.png'
                fig_path = os.path.join(checkpoint_dir, filename)
                plot_voxel_grid_3d(
                    fake[idx, 0],
                    title=f'Fake sample {idx} | mode={mode_tag}',
                    save_path=fig_path,
                    bc_points=bc_points,
                    load_point=load_point,
                    load_vec=load_dir,
                )
                cond_str_num = ' '.join(f'{v:.6f}' for v in cond_np[idx])
                f_txt.write(f'{idx:03d}  {filename}  {cond_str_num}  {cond_strs_vis[idx]}\n')

    print(f'Saved voxel plots and fake_voxel_conditions_final.txt for {args.batch_size} fake samples.')

    batches_eval = 10
    all_mass = []
    all_div = []
    netG.eval()
    with torch.no_grad():
        for _ in trange(batches_eval):
            z = torch.randn(args.batch_size, args.nz, device=device)
            idx_eval = torch.randint(low=0, high=C_P.shape[0], size=(args.batch_size,))
            c_eval = C_P[idx_eval].to(device)
            fake = netG(z, c_eval).cpu().numpy()
            fake_bin = (fake > 0.0).astype(np.float64)
            batch_np = fake_bin[:, 0, :, :, :]
            mass_fracs = mass_fraction_batch(batch_np)
            all_mass.append(mass_fracs)
            div_val = eval_dpp_div_from_voxels(batch_np, device=device)
            all_div.append(div_val)

    all_mass = np.concatenate(all_mass)
    cutoff = meta_info['mass_cutoff_value']
    if meta_info['positive_if'] == 'low_mass':
        positive_rate = float((all_mass <= cutoff).mean() * 100.0)
    elif meta_info['positive_if'] == 'high_mass':
        positive_rate = float((all_mass >= cutoff).mean() * 100.0)
    else:
        raise ValueError(f"Unexpected positive_if in meta: {meta_info['positive_if']}")

    mean_mass = float(all_mass.mean())
    mean_diversity = float(np.mean(all_div))
    print('Mass cutoff value:', cutoff)
    print('Mass cutoff method:', meta_info['mass_cutoff_method'])
    print('positive_if:', meta_info['positive_if'])
    print('Mass-labeled positive rate wrt TRAIN cutoff (%):', positive_rate)
    print('Mean mass fraction:', mean_mass)
    print('Mean DPP diversity:', mean_diversity)

    results_txt = os.path.join(checkpoint_dir, 'evaluation_mass.txt')
    with open(results_txt, 'w') as f:
        f.write(f'mass_cutoff_value_train: {cutoff:.6f}\n')
        f.write(f'mass_cutoff_method_train: {meta_info["mass_cutoff_method"]}\n')
        f.write(f'mass_threshold_source: {meta_info["mass_threshold_source"]}\n')
        f.write(f'mass_quantile: {meta_info["mass_quantile"]}\n')
        f.write(f'positive_if: {meta_info["positive_if"]}\n')
        f.write(f'conditioning_spec: {conditioning_spec}\n')
        f.write(f'Label-positive rate wrt train mass cutoff (%): {positive_rate:.4f}\n')
        f.write(f'Mean mass fraction: {mean_mass:.6f}\n')
        f.write(f'Mean DPP diversity: {mean_diversity:.6f}\n')


if __name__ == '__main__':
    main()